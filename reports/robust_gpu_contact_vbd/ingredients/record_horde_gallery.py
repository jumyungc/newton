# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record additional adversarial contact-gallery trajectories on Horde.

This is a focused report ingredient, not a unit test.  It records a dynamic
bullet/thin-plate impact and a staged rigid-cable pile.  Every accepted pile
frame is independently re-queried and may be restored/replayed with a finer
physical VBD step before it is written to the trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.tests import benchmark_cable_twist_contact_strategy as cable_benchmark


BULLET_RADIUS = 0.05
PLATE_HALF_THICKNESS = 0.02
PLATE_HALF_WIDTH = 0.28
BULLET_SPEED = 80.0
BULLET_DT = 0.02

PILE_LAYERS = 3
PILE_CABLES_PER_LAYER = 2
PILE_SEGMENTS = 24
PILE_RADIUS = 0.018
PILE_LENGTH = 1.60
PILE_FRAME_DT = 1.0 / 60.0
PILE_RELEASE_FRAMES = (0, 90, 180)
PILE_FRAMES = 330
PILE_BUCKETS = (10, 20, 40, 80, 160)
PILE_PENETRATION_TOLERANCE = 1.0e-3

LAYER_COLORS = (
    wp.vec3(54.0 / 255.0, 174.0 / 255.0, 240.0 / 255.0),
    wp.vec3(174.0 / 255.0, 125.0 / 255.0, 238.0 / 255.0),
    wp.vec3(255.0 / 255.0, 177.0 / 255.0, 83.0 / 255.0),
)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _build_bullet_plate(device: str):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    bullet = builder.add_body(xform=wp.transform(wp.vec3(-0.55, 0.0, 0.0)))
    builder.add_shape_sphere(
        bullet,
        radius=BULLET_RADIUS,
        cfg=newton.ModelBuilder.ShapeConfig(density=2000.0, restitution=0.0, mu=0.0),
    )
    plate = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
    builder.add_shape_box(
        plate,
        hx=PLATE_HALF_THICKNESS,
        hy=PLATE_HALF_WIDTH,
        hz=PLATE_HALF_WIDTH,
        cfg=newton.ModelBuilder.ShapeConfig(density=500.0, restitution=0.0, mu=0.0),
    )
    builder.body_qd[bullet] = (BULLET_SPEED, 0.0, 0.0, 0.0, 0.0, 0.0)
    builder.color()
    model = builder.finalize(device=device)
    return model, bullet, plate


def _record_bullet_method(device: str, event_projection: bool) -> tuple[dict, dict[str, np.ndarray]]:
    model, bullet, plate = _build_bullet_plate(device)
    pipeline = newton.CollisionPipeline(
        model,
        contact_matching="disabled",
        deterministic=True,
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=1.05 * BULLET_SPEED * BULLET_DT
        ),
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=5,
        rigid_contact_hard=True,
        rigid_contact_history=False,
        rigid_contact_feasibility_projection=event_projection,
        rigid_contact_event_waves=2,
        rigid_contact_event_closure_iterations=20,
        rigid_contact_event_active_set_iterations=4,
    )
    state_in = model.state()
    state_out = model.state()
    initial_q = state_in.body_q.numpy().copy()
    initial_qd = state_in.body_qd.numpy().copy()

    pipeline.narrow_phase.buffer_overflow_accumulator.zero_()
    solver.body_body_contact_overflow_accumulator.zero_()
    pipeline.collide(state_in, contacts, dt=BULLET_DT)
    query_count = int(contacts.rigid_contact_count.numpy()[0])
    solver.step(state_in, state_out, None, contacts, BULLET_DT)
    wp.synchronize()

    final_q = state_out.body_q.numpy().copy()
    final_qd = state_out.body_qd.numpy().copy()
    bullet_x = float(final_q[bullet, 0])
    plate_x = float(final_q[plate, 0])
    directed_gap = plate_x - PLATE_HALF_THICKNESS - (bullet_x + BULLET_RADIUS)
    masses = model.body_mass.numpy()
    initial_momentum = float(masses[bullet] * BULLET_SPEED)
    final_momentum = float(
        masses[bullet] * final_qd[bullet, 0] + masses[plate] * final_qd[plate, 0]
    )
    common_velocity = initial_momentum / float(masses[bullet] + masses[plate])
    metrics = {
        "method": "event_time_island_impulse" if event_projection else "ordinary_one_step_vbd",
        "speed_m_s": BULLET_SPEED,
        "dt_s": BULLET_DT,
        "potential_displacement_m": BULLET_SPEED * BULLET_DT,
        "initial_directed_gap_m": 0.55 - PLATE_HALF_THICKNESS - BULLET_RADIUS,
        "query_contact_count": query_count,
        "final_directed_gap_m": directed_gap,
        "crossed_plate_midplane": bullet_x > plate_x,
        "bullet_final_velocity_m_s": float(final_qd[bullet, 0]),
        "plate_final_velocity_m_s": float(final_qd[plate, 0]),
        "analytic_inelastic_common_velocity_m_s": common_velocity,
        "momentum_error_kg_m_s": final_momentum - initial_momentum,
        "event_body_updates": int(solver.body_contact_projection_count.numpy().sum()) if event_projection else 0,
        "event_debt_count": int(solver.rigid_contact_event_debt_count.numpy()[0]) if event_projection else 0,
        "query_integrity_debt": int(pipeline.narrow_phase.buffer_overflow_accumulator.numpy()[0]),
        "solver_list_required_capacity": int(solver.body_body_contact_overflow_accumulator.numpy()[0]),
        "finite": bool(np.isfinite(final_q).all() and np.isfinite(final_qd).all()),
    }
    trace = {
        "initial_q": initial_q,
        "initial_qd": initial_qd,
        "final_q": final_q,
        "final_qd": final_qd,
    }
    return metrics, trace


def build_staged_pile(device: str):
    """Build the staged pile model used by both recorder and renderer."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 0.0
    builder.default_shape_cfg.mu = 1.0
    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=0.0, mu=1.0)
    )

    layer_bodies: list[list[int]] = []
    segment_length = PILE_LENGTH / PILE_SEGMENTS
    lane_spacing = 0.19
    for layer in range(PILE_LAYERS):
        orient_x = layer % 2 == 0
        direction = wp.vec3(1.0, 0.0, 0.0) if orient_x else wp.vec3(0.0, 1.0, 0.0)
        orthogonal = wp.vec3(0.0, 1.0, 0.0) if orient_x else wp.vec3(1.0, 0.0, 0.0)
        z = 0.65 + 0.52 * layer
        bodies_in_layer: list[int] = []
        for lane in range(PILE_CABLES_PER_LAYER):
            offset = (lane - 0.5) * lane_spacing
            start = -0.5 * PILE_LENGTH * direction + offset * orthogonal + wp.vec3(0.0, 0.0, z)
            points = []
            for point in range(PILE_SEGMENTS + 1):
                t = point / PILE_SEGMENTS
                wave = 0.035 * math.sin(4.0 * math.pi * t + 0.7 * lane)
                points.append(start + point * segment_length * direction + wave * orthogonal)
            quaternions = newton.utils.create_parallel_transport_cable_quaternions(points)
            rod_bodies, _ = builder.add_rod(
                positions=points,
                quaternions=quaternions,
                radius=PILE_RADIUS,
                cfg=newton.ModelBuilder.ShapeConfig(density=900.0, ke=1.0e5, kd=0.0, mu=1.0),
                stretch_stiffness=5.0e5,
                bend_stiffness=2.0e2,
                bend_damping=2.0e1,
                twist_stiffness=2.0e2,
                twist_damping=2.0e1,
                color=LAYER_COLORS[layer],
                label=f"drop_layer_{layer}_cable_{lane}",
                body_frame_origin="com",
            )
            for body in rod_bodies:
                builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
            bodies_in_layer.extend(rod_bodies)
        layer_bodies.append(bodies_in_layer)

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model, layer_bodies


def _release_layer(model: newton.Model, solver: newton.solvers.SolverVBD, bodies: list[int]) -> None:
    flags = model.body_flags.numpy()
    flags[np.asarray(bodies, dtype=np.int32)] = int(newton.BodyFlags.DYNAMIC)
    model.body_flags.assign(flags)
    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)


def _simulate_pile_attempt(example, substeps: int) -> None:
    dt = PILE_FRAME_DT / substeps
    for _ in range(substeps):
        example.state_0.clear_forces()
        example.collision_pipeline.collide(example.state_0, example.contacts, dt=dt)
        example.solver.set_rigid_history_update(True)
        example.solver.step(example.state_0, example.state_1, example.control, example.contacts, dt)
        example.state_0, example.state_1 = example.state_1, example.state_0


def _record_staged_pile(device: str) -> tuple[dict, dict[str, np.ndarray]]:
    model, layer_bodies = build_staged_pile(device)
    pipeline = newton.CollisionPipeline(model, contact_matching="disabled")
    contacts = pipeline.contacts()
    verifier = newton.CollisionPipeline(model, contact_matching="disabled")
    verification_contacts = verifier.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=5,
        rigid_avbd_contact_alpha=0.0,
        rigid_contact_history=False,
        rigid_contact_inertial_stiffness_ratio=5.0,
        rigid_body_contact_buffer_size=256,
    )
    example = SimpleNamespace(
        model=model,
        solver=solver,
        collision_pipeline=pipeline,
        contacts=contacts,
        state_0=model.state(),
        state_1=model.state(),
        control=model.control(),
    )
    snapshot = cable_benchmark._CableFrameSnapshot(example)

    poses = [example.state_0.body_q.numpy().copy()]
    velocities = [example.state_0.body_qd.numpy().copy()]
    selected_substeps: list[int] = []
    replay_counts: list[int] = []
    fresh_penetration: list[float] = []
    fresh_contact_counts: list[int] = []
    query_debt: list[int] = []
    solver_list_required: list[int] = []
    frame_times_ms: list[float] = []

    for frame in range(PILE_FRAMES):
        if frame in PILE_RELEASE_FRAMES:
            _release_layer(model, solver, layer_bodies[PILE_RELEASE_FRAMES.index(frame)])

        start = time.perf_counter()
        snapshot.save()
        accepted = False
        last_penetration = float("inf")
        for attempt, bucket in enumerate(PILE_BUCKETS):
            pipeline.narrow_phase.buffer_overflow_accumulator.zero_()
            solver.body_body_contact_overflow_accumulator.zero_()
            _simulate_pile_attempt(example, bucket)
            verifier.collide(example.state_0, verification_contacts)
            gap, _ = cable_benchmark._minimum_fresh_gap(model, example.state_0, verification_contacts)
            last_penetration = 0.0 if not math.isfinite(gap) else max(0.0, -float(gap))
            q_debt = int(pipeline.narrow_phase.buffer_overflow_accumulator.numpy()[0]) or int(
                verifier.narrow_phase.buffer_overflow.numpy()[0]
            )
            s_required = int(solver.body_body_contact_overflow_accumulator.numpy()[0])
            if q_debt or s_required:
                raise RuntimeError(
                    "Cannot record a certified pile trajectory with contact capacity debt: "
                    f"frame={frame}, query_debt={q_debt}, solver_required={s_required}"
                )
            if last_penetration <= PILE_PENETRATION_TOLERANCE:
                selected_substeps.append(bucket)
                replay_counts.append(attempt)
                query_debt.append(0)
                solver_list_required.append(0)
                accepted = True
                break
            if bucket != PILE_BUCKETS[-1]:
                snapshot.restore()
        if not accepted:
            selected_substeps.append(PILE_BUCKETS[-1])
            replay_counts.append(len(PILE_BUCKETS) - 1)
            query_debt.append(0)
            solver_list_required.append(0)

        wp.synchronize()
        frame_times_ms.append((time.perf_counter() - start) * 1000.0)
        poses.append(example.state_0.body_q.numpy().copy())
        velocities.append(example.state_0.body_qd.numpy().copy())
        fresh_penetration.append(last_penetration)
        fresh_contact_counts.append(int(verification_contacts.rigid_contact_count.numpy()[0]))

    final_q = example.state_0.body_q.numpy()
    final_qd = example.state_0.body_qd.numpy()
    timed = frame_times_ms[10:]
    rates, counts = np.unique(np.asarray(selected_substeps, dtype=np.int32), return_counts=True)
    metrics = {
        "frames": PILE_FRAMES,
        "frame_dt_s": PILE_FRAME_DT,
        "layers": PILE_LAYERS,
        "cables_per_layer": PILE_CABLES_PER_LAYER,
        "segments_per_cable": PILE_SEGMENTS,
        "release_frames": PILE_RELEASE_FRAMES,
        "penetration_tolerance_m": PILE_PENETRATION_TOLERANCE,
        "maximum_fresh_penetration_m": float(max(fresh_penetration)),
        "maximum_fresh_contact_count": int(max(fresh_contact_counts)),
        "replayed_frames": int(np.count_nonzero(replay_counts)),
        "total_replays": int(np.sum(replay_counts)),
        "replay_cap_debt_frames": int(
            np.count_nonzero(np.asarray(fresh_penetration) > PILE_PENETRATION_TOLERANCE)
        ),
        "selected_substep_histogram": {
            str(int(rate)): int(count) for rate, count in zip(rates, counts, strict=True)
        },
        "query_integrity_debt_frames": int(np.count_nonzero(query_debt)),
        "solver_list_integrity_debt_frames": int(np.count_nonzero(solver_list_required)),
        "maximum_solver_list_required_capacity": int(max(solver_list_required)),
        "median_frame_time_ms": float(statistics.median(timed)),
        "p95_frame_time_ms": float(np.percentile(timed, 95)),
        "minimum_final_body_height_m": float(np.min(final_q[:, 2])),
        "maximum_final_speed_m_s": float(np.max(np.linalg.norm(final_qd[:, :3], axis=1))),
        "finite": bool(np.isfinite(final_q).all() and np.isfinite(final_qd).all()),
    }
    trace = {
        "poses": np.asarray(poses, dtype=np.float32),
        "velocities": np.asarray(velocities, dtype=np.float32),
        "selected_substeps": np.asarray(selected_substeps, dtype=np.int32),
        "replay_counts": np.asarray(replay_counts, dtype=np.int32),
        "fresh_penetration_m": np.asarray(fresh_penetration, dtype=np.float32),
        "fresh_contact_counts": np.asarray(fresh_contact_counts, dtype=np.int32),
        "layer_bodies": np.asarray(layer_bodies, dtype=np.int32),
    }
    return metrics, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", choices=("all", "bullet", "pile"), default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    report: dict[str, object] = {
        "device": args.device,
        "scope": "focused Horde gallery simulations; no unit tests",
    }
    arrays: dict[str, np.ndarray] = {}
    with wp.ScopedDevice(args.device):
        if args.only in ("all", "bullet"):
            bullet_results = []
            for event_projection in (False, True):
                metrics, trace = _record_bullet_method(args.device, event_projection)
                bullet_results.append(metrics)
                prefix = "bullet_event" if event_projection else "bullet_vbd"
                arrays.update({f"{prefix}_{key}": value for key, value in trace.items()})
            report["bullet_thin_dynamic_plate"] = bullet_results
        if args.only in ("all", "pile"):
            metrics, trace = _record_staged_pile(args.device)
            report["staged_cable_pile"] = metrics
            arrays.update({f"pile_{key}": value for key, value in trace.items()})

    np.savez_compressed(args.output_dir / "horde_gallery_traces.npz", **arrays)
    with (args.output_dir / "horde_gallery_results.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, default=_json_default)
        output.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
