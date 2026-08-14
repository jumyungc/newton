# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record the contact-report trajectories and metrics on an NVIDIA GPU.

This is a focused report ingredient, not a unit test.  It imports the benchmark
scenes from the experimental solver branch, runs every physics case on Horde,
and writes only portable JSON/NPZ traces for the renderer.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.cable.example_cable_twist import Example
from newton.tests import benchmark_acm_event_strategy as impact_benchmark
from newton.tests import benchmark_cable_twist_contact_strategy as cable_benchmark
from newton.tests import benchmark_vbd_active_clock_mask as clock_benchmark


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _record_sphere_plane(*, device: str, speed: float, dt: float, event_projection: bool) -> tuple[dict, dict]:
    model, body = impact_benchmark.build_sphere_plane(device, speed)
    pipeline = impact_benchmark.make_pipeline(model, speed * dt)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_contact_hard=True,
        rigid_contact_feasibility_projection=event_projection,
    )
    state_in = model.state()
    state_out = model.state()
    initial_pose = state_in.body_q.numpy()[body].copy()
    initial_velocity = state_in.body_qd.numpy()[body].copy()

    pipeline.collide(state_in, contacts, dt=dt)
    query_count = int(contacts.rigid_contact_count.numpy()[0])
    solver.step(state_in, state_out, None, contacts, dt)
    final_pose = state_out.body_q.numpy()[body].copy()
    final_velocity = state_out.body_qd.numpy()[body].copy()

    metrics = {
        "method": "event" if event_projection else "ordinary_vbd",
        "speed_m_s": speed,
        "dt_s": dt,
        "potential_displacement_m": speed * dt,
        "start_gap_m": impact_benchmark.START_GAP,
        "query_contact_count": query_count,
        "final_signed_gap_m": float(final_pose[2] - impact_benchmark.RADIUS),
        "final_normal_velocity_m_s": float(final_velocity[2]),
        "tunneled": bool(final_pose[2] < 0.0),
        "event_body_updates": int(solver.body_contact_projection_count.numpy()[body]) if event_projection else 0,
        "event_debt": int(solver.body_contact_event_debt.numpy()[body]) if event_projection else 0,
    }
    trace = {
        "initial_pose": initial_pose,
        "initial_velocity": initial_velocity,
        "final_pose": final_pose,
        "final_velocity": final_velocity,
    }
    return metrics, trace


def _record_dense_chain(
    *,
    device: str,
    body_count: int,
    dt: float,
    event_projection: bool,
    mass_ratio: float = 1.0,
    timing_repeats: int = 0,
) -> tuple[dict, dict]:
    radius = impact_benchmark.RADIUS
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 1.0e-4
    bodies: list[int] = []
    shapes: list[int] = []
    for index in range(body_count):
        body = builder.add_body(xform=wp.transform(wp.vec3(2.0 * radius * float(index), 0.0, 0.0)))
        density = 1000.0 * (mass_ratio if index % 2 == 1 else 1.0)
        shape = builder.add_shape_sphere(
            body,
            radius=radius,
            cfg=newton.ModelBuilder.ShapeConfig(density=density),
        )
        builder.body_qd[body] = (20.0 if index == 0 else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        bodies.append(body)
        shapes.append(shape)
    builder.color()
    model = builder.finalize(device=device)
    adjacent_pairs = wp.array(
        [wp.vec2i(shapes[index], shapes[index + 1]) for index in range(body_count - 1)],
        dtype=wp.vec2i,
        device=device,
    )
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="explicit",
        shape_pairs_filtered=adjacent_pairs,
        deterministic=True,
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=1.05 * 20.0 * dt
        ),
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=5,
        rigid_contact_hard=True,
        rigid_contact_feasibility_projection=event_projection,
        rigid_contact_event_waves=1,
        rigid_contact_event_closure_iterations=20,
        rigid_contact_event_closure_relaxation=1.0,
        rigid_contact_event_active_set_iterations=4,
    )
    state_in = model.state()
    state_out = model.state()
    initial_pose = state_in.body_q.numpy()[bodies].copy()
    initial_velocity = state_in.body_qd.numpy()[bodies].copy()

    samples: list[float] = []
    if timing_repeats:
        initial_q = wp.clone(model.body_q)
        initial_qd = wp.clone(model.body_qd)
        pipeline.collide(state_in, contacts, dt=dt)
        solver.step(state_in, state_out, None, contacts, dt)
        with wp.ScopedCapture(device) as capture:
            state_in.body_q.assign(initial_q)
            state_in.body_qd.assign(initial_qd)
            state_out.body_q.assign(initial_q)
            state_out.body_qd.assign(initial_qd)
            solver.reset(state_in)
            state_in.clear_forces()
            pipeline.collide(state_in, contacts, dt=dt)
            solver.step(state_in, state_out, None, contacts, dt)
        for repeat in range(timing_repeats + 1):
            begin = wp.Event(enable_timing=True)
            end = wp.Event(enable_timing=True)
            wp.record_event(begin)
            wp.capture_launch(capture.graph)
            wp.record_event(end)
            elapsed = wp.get_event_elapsed_time(begin, end)
            if repeat:
                samples.append(elapsed)
    else:
        pipeline.collide(state_in, contacts, dt=dt)
        solver.step(state_in, state_out, None, contacts, dt)

    final_pose = state_out.body_q.numpy()[bodies].copy()
    final_velocity = state_out.body_qd.numpy()[bodies].copy()
    positions = final_pose[:, 0]
    velocities = final_velocity[:, 0]
    masses = model.body_mass.numpy()[bodies]
    gaps = positions[1:] - positions[:-1] - 2.0 * radius
    initial_momentum = float(masses[0] * 20.0)
    analytic_velocity = initial_momentum / float(masses.sum())
    metrics = {
        "method": "event_pcg" if event_projection else "ordinary_vbd",
        "body_count": body_count,
        "alternating_mass_ratio": mass_ratio,
        "dt_s": dt,
        "contact_count": int(contacts.rigid_contact_count.numpy()[0]),
        "minimum_gap_m": float(gaps.min()),
        "maximum_gap_m": float(gaps.max()),
        "minimum_velocity_m_s": float(velocities.min()),
        "maximum_velocity_m_s": float(velocities.max()),
        "analytic_common_velocity_m_s": analytic_velocity,
        "maximum_common_velocity_error_m_s": float(np.max(np.abs(velocities - analytic_velocity))),
        "momentum_error_kg_m_s": float(np.dot(masses, velocities) - initial_momentum),
        "translational_energy_ratio": float(
            np.dot(masses, velocities * velocities) / (masses[0] * 20.0 * 20.0)
        ),
        "event_debt_count": int(solver.rigid_contact_event_debt_count.numpy()[0]) if event_projection else 0,
        "median_graph_gpu_ms": statistics.median(samples) if samples else None,
    }
    trace = {
        "initial_pose": initial_pose,
        "initial_velocity": initial_velocity,
        "final_pose": final_pose,
        "final_velocity": final_velocity,
        "masses": masses,
    }
    return metrics, trace


def _record_cable(
    *,
    frames: int,
    target_surface_motion: float,
    twist_rate: float,
    inertial_contact_ratio: float,
    retain_trajectory: bool,
) -> tuple[dict, dict | None]:
    viewer = newton.viewer.ViewerNull(num_frames=frames)
    example = Example(viewer, argparse.Namespace())
    example.sim_substeps = 10
    example.sim_dt = example.frame_dt / example.sim_substeps
    example.update_step_interval = 10
    example.first_twist_rates.assign(
        np.full(example.kinematic_bodies.shape[0], twist_rate, dtype=np.float32)
    )
    if inertial_contact_ratio:
        example.collision_pipeline = newton.CollisionPipeline(
            example.model,
            contact_matching="disabled",
        )
        example.contacts = example.collision_pipeline.contacts()
        example.solver = newton.solvers.SolverVBD(
            example.model,
            iterations=5,
            rigid_avbd_contact_alpha=0.0,
            rigid_contact_history=False,
            rigid_contact_inertial_stiffness_ratio=inertial_contact_ratio,
            rigid_contact_feasibility_projection=True,
        )
    else:
        example.solver.rigid_contact_feasibility_projection = True

    # Match the focused benchmark's initial contact/island priming exactly.
    # This scene is chaotic enough that omitting the initial collision query
    # changes the subsequent contact history and invalidates comparisons with
    # the benchmark evidence.
    collision_radii = cable_benchmark._body_collision_radii(example.model)
    surface_speed_override_np = np.zeros(example.model.body_count, dtype=np.float32)
    kinematic_ids = example.kinematic_bodies.numpy()
    surface_speed_override_np[kinematic_ids] = (
        abs(twist_rate) * collision_radii[kinematic_ids]
    )
    surface_speed_override = wp.array(
        surface_speed_override_np,
        dtype=float,
        device=example.model.device,
    )
    example.collision_pipeline.collide(example.state_0, example.contacts, dt=example.frame_dt)
    example.solver.update_rigid_contact_island_schedule(
        example.state_0,
        example.contacts,
        example.frame_dt,
        target_surface_motion,
        minimum_substeps=10,
        maximum_substeps=320,
        body_surface_speed_override=surface_speed_override,
    )

    adaptive_args = argparse.Namespace(
        target_surface_motion=target_surface_motion,
        twist_rate=twist_rate,
    )
    example.graph, required_substeps = cable_benchmark._capture_adaptive_contact_substeps(example, adaptive_args)
    verification_pipeline = newton.CollisionPipeline(example.model)
    verification_contacts = verification_pipeline.contacts()
    cable_bodies = np.asarray(example.cable_bodies_list, dtype=np.int32)

    positions: list[np.ndarray] = []
    fresh_gaps: list[float] = []
    missing_pairs_per_frame: list[int] = []
    selected_substeps: list[int] = []
    fresh_contact_counts: list[int] = []
    frame_times_ms: list[float] = []
    total_event_updates = 0
    frames_with_event_debt = 0

    if retain_trajectory:
        positions.append(example.state_0.body_q.numpy()[cable_bodies, :3].copy())

    for frame in range(frames):
        begin = time.perf_counter()
        example.step()
        pose = example.state_0.body_q.numpy()
        elapsed_ms = (time.perf_counter() - begin) * 1000.0
        frame_times_ms.append(elapsed_ms)
        selected_substeps.append(int(required_substeps.numpy()[0]))
        if retain_trajectory:
            positions.append(pose[cable_bodies, :3].copy())

        retained_pairs = cable_benchmark._contact_pairs(example.contacts)
        verification_pipeline.collide(example.state_0, verification_contacts)
        fresh_pairs = cable_benchmark._contact_pairs(verification_contacts)
        gap, _ = cable_benchmark._minimum_fresh_gap(example.model, example.state_0, verification_contacts)
        fresh_gaps.append(gap)
        missing_pairs_per_frame.append(len(fresh_pairs - retained_pairs))
        fresh_contact_counts.append(len(fresh_pairs))
        total_event_updates += int(example.solver.body_contact_projection_count.numpy().sum())
        frames_with_event_debt += int(example.solver.rigid_contact_event_debt_count.numpy()[0] > 0)

    timed = frame_times_ms[min(10, max(1, frames // 10)) :]
    rates, counts = np.unique(np.asarray(selected_substeps, dtype=np.int32), return_counts=True)
    finite = bool(np.isfinite(example.state_0.body_q.numpy()).all() and np.isfinite(example.state_0.body_qd.numpy()).all())
    metrics = {
        "target_surface_motion_m": target_surface_motion,
        "inertial_contact_stiffness_ratio": inertial_contact_ratio,
        "twist_rate_rad_s": twist_rate,
        "frames": frames,
        "finite": finite,
        "maximum_fresh_penetration_m": max(0.0, -float(np.min(fresh_gaps))),
        "minimum_fresh_gap_m": float(np.min(fresh_gaps)),
        "max_missing_pairs_per_frame": int(np.max(missing_pairs_per_frame)),
        "frames_with_missing_pairs": int(np.count_nonzero(missing_pairs_per_frame)),
        "median_frame_time_ms": float(np.median(timed)),
        "p95_frame_time_ms": float(np.percentile(timed, 95)),
        "selected_substep_histogram": {str(int(rate)): int(count) for rate, count in zip(rates, counts, strict=True)},
        "total_event_body_updates": total_event_updates,
        "frames_with_event_debt": frames_with_event_debt,
    }
    trace = None
    if retain_trajectory:
        trace = {
            "positions": np.asarray(positions, dtype=np.float32),
            "fresh_gap_m": np.asarray(fresh_gaps, dtype=np.float32),
            "missing_pairs": np.asarray(missing_pairs_per_frame, dtype=np.int32),
            "selected_substeps": np.asarray(selected_substeps, dtype=np.int32),
            "fresh_contact_counts": np.asarray(fresh_contact_counts, dtype=np.int32),
        }
    return metrics, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    report: dict[str, object] = {
        "device": args.device,
        "solver_branch": "codex-sol/contact-strategy",
        "scope": "focused report benchmarks; no unit tests",
    }
    arrays: dict[str, np.ndarray] = {}
    with wp.ScopedDevice(args.device):
        sphere_results = []
        for event_projection in (False, True):
            metrics, trace = _record_sphere_plane(
                device=args.device,
                speed=10000.0,
                dt=0.03,
                event_projection=event_projection,
            )
            sphere_results.append(metrics)
            prefix = "sphere_event" if event_projection else "sphere_vbd"
            arrays.update({f"{prefix}_{key}": value for key, value in trace.items()})
        report["sphere_plane_10000_m_s"] = sphere_results

        dense_results = []
        for event_projection in (False, True):
            metrics, trace = _record_dense_chain(
                device=args.device,
                body_count=16,
                dt=0.03,
                event_projection=event_projection,
                timing_repeats=30 if event_projection else 0,
            )
            dense_results.append(metrics)
            prefix = "dense_event" if event_projection else "dense_vbd"
            arrays.update({f"{prefix}_{key}": value for key, value in trace.items()})
        report["dense_equal_mass_chain"] = dense_results

        extreme_metrics, _ = _record_dense_chain(
            device=args.device,
            body_count=16,
            dt=0.03,
            event_projection=True,
            mass_ratio=1.0e6,
            timing_repeats=30,
        )
        report["dense_extreme_mass_chain"] = extreme_metrics

        baseline_metrics, baseline_trace = _record_cable(
            frames=args.frames,
            target_surface_motion=10.0e-3,
            twist_rate=20.0,
            inertial_contact_ratio=0.0,
            retain_trajectory=True,
        )
        report["cable_twist_unconditioned"] = baseline_metrics
        arrays.update(
            {f"cable_default_10mm_{key}": value for key, value in baseline_trace.items()}
        )

        cable_results = []
        for target_mm in (10.0, 5.0, 1.0):
            keep = target_mm in (10.0, 1.0)
            metrics, trace = _record_cable(
                frames=args.frames,
                target_surface_motion=target_mm * 1.0e-3,
                twist_rate=20.0,
                inertial_contact_ratio=5.0,
                retain_trajectory=keep,
            )
            cable_results.append(metrics)
            if trace is not None:
                prefix = f"cable_conditioned_{int(target_mm)}mm"
                arrays.update({f"{prefix}_{key}": value for key, value in trace.items()})
        report["cable_twist_conditioned"] = cable_results

        clock_args = argparse.Namespace(
            device=args.device,
            dt=1.0e-2,
            prime_dt=1.0e-5,
            radius=5.0e-2,
            contact_speed=2.0,
            free_speed=3.0,
            iterations=5,
            inertial_contact_ratio=5.0,
            target_surface_motion=1.0e-2,
        )
        report["active_clock"] = clock_benchmark.run(clock_args)

    np.savez_compressed(args.output_dir / "horde_traces.npz", **arrays)
    with (args.output_dir / "horde_results.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, default=_json_default)
        output.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
