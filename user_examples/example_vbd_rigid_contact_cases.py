# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Rigid Contact Benchmark Cases
#
# Visual/perf entry point for the rigid contact benchmark scenes used in the
# VBD rigid contact report.
#
# Visual:
#   python user_examples/example_vbd_rigid_contact_cases.py --case box_stack_16
#
# Headless timing:
#   python user_examples/example_vbd_rigid_contact_cases.py --case box_stack_16 --perf --device cuda:0
#
###########################################################################

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import newton
import newton.examples
from newton.examples.cable.example_cable_pile import Example as CablePileExample
from user_examples.example_cable_nv72_tray import Example as NV72Example
from user_examples.example_cable_nv72_tray import _advance_time, _twist_kinematic_bodies


@dataclass(frozen=True)
class CaseSpec:
    family: str
    config: dict


RIGID_CASES: dict[str, CaseSpec] = {
    "box_stack_16": CaseSpec(
        "rigid",
        {
            "kind": "stack",
            "boxes": 16,
            "iterations": 120,
            "shape_ke": 10000.0,
            "ground_ke": 30000.0,
            "gap": 0.5,
            "camera": ((4.0, -8.0, 5.5), -18.0, 118.0),
            "hx": 0.25,
            "hy": 0.25,
            "hz": 0.25,
            "density": 1000.0,
        },
    ),
    "box_stack_16_gap010": CaseSpec(
        "rigid",
        {
            "kind": "stack",
            "boxes": 16,
            "iterations": 120,
            "shape_ke": 10000.0,
            "ground_ke": 30000.0,
            "gap": 0.1,
            "camera": ((4.0, -8.0, 5.5), -18.0, 118.0),
            "hx": 0.25,
            "hy": 0.25,
            "hz": 0.25,
            "density": 1000.0,
        },
    ),
    "box_pyramid_16": CaseSpec(
        "rigid",
        {
            "kind": "single_pyramid",
            "rows": 16,
            "iterations": 120,
            "shape_ke": 10000.0,
            "ground_ke": 30000.0,
            "gap": 0.5,
            "camera": ((0.0, -24.0, 10.0), -8.0, 90.0),
            "hx": 0.5,
            "hy": 0.25,
            "hz": 0.25,
            "density": 1000.0,
        },
    ),
    "box_pyramid_16_gap010": CaseSpec(
        "rigid",
        {
            "kind": "single_pyramid",
            "rows": 16,
            "iterations": 120,
            "shape_ke": 10000.0,
            "ground_ke": 30000.0,
            "gap": 0.1,
            "camera": ((0.0, -24.0, 10.0), -8.0, 90.0),
            "hx": 0.5,
            "hy": 0.25,
            "hz": 0.25,
            "density": 1000.0,
        },
    ),
    "rigid_pyramid_field": CaseSpec(
        "rigid",
        {
            "kind": "pyramid_field",
            "rows": 8,
            "field": (2, 2),
            "iterations": 120,
            "shape_ke": 10000.0,
            "ground_ke": 30000.0,
            "gap": 0.5,
            "camera": ((7.0, -20.0, 9.0), -12.0, 120.0),
            "hx": 0.5,
            "hy": 0.25,
            "hz": 0.25,
            "density": 1000.0,
        },
    ),
    "rigid_pyramid_field_gap010": CaseSpec(
        "rigid",
        {
            "kind": "pyramid_field",
            "rows": 8,
            "field": (2, 2),
            "iterations": 120,
            "shape_ke": 10000.0,
            "ground_ke": 30000.0,
            "gap": 0.1,
            "camera": ((7.0, -20.0, 9.0), -12.0, 120.0),
            "hx": 0.5,
            "hy": 0.25,
            "hz": 0.25,
            "density": 1000.0,
        },
    ),
}

CABLE_CASES: dict[str, CaseSpec] = {
    "cable_pile": CaseSpec(
        "cable",
        {
            "layers": 10,
            "lanes": 10,
            "iterations": 5,
            "substeps": 10,
            "contact_buffer": 256,
            "camera": ((2.5, -4.0, 2.0), -20.0, 125.0),
        },
    ),
    "dense_cable_pile_x2": CaseSpec(
        "cable",
        {
            "layers": 20,
            "lanes": 20,
            "iterations": 5,
            "substeps": 10,
            "contact_buffer": 512,
            "camera": ((2.5, -4.0, 2.0), -20.0, 125.0),
        },
    ),
}

NV72_CASES: dict[str, CaseSpec] = {
    "real_nv72_twist": CaseSpec(
        "nv72",
        {
            "hold_twist": False,
            "iterations": 16,
            "substeps": 2,
            "cable_limit": 24,
            "bracket_rows": 4,
            "bracket_columns": 6,
            "source_twist_amplitude_deg": 40.0,
            "source_twist_frequency_hz": 0.7,
            "contact_buffer": 512,
        },
    ),
    "real_nv72_twisted_hold": CaseSpec(
        "nv72",
        {
            "hold_twist": True,
            "iterations": 16,
            "substeps": 2,
            "cable_limit": 24,
            "bracket_rows": 4,
            "bracket_columns": 6,
            "source_twist_amplitude_deg": 40.0,
            "source_twist_frequency_hz": 0.0,
            "contact_buffer": 512,
        },
    ),
}

CASE_SPECS: dict[str, CaseSpec] = {}
CASE_SPECS.update(RIGID_CASES)
CASE_SPECS.update(CABLE_CASES)
CASE_SPECS.update(NV72_CASES)

CONTACT_MODE_CHOICES = (
    "auto",
    "fused",
    "fused_block32",
    "fused_block64",
    "fused_block128",
    "fused_tile4",
    "fused_tile8",
    "fused_tile16",
    "fused_tile32",
    "tile",
    "atomic",
    "inline_reduce",
)


def _iterations(args: argparse.Namespace, default_value: int) -> int:
    return max(1, int(args.iterations if args.iterations is not None else default_value))


def _substeps(args: argparse.Namespace, default_value: int) -> int:
    return max(1, int(args.substeps if args.substeps is not None else default_value))


def _contact_settings(args: argparse.Namespace) -> tuple[str, int]:
    choice = args.contact_mode
    if choice.startswith("fused_tile"):
        return "fused_tile", int(choice.removeprefix("fused_tile"))
    if choice in ("auto", "fused_block32", "fused_block64", "fused_block128"):
        return choice, 32
    return choice, max(1, int(args.tile_width))


def _solver_contact_kwargs(args: argparse.Namespace) -> dict:
    mode, tile_width = _contact_settings(args)
    kwargs = {
        "rigid_contact_accumulation_mode": mode,
        "rigid_contact_tile_width": tile_width,
    }
    if args.fuse_max is not None:
        kwargs["rigid_contact_fuse_max"] = int(args.fuse_max)
    return kwargs


def _configure_camera(viewer, camera) -> None:
    if viewer is None or not hasattr(viewer, "set_camera"):
        return
    pos, pitch, yaw = camera
    viewer.set_camera(pos=wp.vec3(*pos), pitch=pitch, yaw=yaw)


def _safe_contact_stats(model, solver, contacts) -> dict:
    wp.synchronize()
    out = {
        "contact_count": None,
        "max_contacts_per_body": None,
        "mean_contacts_per_body": None,
    }
    contact_count = getattr(contacts, "rigid_contact_count", None)
    if contact_count is not None:
        out["contact_count"] = int(contact_count.numpy()[0])
    body_counts = getattr(solver, "body_body_contact_counts", None)
    if body_counts is not None:
        counts = body_counts.numpy()
        if counts.size:
            out["max_contacts_per_body"] = int(np.max(counts))
            out["mean_contacts_per_body"] = float(np.mean(counts))
    return out


def _state_quality(state) -> dict:
    wp.synchronize()
    body_q = state.body_q.numpy() if state.body_q is not None else np.zeros((0, 7), dtype=np.float32)
    body_qd = state.body_qd.numpy() if state.body_qd is not None else np.zeros((0, 6), dtype=np.float32)
    return {
        "finite": bool(np.isfinite(body_q).all() and np.isfinite(body_qd).all()),
        "final_min_z": float(np.min(body_q[:, 2])) if body_q.size else 0.0,
        "final_max_z": float(np.max(body_q[:, 2])) if body_q.size else 0.0,
        "final_speed_max": float(np.max(np.linalg.norm(body_qd[:, :3], axis=1))) if body_qd.size else 0.0,
    }


class RigidBenchmarkScene:
    def __init__(self, viewer, args: argparse.Namespace, case: str, spec: CaseSpec):
        self.viewer = viewer
        self.args = args
        self.case = case
        self.config = spec.config

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = _substeps(args, 1)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = _iterations(args, int(self.config["iterations"]))
        self.dynamic_bodies: list[int] = []
        self.cuda_graph = False

        self.shape_ke = float(self.config["shape_ke"])
        self.ground_ke = float(self.config["ground_ke"])
        self.gap = float(self.config["gap"])
        self.hx = float(self.config.get("hx", 0.5))
        self.hy = float(self.config.get("hy", 0.25))
        self.hz = float(self.config.get("hz", 0.25))

        builder = newton.ModelBuilder(gravity=-10.0)
        builder.default_shape_cfg.density = float(self.config.get("density", 1.0))
        builder.default_shape_cfg.mu = 0.5
        builder.default_shape_cfg.ke = self.ground_ke
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.gap = self.gap
        builder.add_ground_plane()
        builder.default_shape_cfg.ke = self.shape_ke

        kind = self.config["kind"]
        if kind == "stack":
            self.expected_top_z = self.hz + 2.0 * self.hz * float(int(self.config["boxes"]) - 1)
            self._add_stack(builder, int(self.config["boxes"]), 0.0, 0.0, "s0")
        elif kind == "single_pyramid":
            self.expected_top_z = self.hz + 2.0 * self.hz * float(int(self.config["rows"]) - 1)
            self._add_pyramid(builder, int(self.config["rows"]), 0.0, 0.0, "p0")
        elif kind == "pyramid_field":
            rows = int(self.config["rows"])
            self.expected_top_z = self.hz + 2.0 * self.hz * float(rows - 1)
            field_x, field_y = self.config["field"]
            spacing_x = float(rows) * 1.15
            spacing_y = 1.7
            for ix in range(int(field_x)):
                for iy in range(int(field_y)):
                    ox = (float(ix) - 0.5 * float(field_x - 1)) * spacing_x
                    oy = (float(iy) - 0.5 * float(field_y - 1)) * spacing_y
                    self._add_pyramid(builder, rows, ox, oy, f"p{ix}_{iy}")
        else:
            raise ValueError(f"Unsupported rigid benchmark kind: {kind!r}")

        builder.color()
        self.model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            self.model,
            deterministic=True,
            contact_matching="latest",
            contact_matching_pos_threshold=0.00075,
            contact_matching_normal_dot_threshold=0.995,
        )
        self.contacts = self.model.contacts(collision_pipeline=pipeline)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.iterations,
            rigid_contact_hard=True,
            rigid_contact_history=True,
            rigid_avbd_contact_alpha=0.0,
            rigid_avbd_linear_beta=0.0,
            rigid_avbd_angular_beta=0.0,
            rigid_avbd_gamma=0.999,
            rigid_contact_k_start=1.0,
            rigid_body_contact_buffer_size=512,
            **_solver_contact_kwargs(args),
        )

        if viewer is not None:
            viewer.set_model(self.model)
        _configure_camera(viewer, self.config["camera"])
        self.capture()

    def _add_stack(self, builder, boxes: int, offset_x: float, offset_y: float, prefix: str) -> None:
        for i in range(boxes):
            z = self.hz + 2.0 * self.hz * float(i)
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(offset_x, offset_y, z), q=wp.quat_identity()),
                label=f"{prefix}_box_{i}",
            )
            builder.add_shape_box(body, hx=self.hx, hy=self.hy, hz=self.hz)
            self.dynamic_bodies.append(body)

    def _add_pyramid(self, builder, rows: int, offset_x: float, offset_y: float, prefix: str) -> None:
        for row in range(rows):
            count = rows - row
            z = self.hz + 2.0 * self.hz * float(row)
            for col in range(count):
                x = float(col) * 1.01 + float(row) * 0.5 - float(rows) / 2.0 + offset_x
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(x, offset_y, z), q=wp.quat_identity()),
                    label=f"{prefix}_box_{row}_{col}",
                )
                builder.add_shape_box(body, hx=self.hx, hy=self.hy, hz=self.hz)
                self.dynamic_bodies.append(body)

    def capture(self) -> None:
        if self.args.no_cuda_graph or not self.solver.device.is_cuda:
            self.graph = None
            self.cuda_graph = False
            return
        try:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
            self.cuda_graph = True
        except Exception as exc:
            print(f"[graph-disabled] {self.case}: {exc}", flush=True)
            self.graph = None
            self.cuda_graph = False

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def contact_stats(self) -> dict:
        return _safe_contact_stats(self.model, self.solver, self.contacts)

    def quality(self) -> dict:
        out = _state_quality(self.state_0)
        body_q = self.state_0.body_q.numpy()
        if self.dynamic_bodies:
            dynamic_q = body_q[self.dynamic_bodies]
            out["dynamic_min_z"] = float(np.min(dynamic_q[:, 2]))
            out["dynamic_max_z"] = float(np.max(dynamic_q[:, 2]))
            out["top_height_ratio"] = (
                float(np.max(dynamic_q[:, 2]) / self.expected_top_z) if self.expected_top_z > 0.0 else 1.0
            )
        return out

    def test_final(self) -> None:
        q = self.state_0.body_q.numpy()
        assert np.isfinite(q).all(), "Non-finite rigid body state"
        assert np.min(q[:, 2]) > -0.5, "Rigid body fell far below the ground"


class CablePileBenchmarkScene:
    def __init__(self, viewer, args: argparse.Namespace, case: str, spec: CaseSpec):
        self.viewer = viewer
        self.args = args
        self.case = case
        self.config = spec.config
        self.cuda_graph = False

        original_capture = CablePileExample.capture
        CablePileExample.capture = lambda inst: setattr(inst, "graph", None)
        try:
            inner = CablePileExample(
                viewer,
                args=None,
                layers=int(self.config["layers"]),
                lanes_per_layer=int(self.config["lanes"]),
            )
        finally:
            CablePileExample.capture = original_capture

        self.inner = inner
        inner.sim_substeps = _substeps(args, int(self.config["substeps"]))
        inner.sim_iterations = _iterations(args, int(self.config["iterations"]))
        inner.sim_dt = inner.frame_dt / inner.sim_substeps
        inner.contacts = inner.model.contacts(collision_pipeline=newton.CollisionPipeline(inner.model, contact_matching="latest"))
        inner.solver = newton.solvers.SolverVBD(
            inner.model,
            iterations=inner.sim_iterations,
            rigid_contact_hard=True,
            rigid_contact_history=True,
            rigid_avbd_contact_alpha=0.0,
            rigid_avbd_linear_beta=0.0,
            rigid_avbd_angular_beta=0.0,
            rigid_avbd_gamma=0.999,
            rigid_contact_k_start=1.0,
            rigid_body_contact_buffer_size=int(self.config["contact_buffer"]),
            **_solver_contact_kwargs(args),
        )
        _configure_camera(viewer, self.config["camera"])
        self.capture()

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def capture(self) -> None:
        if self.args.no_cuda_graph:
            self.inner.graph = None
            self.cuda_graph = False
            return
        self.inner.capture()
        self.cuda_graph = bool(self.inner.graph)

    def step(self) -> None:
        self.inner.step()

    def render(self) -> None:
        self.inner.render()

    def contact_stats(self) -> dict:
        return _safe_contact_stats(self.inner.model, self.inner.solver, self.inner.contacts)

    def quality(self) -> dict:
        return _state_quality(self.inner.state_0)

    def test_final(self) -> None:
        self.inner.test_final()


def _make_nv72_args(args: argparse.Namespace, config: dict) -> SimpleNamespace:
    values = vars(args).copy()
    values.update(
        {
            "layout": "bracket-array",
            "cable_limit": int(config["cable_limit"]),
            "bracket_rows": int(config["bracket_rows"]),
            "bracket_columns": int(config["bracket_columns"]),
            "substeps": _substeps(args, int(config["substeps"])),
            "iterations": _iterations(args, int(config["iterations"])),
            "drive_source_twist": True,
            "drive_receiver": False,
            "source_twist_amplitude_deg": float(config["source_twist_amplitude_deg"]),
            "source_twist_frequency_hz": float(config["source_twist_frequency_hz"]),
            "rigid_body_contact_buffer_size": int(config["contact_buffer"]),
        }
    )
    return SimpleNamespace(**values)


class NV72BenchmarkScene:
    def __init__(self, viewer, args: argparse.Namespace, case: str, spec: CaseSpec):
        self.viewer = viewer
        self.args = args
        self.case = case
        self.config = spec.config
        self.hold_twist = bool(self.config["hold_twist"])
        self.cuda_graph = False

        original_capture = NV72Example.capture
        NV72Example.capture = lambda inst: setattr(inst, "graph", None)
        try:
            inner = NV72Example(viewer, _make_nv72_args(args, self.config))
        finally:
            NV72Example.capture = original_capture

        self.inner = inner
        inner.contacts = inner.model.contacts(
            collision_pipeline=newton.CollisionPipeline(
                inner.model,
                deterministic=True,
                contact_matching="latest",
                contact_matching_pos_threshold=0.00075,
                contact_matching_normal_dot_threshold=0.995,
            )
        )
        inner.solver = newton.solvers.SolverVBD(
            inner.model,
            iterations=inner.sim_iterations,
            friction_epsilon=1.0e-4,
            rigid_contact_hard=True,
            rigid_contact_history=True,
            rigid_avbd_contact_alpha=0.0,
            rigid_avbd_linear_beta=0.0,
            rigid_avbd_angular_beta=0.0,
            rigid_avbd_gamma=0.999,
            rigid_contact_k_start=1.0,
            rigid_body_contact_buffer_size=int(self.config["contact_buffer"]),
            **_solver_contact_kwargs(args),
        )

        if self.hold_twist:
            inner._source_twist_angle.assign([inner.source_twist_amplitude])
            inner._source_twist_angular_speed.assign([0.0])
            inner._last_source_twist_angle = inner.source_twist_amplitude
        _configure_camera(viewer, ((-0.26, 0.07, 0.12), -15.0, -10.0))
        self.capture()

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def _apply_held_source_twist(self) -> None:
        inner = self.inner
        if (
            not inner.drive_source_twist
            or inner._source_driven_body_ids_wp is None
            or inner._source_driven_body_rest_q_wp is None
            or inner.state_0.body_q is None
            or inner.state_0.body_qd is None
        ):
            return
        wp.launch(
            kernel=_twist_kinematic_bodies,
            dim=len(inner._source_driven_body_ids),
            inputs=(
                inner.state_0.body_q,
                inner.state_0.body_qd,
                inner._source_driven_body_ids_wp,
                inner._source_driven_body_rest_q_wp,
                inner._source_twist_pivot,
                inner._source_twist_angle,
                inner._source_twist_angular_speed,
            ),
            device=inner.model.device,
        )

    def capture(self) -> None:
        inner = self.inner
        if self.args.no_cuda_graph or not inner.solver.device.is_cuda:
            inner.graph = None
            self.cuda_graph = False
            return
        try:
            if not self.hold_twist:
                inner._graph_drive_time.zero_()
            with wp.ScopedCapture() as cap:
                graph_time = inner._graph_drive_time if (inner.drive_source_twist and not self.hold_twist) else None
                self.simulate(graph_time=graph_time)
            inner.graph = cap.graph
            self.cuda_graph = True
        except Exception as exc:
            print(f"[graph-disabled] {self.case}: {exc}", flush=True)
            inner.graph = None
            self.cuda_graph = False

    def simulate(self, graph_time: wp.array | None = None) -> None:
        inner = self.inner
        for substep in range(inner.sim_substeps):
            drive_time = inner.sim_time + float(substep) * inner.sim_dt
            if self.hold_twist:
                self._apply_held_source_twist()
            elif graph_time is None:
                inner._apply_source_twist(drive_time)
            else:
                inner._apply_source_twist_from_graph_time(graph_time, float(substep) * inner.sim_dt)
            inner._apply_receiver_drive(drive_time)
            inner.state_0.clear_forces()
            if inner.viewer is not None:
                inner.viewer.apply_forces(inner.state_0)
            inner.model.collide(inner.state_0, inner.contacts)
            inner.solver.step(inner.state_0, inner.state_1, inner.control, inner.contacts, inner.sim_dt)
            inner.state_0, inner.state_1 = inner.state_1, inner.state_0
        if graph_time is not None:
            wp.launch(kernel=_advance_time, dim=1, inputs=(graph_time, inner.frame_dt), device=inner.model.device)

    def step(self) -> None:
        inner = self.inner
        drive_time = inner.sim_time + float(inner.sim_substeps - 1) * inner.sim_dt
        if inner.graph:
            wp.capture_launch(inner.graph)
            if inner.drive_source_twist and not self.hold_twist:
                inner._last_source_twist_angle = inner.source_twist_amplitude * math.sin(
                    2.0 * math.pi * inner.source_twist_frequency * drive_time
                )
        else:
            self.simulate()
        inner.sim_time += inner.frame_dt
        if self.hold_twist:
            inner._last_source_twist_angle = inner.source_twist_amplitude
        inner._update_metrics()

    def render(self) -> None:
        self.inner.render()

    def contact_stats(self) -> dict:
        return _safe_contact_stats(self.inner.model, self.inner.solver, self.inner.contacts)

    def quality(self) -> dict:
        self.inner._update_metrics()
        out = _state_quality(self.inner.state_0)
        out["hold_twist"] = self.hold_twist
        out["source_twist_amplitude_deg"] = float(math.degrees(self.inner.source_twist_amplitude))
        out["source_twist_frequency_hz"] = float(self.inner.source_twist_frequency)
        return out

    def test_final(self) -> None:
        self.inner.test_final()


class Example:
    def __init__(self, viewer, args: argparse.Namespace):
        self.viewer = viewer
        self.args = args
        self.case = args.case
        spec = CASE_SPECS[self.case]
        if spec.family == "rigid":
            self.scene = RigidBenchmarkScene(viewer, args, self.case, spec)
        elif spec.family == "cable":
            self.scene = CablePileBenchmarkScene(viewer, args, self.case, spec)
        elif spec.family == "nv72":
            self.scene = NV72BenchmarkScene(viewer, args, self.case, spec)
        else:
            raise ValueError(f"Unsupported benchmark family: {spec.family!r}")

    def __getattr__(self, name: str):
        return getattr(self.scene, name)

    def step(self) -> None:
        self.scene.step()

    def render(self) -> None:
        self.scene.render()

    def contact_stats(self) -> dict:
        return self.scene.contact_stats()

    def quality(self) -> dict:
        return self.scene.quality()

    def test_final(self) -> None:
        self.scene.test_final()

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = newton.examples.create_parser()
        parser.add_argument("--case", default="box_stack_16", choices=sorted(CASE_SPECS), help="Benchmark scene to run.")
        parser.add_argument(
            "--contact-mode",
            default="auto",
            choices=CONTACT_MODE_CHOICES,
            help="Rigid contact accumulation mode to use.",
        )
        parser.add_argument("--tile-width", type=int, default=4, help="Tile width for the generic tile mode.")
        parser.add_argument("--fuse-max", type=int, default=None, help="Override auto/hybrid fused contact threshold.")
        parser.add_argument("--iterations", type=int, default=None, help="Override the case's benchmark iteration count.")
        parser.add_argument("--substeps", type=int, default=None, help="Override the case's benchmark substep count.")
        parser.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDA graph capture for this run.")
        parser.add_argument("--perf", action="store_true", help="Run headless timing instead of the viewer loop.")
        parser.add_argument("--perf-frames", type=int, default=500, help="Frames measured by --perf.")
        parser.add_argument("--warmup-frames", type=int, default=20, help="Warmup frames before --perf timing.")
        parser.add_argument("--sample-stride", type=int, default=10, help="Contact stat sampling stride for --perf.")
        return parser


def run_perf(example: Example, args: argparse.Namespace) -> dict:
    for _ in range(max(0, int(args.warmup_frames))):
        example.step()
    wp.synchronize()

    samples = []
    t0 = time.perf_counter()
    for frame in range(max(1, int(args.perf_frames))):
        example.step()
        if frame % max(1, int(args.sample_stride)) == 0 or frame == int(args.perf_frames) - 1:
            samples.append(example.contact_stats())
    wp.synchronize()
    elapsed = time.perf_counter() - t0

    def stat(key: str, reducer, default=None):
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        return reducer(values) if values else default

    result = {
        "case": args.case,
        "contact_mode": args.contact_mode,
        "device": str(wp.get_device()),
        "frames": int(args.perf_frames),
        "warmup_frames": int(args.warmup_frames),
        "elapsed_s": elapsed,
        "ms_per_frame": 1000.0 * elapsed / max(1, int(args.perf_frames)),
        "fps": max(1, int(args.perf_frames)) / elapsed if elapsed > 0.0 else None,
        "cuda_graph": bool(getattr(example.scene, "cuda_graph", False)),
        "contact_count_mean": stat("contact_count", lambda v: float(np.mean(v))),
        "contact_count_max": stat("contact_count", lambda v: int(np.max(v))),
        "max_contacts_per_body": stat("max_contacts_per_body", lambda v: int(np.max(v))),
        "mean_contacts_per_body": stat("mean_contacts_per_body", lambda v: float(np.mean(v))),
        "quality": example.quality(),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--perf", action="store_true")
    pre_args, _ = pre_parser.parse_known_args()

    parser = Example.create_parser()
    if pre_args.perf:
        parser.set_defaults(viewer="null", num_frames=500)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if args.perf:
        run_perf(example, args)
        viewer.close()
    else:
        newton.examples.run(example, args)


if __name__ == "__main__":
    main()
