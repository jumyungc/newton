# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visual experiment: single-pulley wrapped cable with equal endpoint loads.

Mirrors the second video in the VBD Cable Tension Readout report, but
exposes every relevant stiffness / damping / contact parameter so you can
sweep them interactively. Reports `state.vbd.cable_tension` live and prints
min/max/ratio per displayed frame.

Defaults match the user's request:
    contact ke = 1e4, no damping anywhere, no bending stiffness.

Run with viewer:
    .venv/bin/python newton/reports/vbd_cable_tension/experiment_single_pulley_visual.py

Common overrides:
    --mu 0.8 --stretch-ke 2e5 --contact-ke 1e4 --bend-ke 0 --iterations 32 --substeps 10
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples

# ============================================================================
# Editable scene constants (CLI flags override the ones marked --like-this)
# ============================================================================
NUM_NODES = 25  # cable nodes on the arc (24 joints)
PULLEY_RADIUS = 0.40  # m
CABLE_RADIUS = 0.035  # m
WRAP_ANGLE_DEG = 240.0  # initial cable arc spanning this angle
CABLE_DENSITY = 1000.0  # kg / m^3
ENDPOINT_LOAD = 20.0  # N applied tangentially outward at each end
CONTACT_GAP = 0.02  # m, shape config gap for contact

DEFAULT_STRETCH_KE = 1.0e4  # cable stretch stiffness [N/m]
DEFAULT_STRETCH_KD = 0.0  # cable stretch damping
DEFAULT_BEND_KE = 0.0  # cable bend stiffness [N m / rad]
DEFAULT_BEND_KD = 0.0  # cable bend damping
DEFAULT_CONTACT_KE = 1.0e4  # contact penalty stiffness
DEFAULT_CONTACT_KD = 0.0  # contact damping
DEFAULT_MU = 0.8  # contact friction coefficient
DEFAULT_ITERATIONS = 32
DEFAULT_SUBSTEPS = 10


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mu", type=float, default=DEFAULT_MU, help="contact friction coefficient")
    parser.add_argument("--stretch-ke", type=float, default=DEFAULT_STRETCH_KE, help="cable stretch stiffness [N/m]")
    parser.add_argument("--stretch-kd", type=float, default=DEFAULT_STRETCH_KD, help="cable stretch damping")
    parser.add_argument("--bend-ke", type=float, default=DEFAULT_BEND_KE, help="cable bend stiffness [N m / rad]")
    parser.add_argument("--bend-kd", type=float, default=DEFAULT_BEND_KD, help="cable bend damping")
    parser.add_argument("--contact-ke", type=float, default=DEFAULT_CONTACT_KE, help="contact penalty stiffness")
    parser.add_argument("--contact-kd", type=float, default=DEFAULT_CONTACT_KD, help="contact damping")
    parser.add_argument("--endpoint-load", type=float, default=ENDPOINT_LOAD, help="N at each end, outward tangent")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="VBD iterations per substep")
    parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS, help="substeps per rendered frame")


@wp.kernel
def _apply_endpoint_loads(
    body_left: int,
    body_right: int,
    load_left: float,
    load_right: float,
    tangent_left: wp.vec3,
    tangent_right: wp.vec3,
    body_f: wp.array[wp.spatial_vector],
):
    body_f[body_left] = wp.spatial_vector(load_left * tangent_left, wp.vec3(0.0))
    body_f[body_right] = wp.spatial_vector(load_right * tangent_right, wp.vec3(0.0))


def build_wrapped_cable(args, device: wp.Device | None) -> tuple[newton.Model, list[int], list[int], np.ndarray]:
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    builder.request_state_attributes("vbd:cable_tension")

    cable_cfg = newton.ModelBuilder.ShapeConfig()
    cable_cfg.density = CABLE_DENSITY
    cable_cfg.ke = args.contact_ke
    cable_cfg.kd = args.contact_kd
    cable_cfg.mu = args.mu
    cable_cfg.gap = CONTACT_GAP

    pulley_cfg = newton.ModelBuilder.ShapeConfig()
    pulley_cfg.density = 0.0
    pulley_cfg.ke = args.contact_ke
    pulley_cfg.kd = args.contact_kd
    pulley_cfg.mu = args.mu
    pulley_cfg.gap = CONTACT_GAP

    center = np.array([0.0, 0.0, 1.0], dtype=float)
    cyl_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * np.pi)
    builder.add_shape_cylinder(
        -1,
        xform=wp.transform(wp.vec3(*center), cyl_q),
        radius=PULLEY_RADIUS,
        half_height=0.25,
        cfg=pulley_cfg,
        color=(0.25, 0.27, 0.30),
        label="pulley",
    )

    centerline_radius = PULLEY_RADIUS + 0.75 * CABLE_RADIUS
    half_wrap = 0.5 * np.deg2rad(WRAP_ANGLE_DEG)
    # Wrap centered on top of pulley (90°); spans 0.5π ± half_wrap.
    angles = np.linspace(0.5 * np.pi + half_wrap, 0.5 * np.pi - half_wrap, NUM_NODES)
    points = [
        wp.vec3(
            float(centerline_radius * np.cos(angle)),
            0.0,
            float(center[2] + centerline_radius * np.sin(angle)),
        )
        for angle in angles
    ]
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(points)
    bodies, joints = builder.add_rod(
        points,
        quaternions,
        radius=CABLE_RADIUS,
        cfg=cable_cfg,
        stretch_stiffness=args.stretch_ke,
        stretch_damping=args.stretch_kd,
        bend_stiffness=args.bend_ke,
        bend_damping=args.bend_kd,
        label="visual_single_pulley_cable",
    )
    builder.color()
    points_np = np.array([(p[0], p[1], p[2]) for p in points], dtype=float)
    return builder.finalize(device=device), bodies, joints, points_np


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        sim_device = wp.get_device(args.device) if getattr(args, "device", None) else None
        self.model, self.bodies, self.joints, points = build_wrapped_cable(args, sim_device)
        self.joints_np = np.asarray(self.joints, dtype=int)

        self.solver = newton.solvers.SolverVBD(self.model, iterations=int(args.iterations), rigid_contact_hard=True)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        left_tangent = points[0] - points[1]
        left_tangent /= np.linalg.norm(left_tangent)
        right_tangent = points[-1] - points[-2]
        right_tangent /= np.linalg.norm(right_tangent)
        self.left_tangent = wp.vec3(*left_tangent)
        self.right_tangent = wp.vec3(*right_tangent)
        self.left_load = float(args.endpoint_load)
        self.right_load = float(args.endpoint_load)
        self.body_left = int(self.bodies[0])
        self.body_right = int(self.bodies[-1])

        self.tension_min = 0.0
        self.tension_max = 0.0
        self.tension_ratio = 1.0
        self.frame_count = 0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.82, -2.10, 1.18), pitch=-4.0, yaw=-118.0)

        print(
            f"[single-pulley visual]  mu={args.mu}  stretch_ke={args.stretch_ke:g}  "
            f"contact_ke={args.contact_ke:g}  bend_ke={args.bend_ke:g}  "
            f"iters={args.iterations}  substeps={args.substeps}  load={args.endpoint_load} N"
        )
        print(f"  Capstan kinetic upper bound exp(mu*theta) = {np.exp(args.mu * np.deg2rad(WRAP_ANGLE_DEG)):.3f}")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                _apply_endpoint_loads,
                dim=1,
                inputs=[
                    self.body_left,
                    self.body_right,
                    self.left_load,
                    self.right_load,
                    self.left_tangent,
                    self.right_tangent,
                ],
                outputs=[self.state_0.body_f],
                device=self.model.device,
            )
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame_count += 1

        tensions = self.state_0.vbd.cable_tension.numpy()[self.joints_np].astype(float)
        self.tension_min = float(np.min(tensions))
        self.tension_max = float(np.max(tensions))
        self.tension_ratio = self.tension_max / max(self.tension_min, 1.0e-12)

        if self.frame_count % 30 == 0:
            print(
                f"  t={self.sim_time:5.2f}s  min={self.tension_min:6.2f} N  "
                f"max={self.tension_max:6.2f} N  max/min={self.tension_ratio:5.2f}"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("min tension [N]", self.tension_min)
        self.viewer.log_scalar("max tension [N]", self.tension_max)
        self.viewer.log_scalar("max / min", self.tension_ratio)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    _add_arguments(parser)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
