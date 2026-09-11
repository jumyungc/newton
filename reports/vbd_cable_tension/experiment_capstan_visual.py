# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visual experiment: two-pulley capstan with unequal endpoint masses.

Mirrors the third video in the VBD Cable Tension Readout report, but exposes
every relevant stiffness / damping / contact parameter. Reports
`state.vbd.cable_tension` live for the middle-span joint and compares against
the ideal capstan reference T_high / T_low = exp(mu * theta) per quarter wrap.

Defaults match the user's request:
    contact ke = 1e4, no damping anywhere, no bending stiffness.

Run with viewer:
    .venv/bin/python newton/reports/vbd_cable_tension/experiment_capstan_visual.py

Common overrides:
    --mu 0.20 --stretch-ke 2e4 --contact-ke 1e4 --bend-ke 0 \
        --left-load 9.81 --right-load 18.39 --iterations 32 --substeps 10
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
PULLEY_RADIUS = 0.32  # m
CABLE_RADIUS = 0.035  # m
LEFT_PULLEY_X = -0.65  # m
RIGHT_PULLEY_X = +0.65  # m
PULLEY_Z = 1.0  # m
CABLE_DENSITY = 1000.0  # kg / m^3
CONTACT_GAP = 0.02  # m

# Ideal capstan-equation references for a quarter wrap per pulley:
#   T_high / T_low = exp(mu * pi/2)
# Defaults use mu=0.20 + left_load=9.81 N (1 kg) → middle = 9.81 * exp(0.2 * pi/2),
# right = middle * exp(0.2 * pi/2).
DEFAULT_LEFT_LOAD = 9.81
DEFAULT_RIGHT_LOAD = 18.39  # = 9.81 * exp(0.2 * pi) ≈ 18.39 (saturates capstan)

DEFAULT_STRETCH_KE = 2.0e4
DEFAULT_STRETCH_KD = 0.0
DEFAULT_BEND_KE = 0.0
DEFAULT_BEND_KD = 0.0
DEFAULT_CONTACT_KE = 1.0e4
DEFAULT_CONTACT_KD = 0.0
DEFAULT_MU = 0.20
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
    parser.add_argument("--left-load", type=float, default=DEFAULT_LEFT_LOAD, help="left endpoint downward load [N]")
    parser.add_argument("--right-load", type=float, default=DEFAULT_RIGHT_LOAD, help="right endpoint downward load [N]")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="VBD iterations per substep")
    parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS, help="substeps per rendered frame")


@wp.kernel
def _apply_endpoint_loads(
    body_left: int,
    body_right: int,
    load_left: float,
    load_right: float,
    body_f: wp.array[wp.spatial_vector],
):
    body_f[body_left] = wp.spatial_vector(wp.vec3(0.0, 0.0, -load_left), wp.vec3(0.0))
    body_f[body_right] = wp.spatial_vector(wp.vec3(0.0, 0.0, -load_right), wp.vec3(0.0))


def _capstan_path() -> tuple[list[wp.vec3], int]:
    """Build the cable centerline path: vertical drops, quarter wraps over the two
    pulleys, and a horizontal span between them.

    Returns ``(points, middle_node)`` where ``middle_node`` is an index into the
    centerline-point list whose neighbourhood lies between the two pulleys.
    ``add_rod`` produces ``len(points) - 1`` bodies; we use those bodies' first
    and last entries for the endpoint loads in the caller.
    """
    centerline_radius = PULLEY_RADIUS + 0.75 * CABLE_RADIUS
    left_center = np.array([LEFT_PULLEY_X, 0.0, PULLEY_Z])
    right_center = np.array([RIGHT_PULLEY_X, 0.0, PULLEY_Z])

    points: list[wp.vec3] = []

    # Left vertical drop (load hangs at z=0.25)
    for z in np.linspace(0.25, left_center[2], 7):
        points.append(wp.vec3(float(left_center[0] - centerline_radius), 0.0, float(z)))

    # Left pulley quarter wrap: 180° → 90°
    for angle in np.linspace(np.pi, 0.5 * np.pi, 9)[1:]:
        points.append(
            wp.vec3(
                float(left_center[0] + centerline_radius * np.cos(angle)),
                0.0,
                float(left_center[2] + centerline_radius * np.sin(angle)),
            )
        )
    middle_start = len(points) - 1

    # Middle horizontal span (between the two pulley tops)
    for x in np.linspace(left_center[0], right_center[0], 13)[1:]:
        points.append(wp.vec3(float(x), 0.0, float(left_center[2] + centerline_radius)))
    middle_node = middle_start + 6

    # Right pulley quarter wrap: 90° → 0°
    for angle in np.linspace(0.5 * np.pi, 0.0, 9)[1:]:
        points.append(
            wp.vec3(
                float(right_center[0] + centerline_radius * np.cos(angle)),
                0.0,
                float(right_center[2] + centerline_radius * np.sin(angle)),
            )
        )

    # Right vertical drop (load hangs at z=0.25)
    for z in np.linspace(right_center[2], 0.25, 7)[1:]:
        points.append(wp.vec3(float(right_center[0] + centerline_radius), 0.0, float(z)))

    return points, middle_node


def build_capstan(args, device: wp.Device | None) -> tuple[newton.Model, list[int], list[int], int]:
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

    cyl_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * np.pi)
    for cx in (LEFT_PULLEY_X, RIGHT_PULLEY_X):
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(cx, 0.0, PULLEY_Z), cyl_q),
            radius=PULLEY_RADIUS,
            half_height=0.25,
            cfg=pulley_cfg,
            color=(0.25, 0.27, 0.30),
            label="capstan_pulley",
        )

    points, middle_node = _capstan_path()
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
        label="visual_capstan_cable",
    )
    builder.color()
    return builder.finalize(device=device), bodies, joints, middle_node


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
        self.model, self.bodies, self.joints, middle_node = build_capstan(args, sim_device)
        self.joints_np = np.asarray(self.joints, dtype=int)

        self.solver = newton.solvers.SolverVBD(self.model, iterations=int(args.iterations), rigid_contact_hard=True)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # add_rod produces len(points)-1 bodies and len(points)-2 joints; pick endpoints
        # straight from the body list to avoid off-by-one with the centerline indices.
        self.body_left = int(self.bodies[0])
        self.body_right = int(self.bodies[-1])
        self.left_load = float(args.left_load)
        self.right_load = float(args.right_load)

        # Each pulley wraps a quarter (pi/2 rad). Ideal capstan saturates with:
        #   T_middle = T_low * exp(mu * pi/2),   T_high = T_low * exp(mu * pi)
        wrap = 0.5 * np.pi
        self.t_low_ref = min(self.left_load, self.right_load)
        self.t_high_ref = self.t_low_ref * float(np.exp(args.mu * 2.0 * wrap))
        self.t_mid_ref = self.t_low_ref * float(np.exp(args.mu * wrap))

        # Joint i connects body i and body i+1, so joint indices are 0..len(joints)-1.
        num_joints = len(self.joints)
        self.joint_left_idx = 0
        self.joint_middle_idx = max(0, min(middle_node, num_joints - 1))
        self.joint_right_idx = num_joints - 1

        self.tension_left = 0.0
        self.tension_middle = 0.0
        self.tension_right = 0.0
        self.frame_count = 0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -2.6, 1.10), pitch=-3.0, yaw=-90.0)

        print(
            f"[capstan visual]  mu={args.mu}  stretch_ke={args.stretch_ke:g}  "
            f"contact_ke={args.contact_ke:g}  bend_ke={args.bend_ke:g}  "
            f"iters={args.iterations}  substeps={args.substeps}"
        )
        print(
            f"  load_left={self.left_load:.3f} N  load_right={self.right_load:.3f} N  "
            f"capstan-ideal: T_low={self.t_low_ref:.3f}  T_mid={self.t_mid_ref:.3f}  "
            f"T_high={self.t_high_ref:.3f}"
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                _apply_endpoint_loads,
                dim=1,
                inputs=[self.body_left, self.body_right, self.left_load, self.right_load],
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
        self.tension_left = float(tensions[self.joint_left_idx])
        self.tension_middle = float(tensions[self.joint_middle_idx])
        self.tension_right = float(tensions[self.joint_right_idx])

        if self.frame_count % 30 == 0:
            print(
                f"  t={self.sim_time:5.2f}s  T_left={self.tension_left:6.2f}  "
                f"T_mid={self.tension_middle:6.2f}  T_right={self.tension_right:6.2f}    "
                f"(ideal mid {self.t_mid_ref:.2f})"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("T_left  [N]", self.tension_left)
        self.viewer.log_scalar("T_mid   [N]", self.tension_middle)
        self.viewer.log_scalar("T_right [N]", self.tension_right)
        self.viewer.log_scalar("T_mid - ideal [N]", self.tension_middle - self.t_mid_ref)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    _add_arguments(parser)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
