# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Tension Capstan
#
# Requests state.vbd.cable_tension for a two-pulley, capstan-style cable
# diagnostic. Endpoint loads are prescribed; the middle span is read from the
# VBD cable-tension output.
#
# Command: uv run -m newton.examples cable_tension_capstan
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

GRAVITY = 9.81
CABLE_COLOR = (0.92, 0.72, 0.16)
ANALYTIC_CONTACT_STICK_EPS = 0.0


@wp.kernel
def apply_endpoint_loads(
    first_body: int,
    last_body: int,
    left_load: float,
    right_load: float,
    left_tangent: wp.vec3,
    right_tangent: wp.vec3,
    body_f: wp.array[wp.spatial_vector],
):
    body_f[first_body] = wp.spatial_vector(left_load * left_tangent, wp.vec3(0.0))
    body_f[last_body] = wp.spatial_vector(right_load * right_tangent, wp.vec3(0.0))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.mu = 0.20
        self.left_load = GRAVITY
        self.middle_reference = self.left_load * np.exp(self.mu * 0.5 * np.pi)
        self.right_load = self.middle_reference * np.exp(self.mu * 0.5 * np.pi)

        builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
        builder.request_state_attributes("vbd:cable_tension")

        cable_radius = 0.035
        pulley_radius = 0.32

        cable_cfg = newton.ModelBuilder.ShapeConfig()
        cable_cfg.density = 1000.0
        cable_cfg.ke = 1.0e4
        cable_cfg.kd = 0.0
        cable_cfg.mu = self.mu
        cable_cfg.gap = 0.02

        pulley_cfg = newton.ModelBuilder.ShapeConfig()
        pulley_cfg.density = 0.0
        pulley_cfg.ke = 1.0e4
        pulley_cfg.kd = 0.0
        pulley_cfg.mu = self.mu
        pulley_cfg.gap = 0.02

        cylinder_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * np.pi)
        for center_x in (-0.65, 0.65):
            builder.add_shape_cylinder(
                -1,
                xform=wp.transform(wp.vec3(center_x, 0.0, 1.0), cylinder_rotation),
                radius=pulley_radius,
                half_height=0.25,
                cfg=pulley_cfg,
                color=(0.25, 0.27, 0.30),
                label="capstan_pulley",
            )

        centerline_radius = pulley_radius + 0.75 * cable_radius
        left_center = np.array([-0.65, 0.0, 1.0], dtype=float)
        right_center = np.array([0.65, 0.0, 1.0], dtype=float)

        points = []
        for z in np.linspace(0.25, left_center[2], 7):
            points.append(wp.vec3(float(left_center[0] - centerline_radius), 0.0, float(z)))
        for angle in np.linspace(np.pi, 0.5 * np.pi, 9)[1:]:
            points.append(
                wp.vec3(
                    float(left_center[0] + centerline_radius * np.cos(angle)),
                    0.0,
                    float(left_center[2] + centerline_radius * np.sin(angle)),
                )
            )
        middle_start = len(points) - 1
        for x in np.linspace(left_center[0], right_center[0], 13)[1:]:
            points.append(wp.vec3(float(x), 0.0, float(left_center[2] + centerline_radius)))
        for angle in np.linspace(0.5 * np.pi, 0.0, 9)[1:]:
            points.append(
                wp.vec3(
                    float(right_center[0] + centerline_radius * np.cos(angle)),
                    0.0,
                    float(right_center[2] + centerline_radius * np.sin(angle)),
                )
            )
        for z in np.linspace(right_center[2], 0.25, 7)[1:]:
            points.append(wp.vec3(float(right_center[0] + centerline_radius), 0.0, float(z)))

        quaternions = newton.utils.create_parallel_transport_cable_quaternions(points)
        bodies, joints = builder.add_rod(
            points,
            quaternions,
            radius=cable_radius,
            cfg=cable_cfg,
            stretch_stiffness=1.0e4,
            stretch_damping=0.0,
            bend_stiffness=0.0,
            bend_damping=0.0,
            label="capstan_two_pulley_cable",
        )
        for shape_index, shape_body in enumerate(builder.shape_body):
            if shape_body in bodies:
                builder.shape_color[shape_index] = CABLE_COLOR
        builder.color()

        self.model = builder.finalize()
        self.bodies = bodies
        self.joints = joints
        self.points = np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=float)
        self.regions = {
            "left": slice(0, 1),
            "middle": slice(middle_start + 6, middle_start + 7),
            "right": slice(-1, None),
        }
        left_tangent = self.points[0] - self.points[1]
        left_tangent = left_tangent / np.linalg.norm(left_tangent)
        right_tangent = self.points[-1] - self.points[-2]
        right_tangent = right_tangent / np.linalg.norm(right_tangent)
        self.left_tangent = wp.vec3(*left_tangent)
        self.right_tangent = wp.vec3(*right_tangent)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=32,
            rigid_contact_hard=True,
            rigid_contact_stick_motion_eps=ANALYTIC_CONTACT_STICK_EPS,
            rigid_contact_stick_freeze_translation_eps=ANALYTIC_CONTACT_STICK_EPS,
            rigid_contact_stick_freeze_angular_eps=ANALYTIC_CONTACT_STICK_EPS,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.last_tensions = np.zeros(len(self.joints), dtype=float)
        self.last_middle = 0.0
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.0, -3.8, 1.0), pitch=-4.0, yaw=90.0)
        self.capture()

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                apply_endpoint_loads,
                dim=1,
                inputs=[
                    self.bodies[0],
                    self.bodies[-1],
                    self.left_load,
                    self.right_load,
                    self.left_tangent,
                    self.right_tangent,
                ],
                outputs=[self.state_0.body_f],
                device=self.solver.device,
            )
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.last_tensions = self.state_0.vbd.cable_tension.numpy()[self.joints].astype(float)
        self.last_middle = float(np.mean(self.last_tensions[self.regions["middle"]]))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_scalar("left endpoint load [N]", self.left_load)
        self.viewer.log_scalar("middle measured tension [N]", self.last_middle)
        self.viewer.log_scalar("middle capstan reference [N]", self.middle_reference)
        self.viewer.log_scalar("right endpoint load [N]", self.right_load)
        self.viewer.end_frame()

    def test_final(self):
        tensions = self.state_0.vbd.cable_tension.numpy()[self.joints].astype(float)
        middle = float(np.mean(tensions[self.regions["middle"]]))
        if not np.isfinite(middle):
            raise ValueError("Middle-span tension is non-finite.")
        if middle <= self.left_load:
            raise ValueError("Middle span did not develop a higher tension than the lighter endpoint load.")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
