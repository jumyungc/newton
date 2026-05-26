# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Tension Hanging
#
# Builds a visible rod cable, requests state.vbd.cable_tension, and compares the
# upper cable-joint tension against the supported cable/load weight plus the
# applied force on the bottom load.
#
# Command: uv run -m newton.examples cable_tension_hanging
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

GRAVITY = 9.81
CABLE_COLOR = (0.92, 0.72, 0.16)


@wp.kernel
def set_body_linear_force(
    body: int,
    force: wp.vec3,
    body_f: wp.array[wp.spatial_vector],
):
    body_f[body] = wp.spatial_vector(force, wp.vec3(0.0))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.applied_downward_force = float(args.applied_force)

        builder = newton.ModelBuilder(gravity=-GRAVITY, up_axis=newton.Axis.Z)
        builder.request_state_attributes("vbd:cable_tension")

        anchor_cfg = newton.ModelBuilder.ShapeConfig()
        anchor_cfg.collision_group = 0
        builder.add_shape_sphere(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            radius=0.055,
            cfg=anchor_cfg,
            color=(0.88, 0.88, 0.88),
        )

        cable_cfg = newton.ModelBuilder.ShapeConfig()
        cable_cfg.density = 1.0
        cable_cfg.ke = 1.0e4
        cable_cfg.kd = 0.0

        num_segments = 8
        cable_length = 1.0
        segment_length = cable_length / num_segments
        points = [wp.vec3(0.0, 0.0, -i * segment_length) for i in range(num_segments + 1)]
        bodies, cable_joints = builder.add_rod(
            points,
            radius=0.015,
            cfg=cable_cfg,
            label="hanging_tension_cable",
        )
        for shape_index, shape_body in enumerate(builder.shape_body):
            if shape_body in bodies:
                builder.shape_color[shape_index] = CABLE_COLOR
        builder.body_flags[bodies[0]] = int(newton.BodyFlags.KINEMATIC)

        load_cfg = newton.ModelBuilder.ShapeConfig()
        load_cfg.density = 1000.0
        builder.add_shape_box(
            bodies[-1],
            xform=wp.transform(wp.vec3(0.0, 0.0, segment_length), wp.quat_identity()),
            hx=0.05,
            hy=0.05,
            hz=0.05,
            cfg=load_cfg,
            color=(0.16, 0.45, 0.72),
        )
        builder.color()

        self.model = builder.finalize()
        self.body = bodies[-1]
        self.cable_joints = cable_joints
        self.supported_bodies = bodies[1:]
        self.solver = newton.solvers.SolverVBD(self.model, iterations=128, rigid_contact_hard=True)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        masses = self.model.body_mass.numpy()
        self.expected_tension = float(np.sum(masses[self.supported_bodies])) * GRAVITY + self.applied_downward_force
        self.last_tension = 0.0

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.0, -3.0, -0.25), pitch=-8.0, yaw=90.0)
        self.capture()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--applied-force",
            type=float,
            default=5.0,
            help="Additional downward force on the hanging mass [N].",
        )
        return parser

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
            if self.applied_downward_force > 0.0:
                wp.launch(
                    set_body_linear_force,
                    dim=1,
                    inputs=[self.body, wp.vec3(0.0, 0.0, -self.applied_downward_force)],
                    outputs=[self.state_0.body_f],
                    device=self.solver.device,
                )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        tensions = self.state_0.vbd.cable_tension.numpy()[self.cable_joints].astype(float)
        self.last_tension = float(tensions[0])

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("cable tension [N]", self.last_tension)
        self.viewer.log_scalar("static load [N]", self.expected_tension)
        self.viewer.log_scalar("top tension error [N]", self.last_tension - self.expected_tension)
        self.viewer.end_frame()

    def test_final(self):
        tensions = self.state_0.vbd.cable_tension.numpy()[self.cable_joints].astype(float)
        tension = float(tensions[0])
        np.testing.assert_allclose(tension, self.expected_tension, rtol=0.0, atol=5.0e-2)


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
