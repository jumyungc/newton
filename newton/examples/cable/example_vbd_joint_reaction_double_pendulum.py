# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Joint Reaction Double Pendulum
#
# Requests State.body_parent_f and checks two-link parent-joint reactions
# against the dynamic force/torque balance used by the VBD reaction report.
#
# Command: uv run -m newton.examples vbd_joint_reaction_double_pendulum
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

GRAVITY = 9.81
BASE_ITERATIONS = 4
RIGID_JOINT_LINEAR_KE = 1.0e4
RIGID_JOINT_ANGULAR_KE = 1.0e5


def quat_y(theta: float) -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta)


def rot_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def transform_rotation_matrix(transform_row: np.ndarray) -> np.ndarray:
    quat = wp.quat(*transform_row[3:7].tolist())
    return np.array(wp.quat_to_matrix(quat), dtype=float).reshape(3, 3)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        theta_0 = 0.45
        theta_1 = -0.35
        length = 1.0
        half = 0.5 * length
        q0 = quat_y(theta_0)
        q1 = quat_y(theta_1)
        r0 = rot_y(theta_0)
        r1 = rot_y(theta_1)
        com0 = r0 @ np.array([0.0, 0.0, -half], dtype=float)
        pivot1 = r0 @ np.array([0.0, 0.0, -length], dtype=float)
        com1 = pivot1 + r1 @ np.array([0.0, 0.0, -half], dtype=float)

        builder = newton.ModelBuilder(gravity=-GRAVITY, up_axis=newton.Axis.Z)
        builder.request_state_attributes("body_parent_f")
        cfg0 = newton.ModelBuilder.ShapeConfig()
        cfg1 = newton.ModelBuilder.ShapeConfig()
        marker_cfg = newton.ModelBuilder.ShapeConfig()
        marker_cfg.collision_group = 0
        builder.add_shape_sphere(
            -1,
            xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
            radius=0.055,
            cfg=marker_cfg,
            color=(0.9, 0.9, 0.9),
        )

        link_0 = builder.add_link(xform=wp.transform(wp.vec3(*com0), q0))
        builder.add_shape_box(link_0, hx=0.04, hy=0.04, hz=half, cfg=cfg0, color=(0.13, 0.47, 0.74))
        link_1 = builder.add_link(xform=wp.transform(wp.vec3(*com1), q1))
        builder.add_shape_box(link_1, hx=0.04, hy=0.04, hz=half, cfg=cfg1, color=(0.78, 0.31, 0.16))

        joint_0 = builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, half), wp.quat_identity()),
            axis=wp.vec3(0.0, 1.0, 0.0),
        )
        joint_1 = builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -half), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, half), wp.quat_identity()),
            axis=wp.vec3(0.0, 1.0, 0.0),
        )
        builder.add_articulation([joint_0, joint_1])
        builder.color()

        self.model = builder.finalize()
        self.links = (link_0, link_1)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=BASE_ITERATIONS,
            rigid_joint_linear_ke=RIGID_JOINT_LINEAR_KE,
            rigid_joint_angular_ke=RIGID_JOINT_ANGULAR_KE,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(3.4, -5.4, 1.45), pitch=-12.0, yaw=32.0)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.0, 0.0, -0.6))

        self.masses = self.model.body_mass.numpy().astype(float)
        self.gravity = self.model.gravity.numpy()[0].astype(float)
        self.arm_local = np.array([0.0, 0.0, half], dtype=float)
        self.last_reaction = np.zeros((self.model.body_count, 6))
        self.expected_force = np.zeros((self.model.body_count, 3), dtype=float)
        self.expected_torque = np.zeros((self.model.body_count, 3), dtype=float)
        self.force_error = 0.0
        self.torque_error = 0.0

    def simulate(self):
        link_indices = list(self.links)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            vel_before = self.state_0.body_qd.numpy()[link_indices, :3].astype(float)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            body_q = self.state_1.body_q.numpy()[link_indices]
            vel_after = self.state_1.body_qd.numpy()[link_indices, :3].astype(float)
            acceleration = (vel_after - vel_before) / self.sim_dt

            reaction = self.state_1.body_parent_f.numpy().astype(float)
            expected_force = np.zeros((self.model.body_count, 3), dtype=float)
            expected_force[self.links[1]] = self.masses[self.links[1]] * (acceleration[1] - self.gravity)
            expected_force[self.links[0]] = (
                self.masses[self.links[0]] * (acceleration[0] - self.gravity) + expected_force[self.links[1]]
            )

            expected_torque = np.zeros((self.model.body_count, 3), dtype=float)
            for row, link in enumerate(self.links):
                arm_world = transform_rotation_matrix(body_q[row]) @ self.arm_local
                expected_torque[link] = np.cross(arm_world, expected_force[link])

            self.last_reaction = reaction
            self.expected_force = expected_force
            self.expected_torque = expected_torque
            self.force_error = float(
                np.max(np.linalg.norm(reaction[link_indices, :3] - expected_force[link_indices], axis=1))
            )
            self.torque_error = float(
                np.max(np.linalg.norm(reaction[link_indices, 3:] - expected_torque[link_indices], axis=1))
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        link_0, link_1 = self.links
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("max reaction force error [N]", self.force_error)
        self.viewer.log_scalar("max reaction torque error [N m]", self.torque_error)
        self.viewer.log_scalar("root reaction force [N]", float(np.linalg.norm(self.last_reaction[link_0, :3])))
        self.viewer.log_scalar("middle reaction force [N]", float(np.linalg.norm(self.last_reaction[link_1, :3])))
        self.viewer.end_frame()

    def test_final(self):
        if self.force_error > 2.5e-1:
            raise ValueError(f"Reaction force error is too large: {self.force_error:.6f} N")
        if self.torque_error > 5.0e-2:
            raise ValueError(f"Reaction torque error is too large: {self.torque_error:.6f} N m")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
