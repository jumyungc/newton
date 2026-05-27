# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Joint Reaction Closed Cable
#
# Demonstrates `state.vbd.joint_reaction_f` and `state.vbd.cable_tension` on a
# standalone closed cable loop. The cable is a real segmented rod built with
# `add_rod(..., closed=True)`. Two highlighted cable segments receive equal and
# opposite world-X body forces. The net external force is zero, but the loaded
# closed loop develops measurable per-joint reactions and cable tension.
#
# Command: uv run -m newton.examples vbd_joint_reaction_closed_cable
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

CABLE_COLOR = (0.92, 0.72, 0.16)
PULL_BODY_COLOR = (0.92, 0.22, 0.18)
CLOSURE_JOINT_COLOR = (0.49, 0.23, 0.93)
PULL_FORCE = 2.0
NUM_SEGMENTS = 24
RING_RADIUS = 0.55
RIGHT_PULL_BODY_INDEX = 0
LEFT_PULL_BODY_INDEX = NUM_SEGMENTS // 2


@wp.kernel
def apply_opposing_pull_forces(
    left_body: int,
    right_body: int,
    pull_force: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    body_f[left_body] = wp.spatial_vector(wp.vec3(-pull_force, 0.0, 0.0), wp.vec3(0.0))
    body_f[right_body] = wp.spatial_vector(wp.vec3(pull_force, 0.0, 0.0), wp.vec3(0.0))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
        builder.request_state_attributes("body_parent_f", "vbd:joint_reaction_f", "vbd:cable_tension")

        cable_cfg = newton.ModelBuilder.ShapeConfig()
        cable_cfg.density = 20.0
        cable_cfg.ke = 1.0e4
        cable_cfg.kd = 0.0
        cable_cfg.collision_group = 0
        marker_cfg = newton.ModelBuilder.ShapeConfig()
        marker_cfg.mark_as_site()

        angles = np.linspace(0.0, 2.0 * np.pi, NUM_SEGMENTS + 1)
        points_np = np.column_stack(
            (
                RING_RADIUS * np.cos(angles),
                RING_RADIUS * np.sin(angles),
                np.zeros(NUM_SEGMENTS + 1),
            )
        )
        points = [wp.vec3(*point) for point in points_np]
        quaternions = newton.utils.create_parallel_transport_cable_quaternions(points)
        bodies, joints = builder.add_rod(
            points,
            quaternions,
            radius=0.014,
            cfg=cable_cfg,
            stretch_stiffness=1.0e4,
            stretch_damping=1.0e-2,
            bend_stiffness=0.35,
            bend_damping=1.0e-2,
            closed=True,
            label="closed_cable",
        )
        closure_marker_shape = builder.add_shape_sphere(
            bodies[-1],
            xform=wp.transform(
                wp.vec3(0.0, 0.0, float(np.linalg.norm(points_np[-1] - points_np[-2]))),
                wp.quat_identity(),
            ),
            radius=0.04,
            cfg=marker_cfg,
            color=CLOSURE_JOINT_COLOR,
            label="purple_closure_joint_marker",
        )
        left_body = bodies[LEFT_PULL_BODY_INDEX]
        right_body = bodies[RIGHT_PULL_BODY_INDEX]
        for shape_index, shape_body in enumerate(builder.shape_body):
            if shape_index == closure_marker_shape:
                builder.shape_color[shape_index] = CLOSURE_JOINT_COLOR
            elif shape_body in bodies:
                builder.shape_color[shape_index] = (
                    PULL_BODY_COLOR if shape_body in (left_body, right_body) else CABLE_COLOR
                )
        builder.color()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=48)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.bodies = bodies
        self.joints = joints
        self.left_body = left_body
        self.right_body = right_body
        self.loop_joint = joints[-1]

        self.sum_error = 0.0
        self.max_reaction = 0.0
        self.mean_reaction = 0.0
        self.loop_reaction = 0.0
        self.max_tension = 0.0
        self.mean_tension = 0.0
        self.loaded_body_reaction = 0.0
        self.applied_pull_force = PULL_FORCE

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.0, -2.0, 1.0), pitch=-30.0, yaw=90.0)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.0, 0.0, 0.0))
        self.capture()

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate(update_metrics=False)
            self.graph = capture.graph
        else:
            self.graph = None

    def update_metrics(self):
        joint_reaction = self.state_0.vbd.joint_reaction_f.numpy()
        body_parent_f = self.state_0.body_parent_f.numpy()
        joint_child = self.model.joint_child.numpy()

        summed = np.zeros_like(body_parent_f)
        for joint_index, child in enumerate(joint_child):
            if child >= 0:
                summed[child] += joint_reaction[joint_index]

        reaction_force = np.linalg.norm(joint_reaction[self.joints, :3], axis=1)
        tensions = self.state_0.vbd.cable_tension.numpy()[self.joints].astype(float)

        self.sum_error = float(np.max(np.linalg.norm(body_parent_f - summed, axis=1)))
        self.max_reaction = float(np.max(reaction_force))
        self.mean_reaction = float(np.mean(reaction_force))
        self.loop_reaction = float(np.linalg.norm(joint_reaction[self.loop_joint, :3]))
        self.max_tension = float(np.max(tensions))
        self.mean_tension = float(np.mean(tensions))
        self.loaded_body_reaction = float(
            max(np.linalg.norm(body_parent_f[self.left_body, :3]), np.linalg.norm(body_parent_f[self.right_body, :3]))
        )

    def simulate(self, update_metrics: bool = True):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                apply_opposing_pull_forces,
                dim=1,
                inputs=[self.left_body, self.right_body, PULL_FORCE],
                outputs=[self.state_0.body_f],
                device=self.solver.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        if update_metrics:
            self.update_metrics()

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
            self.update_metrics()
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("applied pull on each red segment [N]", self.applied_pull_force)
        self.viewer.log_scalar("mean per-joint cable reaction [N]", self.mean_reaction)
        self.viewer.log_scalar("mean cable tension [N]", self.mean_tension)
        self.viewer.log_scalar("purple closure-joint reaction [N]", self.loop_reaction)
        self.viewer.log_scalar("max red-segment net joint load [N]", self.loaded_body_reaction)
        self.viewer.log_scalar("body load sum error [N]", self.sum_error)
        self.viewer.end_frame()

    def test_final(self):
        values = np.array(
            [
                self.sum_error,
                self.max_reaction,
                self.mean_reaction,
                self.loop_reaction,
                self.loaded_body_reaction,
                self.max_tension,
                self.mean_tension,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Closed cable reaction metrics contain non-finite values.")
        if self.sum_error > 1.0e-4:
            raise ValueError(f"Per-body sum consistency error is too large: {self.sum_error:.6g} N")
        if min(self.max_reaction, self.loop_reaction, self.loaded_body_reaction, self.max_tension) <= 1.0e-4:
            raise ValueError("Closed cable did not develop meaningful joint reactions and tension.")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
