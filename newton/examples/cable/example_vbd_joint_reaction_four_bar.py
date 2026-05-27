# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Joint Reaction Four Bar
#
# Demonstrates the VBD-only `state.vbd.joint_reaction_f` readout on a closed
# four-bar linkage. The orange middle link and gray ground both constrain the
# green right link. `state.vbd.joint_reaction_f[joint]` reports those two joint
# loads separately, while `body_parent_f[green link]` is their merged body-level
# load.
#
# Command: uv run -m newton.examples vbd_joint_reaction_four_bar
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

DRIVE_TORQUE = 0.2
LINK_THICKNESS = 0.02
PIN_RADIUS = 0.028
GROUND_LAYER_Z = -0.045
ROCKER_LAYER_Z = 0.0
CRANK_LAYER_Z = 0.045
COUPLER_LAYER_Z = 0.09


def fourbar_geometry(theta: float = 0.55):
    a_link, b_link, c_link, d_link = 0.28, 0.62, 0.48, 0.58
    point_a = np.array([0.0, 0.0, 0.0], dtype=float)
    point_d = np.array([d_link, 0.0, 0.0], dtype=float)
    point_b = point_a + a_link * np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)

    bd = point_d - point_b
    dist_bd = float(np.linalg.norm(bd))
    ex = bd / dist_bd
    ey = np.array([-ex[1], ex[0], 0.0], dtype=float)
    x = (b_link * b_link - c_link * c_link + dist_bd * dist_bd) / (2.0 * dist_bd)
    h = np.sqrt(max(b_link * b_link - x * x, 0.0))
    point_c = point_b + x * ex + h * ey
    return (point_a, point_b, point_c, point_d), (a_link, b_link, c_link)


def link_pose(p0: np.ndarray, p1: np.ndarray) -> wp.transform:
    midpoint = 0.5 * (p0 + p1)
    angle = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
    return wp.transform(wp.vec3(*midpoint), q)


def set_joint_force(control: newton.Control, model: newton.Model, joint: int, value: float) -> None:
    joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_f[int(model.joint_qd_start.numpy()[joint])] = value
    control.joint_f.assign(joint_f)


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
        builder.request_state_attributes("body_parent_f", "vbd:joint_reaction_f")

        marker_cfg = newton.ModelBuilder.ShapeConfig()
        marker_cfg.mark_as_site()
        visual_cfg = newton.ModelBuilder.ShapeConfig()
        visual_cfg.mark_as_site()
        ground_cfg = newton.ModelBuilder.ShapeConfig()
        ground_cfg.density = 0.0
        ground_cfg.collision_group = 0
        ground_cfg.has_shape_collision = False
        ground_cfg.has_particle_collision = False
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = 1000.0
        cfg.collision_group = 0
        cfg.is_visible = False

        (point_a, point_b, point_c, point_d), (a_link, b_link, c_link) = fourbar_geometry()
        ground_pose = link_pose(
            point_a + np.array([0.0, 0.0, GROUND_LAYER_Z]), point_d + np.array([0.0, 0.0, GROUND_LAYER_Z])
        )
        builder.add_shape_box(
            -1,
            xform=ground_pose,
            hx=0.5 * float(np.linalg.norm(point_d - point_a)),
            hy=0.012,
            hz=0.012,
            cfg=ground_cfg,
            color=(0.33, 0.36, 0.38),
        )
        for point, layer_z in ((point_a, CRANK_LAYER_Z), (point_d, ROCKER_LAYER_Z)):
            builder.add_shape_sphere(
                -1,
                xform=wp.transform(wp.vec3(point[0], point[1], layer_z), wp.quat_identity()),
                radius=PIN_RADIUS,
                cfg=marker_cfg,
                color=(0.86, 0.86, 0.82),
            )

        crank = builder.add_link(xform=link_pose(point_a, point_b))
        coupler = builder.add_link(xform=link_pose(point_b, point_c))
        rocker = builder.add_link(xform=link_pose(point_d, point_c))
        builder.add_shape_box(
            crank, hx=0.5 * a_link, hy=LINK_THICKNESS, hz=LINK_THICKNESS, cfg=cfg, color=(0.18, 0.46, 0.72)
        )
        builder.add_shape_box(
            coupler, hx=0.5 * b_link, hy=LINK_THICKNESS, hz=LINK_THICKNESS, cfg=cfg, color=(0.82, 0.55, 0.18)
        )
        builder.add_shape_box(
            rocker, hx=0.5 * c_link, hy=LINK_THICKNESS, hz=LINK_THICKNESS, cfg=cfg, color=(0.32, 0.63, 0.36)
        )
        builder.add_shape_box(
            crank,
            xform=wp.transform(wp.vec3(0.0, 0.0, CRANK_LAYER_Z), wp.quat_identity()),
            hx=0.5 * a_link,
            hy=LINK_THICKNESS,
            hz=0.012,
            cfg=visual_cfg,
            color=(0.18, 0.46, 0.72),
        )
        builder.add_shape_box(
            coupler,
            xform=wp.transform(wp.vec3(0.0, 0.0, COUPLER_LAYER_Z), wp.quat_identity()),
            hx=0.5 * b_link,
            hy=LINK_THICKNESS,
            hz=0.012,
            cfg=visual_cfg,
            color=(0.82, 0.55, 0.18),
        )
        builder.add_shape_box(
            rocker,
            xform=wp.transform(wp.vec3(0.0, 0.0, ROCKER_LAYER_Z), wp.quat_identity()),
            hx=0.5 * c_link,
            hy=LINK_THICKNESS,
            hz=0.012,
            cfg=visual_cfg,
            color=(0.32, 0.63, 0.36),
        )
        builder.add_shape_sphere(
            coupler,
            xform=wp.transform(wp.vec3(-0.5 * b_link, 0.0, COUPLER_LAYER_Z), wp.quat_identity()),
            radius=PIN_RADIUS,
            cfg=marker_cfg,
            color=(0.96, 0.92, 0.72),
        )
        builder.add_shape_sphere(
            rocker,
            xform=wp.transform(wp.vec3(0.5 * c_link, 0.0, ROCKER_LAYER_Z), wp.quat_identity()),
            radius=PIN_RADIUS,
            cfg=marker_cfg,
            color=(0.96, 0.92, 0.72),
        )

        j_crank = builder.add_joint_revolute(
            parent=-1,
            child=crank,
            parent_xform=wp.transform(wp.vec3(*point_a), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * a_link, 0.0, 0.0), wp.quat_identity()),
            axis=wp.vec3(0.0, 0.0, 1.0),
            label="ground_crank",
        )
        j_coupler = builder.add_joint_revolute(
            parent=crank,
            child=coupler,
            parent_xform=wp.transform(wp.vec3(0.5 * a_link, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * b_link, 0.0, 0.0), wp.quat_identity()),
            axis=wp.vec3(0.0, 0.0, 1.0),
            label="crank_coupler",
        )
        j_rocker = builder.add_joint_revolute(
            parent=coupler,
            child=rocker,
            parent_xform=wp.transform(wp.vec3(0.5 * b_link, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.5 * c_link, 0.0, 0.0), wp.quat_identity()),
            axis=wp.vec3(0.0, 0.0, 1.0),
            label="coupler_rocker",
        )
        j_loop = builder.add_joint_revolute(
            parent=-1,
            child=rocker,
            parent_xform=wp.transform(wp.vec3(*point_d), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * c_link, 0.0, 0.0), wp.quat_identity()),
            axis=wp.vec3(0.0, 0.0, 1.0),
            label="loop_ground_rocker",
        )
        builder.add_articulation([j_crank, j_coupler, j_rocker])
        builder.joint_articulation[j_loop] = -1
        builder.color()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=32)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.joints = {
            "ground_crank": j_crank,
            "crank_coupler": j_coupler,
            "coupler_rocker": j_rocker,
            "loop_ground_rocker": j_loop,
        }
        self.rocker = rocker
        self.sum_error = 0.0
        self.loop_force = 0.0
        self.tree_force = 0.0
        self.body_force = 0.0
        self.summed_body_force = 0.0

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.05, -1.55, 0.8), pitch=-34.0, yaw=32.0)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.3, 0.22, 0.0))

    def update_metrics(self):
        joint_reaction = self.state_0.vbd.joint_reaction_f.numpy()
        body_parent_f = self.state_0.body_parent_f.numpy()
        joint_child = self.model.joint_child.numpy()

        summed = np.zeros_like(body_parent_f)
        for joint_index, child in enumerate(joint_child):
            if child >= 0:
                summed[child] += joint_reaction[joint_index]

        self.sum_error = float(np.max(np.linalg.norm(body_parent_f - summed, axis=1)))
        self.tree_force = float(np.linalg.norm(joint_reaction[self.joints["coupler_rocker"], :3]))
        self.loop_force = float(np.linalg.norm(joint_reaction[self.joints["loop_ground_rocker"], :3]))
        self.body_force = float(np.linalg.norm(body_parent_f[self.rocker, :3]))
        self.summed_body_force = float(np.linalg.norm(summed[self.rocker, :3]))

    def simulate(self):
        set_joint_force(self.control, self.model, self.joints["ground_crank"], DRIVE_TORQUE)
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.update_metrics()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("orange-green internal joint reaction [N]", self.tree_force)
        self.viewer.log_scalar("gray-green closure joint reaction [N]", self.loop_force)
        self.viewer.log_scalar("green link net joint load [N]", self.body_force)
        self.viewer.log_scalar("vector sum of green-link joint reactions [N]", self.summed_body_force)
        self.viewer.log_scalar("green body load sum error [N]", self.sum_error)
        self.viewer.end_frame()

    def test_final(self):
        if self.sum_error > 1.0e-4:
            raise ValueError(f"Per-body sum consistency error is too large: {self.sum_error:.6g} N")
        if min(self.loop_force, self.tree_force) <= 1.0e-4:
            raise ValueError("Internal and closure joint reactions should both be nonzero.")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
