# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Cable/Rod Simulation
#
# Shows how to create cables with helical initial configurations.
# Creates two helix cables side by side - one untwisted, one twisted.
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton

wp.config.enable_backward = False


class Example:
    def create_helix_geometry(
        self,
        pos: wp.vec3 | None = None,
        num_elements: int = 30,
        radius: float = 1.0,
        height: float = 3.0,
        total_turns: float = 2.0,
        twisting_angle: float = 0.0,
    ):
        """
        Creates the rest geometry (points and quaternions) for a helix.
        This uses a robust two-pass parallel transport method to generate orientations.
        """

        if pos is None:
            pos = wp.vec3()

        # --- 1. Generate the points of the helix centerline ---
        points = []
        num_points = num_elements + 1
        total_angle = total_turns * 2.0 * math.pi

        for i in range(num_points):
            t = float(i) / float(num_elements)
            theta = total_angle * t
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            z = height * t
            points.append(pos + wp.vec3(x, y, z))

        if num_elements <= 0:
            return points, []

        # --- 2. First Pass: Generate untwisted quaternions using Parallel Transport ---
        # This creates a smooth "ribbon" that follows the curve's natural torsion without adding any twist.
        edge_q = []
        local_axis = wp.vec3(0.0, 0.0, 1.0)  # Capsule's local axis is Z

        # Initial orientation aligns the local Z axis with the first segment.
        first_segment_dir = wp.normalize(points[1] - points[0])
        q = wp.quat_between_vectors(local_axis, first_segment_dir)
        edge_q.append(q)

        for i in range(1, num_elements):
            p_prev = points[i - 1]
            p_curr = points[i]
            p_next = points[i + 1]

            v_prev = wp.normalize(p_curr - p_prev)
            v_curr = wp.normalize(p_next - p_curr)

            # Rotation from the previous segment to the current one
            delta_q = wp.quat_between_vectors(v_prev, v_curr)

            # Apply this rotation to the previous orientation to get the new one
            q = wp.mul(delta_q, q)
            edge_q.append(q)

        # --- 3. Second Pass: Apply the desired end-to-end twist ---
        if twisting_angle != 0.0:
            angle_step = twisting_angle / float(num_elements)

            for i in range(num_elements):
                # Axis of twist is the capsule's own segment direction
                segment_direction = wp.normalize(points[i + 1] - points[i])

                # Create a cumulative rotation for this segment
                twist_rot = wp.quat_from_axis_angle(segment_direction, angle_step * (i + 1))

                # Apply it to the existing (untwisted) quaternion
                edge_q[i] = wp.mul(twist_rot, edge_q[i])

        return points, edge_q

    def __init__(self, stage_path: str | None = "example_cable_helix.usd", headless: bool = False):
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 50
        self.sim_iterations = 20
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()

        # Helix parameters
        num_elements = 30
        helix_radius = 1.0
        helix_height = 3.0
        helix_turns = 2.0
        cable_radius = 0.01

        # Place helixes above the ground
        z_offset = helix_height * 0.5

        # Create two helix cables side by side
        helix1_points, helix1_edge_q = self.create_helix_geometry(
            pos=wp.vec3(0.0, -2.0, z_offset),
            num_elements=num_elements,
            radius=helix_radius,
            height=helix_height,
            total_turns=helix_turns,
            twisting_angle=0.0,  # No additional twist
        )

        helix2_points, helix2_edge_q = self.create_helix_geometry(
            pos=wp.vec3(0.0, 2.0, z_offset),
            num_elements=num_elements,
            radius=helix_radius,
            height=helix_height,
            total_turns=helix_turns,
            twisting_angle=np.pi * 4.0,  # Two full twists
        )

        # Physical properties
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=500.0,
            ke=1e3,
            kd=1e1,
            restitution=0.0,
            mu=0.5,
        )

        bend_stiffness = 1.0e2
        twist_stiffness = 1.0e2

        # Add first helix cable (untwisted)
        rod1_bodies, rod1_joints = builder.add_rod_mesh(
            positions=helix1_points,
            quaternions=helix1_edge_q,
            radius=cable_radius,
            cfg=shape_cfg,
            bend_stiffness=bend_stiffness,
            twist_stiffness=twist_stiffness,
        )

        # Add second helix cable (twisted)
        rod2_bodies, rod2_joints = builder.add_rod_mesh(
            positions=helix2_points,
            quaternions=helix2_edge_q,
            radius=cable_radius,
            cfg=shape_cfg,
            bend_stiffness=bend_stiffness,
            twist_stiffness=twist_stiffness,
        )

        # # Fix the top (last) capsule of each helix to make it kinematic (hanging from top)
        # self.top_body_helix1 = rod1_bodies[-1]
        # builder.body_mass[self.top_body_helix1] = 0.0
        # builder.body_inv_mass[self.top_body_helix1] = 0.0

        # self.top_body_helix2 = rod2_bodies[-1]
        # builder.body_mass[self.top_body_helix2] = 0.0
        # builder.body_inv_mass[self.top_body_helix2] = 0.0

        # Add ground plane
        builder.add_ground_plane()

        self.model = builder.finalize()

        # Optimized XPBD parameters for cable simulation
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=self.sim_iterations,
            joint_linear_compliance=0.0,  # Make stretch constraint stiff
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if not headless:
            self.renderer = newton.viewer.RendererOpenGL(
                self.model,
                stage_path,
                show_body_frames=True,
                body_frame_axis_length=0.2,
                capsule_radius_scale=2.0,
            )
        elif stage_path:
            self.renderer = newton.viewer.RendererUsd(self.model, stage_path)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.contacts = self.model.collide(self.state_0, rigid_contact_margin=1.0)
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.sim_time)
        self.renderer.render(self.state_0)
        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cable_helix.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Toggles the opening of an interactive window to play back animations in real time. Ignores --num-frames if used.",
    )
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, headless=args.headless)

        if not args.headless:
            example.renderer.paused = True
            while example.renderer.is_running():
                example.step()
                example.render()
        else:
            for _ in range(args.num_frames):
                example.step()
                example.render()

        if example.renderer:
            example.renderer.save()
