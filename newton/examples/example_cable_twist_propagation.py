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
# Shows how twist propagates along a cable by manually rotating one end.
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton


@wp.kernel
def kinematic_twist_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_indices: wp.array(dtype=wp.int32),
    base_angle_increment: float,
    sim_time_wp: wp.array(dtype=float),
    end_time: float,
):
    # This kernel is always called by the captured graph.
    # We control its behavior by passing in the current simulation time
    # and conditionally setting the rotation angle to zero inside the kernel.
    angle_increment = base_angle_increment
    if sim_time_wp[0] >= end_time:
        angle_increment = 0.0

    tid = wp.tid()
    body_index = body_indices[tid]

    # Get current transform from the state array
    current_transform = body_q[body_index]
    current_quat = wp.transform_get_rotation(current_transform)

    # The capsule's length is its local Z-axis
    local_twist_axis = wp.vec3(0.0, 0.0, 1.0)

    # Rotate this local axis into world space to get the correct axis for rotation
    world_twist_axis = wp.quat_rotate(current_quat, local_twist_axis)

    # Create a small incremental rotation quaternion
    incremental_rotation = wp.quat_from_axis_angle(world_twist_axis, angle_increment)

    # Apply the incremental rotation to the current orientation
    new_q = wp.mul(incremental_rotation, current_quat)

    # Write the new transform (preserving position, updating rotation) back to the state array
    body_q[body_index] = wp.transform(wp.transform_get_translation(current_transform), new_q)


class Example:
    def create_cable_geometry(self, pos: wp.vec3 | None = None, num_elements=40, total_length=8.0, twisting_angle=0.0):
        if pos is None:
            pos = wp.vec3()

        # num_points = num_elements + 1
        points = []

        # Create a serpentine path with 3 right-angle turns on the XY plane
        num_segments = 4
        elements_per_segment = num_elements // num_segments
        segment_length = total_length / num_segments

        path_directions = [
            wp.vec3(1.0, 0.0, 0.0),  # Leg 1: +X
            wp.vec3(0.0, 1.0, 0.0),  # Leg 2: +Y
            wp.vec3(-1.0, 0.0, 0.0),  # Leg 3: -X
            wp.vec3(0.0, 1.0, 0.0),  # Leg 4: +Y
        ]

        current_pos = pos
        points.append(current_pos)

        for seg_idx in range(num_segments):
            direction = path_directions[seg_idx]
            for _ in range(elements_per_segment):
                current_pos += direction * (segment_length / elements_per_segment)
                points.append(current_pos)

        edge_q = []
        if num_elements > 0:
            local_axis = wp.vec3(0.0, 0.0, 1.0)

            # Initial direction must match the first segment of the path
            from_direction = wp.normalize(points[1] - points[0])

            angle_step = twisting_angle / num_elements if num_elements > 0 else 0.0

            # Use parallel transport for smooth quaternion generation
            # Initialize the first quaternion to align the capsule's Z-axis with the first segment
            edge_q.append(wp.quat_between_vectors(local_axis, from_direction))

            for i in range(1, num_elements):
                p0 = points[i]
                p1 = points[i + 1]
                to_direction = wp.normalize(p1 - p0)

                # Get rotation from previous segment to current one
                dq = wp.quat_between_vectors(from_direction, to_direction)

                # Apply it to the previous quaternion to continue the chain
                base_quaternion = wp.mul(dq, edge_q[i - 1])

                if twisting_angle != 0.0:
                    # Apply an incremental twist around the segment's own axis
                    twist_rot = wp.quat_from_axis_angle(to_direction, angle_step)
                    final_quaternion = wp.mul(twist_rot, base_quaternion)
                else:
                    final_quaternion = base_quaternion

                edge_q.append(final_quaternion)
                from_direction = to_direction

        return points, edge_q

    def __init__(self, stage_path: str | None = "example_cable_twist_propagation.usd", headless: bool = False):
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 20
        self.sim_iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()

        num_elements = 40
        cable_length = 8.0
        cable_radius = 0.05

        # Place cables slightly above the ground
        z_offset = cable_radius * 2.0

        # Create two serpentine cables
        cable1_points, cable1_edge_q = self.create_cable_geometry(
            pos=wp.vec3(-4.0, -3.0, z_offset), num_elements=num_elements, total_length=cable_length, twisting_angle=0.0
        )

        cable2_points, cable2_edge_q = self.create_cable_geometry(
            pos=wp.vec3(-4.0, 3.0, z_offset),
            num_elements=num_elements,
            total_length=cable_length,
            twisting_angle=np.pi,  # Pre-twisted by 180 degrees
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig(density=100.0)
        bend_stiffness = 1.0e2
        twist_stiffness = 1.0e2

        # if bend_stiffness > twist_stiffness and bend_stiffness > 100.0 * twist_stiffness:
        #     twist_stiffness = bend_stiffness / 100.0

        # if twist_stiffness > bend_stiffness and twist_stiffness > 100.0 * bend_stiffness:
        #     bend_stiffness = twist_stiffness / 100.0

        # Add first cable
        rod1_bodies, _ = builder.add_rod_mesh(
            positions=cable1_points,
            quaternions=cable1_edge_q,
            radius=cable_radius,
            cfg=shape_cfg,
            bend_stiffness=bend_stiffness,
            twist_stiffness=twist_stiffness,
        )

        # Add second cable
        rod2_bodies, _ = builder.add_rod_mesh(
            positions=cable2_points,
            quaternions=cable2_edge_q,
            radius=cable_radius,
            cfg=shape_cfg,
            bend_stiffness=bend_stiffness,
            twist_stiffness=twist_stiffness,
        )

        # Make the first capsule of each cable kinematic
        self.first_body_cable1 = rod1_bodies[0]
        builder.body_mass[self.first_body_cable1] = 0.0
        builder.body_inv_mass[self.first_body_cable1] = 0.0

        self.first_body_cable2 = rod2_bodies[0]
        builder.body_mass[self.first_body_cable2] = 0.0
        builder.body_inv_mass[self.first_body_cable2] = 0.0

        self.kinematic_bodies = wp.array([self.first_body_cable1, self.first_body_cable2], dtype=wp.int32)

        # builder.add_joint_d6(child=rod1_bodies[-1], parent=-1, parent_xform=builder.body_q[rod1_bodies[-1]], child_xform=wp.transform_identity(), linear_axes=[], angular_axes=[])
        # builder.add_joint_d6(child=rod2_bodies[-1], parent=-1, parent_xform=builder.body_q[rod2_bodies[-1]], child_xform=wp.transform_identity(), linear_axes=[], angular_axes=[])
        # Note: Not fixing the last capsule - the D6 anchor joints were causing the first cable to disconnect

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.ground = True

        self.solver = newton.solvers.SolverXPBD(self.model, iterations=self.sim_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Create a Warp array to hold the simulation time for the kernel
        self.sim_time_wp = wp.zeros(1, dtype=float)

        if not headless:
            self.renderer = newton.viewer.RendererOpenGL(
                self.model, stage_path, show_body_frames=True, body_frame_axis_length=0.2
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
        # The base angle increment is now constant from the graph's perspective
        angle_increment = self.sim_dt * 4.0
        twist_end_time = 4.0

        for _ in range(self.sim_substeps):
            # The conditional logic is now inside the kernel
            wp.launch(
                kernel=kinematic_twist_kernel,
                dim=self.kinematic_bodies.shape[0],
                inputs=[
                    self.state_0.body_q,
                    self.kinematic_bodies,
                    angle_increment,
                    self.sim_time_wp,  # Pass time array to kernel
                    twist_end_time,  # Pass end_time as a constant
                ],
            )
            self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        # Update the time array on the device before launching the graph
        self.sim_time_wp.assign(np.array([self.sim_time], dtype=np.float32))

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


# main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cable_twist_propagation.usd",
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
