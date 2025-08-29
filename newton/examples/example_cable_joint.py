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
# Shows how to set up a cable/rod simulation.
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton

wp.config.enable_backward = False


@wp.kernel
def kinematic_twist_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_indices: wp.array(dtype=wp.int32),
    angle_increment: float,
):
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
    def create_cable_geometry(self, pos: wp.vec3 | None = None, num_elements=10, length=10.0, twisting_angle=0.0):
        """
        Create cable geometry with points, edge indices, and quaternions using parallel transport.

        Uses parallel transport to maintain smooth rotational continuity along the cable,
        which is more physically accurate than computing each segment independently.

        Args:
            pos: The starting position of the cable.
            num_elements: Number of cable elements (edges)
            length: Total length of the cable
            twisting_angle: Twist angle in radians (default 0.0)

        Returns:
            tuple: (points, edge_indices, quaternions)
        """
        if pos is None:
            pos = wp.vec3()
        # Create points along straight line in x direction.
        num_points = num_elements + 1
        points = []

        for i in range(num_points):
            t = i / num_elements
            x = length * t
            y = 0.0
            z = 0.0
            points.append(pos + wp.vec3(x, y, z))

        # Create edge indices connecting consecutive points
        edge_indices = []
        for i in range(num_elements):
            vertex_0 = i  # First vertex of edge
            vertex_1 = i + 1  # Second vertex of edge
            edge_indices.extend([vertex_0, vertex_1])

        edge_indices = np.array(edge_indices, dtype=np.int32)

        # Create quaternions for each edge using parallel transport
        edge_q = []
        if num_elements > 0:
            # Capsule internal axis is +Z (from capsule code: "internally capsule axis is always +Z")
            local_axis = wp.vec3(0.0, 0.0, 1.0)

            # Parallel transport: maintain smooth rotational continuity along cable
            from_direction = local_axis  # Start with local Z-axis

            # The total twist will be distributed along the cable
            angle_step = twisting_angle / num_elements if num_elements > 0 else 0.0

            for i in range(num_elements):
                p0 = points[i]
                p1 = points[i + 1]

                # Current segment direction
                to_direction = wp.normalize(p1 - p0)

                # Compute rotation from previous direction to current direction
                # This maintains smooth rotational continuity (parallel transport)
                dq = wp.quat_between_vectors(from_direction, to_direction)

                if i == 0:
                    # First segment: just the directional alignment
                    base_quaternion = dq
                else:
                    # Subsequent segments: multiply with previous quaternion (parallel transport)
                    base_quaternion = wp.mul(dq, edge_q[i - 1])

                # Apply incremental twist around the current segment direction
                if twisting_angle != 0.0:
                    twist_increment = angle_step
                    twist_rot = wp.quat_from_axis_angle(to_direction, twist_increment)
                    final_quaternion = wp.mul(twist_rot, base_quaternion)
                else:
                    final_quaternion = base_quaternion

                edge_q.append(final_quaternion)

                # Update for next iteration (parallel transport)
                from_direction = to_direction

        return points, edge_indices, edge_q

    def __init__(self, stage_path: str | None = "example_cable_joint.usd", headless: bool = False):
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 20
        self.sim_iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.use_cuda_graph = wp.get_device().is_cuda

        # Generate cable geometry for two cables
        num_elements = 5
        cable_length = 5.0
        cable_radius = 0.01

        builder = newton.ModelBuilder()

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=100.0,
            ke=1e3,
            kd=1e2,
            restitution=0.0,
            mu=0.5,
        )

        initial_bend_stiffness = 1.0e-1

        bend_scale = 10

        # Store all rod bodies and joints for compatibility with existing code
        self.rod_bodies = []
        self.rod_joints = []
        kinematic_body_indices = []

        y_separation = 2.0
        start_x = 0.0

        # Create 6 cables in a row along the y-axis
        for i in range(6):
            y_pos = i * y_separation

            # First 3 are untwisted, next 3 are twisted
            if i < 3:
                twist = 0.0
                bend_stiffness = initial_bend_stiffness * bend_scale**i
                twist_stiffness = bend_stiffness

            else:
                twist = np.pi / 2
                bend_stiffness = initial_bend_stiffness * bend_scale ** (i - 3)
                twist_stiffness = bend_stiffness

            cable_points, _, cable_edge_q = self.create_cable_geometry(
                pos=wp.vec3(start_x, y_pos, 2.0),
                num_elements=num_elements,
                length=cable_length,
                twisting_angle=twist,
            )
            rod_bodies, rod_joints = builder.add_rod_mesh(
                positions=cable_points,
                quaternions=cable_edge_q,
                radius=cable_radius,
                cfg=shape_cfg,
                bend_stiffness=bend_stiffness,
                twist_stiffness=twist_stiffness,
            )
            self.rod_bodies.extend(rod_bodies)
            self.rod_joints.extend(rod_joints)

            # Fix the first body to make it kinematic
            first_body = rod_bodies[0]
            builder.body_mass[first_body] = 0.0
            builder.body_inv_mass[first_body] = 0.0
            kinematic_body_indices.append(first_body)

            bend_stiffness *= bend_scale

        # Create a GPU array of the body indices we will be controlling
        self.kinematic_bodies = wp.array(kinematic_body_indices, dtype=wp.int32)

        # b = builder.add_body(xform=wp.transform((0.0, 10.0, 0.0), wp.quat_identity()))
        # builder.add_shape_box(body=b, hx=1.0, hy=1.0, hz=1.0, cfg=newton.ModelBuilder.ShapeConfig(density=100.0))
        # builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.ground = True

        # Optimized XPBD parameters for cable simulation
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=self.sim_iterations,
            joint_linear_compliance=0.001,  # Make joints softer (less bouncy)
            # joint_linear_relaxation=0.9,    # Stable constraint resolution
            # joint_angular_relaxation=0.9,   # Dampen angular corrections
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if not headless:
            self.renderer = newton.viewer.RendererOpenGL(
                self.model,
                stage_path,
                show_body_frames=True,
                body_frame_axis_length=0.5,
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
        # Calculate the small angle to rotate by at each substep
        # angle_increment = self.sim_dt * 4.0

        for _ in range(self.sim_substeps):
            # # Launch our custom kernel to update the capsule orientations on the GPU
            # wp.launch(
            #     kernel=kinematic_twist_kernel,
            #     dim=self.kinematic_bodies.shape[0],
            #     inputs=[
            #         self.state_0.body_q,
            #         self.kinematic_bodies,
            #         angle_increment,
            #     ],
            # )

            self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)

            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        # with wp.ScopedTimer("step"):  # Commented out to reduce console output
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        # with wp.ScopedTimer("render"):  # Commented out to reduce console output
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
        default="example_cable.usd",
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
