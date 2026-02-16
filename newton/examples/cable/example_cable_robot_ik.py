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
# Example Robot IK
#
# Loads a robot from URDF and sets up IK with GUI slider controls
# for the left and right gripper end effector targets.
#
# Command: python -m newton.examples.cable example_robot_ik
#
###########################################################################

from enum import Enum, IntEnum

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton import Contacts


class CollisionMode(Enum):
    """Collision pipeline modes for the robot IK example.

    MUJOCO: Uses MuJoCo's native contact solver (use_mujoco_contacts=True).
    NEWTON_DEFAULT: Newton's standard collision pipeline with GJK/MPR.
    NEWTON_SDF: Newton collision with SDF (Signed Distance Field) for mesh shapes.
    NEWTON_HYDROELASTIC: Newton collision with hydroelastic contact model.
    """

    MUJOCO = "mujoco"
    NEWTON_DEFAULT = "newton_default"
    NEWTON_SDF = "newton_sdf"
    NEWTON_HYDROELASTIC = "newton_hydroelastic"


class TaskType(IntEnum):
    """State machine states for automated capsule grasping and extraction."""

    IDLE = 0
    APPROACH = 1
    ENGAGE = 2
    GRASP = 3
    HOLD_GRASP = 4
    EXTRACT = 5
    DONE = 6


@wp.kernel
def merge_ik_with_gripper_targets(
    ik_solution: wp.array(dtype=wp.float32),
    gripper_targets: wp.array(dtype=wp.float32),
    gripper_mask: wp.array(dtype=wp.int32),
    dof_count: int,
    output: wp.array(dtype=wp.float32),
):
    """Merge IK solution with gripper targets based on mask.

    For each DOF:
    - If gripper_mask[i] >= 0, use gripper_targets[gripper_mask[i]]
    - Otherwise, use ik_solution[i]
    """
    i = wp.tid()
    if i >= dof_count:
        return

    mask_val = gripper_mask[i]
    if mask_val >= 0:
        output[i] = gripper_targets[mask_val]
    else:
        output[i] = ik_solution[i]


NUM_ARMS = 2


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_schedule: wp.array(dtype=wp.int32),
    task_time_soft_limits: wp.array(dtype=float),
    task_idx: wp.array(dtype=int),
    task_time_elapsed: wp.array(dtype=float),
    task_dt: float,
    approach_offsets: wp.array(dtype=wp.vec3),
    capsule_grasp_offset_from_com: wp.array(dtype=wp.vec3),
    extract_distance: float,
    capsule_body_indices: wp.array(dtype=int),
    capsule_extract_dirs: wp.array(dtype=wp.quat),
    grasp_orientation_offset: wp.array(dtype=wp.vec4),
    gripper_open_values: wp.array(dtype=wp.float32),
    task_ee_init_body_q: wp.array(dtype=wp.transform),
    task_capsule_body_q_prev: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    ee_body_indices: wp.array(dtype=int),
    # outputs
    ee_pos_target: wp.array(dtype=wp.vec3),
    ee_pos_target_interpolated: wp.array(dtype=wp.vec3),
    ee_rot_target: wp.array(dtype=wp.vec4),
    ee_rot_target_interpolated: wp.array(dtype=wp.vec4),
    gripper_target: wp.array2d(dtype=wp.float32),
):
    """Compute EE target pose and gripper targets for one arm per thread."""
    arm_idx = wp.tid()

    idx = task_idx[arm_idx]
    task = task_schedule[idx]
    time_limit = task_time_soft_limits[idx]

    task_time_elapsed[arm_idx] += task_dt

    # Interpolation parameter t in [0, 1]
    t = wp.min(1.0, task_time_elapsed[arm_idx] / time_limit)

    # EE position and rotation at the start of this task
    ee_pos_prev = wp.transform_get_translation(task_ee_init_body_q[arm_idx])
    ee_quat_prev = wp.transform_get_rotation(task_ee_init_body_q[arm_idx])

    # Capsule orientation at the start of this task
    capsule_quat_prev = wp.transform_get_rotation(task_capsule_body_q_prev[arm_idx])

    # Current capsule world position
    capsule_pos = wp.transform_get_translation(body_q[capsule_body_indices[arm_idx]])
    capsule_quat = wp.transform_get_rotation(body_q[capsule_body_indices[arm_idx]])

    # Grasp orientation for this arm (stored as vec4 xyzw)
    gv = grasp_orientation_offset[arm_idx]
    grasp_quat_offset = wp.quaternion(gv[:3], gv[3])

    # Default: hold previous pose
    ee_quat_target = ee_quat_prev
    t_gripper = 0.0

    if task == TaskType.APPROACH.value:
        grasp_pos_offset = wp.quat_rotate(capsule_quat, capsule_grasp_offset_from_com[arm_idx])
        ee_pos_target[arm_idx] = capsule_pos + grasp_pos_offset + approach_offsets[arm_idx]
        ee_quat_target = capsule_quat * grasp_quat_offset
    elif task == TaskType.ENGAGE.value:
        grasp_pos_offset = wp.quat_rotate(capsule_quat_prev, capsule_grasp_offset_from_com[arm_idx])
        ee_pos_target[arm_idx] = capsule_pos + grasp_pos_offset
        ee_quat_target = ee_quat_prev
    elif task == TaskType.GRASP.value:
        ee_pos_target[arm_idx] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = t
    elif task == TaskType.HOLD_GRASP.value:
        ee_pos_target[arm_idx] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.EXTRACT.value:
        extract_vec = wp.quat_rotate(capsule_quat_prev, wp.vec3(0.0, 0.0, 1.0)) * 2.0 * extract_distance
        ee_pos_target[arm_idx] = ee_pos_prev + extract_vec
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.DONE.value:
        ee_pos_target[arm_idx] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    else:
        ee_pos_target[arm_idx] = ee_pos_prev
        t_gripper = 1.0

    # Interpolate position (lerp) and rotation (slerp)
    ee_pos_target_interpolated[arm_idx] = ee_pos_prev * (1.0 - t) + ee_pos_target[arm_idx] * t
    ee_quat_interpolated = wp.quat_slerp(ee_quat_prev, ee_quat_target, t)

    ee_rot_target[arm_idx] = ee_quat_target[:4]
    ee_rot_target_interpolated[arm_idx] = ee_quat_interpolated[:4]

    # Gripper targets: open_val * (1 - t_gripper)
    base = arm_idx * 2
    # Over-scale the gripper targets to make the gripper open and close faster/stronger
    over_scale = 4.0
    gripper_target[arm_idx, 0] = over_scale * gripper_open_values[base] * (1.0 - t_gripper)
    gripper_target[arm_idx, 1] = over_scale * gripper_open_values[base + 1] * (1.0 - t_gripper)


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_schedule: wp.array(dtype=wp.int32),
    task_time_soft_limits: wp.array(dtype=float),
    ee_pos_target: wp.array(dtype=wp.vec3),
    ee_rot_target: wp.array(dtype=wp.vec4),
    body_q: wp.array(dtype=wp.transform),
    ee_body_indices: wp.array(dtype=int),
    capsule_body_indices: wp.array(dtype=int),
    pos_error_thresholds: wp.array(dtype=float),
    rot_error_thresholds: wp.array(dtype=float),
    # outputs
    task_idx: wp.array(dtype=int),
    task_time_elapsed: wp.array(dtype=float),
    task_ee_init_body_q: wp.array(dtype=wp.transform),
    task_capsule_body_q_prev: wp.array(dtype=wp.transform),
):
    """Check convergence and advance to the next task when ready."""
    arm_idx = wp.tid()

    idx = task_idx[arm_idx]
    time_limit = task_time_soft_limits[idx]

    # Current EE world pose
    ee_body_id = ee_body_indices[arm_idx]
    ee_pos_current = wp.transform_get_translation(body_q[ee_body_id])
    ee_rot_current = wp.transform_get_rotation(body_q[ee_body_id])

    pos_err = wp.length(ee_pos_target[arm_idx] - ee_pos_current)

    # Rotation error: angular distance between current and target quaternions
    rv = ee_rot_target[arm_idx]
    target_quat = wp.quaternion(rv[:3], rv[3])

    quat_rel = ee_rot_current * wp.quat_inverse(target_quat)
    rot_err = wp.abs(2.0 * wp.atan2(wp.length(quat_rel[:3]), quat_rel[3]))

    # Advance when time limit is exceeded, task index is not the last one,
    # and both position and rotation errors are below their thresholds.
    if (
        task_time_elapsed[arm_idx] >= time_limit
        and task_idx[arm_idx] < wp.len(task_time_soft_limits) - 1
        and pos_err < pos_error_thresholds[idx]
        and rot_err < rot_error_thresholds[idx]
    ):
        task_idx[arm_idx] += 1
        task_time_elapsed[arm_idx] = 0.0
        # Snapshot current EE transform as entry point for the next task
        task_ee_init_body_q[arm_idx] = body_q[ee_body_id]
        task_capsule_body_q_prev[arm_idx] = body_q[capsule_body_indices[arm_idx]]


import os
from pathlib import Path


def _default_assets_path() -> str:
    """Return the default examples asset path.

    Avoid hardcoded absolute paths so examples run on any machine/CI checkout.
    """
    # `newton.examples` is a package; its folder contains the `assets/` directory.
    import newton.examples  # noqa: PLC0415

    return str(Path(newton.examples.__file__).resolve().parent / "assets")


_assets_root = Path(os.environ.get("NEWTON_EXAMPLES_ASSETS_PATH", _default_assets_path())).resolve()
# Many legacy examples use string concatenation: `ASSETS_PATH + "foo.ext"`. Ensure trailing separator.
ASSETS_PATH = os.path.join(os.fspath(_assets_root), "")
ROBOT_PATH = os.fspath(_assets_root / "rby1df" / "urdf")


class Example:
    def __init__(self, viewer, num_worlds=1, args=None):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.collide_substeps = 2  # run collision detection every X simulation substeps

        self.num_worlds = num_worlds
        self.viewer = viewer
        self.collision_mode = CollisionMode.NEWTON_SDF

        self.viewer._paused = True
        self.show_isosurface = False
        self.rigid_contact_max = 100000

        # IK settings
        self.ik_iters = 24

        # Configure shape settings based on collision mode
        self.default_shape_cfg = self._create_shape_config(self.collision_mode)

        self.default_ground_cfg = self.default_shape_cfg.copy()
        self.default_ground_cfg.ke = 5e5
        self.default_ground_cfg.kd = 5e2
        self.default_ground_cfg.mu = 0.3

        self.table_half_size = [0.25, 0.5, 0.02]
        self.table_pos = [0.5, 0, 0.75]

        self.gripper_joint_dofs = [14, 15, 24, 25]
        robot = self.setup_robot_builder()
        scene = self.setup_scene_builder(robot)

        self.single_robot_model = robot.finalize()
        self.model = scene.finalize()

        # Create collision pipeline based on collision mode
        self.collision_pipeline = self._create_collision_pipeline(self.collision_mode, args)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        # Configure solver based on collision mode
        self._use_mujoco_contacts = self.collision_mode == CollisionMode.MUJOCO
        num_per_world = self.rigid_contact_max // self.num_worlds
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=num_per_world,
            nconmax=num_per_world,
            ls_parallel=True,
            iterations=50,
            ls_iterations=25,
            use_mujoco_contacts=self._use_mujoco_contacts,
            impratio=1000.0,
        )

        print(f"Collision mode: {self.collision_mode.value}")
        print(f"  use_mujoco_contacts: {self._use_mujoco_contacts}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        if self._use_mujoco_contacts:
            self.contacts = Contacts(0, 0)
        else:
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.setup_end_effectors()
        self.setup_ik()
        self.setup_gripper_targets()
        self.setup_state_machine()

        # Store joint target positions for merging
        self.joint_target_pos = wp.zeros_like(self.control.joint_target_pos)
        wp.copy(self.joint_target_pos, self.control.joint_target_pos)

        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(wp.vec3(6.5, 0, 1.6), pitch=-5.0, yaw=-180.0)
            self.viewer.camera.fov = 15.0
            # self.viewer.picking_enabled = False  # Disable interactive GUI picking for this example

        self.capture()

    def capture(self):
        self.capture_sim()
        self.capture_ik()

    def capture_sim(self):
        self.graph_sim = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph_sim = capture.graph

    def capture_ik(self):
        self.graph_ik = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
            self.graph_ik = capture.graph

    def simulate(self):
        for i in range(self.sim_substeps):
            if not self._use_mujoco_contacts and (i % self.collide_substeps == 0):
                self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

            self.state_0.clear_forces()

            ## apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def setup_ik(self):
        """Set up IK solver with position and rotation objectives for each end effector."""

        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        # Get current body transforms
        body_q_np = self.state_0.body_q.numpy()

        # Create target transforms and objectives for each end effector
        self.ee_tfs = []
        self.pos_objs = []
        self.rot_objs = []

        for _name, link_idx in self.ee_configs:
            # Create persistent target transform (updated via GUI sliders)
            tf = wp.transform(*body_q_np[link_idx])
            self.ee_tfs.append(tf)

            # Position objective
            self.pos_objs.append(
                ik.IKObjectivePosition(
                    link_index=link_idx,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([wp.transform_get_translation(tf)], dtype=wp.vec3),
                )
            )

            # Rotation objective
            self.rot_objs.append(
                ik.IKObjectiveRotation(
                    link_index=link_idx,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([_q2v4(wp.transform_get_rotation(tf))], dtype=wp.vec4),
                )
            )

        # Joint limit objective to keep joints within bounds
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        # Joint state for IK solver
        self.ik_joint_q = wp.array(
            self.single_robot_model.joint_q, shape=(1, self.single_robot_model.joint_coord_count)
        )

        # Create IK solver with all objectives
        objectives = [*self.pos_objs, *self.rot_objs, self.obj_joint_limits]
        self.ik_solver = ik.IKSolver(
            model=self.single_robot_model,
            n_problems=1,
            objectives=objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        print(f"IK solver initialized with {len(self.ee_configs)} end effector(s)")

    def setup_end_effectors(self):
        # Print body information for debugging
        print("\n=== Body Information ===")
        for i, key in enumerate(self.model.body_key):
            print(f"  Body {i}: {key}")
        print("========================\n")

        # ----------------------------------------------------------------
        # End effector configuration
        # ----------------------------------------------------------------
        ee_body_keys = [
            "right_gripper_end_effector",
            "left_gripper_end_effector",
            "torso_hip_yaw",  # This target helps to keep the robot upright.
        ]

        self.ee_configs = []
        for key in ee_body_keys:
            try:
                idx = self.model.body_key.index(key)
                self.ee_configs.append((key, idx))
                print(f"End effector: {key} (body index {idx})")
            except ValueError:
                print(f"WARNING: End effector key not found: {key}")
                print(f"  Available keys: {self.model.body_key}")

    def setup_gripper_targets(self):
        # ----------------------------------------------------------------
        # Gripper finger joint configuration
        # ----------------------------------------------------------------
        self.gripper_joint_names = [
            "right_gripper_left_finger_joint",
            "right_gripper_right_finger_joint",
            "left_gripper_left_finger_joint",
            "left_gripper_right_finger_joint",
        ]

        # # Build mapping from joint key to DOF index
        self.gripper_limits_lower = self.model.joint_limit_lower.numpy()[self.gripper_joint_dofs]
        self.gripper_limits_upper = self.model.joint_limit_upper.numpy()[self.gripper_joint_dofs]

        # Initialize gripper target positions
        self.gripper_targets_list = [-0.04, 0.04, -0.04, 0.04]

        # Create GPU arrays for gripper targets
        self.gripper_targets = wp.array(self.gripper_targets_list, dtype=wp.float32)

        # Create mask array for merging IK solution with gripper targets
        # mask[i] = gripper_index if DOF i is a gripper joint, else -1
        gripper_mask_np = [-1] * self.single_robot_model.joint_dof_count
        for gripper_idx, dof_idx in enumerate(self.gripper_joint_dofs):
            gripper_mask_np[dof_idx] = gripper_idx
        self.gripper_mask = wp.array(gripper_mask_np, dtype=wp.int32)

    def setup_state_machine(self):
        """Initialize the state machine for automated capsule grasping.

        Creates GPU arrays that mirror the pattern used by
        ``example_ik_cube_stacking.py`` so that target-pose computation and
        task advancement run entirely inside Warp kernels.
        """
        self.auto_mode = False
        self.num_arms = NUM_ARMS

        # Task schedule: both arms follow the same state sequence.
        task_schedule_list = [
            TaskType.APPROACH,
            TaskType.ENGAGE,
            TaskType.GRASP,
            TaskType.HOLD_GRASP,
            TaskType.EXTRACT,
            TaskType.DONE,
        ]
        self.num_tasks = len(task_schedule_list)
        self.sm_task_schedule = wp.array(task_schedule_list, dtype=wp.int32)

        # Time limits per task entry (DONE gets a large sentinel value).
        task_time_limits_list = [1.0, 1.0, 0.5, 0.5, 2.0, 999.0]
        self.sm_task_time_limits = wp.array(task_time_limits_list, dtype=float)

        # Per-arm mutable state
        self.sm_task_idx = wp.zeros(self.num_arms, dtype=int)
        self.sm_task_time_elapsed = wp.zeros(self.num_arms, dtype=float)

        # Snapshot of each arm's EE transform at the start of the current task.
        body_q_np = self.state_0.body_q.numpy()
        init_tfs = []
        for arm_idx in range(self.num_arms):
            _, ee_link_idx = self.ee_configs[arm_idx]
            init_tfs.append(wp.transform(*body_q_np[ee_link_idx]))
        self.sm_task_init_body_q = wp.array(init_tfs, dtype=wp.transform)

        # State machine parameters
        # Especfied in the frame of the capsule
        self.capsule_grasp_offset_from_com = wp.array(
            [wp.vec3(0.0, 0.01, 0.01), wp.vec3(0.0, -0.01, 0.01)], dtype=wp.vec3
        )

        self.approach_offsets = wp.array(
            [
                wp.vec3(0.0, -0.1, 0.0),  # Right arm -> Capsule A
                wp.vec3(0.0, 0.1, 0.0),  # Left arm  -> Capsule B
            ],
            dtype=wp.vec3,
        )

        self.extract_distance = 0.05
        self.pos_error_threshold = wp.array([0.0025] * self.num_tasks, dtype=float)
        rot_error_threshold = [0.20 * wp.pi / 180.0] * self.num_tasks  # in radians
        self.rot_error_threshold = wp.array(rot_error_threshold, dtype=float)

        # Capsule body indices (right arm -> capsule A, left arm -> capsule B)
        capsule_a_idx = self.model.body_key.index("test_capsule_a")
        capsule_b_idx = self.model.body_key.index("test_capsule_b")
        self.sm_capsule_body_indices = wp.array([capsule_a_idx, capsule_b_idx], dtype=int)
        print(f"Capsule A  position: {body_q_np[capsule_a_idx, :3]}, quaternion: {body_q_np[capsule_a_idx, 3:]}")
        print(f"Capsule B  position: {body_q_np[capsule_b_idx, :3]}, quaternion: {body_q_np[capsule_b_idx, 3:]}")

        # Extraction directions (precomputed in setup_scene_builder)
        self.sm_capsule_extract_dirs = wp.array(self.capsule_extract_dirs, dtype=wp.quat)

        # Snapshot of the capsule transform at the start of the current task
        capsule_tf_a = wp.transform(*body_q_np[capsule_a_idx])
        capsule_tf_b = wp.transform(*body_q_np[capsule_b_idx])
        self.sm_task_capsule_body_q_prev = wp.array([capsule_tf_a, capsule_tf_b], dtype=wp.transform)

        # Grasp orientations stored as vec4 (xyzw) for the kernel
        # right_arm_ee_idx = self.ee_configs[0][1]
        # left_arm_ee_idx = self.ee_configs[1][1]

        # quat_a = wp.quat(body_q_np[right_arm_ee_idx, 3:])
        quat_a_offset = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi)
        quat_a_offset = quat_a_offset * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.half_pi)
        quat_a_offset = quat_a_offset * wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -45.0 * wp.half_pi / 180.0)

        # quat_b = wp.quat(body_q_np[left_arm_ee_idx, 3:])
        quat_b_offset = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi)
        quat_b_offset = quat_b_offset * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.half_pi)
        quat_b_offset = quat_b_offset * wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -45.0 * wp.half_pi / 180.0)

        self.sm_grasp_orientation_offset = wp.array([quat_a_offset, quat_b_offset], dtype=wp.vec4)

        # Gripper open values (flat array of 4 floats, used by the kernel)
        self.sm_gripper_open_values = wp.array(self.gripper_targets_list, dtype=wp.float32)

        # EE body indices (for reading current EE position from body_q)
        self.sm_ee_body_indices = wp.array(
            [self.ee_configs[0][1], self.ee_configs[1][1]],
            dtype=int,
        )

        # Kernel output arrays
        self.sm_ee_pos_target = wp.zeros(self.num_arms, dtype=wp.vec3)
        self.sm_ee_pos_interp = wp.zeros(self.num_arms, dtype=wp.vec3)
        self.sm_ee_rot_target = wp.zeros(self.num_arms, dtype=wp.vec4)
        self.sm_ee_rot_interp = wp.zeros(self.num_arms, dtype=wp.vec4)
        self.sm_gripper_target = wp.zeros(shape=(self.num_arms, 2), dtype=wp.float32)

        print("State machine initialized (kernel-based)")
        print(f"  Capsule A body index: {capsule_a_idx}")
        print(f"  Capsule B body index: {capsule_b_idx}")

    def set_joint_targets(self):
        """Update IK targets, run IK solver, and set joint target positions.

        When ``auto_mode`` is active the two Warp kernels drive the arm EE
        targets and gripper values.  Otherwise the GUI-driven ``ee_tfs`` list
        is used (manual control).
        """
        if self.auto_mode:
            # --- kernel path: compute targets on GPU ---
            wp.launch(
                set_target_pose_kernel,
                dim=self.num_arms,
                inputs=[
                    self.sm_task_schedule,
                    self.sm_task_time_limits,
                    self.sm_task_idx,
                    self.sm_task_time_elapsed,
                    self.frame_dt,
                    self.approach_offsets,
                    self.capsule_grasp_offset_from_com,
                    self.extract_distance,
                    self.sm_capsule_body_indices,
                    self.sm_capsule_extract_dirs,
                    self.sm_grasp_orientation_offset,
                    self.sm_gripper_open_values,
                    self.sm_task_init_body_q,
                    self.sm_task_capsule_body_q_prev,
                    self.state_0.body_q,
                    self.sm_ee_body_indices,
                ],
                outputs=[
                    self.sm_ee_pos_target,
                    self.sm_ee_pos_interp,
                    self.sm_ee_rot_target,
                    self.sm_ee_rot_interp,
                    self.sm_gripper_target,
                ],
            )

            # Push kernel outputs into IK objectives (arms only, no CPU sync)
            for arm_idx in range(self.num_arms):
                self.pos_objs[arm_idx].set_target_positions(self.sm_ee_pos_interp[arm_idx : arm_idx + 1])
                self.rot_objs[arm_idx].set_target_rotations(self.sm_ee_rot_interp[arm_idx : arm_idx + 1])

            # Torso objective (index 2) is still driven by the GUI
            tf = self.ee_tfs[2]
            self.pos_objs[2].set_target_position(0, wp.transform_get_translation(tf))
            q = wp.transform_get_rotation(tf)
            self.rot_objs[2].set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

            # Copy kernel gripper output to the main gripper_targets array
            wp.copy(self.gripper_targets, self.sm_gripper_target.flatten())

        else:
            # --- manual GUI path ---
            for i, tf in enumerate(self.ee_tfs):
                self.pos_objs[i].set_target_position(0, wp.transform_get_translation(tf))
                q = wp.transform_get_rotation(tf)
                self.rot_objs[i].set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

        # Step the IK solver
        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)

        # Merge IK solution with gripper targets using GPU kernel
        wp.launch(
            merge_ik_with_gripper_targets,
            dim=self.single_robot_model.joint_dof_count,
            inputs=[
                self.ik_joint_q.flatten(),
                self.gripper_targets,
                self.gripper_mask,
                self.single_robot_model.joint_dof_count,
            ],
            outputs=[self.joint_target_pos],
        )

        # Copy merged joint targets to control
        wp.copy(self.control.joint_target_pos, self.joint_target_pos)

        # Advance the task state machine (after IK, before next sim step)
        if self.auto_mode:
            wp.launch(
                advance_task_kernel,
                dim=self.num_arms,
                inputs=[
                    self.sm_task_schedule,
                    self.sm_task_time_limits,
                    self.sm_ee_pos_target,
                    self.sm_ee_rot_target,
                    self.state_0.body_q,
                    self.sm_ee_body_indices,
                    self.sm_capsule_body_indices,
                    self.pos_error_threshold,
                    self.rot_error_threshold,
                ],
                outputs=[
                    self.sm_task_idx,
                    self.sm_task_time_elapsed,
                    self.sm_task_init_body_q,
                    self.sm_task_capsule_body_q_prev,
                ],
            )

    def step(self):
        # Update IK targets and compute joint positions
        self.set_joint_targets()

        # Run physics simulation
        if self.graph_sim:
            wp.capture_launch(self.graph_sim)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self.collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            self.viewer.log_hydro_contact_surface(
                self.collision_pipeline.get_hydro_contact_surface(), penetrating_only=True
            )
        self.viewer.end_frame()

    def _create_shape_config(self, collision_mode: CollisionMode) -> newton.ModelBuilder.ShapeConfig:
        """Create shape configuration based on collision mode.

        Args:
            collision_mode: The collision pipeline mode to configure for.

        Returns:
            ShapeConfig configured for the specified collision mode.
        """
        # Base configuration common to all modes
        base_cfg = newton.ModelBuilder.ShapeConfig(
            thickness=0.0,
            contact_margin=0.01,
            ke=5.0e4,
            kd=5.0e2,
            mu=2.0,
            # torsional_friction=0.0,
            # rolling_friction=0.0,
        )

        if collision_mode == CollisionMode.NEWTON_SDF:
            # Enable SDF for mesh collision
            shape_cfg = base_cfg.copy()
            shape_cfg.sdf_max_resolution = 64
            shape_cfg.sdf_narrow_band_range = (-0.01, 0.01)
            shape_cfg.is_hydroelastic = False
            return shape_cfg
        elif collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            # Enable hydroelastic contact model
            shape_cfg = base_cfg.copy()
            shape_cfg.sdf_max_resolution = 512
            shape_cfg.sdf_narrow_band_range = (-0.01, 0.01)
            shape_cfg.is_hydroelastic = True
            shape_cfg.k_hydro = 1e11
            return shape_cfg
        else:
            # MUJOCO or NEWTON_DEFAULT: no SDF, no hydroelastic
            shape_cfg = base_cfg.copy()
            shape_cfg.is_hydroelastic = False
            return shape_cfg

    def _create_collision_pipeline(self, collision_mode: CollisionMode, args=None):
        """Create collision pipeline based on collision mode.

        Args:
            collision_mode: The collision pipeline mode to use.
            args: Optional command-line arguments.

        Returns:
            CollisionPipelineUnified instance or None for MuJoCo native contacts.
        """
        if collision_mode == CollisionMode.MUJOCO:
            # MuJoCo uses its own contact solver, but we still need Newton's collision pipeline
            # for collision detection
            return None
        elif collision_mode == CollisionMode.NEWTON_DEFAULT:
            # Standard Newton collision pipeline
            return newton.examples.create_collision_pipeline(self.model, args)
        elif collision_mode == CollisionMode.NEWTON_SDF:
            # Newton with SDF for mesh collision
            return newton.CollisionPipelineUnified.from_model(
                self.model,
                reduce_contacts=True,
                broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
            )
        elif collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            # Newton with hydroelastic contacts
            from newton._src.geometry.sdf_hydroelastic import SDFHydroelasticConfig  # noqa: PLC0415

            return newton.CollisionPipelineUnified.from_model(
                self.model,
                reduce_contacts=True,
                broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
                sdf_hydroelastic_config=SDFHydroelasticConfig(output_contact_surface=True),
            )
        else:
            raise ValueError(f"Unknown collision mode: {collision_mode}")

    def setup_robot_builder(self):
        robot = newton.ModelBuilder()
        robot.default_shape_cfg = self.default_shape_cfg

        robot_file = str(Path(ROBOT_PATH) / "robot_edited.urdf")
        if not os.path.isfile(robot_file):
            raise FileNotFoundError(
                f"Robot URDF not found: {robot_file}. "
                f"Set NEWTON_EXAMPLES_ASSETS_PATH to override (currently ASSETS_PATH={ASSETS_PATH})."
            )

        robot.add_urdf(
            robot_file,
            xform=wp.transform(wp.vec3(0, 0, 0.00)),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

        # Setting some gains to make the stay upright.
        # TODO: Set better control gains. Ask the manufacturer for better gains.
        robot.joint_target_ke[: robot.joint_dof_count] = [45000.0] * robot.joint_dof_count
        robot.joint_target_kd[: robot.joint_dof_count] = [4500.0] * robot.joint_dof_count

        # These values are based/inspired from the specifications of the Franka Panda robot.
        robot.joint_effort_limit[: robot.joint_dof_count] = [100.0] * robot.joint_dof_count
        robot.joint_armature[: robot.joint_dof_count] = [0.2] * robot.joint_dof_count

        # Set smaller gains for gripper joints to allow for more precise control.
        # These values are based on the specifications of the Franka Panda Gripper
        for dof in self.gripper_joint_dofs:
            robot.joint_target_ke[dof] = 100.0
            robot.joint_target_kd[dof] = 10.0
            robot.joint_effort_limit[dof] = 150.0
            robot.joint_armature[dof] = 0.5

        # Set some initial joint positions so that the end effectors are on top of the table.
        # TODO: Set better initial joint positions. Ask the manufacturer for better initial positions.
        # right_arm_idxs = range(6, 6 + 7)
        # left_arm_idxs = range(16, 16 + 7)

        robot.joint_q = [
            4.8646115e-02,
            -1.1358134e-01,
            2.8509942e-01,
            3.0236751e-01,
            -4.3634601e-02,
            9.6731670e-03,
            -8.5306484e-01,
            -1.0891527e00,
            6.6765565e-01,
            -2.0121396e00,
            -1.0203781e00,
            1.5501461e00,
            5.6562239e-01,
            1.9687047e-07,
            -3.9999921e-02,
            4.0000085e-02,
            -7.0531148e-01,
            1.0506693e00,
            -4.4851208e-01,
            -1.9159117e00,
            1.0035634e00,
            1.5637023e00,
            -8.4481186e-01,
            -3.3100471e-07,
            -4.0000360e-02,
            3.9999638e-02,
            -5.4279553e-06,
            1.4788106e-04,
        ]

        finger_shape_indices = {
            robot.body_key.index("right_gripper_leftfinger"),
            robot.body_key.index("right_gripper_rightfinger"),
            robot.body_key.index("left_gripper_leftfinger"),
            robot.body_key.index("left_gripper_rightfinger"),
        }
        non_finger_shape_indices = []
        for shape_idx, body_idx in enumerate(robot.shape_body):
            if body_idx not in finger_shape_indices:
                robot.shape_flags[shape_idx] &= ~newton.ShapeFlags.HYDROELASTIC
                non_finger_shape_indices.append(shape_idx)

        # Fingers
        # robot.approximate_meshes(method="convex_hull", shape_indices=non_finger_shape_indices, keep_visual_shapes=True)

        return robot

    def setup_scene_builder(self, robot):
        scene = newton.ModelBuilder()
        scene.default_shape_cfg = self.default_shape_cfg
        scene.add_builder(robot)

        table_top_center = [self.table_pos[0], self.table_pos[1], self.table_pos[2] + self.table_half_size[2]]

        right_hose_y = -0.15
        left_hose_y = -right_hose_y

        # Add capsule for testing.
        capsule_radius = 0.0025  # 0.01
        capsule_height = 4.0 / 60.0
        tilt_angle_rad = 30.0 * wp.pi / 180.0

        capsule_length_offset = 0.01
        capsule_total_length = capsule_height + 2.0 * capsule_radius + capsule_length_offset

        # Compute pose for capsule A
        x_pos = self.table_pos[0] - 0.5 * capsule_total_length * wp.sin(tilt_angle_rad)
        # Modify x_pos to avoid initial penetration.
        x_pos += 0.005
        y_pos = right_hose_y
        z_pos = table_top_center[2] + 0.5 * capsule_total_length * wp.cos(tilt_angle_rad)

        quat_a = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -tilt_angle_rad)
        xform = wp.transform(wp.vec3(x_pos, y_pos, z_pos), quat_a)

        # Add capsule A
        body_id = scene.add_body(xform=xform, key="test_capsule_a")
        scene.add_shape_capsule(
            body=body_id, radius=capsule_radius, half_height=0.5 * capsule_height, cfg=self.default_shape_cfg
        )

        # Compute pose for capsule B
        x_pos = self.table_pos[0] - 0.5 * capsule_total_length * wp.sin(tilt_angle_rad)
        # Modify x_pos to avoid initial penetration.
        x_pos += 0.005
        y_pos = left_hose_y
        z_pos = table_top_center[2] + 0.5 * capsule_total_length * wp.cos(tilt_angle_rad)

        quat_b = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -tilt_angle_rad)
        xform = wp.transform(wp.vec3(x_pos, y_pos, z_pos), quat_b)

        # Add capsule B
        body_id = scene.add_body(xform=xform, key="test_capsule_b")
        scene.add_shape_capsule(
            body=body_id, radius=capsule_radius, half_height=0.5 * capsule_height, cfg=self.default_shape_cfg
        )

        # Store capsule metadata for the state machine.
        # Arm-to-capsule mapping: right arm (index 0) -> capsule A, left arm (index 1) -> capsule B.
        self.capsule_body_keys = ["test_capsule_a", "test_capsule_b"]
        self.capsule_tilt_angles = [tilt_angle_rad, -tilt_angle_rad]

        # Extraction directions: along capsule axis, away from its hose connector.
        self.capsule_extract_dirs = [quat_a, quat_b]

        # Additional capsule to drop just for debugging purposes.
        # xform.p = wp.vec3(0.0, 0.0, 0.1) + wp.vec3f(*table_top_center)
        # xform.q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)
        # body_id = scene.add_body(xform=xform, key="test_capsule_drop")
        # scene.add_shape_capsule(
        #     body=body_id, radius=capsule_radius, half_height=0.5 * capsule_height, cfg=self.default_sdf_shape_cfg
        # )

        # Mirror asset loading from example_cable_hose_connector.py
        from newton._src.geometry.utils import load_mesh  # noqa: PLC0415

        mesh_vertices, mesh_indices = load_mesh(ASSETS_PATH + "rby1_hose_connectorv3.stl")

        mesh_vertices = np.asarray(mesh_vertices, dtype=np.float32).reshape(-1, 3)
        mesh_indices = np.asarray(mesh_indices, dtype=np.int32).flatten()

        # Apply a fixed scale so the scene stays predictable (asset is authored in different units).
        scale_factor = 0.001  # Here we use smaller scale factor.
        mesh_center = wp.vec3(0.0, 0.0, 0.0)

        mesh_vertices_centered = (mesh_vertices - mesh_center) * scale_factor
        mesh = newton.Mesh(mesh_vertices_centered, mesh_indices, compute_inertia=True, is_solid=True)

        min_z = float(np.min(mesh_vertices_centered[:, 2]))
        mesh_z = float(-min_z)

        mesh_y_a = float(right_hose_y)
        mesh_pos_a = wp.vec3f(wp.float32(0.0), wp.float32(mesh_y_a), wp.float32(mesh_z)) + wp.vec3f(*table_top_center)

        mesh_y_b = float(left_hose_y)
        mesh_pos_b = wp.vec3f(wp.float32(0.0), wp.float32(mesh_y_b), wp.float32(mesh_z)) + wp.vec3f(*table_top_center)

        q_a = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.half_pi)
        q_b = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.half_pi)

        scene.add_shape_mesh(
            body=-1,
            mesh=mesh,
            xform=wp.transform(p=mesh_pos_a, q=q_a),
            cfg=self.default_ground_cfg,
            key="rby1_hose_connectorv3_a",
        )

        scene.add_shape_mesh(
            body=-1,
            mesh=mesh,
            xform=wp.transform(p=mesh_pos_b, q=q_b),
            cfg=self.default_ground_cfg,
            key="rby1_hose_connectorv3_b",
        )

        # Add table
        scene.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(self.table_pos)),
            hx=self.table_half_size[0],
            hy=self.table_half_size[1],
            hz=self.table_half_size[2],
            cfg=self.default_ground_cfg,
        )

        # Add ground plane
        scene.add_ground_plane(
            cfg=self.default_ground_cfg,
        )

        return scene

    def _start_auto_mode(self):
        """Begin the automated grasping sequence for both arms."""
        # Reset task indices to start of schedule (APPROACH)
        self.sm_task_idx.zero_()
        self.sm_task_time_elapsed.zero_()

        # Snapshot current EE transforms as entry points for interpolation
        body_q_np = self.state_0.body_q.numpy()
        init_tfs = []
        for arm_idx in range(self.num_arms):
            _, ee_link_idx = self.ee_configs[arm_idx]
            init_tfs.append(wp.transform(*body_q_np[ee_link_idx]))
        wp.copy(self.sm_task_init_body_q, wp.array(init_tfs, dtype=wp.transform))

        # Reset gripper targets to fully open
        wp.copy(self.gripper_targets, self.sm_gripper_open_values)

        print("Auto-grasp mode STARTED")

    def _stop_auto_mode(self):
        """Cancel the automated sequence and return to manual GUI control."""
        # Reset gripper targets to fully open
        wp.copy(self.gripper_targets, self.sm_gripper_open_values)
        print("Auto-grasp mode STOPPED")

    def _reset_state_machine(self):
        """Restart the automated sequence from the APPROACH state."""
        if self.auto_mode:
            self._start_auto_mode()

    def gui(self, ui):
        changed, self.show_isosurface = ui.checkbox("Show Isosurface", self.show_isosurface)
        if changed:
            self.viewer.show_hydro_contact_surface = self.show_isosurface

        self.gui_auto_grasp(ui)
        self.gui_gripper_controls(ui)
        self.gui_ee_target_controls(ui)
        self.gui_ik_settings(ui)

    def gui_auto_grasp(self, ui):
        """GUI section for the automated capsule grasping state machine."""
        if not ui.collapsing_header("Auto Grasp", flags=0):
            return

        # Toggle auto mode
        changed, value = ui.checkbox("Enable Auto Grasp", self.auto_mode)
        if changed:
            self.auto_mode = value
            if self.auto_mode:
                self._start_auto_mode()
            else:
                self._stop_auto_mode()

        # Reset button (only shown when auto mode is active)
        if self.auto_mode:
            if ui.button("Reset State Machine"):
                self._reset_state_machine()

        ui.separator()

        # Status display for each arm (read GPU arrays once)
        arm_labels = ["Right Arm", "Left Arm"]
        capsule_labels = ["Capsule A", "Capsule B"]
        if self.auto_mode:
            task_idx_np = self.sm_task_idx.numpy()
            task_time_np = self.sm_task_time_elapsed.numpy()
            schedule_np = self.sm_task_schedule.numpy()
            time_limits_np = self.sm_task_time_limits.numpy()
            ee_pos_target_np = self.sm_ee_pos_target.numpy()
            ee_rot_target_np = self.sm_ee_rot_target.numpy()
            body_q_np = self.state_0.body_q.numpy()
            ee_body_indices_np = self.sm_ee_body_indices.numpy()
            for arm_idx in range(self.num_arms):
                idx = int(task_idx_np[arm_idx])
                task_type = TaskType(int(schedule_np[idx]))
                elapsed = float(task_time_np[arm_idx])
                time_limit = float(time_limits_np[idx])
                ee_body_id = int(ee_body_indices_np[arm_idx])
                ee_pos = body_q_np[ee_body_id][:3]
                ee_rot = body_q_np[ee_body_id][3:]  # quaternion xyzw
                pos_err = ee_pos - ee_pos_target_np[arm_idx]
                pos_err_norm = float(np.linalg.norm(pos_err))
                # Rotation error (angular distance in radians)
                target_rot = ee_rot_target_np[arm_idx]
                quat_rel = wp.quat(*ee_rot) * wp.quat_inverse(wp.quat(*target_rot))
                rot_err = wp.abs(2.0 * wp.atan2(wp.length(quat_rel[:3]), quat_rel[3]))
                rot_err_deg = float(np.degrees(rot_err))
                ui.text(f"{arm_labels[arm_idx]} ({capsule_labels[arm_idx]}): {task_type.name}")
                if task_type != TaskType.DONE:
                    ui.text(f"  Time: {elapsed:.2f} / {time_limit:.1f} s")
                    ui.text(f"  Pos error: {pos_err_norm:.4f} m")
                    ui.text(f"  Pos error: {pos_err[0]:.4f} m, {pos_err[1]:.4f} m, {pos_err[2]:.4f} m")
                    ui.text(f"  Rot error: {rot_err:.4f} rad ({rot_err_deg:.2f} deg)")
        else:
            for arm_idx in range(self.num_arms):
                ui.text(f"{arm_labels[arm_idx]} ({capsule_labels[arm_idx]}): IDLE")

        ui.separator()

    def gui_ee_target_controls(self, ui):
        """GUI controls for end effector target positions and rotations."""
        if not ui.collapsing_header("End Effector Targets", flags=0):
            return

        # Position limits for sliders
        min_z_pos = self.table_pos[2] + self.table_half_size[2] + 0.001
        pos_limit_lower = [-1.0, -1.0, min_z_pos]  # Table height is 0.75, table thickness is 0.02
        pos_limit_upper = [1.0, 1.0, 0.9]

        # Rotation limits for sliders (in radians)
        rot_limit_lower = -np.pi
        rot_limit_upper = np.pi

        def update_ee_position(ee_idx, axis, value):
            """Update a single axis of an end effector's target position."""
            tf = self.ee_tfs[ee_idx]
            pos = list(wp.transform_get_translation(tf))
            pos[axis] = value
            rot = wp.transform_get_rotation(tf)
            self.ee_tfs[ee_idx] = wp.transform(wp.vec3(*pos), rot)

        def update_ee_rotation(ee_idx, axis, value):
            """Update a single axis of an end effector's target rotation (Euler angles)."""
            tf = self.ee_tfs[ee_idx]
            pos = wp.transform_get_translation(tf)
            # Get current rotation as Euler angles
            current_rot = wp.transform_get_rotation(tf)
            euler = self._quat_to_euler(current_rot)
            euler[axis] = value
            # Convert back to quaternion
            new_rot = self._euler_to_quat(euler)
            self.ee_tfs[ee_idx] = wp.transform(pos, new_rot)

        axis_names = ["X", "Y", "Z"]

        for ee_idx, (ee_name, _link_idx) in enumerate(self.ee_configs):
            # Short display name
            short_name = ee_name.replace("_end_effector", "").replace("_", " ").title()

            ui.text(f"{short_name}:")
            ui.separator()

            tf = self.ee_tfs[ee_idx]
            pos = wp.transform_get_translation(tf)
            rot = wp.transform_get_rotation(tf)
            euler = self._quat_to_euler(rot)

            # Position controls
            for axis in range(3):
                # Slider for position
                ui.text(f"{short_name} {axis_names[axis]}:")
                changed, value = ui.slider_float(
                    f"{axis_names[axis]}##pos_slider_{ee_idx}_{axis}",
                    pos[axis],
                    pos_limit_lower[axis],
                    pos_limit_upper[axis],
                    format="%.3f",
                )
                if changed:
                    update_ee_position(ee_idx, axis, value)

                # Input field for precise position input
                changed, value = ui.input_float(
                    f"{axis_names[axis]}##pos_input_{ee_idx}_{axis}",
                    pos[axis],
                    format="%.4f",
                )
                if changed:
                    value = min(max(value, pos_limit_lower[axis]), pos_limit_upper[axis])
                    update_ee_position(ee_idx, axis, value)

            # Rotation controls (Euler angles)
            rot_axis_names = ["Roll", "Pitch", "Yaw"]
            for axis in range(3):
                # Slider for rotation
                ui.text(f"{short_name} {rot_axis_names[axis]}:")
                changed, value = ui.slider_float(
                    f"{rot_axis_names[axis]}##rot_slider_{ee_idx}_{axis}",
                    euler[axis],
                    rot_limit_lower,
                    rot_limit_upper,
                    format="%.3f",
                )
                if changed:
                    update_ee_rotation(ee_idx, axis, value)

                # Input field for precise rotation input
                changed, value = ui.input_float(
                    f"{rot_axis_names[axis]}##rot_input_{ee_idx}_{axis}",
                    euler[axis],
                    format="%.4f",
                )
                if changed:
                    value = min(max(value, rot_limit_lower), rot_limit_upper)
                    update_ee_rotation(ee_idx, axis, value)

            ui.separator()

    def _quat_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # Extract quaternion components
        x, y, z, w = q[0], q[1], q[2], q[3]

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return [roll, pitch, yaw]

    def _euler_to_quat(self, euler):
        """Convert Euler angles (roll, pitch, yaw) to quaternion."""
        roll, pitch, yaw = euler

        # Compute half angles
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        # Compute quaternion components
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return wp.quat(x, y, z, w)

    def gui_gripper_controls(self, ui):
        if not ui.collapsing_header("Gripper Controls", flags=0):
            return

        def update_gripper_target(joint_idx, value):
            """Update a single gripper joint target."""
            self.gripper_targets_list[joint_idx] = value
            # Update GPU array
            gripper_np = self.gripper_targets.numpy()
            gripper_np[joint_idx] = value
            wp.copy(self.gripper_targets, wp.array(gripper_np, dtype=wp.float32))

        # Coupled control: move left and right grippers together
        ui.text("Coupled Controls:")
        ui.separator()

        # Left gripper (both fingers) - use limits from second left finger joint
        if len(self.gripper_targets_list) >= 2:
            changed, value = ui.slider_float(
                "Right Gripper",
                self.gripper_targets_list[1],
                self.gripper_limits_lower[1],
                self.gripper_limits_upper[1],
            )
            if changed:
                update_gripper_target(0, -value)
                update_gripper_target(1, value)

        # Right gripper (both fingers) - use limits from second right finger joint
        if len(self.gripper_targets_list) >= 4:
            changed, value = ui.slider_float(
                "Left Gripper",
                self.gripper_targets_list[3],
                self.gripper_limits_lower[3],
                self.gripper_limits_upper[3],
            )
            if changed:
                update_gripper_target(2, -value)
                update_gripper_target(3, value)

        ui.separator()

    def gui_ik_settings(self, ui):
        if not ui.collapsing_header("IK Settings", flags=0):
            return

        # Show end effector info
        ui.text("End Effectors:")
        for i, (name, idx) in enumerate(self.ee_configs):
            tf = self.ee_tfs[i]
            pos = wp.transform_get_translation(tf)
            ui.text(f"  {name} (body {idx})")
            ui.text(f"    pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    def test_final(self):
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=1, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, num_worlds=args.num_worlds, args=args)

    newton.examples.run(example, args)
