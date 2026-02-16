# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
#
###########################################################################
# Example MuJoCo-VBD Two-Way Coupling
#
# Demonstrates two-way coupling between MuJoCo (robot dynamics) and
# VBD (cable/soft body dynamics) following the virtual inertia method.
#
# Architecture:
# - Universe A (MuJoCo): Robot with articulation, motors, sensors
# - Universe B (VBD): Cable + Robot Proxies (approximation of dynamics)
# - Staggered coupling loop with lagged impulses
#
# Key Features:
# 1. Proxy Bodies: MuJoCo robot bodies duplicated in VBD for collision
# 2. State Sync: MuJoCo poses copied to VBD proxies each timestep
# 3. Force Harvesting: Contact forces from VBD applied to MuJoCo (contact-based)
# 4. Force Subtraction: Previously applied forces removed before VBD step
# 5. Lagged Impulses: Forces from step k applied at step k+1
#
# Command:
#   uv run -m newton.examples.cable.example_coupling_twoway
#
###########################################################################

import os
import struct
from enum import Enum, IntEnum
from numbers import Integral
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton import Contacts, GeoType


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


# Examples assets live in `newton/examples/assets/`.
def _default_assets_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assets"


ASSETS_ROOT = Path(os.environ.get("NEWTON_EXAMPLES_ASSETS_PATH", os.fspath(_default_assets_root()))).resolve()
HOSE_CONNECTOR_PATH = ASSETS_ROOT / "rby1_hose_connectorv3.stl"


def _load_stl_as_tri_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an STL file into (vertices, indices) arrays.

    This keeps the example self-contained:
    - No `newton._src` imports
    - No public API changes (e.g. no `newton.geometry.load_mesh`)

    Args:
        path: Path to an ASCII or binary STL.

    Returns:
        vertices: float32 array of shape (N, 3)
        indices: int32 array of shape (M,) (triangle indices, 3 per triangle)
    """
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL file too small: {path}")

    # Heuristic: if size matches binary STL layout, treat as binary.
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + 50 * tri_count
    is_binary = expected_size == len(data)

    if is_binary:
        vertices = np.empty((tri_count * 3, 3), dtype=np.float32)
        indices = np.arange(tri_count * 3, dtype=np.int32)

        offset = 84
        for t in range(tri_count):
            # normal (3 floats) then 3 vertices (9 floats) then 2-byte attribute
            offset += 12
            v = struct.unpack_from("<fffffffff", data, offset)
            offset += 36
            offset += 2  # attribute byte count

            base = 3 * t
            vertices[base + 0] = (v[0], v[1], v[2])
            vertices[base + 1] = (v[3], v[4], v[5])
            vertices[base + 2] = (v[6], v[7], v[8])

        return vertices, indices

    # ASCII STL
    text = data.decode("utf-8", errors="ignore")
    verts: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("vertex"):
            continue
        _tag, xs, ys, zs = s.split(maxsplit=3)
        verts.append([float(xs), float(ys), float(zs)])

    if len(verts) == 0 or (len(verts) % 3) != 0:
        raise ValueError(f"Failed to parse ASCII STL (no vertices): {path}")

    vertices = np.asarray(verts, dtype=np.float32)
    indices = np.arange(vertices.shape[0], dtype=np.int32)
    return vertices, indices


@wp.kernel
def sync_proxy_states_kernel(
    mj_body_q: wp.array(dtype=wp.transform),
    mj_body_qd: wp.array(dtype=wp.spatial_vector),
    mj_to_vbd_map: wp.array(dtype=int),
    vbd_body_q: wp.array(dtype=wp.transform),
    vbd_body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Copy MuJoCo body states to VBD proxy bodies."""
    mj_body_id = wp.tid()
    vbd_body_id = mj_to_vbd_map[mj_body_id]

    if vbd_body_id >= 0:
        vbd_body_q[vbd_body_id] = mj_body_q[mj_body_id]
        vbd_body_qd[vbd_body_id] = mj_body_qd[mj_body_id]


@wp.kernel
def sync_vbd_solver_prev_poses_from_velocity_kernel(
    proxy_vbd_body_ids: wp.array(dtype=int),
    dt: float,
    vbd_body_q: wp.array(dtype=wp.transform),
    vbd_body_qd: wp.array(dtype=wp.spatial_vector),
    vbd_solver_body_q_prev: wp.array(dtype=wp.transform),
):
    """Update SolverVBD.body_q_prev for teleported proxy bodies using their velocities.

    Notes:
    - Rotation integration is first-order using an axis-angle delta in world space.
    """
    i = wp.tid()
    if i >= proxy_vbd_body_ids.shape[0]:
        return

    b = proxy_vbd_body_ids[i]

    q_now = vbd_body_q[b]
    p_now = wp.transform_get_translation(q_now)
    r_now = wp.transform_get_rotation(q_now)

    qd = vbd_body_qd[b]
    v = wp.spatial_top(qd)
    w = wp.spatial_bottom(qd)

    wmag = wp.length(w)

    # Backwards integrate one step to build a plausible "previous" pose.
    p_prev = p_now - v * dt

    # q_prev = delta(-w*dt) * q_now
    angle = wmag * dt
    if angle > 0.0:
        axis = w / wmag
        delta_inv = wp.quat_from_axis_angle(axis, -angle)
        r_prev = delta_inv * r_now
    else:
        r_prev = r_now

    vbd_solver_body_q_prev[b] = wp.transform(p_prev, r_prev)


@wp.kernel
def capture_proxy_prev_poses_kernel(
    proxy_vbd_body_ids: wp.array(dtype=int),
    vbd_body_q: wp.array(dtype=wp.transform),
    proxy_prev_poses: wp.array(dtype=wp.transform),
):
    """Capture current VBD proxy poses into a compact per-proxy buffer."""
    i = wp.tid()
    if i >= proxy_vbd_body_ids.shape[0]:
        return
    b = proxy_vbd_body_ids[i]
    proxy_prev_poses[i] = vbd_body_q[b]


@wp.kernel
def write_proxy_prev_poses_to_solver_prev_kernel(
    proxy_vbd_body_ids: wp.array(dtype=int),
    proxy_prev_poses: wp.array(dtype=wp.transform),
    vbd_solver_body_q_prev: wp.array(dtype=wp.transform),
):
    """Write captured per-proxy previous poses back to SolverVBD.body_q_prev."""
    i = wp.tid()
    if i >= proxy_vbd_body_ids.shape[0]:
        return
    b = proxy_vbd_body_ids[i]
    vbd_solver_body_q_prev[b] = proxy_prev_poses[i]


@wp.kernel
def subtract_proxy_forces_kernel(
    dt: float,
    vbd_body_q: wp.array(dtype=wp.transform),
    proxy_forces: wp.array(dtype=wp.spatial_vector),
    proxy_mj_body_ids: wp.array(dtype=int),
    proxy_vbd_body_ids: wp.array(dtype=int),
    vbd_body_inv_mass: wp.array(dtype=float),
    vbd_body_inv_inertia: wp.array(dtype=wp.mat33),
    vbd_body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Subtract previously applied forces from VBD proxy velocities."""
    proxy_idx = wp.tid()

    if proxy_idx >= proxy_mj_body_ids.shape[0]:
        return

    mj_body_id = proxy_mj_body_ids[proxy_idx]
    vbd_body_id = proxy_vbd_body_ids[proxy_idx]

    # Get previously applied force (this was applied to MuJoCo)
    f = proxy_forces[mj_body_id]

    # Compute velocity change using VBD proxy mass/inertia (accounts for proxy_mass_scale)
    # This is the velocity change that VBD proxy would experience from this force.
    delta_v = dt * vbd_body_inv_mass[vbd_body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(vbd_body_q[vbd_body_id])
    delta_w = dt * wp.quat_rotate(r, vbd_body_inv_inertia[vbd_body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))

    # Subtract from VBD proxy velocity
    vbd_body_qd[vbd_body_id] = vbd_body_qd[vbd_body_id] - wp.spatial_vector(delta_v, delta_w)


@wp.kernel
def harvest_proxy_wrenches_from_contact_forces_kernel(
    # Contact buffers (per-contact, world frame)
    rigid_contact_count: wp.array(dtype=int),
    contact_body0: wp.array(dtype=wp.int32),
    contact_body1: wp.array(dtype=wp.int32),
    contact_point0_world: wp.array(dtype=wp.vec3),
    contact_point1_world: wp.array(dtype=wp.vec3),
    contact_force_on_body1: wp.array(dtype=wp.vec3),
    # VBD body inv mass (to filter static/kinematic env)
    vbd_body_inv_mass: wp.array(dtype=float),
    # Proxy mapping
    proxy_vbd_body_ids: wp.array(dtype=int),
    proxy_mj_body_ids: wp.array(dtype=int),
    # MuJoCo model data (to compute torque about COM)
    mj_body_com: wp.array(dtype=wp.vec3),
    mj_body_q: wp.array(dtype=wp.transform),
    # Output
    out_mj_body_f: wp.array(dtype=wp.spatial_vector),
):
    """Harvest coupling wrenches from solver-produced per-contact forces.

    The VBD solver provides per-contact forces in world frame via:
      `SolverVBD.collect_rigid_contact_forces(...)`

    Filtering:
      - Include contacts where exactly one body is a proxy
      - Exclude proxy<->proxy
      - Exclude proxy<->static/kinematic env (other body inv_mass <= 0), temporarily

    Force convention:
      - `contact_force_on_body1` is force applied to body1 at `contact_point1_world`
      - Force on body0 is `-contact_force_on_body1` at `contact_point0_world`
    """
    contact_id = wp.tid()
    rc = rigid_contact_count[0]
    if contact_id >= rc:
        return

    body0 = int(contact_body0[contact_id])
    body1 = int(contact_body1[contact_id])
    if body0 < 0 or body1 < 0:
        return

    # Determine if body0/body1 are proxies.
    is_proxy0 = int(0)
    is_proxy1 = int(0)
    mj_body_id0 = int(-1)
    mj_body_id1 = int(-1)
    for i in range(proxy_vbd_body_ids.shape[0]):
        proxy_body_id = int(proxy_vbd_body_ids[i])
        if proxy_body_id == body0:
            is_proxy0 = int(1)
            mj_body_id0 = int(proxy_mj_body_ids[i])
        if proxy_body_id == body1:
            is_proxy1 = int(1)
            mj_body_id1 = int(proxy_mj_body_ids[i])

    if (is_proxy0 + is_proxy1) != 1:
        return

    # Filter out env: require the non-proxy body to be dynamic (inv_mass > 0)
    other_body_id = body1 if is_proxy0 == 1 else body0
    if other_body_id < 0 or other_body_id >= int(vbd_body_inv_mass.shape[0]) or vbd_body_inv_mass[other_body_id] <= 0.0:
        return

    # Select MuJoCo body id, a point of application in world, and the force on the proxy in world.
    force_on_body1_world = contact_force_on_body1[contact_id]
    if is_proxy1 == 1:
        mj_body_id = mj_body_id1
        contact_point_world = contact_point1_world[contact_id]
        force_on_proxy_world = force_on_body1_world
    else:
        mj_body_id = mj_body_id0
        contact_point_world = contact_point0_world[contact_id]
        force_on_proxy_world = -force_on_body1_world

    if mj_body_id < 0:
        return
    if (
        mj_body_id >= int(out_mj_body_f.shape[0])
        or mj_body_id >= int(mj_body_q.shape[0])
        or mj_body_id >= int(mj_body_com.shape[0])
    ):
        return

    mj_body_tf_world = mj_body_q[mj_body_id]
    com_world = wp.transform_point(mj_body_tf_world, mj_body_com[mj_body_id])
    r_world = contact_point_world - com_world
    torque_world = wp.cross(r_world, force_on_proxy_world)
    wp.atomic_add(out_mj_body_f, mj_body_id, wp.spatial_vector(force_on_proxy_world, torque_world))


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
    extract_unseat_lift: float,
    extract_unseat_fraction: float,
    capsule_body_indices: wp.array(dtype=int),
    capsule_extract_dirs: wp.array(dtype=wp.quat),
    grasp_orientation_offset: wp.array(dtype=wp.vec4),
    gripper_open_values: wp.array(dtype=wp.float32),
    gripper_closed_values: wp.array(dtype=wp.float32),
    extract_offsets_world: wp.array(dtype=wp.vec3),
    task_ee_init_body_q: wp.array(dtype=wp.transform),
    task_capsule_body_q_prev: wp.array(dtype=wp.transform),
    robot_body_q: wp.array(dtype=wp.transform),
    capsule_body_q: wp.array(dtype=wp.transform),
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
    capsule_pos = wp.transform_get_translation(capsule_body_q[capsule_body_indices[arm_idx]])
    capsule_quat = wp.transform_get_rotation(capsule_body_q[capsule_body_indices[arm_idx]])

    # Grasp orientation for this arm (stored as vec4 xyzw)
    gv = grasp_orientation_offset[arm_idx]
    grasp_quat_offset = wp.quaternion(gv[:3], gv[3])

    # Default: hold previous pose
    ee_quat_target = ee_quat_prev
    t_gripper = 0.0

    if task == TaskType.APPROACH.value:
        grasp_pos_offset = wp.quat_rotate(capsule_quat, capsule_grasp_offset_from_com[arm_idx])
        ee_pos_target[arm_idx] = capsule_pos + grasp_pos_offset + approach_offsets[arm_idx]

        # Align EE orientation to capsule axis (ignore twist).
        capsule_axis = wp.quat_rotate(capsule_quat, wp.vec3(0.0, 0.0, 1.0))
        align_quat = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), capsule_axis)
        ee_quat_target = align_quat * grasp_quat_offset
    elif task == TaskType.ENGAGE.value:
        grasp_pos_offset = wp.quat_rotate(capsule_quat_prev, capsule_grasp_offset_from_com[arm_idx])
        ee_pos_target[arm_idx] = capsule_pos + grasp_pos_offset
        ee_quat_target = ee_quat_prev
    elif task == TaskType.GRASP.value:
        # While closing, keep the EE position aimed at the capsule grasp point (typically COM)
        # instead of freezing a potentially surface-biased pose from the ENGAGE transition.
        grasp_pos_offset = wp.quat_rotate(capsule_quat_prev, capsule_grasp_offset_from_com[arm_idx])
        ee_pos_target[arm_idx] = capsule_pos + grasp_pos_offset
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.HOLD_GRASP.value:
        # Maintain centering during hold as the capsule may move slightly under contact.
        grasp_pos_offset = wp.quat_rotate(capsule_quat_prev, capsule_grasp_offset_from_com[arm_idx])
        ee_pos_target[arm_idx] = capsule_pos + grasp_pos_offset
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.EXTRACT.value:
        # Follow the robot_ik-style extract reference: pull from the task-start EE pose.
        # Using capsule_pos + grasp_offset here can "double count" backward motion and over-pull.
        extract_axis = wp.quat_rotate(capsule_quat_prev, wp.vec3(0.0, 0.0, 1.0))
        # pull_t = t * t  # smooth ramp from 0 -> 1
        pull_t = t
        extract_vec = extract_axis * (2.0 * pull_t * extract_distance)
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

    # Gripper targets: ramp from open (`gripper_open_values`) -> closed (`gripper_closed_values`)
    # as t_gripper goes 0 -> 1.
    base = arm_idx * 2
    # Over-scale the gripper targets to make the gripper open and close faster/stronger
    # Avoid overly aggressive gripper commands; too-strong closing can cause deep penetrations
    # and penalty-force blow-ups in the VBD proxy<->capsule contacts.
    over_scale = 1.0
    g0_open = gripper_open_values[base]
    g1_open = gripper_open_values[base + 1]
    g0_closed = gripper_closed_values[base]
    g1_closed = gripper_closed_values[base + 1]

    gripper_target[arm_idx, 0] = over_scale * (g0_open * (1.0 - t_gripper) + g0_closed * t_gripper)
    gripper_target[arm_idx, 1] = over_scale * (g1_open * (1.0 - t_gripper) + g1_closed * t_gripper)


@wp.kernel(enable_backward=False)
def apply_gripper_centering_correction_kernel(
    task_schedule: wp.array(dtype=wp.int32),
    task_idx: wp.array(dtype=int),
    vbd_body_q: wp.array(dtype=wp.transform),
    capsule_body_indices: wp.array(dtype=int),
    finger_proxy_body_indices: wp.array2d(dtype=int),
    k_center: float,
    max_step: float,
    # in/out
    ee_pos_target_interpolated: wp.array(dtype=wp.vec3),
):
    """Bias the EE target laterally to keep the capsule centered between the two finger proxies.

    This is a purely geometric correction computed in VBD world coordinates using the proxy
    finger body positions. It helps maintain symmetric two-sided contact (normal load on both
    sides), which is required for reliable towing during EXTRACT.
    """
    arm = wp.tid()
    idx = task_idx[arm]
    task = task_schedule[idx]
    # Only apply when we want to maintain a symmetric pinch.
    if not (
        task == TaskType.ENGAGE.value
        or task == TaskType.GRASP.value
        or task == TaskType.HOLD_GRASP.value
        or task == TaskType.EXTRACT.value
    ):
        return

    cap = capsule_body_indices[arm]
    f0 = finger_proxy_body_indices[arm, 0]
    f1 = finger_proxy_body_indices[arm, 1]

    if cap < 0 or f0 < 0 or f1 < 0:
        return

    pcap = wp.transform_get_translation(vbd_body_q[cap])
    pf0 = wp.transform_get_translation(vbd_body_q[f0])
    pf1 = wp.transform_get_translation(vbd_body_q[f1])

    u = pf1 - pf0
    ulen = wp.length(u)
    if ulen < 1.0e-8:
        return
    u = u / ulen

    mid = 0.5 * (pf0 + pf1)
    off_u = wp.dot(pcap - mid, u)  # + means capsule shifted toward finger1 side along closing axis

    # Move EE opposite the offset to re-center capsule between fingers.
    delta = -k_center * off_u * u
    dlen = wp.length(delta)
    if dlen > max_step and dlen > 0.0:
        delta = delta * (max_step / dlen)

    ee_pos_target_interpolated[arm] = ee_pos_target_interpolated[arm] + delta


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_schedule: wp.array(dtype=wp.int32),
    task_time_soft_limits: wp.array(dtype=float),
    ee_pos_target: wp.array(dtype=wp.vec3),
    ee_rot_target: wp.array(dtype=wp.vec4),
    robot_body_q: wp.array(dtype=wp.transform),
    capsule_body_q: wp.array(dtype=wp.transform),
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
    ee_pos_current = wp.transform_get_translation(robot_body_q[ee_body_id])
    ee_rot_current = wp.transform_get_rotation(robot_body_q[ee_body_id])

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
        task_ee_init_body_q[arm_idx] = robot_body_q[ee_body_id]
        task_capsule_body_q_prev[arm_idx] = capsule_body_q[capsule_body_indices[arm_idx]]


ROBOT_PATH = ASSETS_ROOT / "rby1df" / "urdf"


class Example:
    def __init__(self, viewer, num_worlds=1, args=None):
        # ----------------------------------------------------------------
        # Common (both MuJoCo and VBD)
        # ----------------------------------------------------------------
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.num_worlds = num_worlds
        self.viewer = viewer

        # Keep the example quiet by default (avoid spamming stdout).
        self.verbose = False
        self.frame_count = 0
        self.use_graph = wp.get_device().is_cuda
        self.enable_auto_grasp = True

        # ----------------------------------------------------------------
        # Coupling
        # ----------------------------------------------------------------
        self.enable_two_way_coupling = True
        self.exchange_forces_every_substep = True

        # Source used when reconstructing SolverVBD proxy previous pose:
        # - "snapshot": store proxy previous pose each substep and reuse it (default).
        # - "integrate_velocity": reconstruct previous pose from current VBD velocity.
        self.coupling_proxy_prevpose_velocity_source = "snapshot"
        # self.coupling_proxy_prevpose_velocity_source = "integrate_velocity"

        if self.coupling_proxy_prevpose_velocity_source not in ("snapshot", "integrate_velocity"):
            raise ValueError("coupling_proxy_prevpose_velocity_source must be one of: 'snapshot', 'integrate_velocity'")

        # Shared table geometry used by both MuJoCo and VBD setup.
        self.table_half_size = [0.25, 0.5, 0.02]
        self.table_pos = [0.5, 0, 0.75]

        # ----------------------------------------------------------------
        # MuJoCo
        # ----------------------------------------------------------------
        self.mujoco_substeps = 20
        self.mujoco_substep_dt = self.frame_dt / self.mujoco_substeps

        self.mujoco_collision_mode = CollisionMode.NEWTON_SDF
        self.mujoco_collide_substeps = 2  # run MuJoCo collision every X MuJoCo substeps
        self.mujoco_iterations = 50
        self.mujoco_ls_iterations = 25
        self.enable_robot = True
        self.rigid_contact_max = 100000

        # IK settings.
        self.ik_iters = 24

        # Gripper joint drive tuning.
        self.gripper_joint_target_ke = 50.0
        self.gripper_joint_target_kd = 80.0
        self.gripper_joint_effort_limit = 500.0

        # MuJoCo scene defaults derived from collision mode.
        self.mujoco_default_shape_cfg = self._create_shape_config(self.mujoco_collision_mode)

        # Keep ground settings separate from default shape settings.
        self.mujoco_default_ground_cfg = self.mujoco_default_shape_cfg.copy()
        self.mujoco_default_ground_cfg.ke = 1e4
        self.mujoco_default_ground_cfg.kd = 1e0
        self.mujoco_default_ground_cfg.mu = 0.3

        # ----------------------------------------------------------------
        # State machine tuning
        # ----------------------------------------------------------------
        # Approach/grasp alignment.
        # APPROACH aims near capsule COM with +/-Y offset to avoid early collisions.
        # self.sm_approach_offset_y = 0.03
        # self.sm_approach_offset_z = 0.01
        self.sm_approach_offset_y = 0.0
        self.sm_approach_offset_z = 0.0

        # Grasp point along capsule local axis (0.5 == COM).
        self.sm_grasp_axis_fraction = 2.0 / 3.0

        # Task durations (seconds).
        self.sm_time_approach = 1.0
        self.sm_time_engage = 1.0
        self.sm_time_grasp = 3.0
        self.sm_time_hold_grasp = 3.0
        self.sm_time_extract = 3.0

        # Extraction helper offsets (currently not applied by set_target_pose_kernel,
        # but kept for optional future tuning).
        self.sm_extract_lift_z = 0.03
        self.sm_extract_center_y = 0.05

        # Pure pull in EXTRACT: keep unseat lift disabled.
        self.extract_unseat_lift = 0.0
        self.extract_unseat_fraction = 0.0

        self._setup_mujoco_world(args)

        # ----------------------------------------------------------------
        # VBD
        # ----------------------------------------------------------------
        self.vbd_substeps = 20
        self.vbd_iterations = 20

        self.vbd_collide_substeps = 1  # run VBD collision every X VBD substeps
        self.vbd_collision_pipeline_type = "unified"
        self.vbd_mesh_use_sdf = True
        self.vbd_mesh_sdf_max_resolution = 64

        self.vbd_default_contact_ke = 1.0e5
        self.vbd_default_contact_kd = 1.0e-1
        self.vbd_default_contact_margin = 0.001

        self.vbd_solver_friction_epsilon = 0.1
        self.vbd_rigid_contact_buffer_size = 512

        self.proxy_mass_scale = 1.0  # Start with 1x (real mass).
        self.vbd_proxy_mu = 1000.0
        self.vbd_proxy_thickness = 0.001

        self.vbd_capsule_mu = 1.0
        self.vbd_capsule_thickness = 0.0
        self.vbd_capsule_contact_margin = 0.001
        self.vbd_capsule_mass = 0.01

        self.vbd_env_mu = 0.1  # Static colliding objects
        self.vbd_env_thickness = 1.0e-4
        self.vbd_env_contact_margin = 0.001

        # Capsule geometry
        self.capsule_radius = 0.003
        self.capsule_cylinder_height = 4.0 / 60.0
        self.capsule_tilt_angle_deg = 30.0
        self.capsule_length_offset = 0.01
        self.capsule_spawn_x_bias = 0.005
        self.hose_y_offset = 0.15

        # Initial capsule spawn tweak: offset the capsule a bit along its own axis
        self.capsule_spawn_axis_offset = 0.01

        self.vbd_rigid_avbd_beta = 1.0e5
        self.vbd_rigid_contact_k_start = 1.0e2

        self._setup_vbd_world_and_coupling(args)

        # ----------------------------------------------------------------
        # MuJoCo planning/control setup (depends on VBD capsule setup)
        # ----------------------------------------------------------------

        if self.enable_robot:
            self.setup_end_effectors()
            self.setup_ik()
            self.setup_gripper_targets()
            self.setup_state_machine()
            # Optional auto grasp/state-machine startup.
            self.auto_mode = bool(self.enable_auto_grasp)
            if self.auto_mode:
                self._start_auto_mode()

            # Store joint target positions for merging.
            self.joint_target_pos = wp.zeros_like(self.control.joint_target_pos)
            wp.copy(self.joint_target_pos, self.control.joint_target_pos)

        # ----------------------------------------------------------------
        # Viewer
        # ----------------------------------------------------------------
        # Start paused in interactive GL viewer; leave other viewer types untouched.
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = isinstance(self.viewer, newton.viewer.ViewerGL)
        self.show_isosurface = False
        # self.viewer_primary_model = "mujoco"  # Change to "vbd" to view VBD as primary
        self.viewer_primary_model = "vbd"
        # self.viewer_camera_mode = "front_view"
        self.viewer_camera_mode = "side_view"

        if self.enable_robot:
            if self.viewer_primary_model == "mujoco":
                self.viewer.set_model(self.mujoco_model)
            else:
                self.viewer.set_model(self.vbd_model)
        else:
            self.viewer.set_model(self.vbd_model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            if self.viewer_camera_mode == "front_view":
                self.viewer.set_camera(wp.vec3(6.5, 0.0, 1.6), pitch=-5.0, yaw=-180.0)
            elif self.viewer_camera_mode == "side_view":
                self.viewer.set_camera(wp.vec3(0.5, 4.0, 1.25), pitch=-5.0, yaw=-90.0)
            else:
                raise ValueError(
                    f"viewer_camera_mode must be 'front_view' or 'side_view', got: {self.viewer_camera_mode}"
                )
            self.viewer.camera.fov = 15.0
            # self.viewer.picking_enabled = False  # Disable interactive GUI picking for this example

        self.capture()

    def _setup_mujoco_world(self, args=None):
        """Build and initialize the MuJoCo-side robot world and solver state."""
        # Gripper DOF indices are discovered from joint names after URDF import
        # (see setup_robot_builder()).
        self.gripper_joint_dofs = []

        if not self.enable_robot:
            self.single_robot_model = None
            self.mujoco_model = None
            self.collision_pipeline = None
            self._use_mujoco_contacts = False
            self.mujoco_solver = None
            self.mujoco_state_0 = None
            self.mujoco_state_1 = None
            self.control = None
            self.mujoco_contacts = None
            return

        robot = self.setup_robot_builder()
        scene = self.setup_scene_builder(robot)

        self.single_robot_model = robot.finalize()
        self.mujoco_model = scene.finalize()

        self.collision_pipeline = self._create_collision_pipeline(self.mujoco_collision_mode, args)
        newton.eval_fk(self.mujoco_model, self.mujoco_model.joint_q, self.mujoco_model.joint_qd, self.mujoco_model)

        # Collision path:
        # - MUJOCO mode: MuJoCo native contacts
        # - otherwise: Newton collision pipeline contacts
        self._use_mujoco_contacts = self.mujoco_collision_mode == CollisionMode.MUJOCO
        num_per_world = self.rigid_contact_max // self.num_worlds
        self.mujoco_solver = newton.solvers.SolverMuJoCo(
            self.mujoco_model,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=num_per_world,
            nconmax=num_per_world,
            ls_parallel=True,
            iterations=int(self.mujoco_iterations),
            ls_iterations=int(self.mujoco_ls_iterations),
            use_mujoco_contacts=self._use_mujoco_contacts,
            impratio=1000.0,
        )

        if self.verbose:
            print(f"Collision mode: {self.mujoco_collision_mode.value}")
            print(f"  use_mujoco_contacts: {self._use_mujoco_contacts}")

        self.mujoco_state_0 = self.mujoco_model.state()
        self.mujoco_state_1 = self.mujoco_model.state()
        self.control = self.mujoco_model.control()

        newton.eval_fk(self.mujoco_model, self.mujoco_model.joint_q, self.mujoco_model.joint_qd, self.mujoco_state_0)
        if self._use_mujoco_contacts:
            self.mujoco_contacts = Contacts(0, 0)
        else:
            self.mujoco_contacts = self.mujoco_model.collide(
                self.mujoco_state_0, collision_pipeline=self.collision_pipeline
            )

    def capture(self):
        self.capture_sim()
        if self.enable_robot:
            self.capture_ik()

    def capture_sim(self):
        self.graph_sim = None
        if not self.use_graph:
            return
        # Preserve initial states so graph capture doesn't advance them.
        if self.enable_robot:
            state_0_backup = self.mujoco_model.state()
            state_1_backup = self.mujoco_model.state()
            state_0_backup.assign(self.mujoco_state_0)
            state_1_backup.assign(self.mujoco_state_1)
        else:
            state_0_backup = None
            state_1_backup = None

        vbd_state_0_backup = self.vbd_model.state()
        vbd_state_1_backup = self.vbd_model.state()
        vbd_state_0_backup.assign(self.vbd_state_0)
        vbd_state_1_backup.assign(self.vbd_state_1)

        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph_sim = capture.graph

        if self.enable_robot:
            self.mujoco_state_0.assign(state_0_backup)
            self.mujoco_state_1.assign(state_1_backup)
        self.vbd_state_0.assign(vbd_state_0_backup)
        self.vbd_state_1.assign(vbd_state_1_backup)

    def capture_ik(self):
        self.graph_ik = None
        if self.use_graph:
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
            self.graph_ik = capture.graph

    def simulate(self):
        """Two-way coupling simulation loop.

        Implements staggered coupling:
        1. Apply VBD forces to MuJoCo (from previous frame)
        2. Step MuJoCo (with substeps)
        3. Sync proxy states (MuJoCo -> VBD)
        4. Subtract previously applied forces from VBD
        5. Step VBD (once per frame)
        6. Harvest new forces from VBD contacts
        """
        mujoco_substep_count = max(int(self.mujoco_substeps), 1)
        vbd_substep_count = max(int(self.vbd_substeps), 1)
        vbd_collide_substeps = max(int(self.vbd_collide_substeps), 1)
        vbd_collide_substeps_independent = vbd_collide_substeps

        # Local aliases keep solver ownership explicit in this long routine.
        mujoco_state_0 = self.mujoco_state_0
        mujoco_state_1 = self.mujoco_state_1
        vbd_state_0 = self.vbd_state_0
        vbd_state_1 = self.vbd_state_1
        mujoco_collision_step_counter = 0
        vbd_collision_step_counter = 0

        if self.exchange_forces_every_substep:
            # ------------------------------------------------------------
            # Shared cadence:
            # use N=max(mujoco_substeps, vbd_substeps) and exchange forces every substep.
            # ------------------------------------------------------------
            shared_substep_count = max(mujoco_substep_count, vbd_substep_count)
            shared_substep_dt = float(self.frame_dt / shared_substep_count)

            for _shared_substep_idx in range(shared_substep_count):
                # --- Step 1: Apply lagged VBD->MuJoCo wrench ---
                if self.enable_robot:
                    mujoco_state_0.clear_forces()
                    self.viewer.apply_forces(mujoco_state_0)

                    if self.enable_two_way_coupling and hasattr(self, "proxy_forces"):
                        # Build the exact wrench we will apply this substep.
                        self.coupling_wrenches_applied.assign(self.proxy_forces)
                        applied_coupling_wrenches = self.coupling_wrenches_applied
                        mujoco_state_0.body_f.assign(mujoco_state_0.body_f + self.coupling_wrenches_applied)
                    else:
                        applied_coupling_wrenches = None

                    # --- Step 2: Step MuJoCo ---
                    if not self._use_mujoco_contacts and (
                        mujoco_collision_step_counter % self.mujoco_collide_substeps == 0
                    ):
                        self.mujoco_contacts = self.mujoco_model.collide(
                            mujoco_state_0, collision_pipeline=self.collision_pipeline
                        )

                    self.mujoco_solver.step(
                        mujoco_state_0, mujoco_state_1, self.control, self.mujoco_contacts, shared_substep_dt
                    )
                    mujoco_collision_step_counter += 1
                    mujoco_state_0, mujoco_state_1 = mujoco_state_1, mujoco_state_0
                    self.mujoco_state_0, self.mujoco_state_1 = mujoco_state_0, mujoco_state_1

                # --- Step 3: Sync proxy states (MuJoCo -> VBD) ---
                if self.enable_robot and hasattr(self, "mj_to_vbd_body_map_array"):
                    # Optionally snapshot current proxy poses before teleport-sync from MuJoCo.
                    if (
                        self.coupling_proxy_prevpose_velocity_source == "snapshot"
                        and hasattr(self, "proxy_vbd_body_ids_array")
                        and hasattr(self, "proxy_prev_poses")
                    ):
                        wp.launch(
                            capture_proxy_prev_poses_kernel,
                            dim=len(self.proxy_body_ids),
                            inputs=[
                                self.proxy_vbd_body_ids_array,
                                vbd_state_0.body_q,
                                self.proxy_prev_poses,
                            ],
                        )

                    wp.launch(
                        sync_proxy_states_kernel,
                        dim=self.mujoco_model.body_count,
                        inputs=[
                            mujoco_state_0.body_q,
                            mujoco_state_0.body_qd,
                            self.mj_to_vbd_body_map_array,
                            vbd_state_0.body_q,
                            vbd_state_0.body_qd,
                        ],
                    )

                    # Keep SolverVBD's `body_q_prev` consistent for teleported proxies, but preserve
                    # tangential slip so friction can actually act (use velocity-consistent prev poses).
                    if hasattr(self, "vbd_solver") and hasattr(self.vbd_solver, "body_q_prev"):
                        if self.coupling_proxy_prevpose_velocity_source == "snapshot" and hasattr(
                            self, "proxy_prev_poses"
                        ):
                            wp.launch(
                                write_proxy_prev_poses_to_solver_prev_kernel,
                                dim=len(self.proxy_body_ids),
                                inputs=[
                                    self.proxy_vbd_body_ids_array,
                                    self.proxy_prev_poses,
                                    self.vbd_solver.body_q_prev,
                                ],
                            )
                        else:
                            wp.launch(
                                sync_vbd_solver_prev_poses_from_velocity_kernel,
                                dim=len(self.proxy_body_ids),
                                inputs=[
                                    self.proxy_vbd_body_ids_array,
                                    float(shared_substep_dt),
                                    vbd_state_0.body_q,
                                    vbd_state_0.body_qd,
                                    self.vbd_solver.body_q_prev,
                                ],
                            )

                # --- Step 4: Subtract previously applied wrench effect before VBD contact ---
                if (
                    self.enable_two_way_coupling
                    and self.enable_robot
                    and applied_coupling_wrenches is not None
                    and hasattr(self, "proxy_mj_body_ids_array")
                ):
                    # Store the *applied* wrench so subtraction matches what MuJoCo felt this substep.
                    self.proxy_forces_prev.assign(applied_coupling_wrenches)
                    wp.launch(
                        subtract_proxy_forces_kernel,
                        dim=len(self.proxy_body_ids),
                        inputs=[
                            shared_substep_dt,
                            vbd_state_0.body_q,
                            self.proxy_forces_prev,
                            self.proxy_mj_body_ids_array,
                            self.proxy_vbd_body_ids_array,
                            self.vbd_model.body_inv_mass,
                            self.vbd_model.body_inv_inertia,
                            vbd_state_0.body_qd,  # input/output
                        ],
                    )

                # --- Step 5: Step VBD ---
                update_vbd_history = (vbd_collision_step_counter % vbd_collide_substeps == 0) or (
                    self.vbd_contacts is None
                )
                if update_vbd_history:
                    self.vbd_contacts = self.vbd_model.collide(
                        vbd_state_0, collision_pipeline=self.vbd_collision_pipeline
                    )
                self.vbd_solver.set_rigid_history_update(bool(update_vbd_history))

                self.vbd_solver.step(vbd_state_0, vbd_state_1, self.vbd_control, self.vbd_contacts, shared_substep_dt)
                vbd_collision_step_counter += 1

                # --- Step 6: Harvest new VBD->MuJoCo wrench for the next substep ---
                if self.enable_two_way_coupling and self.enable_robot and hasattr(self, "proxy_vbd_body_ids_array"):
                    self.proxy_forces.zero_()

                    # Contact-filtered harvesting:
                    # Ask SolverVBD for per-contact forces and application points, then map only
                    # proxy<->dynamic contacts back to MuJoCo.
                    if (
                        hasattr(self, "vbd_contacts")
                        and self.vbd_contacts is not None
                        and hasattr(self, "vbd_solver")
                        and hasattr(self.vbd_solver, "collect_rigid_contact_forces")
                    ):
                        c_b0, c_b1, c_p0w, c_p1w, c_f_b1, c_count = self.vbd_solver.collect_rigid_contact_forces(
                            vbd_state_1, self.vbd_contacts, float(shared_substep_dt)
                        )
                        wp.launch(
                            harvest_proxy_wrenches_from_contact_forces_kernel,
                            dim=c_b0.shape[0],
                            inputs=[
                                c_count,
                                c_b0,
                                c_b1,
                                c_p0w,
                                c_p1w,
                                c_f_b1,
                                self.vbd_model.body_inv_mass,
                                self.proxy_vbd_body_ids_array,
                                self.proxy_mj_body_ids_array,
                                self.mujoco_model.body_com,
                                mujoco_state_0.body_q,
                                self.proxy_forces,
                            ],
                        )

                vbd_state_0, vbd_state_1 = vbd_state_1, vbd_state_0
                self.vbd_state_0, self.vbd_state_1 = vbd_state_0, vbd_state_1

        else:
            # ------------------------------------------------------------
            # Independent cadence:
            # MuJoCo steps `mujoco_substep_count` times (`mujoco_substep_dt`),
            # VBD steps `vbd_substep_count` times (`dt_vbd`).
            # Coupling is exchanged at the frame boundary (lagged).
            # ------------------------------------------------------------
            mujoco_substep_dt = float(self.frame_dt / mujoco_substep_count)
            dt_vbd = float(self.frame_dt / vbd_substep_count)

            # --- MuJoCo substeps (apply lagged proxy_forces throughout the frame) ---
            for _mujoco_substep_idx in range(mujoco_substep_count):
                if self.enable_robot:
                    mujoco_state_0.clear_forces()
                    self.viewer.apply_forces(mujoco_state_0)

                    if self.enable_two_way_coupling and hasattr(self, "proxy_forces"):
                        self.coupling_wrenches_applied.assign(self.proxy_forces)
                        mujoco_state_0.body_f.assign(mujoco_state_0.body_f + self.coupling_wrenches_applied)

                    if not self._use_mujoco_contacts and (
                        mujoco_collision_step_counter % self.mujoco_collide_substeps == 0
                    ):
                        self.mujoco_contacts = self.mujoco_model.collide(
                            mujoco_state_0, collision_pipeline=self.collision_pipeline
                        )

                    self.mujoco_solver.step(
                        mujoco_state_0, mujoco_state_1, self.control, self.mujoco_contacts, mujoco_substep_dt
                    )
                    mujoco_collision_step_counter += 1
                    mujoco_state_0, mujoco_state_1 = mujoco_state_1, mujoco_state_0
                    self.mujoco_state_0, self.mujoco_state_1 = mujoco_state_0, mujoco_state_1

            # --- Sync proxies at the end of MuJoCo frame ---
            if self.enable_robot and hasattr(self, "mj_to_vbd_body_map_array"):
                # Optionally snapshot current proxy poses before teleport-sync from MuJoCo.
                if (
                    self.coupling_proxy_prevpose_velocity_source == "snapshot"
                    and hasattr(self, "proxy_vbd_body_ids_array")
                    and hasattr(self, "proxy_prev_poses")
                ):
                    wp.launch(
                        capture_proxy_prev_poses_kernel,
                        dim=len(self.proxy_body_ids),
                        inputs=[
                            self.proxy_vbd_body_ids_array,
                            vbd_state_0.body_q,
                            self.proxy_prev_poses,
                        ],
                    )

                wp.launch(
                    sync_proxy_states_kernel,
                    dim=self.mujoco_model.body_count,
                    inputs=[
                        mujoco_state_0.body_q,
                        mujoco_state_0.body_qd,
                        self.mj_to_vbd_body_map_array,
                        vbd_state_0.body_q,
                        vbd_state_0.body_qd,
                    ],
                )
                # Keep SolverVBD previous-pose history consistent for teleported proxies.
                if hasattr(self, "vbd_solver") and hasattr(self.vbd_solver, "body_q_prev"):
                    if self.coupling_proxy_prevpose_velocity_source == "snapshot" and hasattr(self, "proxy_prev_poses"):
                        wp.launch(
                            write_proxy_prev_poses_to_solver_prev_kernel,
                            dim=len(self.proxy_body_ids),
                            inputs=[
                                self.proxy_vbd_body_ids_array,
                                self.proxy_prev_poses,
                                self.vbd_solver.body_q_prev,
                            ],
                        )
                    else:
                        wp.launch(
                            sync_vbd_solver_prev_poses_from_velocity_kernel,
                            dim=len(self.proxy_body_ids),
                            inputs=[
                                self.proxy_vbd_body_ids_array,
                                float(dt_vbd),
                                vbd_state_0.body_q,
                                vbd_state_0.body_qd,
                                self.vbd_solver.body_q_prev,
                            ],
                        )

            # --- Subtract previously applied (lagged) wrench effect before VBD frame ---
            if self.enable_two_way_coupling and self.enable_robot and hasattr(self, "proxy_mj_body_ids_array"):
                # Subtract the effect of the applied coupling wrench over the whole frame.
                self.coupling_wrenches_applied.assign(self.proxy_forces)
                self.proxy_forces_prev.assign(self.coupling_wrenches_applied)
                wp.launch(
                    subtract_proxy_forces_kernel,
                    dim=len(self.proxy_body_ids),
                    inputs=[
                        self.frame_dt,
                        vbd_state_0.body_q,
                        self.proxy_forces_prev,
                        self.proxy_mj_body_ids_array,
                        self.proxy_vbd_body_ids_array,
                        self.vbd_model.body_inv_mass,  # Use VBD mass (accounts for proxy_mass_scale)
                        self.vbd_model.body_inv_inertia,  # Use VBD inertia (accounts for proxy_mass_scale)
                        vbd_state_0.body_qd,  # input/output
                    ],
                )

            # --- VBD substeps ---
            if self.enable_two_way_coupling and self.enable_robot and hasattr(self, "proxy_vbd_body_ids_array"):
                self.proxy_forces.zero_()

            for _ in range(vbd_substep_count):
                update_vbd_history = (vbd_collision_step_counter % vbd_collide_substeps_independent == 0) or (
                    self.vbd_contacts is None
                )
                if update_vbd_history:
                    self.vbd_contacts = self.vbd_model.collide(
                        vbd_state_0, collision_pipeline=self.vbd_collision_pipeline
                    )
                self.vbd_solver.set_rigid_history_update(bool(update_vbd_history))
                self.vbd_solver.step(vbd_state_0, vbd_state_1, self.vbd_control, self.vbd_contacts, dt_vbd)
                vbd_collision_step_counter += 1

                # Harvest per-substep solver forces and accumulate a frame-average coupling force:
                # proxy_forces ~ (1/frame_dt) * sum(F_sub * dt_vbd)
                if self.enable_two_way_coupling and self.enable_robot and hasattr(self, "proxy_vbd_body_ids_array"):
                    self.proxy_forces_step.zero_()
                    if (
                        hasattr(self, "vbd_contacts")
                        and self.vbd_contacts is not None
                        and hasattr(self, "vbd_solver")
                        and hasattr(self.vbd_solver, "collect_rigid_contact_forces")
                    ):
                        c_b0, c_b1, c_p0w, c_p1w, c_f_b1, c_count = self.vbd_solver.collect_rigid_contact_forces(
                            vbd_state_1, self.vbd_contacts, float(dt_vbd)
                        )
                        # GPU-only path: launch over fixed contact-buffer capacity and early-out in kernel
                        # using c_count[0]. This avoids device->host readback here.
                        wp.launch(
                            harvest_proxy_wrenches_from_contact_forces_kernel,
                            dim=c_b0.shape[0],
                            inputs=[
                                c_count,
                                c_b0,
                                c_b1,
                                c_p0w,
                                c_p1w,
                                c_f_b1,
                                self.vbd_model.body_inv_mass,
                                self.proxy_vbd_body_ids_array,
                                self.proxy_mj_body_ids_array,
                                self.mujoco_model.body_com,
                                mujoco_state_0.body_q,
                                self.proxy_forces_step,
                            ],
                        )
                    alpha = float(dt_vbd / self.frame_dt)
                    self.proxy_forces.assign(self.proxy_forces + self.proxy_forces_step * alpha)

                vbd_state_0, vbd_state_1 = vbd_state_1, vbd_state_0
                self.vbd_state_0, self.vbd_state_1 = vbd_state_0, vbd_state_1

    def setup_ik(self):
        """Set up IK solver with position and rotation objectives for each end effector."""

        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        # Get current body transforms
        body_q_np = self.mujoco_state_0.body_q.numpy()

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
            # IMPORTANT: joint limit arrays must match the IK solver model (single_robot_model),
            # not the full scene model.
            joint_limit_lower=self.single_robot_model.joint_limit_lower,
            joint_limit_upper=self.single_robot_model.joint_limit_upper,
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

        if self.verbose:
            print(f"IK solver initialized with {len(self.ee_configs)} end effector(s)")

    def setup_end_effectors(self):
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
                idx = self.mujoco_model.body_key.index(key)
                self.ee_configs.append((key, idx))
                if self.verbose:
                    print(f"End effector: {key} (body index {idx})")
            except ValueError:
                if self.verbose:
                    print(f"WARNING: End effector key not found: {key}")
                    print(f"  Available keys: {self.mujoco_model.body_key}")

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
        self.gripper_limits_lower = self.mujoco_model.joint_limit_lower.numpy()[self.gripper_joint_dofs]
        self.gripper_limits_upper = self.mujoco_model.joint_limit_upper.numpy()[self.gripper_joint_dofs]

        # Initialize gripper target positions
        # Open values (used at APPROACH/ENGAGE). These are within the joint limits.
        self.gripper_targets_list = [-0.04, 0.04, -0.04, 0.04]

        # Closed values (used at HOLD_GRASP/EXTRACT). For this model, the "more closed" posture is
        # near the inner joint limits (right finger joint near lower, left finger joint near upper).
        # We stay slightly inside limits to avoid hard-limit chatter.
        eps = 1.0e-4
        gl = self.gripper_limits_lower.astype(np.float64, copy=False)
        gu = self.gripper_limits_upper.astype(np.float64, copy=False)
        self.gripper_closed_values_list = [
            float(gu[0] - eps),  # right_gripper_left_finger_joint (upper)
            float(gl[1] + eps),  # right_gripper_right_finger_joint (lower)
            float(gu[2] - eps),  # left_gripper_left_finger_joint (upper)
            float(gl[3] + eps),  # left_gripper_right_finger_joint (lower)
        ]

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
        # Initialize state machine data, but start disabled by default.
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
        task_time_limits_list = [
            float(self.sm_time_approach),
            float(self.sm_time_engage),
            float(self.sm_time_grasp),
            float(self.sm_time_hold_grasp),
            float(self.sm_time_extract),
            999.0,
        ]
        self.sm_task_time_limits = wp.array(task_time_limits_list, dtype=float)

        # Per-arm mutable state
        self.sm_task_idx = wp.zeros(self.num_arms, dtype=int)
        self.sm_task_time_elapsed = wp.zeros(self.num_arms, dtype=float)

        # Snapshot of each arm's EE transform at the start of the current task.
        body_q_np = self.mujoco_state_0.body_q.numpy()
        init_tfs = []
        for arm_idx in range(self.num_arms):
            _, ee_link_idx = self.ee_configs[arm_idx]
            init_tfs.append(wp.transform(*body_q_np[ee_link_idx]))
        self.sm_task_init_body_q = wp.array(init_tfs, dtype=wp.transform)

        # State machine parameters
        # Specified in the frame of the capsule (target center-aligned grasps).
        # Aim for capsule COM; approach offsets stay on +/-Y with a small Z lift.
        approach_offset_y = float(self.sm_approach_offset_y)
        approach_offset_z = float(self.sm_approach_offset_z)

        # Grasp offset specified in the capsule's local frame and rotated into world by the kernel
        # via `wp.quat_rotate(capsule_quat(_prev), capsule_grasp_offset_from_com[arm_idx])`.
        #
        # Interpret "2/3 of capsule axis" as a point along the full capsule length
        # (including hemispheres): full_len ~ 2*(half_height + radius).
        # COM is at half-length, so offset from COM is:
        #   s = (fraction - 0.5) * full_len
        frac = float(self.sm_grasp_axis_fraction)
        full_len = 2.0 * (float(self.vbd_capsule_half_height) + float(self.vbd_capsule_radius))
        s = (frac - 0.5) * full_len
        self.capsule_grasp_offset_from_com = wp.array(
            [wp.vec3(0.0, 0.0, float(s)), wp.vec3(0.0, 0.0, float(s))],
            dtype=wp.vec3,
        )

        self.approach_offsets = wp.array(
            [
                wp.vec3(0.0, -approach_offset_y, approach_offset_z),  # Right arm -> Capsule A
                wp.vec3(0.0, approach_offset_y, approach_offset_z),  # Left arm  -> Capsule B
            ],
            dtype=wp.vec3,
        )

        self.extract_distance = 0.03  # Reduced from 0.05 to make extraction gentler
        # Keep unseat disabled by default; EXTRACT follows capsule-axis pull only.
        self.extract_unseat_lift = float(getattr(self, "extract_unseat_lift", 0.0))
        self.extract_unseat_fraction = float(getattr(self, "extract_unseat_fraction", 0.0))
        # Relax thresholds so auto-grasp can advance with VBD pose noise.
        self.pos_error_threshold = wp.array([0.003] * self.num_tasks, dtype=float)
        rot_error_threshold = [1.0 * wp.pi / 180.0] * self.num_tasks  # in radians
        self.rot_error_threshold = wp.array(rot_error_threshold, dtype=float)

        # Capsule body indices (right arm -> capsule A, left arm -> capsule B)
        capsule_a_idx = self.vbd_capsule_body_ids[0]
        capsule_b_idx = self.vbd_capsule_body_ids[1]
        self.sm_capsule_body_indices = wp.array([capsule_a_idx, capsule_b_idx], dtype=int)
        vbd_body_q_np = self.vbd_state_0.body_q.numpy()
        if self.verbose:
            print(
                f"Capsule A  position: {vbd_body_q_np[capsule_a_idx, :3]}, quaternion: {vbd_body_q_np[capsule_a_idx, 3:]}"
            )
            print(
                f"Capsule B  position: {vbd_body_q_np[capsule_b_idx, :3]}, quaternion: {vbd_body_q_np[capsule_b_idx, 3:]}"
            )

        # Extraction directions (precomputed in setup_scene_builder)
        self.sm_capsule_extract_dirs = wp.array(self.capsule_extract_dirs, dtype=wp.quat)

        # Per-arm world-frame extraction offsets:
        # - Lift up to reduce sticking on environment geometry
        # - Move toward y=0 to improve pinch alignment during pull
        lift_z = float(getattr(self, "sm_extract_lift_z", 0.03))
        center_y = float(getattr(self, "sm_extract_center_y", 0.05))
        self.sm_extract_offsets_world = wp.array(
            [
                wp.vec3(0.0, +center_y, lift_z),  # right arm (starts at -Y) -> move toward center (+Y)
                wp.vec3(0.0, -center_y, lift_z),  # left arm  (starts at +Y) -> move toward center (-Y)
            ],
            dtype=wp.vec3,
        )

        # Snapshot of the capsule transform at the start of the current task
        capsule_tf_a = wp.transform(*vbd_body_q_np[capsule_a_idx])
        capsule_tf_b = wp.transform(*vbd_body_q_np[capsule_b_idx])
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
        # Gripper closed values (flat array of 4 floats, used by the kernel)
        self.sm_gripper_closed_values = wp.array(self.gripper_closed_values_list, dtype=wp.float32)

        # EE body indices (for reading current EE position from body_q)
        self.sm_ee_body_indices = wp.array(
            [self.ee_configs[0][1], self.ee_configs[1][1]],
            dtype=int,
        )

        # Finger proxy body indices per arm (VBD body ids). Used for centering correction.
        # We map from MuJoCo finger body ids -> VBD proxy body ids using proxy_mj_body_ids/proxy_body_ids.
        body_key_to_id = (
            {k: i for i, k in enumerate(self.mujoco_model.body_key)} if hasattr(self.mujoco_model, "body_key") else {}
        )
        arm_finger_keys = [
            ("right_gripper_leftfinger", "right_gripper_rightfinger"),
            ("left_gripper_leftfinger", "left_gripper_rightfinger"),
        ]
        mujoco_to_vbd = {}
        if hasattr(self, "proxy_mj_body_ids") and hasattr(self, "proxy_body_ids"):
            for mujoco_body_id, vbd_body_id in zip(self.proxy_mj_body_ids, self.proxy_body_ids, strict=True):
                mujoco_to_vbd[int(mujoco_body_id)] = int(vbd_body_id)
        finger_proxy_ids = []
        for arm in range(self.num_arms):
            fk0, fk1 = arm_finger_keys[arm]
            mujoco_finger0_body_id = int(body_key_to_id.get(fk0, -1))
            mujoco_finger1_body_id = int(body_key_to_id.get(fk1, -1))
            vbd_f0 = int(mujoco_to_vbd.get(mujoco_finger0_body_id, -1))
            vbd_f1 = int(mujoco_to_vbd.get(mujoco_finger1_body_id, -1))
            finger_proxy_ids.append((vbd_f0, vbd_f1))
        self.sm_finger_proxy_body_indices = wp.array(finger_proxy_ids, dtype=int)

        # Centering correction parameters (helps ensure both fingers touch).
        # Disabled by default: finger proxy body origins are not necessarily at the fingertip, so
        # geometric centering based on body origins can be misleading. Re-enable only after switching
        # to a contact-point-based centering signal.
        self.gripper_centering_enable = True
        self.gripper_centering_k = 0.5  # fraction of measured offset to correct each frame
        self.gripper_centering_max_step = 0.01  # meters per frame (clamp for stability)

        # Kernel output arrays
        self.sm_ee_pos_target = wp.zeros(self.num_arms, dtype=wp.vec3)
        self.sm_ee_pos_interp = wp.zeros(self.num_arms, dtype=wp.vec3)
        self.sm_ee_rot_target = wp.zeros(self.num_arms, dtype=wp.vec4)
        self.sm_ee_rot_interp = wp.zeros(self.num_arms, dtype=wp.vec4)
        self.sm_gripper_target = wp.zeros(shape=(self.num_arms, 2), dtype=wp.float32)

        if self.verbose:
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
                    self.extract_unseat_lift,
                    self.extract_unseat_fraction,
                    self.sm_capsule_body_indices,
                    self.sm_capsule_extract_dirs,
                    self.sm_grasp_orientation_offset,
                    self.sm_gripper_open_values,
                    self.sm_gripper_closed_values,
                    self.sm_extract_offsets_world,
                    self.sm_task_init_body_q,
                    self.sm_task_capsule_body_q_prev,
                    self.mujoco_state_0.body_q,
                    self.vbd_state_0.body_q,
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

            # Optional centering correction: keep capsule centered between finger proxies.
            if bool(getattr(self, "gripper_centering_enable", True)) and hasattr(self, "sm_finger_proxy_body_indices"):
                wp.launch(
                    apply_gripper_centering_correction_kernel,
                    dim=self.num_arms,
                    inputs=[
                        self.sm_task_schedule,
                        self.sm_task_idx,
                        self.vbd_state_0.body_q,
                        self.sm_capsule_body_indices,
                        self.sm_finger_proxy_body_indices,
                        float(getattr(self, "gripper_centering_k", 0.0)),
                        float(getattr(self, "gripper_centering_max_step", 0.0)),
                    ],
                    outputs=[self.sm_ee_pos_interp],
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
                    self.sm_ee_pos_interp,
                    self.sm_ee_rot_interp,
                    self.mujoco_state_0.body_q,
                    self.vbd_state_0.body_q,
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
        if self.enable_robot:
            # Update IK targets and compute joint positions
            self.set_joint_targets()

        # Run physics simulation
        if self.graph_sim:
            wp.capture_launch(self.graph_sim)
        else:
            self.simulate()

        if self.graph_sim:
            # Keep input buffers in sync for the next graph launch.
            mujoco_step_count_for_swap = (
                max(int(self.mujoco_substeps), int(self.vbd_substeps))
                if self.exchange_forces_every_substep
                else int(self.mujoco_substeps)
            )
            if self.enable_robot and (mujoco_step_count_for_swap % 2 == 1):
                self.mujoco_state_0.assign(self.mujoco_state_1)
            # VBD copies output into `self.vbd_state_0` inside `simulate()` when graph mode is enabled.

        self.sim_time += self.frame_dt
        self.frame_count += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        if self.enable_robot:
            if self.viewer_primary_model == "mujoco":
                # MuJoCo is primary - render MuJoCo state via viewer
                self.viewer.log_state(self.mujoco_state_0)
                self.viewer.log_contacts(self.mujoco_contacts, self.mujoco_state_0)
                if self.mujoco_collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
                    self.viewer.log_hydro_contact_surface(
                        self.collision_pipeline.get_hydro_contact_surface(), penetrating_only=True
                    )

                # Manually render VBD capsules, table, and meshes
                self._render_vbd_objects()
                # Render proxy bodies with distinct color
                self._render_proxy_bodies()
            else:
                # VBD is primary - render VBD state via viewer
                self.viewer.log_state(self.vbd_state_0)
                self.viewer.log_contacts(self.vbd_contacts, self.vbd_state_0)

                # Render proxy bodies with distinct color to highlight them
                self._render_proxy_bodies()
        else:
            # Robot disabled - only render VBD
            self.viewer.log_state(self.vbd_state_0)
            self.viewer.log_contacts(self.vbd_contacts, self.vbd_state_0)

        self.viewer.end_frame()

    def _render_vbd_objects(self):
        """Render VBD capsules, table, and mesh connectors as separate shapes."""
        vbd_body_q_np = self.vbd_state_0.body_q.numpy()
        capsule_xforms = []
        for body_idx in self.vbd_capsule_body_ids:
            bq = vbd_body_q_np[body_idx]
            capsule_xforms.append(
                wp.transform(
                    wp.vec3(bq[0], bq[1], bq[2]),
                    wp.quat(bq[3], bq[4], bq[5], bq[6]),
                )
            )

        if capsule_xforms:
            self.viewer.log_shapes(
                "/vbd_capsules",
                GeoType.CAPSULE,
                (self.vbd_capsule_radius, self.vbd_capsule_half_height),
                wp.array(capsule_xforms, dtype=wp.transform),
                colors=wp.array([wp.vec3(0.2, 0.6, 0.9)], dtype=wp.vec3),
            )

        # Render VBD table
        self.viewer.log_shapes(
            "/vbd_table",
            GeoType.BOX,
            (self.table_half_size[0], self.table_half_size[1], self.table_half_size[2]),
            wp.array([self.vbd_table_xform], dtype=wp.transform),
            colors=wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3),
        )

        # Render VBD mesh connectors
        self.viewer.log_shapes(
            "/vbd_mesh_connectors",
            GeoType.MESH,
            1.0,
            wp.array(self.vbd_mesh_xforms, dtype=wp.transform),
            colors=wp.array([wp.vec3(0.6, 0.6, 0.6)], dtype=wp.vec3),
            geo_src=self.vbd_mesh,
        )

    def _render_proxy_bodies(self):
        """Render VBD proxy bodies with distinct color to visualize two-way coupling."""
        if not hasattr(self, "proxy_body_ids"):
            return

        if len(self.proxy_body_ids) == 0:
            return

        vbd_body_q_np = self.vbd_state_0.body_q.numpy()
        shape_body_np = self.vbd_model.shape_body.numpy()
        shape_type_np = self.vbd_model.shape_type.numpy()
        shape_scale_np = self.vbd_model.shape_scale.numpy()

        # Debug: Count how many proxy shapes we find
        total_proxy_shapes = 0

        # Collect proxy body shapes
        for proxy_vbd_id in self.proxy_body_ids:
            # Find all shapes attached to this proxy body
            proxy_shapes = []
            for shape_id in range(self.vbd_model.shape_count):
                if shape_body_np[shape_id] == proxy_vbd_id:
                    proxy_shapes.append(shape_id)

            if not proxy_shapes:
                continue

            # Get proxy transform
            bq = vbd_body_q_np[proxy_vbd_id]
            proxy_xform = wp.transform(
                wp.vec3(bq[0], bq[1], bq[2]),
                wp.quat(bq[3], bq[4], bq[5], bq[6]),
            )

            # Render each shape with distinct color (bright orange, slightly enlarged)
            for shape_id in proxy_shapes:
                total_proxy_shapes += 1
                geo_type = shape_type_np[shape_id]

                # Scale factor to make proxies slightly larger/more visible
                scale_factor = 1.05

                if geo_type == GeoType.SPHERE:
                    radius = float(shape_scale_np[shape_id][0]) * scale_factor
                    self.viewer.log_shapes(
                        f"/proxy_body_{proxy_vbd_id}_shape_{shape_id}",
                        GeoType.SPHERE,
                        radius,
                        wp.array([proxy_xform], dtype=wp.transform),
                        colors=wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3),  # Bright orange
                    )
                elif geo_type == GeoType.BOX:
                    hx = float(shape_scale_np[shape_id][0]) * scale_factor
                    hy = float(shape_scale_np[shape_id][1]) * scale_factor
                    hz = float(shape_scale_np[shape_id][2]) * scale_factor
                    self.viewer.log_shapes(
                        f"/proxy_body_{proxy_vbd_id}_shape_{shape_id}",
                        GeoType.BOX,
                        (hx, hy, hz),
                        wp.array([proxy_xform], dtype=wp.transform),
                        colors=wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3),  # Bright orange
                    )
                elif geo_type == GeoType.CAPSULE:
                    radius = float(shape_scale_np[shape_id][0]) * scale_factor
                    half_height = float(shape_scale_np[shape_id][1]) * scale_factor
                    self.viewer.log_shapes(
                        f"/proxy_body_{proxy_vbd_id}_shape_{shape_id}",
                        GeoType.CAPSULE,
                        (radius, half_height),
                        wp.array([proxy_xform], dtype=wp.transform),
                        colors=wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3),  # Bright orange
                    )
                elif geo_type == GeoType.MESH:
                    # Render mesh with orange color
                    geo_src = self.vbd_model.shape_source[shape_id]
                    mesh = None
                    if isinstance(geo_src, newton.Mesh):
                        mesh = geo_src
                    elif isinstance(geo_src, Integral):
                        mesh_id = int(geo_src)
                        if 0 <= mesh_id < len(self.vbd_model.geo_meshes):
                            mesh = self.vbd_model.geo_meshes[mesh_id]

                    if mesh is not None:
                        self.viewer.log_shapes(
                            f"/proxy_body_{proxy_vbd_id}_shape_{shape_id}",
                            GeoType.MESH,
                            scale_factor,
                            wp.array([proxy_xform], dtype=wp.transform),
                            colors=wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3),  # Bright orange
                            geo_src=mesh,
                        )

    def _render_mujoco_robot(self):
        """Render MuJoCo robot bodies as separate shapes when VBD is primary."""
        # Get MuJoCo body transforms
        body_q_np = self.mujoco_state_0.body_q.numpy()

        # Collect transforms for all non-ground bodies
        robot_xforms = []
        for body_idx in range(self.mujoco_model.body_count):
            # Skip ground body (body -1 or body 0 depending on convention)
            if body_idx == 0:
                continue
            bq = body_q_np[body_idx]
            robot_xforms.append(
                wp.transform(
                    wp.vec3(bq[0], bq[1], bq[2]),
                    wp.quat(bq[3], bq[4], bq[5], bq[6]),
                )
            )

        # Render robot bodies with semi-transparent color to distinguish from VBD objects
        if robot_xforms:
            # Note: This renders all bodies as points for now
            # For full robot visualization, we'd need to iterate through shapes
            # and render each with its proper geometry
            positions = wp.array([wp.transform_get_translation(xf) for xf in robot_xforms], dtype=wp.vec3)
            self.viewer.log_points(
                "/mujoco_robot_bodies",
                points=positions,
                radii=wp.full(len(robot_xforms), 0.02, dtype=float),
                colors=wp.array([wp.vec3(0.9, 0.3, 0.3)], dtype=wp.vec3),
            )

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
            contact_margin=0.005,
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
            return newton.examples.create_collision_pipeline(self.mujoco_model, args)
        elif collision_mode == CollisionMode.NEWTON_SDF:
            # Newton with SDF for mesh collision
            return newton.CollisionPipelineUnified.from_model(
                self.mujoco_model,
                reduce_contacts=True,
                broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
            )
        elif collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            # Newton with hydroelastic contacts
            from newton.geometry import SDFHydroelasticConfig  # noqa: PLC0415

            return newton.CollisionPipelineUnified.from_model(
                self.mujoco_model,
                reduce_contacts=True,
                broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
                sdf_hydroelastic_config=SDFHydroelasticConfig(output_contact_surface=True),
            )
        else:
            raise ValueError(f"Unknown collision mode: {collision_mode}")

    def setup_robot_builder(self):
        robot = newton.ModelBuilder()
        robot.default_shape_cfg = self.mujoco_default_shape_cfg

        robot_file = ROBOT_PATH / "robot_edited.urdf"
        if not robot_file.is_file():
            raise FileNotFoundError(
                f"Robot URDF not found: {robot_file}. "
                f"Set NEWTON_EXAMPLES_ASSETS_PATH to override (currently ASSETS_ROOT={ASSETS_ROOT})."
            )

        robot.add_urdf(
            str(robot_file),
            xform=wp.transform(wp.vec3(0, 0, 0.00)),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

        # Discover gripper DOFs by joint key (robust to URDF changes).
        # ModelBuilder.joint_key is per-joint; convert to DOF index using joint_qd_start.
        gripper_joint_keys = [
            "right_gripper_left_finger_joint",
            "right_gripper_right_finger_joint",
            "left_gripper_left_finger_joint",
            "left_gripper_right_finger_joint",
        ]
        dofs: list[int] = []
        for key in gripper_joint_keys:
            try:
                j = robot.joint_key.index(key)
            except ValueError:
                dofs.append(-1)
                continue
            dof_start = int(robot.joint_qd_start[j])
            # If the joint has multiple DOFs, take the first.
            dofs.append(dof_start)
        # Filter invalid entries.
        self.gripper_joint_dofs = [d for d in dofs if d >= 0]

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
            robot.joint_target_ke[dof] = float(self.gripper_joint_target_ke)
            robot.joint_target_kd[dof] = float(self.gripper_joint_target_kd)
            robot.joint_effort_limit[dof] = float(self.gripper_joint_effort_limit)
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
        scene.default_shape_cfg = self.mujoco_default_shape_cfg
        scene.add_builder(robot)

        return scene

    def _compute_hose_layout(self):
        """Return shared table-top center and symmetric hose lane y positions."""
        table_top_center = [self.table_pos[0], self.table_pos[1], self.table_pos[2] + self.table_half_size[2]]
        right_hose_y = -float(self.hose_y_offset)
        left_hose_y = -right_hose_y
        return table_top_center, right_hose_y, left_hose_y

    def _compute_capsule_specs(self):
        """Compute capsule transforms and parameters without changing geometry."""
        table_top_center, right_hose_y, left_hose_y = self._compute_hose_layout()

        capsule_radius = float(self.capsule_radius)
        capsule_height = float(self.capsule_cylinder_height)
        tilt_angle_rad = float(self.capsule_tilt_angle_deg) * wp.pi / 180.0

        # Spawn offset along the capsule's local +Z axis in world frame (same direction used for extraction).
        # This helps slightly "unseat" the capsule from the static STL cradle at t=0.
        spawn_axis_offset = float(self.capsule_spawn_axis_offset)

        capsule_length_offset = float(self.capsule_length_offset)
        capsule_total_length = capsule_height + 2.0 * capsule_radius + capsule_length_offset
        base_x = (
            self.table_pos[0] - 0.5 * capsule_total_length * wp.sin(tilt_angle_rad) + float(self.capsule_spawn_x_bias)
        )
        base_z = table_top_center[2] + 0.5 * capsule_total_length * wp.cos(tilt_angle_rad)

        # Keep both capsules with the same orientation (mirrors `example_cable_robot_ik.py`).
        capsule_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -tilt_angle_rad)
        capsule_axis = wp.quat_rotate(capsule_quat, wp.vec3(0.0, 0.0, 1.0))

        # Capsule A
        pos_a = wp.vec3(base_x, right_hose_y, base_z) + capsule_axis * spawn_axis_offset
        xform_a = wp.transform(pos_a, capsule_quat)

        # Capsule B
        pos_b = wp.vec3(base_x, left_hose_y, base_z) + capsule_axis * spawn_axis_offset
        xform_b = wp.transform(pos_b, capsule_quat)

        return {
            "capsule_radius": capsule_radius,
            "capsule_half_height": 0.5 * capsule_height,
            "capsules": [("test_capsule_a", xform_a), ("test_capsule_b", xform_b)],
            # Keep metadata consistent with the actual spawned transforms above.
            "tilt_angles": [-tilt_angle_rad, -tilt_angle_rad],
            "extract_dirs": [capsule_quat, capsule_quat],
        }

    def _setup_vbd_world_and_coupling(self, args):
        """Create VBD world state and optional proxy plumbing for two-way coupling."""

        vbd_builder = newton.ModelBuilder()

        vbd_builder.default_shape_cfg.ke = float(self.vbd_default_contact_ke)
        vbd_builder.default_shape_cfg.kd = float(self.vbd_default_contact_kd)

        vbd_builder.default_shape_cfg.contact_margin = float(self.vbd_default_contact_margin)
        vbd_builder.default_shape_cfg.mu = float(self.vbd_proxy_mu)

        # For mesh collisions: disable SDF volume generation unless explicitly requested.
        if not self.vbd_mesh_use_sdf:
            vbd_builder.default_shape_cfg.sdf_max_resolution = None
            vbd_builder.default_shape_cfg.sdf_target_voxel_size = None
            vbd_builder.default_shape_cfg.is_hydroelastic = False
        else:
            # Enable SDF volumes for VBD mesh shapes (used by unified mesh contact path).
            vbd_builder.default_shape_cfg.sdf_max_resolution = int(self.vbd_mesh_sdf_max_resolution)

        # Create dynamic capsules in the VBD model from the shared capsule spec helper.
        capsule_specs = self._compute_capsule_specs()
        self.capsule_body_keys = [name for name, _xform in capsule_specs["capsules"]]
        self.capsule_tilt_angles = capsule_specs["tilt_angles"]
        self.capsule_extract_dirs = capsule_specs["extract_dirs"]

        self.vbd_capsule_radius = float(capsule_specs["capsule_radius"])
        self.vbd_capsule_half_height = float(capsule_specs["capsule_half_height"])
        self.vbd_capsule_body_ids = []

        capsule_cfg = vbd_builder.default_shape_cfg.copy()
        capsule_cfg.mu = float(self.vbd_capsule_mu)
        capsule_cfg.thickness = float(self.vbd_capsule_thickness)
        capsule_cfg.contact_margin = float(self.vbd_capsule_contact_margin)
        capsule_mass = float(self.vbd_capsule_mass)

        for key, xform in capsule_specs["capsules"]:
            body_id = vbd_builder.add_body(xform=xform, key=key, mass=capsule_mass)
            vbd_builder.add_shape_capsule(
                body=body_id,
                radius=capsule_specs["capsule_radius"],
                half_height=capsule_specs["capsule_half_height"],
                cfg=capsule_cfg,
            )
            self.vbd_capsule_body_ids.append(body_id)

        # Add static scene geometry to VBD (table + STL connectors + ground plane).
        if not HOSE_CONNECTOR_PATH.exists():
            raise FileNotFoundError(f"Missing STL asset: {HOSE_CONNECTOR_PATH}")

        mesh_vertices, mesh_indices = _load_stl_as_tri_mesh(HOSE_CONNECTOR_PATH)

        scale_factor = 0.001
        mesh_vertices_centered = mesh_vertices * scale_factor
        mesh = newton.Mesh(mesh_vertices_centered, mesh_indices, compute_inertia=True, is_solid=True)
        self.vbd_mesh = mesh

        table_top_center, right_hose_y, left_hose_y = self._compute_hose_layout()

        min_z = float(np.min(mesh_vertices_centered[:, 2])) if mesh_vertices_centered.size else 0.0
        mesh_z = float(-min_z)
        mesh_pos_a = wp.vec3f(wp.float32(0.0), wp.float32(float(right_hose_y)), wp.float32(mesh_z)) + wp.vec3f(
            *table_top_center
        )
        mesh_pos_b = wp.vec3f(wp.float32(0.0), wp.float32(float(left_hose_y)), wp.float32(mesh_z)) + wp.vec3f(
            *table_top_center
        )

        q_a = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.half_pi)
        q_b = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.half_pi)
        self.vbd_mesh_xforms = [
            wp.transform(p=mesh_pos_a, q=q_a),
            wp.transform(p=mesh_pos_b, q=q_b),
        ]

        # Use the VBD-tuned contact parameters for static environment shapes as well.
        vbd_ground_cfg = vbd_builder.default_shape_cfg.copy()
        vbd_ground_cfg.mu = float(self.vbd_env_mu)

        vbd_ground_cfg.thickness = float(self.vbd_env_thickness)
        vbd_ground_cfg.contact_margin = float(self.vbd_env_contact_margin)

        if not self.vbd_mesh_use_sdf:
            vbd_ground_cfg.sdf_max_resolution = None
            vbd_ground_cfg.sdf_target_voxel_size = None
            vbd_ground_cfg.is_hydroelastic = False

        vbd_builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            xform=wp.transform(p=mesh_pos_a, q=q_a),
            cfg=vbd_ground_cfg,
            key="rby1_hose_connectorv3_a",
        )
        vbd_builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            xform=wp.transform(p=mesh_pos_b, q=q_b),
            cfg=vbd_ground_cfg,
            key="rby1_hose_connectorv3_b",
        )

        vbd_builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(self.table_pos)),
            hx=self.table_half_size[0],
            hy=self.table_half_size[1],
            hz=self.table_half_size[2],
            cfg=vbd_ground_cfg,
        )
        self.vbd_table_xform = wp.transform(wp.vec3(self.table_pos))

        vbd_builder.add_ground_plane(cfg=vbd_ground_cfg)

        # Create proxy bodies only when two-way coupling is enabled.
        if self.enable_robot and self.enable_two_way_coupling:
            self._create_proxy_bodies(vbd_builder)

        vbd_builder.color()

        self.vbd_model = vbd_builder.finalize()
        self.vbd_solver = newton.solvers.SolverVBD(
            self.vbd_model,
            iterations=self.vbd_iterations,
            friction_epsilon=float(self.vbd_solver_friction_epsilon),
            rigid_avbd_beta=float(self.vbd_rigid_avbd_beta),
            rigid_contact_k_start=float(self.vbd_rigid_contact_k_start),
            rigid_body_contact_buffer_size=int(self.vbd_rigid_contact_buffer_size),
        )

        self.vbd_state_0 = self.vbd_model.state()
        self.vbd_state_1 = self.vbd_model.state()
        self.vbd_control = self.vbd_model.control()

        self.vbd_collision_pipeline = newton.examples.create_collision_pipeline(
            self.vbd_model,
            args,
            collision_pipeline_type=self.vbd_collision_pipeline_type,
        )
        self.vbd_contacts = self.vbd_model.collide(self.vbd_state_0, collision_pipeline=self.vbd_collision_pipeline)

        # Initialize coupling buffers only when two-way coupling is enabled.
        if self.enable_robot and self.enable_two_way_coupling:
            self._init_coupling_buffers()

        if self.verbose:
            print(f"Created VBD world with {self.vbd_model.body_count} bodies")
            print(f"  VBD capsules: {len(self.vbd_capsule_body_ids)}")
            if self.enable_robot and self.enable_two_way_coupling:
                print(f"  Proxy bodies: {len(self.proxy_body_ids)}")

    def _create_proxy_bodies(self, vbd_builder):
        """Create proxy bodies in VBD that mirror MuJoCo robot bodies.

        Uses a duplication strategy: create rigid proxy bodies in VBD.
        We create rigid bodies in VBD with the same mass, inertia, and shapes as MuJoCo.
        """
        if self.verbose:
            print("Creating proxy bodies for two-way coupling...")

        # Mapping from MuJoCo body ID to VBD proxy body ID
        # -1 means no proxy (e.g., ground body)
        self.mj_to_vbd_body_map = {}
        self.proxy_body_ids = []  # List of VBD proxy body IDs
        self.proxy_mj_body_ids = []  # List of corresponding MuJoCo body IDs

        # Convert arrays to numpy for CPU access
        body_inv_mass_np = self.mujoco_model.body_inv_mass.numpy()
        body_q_np = self.mujoco_state_0.body_q.numpy()
        shape_body_np = self.mujoco_model.shape_body.numpy()
        shape_type_np = self.mujoco_model.shape_type.numpy()
        shape_scale_np = self.mujoco_model.shape_scale.numpy()
        shape_transform_np = self.mujoco_model.shape_transform.numpy()

        # Only create proxies for gripper finger bodies
        gripper_finger_keys = {
            "right_gripper_leftfinger",
            "right_gripper_rightfinger",
            "left_gripper_leftfinger",
            "left_gripper_rightfinger",
        }

        # Track per-finger VBD shape ids so we can filter left/right finger self-collisions.
        # This is important when `vbd_proxy_thickness` is non-zero: closed fingers can otherwise
        # interpenetrate and generate large proxy-proxy forces unrelated to capsule grasping.
        proxy_finger_shape_ids: dict[str, list[int]] = {}

        if self.verbose:
            print(f"  Looking for gripper fingers in {self.mujoco_model.body_count} bodies...")

        # Iterate through MuJoCo bodies and create proxies
        for mj_body_id in range(self.mujoco_model.body_count):
            # Skip ground body (body 0)
            if mj_body_id == 0:
                self.mj_to_vbd_body_map[mj_body_id] = -1
                continue

            # Only create proxies for gripper fingers
            body_key = self.mujoco_model.body_key[mj_body_id] if mj_body_id < len(self.mujoco_model.body_key) else ""
            if body_key not in gripper_finger_keys:
                self.mj_to_vbd_body_map[mj_body_id] = -1
                continue

            if self.verbose:
                print(f"  Found gripper finger: {body_key} (body {mj_body_id})")

            # Get body properties from MuJoCo model
            body_inv_mass = body_inv_mass_np[mj_body_id]
            body_mass = float(1.0 / body_inv_mass) if body_inv_mass > 0 else 0.0

            # Skip bodies with zero or infinite mass
            if body_mass <= 0.0 or body_mass > 1e9:
                self.mj_to_vbd_body_map[mj_body_id] = -1
                continue

            # Apply mass scaling
            proxy_mass_scale = float(getattr(self, "proxy_mass_scale", 1.0))
            scaled_body_mass = body_mass * proxy_mass_scale

            # Get initial transform
            body_q = body_q_np[mj_body_id]
            initial_xform = wp.transform(
                wp.vec3(body_q[0], body_q[1], body_q[2]), wp.quat(body_q[3], body_q[4], body_q[5], body_q[6])
            )

            # Create proxy body in VBD with scaled mass
            proxy_body_id = vbd_builder.add_body(
                xform=initial_xform,
                mass=scaled_body_mass,
                key=f"proxy_{self.mujoco_model.body_key[mj_body_id] if mj_body_id < len(self.mujoco_model.body_key) else mj_body_id}",
            )

            # Copy shapes from base body to proxy body.
            # Important: use the VBD-tuned contact parameters (ke/kd/mu) rather than the MuJoCo-side ones,
            # otherwise proxy contact in the VBD solver can become excessively stiff.
            proxy_shape_cfg = vbd_builder.default_shape_cfg.copy()
            proxy_shape_cfg.mu = float(self.vbd_proxy_mu)
            proxy_shape_cfg.thickness = float(self.vbd_proxy_thickness)
            shape_count = 0
            shape_ids: list[int] = []

            # Find all shapes attached to this body
            for shape_idx in range(len(shape_body_np)):
                if shape_body_np[shape_idx] != mj_body_id:
                    continue

                shape_type = shape_type_np[shape_idx]
                shape_scale = shape_scale_np[shape_idx]
                shape_xform_data = shape_transform_np[shape_idx]

                # Extract shape transform
                pos = wp.vec3(shape_xform_data[0], shape_xform_data[1], shape_xform_data[2])
                rot = wp.quat(shape_xform_data[3], shape_xform_data[4], shape_xform_data[5], shape_xform_data[6])
                shape_xform = wp.transform(p=pos, q=rot)

                if shape_type == GeoType.SPHERE:
                    radius = float(shape_scale[0])
                    sid = vbd_builder.add_shape_sphere(
                        body=proxy_body_id, radius=radius, pos=pos, rot=rot, cfg=proxy_shape_cfg
                    )
                    shape_ids.append(int(sid))
                    if self.verbose:
                        print(f"    Added SPHERE (r={radius:.3f})")
                    shape_count += 1

                elif shape_type == GeoType.BOX:
                    hx, hy, hz = float(shape_scale[0]), float(shape_scale[1]), float(shape_scale[2])
                    sid = vbd_builder.add_shape_box(
                        body=proxy_body_id, hx=hx, hy=hy, hz=hz, pos=pos, rot=rot, cfg=proxy_shape_cfg
                    )
                    shape_ids.append(int(sid))
                    if self.verbose:
                        print(f"    Added BOX ({hx:.3f}, {hy:.3f}, {hz:.3f})")
                    shape_count += 1

                elif shape_type == GeoType.CAPSULE:
                    radius = float(shape_scale[0])
                    half_height = float(shape_scale[1])
                    sid = vbd_builder.add_shape_capsule(
                        body=proxy_body_id,
                        radius=radius,
                        half_height=half_height,
                        pos=pos,
                        rot=rot,
                        cfg=proxy_shape_cfg,
                    )
                    shape_ids.append(int(sid))
                    if self.verbose:
                        print(f"    Added CAPSULE (r={radius:.3f}, h={half_height:.3f})")
                    shape_count += 1

                elif shape_type == GeoType.MESH:
                    # For mesh, we need the shape source (mesh data reference)
                    shape_source = self.mujoco_model.shape_source[shape_idx]
                    # Mesh uses xform instead of pos/rot
                    sid = vbd_builder.add_shape_mesh(
                        body=proxy_body_id, mesh=shape_source, xform=shape_xform, cfg=proxy_shape_cfg
                    )
                    shape_ids.append(int(sid))
                    if self.verbose:
                        print(f"    Added MESH (source={shape_source})")
                    shape_count += 1

                else:
                    if self.verbose:
                        print(f"    WARNING: Unsupported shape type {shape_type}, skipping")

            if shape_count == 0:
                if self.verbose:
                    print(f"    WARNING: No shapes found for body {mj_body_id}, adding fallback box")
                # Add a small fallback box if no shapes were found
                sid = vbd_builder.add_shape_box(body=proxy_body_id, hx=0.02, hy=0.01, hz=0.04, cfg=proxy_shape_cfg)
                shape_ids.append(int(sid))

            proxy_finger_shape_ids[str(body_key)] = shape_ids

            # Store mapping
            self.mj_to_vbd_body_map[mj_body_id] = proxy_body_id
            self.proxy_body_ids.append(proxy_body_id)
            self.proxy_mj_body_ids.append(mj_body_id)

        # Filter self-collisions between the two fingers of each gripper.
        for a, b in [
            ("right_gripper_leftfinger", "right_gripper_rightfinger"),
            ("left_gripper_leftfinger", "left_gripper_rightfinger"),
        ]:
            sa = proxy_finger_shape_ids.get(a, [])
            sb = proxy_finger_shape_ids.get(b, [])
            if not sa or not sb:
                continue
            for s1 in sa:
                for s2 in sb:
                    vbd_builder.add_shape_collision_filter_pair(int(s1), int(s2))

        if self.verbose:
            print(f"  Created {len(self.proxy_body_ids)} proxy bodies")

    def _init_coupling_buffers(self):
        """Initialize buffers for two-way coupling force exchange."""

        mujoco_device = wp.get_device(self.mujoco_model.device)
        vbd_device = wp.get_device(self.vbd_model.device)
        if str(mujoco_device) != str(vbd_device):
            raise RuntimeError(
                "Two-way coupling requires MuJoCo and VBD models on the same device. "
                f"Got mujoco_device={mujoco_device}, vbd_device={vbd_device}"
            )
        device = mujoco_device

        # Convert MuJoCo->VBD body map to a dense array for GPU kernels.
        mj_to_vbd_map_list = [-1] * int(self.mujoco_model.body_count)
        for mj_id, vbd_id in self.mj_to_vbd_body_map.items():
            mj_to_vbd_map_list[mj_id] = vbd_id
        self.mj_to_vbd_body_map_array = wp.array(mj_to_vbd_map_list, dtype=int, device=device)

        # Convert proxy ID lists to GPU arrays for kernels.
        self.proxy_vbd_body_ids_array = wp.array(self.proxy_body_ids, dtype=int, device=device)
        self.proxy_mj_body_ids_array = wp.array(self.proxy_mj_body_ids, dtype=int, device=device)
        # Keep proxies MuJoCo-driven: skip VBD rigid forward integration for proxy bodies.
        if hasattr(self.vbd_solver, "set_rigid_forward_skip_body_ids"):
            self.vbd_solver.set_rigid_forward_skip_body_ids(self.proxy_body_ids)

        # Coupling wrench/force buffers on MuJoCo bodies.
        body_force_template = self.mujoco_state_0.body_f
        self.proxy_forces = wp.zeros_like(body_force_template)
        self.proxy_forces_step = wp.zeros_like(body_force_template)
        self.coupling_wrenches_applied = wp.zeros_like(body_force_template)
        self.proxy_forces_prev = wp.zeros_like(body_force_template)

        # Per-proxy previous-pose buffer (used when coupling_proxy_prevpose_velocity_source="snapshot")
        self.proxy_prev_poses = wp.zeros(len(self.proxy_body_ids), dtype=wp.transform, device=device)

        if self.verbose:
            print("  Initialized coupling buffers")

    def _start_auto_mode(self):
        """Begin the automated grasping sequence for both arms."""
        # Reset task indices to start of schedule (APPROACH)
        self.sm_task_idx.zero_()
        self.sm_task_time_elapsed.zero_()

        # Snapshot current EE transforms as entry points for interpolation
        body_q_np = self.mujoco_state_0.body_q.numpy()
        init_tfs = []
        for arm_idx in range(self.num_arms):
            _, ee_link_idx = self.ee_configs[arm_idx]
            init_tfs.append(wp.transform(*body_q_np[ee_link_idx]))
        wp.copy(self.sm_task_init_body_q, wp.array(init_tfs, dtype=wp.transform))

        # Refresh capsule snapshot from current VBD state for auto-grasp.
        vbd_body_q_np = self.vbd_state_0.body_q.numpy()
        capsule_indices = self.sm_capsule_body_indices.numpy()
        capsule_tfs = [wp.transform(*vbd_body_q_np[idx]) for idx in capsule_indices]
        wp.copy(self.sm_task_capsule_body_q_prev, wp.array(capsule_tfs, dtype=wp.transform))

        # Reset gripper targets to fully open
        wp.copy(self.gripper_targets, self.sm_gripper_open_values)

        if self.verbose:
            print("Auto-grasp mode STARTED")

    def _stop_auto_mode(self):
        """Cancel the automated sequence and return to manual GUI control."""
        # Reset gripper targets to fully open
        wp.copy(self.gripper_targets, self.sm_gripper_open_values)
        if self.verbose:
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
            ee_pos_target_np = self.sm_ee_pos_interp.numpy()
            ee_rot_target_np = self.sm_ee_rot_interp.numpy()
            body_q_np = self.mujoco_state_0.body_q.numpy()
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
