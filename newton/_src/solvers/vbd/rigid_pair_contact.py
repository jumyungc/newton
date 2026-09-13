# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic constraint-block contacts for free rigid bodies in VBD."""

import warp as wp

from newton._src.sim.contacts import contact_surface_point, contact_surface_separation
from newton._src.solvers.vbd.rigid_vbd_kernels import orthonormal_basis

wp.set_module_options({"enable_backward": False})

_COLOR_COUNT = 30
_DIRECT_COLOR_COUNT = 8
_PARALLEL_BUILD_ROUNDS = 10
_FUSED_THREADS = 256
_CAPTURED_DIRECT_COLOR_MIN_BODIES_PER_SM = 1
_EAGER_DIRECT_COLOR_MIN_BODIES_PER_SM = 32
_EPS = wp.constant(1.0e-12)
_TANGENT_EPS = wp.constant(1.0e-6)


def select_rigid_pair_block_dim(launch_dim: int, device, target_blocks: int | None = None) -> int:
    """Choose a power-of-two CUDA block width for pair-contact kernels."""
    if not device.is_cuda:
        return 256
    if target_blocks is None:
        target_blocks = max(1, (device.sm_count + 1) // 2)
    block_dim = 4
    while block_dim < 256:
        candidate = 2 * block_dim
        if (launch_dim + candidate - 1) // candidate < target_blocks:
            break
        block_dim = candidate
    return block_dim


@wp.func_native(
    """
    #if defined(__CUDA_ARCH__)
    __syncthreads();
    #endif
    """
)
def _pair_sync_block(): ...


@wp.kernel
def build_and_color_pair_groups_serial(
    body_count: int,
    color_count: int,
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_uses_pair_contacts: wp.array[wp.int32],
    body_color_mask: wp.array[wp.int32],
    group_start: wp.array[wp.int32],
    group_end: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    group_color: wp.array[wp.int32],
    group_list: wp.array[wp.int32],
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    color_cursors: wp.array[wp.int32],
    group_count_out: wp.array[wp.int32],
    overflow_count_out: wp.array[wp.int32],
):
    """Build contiguous body-pair runs and reverse-greedy exact edge colors."""
    if wp.tid() != 0:
        return

    for group in range(group_color.shape[0]):
        group_color[group] = -2
    for body in range(body_count):
        body_color_mask[body] = 0
    for color in range(color_count):
        color_counts[color] = 0
        color_offsets[color] = 0
        color_cursors[color] = 0
    color_offsets[color_count] = 0

    active_contacts = contact_count[0]
    group_count = wp.int32(0)
    contact = wp.int32(0)
    while contact < active_contacts:
        shape0 = contact_shape0[contact]
        shape1 = contact_shape1[contact]
        body0 = shape_body[shape0] if shape0 >= 0 else -1
        body1 = shape_body[shape1] if shape1 >= 0 else -1

        end = contact + 1
        while end < active_contacts:
            next_shape0 = contact_shape0[end]
            next_shape1 = contact_shape1[end]
            next_body0 = shape_body[next_shape0] if next_shape0 >= 0 else -1
            next_body1 = shape_body[next_shape1] if next_shape1 >= 0 else -1
            if next_body0 != body0 or next_body1 != body1:
                break
            end += 1

        dynamic0 = body0 >= 0 and body_inv_mass[body0] > 0.0
        dynamic1 = body1 >= 0 and body_inv_mass[body1] > 0.0
        pair0 = body0 < 0 or body_uses_pair_contacts[body0] != 0
        pair1 = body1 < 0 or body_uses_pair_contacts[body1] != 0
        if body0 != body1 and (dynamic0 or dynamic1) and pair0 and pair1:
            group_start[group_count] = contact
            group_end[group_count] = end
            group_body0[group_count] = body0
            group_body1[group_count] = body1
            group_color[group_count] = -1
            group_count += 1
        contact = end

    overflow_count = wp.int32(0)
    group = group_count - 1
    while group >= 0:
        body0 = group_body0[group]
        body1 = group_body1[group]
        dynamic0 = body0 >= 0 and body_inv_mass[body0] > 0.0
        dynamic1 = body1 >= 0 and body_inv_mass[body1] > 0.0
        used_mask = wp.int32(0)
        if dynamic0:
            used_mask |= body_color_mask[body0]
        if dynamic1:
            used_mask |= body_color_mask[body1]

        selected = wp.int32(-1)
        color = wp.int32(0)
        while color < color_count and selected < 0:
            bit = wp.int32(1) << color
            if (used_mask & bit) == 0:
                selected = color
            color += 1

        group_color[group] = selected
        if selected >= 0:
            bit = wp.int32(1) << selected
            if dynamic0:
                body_color_mask[body0] |= bit
            if dynamic1:
                body_color_mask[body1] |= bit
            color_counts[selected] += 1
        else:
            overflow_count += 1
        group -= 1

    offset = wp.int32(0)
    for color in range(color_count):
        color_offsets[color] = offset
        color_cursors[color] = offset
        offset += color_counts[color]
    color_offsets[color_count] = offset

    for group_index in range(group_count):
        color = group_color[group_index]
        if color >= 0:
            slot = color_cursors[color]
            group_list[slot] = group_index
            color_cursors[color] = slot + 1

    group_count_out[0] = group_count
    overflow_count_out[0] = overflow_count


@wp.kernel
def mark_pair_groups_parallel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_uses_pair_contacts: wp.array[wp.int32],
    group_start: wp.array[wp.int32],
    group_end: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    group_color: wp.array[wp.int32],
    group_count_out: wp.array[wp.int32],
):
    """Mark contiguous body-pair leaders independently on large worlds."""
    contact = wp.tid()
    group_color[contact] = -2
    if contact >= contact_count[0]:
        return

    shape0 = contact_shape0[contact]
    shape1 = contact_shape1[contact]
    body0 = shape_body[shape0] if shape0 >= 0 else -1
    body1 = shape_body[shape1] if shape1 >= 0 else -1
    if contact > 0:
        previous_shape0 = contact_shape0[contact - 1]
        previous_shape1 = contact_shape1[contact - 1]
        previous_body0 = shape_body[previous_shape0] if previous_shape0 >= 0 else -1
        previous_body1 = shape_body[previous_shape1] if previous_shape1 >= 0 else -1
        if previous_body0 == body0 and previous_body1 == body1:
            return

    dynamic0 = body0 >= 0 and body_inv_mass[body0] > 0.0
    dynamic1 = body1 >= 0 and body_inv_mass[body1] > 0.0
    pair0 = body0 < 0 or body_uses_pair_contacts[body0] != 0
    pair1 = body1 < 0 or body_uses_pair_contacts[body1] != 0
    if body0 == body1 or not (dynamic0 or dynamic1) or not pair0 or not pair1:
        return

    end = contact + 1
    while end < contact_count[0]:
        next_shape0 = contact_shape0[end]
        next_shape1 = contact_shape1[end]
        next_body0 = shape_body[next_shape0] if next_shape0 >= 0 else -1
        next_body1 = shape_body[next_shape1] if next_shape1 >= 0 else -1
        if next_body0 != body0 or next_body1 != body1:
            break
        end += 1

    group_start[contact] = contact
    group_end[contact] = end
    group_body0[contact] = body0
    group_body1[contact] = body1
    group_color[contact] = -1
    wp.atomic_add(group_count_out, 0, 1)


@wp.kernel
def propose_pair_color_parallel(
    color_count: int,
    round_index: int,
    group_color: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_color_mask: wp.array[wp.int32],
    group_proposal: wp.array[wp.int32],
    body_color_owner: wp.array[wp.int64],
):
    """Propose the lowest free color and elect unique endpoint owners."""
    group = wp.tid()
    group_proposal[group] = -1
    if group_color[group] != -1:
        return
    body0 = group_body0[group]
    body1 = group_body1[group]
    used_mask = wp.int32(0)
    if body0 >= 0 and body_inv_mass[body0] > 0.0:
        used_mask |= body_color_mask[body0]
    if body1 >= 0 and body_inv_mass[body1] > 0.0:
        used_mask |= body_color_mask[body1]

    available = (~used_mask) & ((wp.int32(1) << color_count) - 1)
    selected = wp.int32(-1)
    color = wp.int32(0)
    while available != 0 and selected < 0:
        if (available & (wp.int32(1) << color)) != 0:
            selected = color
        color += 1
    if selected < 0:
        return

    group_proposal[group] = selected
    priority = (group * wp.int32(1103515245) + round_index * wp.int32(12345)) & wp.int32(0x7FFFFFFF)
    priority_tag = (wp.int64(round_index) << wp.int64(32)) | wp.int64(priority)
    if body0 >= 0 and body_inv_mass[body0] > 0.0:
        wp.atomic_max(body_color_owner, body0 * color_count + selected, priority_tag)
    if body1 >= 0 and body_inv_mass[body1] > 0.0:
        wp.atomic_max(body_color_owner, body1 * color_count + selected, priority_tag)


@wp.kernel
def commit_pair_color_parallel(
    color_count: int,
    round_index: int,
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_color_owner: wp.array[wp.int64],
    group_proposal: wp.array[wp.int32],
    body_color_mask: wp.array[wp.int32],
    group_color: wp.array[wp.int32],
    color_counts: wp.array[wp.int32],
):
    """Commit an edge only when it won at every dynamic endpoint."""
    group = wp.tid()
    if group_color[group] != -1:
        return
    color = group_proposal[group]
    if color < 0:
        return
    body0 = group_body0[group]
    body1 = group_body1[group]
    priority = (group * wp.int32(1103515245) + round_index * wp.int32(12345)) & wp.int32(0x7FFFFFFF)
    priority_tag = (wp.int64(round_index) << wp.int64(32)) | wp.int64(priority)
    wins0 = body0 < 0 or body_inv_mass[body0] <= 0.0 or body_color_owner[body0 * color_count + color] == priority_tag
    wins1 = body1 < 0 or body_inv_mass[body1] <= 0.0 or body_color_owner[body1 * color_count + color] == priority_tag
    if wins0 and wins1:
        group_color[group] = color
        bit = wp.int32(1) << color
        if body0 >= 0 and body_inv_mass[body0] > 0.0:
            wp.atomic_or(body_color_mask, body0, bit)
        if body1 >= 0 and body_inv_mass[body1] > 0.0:
            wp.atomic_or(body_color_mask, body1, bit)
        wp.atomic_add(color_counts, color, 1)


@wp.kernel
def mark_pending_pair_groups(
    group_count: wp.array[wp.int32],
    color_counts: wp.array[wp.int32],
    pending: wp.array[wp.int32],
):
    """Check completion on-device before extending the parallel color rounds."""
    colored = wp.int32(0)
    for color in range(color_counts.shape[0]):
        colored += color_counts[color]
    pending[0] = wp.int32(colored < group_count[0])


@wp.kernel
def color_pair_tail_serial(
    first_color: int,
    color_count: int,
    contact_count: wp.array[wp.int32],
    group_count: wp.array[wp.int32],
    group_color: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_color_mask: wp.array[wp.int32],
    color_counts: wp.array[wp.int32],
    overflow_count: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    color_cursors: wp.array[wp.int32],
):
    """Color the sparse tail deterministically, then build color offsets."""
    if wp.tid() != 0:
        return

    overflow_count[0] = 0
    colored_count = wp.int32(0)
    for color in range(first_color):
        colored_count += color_counts[color]
    if colored_count < group_count[0]:
        remaining = group_count[0] - colored_count
        group = contact_count[0] - 1
        while group >= 0 and remaining > 0:
            if group_color[group] == -1:
                body0 = group_body0[group]
                body1 = group_body1[group]
                used_mask = wp.int32(0)
                if body0 >= 0 and body_inv_mass[body0] > 0.0:
                    used_mask |= body_color_mask[body0]
                if body1 >= 0 and body_inv_mass[body1] > 0.0:
                    used_mask |= body_color_mask[body1]

                selected = wp.int32(-1)
                # Parallel rounds may leave a group even when a low color is
                # reusable.  Search the full palette so used colors remain a
                # contiguous prefix, which also makes the fused-solve early
                # exit below valid.
                color = wp.int32(0)
                available = (~used_mask) & ((wp.int32(1) << color_count) - 1)
                while available != 0 and selected < 0:
                    if (available & (wp.int32(1) << color)) != 0:
                        selected = color
                    color += 1

                group_color[group] = selected
                if selected >= 0:
                    bit = wp.int32(1) << selected
                    if body0 >= 0 and body_inv_mass[body0] > 0.0:
                        body_color_mask[body0] |= bit
                    if body1 >= 0 and body_inv_mass[body1] > 0.0:
                        body_color_mask[body1] |= bit
                    color_counts[selected] += 1
                else:
                    overflow_count[0] += 1
                remaining -= 1
            group -= 1

    offset = wp.int32(0)
    for color in range(color_count):
        color_offsets[color] = offset
        color_cursors[color] = offset
        offset += color_counts[color]
    color_offsets[color_count] = offset


@wp.kernel
def scatter_pair_groups_by_color(
    group_color: wp.array[wp.int32],
    color_cursors: wp.array[wp.int32],
    group_list: wp.array[wp.int32],
):
    """Pack exact-color groups; within-color order is irrelevant."""
    group = wp.tid()
    color = group_color[group]
    if color >= 0:
        slot = wp.atomic_add(color_cursors, color, 1)
        group_list[slot] = group


@wp.kernel
def sort_constraint_contact_lists(
    body_contact_buffer_size: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
):
    """Sort each bounded per-body list so multiplier sums are reproducible."""
    body = wp.tid()
    count = wp.min(body_contact_counts[body], body_contact_buffer_size)
    for i in range(1, count):
        value = body_contact_indices[body * body_contact_buffer_size + i]
        j = i - 1
        while j >= 0 and body_contact_indices[body * body_contact_buffer_size + j] > value:
            body_contact_indices[body * body_contact_buffer_size + j + 1] = body_contact_indices[
                body * body_contact_buffer_size + j
            ]
            j -= 1
        body_contact_indices[body * body_contact_buffer_size + j + 1] = value


@wp.kernel
def accumulate_constraint_contact_forces_per_body(
    color_group: wp.array[wp.int32],
    dynamic_body_ids: wp.array[wp.int32],
    dynamic_color_starts: wp.array[wp.int32],
    active_dynamic_color: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_uses_pair_contacts: wp.array[wp.int32],
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    contact_lambda: wp.array[wp.vec3],
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
):
    """Accumulate the current contact reaction in each VBD primal block."""
    slot = wp.tid()
    body = wp.int32(-1)
    if active_dynamic_color >= 0:
        start = dynamic_color_starts[active_dynamic_color]
        count_in_color = dynamic_color_starts[active_dynamic_color + 1] - start
        if slot < count_in_color:
            body = dynamic_body_ids[start + slot]
    elif slot < color_group.shape[0]:
        body = color_group[slot]
    if body < 0 or body_inv_mass[body] <= 0.0:
        return

    force_sum = wp.vec3(0.0)
    torque_sum = wp.vec3(0.0)
    center = wp.transform_point(body_q[body], body_com[body])
    count = wp.min(body_contact_counts[body], body_contact_buffer_pre_alloc)
    for slot in range(count):
        contact = body_contact_indices[body * body_contact_buffer_pre_alloc + slot]
        if contact >= contact_count[0]:
            continue
        shape0 = contact_shape0[contact]
        shape1 = contact_shape1[contact]
        body0 = shape_body[shape0] if shape0 >= 0 else -1
        body1 = shape_body[shape1] if shape1 >= 0 else -1
        pair0 = body0 < 0 or body_uses_pair_contacts[body0] != 0
        pair1 = body1 < 0 or body_uses_pair_contacts[body1] != 0
        if not pair0 or not pair1:
            continue
        reaction = contact_lambda[contact]
        point = wp.vec3(0.0)
        force = wp.vec3(0.0)
        if body == body0:
            point = contact_surface_point(body_q[body], contact_point0[contact], contact_offset0[contact])
            force = -reaction
        elif body == body1:
            point = contact_surface_point(body_q[body], contact_point1[contact], contact_offset1[contact])
            force = reaction
        else:
            continue
        force_sum += force
        torque_sum += wp.cross(point - center, force)

    body_forces[body] += force_sum
    body_torques[body] += torque_sum


@wp.kernel
def query_constraint_contact_forces(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    contact_lambda: wp.array[wp.vec3],
    out_body0: wp.array[wp.int32],
    out_body1: wp.array[wp.int32],
    out_point0_world: wp.array[wp.vec3],
    out_point1_world: wp.array[wp.vec3],
    out_force_on_body1: wp.array[wp.vec3],
):
    """Expose the solved force multiplier and current surface points."""
    contact = wp.tid()
    if contact >= contact_count[0]:
        out_body0[contact] = -1
        out_body1[contact] = -1
        out_point0_world[contact] = wp.vec3(0.0)
        out_point1_world[contact] = wp.vec3(0.0)
        out_force_on_body1[contact] = wp.vec3(0.0)
        return

    shape0 = contact_shape0[contact]
    shape1 = contact_shape1[contact]
    body0 = shape_body[shape0] if shape0 >= 0 else -1
    body1 = shape_body[shape1] if shape1 >= 0 else -1
    out_body0[contact] = body0
    out_body1[contact] = body1
    out_point0_world[contact] = (
        contact_surface_point(body_q[body0], contact_point0[contact], contact_offset0[contact])
        if body0 >= 0
        else contact_point0[contact] + contact_offset0[contact]
    )
    out_point1_world[contact] = (
        contact_surface_point(body_q[body1], contact_point1[contact], contact_offset1[contact])
        if body1 >= 0
        else contact_point1[contact] + contact_offset1[contact]
    )
    out_force_on_body1[contact] = contact_lambda[contact]


@wp.func
def _constraint_body_response(
    body: int,
    pose: wp.transform,
    point_world: wp.vec3,
    force_world: wp.vec3,
    dt: float,
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
):
    """Return the inertial-metric pose response to a unit force."""
    translation = wp.vec3(0.0)
    rotation = wp.vec3(0.0)
    if body >= 0 and body_inv_mass[body] > 0.0:
        center = wp.transform_point(pose, body_com[body])
        torque = wp.cross(point_world - center, force_world)
        q = wp.transform_get_rotation(pose)
        dt_sq = dt * dt
        translation = dt_sq * body_inv_mass[body] * force_world
        rotation = dt_sq * wp.quat_rotate(
            q,
            body_inv_inertia[body] * wp.quat_rotate_inv(q, torque),
        )
    return translation, rotation


@wp.func
def _apply_constraint_pose_increment(
    body: int,
    pose: wp.transform,
    translation: wp.vec3,
    rotation: wp.vec3,
    scale: float,
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
):
    """Apply one Hessian-metric SE(3) increment to a dynamic body."""
    if body < 0 or body_inv_mass[body] <= 0.0 or scale == 0.0:
        return pose

    q = wp.transform_get_rotation(pose)
    center = wp.transform_point(pose, body_com[body])
    w = rotation * scale
    half_w = 0.5 * w
    dq = wp.normalize(wp.quat(half_w[0], half_w[1], half_w[2], 1.0))
    q_new = wp.normalize(wp.mul(dq, q))
    center_new = center + translation * scale
    origin_new = center_new - wp.quat_rotate(q_new, body_com[body])
    return wp.transform(origin_new, q_new)


@wp.func
def _constraint_direction_response(
    body0: int,
    body1: int,
    pose0: wp.transform,
    pose1: wp.transform,
    point0: wp.vec3,
    point1: wp.vec3,
    direction: wp.vec3,
    dt: float,
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
):
    """Return endpoint responses and relative point motion for a unit pair force."""
    x0, w0 = _constraint_body_response(
        body0,
        pose0,
        point0,
        -direction,
        dt,
        body_com,
        body_inv_mass,
        body_inv_inertia,
    )
    x1, w1 = _constraint_body_response(
        body1,
        pose1,
        point1,
        direction,
        dt,
        body_com,
        body_inv_mass,
        body_inv_inertia,
    )

    point_delta0 = wp.vec3(0.0)
    point_delta1 = wp.vec3(0.0)
    if body0 >= 0 and body_inv_mass[body0] > 0.0:
        center0 = wp.transform_point(pose0, body_com[body0])
        point_delta0 = x0 + wp.cross(w0, point0 - center0)
    if body1 >= 0 and body_inv_mass[body1] > 0.0:
        center1 = wp.transform_point(pose1, body_com[body1])
        point_delta1 = x1 + wp.cross(w1, point1 - center1)
    return x0, w0, x1, w1, point_delta1 - point_delta0


@wp.kernel
def prepare_constraint_contact_blocks(
    dt: float,
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    tangent0: wp.array[wp.vec3],
    tangent1: wp.array[wp.vec3],
    response0_normal: wp.array[wp.spatial_vector],
    response1_normal: wp.array[wp.spatial_vector],
    response0_tangent0: wp.array[wp.spatial_vector],
    response1_tangent0: wp.array[wp.spatial_vector],
    response0_tangent1: wp.array[wp.spatial_vector],
    response1_tangent1: wp.array[wp.spatial_vector],
    normal_support: wp.array[float],
    tangent_support: wp.array[wp.mat22],
):
    """Prepare a fixed inertial Delassus linearization for each contact."""
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    shape0 = contact_shape0[contact]
    shape1 = contact_shape1[contact]
    body0 = shape_body[shape0] if shape0 >= 0 else -1
    body1 = shape_body[shape1] if shape1 >= 0 else -1
    pose0 = body_q[body0] if body0 >= 0 else wp.transform_identity()
    pose1 = body_q[body1] if body1 >= 0 else wp.transform_identity()
    skeleton0 = wp.transform_point(pose0, contact_point0[contact]) if body0 >= 0 else contact_point0[contact]
    skeleton1 = wp.transform_point(pose1, contact_point1[contact]) if body1 >= 0 else contact_point1[contact]
    anchor0 = (
        contact_surface_point(pose0, contact_point0[contact], contact_offset0[contact])
        if body0 >= 0
        else contact_point0[contact] + contact_offset0[contact]
    )
    anchor1 = (
        contact_surface_point(pose1, contact_point1[contact], contact_offset1[contact])
        if body1 >= 0
        else contact_point1[contact] + contact_offset1[contact]
    )
    n = contact_normal[contact]
    t0, t1 = orthonormal_basis(n)
    x0_n, w0_n, x1_n, w1_n, relative_n = _constraint_direction_response(
        body0,
        body1,
        pose0,
        pose1,
        skeleton0,
        skeleton1,
        n,
        dt,
        body_com,
        body_inv_mass,
        body_inv_inertia,
    )
    x0_t0, w0_t0, x1_t0, w1_t0, relative_t0 = _constraint_direction_response(
        body0,
        body1,
        pose0,
        pose1,
        anchor0,
        anchor1,
        t0,
        dt,
        body_com,
        body_inv_mass,
        body_inv_inertia,
    )
    x0_t1, w0_t1, x1_t1, w1_t1, relative_t1 = _constraint_direction_response(
        body0,
        body1,
        pose0,
        pose1,
        anchor0,
        anchor1,
        t1,
        dt,
        body_com,
        body_inv_mass,
        body_inv_inertia,
    )
    W00 = wp.dot(t0, relative_t0)
    W11 = wp.dot(t1, relative_t1)
    W01 = 0.5 * (wp.dot(t1, relative_t0) + wp.dot(t0, relative_t1))
    tangent0[contact] = t0
    tangent1[contact] = t1
    response0_normal[contact] = wp.spatial_vector(x0_n, w0_n)
    response1_normal[contact] = wp.spatial_vector(x1_n, w1_n)
    response0_tangent0[contact] = wp.spatial_vector(x0_t0, w0_t0)
    response1_tangent0[contact] = wp.spatial_vector(x1_t0, w1_t0)
    response0_tangent1[contact] = wp.spatial_vector(x0_t1, w0_t1)
    response1_tangent1[contact] = wp.spatial_vector(x1_t1, w1_t1)
    normal_support[contact] = wp.dot(n, relative_n)
    tangent_support[contact] = wp.mat22(W00, W01, W01, W11)


@wp.func
def _solve_constraint_contact_group(
    group: int,
    dt: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    group_start: wp.array[wp.int32],
    group_end: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_C0: wp.array[wp.vec3],
    contact_tangent0: wp.array[wp.vec3],
    contact_tangent1: wp.array[wp.vec3],
    response0_normal: wp.array[wp.spatial_vector],
    response1_normal: wp.array[wp.spatial_vector],
    response0_tangent0: wp.array[wp.spatial_vector],
    response1_tangent0: wp.array[wp.spatial_vector],
    response0_tangent1: wp.array[wp.spatial_vector],
    response1_tangent1: wp.array[wp.spatial_vector],
    contact_normal_support: wp.array[float],
    contact_tangent_support: wp.array[wp.mat22],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    contact_lambda: wp.array[wp.vec3],
):
    """Solve one body-pair manifold in the free-body inertial metric.

    Normal compliance is the implicit Kelvin--Voigt relation
    ``lambda = k*C + d*(C-C_prev)/dt``. Tangential force minimizes the local
    two-dimensional quadratic over the Coulomb disk using a bounded root solve.
    """
    body0 = group_body0[group]
    body1 = group_body1[group]
    pose0 = body_q[body0] if body0 >= 0 else wp.transform_identity()
    pose1 = body_q[body1] if body1 >= 0 else wp.transform_identity()

    for contact in range(group_start[group], group_end[group]):
        n = contact_normal[contact]
        skeleton0 = wp.transform_point(pose0, contact_point0[contact]) if body0 >= 0 else contact_point0[contact]
        skeleton1 = wp.transform_point(pose1, contact_point1[contact]) if body1 >= 0 else contact_point1[contact]
        penetration = -contact_surface_separation(
            skeleton0,
            skeleton1,
            n,
            contact_margin0[contact],
            contact_margin1[contact],
        )

        old_lambda = contact_lambda[contact]
        old_normal = wp.max(wp.dot(old_lambda, n), 0.0)
        old_tangent = old_lambda - n * wp.dot(old_lambda, n)
        C0 = contact_C0[contact]
        C0_n = wp.dot(C0, n)

        compliance = float(0.0)
        damping_fraction = float(0.0)
        use_finite_material = contact_compliant_alm == 1 or legacy_hard_contacts == 0
        if use_finite_material:
            stiffness = contact_material_ke[contact]
            damping_rate = contact_material_kd[contact] / dt
            effective_stiffness = stiffness + damping_rate
            if effective_stiffness > 0.0:
                compliance = 1.0 / effective_stiffness
                damping_fraction = damping_rate / effective_stiffness

        normal_residual = penetration - damping_fraction * C0_n - compliance * old_normal
        response0_n = response0_normal[contact]
        response1_n = response1_normal[contact]
        x0_n = wp.spatial_top(response0_n)
        w0_n = wp.spatial_bottom(response0_n)
        x1_n = wp.spatial_top(response1_n)
        w1_n = wp.spatial_bottom(response1_n)
        normal_support = contact_normal_support[contact]
        new_normal = float(0.0)
        denominator = normal_support + compliance
        if denominator > _EPS:
            new_normal = wp.max(old_normal + normal_residual / denominator, 0.0)
        delta_normal = new_normal - old_normal

        # Evaluate the two tangential rows at the same linearization point as
        # the normal row, then apply their combined 3-DOF contact increment.
        anchor0_local = contact_point0[contact] + contact_offset0[contact]
        anchor1_local = contact_point1[contact] + contact_offset1[contact]
        anchor0 = (
            contact_surface_point(pose0, contact_point0[contact], contact_offset0[contact])
            if body0 >= 0
            else anchor0_local
        )
        anchor1 = (
            contact_surface_point(pose1, contact_point1[contact], contact_offset1[contact])
            if body1 >= 0
            else anchor1_local
        )
        prev_pose0 = body_q_prev[body0] if body0 >= 0 else wp.transform_identity()
        prev_pose1 = body_q_prev[body1] if body1 >= 0 else wp.transform_identity()
        anchor0_prev = (
            contact_surface_point(prev_pose0, contact_point0[contact], contact_offset0[contact])
            if body0 >= 0
            else anchor0_local
        )
        anchor1_prev = (
            contact_surface_point(prev_pose1, contact_point1[contact], contact_offset1[contact])
            if body1 >= 0
            else anchor1_local
        )
        relative_motion = (anchor1 - anchor1_prev) - (anchor0 - anchor0_prev)

        slip = relative_motion - n * wp.dot(n, relative_motion)
        tangent_drive = -slip

        t0 = contact_tangent0[contact]
        t1 = contact_tangent1[contact]
        response0_t0 = response0_tangent0[contact]
        response1_t0 = response1_tangent0[contact]
        response0_t1 = response0_tangent1[contact]
        response1_t1 = response1_tangent1[contact]
        x0_t0 = wp.spatial_top(response0_t0)
        w0_t0 = wp.spatial_bottom(response0_t0)
        x1_t0 = wp.spatial_top(response1_t0)
        w1_t0 = wp.spatial_bottom(response1_t0)
        x0_t1 = wp.spatial_top(response0_t1)
        w0_t1 = wp.spatial_bottom(response0_t1)
        x1_t1 = wp.spatial_top(response1_t1)
        w1_t1 = wp.spatial_bottom(response1_t1)
        tangent_W = contact_tangent_support[contact]
        W00 = tangent_W[0, 0]
        W01 = tangent_W[0, 1]
        W11 = tangent_W[1, 1]
        determinant = W00 * W11 - W01 * W01
        support_product = W00 * W11
        candidate_tangent = wp.vec3(0.0)
        if W00 > 0.0 and W11 > 0.0 and determinant > _TANGENT_EPS * support_product:
            rhs0 = wp.dot(tangent_drive, t0)
            rhs1 = wp.dot(tangent_drive, t1)
            delta0 = (W11 * rhs0 - W01 * rhs1) / determinant
            delta1 = (W00 * rhs1 - W01 * rhs0) / determinant
            candidate_tangent = old_tangent + t0 * delta0 + t1 * delta1
        friction_limit = contact_material_mu[contact] * new_normal
        candidate_length = wp.length(candidate_tangent)
        if candidate_length > friction_limit and candidate_length > _EPS:
            if friction_limit > _EPS:
                # Minimize the local quadratic in the W metric on the Coulomb disk.
                c0 = wp.dot(candidate_tangent, t0)
                c1 = wp.dot(candidate_tangent, t1)
                rhs0 = W00 * c0 + W01 * c1
                rhs1 = W01 * c0 + W11 * c1
                lower = float(0.0)
                upper = wp.sqrt(rhs0 * rhs0 + rhs1 * rhs1) / friction_limit
                for _root_iter in range(16):
                    eta = 0.5 * (lower + upper)
                    d0 = W00 + eta
                    d1 = W11 + eta
                    inverse_det = 1.0 / (d0 * d1 - W01 * W01)
                    x0 = (d1 * rhs0 - W01 * rhs1) * inverse_det
                    x1 = (d0 * rhs1 - W01 * rhs0) * inverse_det
                    if x0 * x0 + x1 * x1 > friction_limit * friction_limit:
                        lower = eta
                    else:
                        upper = eta
                d0 = W00 + upper
                d1 = W11 + upper
                inverse_det = 1.0 / (d0 * d1 - W01 * W01)
                candidate_tangent = t0 * ((d1 * rhs0 - W01 * rhs1) * inverse_det) + t1 * (
                    (d0 * rhs1 - W01 * rhs0) * inverse_det
                )
            else:
                candidate_tangent = wp.vec3(0.0)
        delta_tangent = candidate_tangent - old_tangent
        delta_t0 = wp.dot(delta_tangent, t0)
        delta_t1 = wp.dot(delta_tangent, t1)
        pose0 = _apply_constraint_pose_increment(
            body0,
            pose0,
            x0_n * delta_normal + x0_t0 * delta_t0 + x0_t1 * delta_t1,
            w0_n * delta_normal + w0_t0 * delta_t0 + w0_t1 * delta_t1,
            1.0,
            body_com,
            body_inv_mass,
        )
        pose1 = _apply_constraint_pose_increment(
            body1,
            pose1,
            x1_n * delta_normal + x1_t0 * delta_t0 + x1_t1 * delta_t1,
            w1_n * delta_normal + w1_t0 * delta_t0 + w1_t1 * delta_t1,
            1.0,
            body_com,
            body_inv_mass,
        )
        contact_lambda[contact] = n * new_normal + candidate_tangent

    if body0 >= 0 and body_inv_mass[body0] > 0.0:
        body_q[body0] = pose0
    if body1 >= 0 and body_inv_mass[body1] > 0.0:
        body_q[body1] = pose1


@wp.kernel
def solve_constraint_contact_color(
    launch_dim: int,
    active_color: int,
    dt: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    group_list: wp.array[wp.int32],
    group_start: wp.array[wp.int32],
    group_end: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_C0: wp.array[wp.vec3],
    contact_tangent0: wp.array[wp.vec3],
    contact_tangent1: wp.array[wp.vec3],
    response0_normal: wp.array[wp.spatial_vector],
    response1_normal: wp.array[wp.spatial_vector],
    response0_tangent0: wp.array[wp.spatial_vector],
    response1_tangent0: wp.array[wp.spatial_vector],
    response0_tangent1: wp.array[wp.spatial_vector],
    response1_tangent1: wp.array[wp.spatial_vector],
    contact_normal_support: wp.array[float],
    contact_tangent_support: wp.array[wp.mat22],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    contact_lambda: wp.array[wp.vec3],
):
    slot = wp.tid()
    count = color_counts[active_color]
    offset = color_offsets[active_color]
    while slot < count:
        group = group_list[offset + slot]
        _solve_constraint_contact_group(
            group,
            dt,
            legacy_hard_contacts,
            contact_compliant_alm,
            group_start,
            group_end,
            group_body0,
            group_body1,
            contact_point0,
            contact_point1,
            contact_offset0,
            contact_offset1,
            contact_normal,
            contact_margin0,
            contact_margin1,
            contact_material_ke,
            contact_material_kd,
            contact_material_mu,
            contact_C0,
            contact_tangent0,
            contact_tangent1,
            response0_normal,
            response1_normal,
            response0_tangent0,
            response1_tangent0,
            response0_tangent1,
            response1_tangent1,
            contact_normal_support,
            contact_tangent_support,
            body_q_prev,
            body_com,
            body_inv_mass,
            body_q,
            contact_lambda,
        )
        slot += launch_dim


@wp.kernel
def solve_constraint_contact_colors_fused(
    first_color: int,
    color_count: int,
    dt: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    group_list: wp.array[wp.int32],
    group_start: wp.array[wp.int32],
    group_end: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_C0: wp.array[wp.vec3],
    contact_tangent0: wp.array[wp.vec3],
    contact_tangent1: wp.array[wp.vec3],
    response0_normal: wp.array[wp.spatial_vector],
    response1_normal: wp.array[wp.spatial_vector],
    response0_tangent0: wp.array[wp.spatial_vector],
    response1_tangent0: wp.array[wp.spatial_vector],
    response0_tangent1: wp.array[wp.spatial_vector],
    response1_tangent1: wp.array[wp.spatial_vector],
    contact_normal_support: wp.array[float],
    contact_tangent_support: wp.array[wp.mat22],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    contact_lambda: wp.array[wp.vec3],
):
    """Solve all colors in one block, using block barriers as color barriers."""
    lane = wp.tid()
    color = first_color
    while color < color_count:
        count = color_counts[color]
        # Greedy coloring produces a contiguous palette.  Once a color is
        # empty, every later color is empty too, so avoid useless barriers.
        if count == 0:
            break
        slot = lane
        offset = color_offsets[color]
        while slot < count:
            group = group_list[offset + slot]
            _solve_constraint_contact_group(
                group,
                dt,
                legacy_hard_contacts,
                contact_compliant_alm,
                group_start,
                group_end,
                group_body0,
                group_body1,
                contact_point0,
                contact_point1,
                contact_offset0,
                contact_offset1,
                contact_normal,
                contact_margin0,
                contact_margin1,
                contact_material_ke,
                contact_material_kd,
                contact_material_mu,
                contact_C0,
                contact_tangent0,
                contact_tangent1,
                response0_normal,
                response1_normal,
                response0_tangent0,
                response1_tangent0,
                response0_tangent1,
                response1_tangent1,
                contact_normal_support,
                contact_tangent_support,
                body_q_prev,
                body_com,
                body_inv_mass,
                body_q,
                contact_lambda,
            )
            slot += _FUSED_THREADS
        _pair_sync_block()
        color += 1


@wp.kernel
def solve_constraint_contact_overflow(
    group_capacity: int,
    solve_all: int,
    dt: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    overflow_count: wp.array[wp.int32],
    group_color: wp.array[wp.int32],
    group_start: wp.array[wp.int32],
    group_end: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_C0: wp.array[wp.vec3],
    contact_tangent0: wp.array[wp.vec3],
    contact_tangent1: wp.array[wp.vec3],
    response0_normal: wp.array[wp.spatial_vector],
    response1_normal: wp.array[wp.spatial_vector],
    response0_tangent0: wp.array[wp.spatial_vector],
    response1_tangent0: wp.array[wp.spatial_vector],
    response0_tangent1: wp.array[wp.spatial_vector],
    response1_tangent1: wp.array[wp.spatial_vector],
    contact_normal_support: wp.array[float],
    contact_tangent_support: wp.array[wp.mat22],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    contact_lambda: wp.array[wp.vec3],
):
    if wp.tid() != 0 or (solve_all == 0 and overflow_count[0] == 0):
        return
    for group in range(group_capacity):
        color = group_color[group]
        if color == -1 or (solve_all == 1 and color >= 0):
            _solve_constraint_contact_group(
                group,
                dt,
                legacy_hard_contacts,
                contact_compliant_alm,
                group_start,
                group_end,
                group_body0,
                group_body1,
                contact_point0,
                contact_point1,
                contact_offset0,
                contact_offset1,
                contact_normal,
                contact_margin0,
                contact_margin1,
                contact_material_ke,
                contact_material_kd,
                contact_material_mu,
                contact_C0,
                contact_tangent0,
                contact_tangent1,
                response0_normal,
                response1_normal,
                response0_tangent0,
                response1_tangent0,
                response0_tangent1,
                response1_tangent1,
                contact_normal_support,
                contact_tangent_support,
                body_q_prev,
                body_com,
                body_inv_mass,
                body_q,
                contact_lambda,
            )


class RigidPairContact:
    """Fixed-capacity exact edge-color schedule outside the rigid-joint graph."""

    def __init__(
        self,
        model,
        body_inv_mass: wp.array,
        body_inv_inertia: wp.array,
        body_uses_pair_contacts: wp.array,
        pair_body_count: int,
    ):
        self.model = model
        self.device = model.device
        self.body_inv_mass = body_inv_mass
        self.body_inv_inertia = body_inv_inertia
        self.body_uses_pair_contacts = body_uses_pair_contacts
        self.pair_body_count = pair_body_count
        self.capacity = 0

        body_count = model.body_count
        self.body_color_mask = wp.zeros(body_count, dtype=wp.int32, device=self.device)
        self.body_color_owner = wp.full(body_count * _COLOR_COUNT, -1, dtype=wp.int64, device=self.device)
        self.color_counts = wp.zeros(_COLOR_COUNT, dtype=wp.int32, device=self.device)
        self.color_offsets = wp.zeros(_COLOR_COUNT + 1, dtype=wp.int32, device=self.device)
        self.color_cursors = wp.zeros(_COLOR_COUNT, dtype=wp.int32, device=self.device)
        self.group_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.overflow_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.resize(0)

    def resize(self, capacity: int) -> None:
        if capacity <= self.capacity:
            return
        self.capacity = capacity
        self.group_start = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.group_end = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.group_body0 = wp.full(capacity, -1, dtype=wp.int32, device=self.device)
        self.group_body1 = wp.full(capacity, -1, dtype=wp.int32, device=self.device)
        self.group_color = wp.full(capacity, -2, dtype=wp.int32, device=self.device)
        self.group_proposal = wp.full(capacity, -1, dtype=wp.int32, device=self.device)
        self.group_list = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.tangent0 = wp.zeros(capacity, dtype=wp.vec3, device=self.device)
        self.tangent1 = wp.zeros(capacity, dtype=wp.vec3, device=self.device)
        self.response0_normal = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self.response1_normal = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self.response0_tangent0 = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self.response1_tangent0 = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self.response0_tangent1 = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self.response1_tangent1 = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self.normal_support = wp.zeros(capacity, dtype=float, device=self.device)
        self.tangent_support = wp.zeros(capacity, dtype=wp.mat22, device=self.device)

    def build(self, contacts) -> None:
        """Build body-pair manifolds and a deterministic conflict-free schedule."""
        capacity = contacts.rigid_contact_max
        if capacity <= 0:
            return
        self.resize(capacity)
        model = self.model

        if self.device.is_cuda and model.body_count > self.device.sm_count:
            self.group_count.zero_()
            self.overflow_count.zero_()
            self.color_counts.zero_()
            self.body_color_mask.zero_()
            wp.launch(
                kernel=mark_pair_groups_parallel,
                dim=capacity,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    model.shape_body,
                    self.body_inv_mass,
                    self.body_uses_pair_contacts,
                ],
                outputs=[
                    self.group_start,
                    self.group_end,
                    self.group_body0,
                    self.group_body1,
                    self.group_color,
                    self.group_count,
                ],
                device=self.device,
            )
            # Round tags invalidate old owners without clearing between rounds.
            # Common graphs finish in the initial rounds; denser graphs can use
            # the remaining palette in parallel before deterministic completion.
            self.body_color_owner.fill_(-1)

            def color_rounds(first: int, last: int) -> None:
                for round_index in range(first, last):
                    wp.launch(
                        kernel=propose_pair_color_parallel,
                        dim=capacity,
                        inputs=[
                            _COLOR_COUNT,
                            round_index,
                            self.group_color,
                            self.group_body0,
                            self.group_body1,
                            self.body_inv_mass,
                            self.body_color_mask,
                        ],
                        outputs=[self.group_proposal, self.body_color_owner],
                        device=self.device,
                    )
                    wp.launch(
                        kernel=commit_pair_color_parallel,
                        dim=capacity,
                        inputs=[
                            _COLOR_COUNT,
                            round_index,
                            self.group_body0,
                            self.group_body1,
                            self.body_inv_mass,
                            self.body_color_owner,
                            self.group_proposal,
                        ],
                        outputs=[self.body_color_mask, self.group_color, self.color_counts],
                        device=self.device,
                    )

            color_rounds(0, _PARALLEL_BUILD_ROUNDS)
            if self.device.is_capturing and wp.is_conditional_graph_supported():
                # The tail overwrites this temporary flag with the overflow count.
                wp.launch(
                    mark_pending_pair_groups,
                    dim=1,
                    block_dim=1,
                    inputs=[self.group_count, self.color_counts],
                    outputs=[self.overflow_count],
                    device=self.device,
                )
                wp.capture_if(
                    self.overflow_count,
                    on_true=lambda: color_rounds(_PARALLEL_BUILD_ROUNDS, _COLOR_COUNT),
                )
            else:
                # Keep eager and non-conditional captures readback-free.
                color_rounds(_PARALLEL_BUILD_ROUNDS, _COLOR_COUNT)
            wp.launch(
                kernel=color_pair_tail_serial,
                dim=1,
                inputs=[
                    _COLOR_COUNT,
                    _COLOR_COUNT,
                    contacts.rigid_contact_count,
                    self.group_count,
                    self.group_color,
                    self.group_body0,
                    self.group_body1,
                    self.body_inv_mass,
                ],
                outputs=[
                    self.body_color_mask,
                    self.color_counts,
                    self.overflow_count,
                    self.color_offsets,
                    self.color_cursors,
                ],
                device=self.device,
                block_dim=1,
            )
            wp.launch(
                kernel=scatter_pair_groups_by_color,
                dim=capacity,
                inputs=[self.group_color],
                outputs=[self.color_cursors, self.group_list],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=build_and_color_pair_groups_serial,
                dim=1,
                inputs=[
                    model.body_count,
                    _COLOR_COUNT,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    model.shape_body,
                    self.body_inv_mass,
                    self.body_uses_pair_contacts,
                ],
                outputs=[
                    self.body_color_mask,
                    self.group_start,
                    self.group_end,
                    self.group_body0,
                    self.group_body1,
                    self.group_color,
                    self.group_list,
                    self.color_counts,
                    self.color_offsets,
                    self.color_cursors,
                    self.group_count,
                    self.overflow_count,
                ],
                device=self.device,
                block_dim=1,
            )

    def solve(
        self,
        body_q: wp.array,
        body_q_prev: wp.array,
        contacts,
        dt: float,
        contact_material_ke: wp.array,
        contact_material_kd: wp.array,
        contact_material_mu: wp.array,
        contact_C0: wp.array,
        contact_lambda: wp.array,
        legacy_hard_contacts: int,
        contact_compliant_alm: bool,
    ) -> None:
        """Apply one colored pair-manifold block sweep."""
        capacity = contacts.rigid_contact_max
        if capacity <= 0:
            return
        model = self.model
        wp.launch(
            kernel=prepare_constraint_contact_blocks,
            dim=capacity,
            inputs=[
                dt,
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                model.shape_body,
                body_q,
                model.body_com,
                self.body_inv_mass,
                self.body_inv_inertia,
            ],
            outputs=[
                self.tangent0,
                self.tangent1,
                self.response0_normal,
                self.response1_normal,
                self.response0_tangent0,
                self.response1_tangent0,
                self.response0_tangent1,
                self.response1_tangent1,
                self.normal_support,
                self.tangent_support,
            ],
            device=self.device,
            block_dim=select_rigid_pair_block_dim(capacity, self.device),
        )

        contact_inputs = [
            self.group_start,
            self.group_end,
            self.group_body0,
            self.group_body1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_offset0,
            contacts.rigid_contact_offset1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
            contact_material_ke,
            contact_material_kd,
            contact_material_mu,
            contact_C0,
            self.tangent0,
            self.tangent1,
            self.response0_normal,
            self.response1_normal,
            self.response0_tangent0,
            self.response1_tangent0,
            self.response0_tangent1,
            self.response1_tangent1,
            self.normal_support,
            self.tangent_support,
            body_q_prev,
            model.body_com,
            self.body_inv_mass,
        ]
        # One exact color contains at most one group per pair-routed body.
        launch_dim = max(1, min(capacity, max(self.pair_body_count, 128)))
        block_dim = select_rigid_pair_block_dim(launch_dim, self.device)
        if not self.device.is_cuda:
            wp.launch(
                kernel=solve_constraint_contact_overflow,
                dim=1,
                inputs=[
                    capacity,
                    1,
                    dt,
                    legacy_hard_contacts,
                    int(contact_compliant_alm),
                    self.overflow_count,
                    self.group_color,
                    *contact_inputs,
                ],
                outputs=[body_q, contact_lambda],
                device=self.device,
                block_dim=1,
            )
            return

        # An exact edge color contains at most one group per dynamic body, so
        # each color can be launched independently. Graph capture amortizes the
        # direct launches, while eager execution favors the one-block fused sweep
        # until its single-SM occupancy limit dominates. Structural bodies cannot
        # contribute pair work, so they must not select the schedule.
        min_bodies_per_sm = (
            _CAPTURED_DIRECT_COLOR_MIN_BODIES_PER_SM
            if self.device.is_capturing
            else _EAGER_DIRECT_COLOR_MIN_BODIES_PER_SM
        )
        direct_color_count = (
            0 if self.pair_body_count <= min_bodies_per_sm * self.device.sm_count else _DIRECT_COLOR_COUNT
        )
        for color in range(direct_color_count):
            wp.launch(
                kernel=solve_constraint_contact_color,
                dim=launch_dim,
                inputs=[
                    launch_dim,
                    color,
                    dt,
                    legacy_hard_contacts,
                    int(contact_compliant_alm),
                    self.color_counts,
                    self.color_offsets,
                    self.group_list,
                    *contact_inputs,
                ],
                outputs=[body_q, contact_lambda],
                device=self.device,
                block_dim=block_dim,
            )
        if direct_color_count < _COLOR_COUNT:
            wp.launch(
                kernel=solve_constraint_contact_colors_fused,
                dim=_FUSED_THREADS,
                inputs=[
                    direct_color_count,
                    _COLOR_COUNT,
                    dt,
                    legacy_hard_contacts,
                    int(contact_compliant_alm),
                    self.color_counts,
                    self.color_offsets,
                    self.group_list,
                    *contact_inputs,
                ],
                outputs=[body_q, contact_lambda],
                device=self.device,
                block_dim=_FUSED_THREADS,
                max_blocks=1,
            )
        wp.launch(
            kernel=solve_constraint_contact_overflow,
            dim=1,
            inputs=[
                capacity,
                0,
                dt,
                legacy_hard_contacts,
                int(contact_compliant_alm),
                self.overflow_count,
                self.group_color,
                *contact_inputs,
            ],
            outputs=[body_q, contact_lambda],
            device=self.device,
            block_dim=1,
        )
