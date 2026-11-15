# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic constraint-block contacts for free rigid bodies in VBD."""

import warp as wp

from newton._src.math import orthonormal_basis
from newton._src.sim.contacts import contact_surface_point, contact_surface_separation
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _body_id_for_color_slot,
    _select_rigid_block_dim,
    _sync_cuda_block,
)

wp.set_module_options({"enable_backward": False})

_COLOR_COUNT = 30
_DIRECT_COLOR_COUNT = 8
_PARALLEL_BUILD_ROUNDS = 10
_TAIL_SCAN_THREADS = 256
_FUSED_THREADS = 256
_CAPTURED_DIRECT_COLOR_MIN_BODIES_PER_SM = 1
_EAGER_DIRECT_COLOR_MIN_BODIES_PER_SM = 32
_EPS = wp.constant(1.0e-12)
_TANGENT_EPS = wp.constant(1.0e-6)


@wp.struct
class ContactConstants:
    """One contact as the solve reads it: geometry, material and step-start terms.

    The geometry is packed with the manifolds at every collision refresh; material and
    step-start terms are refreshed at the start of each step.
    Points and anchors are body-local (world-space for a world endpoint).
    Subtract the live center of mass in the solve so inertial-property updates
    do not invalidate cached collision geometry.
    """

    normal: wp.vec3
    point0: wp.vec3
    point1: wp.vec3
    anchor0: wp.vec3
    anchor1: wp.vec3
    anchor0_prev: wp.vec3
    anchor1_prev: wp.vec3
    surface_displacement: wp.vec3
    margin: float
    friction: float
    previous_penetration: float
    compliance: float
    damping_fraction: float
    elastic_compliance: float
    damping_bidirectional: bool


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

    # Serial builds pack groups densely; unused capacity already holds -2.
    for group in range(group_count_out[0]):
        group_color[group] = -2
    for body in range(body_count):
        body_color_mask[body] = 0
    for color in range(color_count):
        color_counts[color] = 0

    active_contacts = wp.min(contact_count[0], contact_shape0.shape[0])
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
            # Reverse traversal packs the tail in ascending group order, matching
            # the serial solve without rescanning unused contact capacity.
            group_list[group_list.shape[0] - 1 - overflow_count] = group
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
    color_count: int,
    pair_partner: wp.array[wp.int32],
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
    group_class: wp.array[wp.int32],
    group_count_out: wp.array[wp.int32],
):
    """Mark contiguous body-pair leaders independently on large worlds.

    A group's class is the color its body pair held in the previous build, or -1.
    """
    contact = wp.tid()
    group_color[contact] = -2
    active_contacts = wp.min(contact_count[0], contact_shape0.shape[0])
    if contact >= active_contacts:
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
    while end < active_contacts:
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
    group_class[contact] = -1
    body = body0
    partner = body1
    if not dynamic0:
        body = body1
        partner = body0
    for color in range(color_count):
        if pair_partner[body * color_count + color] == partner:
            group_class[contact] = color
    wp.atomic_add(group_count_out, 0, 1)


@wp.func
def _pair_color_tag(group: int, round_index: int, group_class: int, build_index: int) -> wp.int64:
    """Order groups by their previous color class, then by a hash that is unique per group.

    Even builds take the highest class first and odd builds a hashed class order, so successive builds are passes of
    iterated greedy with varying class orders. Groups without a class come last. Later rounds dominate earlier tags.
    """
    class_key = wp.int64(0)
    if group_class >= 0:
        order = wp.int64(group_class)
        if build_index % 2 == 1:
            spread = ((group_class + 7) * wp.int32(1103515245) + build_index * wp.int32(12345)) & wp.int32(0x3FF)
            order = (wp.int64(spread) << wp.int64(5)) | wp.int64(group_class)
        class_key = order + wp.int64(1)
    unique = wp.int64((group * wp.int32(1103515245) + build_index * wp.int32(12345)) & wp.int32(0x7FFFFFFF))
    return (wp.int64(round_index) << wp.int64(48)) | (class_key << wp.int64(31)) | unique


@wp.kernel
def propose_pair_color_parallel(
    round_index: int,
    build_index: wp.array[wp.int32],
    group_color: wp.array[wp.int32],
    group_class: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_color_owner: wp.array[wp.int64],
):
    """Post each uncolored group's tag to its dynamic bodies; the largest tag at a body may color next."""
    group = wp.tid()
    if group_color[group] != -1:
        return
    tag = _pair_color_tag(group, round_index, group_class[group], build_index[0])
    body0 = group_body0[group]
    body1 = group_body1[group]
    if body0 >= 0 and body_inv_mass[body0] > 0.0:
        wp.atomic_max(body_color_owner, body0, tag)
    if body1 >= 0 and body_inv_mass[body1] > 0.0:
        wp.atomic_max(body_color_owner, body1, tag)


@wp.kernel
def commit_pair_color_parallel(
    color_count: int,
    round_index: int,
    build_index: wp.array[wp.int32],
    group_class: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_color_owner: wp.array[wp.int64],
    body_color_mask: wp.array[wp.int32],
    group_color: wp.array[wp.int32],
    color_counts: wp.array[wp.int32],
):
    """Color a group holding the largest tag at all its dynamic bodies with the lowest color free there.

    Each body has one such group per round, so commits in a round never share a body.
    """
    group = wp.tid()
    if group_color[group] != -1:
        return
    tag = _pair_color_tag(group, round_index, group_class[group], build_index[0])
    body0 = group_body0[group]
    body1 = group_body1[group]
    dynamic0 = body0 >= 0 and body_inv_mass[body0] > 0.0
    dynamic1 = body1 >= 0 and body_inv_mass[body1] > 0.0
    if dynamic0 and body_color_owner[body0] != tag:
        return
    if dynamic1 and body_color_owner[body1] != tag:
        return
    used_mask = wp.int32(0)
    if dynamic0:
        used_mask |= body_color_mask[body0]
    if dynamic1:
        used_mask |= body_color_mask[body1]
    available = (~used_mask) & ((wp.int32(1) << color_count) - 1)
    if available == 0:
        return
    selected = wp.int32(0)
    while (available & (wp.int32(1) << selected)) == 0:
        selected += 1
    group_color[group] = selected
    bit = wp.int32(1) << selected
    if dynamic0:
        wp.atomic_or(body_color_mask, body0, bit)
    if dynamic1:
        wp.atomic_or(body_color_mask, body1, bit)
    wp.atomic_add(color_counts, selected, 1)


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
    group_list: wp.array[wp.int32],
    parallel_tail: wp.array[wp.int32],
    build_index: wp.array[wp.int32],
):
    """Scan pending groups cooperatively; color in descending order on lane zero."""
    lane = wp.tid()
    if lane == 0:
        build_index[0] = build_index[0] + 1
        overflow_count[0] = 0
    colored_count = wp.int32(0)
    for color in range(first_color):
        colored_count += color_counts[color]
    if colored_count < group_count[0]:
        remaining = group_count[0] - colored_count
        pending = wp.tile_zeros(shape=(_TAIL_SCAN_THREADS,), dtype=wp.int32, storage="shared")
        chunk_end = wp.min(contact_count[0], group_color.shape[0]) - 1
        while chunk_end >= 0 and remaining > 0:
            candidate = chunk_end - lane
            active = wp.int32(0)
            if candidate >= 0:
                active = wp.int32(group_color[candidate] == -1)
            offsets = wp.tile_scan_inclusive(wp.tile(active))
            count = offsets[_TAIL_SCAN_THREADS - 1]
            prefix = wp.untile(offsets)
            slot = count + lane - prefix
            if active != 0:
                slot = prefix - 1
            if count > 0:
                # Tile element assignment synchronizes the block. All lanes
                # write unique slots: pending IDs first, inactive IDs afterward.
                pending[slot] = candidate
                if lane == 0:
                    for index in range(count):
                        group = pending[index]
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
                            group_list[group_list.shape[0] - 1 - overflow_count[0]] = group
                            overflow_count[0] += 1
                _sync_cuda_block()
                remaining -= count
            chunk_end -= _TAIL_SCAN_THREADS

    if lane != 0:
        return

    offset = wp.int32(0)
    for color in range(color_count):
        color_offsets[color] = offset
        color_cursors[color] = offset
        offset += color_counts[color]
    color_offsets[color_count] = offset

    # More than one block of groups in the next batch warrants parallel colors.
    parallel_tail[0] = 0
    for color in range(_DIRECT_COLOR_COUNT, 2 * _DIRECT_COLOR_COUNT):
        if color_counts[color] > _FUSED_THREADS:
            parallel_tail[0] = 1


# Exact-color groups have disjoint dynamic endpoints. Their packed order is
# unobservable; overflow retains its separate, ascending serial order.
@wp.kernel(
    enable_backward=False,
    module="unique",
    module_options={"deterministic": wp.DeterministicMode.NOT_GUARANTEED},
)
def scatter_pair_groups_by_color(
    color_count: int,
    group_color: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    color_cursors: wp.array[wp.int32],
    group_list: wp.array[wp.int32],
    pair_partner: wp.array[wp.int32],
):
    """Pack exact-color groups; within-color order is irrelevant.

    Also record each dynamic body's partner per color, the next build's classes.
    """
    group = wp.tid()
    color = group_color[group]
    if color >= 0:
        slot = wp.atomic_add(color_cursors, color, 1)
        group_list[slot] = group
        body0 = group_body0[group]
        body1 = group_body1[group]
        if body0 >= 0 and body_inv_mass[body0] > 0.0:
            pair_partner[body0 * color_count + color] = body1
        if body1 >= 0 and body_inv_mass[body1] > 0.0:
            pair_partner[body1 * color_count + color] = body0


@wp.kernel
def accumulate_constraint_contact_forces_per_body(
    color_group: wp.array[wp.int32],
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
    contact_normal: wp.array[wp.vec3],
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
    body = _body_id_for_color_slot(slot, color_group, dynamic_color_starts, active_dynamic_color)
    if body < 0 or body_inv_mass[body] <= 0.0 or body_uses_pair_contacts[body] == 0:
        return

    force_sum = wp.vec3(0.0)
    torque_sum = wp.vec3(0.0)
    center = wp.transform_point(body_q[body], body_com[body])
    count = body_contact_counts[body]
    overflow = count > body_contact_buffer_pre_alloc
    if overflow:
        # Pair multipliers cover all contacts, including those absent from a
        # truncated body list. Gather them in contact order without atomics.
        count = wp.min(contact_count[0], contact_lambda.shape[0])
    for slot in range(count):
        contact = slot
        if not overflow:
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
        offset = wp.vec3(0.0)
        force = wp.vec3(0.0)
        if body == body0:
            point = contact_surface_point(body_q[body], contact_point0[contact], contact_offset0[contact])
            offset = wp.transform_vector(body_q[body], contact_offset0[contact])
            force = -reaction
        elif body == body1:
            point = contact_surface_point(body_q[body], contact_point1[contact], contact_offset1[contact])
            offset = wp.transform_vector(body_q[body], contact_offset1[contact])
            force = reaction
        else:
            continue
        force_sum += force
        # Normal rows use skeleton points; only friction uses the surface offset.
        normal = contact_normal[contact]
        normal_force = normal * wp.dot(normal, force)
        torque_sum += wp.cross(point - center, force) - wp.cross(offset, normal_force)

    # One thread owns each body. Use explicit assignment so Warp does not
    # lower the ordered writes to unnecessary floating-point atomics.
    body_forces[body] = body_forces[body] + force_sum
    body_torques[body] = body_torques[body] + torque_sum


@wp.func
def _apply_constraint_pose_increment(
    pose: wp.transform,
    center: wp.vec3,
    translation: wp.vec3,
    turn: wp.vec3,
    com: wp.vec3,
):
    """Move a pose by a translation and a small turn about its center of mass."""
    q = wp.transform_get_rotation(pose)
    half_turn = 0.5 * turn
    # Normalizing the product also removes the increment quaternion's scale.
    dq = wp.quat(half_turn[0], half_turn[1], half_turn[2], 1.0)
    q_new = wp.normalize(wp.mul(dq, q))
    return wp.transform(center + translation - wp.quat_rotate(q_new, com), q_new)


@wp.kernel
def pack_constraint_contact_geometry(
    contact_count: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    contact_constants: wp.array[ContactConstants],
):
    """Pack each contact's geometry into its record; it changes only with the contact set."""
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    contact_constants[contact].normal = contact_normal[contact]
    contact_constants[contact].point0 = contact_point0[contact]
    contact_constants[contact].point1 = contact_point1[contact]
    contact_constants[contact].anchor0 = contact_point0[contact] + contact_offset0[contact]
    contact_constants[contact].anchor1 = contact_point1[contact] + contact_offset1[contact]
    contact_constants[contact].margin = contact_margin0[contact] + contact_margin1[contact]


@wp.kernel
def prepare_constraint_contact_constants(
    dt: float,
    contact_damping_bidirectional: bool,
    body_q_prev: wp.array[wp.transform],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_surface_velocity: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    contact_constants: wp.array[ContactConstants],
):
    """Refresh each contact's material and step-start terms in its record.

    They stay fixed between collision refreshes within a step. The geometry
    was packed with the manifolds, and the inertial responses depend on the
    current poses and are evaluated by the solve.
    """
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    shape0 = contact_shape0[contact]
    shape1 = contact_shape1[contact]
    body0 = shape_body[shape0] if shape0 >= 0 else -1
    body1 = shape_body[shape1] if shape1 >= 0 else -1
    geometry = contact_constants[contact]
    # Keep the two previous anchors separate to preserve the slip subtraction order.
    point0_prev = geometry.point0
    point1_prev = geometry.point1
    anchor0_prev = geometry.anchor0
    anchor1_prev = geometry.anchor1
    if body0 >= 0:
        prev_pose0 = body_q_prev[body0]
        point0_prev = wp.transform_point(prev_pose0, point0_prev)
        anchor0_prev = wp.transform_point(prev_pose0, anchor0_prev)
    if body1 >= 0:
        prev_pose1 = body_q_prev[body1]
        point1_prev = wp.transform_point(prev_pose1, point1_prev)
        anchor1_prev = wp.transform_point(prev_pose1, anchor1_prev)
    compliance = float(0.0)
    damping_fraction = float(0.0)
    stiffness = contact_material_ke[contact]
    elastic_compliance = 1.0 / stiffness if stiffness > 0.0 else -1.0
    damping_rate = contact_material_kd[contact] / dt
    effective_stiffness = stiffness + damping_rate
    if effective_stiffness > 0.0:
        compliance = 1.0 / effective_stiffness
        damping_fraction = damping_rate / effective_stiffness

    # Damping uses step-start poses, not C0: collision refreshes may rebase
    # the ALM contact frame at a predicted or intermediate iterate.
    contact_constants[contact].previous_penetration = -contact_surface_separation(
        point0_prev, point1_prev, geometry.normal, geometry.margin, 0.0
    )
    contact_constants[contact].friction = contact_material_mu[contact]
    contact_constants[contact].anchor0_prev = anchor0_prev
    contact_constants[contact].anchor1_prev = anchor1_prev
    # Match the collision pipeline's relative surface-velocity convention.
    contact_constants[contact].surface_displacement = (
        dt * contact_surface_velocity[contact] if contact_surface_velocity else wp.vec3(0.0)
    )
    contact_constants[contact].compliance = compliance if effective_stiffness > 0.0 else -1.0
    contact_constants[contact].damping_fraction = damping_fraction
    contact_constants[contact].elastic_compliance = elastic_compliance
    contact_constants[contact].damping_bidirectional = contact_damping_bidirectional


@wp.func
def _project_sliding_friction(W00: float, W01: float, W11: float, c0: float, c1: float, friction_limit: float):
    """Project an exterior tangent onto a positive-radius Coulomb disk in an SPD metric."""
    rhs0 = W00 * c0 + W01 * c1
    rhs1 = W01 * c0 + W11 * c1
    # Normalize the SPD metric before comparing the squared root
    # residual. Its positive determinant can be multiplied through,
    # avoiding a reciprocal in every bisection step.
    inverse_scale = 1.0 / wp.max(W00, W11)
    a = W00 * inverse_scale
    b = W01 * inverse_scale
    d = W11 * inverse_scale
    u = rhs0 * (inverse_scale / friction_limit)
    v = rhs1 * (inverse_scale / friction_limit)
    # Keep all coefficients bounded, including when the friction radius tends
    # to zero. Otherwise squaring the determinant can overflow during sliding.
    inverse_bound = 1.0 / wp.max(1.0, wp.max(wp.abs(u), wp.abs(v)))
    a *= inverse_bound
    b *= inverse_bound
    d *= inverse_bound
    u *= inverse_bound
    v *= inverse_bound
    lower = float(0.0)
    upper = wp.sqrt(u * u + v * v)
    for _root_iter in range(16):
        eta = 0.5 * (lower + upper)
        d0 = a + eta
        d1 = d + eta
        determinant = d0 * d1 - b * b
        x0 = d1 * u - b * v
        x1 = d0 * v - b * u
        if x0 * x0 + x1 * x1 > determinant * determinant:
            lower = eta
        else:
            upper = eta
    d0 = a + upper
    d1 = d + upper
    scale = friction_limit / (d0 * d1 - b * b)
    return wp.vec2((d1 * u - b * v) * scale, (d0 * v - b * u) * scale)


@wp.func
def _solve_constraint_contact_group(
    group: int,
    dt: float,
    group_start: wp.array[wp.int32],
    group_end: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    contact_constants: wp.array[ContactConstants],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    contact_lambda: wp.array[wp.vec3],
):
    """Solve one body-pair manifold in the free-body inertial metric.

    Normal compliance uses the implicit spring-damper relation, with the damper
    active only in compression unless bidirectional damping is enabled.
    Tangential force minimizes the local
    two-dimensional quadratic over the Coulomb disk using a bounded root solve.

    Each endpoint moves by a translation and a small turn about its center of
    mass, accumulated over the rows and written to its pose once. Every row is
    linearized where it is solved: its arms follow the accumulated turn to
    first order, and the frames and world inverse inertias are those of the
    poses on entry.

    The contacts form one serial chain through the two endpoints, so each
    contact is one packed record read before the dependent arithmetic, and
    endpoint data is read once per manifold.
    """
    body0 = group_body0[group]
    body1 = group_body1[group]
    pose0 = wp.transform_identity()
    pose1 = wp.transform_identity()
    dynamic0 = False
    dynamic1 = False
    com0 = wp.vec3(0.0)
    com1 = wp.vec3(0.0)
    center0 = wp.vec3(0.0)
    center1 = wp.vec3(0.0)
    frame0 = wp.identity(n=3, dtype=float)
    frame1 = wp.identity(n=3, dtype=float)
    mobility0 = float(0.0)
    mobility1 = float(0.0)
    inertia0 = wp.mat33(0.0)
    inertia1 = wp.mat33(0.0)
    dt_sq = dt * dt
    if body0 >= 0:
        pose0 = body_q[body0]
        com0 = body_com[body0]
        frame0 = wp.quat_to_matrix(wp.transform_get_rotation(pose0))
        center0 = wp.transform_point(pose0, com0)
        dynamic0 = body_inv_mass[body0] > 0.0
        if dynamic0:
            mobility0 = dt_sq * body_inv_mass[body0]
            inertia0 = dt_sq * frame0 * body_inv_inertia[body0] * wp.transpose(frame0)
    if body1 >= 0:
        pose1 = body_q[body1]
        com1 = body_com[body1]
        frame1 = wp.quat_to_matrix(wp.transform_get_rotation(pose1))
        center1 = wp.transform_point(pose1, com1)
        dynamic1 = body_inv_mass[body1] > 0.0
        if dynamic1:
            mobility1 = dt_sq * body_inv_mass[body1]
            inertia1 = dt_sq * frame1 * body_inv_inertia[body1] * wp.transpose(frame1)
    mobility = mobility0 + mobility1
    center_offset = center1 - center0
    translation0 = wp.vec3(0.0)
    translation1 = wp.vec3(0.0)
    turn0 = wp.vec3(0.0)
    turn1 = wp.vec3(0.0)

    for contact in range(group_start[group], group_end[group]):
        constants = contact_constants[contact]
        old_lambda = contact_lambda[contact]
        n = constants.normal
        mu = constants.friction

        # A unit force along d on endpoint 1 and its reaction on endpoint 0 turn
        # them by inertia * cross(arm, d), with opposite signs.
        arm0 = frame0 * (constants.point0 - com0)
        arm1 = frame1 * (constants.point1 - com1)
        arm0 += wp.cross(turn0, arm0)
        arm1 += wp.cross(turn1, arm1)
        offset = center_offset + (translation1 - translation0)
        penetration = constants.margin - wp.dot(n, offset + (arm1 - arm0))
        lever0_n = wp.cross(arm0, n)
        lever1_n = wp.cross(arm1, n)
        spin0_n = inertia0 * lever0_n
        spin1_n = inertia1 * lever1_n
        normal_support = mobility + wp.dot(lever0_n, spin0_n) + wp.dot(lever1_n, spin1_n)

        old_normal = wp.max(wp.dot(old_lambda, n), 0.0)
        old_tangent = old_lambda - n * wp.dot(old_lambda, n)
        compliance = constants.compliance
        damping_fraction = constants.damping_fraction

        # Evaluate the two tangential rows at the same linearization point as
        # the normal row, then apply their combined 3-DOF contact increment.
        # They do not depend on the new normal force; only the friction limit does.
        anchor_arm0 = frame0 * (constants.anchor0 - com0)
        anchor_arm1 = frame1 * (constants.anchor1 - com1)
        anchor_arm0 += wp.cross(turn0, anchor_arm0)
        anchor_arm1 += wp.cross(turn1, anchor_arm1)
        anchor0 = center0 + translation0 + anchor_arm0
        anchor1 = center1 + translation1 + anchor_arm1
        relative_motion = (
            (anchor1 - constants.anchor1_prev) - (anchor0 - constants.anchor0_prev) + constants.surface_displacement
        )

        slip = relative_motion - n * wp.dot(n, relative_motion)
        tangent_drive = -slip

        t0, t1 = orthonormal_basis(n)
        lever0_t0 = wp.cross(anchor_arm0, t0)
        lever1_t0 = wp.cross(anchor_arm1, t0)
        lever0_t1 = wp.cross(anchor_arm0, t1)
        lever1_t1 = wp.cross(anchor_arm1, t1)
        spin0_t0 = inertia0 * lever0_t0
        spin1_t0 = inertia1 * lever1_t0
        spin0_t1 = inertia0 * lever0_t1
        spin1_t1 = inertia1 * lever1_t1
        W00 = mobility + wp.dot(lever0_t0, spin0_t0) + wp.dot(lever1_t0, spin1_t0)
        W11 = mobility + wp.dot(lever0_t1, spin0_t1) + wp.dot(lever1_t1, spin1_t1)
        W01 = wp.dot(lever0_t0, spin0_t1) + wp.dot(lever1_t0, spin1_t1)
        determinant = W00 * W11 - W01 * W01
        support_product = W00 * W11
        candidate_tangent = wp.vec3(0.0)
        if W00 > 0.0 and W11 > 0.0 and determinant > _TANGENT_EPS * support_product:
            rhs0 = wp.dot(tangent_drive, t0)
            rhs1 = wp.dot(tangent_drive, t1)
            delta0 = (W11 * rhs0 - W01 * rhs1) / determinant
            delta1 = (W00 * rhs1 - W01 * rhs0) / determinant
            candidate_tangent = old_tangent + t0 * delta0 + t1 * delta1

        normal_residual = penetration - damping_fraction * constants.previous_penetration - compliance * old_normal
        denominator = normal_support + compliance
        compliant_valid = denominator > 0.0 and compliance >= 0.0
        new_normal = wp.where(compliant_valid, wp.max(old_normal + normal_residual / denominator, 0.0), 0.0)
        # Select the piecewise damper from the solved displacement. Testing
        # only the current pose can choose the wrong branch across reversal.
        # Both candidates are evaluated; the selection matches the branches.
        solved_penetration = penetration - normal_support * (new_normal - old_normal)
        elastic_compliance = constants.elastic_compliance
        elastic_denominator = normal_support + elastic_compliance
        elastic_valid = elastic_compliance >= 0.0 and elastic_denominator > 0.0
        elastic_normal = wp.where(
            elastic_valid,
            wp.max(old_normal + (penetration - elastic_compliance * old_normal) / elastic_denominator, 0.0),
            0.0,
        )
        opening = not constants.damping_bidirectional and damping_fraction > 0.0
        new_normal = wp.where(
            opening and solved_penetration <= constants.previous_penetration, elastic_normal, new_normal
        )
        delta_normal = new_normal - old_normal

        friction_limit = mu * new_normal
        candidate_length = wp.length(candidate_tangent)
        if candidate_length > friction_limit:
            if friction_limit > _EPS:
                # Minimize the local quadratic in the W metric on the Coulomb disk.
                c0 = wp.dot(candidate_tangent, t0)
                c1 = wp.dot(candidate_tangent, t1)
                projected = _project_sliding_friction(W00, W01, W11, c0, c1, friction_limit)
                candidate_tangent = t0 * projected[0] + t1 * projected[1]
            else:
                candidate_tangent = wp.vec3(0.0)
        delta_tangent = candidate_tangent - old_tangent
        delta_t0 = wp.dot(delta_tangent, t0)
        delta_t1 = wp.dot(delta_tangent, t1)
        impulse = n * delta_normal + t0 * delta_t0 + t1 * delta_t1
        translation0 -= mobility0 * impulse
        translation1 += mobility1 * impulse
        turn0 -= spin0_n * delta_normal + spin0_t0 * delta_t0 + spin0_t1 * delta_t1
        turn1 += spin1_n * delta_normal + spin1_t0 * delta_t0 + spin1_t1 * delta_t1
        contact_lambda[contact] = n * new_normal + candidate_tangent

    if dynamic0:
        body_q[body0] = _apply_constraint_pose_increment(pose0, center0, translation0, turn0, com0)
    if dynamic1:
        body_q[body1] = _apply_constraint_pose_increment(pose1, center1, translation1, turn1, com1)


@wp.kernel(grid_stride=False)
def solve_constraint_contact_color(
    launch_dim: int,
    active_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    group_list: wp.array[wp.int32],
    dt: float,
    group_start: wp.array[wp.int32],
    group_end: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    contact_constants: wp.array[ContactConstants],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
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
            group_start,
            group_end,
            group_body0,
            group_body1,
            contact_constants,
            body_com,
            body_inv_mass,
            body_inv_inertia,
            body_q,
            contact_lambda,
        )
        slot += launch_dim


@wp.kernel
def build_pair_overflow_waves(
    overflow_count: wp.array[wp.int32],
    group_list: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_level: wp.array[wp.int32],
    group_level: wp.array[wp.int32],
    wave_count: wp.array[wp.int32],
    wave_offsets: wp.array[wp.int32],
    wave_groups: wp.array[wp.int32],
):
    """Pack disjoint overflow waves, preserving serial order at every dynamic body."""
    count = overflow_count[0]
    wave_count[0] = 0
    if count == 0:
        return
    start = group_list.shape[0] - count
    # Avoid repeated endpoint clears on dense overflow graphs.
    if body_level.shape[0] <= 2 * count:
        for body in range(body_level.shape[0]):
            body_level[body] = -1
    else:
        for slot in range(count):
            group = group_list[start + slot]
            body0 = group_body0[group]
            body1 = group_body1[group]
            if body0 >= 0:
                body_level[body0] = -1
            if body1 >= 0:
                body_level[body1] = -1
    for slot in range(count + 1):
        wave_offsets[slot] = 0
    levels = int(0)
    for slot in range(count):
        group = group_list[start + slot]
        body0 = group_body0[group]
        body1 = group_body1[group]
        level = int(0)
        if body0 >= 0 and body_inv_mass[body0] > 0.0:
            level = wp.max(level, body_level[body0] + 1)
        if body1 >= 0 and body_inv_mass[body1] > 0.0:
            level = wp.max(level, body_level[body1] + 1)
        if body0 >= 0 and body_inv_mass[body0] > 0.0:
            body_level[body0] = level
        if body1 >= 0 and body_inv_mass[body1] > 0.0:
            body_level[body1] = level
        group_level[slot] = level
        wave_offsets[level + 1] += 1
        levels = wp.max(levels, level + 1)
    for level in range(levels):
        wave_offsets[level + 1] += wave_offsets[level]
    # Reverse stable scatter consumes end offsets into starts. Shift them back
    # afterwards, avoiding a second cursor array.
    for index in range(count):
        slot = count - 1 - index
        level = group_level[slot]
        wave_offsets[level + 1] -= 1
        wave_groups[wave_offsets[level + 1]] = group_list[start + slot]
    for level in range(levels):
        wave_offsets[level] = wave_offsets[level + 1]
    wave_offsets[levels] = count
    wave_count[0] = levels


@wp.kernel
def solve_constraint_contact_colors_fused(
    sweeps: int,
    first_color: int,
    color_count: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    wave_count: wp.array[wp.int32],
    wave_offsets: wp.array[wp.int32],
    wave_groups: wp.array[wp.int32],
    group_list: wp.array[wp.int32],
    dt: float,
    group_start: wp.array[wp.int32],
    group_end: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    contact_constants: wp.array[ContactConstants],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    contact_lambda: wp.array[wp.vec3],
):
    """Run ``sweeps`` passes over exact colors and dependency-preserving overflow waves on one block."""
    lane = wp.tid()
    # Overflow waves are built before this launch and stay fixed during it.
    color_end = color_count + wave_count[0]
    for _sweep in range(sweeps):
        color = first_color
        while color < color_end:
            count = int(0)
            offset = int(0)
            slot = lane
            stride = _FUSED_THREADS
            if color >= color_count:
                level = color - color_count
                offset = wave_offsets[level]
                count = wave_offsets[level + 1] - offset
            else:
                count = color_counts[color]
                # The greedy palette is contiguous; skip empty colors, but not overflow.
                if count == 0:
                    color = color_count
                    continue
                offset = color_offsets[color]
            while slot < count:
                group = int(0)
                if color >= color_count:
                    group = wave_groups[offset + slot]
                else:
                    group = group_list[offset + slot]
                _solve_constraint_contact_group(
                    group,
                    dt,
                    group_start,
                    group_end,
                    group_body0,
                    group_body1,
                    contact_constants,
                    body_com,
                    body_inv_mass,
                    body_inv_inertia,
                    body_q,
                    contact_lambda,
                )
                slot += stride
            _sync_cuda_block()
            color += 1


@wp.kernel
def solve_constraint_contacts_serial(
    sweeps: int,
    group_capacity: int,
    group_color: wp.array[wp.int32],
    dt: float,
    group_start: wp.array[wp.int32],
    group_end: wp.array[wp.int32],
    group_body0: wp.array[wp.int32],
    group_body1: wp.array[wp.int32],
    contact_constants: wp.array[ContactConstants],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    contact_lambda: wp.array[wp.vec3],
):
    """Run ``sweeps`` passes over all active groups in contact order on the CPU."""
    if wp.tid() != 0:
        return
    for _sweep in range(sweeps):
        for group in range(group_capacity):
            if group_color[group] >= -1:
                _solve_constraint_contact_group(
                    group,
                    dt,
                    group_start,
                    group_end,
                    group_body0,
                    group_body1,
                    contact_constants,
                    body_com,
                    body_inv_mass,
                    body_inv_inertia,
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
        self.body_color_owner = wp.full(body_count, -1, dtype=wp.int64, device=self.device)
        self.pair_partner = wp.full(body_count * _COLOR_COUNT, -2, dtype=wp.int32, device=self.device)
        self.color_counts = wp.zeros(_COLOR_COUNT, dtype=wp.int32, device=self.device)
        self.color_offsets = wp.zeros(_COLOR_COUNT + 1, dtype=wp.int32, device=self.device)
        self.color_cursors = wp.zeros(_COLOR_COUNT, dtype=wp.int32, device=self.device)
        self.group_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.overflow_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.parallel_tail = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.build_index = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.overflow_wave_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.overflow_body_level = wp.zeros(body_count, dtype=wp.int32, device=self.device)

    def resize(self, capacity: int) -> None:
        if capacity <= self.capacity:
            return
        self.capacity = capacity
        self.group_start = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.group_end = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.group_body0 = wp.full(capacity, -1, dtype=wp.int32, device=self.device)
        self.group_body1 = wp.full(capacity, -1, dtype=wp.int32, device=self.device)
        self.group_color = wp.full(capacity, -2, dtype=wp.int32, device=self.device)
        self.group_class = wp.full(capacity, -1, dtype=wp.int32, device=self.device)
        self.group_list = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.overflow_group_level = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.overflow_wave_offsets = wp.zeros(capacity + 1, dtype=wp.int32, device=self.device)
        self.overflow_wave_groups = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.constants = wp.zeros(capacity, dtype=ContactConstants, device=self.device)

    def pack_geometry(self, contacts) -> None:
        """Pack each contact's geometry for the solve; required whenever the contact set changes."""
        capacity = contacts.rigid_contact_max
        if capacity <= 0:
            return
        self.resize(capacity)
        wp.launch(
            kernel=pack_constraint_contact_geometry,
            dim=capacity,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
            ],
            outputs=[self.constants],
            device=self.device,
            block_dim=_select_rigid_block_dim(capacity, self.device),
        )

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
                    _COLOR_COUNT,
                    self.pair_partner,
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
                    self.group_class,
                    self.group_count,
                ],
                device=self.device,
            )
            self.pair_partner.fill_(-2)
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
                            round_index,
                            self.build_index,
                            self.group_color,
                            self.group_class,
                            self.group_body0,
                            self.group_body1,
                            self.body_inv_mass,
                        ],
                        outputs=[self.body_color_owner],
                        device=self.device,
                    )
                    wp.launch(
                        kernel=commit_pair_color_parallel,
                        dim=capacity,
                        inputs=[
                            _COLOR_COUNT,
                            round_index,
                            self.build_index,
                            self.group_class,
                            self.group_body0,
                            self.group_body1,
                            self.body_inv_mass,
                            self.body_color_owner,
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
                dim=_TAIL_SCAN_THREADS,
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
                    self.group_list,
                    self.parallel_tail,
                    self.build_index,
                ],
                device=self.device,
                block_dim=_TAIL_SCAN_THREADS,
            )
            wp.launch(
                kernel=scatter_pair_groups_by_color,
                dim=capacity,
                inputs=[_COLOR_COUNT, self.group_color, self.group_body0, self.group_body1, self.body_inv_mass],
                outputs=[self.color_cursors, self.group_list, self.pair_partner],
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

        if self.device.is_cuda:
            wp.launch(
                build_pair_overflow_waves,
                dim=1,
                inputs=[self.overflow_count, self.group_list, self.group_body0, self.group_body1, self.body_inv_mass],
                outputs=[
                    self.overflow_body_level,
                    self.overflow_group_level,
                    self.overflow_wave_count,
                    self.overflow_wave_offsets,
                    self.overflow_wave_groups,
                ],
                device=self.device,
                block_dim=1,
            )

    def _direct_color_count(self) -> int:
        """Return how many leading colors get their own launch on this device."""
        if not self.device.is_cuda:
            return 0
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
        return 0 if self.pair_body_count <= min_bodies_per_sm * self.device.sm_count else _DIRECT_COLOR_COUNT

    def sweeps_on_one_block(self) -> bool:
        """Return whether one launch covers every color, so that it can also run every sweep."""
        return self._direct_color_count() == 0

    def solve(
        self,
        body_q: wp.array,
        body_q_prev: wp.array,
        contacts,
        dt: float,
        contact_material_ke: wp.array,
        contact_material_kd: wp.array,
        contact_material_mu: wp.array,
        contact_lambda: wp.array,
        parallel_tail: bool | None = None,
        refresh_constants: bool = True,
        contact_damping_bidirectional: bool = False,
        sweeps: int = 1,
    ) -> None:
        """Apply ``sweeps`` sweeps; an optional enclosing graph branch selects the tail.

        More than one sweep per call requires :meth:`sweeps_on_one_block`.
        """
        capacity = contacts.rigid_contact_max
        if capacity <= 0:
            return
        model = self.model
        if refresh_constants:
            wp.launch(
                kernel=prepare_constraint_contact_constants,
                dim=capacity,
                inputs=[
                    dt,
                    contact_damping_bidirectional,
                    body_q_prev,
                    contact_material_ke,
                    contact_material_kd,
                    contact_material_mu,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_surface_velocity,
                    model.shape_body,
                ],
                outputs=[self.constants],
                device=self.device,
                block_dim=_select_rigid_block_dim(capacity, self.device),
            )

        contact_inputs = [
            dt,
            self.group_start,
            self.group_end,
            self.group_body0,
            self.group_body1,
            self.constants,
            model.body_com,
            self.body_inv_mass,
            self.body_inv_inertia,
        ]
        # One exact color contains at most one group per pair-routed body.
        launch_dim = max(1, min(capacity, max(self.pair_body_count, 128)))
        block_dim = _select_rigid_block_dim(launch_dim, self.device)
        if not self.device.is_cuda:
            wp.launch(
                kernel=solve_constraint_contacts_serial,
                dim=1,
                inputs=[sweeps, capacity, self.group_color, *contact_inputs],
                outputs=[body_q, contact_lambda],
                device=self.device,
                block_dim=1,
            )
            return

        direct_color_count = self._direct_color_count()
        if sweeps != 1 and direct_color_count:
            raise ValueError("Several sweeps per call require a one-block schedule.")

        def solve_direct(first: int, last: int) -> None:
            for color in range(first, last):
                wp.launch(
                    kernel=solve_constraint_contact_color,
                    dim=launch_dim,
                    inputs=[
                        launch_dim,
                        color,
                        self.color_counts,
                        self.color_offsets,
                        self.group_list,
                        *contact_inputs,
                    ],
                    outputs=[body_q, contact_lambda],
                    device=self.device,
                    block_dim=block_dim,
                )

        def solve_tail(first_color: int) -> None:
            # Direct passes cover at most 16 of 30 colors. The fused kernel
            # always handles the remaining colors and the ordered overflow waves.
            wp.launch(
                kernel=solve_constraint_contact_colors_fused,
                dim=_FUSED_THREADS,
                inputs=[
                    sweeps,
                    first_color,
                    _COLOR_COUNT,
                    self.color_counts,
                    self.color_offsets,
                    self.overflow_wave_count,
                    self.overflow_wave_offsets,
                    self.overflow_wave_groups,
                    self.group_list,
                    *contact_inputs,
                ],
                outputs=[body_q, contact_lambda],
                device=self.device,
                block_dim=_FUSED_THREADS,
                max_blocks=1,
            )

        if direct_color_count and parallel_tail is not None:
            direct_color_count *= 2 if parallel_tail else 1
        solve_direct(0, direct_color_count)
        if (
            parallel_tail is None
            and direct_color_count
            and self.device.is_capturing
            and wp.is_conditional_graph_supported()
        ):

            def solve_parallel_tail() -> None:
                expanded_count = 2 * _DIRECT_COLOR_COUNT
                solve_direct(direct_color_count, expanded_count)
                solve_tail(expanded_count)

            wp.capture_if(
                self.parallel_tail,
                on_true=solve_parallel_tail,
                on_false=lambda: solve_tail(direct_color_count),
            )
        else:
            solve_tail(direct_color_count)
