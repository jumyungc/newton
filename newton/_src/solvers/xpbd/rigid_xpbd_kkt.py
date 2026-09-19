# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Global compliance-KKT correction for XPBD rigid-joint islands.

The local XPBD joint kernel resolves every joint independently and combines
incident corrections with atomics. This module linearizes those same scalar
joint rows together and reuses VBD's graph-capturable path, tree, and closure
factorizations to communicate a correction across each complete joint island.

Active static-contact normals contribute body-local terms to the global system.
Friction and contacts between dynamic bodies remain local. A global pass is
scheduled between local iterations to relinearize unilateral constraints afterward.
"""

from __future__ import annotations

import warp as wp

from ...math import vec_max, vec_min
from ...sim import JointType
from ...sim.contacts import contact_surface_separation
from ..vbd.rigid_vbd_kkt import (
    StructuralGraphKKT,
    _ClosedTreeBucket,
    _inverse_spatial_robust,
    _PathBucket,
    _scale_spatial_vector,
    accumulate_body_quadratic_merit,
    accumulate_joint_quadratic_merit,
    assemble_joint_path_system,
    compute_path_correction,
    minimize_quadratic_step,
    suppress_nonfinite_correction,
)
from .kernels import update_joint_axis_limits, update_joint_axis_weighted_target

wp.set_module_options({"enable_backward": False})

_SUPPORTED_JOINT_TYPES = {
    int(JointType.BALL),
    int(JointType.FIXED),
    int(JointType.REVOLUTE),
    int(JointType.PRISMATIC),
    int(JointType.D6),
    int(JointType.DISTANCE),
}
# A nonzero constraint pivot is required when leaf elimination starts at a
# hard world-attachment row. Open paths use a positive Schur system directly
# and therefore do not need this numerical compliance.
_TREE_COMPLIANCE_FLOOR = 1.0e-6


@wp.func
def _static_contact_row(
    tid: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_slot: wp.array[int],
    shape_body: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape0: wp.array[int],
    shape1: wp.array[int],
):
    a, b = int(-1), int(-1)
    if shape0[tid] >= 0:
        a = shape_body[shape0[tid]]
    if shape1[tid] >= 0:
        b = shape_body[shape1[tid]]
    dynamic_a, dynamic_b = bool(False), bool(False)
    if a >= 0:
        dynamic_a = body_inv_mass[a] > 0.0
    if b >= 0:
        dynamic_b = body_inv_mass[b] > 0.0
    if dynamic_a == dynamic_b:
        return -1, 0.0, wp.spatial_vector(), wp.spatial_vector()
    body = b
    if dynamic_a:
        body = a
    slot = body_slot[body]
    if slot < 0:
        return -1, 0.0, wp.spatial_vector(), wp.spatial_vector()
    pose_a, pose_b = wp.transform_identity(), wp.transform_identity()
    com_a, com_b = wp.vec3(0.0), wp.vec3(0.0)
    if a >= 0:
        pose_a = body_q[a]
        com_a = wp.transform_point(pose_a, body_com[a])
    if b >= 0:
        pose_b = body_q[b]
        com_b = wp.transform_point(pose_b, body_com[b])
    pa, pb = wp.transform_point(pose_a, point0[tid]), wp.transform_point(pose_b, point1[tid])
    n = normal[tid]
    gap = contact_surface_separation(pa, pb, n, margin0[tid], margin1[tid])
    ja = wp.spatial_vector(-n, -wp.cross(pa - com_a, n))
    jb = wp.spatial_vector(n, wp.cross(pb - com_b, n))
    if dynamic_a:
        return slot, gap, ja, ja
    return slot, gap, jb, ja


@wp.kernel
def add_static_contact_terms(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_slot: wp.array[int],
    shape_body: wp.array[int],
    count: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape0: wp.array[int],
    shape1: wp.array[int],
    matrix: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Condense frozen active static-normal rows into the body metric.

    For a regularized hard row ``g + J dx + c lambda = 0``, elimination
    adds ``J.T J / c`` and ``-J.T g / c``. Use the existing hard-row
    numerical compliance, not a material/contact stiffness tuning parameter.
    Static endpoints make this condensation body-local and topology preserving.
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    slot, gap, jacobian, _ = _static_contact_row(
        tid,
        body_q,
        body_com,
        body_inv_mass,
        body_slot,
        shape_body,
        point0,
        point1,
        normal,
        margin0,
        margin1,
        shape0,
        shape1,
    )
    if slot >= 0 and gap < 0.0:
        weight = 1.0 / _TREE_COMPLIANCE_FLOOR
        wp.atomic_add(matrix, slot, weight * wp.outer(jacobian, jacobian))
        wp.atomic_add(rhs, slot, -weight * gap * jacobian)


@wp.kernel
def accumulate_static_contact_impulse(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_slot: wp.array[int],
    shape_body: wp.array[int],
    count: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape0: wp.array[int],
    shape1: wp.array[int],
    body_island: wp.array[int],
    island_state: wp.array[int],
    island_step_scale: wp.array[float],
    correction: wp.array[wp.spatial_vector],
    relaxation: float,
    impulse: wp.array[wp.spatial_vector],
):
    """Report the condensed normal impulse with XPBD's shape-0 convention."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    slot, gap, jacobian, jacobian_a = _static_contact_row(
        tid,
        body_q,
        body_com,
        body_inv_mass,
        body_slot,
        shape_body,
        point0,
        point1,
        normal,
        margin0,
        margin1,
        shape0,
        shape1,
    )
    if slot >= 0 and gap < 0.0:
        if island_state[body_island[slot]] >= -1:
            multiplier = -(gap + wp.dot(jacobian, correction[slot])) / _TREE_COMPLIANCE_FLOOR
            scale = relaxation * island_step_scale[body_island[slot]]
            impulse[tid] = impulse[tid] + scale * multiplier * jacobian_a


@wp.func
def _set_spatial_row(matrix: wp.spatial_matrix, row: int, linear: wp.vec3, angular: wp.vec3):
    for column in range(3):
        matrix[row, column] = linear[column]
        matrix[row, column + 3] = angular[column]
    return matrix


@wp.func
def _set_spatial_diagonal(matrix: wp.spatial_matrix, row: int, value: float):
    matrix[row, row] = value
    return matrix


@wp.func
def _axis_controls(
    axis_start: int,
    target_start: int,
    axis_offset: int,
    axis_count: int,
    joint_axis: wp.array[wp.vec3],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
):
    limits = wp.spatial_vector()
    position = wp.spatial_vector()
    velocity = wp.spatial_vector()
    for local_axis in range(3):
        if local_axis < axis_count:
            dof = axis_start + axis_offset + local_axis
            target = target_start + axis_offset + local_axis
            axis = joint_axis[dof]
            if local_axis == 0:
                lower = axis * joint_limit_lower[dof]
                upper = axis * joint_limit_upper[dof]
                limits = wp.spatial_vector(vec_min(lower, upper), vec_max(lower, upper))
            else:
                limits = update_joint_axis_limits(axis, joint_limit_lower[dof], joint_limit_upper[dof], limits)
            stiffness = joint_target_ke[dof]
            damping = joint_target_kd[dof]
            if stiffness > 0.0:
                position = update_joint_axis_weighted_target(axis, joint_target_q[target], stiffness, position)
            if damping > 0.0:
                velocity = update_joint_axis_weighted_target(axis, joint_target_qd[dof], damping, velocity)

    position_value = wp.spatial_top(position)
    position_weight = wp.spatial_bottom(position)
    velocity_value = wp.spatial_top(velocity)
    velocity_weight = wp.spatial_bottom(velocity)
    for axis in range(3):
        if position_weight[axis] > 0.0:
            position_value[axis] = position_value[axis] / position_weight[axis]
        if velocity_weight[axis] > 0.0:
            velocity_value[axis] = velocity_value[axis] / velocity_weight[axis]

    return (
        wp.spatial_top(limits),
        wp.spatial_bottom(limits),
        position_value,
        position_weight,
        velocity_value,
        velocity_weight,
    )


@wp.func
def _scaled_row_terms(
    error: float,
    error_rate: float,
    compliance: float,
    damping: float,
    dt: float,
    hard_row_regularization: float,
):
    """Return symmetric row scale, residual, and compliance for one XPBD row."""
    gamma = compliance * damping
    scale = wp.sqrt(wp.max(1.0 + gamma / dt, 1.0e-12))
    residual = (error + gamma * error_rate) / scale
    return scale, residual, wp.max(compliance / dt, hard_row_regularization)


@wp.kernel
def build_xpbd_body_metric(
    body_ids: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    dt: float,
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_matrix: wp.array[wp.spatial_matrix],
    body_rhs: wp.array[wp.spatial_vector],
    island_state: wp.array[wp.int32],
):
    """Build the ``M / dt`` metric used by XPBD's impulse correction."""
    slot = wp.tid()
    wp.atomic_max(island_state, graph_body_island[slot], 1)
    body = body_ids[slot]
    inv_mass = body_inv_mass[body]
    if inv_mass <= 0.0:
        body_matrix[slot] = wp.spatial_matrix(0.0)
        body_rhs[slot] = wp.spatial_vector()
        return

    rotation = wp.quat_to_matrix(wp.transform_get_rotation(body_q[body]))
    inertia_world = rotation * body_inertia[body] * wp.transpose(rotation)
    metric = wp.spatial_matrix(0.0)
    mass_dt = 1.0 / (inv_mass * dt)
    for row in range(3):
        metric[row, row] = mass_dt
        for column in range(3):
            metric[row + 3, column + 3] = inertia_world[row, column] / dt
    body_matrix[slot] = metric
    body_rhs[slot] = wp.spatial_vector()


@wp.kernel
def build_xpbd_body_inverse(
    body_ids: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    reset_island_state: int,
    dt: float,
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    contact_metric: bool,
    body_matrix: wp.array[wp.spatial_matrix],
    body_rhs: wp.array[wp.spatial_vector],
    body_inverse: wp.array[wp.spatial_matrix],
    body_free: wp.array[wp.spatial_vector],
    island_state: wp.array[wp.int32],
):
    """Invert the contact metric, or build ``dt M^-1`` analytically."""
    slot = wp.tid()
    if reset_island_state != 0:
        wp.atomic_max(island_state, graph_body_island[slot], 1)
    body = body_ids[slot]
    inv_mass = body_inv_mass[body]
    if inv_mass <= 0.0:
        body_inverse[slot] = wp.spatial_matrix(0.0)
        body_free[slot] = wp.spatial_vector()
        return

    if contact_metric:
        inverse = _inverse_spatial_robust(body_matrix[slot])
        body_inverse[slot] = inverse
        body_free[slot] = inverse * body_rhs[slot]
        return

    rotation = wp.quat_to_matrix(wp.transform_get_rotation(body_q[body]))
    inverse_inertia_world = rotation * body_inv_inertia[body] * wp.transpose(rotation)
    inverse = wp.spatial_matrix(0.0)
    for row in range(3):
        inverse[row, row] = dt * inv_mass
        for column in range(3):
            inverse[row + 3, column + 3] = dt * inverse_inertia_world[row, column]
    body_inverse[slot] = inverse
    body_free[slot] = wp.spatial_vector()


@wp.kernel
def linearize_xpbd_joint_rows(
    joint_ids: wp.array[wp.int32],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_axis: wp.array[wp.vec3],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_linear_compliance: float,
    joint_angular_compliance: float,
    hard_row_regularization: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    dt: float,
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    compliance_out: wp.array[wp.spatial_matrix],
    residual_out: wp.array[wp.spatial_vector],
    row_active: wp.array[wp.int32],
    joint_jacobian_child: wp.array[wp.spatial_matrix],
):
    """Linearize one six-row XPBD joint block at the current iterate."""
    row = wp.tid()
    joint = joint_ids[row]
    kind = joint_type[joint]

    jp = wp.spatial_matrix(0.0)
    jc = wp.spatial_matrix(0.0)
    compliance = wp.identity(6, float)
    residual = wp.spatial_vector()
    active_count = int(0)

    supported = (
        kind == JointType.BALL
        or kind == JointType.FIXED
        or kind == JointType.REVOLUTE
        or kind == JointType.PRISMATIC
        or kind == JointType.D6
        or kind == JointType.DISTANCE
    )
    if not joint_enabled[joint] or not supported:
        jacobian_parent[row] = jp
        jacobian_child[row] = jc
        compliance_out[row] = compliance
        residual_out[row] = residual
        row_active[row] = 0
        joint_jacobian_child[joint] = jc
        return

    parent = joint_parent[joint]
    child = joint_child[joint]
    X_wp = joint_X_p[joint]
    parent_pose = X_wp
    parent_velocity = wp.vec3(0.0)
    parent_omega = wp.vec3(0.0)
    parent_com = wp.vec3(0.0)
    if parent >= 0:
        parent_pose = body_q[parent]
        X_wp = parent_pose * X_wp
        parent_velocity = wp.spatial_top(body_qd[parent])
        parent_omega = wp.spatial_bottom(body_qd[parent])
        parent_com = wp.transform_point(parent_pose, body_com[parent])

    child_pose = body_q[child]
    X_wc = child_pose * joint_X_c[joint]
    child_velocity = wp.spatial_top(body_qd[child])
    child_omega = wp.spatial_bottom(body_qd[child])
    child_com = wp.transform_point(child_pose, body_com[child])

    axis_start = joint_qd_start[joint]
    target_start = joint_target_q_start[joint]
    linear_count = joint_dof_dim[joint, 0]
    angular_count = joint_dof_dim[joint, 1]

    rel_pose = wp.transform_inverse(X_wp) * X_wc
    rel_position = wp.transform_get_translation(rel_pose)
    x_child = wp.transform_get_translation(X_wc)

    if kind == JointType.DISTANCE:
        # Distance joints have one radial row, not six Cartesian limits.
        # Match local XPBD's inactive interval and coincident-anchor fallback.
        x_parent = wp.transform_get_translation(X_wp)
        anchor_delta = x_child - x_parent
        distance = wp.length(anchor_delta)
        lower = joint_limit_lower[axis_start]
        upper = joint_limit_upper[axis_start]
        error = 0.0
        if lower >= 0.0 and distance < lower:
            error = distance - lower
        elif upper >= 0.0 and distance > upper:
            error = distance - upper
        if wp.abs(error) > 1.0e-9:
            direction = wp.transform_vector(X_wp, wp.vec3(1.0, 0.0, 0.0))
            if distance > 1.0e-9:
                direction = anchor_delta / distance
            elif wp.length_sq(child_com - parent_com) > 1.0e-18:
                direction = wp.normalize(child_com - parent_com)
            angular_p = -wp.cross(x_parent - parent_com, direction)
            angular_c = wp.cross(x_child - child_com, direction)
            error_rate = (
                wp.dot(direction, child_velocity - parent_velocity)
                + wp.dot(angular_p, parent_omega)
                + wp.dot(angular_c, child_omega)
            )
            row_compliance = joint_linear_compliance
            if joint_target_ke[axis_start] > 0.0:
                row_compliance = 1.0 / joint_target_ke[axis_start]
            scale, row_residual, row_compliance_dt = _scaled_row_terms(
                error,
                error_rate,
                row_compliance,
                joint_target_kd[axis_start],
                dt,
                hard_row_regularization,
            )
            jp = _set_spatial_row(jp, 0, -scale * direction, scale * angular_p)
            jc = _set_spatial_row(jc, 0, scale * direction, scale * angular_c)
            compliance = _set_spatial_diagonal(compliance, 0, row_compliance_dt)
            residual[0] = row_residual
            active_count = 1
        jacobian_parent[row] = jp
        jacobian_child[row] = jc
        compliance_out[row] = compliance
        residual_out[row] = residual
        row_active[row] = active_count
        joint_jacobian_child[joint] = jc
        return

    (
        linear_lower,
        linear_upper,
        linear_target,
        linear_stiffness,
        linear_target_velocity,
        linear_damping,
    ) = _axis_controls(
        axis_start,
        target_start,
        0,
        linear_count,
        joint_axis,
        joint_limit_lower,
        joint_limit_upper,
        joint_target_q,
        joint_target_qd,
        joint_target_ke,
        joint_target_kd,
    )

    admissible_position = rel_position
    for axis in range(3):
        if rel_position[axis] < linear_lower[axis]:
            admissible_position[axis] = linear_lower[axis]
        elif rel_position[axis] > linear_upper[axis]:
            admissible_position[axis] = linear_upper[axis]
        elif linear_stiffness[axis] > 0.0:
            admissible_position[axis] = wp.clamp(linear_target[axis], linear_lower[axis], linear_upper[axis])

    parent_frame = wp.quat_to_matrix(wp.transform_get_rotation(X_wp))
    parent_arm = wp.transform_point(X_wp, admissible_position) - parent_com
    child_arm = x_child - child_com
    for axis in range(3):
        value = rel_position[axis]
        lower = linear_lower[axis]
        upper = linear_upper[axis]
        error = 0.0
        row_compliance = joint_linear_compliance
        damping = 0.0
        active = lower == upper
        if value < lower:
            error = value - lower
            active = True
        elif value > upper:
            error = value - upper
            active = True
        elif linear_stiffness[axis] > 0.0:
            error = value - wp.clamp(linear_target[axis], lower, upper)
            row_compliance = 1.0 / linear_stiffness[axis]
            damping = linear_damping[axis]
            active = True
        elif linear_damping[axis] > 0.0:
            row_compliance = 1.0 / linear_damping[axis]
            damping = linear_damping[axis]
            active = True

        if active:
            linear_c = wp.vec3(parent_frame[0, axis], parent_frame[1, axis], parent_frame[2, axis])
            linear_p = -linear_c
            angular_p = -wp.cross(parent_arm, linear_c)
            angular_c = wp.cross(child_arm, linear_c)
            error_rate = (
                wp.dot(linear_p, parent_velocity)
                + wp.dot(linear_c, child_velocity)
                + wp.dot(angular_p, parent_omega)
                + wp.dot(angular_c, child_omega)
                - linear_target_velocity[axis]
            )
            scale, row_residual, row_compliance_dt = _scaled_row_terms(
                error,
                error_rate,
                row_compliance,
                damping,
                dt,
                hard_row_regularization,
            )
            jp = _set_spatial_row(jp, axis, scale * linear_p, scale * angular_p)
            jc = _set_spatial_row(jc, axis, scale * linear_c, scale * angular_c)
            compliance = _set_spatial_diagonal(compliance, axis, row_compliance_dt)
            residual[axis] = row_residual
            active_count += 1

    if kind == JointType.FIXED or kind == JointType.PRISMATIC or kind == JointType.REVOLUTE or kind == JointType.D6:
        q_parent = wp.transform_get_rotation(X_wp)
        q_child = wp.transform_get_rotation(X_wc)
        if wp.dot(q_parent, q_child) < 0.0:
            q_child = -q_child
        relative_q = wp.quat_inverse(q_parent) * q_child

        twist = wp.normalize(wp.quat(relative_q[0], 0.0, 0.0, relative_q[3]))
        swing = relative_q * wp.quat_inverse(twist)
        s = wp.sqrt(relative_q[0] * relative_q[0] + relative_q[3] * relative_q[3])
        inv_s = 1.0 / wp.max(s, 1.0e-9)
        inv_s_cubed = inv_s * inv_s * inv_s
        twist_w = wp.max(wp.abs(twist[3]), 1.0e-9)

        errors = wp.vec3(
            2.0 * wp.asin(wp.clamp(twist[0], -1.0, 1.0)),
            swing[1],
            swing[2],
        )
        grad_0 = wp.quat(
            inv_s - relative_q[0] * relative_q[0] * inv_s_cubed,
            0.0,
            0.0,
            -(relative_q[3] * relative_q[0]) * inv_s_cubed,
        )
        grad_1 = wp.quat(
            -relative_q[3] * (relative_q[3] * relative_q[2] + relative_q[0] * relative_q[1]) * inv_s_cubed,
            relative_q[3] * inv_s,
            -relative_q[0] * inv_s,
            relative_q[0] * (relative_q[3] * relative_q[2] + relative_q[0] * relative_q[1]) * inv_s_cubed,
        )
        grad_2 = wp.quat(
            relative_q[3] * (relative_q[3] * relative_q[1] - relative_q[0] * relative_q[2]) * inv_s_cubed,
            relative_q[0] * inv_s,
            relative_q[3] * inv_s,
            relative_q[0] * (relative_q[2] * relative_q[0] - relative_q[3] * relative_q[1]) * inv_s_cubed,
        )
        grad_0 = grad_0 * (2.0 / twist_w)

        swing_sq = swing[3] * swing[3]
        if swing_sq + 1.0e-4 < 1.0:
            denominator = wp.sqrt(1.0 - swing_sq)
            angle_scale = 2.0 * wp.acos(wp.clamp(swing[3], -1.0, 1.0)) / denominator
            errors[1] = errors[1] * angle_scale
            errors[2] = errors[2] * angle_scale
            grad_1 = grad_1 * angle_scale
            grad_2 = grad_2 * angle_scale

        grad_x = wp.vec3(grad_0[0], grad_1[0], grad_2[0])
        grad_y = wp.vec3(grad_0[1], grad_1[1], grad_2[1])
        grad_z = wp.vec3(grad_0[2], grad_1[2], grad_2[2])
        grad_w = wp.vec3(grad_0[3], grad_1[3], grad_2[3])
        (
            angular_lower,
            angular_upper,
            angular_target,
            angular_stiffness,
            angular_target_velocity,
            angular_damping,
        ) = _axis_controls(
            axis_start,
            target_start,
            linear_count,
            angular_count,
            joint_axis,
            joint_limit_lower,
            joint_limit_upper,
            joint_target_q,
            joint_target_qd,
            joint_target_ke,
            joint_target_kd,
        )

        for axis in range(3):
            value = errors[axis]
            lower = angular_lower[axis]
            upper = angular_upper[axis]
            error = 0.0
            row_compliance = joint_angular_compliance
            damping = 0.0
            active = lower == upper
            if value < lower:
                error = value - lower
                active = True
            elif value > upper:
                error = value - upper
                active = True
            elif angular_stiffness[axis] > 0.0:
                error = value - wp.clamp(angular_target[axis], lower, upper)
                row_compliance = 1.0 / angular_stiffness[axis]
                damping = angular_damping[axis]
                active = True
            elif angular_damping[axis] > 0.0:
                row_compliance = 1.0 / angular_damping[axis]
                damping = angular_damping[axis]
                active = True

            if active:
                output_row = axis + 3
                gradient = wp.quat(grad_x[axis], grad_y[axis], grad_z[axis], grad_w[axis])
                child_gradient_q = 0.5 * q_parent * gradient * wp.quat_inverse(q_child)
                angular_c = wp.vec3(child_gradient_q[0], child_gradient_q[1], child_gradient_q[2])
                angular_p = -angular_c
                gradient_length = wp.length(angular_c)
                error_rate = (
                    wp.dot(angular_p, parent_omega)
                    + wp.dot(angular_c, child_omega)
                    - angular_target_velocity[axis] * gradient_length
                )
                scale, row_residual, row_compliance_dt = _scaled_row_terms(
                    error,
                    error_rate,
                    row_compliance,
                    damping,
                    dt,
                    hard_row_regularization,
                )
                jp = _set_spatial_row(jp, output_row, wp.vec3(0.0), scale * angular_p)
                jc = _set_spatial_row(jc, output_row, wp.vec3(0.0), scale * angular_c)
                compliance = _set_spatial_diagonal(compliance, output_row, row_compliance_dt)
                residual[output_row] = row_residual
                active_count += 1

    jacobian_parent[row] = jp
    jacobian_child[row] = jc
    compliance_out[row] = compliance
    residual_out[row] = residual
    row_active[row] = active_count
    joint_jacobian_child[joint] = jc


@wp.kernel
def scatter_path_multiplier(
    joint_ids: wp.array[wp.int32],
    solution: wp.array[wp.spatial_vector],
    joint_multiplier: wp.array[wp.spatial_vector],
):
    row = wp.tid()
    joint_multiplier[joint_ids[row]] = solution[row]


@wp.kernel
def scatter_tree_multiplier(
    node_row: wp.array[wp.int32],
    joint_ids: wp.array[wp.int32],
    row_scale: wp.array[wp.spatial_vector],
    solution: wp.array[wp.spatial_vector],
    joint_multiplier: wp.array[wp.spatial_vector],
):
    node = wp.tid()
    row = node_row[node]
    if row >= 0:
        joint_multiplier[joint_ids[row]] = _scale_spatial_vector(row_scale[row], solution[node])


@wp.kernel
def scatter_closed_tree_multiplier(
    tree_node_count: int,
    closure_count: int,
    node_row: wp.array[wp.int32],
    tree_joint_ids: wp.array[wp.int32],
    tree_row_scale: wp.array[wp.spatial_vector],
    tree_solution: wp.array[wp.spatial_vector],
    response_solution: wp.array[wp.spatial_matrix],
    closure_multiplier: wp.array[wp.spatial_vector],
    joint_multiplier: wp.array[wp.spatial_vector],
):
    node = wp.tid()
    row = node_row[node]
    if row >= 0:
        batch = node // tree_node_count
        value = tree_solution[node]
        for closure in range(closure_count):
            value = (
                value
                - response_solution[node * closure_count + closure]
                * closure_multiplier[batch * closure_count + closure]
            )
        joint_multiplier[tree_joint_ids[row]] = _scale_spatial_vector(tree_row_scale[row], value)


@wp.kernel
def scatter_closure_multiplier(
    joint_ids: wp.array[wp.int32],
    row_scale: wp.array[wp.spatial_vector],
    solution: wp.array[wp.spatial_vector],
    joint_multiplier: wp.array[wp.spatial_vector],
):
    row = wp.tid()
    joint_multiplier[joint_ids[row]] = _scale_spatial_vector(row_scale[row], solution[row])


@wp.kernel
def accumulate_global_joint_impulse(
    joint_ids: wp.array[wp.int32],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    body_slot_by_id: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    island_state: wp.array[wp.int32],
    island_step_scale: wp.array[float],
    joint_jacobian_child: wp.array[wp.spatial_matrix],
    joint_multiplier: wp.array[wp.spatial_vector],
    relaxation: float,
    joint_impulse: wp.array[wp.spatial_vector],
):
    index = wp.tid()
    joint = joint_ids[index]
    slot = body_slot_by_id[joint_child[joint]]
    if slot < 0:
        slot = body_slot_by_id[joint_parent[joint]]
    if slot < 0 or island_state[graph_body_island[slot]] < -1:
        return
    scale = relaxation * island_step_scale[graph_body_island[slot]]
    impulse = -scale * wp.transpose(joint_jacobian_child[joint]) * joint_multiplier[joint]
    joint_impulse[joint] = joint_impulse[joint] + impulse


@wp.kernel
def apply_xpbd_global_correction(
    body_ids: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    island_state: wp.array[wp.int32],
    island_step_scale: wp.array[float],
    correction: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_inv_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    relaxation: float,
    dt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Apply a tangent-space pose correction using XPBD's pose convention."""
    slot = wp.tid()
    if island_state[graph_body_island[slot]] < -1:
        return
    body = body_ids[slot]
    if body_inv_mass[body] == 0.0:
        return
    delta = (relaxation * island_step_scale[graph_body_island[slot]]) * correction[slot]

    pose = body_q[body]
    position = wp.transform_get_translation(pose)
    rotation = wp.transform_get_rotation(pose)
    linear = wp.spatial_top(delta)
    angular = wp.spatial_bottom(delta)

    # Match apply_body_deltas: the structural metric supplies the velocity
    # increment before the gyroscopic impulse correction in the body frame.
    velocity = wp.spatial_top(body_qd[body])
    omega = wp.spatial_bottom(body_qd[body])
    wb = wp.quat_rotate_inv(rotation, omega)
    dwb = wp.quat_rotate_inv(rotation, angular / dt)
    inertia = body_inertia[body]
    tb = wp.cross(dwb, inertia * (wb + dwb)) + wp.cross(wb, inertia * dwb)
    dw = wp.quat_rotate(rotation, dwb - dt * body_inv_inertia[body] * tb)
    rotation_new = wp.normalize(rotation + 0.5 * wp.quat(dw * dt, 0.0) * rotation)
    com_world = position + wp.quat_rotate(rotation, body_com[body])
    position_new = com_world + linear - wp.quat_rotate(rotation_new, body_com[body])
    body_q[body] = wp.transform(position_new, rotation_new)
    velocity_new = velocity + linear / dt
    omega_new = omega + dw
    if wp.length(velocity_new) < 1.0e-4:
        velocity_new = wp.vec3(0.0)
    if wp.length(omega_new) < 1.0e-4:
        omega_new = wp.vec3(0.0)
    body_qd[body] = wp.spatial_vector(velocity_new, omega_new)


class RigidJointGlobalXPBD:
    """XPBD linearization over VBD's private structural factorization backend."""

    def __init__(self, model, body_inv_mass):
        self.device = model.device
        self.backend = StructuralGraphKKT(
            model,
            body_inv_mass,
            allow_general_joint_paths=True,
            supported_joint_types=_SUPPORTED_JOINT_TYPES,
            ignore_free_completion_joints=True,
        )
        self.joint_jacobian_child = wp.zeros(model.joint_count, dtype=wp.spatial_matrix, device=self.device)
        self.joint_multiplier = wp.zeros(model.joint_count, dtype=wp.spatial_vector, device=self.device)
        self.merit_rows = []
        if self.backend.active:
            # The usual tree aliases Jacobians with its factor workspace. The
            # original primal-model guard needs owned, unfactorized rows.
            for bucket in self.backend.tree_buckets + self.backend.closed_tree_buckets:
                tree = bucket.tree if isinstance(bucket, _ClosedTreeBucket) else bucket
                tree.jacobian_parent = wp.empty_like(tree.jacobian_parent)
                tree.jacobian_child = wp.empty_like(tree.jacobian_child)
                self.merit_rows.append(
                    (tree.joint_ids, tree.jacobian_parent, tree.jacobian_child, tree.compliance, tree.residual)
                )
                if isinstance(bucket, _ClosedTreeBucket):
                    self.merit_rows.append(
                        (
                            bucket.closure_joint_ids,
                            bucket.closure_jacobian_parent,
                            bucket.closure_jacobian_child,
                            bucket.closure_compliance,
                            bucket.closure_residual,
                        )
                    )
        if self.merit_rows:
            backend = self.backend
            guarded = [False] * backend.island_count
            slots, islands = backend.body_slot_by_id.numpy(), backend.graph_body_island.numpy()
            parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
            for ids, *_ in self.merit_rows:
                for joint in ids.numpy():
                    for body in (parents[joint], children[joint]):
                        if body >= 0 and slots[body] >= 0:
                            guarded[islands[slots[body]]] = True
            self.guarded_islands = wp.array(guarded, dtype=bool, device=self.device)
            self.quadratic_merit = wp.zeros(backend.island_count, dtype=wp.vec2d, device=self.device)

    def _limit_regularized_tree_step(self, model):
        """Guard approximate tree directions with the original quadratic model.

        This is exact one-dimensional minimization, not a nonlinear/contact
        feasibility guarantee. Unregularized hard-row paths keep a full step.
        It must run before the path inverse overwrites the body metric/rhs.
        """
        if not self.merit_rows:
            return
        backend = self.backend
        self.quadratic_merit.zero_()
        wp.launch(
            accumulate_body_quadratic_merit,
            backend.graph_body_count,
            inputs=[
                backend.graph_body_island,
                self.guarded_islands,
                backend.body_matrix,
                backend.body_rhs,
                backend.body_correction,
            ],
            outputs=[self.quadratic_merit],
            device=self.device,
        )
        for ids, jp, jc, compliance, residual in self.merit_rows:
            wp.launch(
                accumulate_joint_quadratic_merit,
                ids.shape[0],
                inputs=[
                    ids,
                    model.joint_parent,
                    model.joint_child,
                    backend.body_slot_by_id,
                    backend.graph_body_island,
                    jp,
                    jc,
                    compliance,
                    residual,
                    backend.body_correction,
                ],
                outputs=[self.quadratic_merit],
                device=self.device,
            )
        wp.launch(
            minimize_quadratic_step,
            backend.island_count,
            inputs=[self.guarded_islands, self.quadratic_merit],
            outputs=[backend.island_step_scale],
            device=self.device,
        )

    @property
    def active(self) -> bool:
        return self.backend.active

    def solve(
        self,
        *,
        model,
        body_q,
        body_qd,
        body_inv_mass,
        body_inv_inertia,
        control,
        joint_linear_compliance,
        joint_angular_compliance,
        relaxation,
        joint_impulse,
        dt,
        contacts=None,
        contact_impulse=None,
    ) -> None:
        """Apply one global XPBD rigid-joint correction in place."""
        backend = self.backend
        if not backend.active:
            return

        has_tree_route = bool(backend.tree_buckets or backend.closed_tree_buckets)
        if has_tree_route or contacts is not None:
            wp.launch(
                build_xpbd_body_metric,
                backend.graph_body_count,
                inputs=[
                    backend.graph_body_ids,
                    backend.graph_body_island,
                    dt,
                    body_q,
                    body_inv_mass,
                    model.body_inertia,
                ],
                outputs=[backend.body_matrix, backend.body_rhs, backend.island_contact_state],
                device=self.device,
                block_dim=backend.spatial_block_dim,
            )

        if contacts is not None:
            contact_inputs = [
                body_q,
                model.body_com,
                body_inv_mass,
                backend.body_slot_by_id,
                model.shape_body,
                contacts.rigid_contact_count,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
            ]
            wp.launch(
                add_static_contact_terms,
                contacts.rigid_contact_max,
                inputs=contact_inputs,
                outputs=[backend.body_matrix, backend.body_rhs],
                device=self.device,
            )

        for bucket in backend.buckets:
            linearizations = (
                (
                    (
                        bucket.tree.joint_ids,
                        bucket.tree.size,
                        bucket.tree.jacobian_parent,
                        bucket.tree.jacobian_child,
                        bucket.tree.compliance,
                        bucket.tree.residual,
                        bucket.tree.row_active,
                        _TREE_COMPLIANCE_FLOOR,
                    ),
                    (
                        bucket.closure_joint_ids,
                        bucket.closure_size,
                        bucket.closure_jacobian_parent,
                        bucket.closure_jacobian_child,
                        bucket.closure_compliance,
                        bucket.closure_residual,
                        bucket.closure_row_active,
                        _TREE_COMPLIANCE_FLOOR,
                    ),
                )
                if isinstance(bucket, _ClosedTreeBucket)
                else (
                    (
                        bucket.joint_ids,
                        bucket.size,
                        bucket.jacobian_parent,
                        bucket.jacobian_child,
                        bucket.compliance,
                        bucket.residual,
                        bucket.row_active,
                        0.0 if isinstance(bucket, _PathBucket) else _TREE_COMPLIANCE_FLOOR,
                    ),
                )
            )
            for joint_ids, size, jp, jc, compliance, residual, active, regularization in linearizations:
                wp.launch(
                    linearize_xpbd_joint_rows,
                    size,
                    inputs=[
                        joint_ids,
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_qd_start,
                        model.joint_target_q_start,
                        model.joint_dof_dim,
                        model.joint_axis,
                        control.joint_target_q,
                        control.joint_target_qd,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        joint_linear_compliance,
                        joint_angular_compliance,
                        regularization,
                        body_q,
                        body_qd,
                        model.body_com,
                        dt,
                    ],
                    outputs=[jp, jc, compliance, residual, active, self.joint_jacobian_child],
                    device=self.device,
                    block_dim=backend.spatial_block_dim,
                )

        for bucket in backend.tree_buckets:
            bucket.solve_tree(
                backend.body_matrix,
                backend.body_rhs,
                backend.body_scale,
                backend.body_correction,
            )
            if joint_impulse is not None:
                wp.launch(
                    scatter_tree_multiplier,
                    bucket.node_count,
                    inputs=[bucket.node_row, bucket.joint_ids, bucket.row_scale, bucket.solution],
                    outputs=[self.joint_multiplier],
                    device=self.device,
                    block_dim=backend.spatial_block_dim,
                )
        for bucket in backend.closed_tree_buckets:
            bucket.solve_tree(
                backend.body_matrix,
                backend.body_rhs,
                backend.body_scale,
                backend.body_correction,
            )
            if joint_impulse is not None:
                wp.launch(
                    scatter_closed_tree_multiplier,
                    bucket.tree.node_count,
                    inputs=[
                        bucket.tree.tree_node_count,
                        bucket.closure_count,
                        bucket.tree.node_row,
                        bucket.tree.joint_ids,
                        bucket.tree.row_scale,
                        bucket.tree.solution,
                        bucket.response_solution,
                        bucket.closure_multiplier,
                    ],
                    outputs=[self.joint_multiplier],
                    device=self.device,
                    block_dim=backend.spatial_block_dim,
                )
                wp.launch(
                    scatter_closure_multiplier,
                    bucket.closure_size,
                    inputs=[
                        bucket.closure_joint_ids,
                        bucket.closure_row_scale,
                        bucket.closure_multiplier,
                    ],
                    outputs=[self.joint_multiplier],
                    device=self.device,
                    block_dim=backend.spatial_block_dim,
                )

        self._limit_regularized_tree_step(model)
        if backend.path_buckets:
            wp.launch(
                build_xpbd_body_inverse,
                backend.graph_body_count,
                inputs=[
                    backend.graph_body_ids,
                    backend.graph_body_island,
                    int(not has_tree_route),
                    dt,
                    body_q,
                    body_inv_mass,
                    body_inv_inertia,
                    contacts is not None,
                    backend.body_matrix,
                    backend.body_rhs,
                ],
                outputs=[backend.body_inverse, backend.body_free, backend.island_contact_state],
                device=self.device,
                block_dim=backend.spatial_block_dim,
            )
        for bucket in backend.path_buckets:
            wp.launch(
                assemble_joint_path_system,
                bucket.size,
                inputs=[
                    bucket.row_count,
                    bucket.row_body,
                    backend.body_inverse,
                    backend.body_free,
                    bucket.jacobian_parent,
                    bucket.jacobian_child,
                    bucket.compliance,
                    bucket.residual,
                ],
                outputs=[bucket.lower[0], bucket.diagonal[0], bucket.upper[0], bucket.rhs[0]],
                device=self.device,
                block_dim=backend.spatial_block_dim,
            )
            bucket.solve_rows()
            wp.launch(
                compute_path_correction,
                bucket.body_size,
                inputs=[
                    bucket.row_count,
                    bucket.body_count,
                    bucket.body_ids,
                    backend.body_slot_by_id,
                    bucket.body_incident_rows,
                    bucket.row_active,
                    bucket.jacobian_parent,
                    bucket.jacobian_child,
                    bucket.solution[0],
                    backend.body_inverse,
                    backend.body_free,
                    body_inv_mass,
                ],
                outputs=[backend.body_correction],
                device=self.device,
                block_dim=backend.spatial_block_dim,
            )
            if joint_impulse is not None:
                wp.launch(
                    scatter_path_multiplier,
                    bucket.size,
                    inputs=[bucket.joint_ids, bucket.solution[0]],
                    outputs=[self.joint_multiplier],
                    device=self.device,
                    block_dim=backend.spatial_block_dim,
                )

        wp.launch(
            suppress_nonfinite_correction,
            backend.graph_body_count,
            inputs=[backend.graph_body_island, backend.body_correction],
            outputs=[backend.island_contact_state],
            device=self.device,
        )
        if joint_impulse is not None:
            wp.launch(
                accumulate_global_joint_impulse,
                backend.graph_joint_ids.shape[0],
                inputs=[
                    backend.graph_joint_ids,
                    model.joint_parent,
                    model.joint_child,
                    backend.body_slot_by_id,
                    backend.graph_body_island,
                    backend.island_contact_state,
                    backend.island_step_scale,
                    self.joint_jacobian_child,
                    self.joint_multiplier,
                    relaxation,
                ],
                outputs=[joint_impulse],
                device=self.device,
            )
        if contacts is not None and contact_impulse is not None:
            wp.launch(
                accumulate_static_contact_impulse,
                contacts.rigid_contact_max,
                inputs=[
                    *contact_inputs,
                    backend.graph_body_island,
                    backend.island_contact_state,
                    backend.island_step_scale,
                    backend.body_correction,
                    relaxation,
                ],
                outputs=[contact_impulse],
                device=self.device,
            )
        wp.launch(
            apply_xpbd_global_correction,
            backend.graph_body_count,
            inputs=[
                backend.graph_body_ids,
                backend.graph_body_island,
                backend.island_contact_state,
                backend.island_step_scale,
                backend.body_correction,
                model.body_com,
                model.body_inertia,
                body_inv_inertia,
                body_inv_mass,
                relaxation,
                dt,
            ],
            outputs=[body_q, body_qd],
            device=self.device,
        )
