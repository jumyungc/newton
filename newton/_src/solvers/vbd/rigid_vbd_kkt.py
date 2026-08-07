# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal compliance-KKT accelerator for persistent VBD joint graphs.

Solves stiff, long-range joint structures in one global pass so a reaction
propagates across a whole connected island at once, instead of creeping one
joint per local sweep. Local VBD keeps ownership of inertia, contact, friction,
history, and multipliers; this module only accelerates the structural coupling.

The system is written in compliance form,

    (C + J B^-1 J^T) lambda = c + J B^-1 f,

and never forms ``J^T K J``, so it stays well-conditioned as joints approach
rigid (compliance tends to zero rather than stiffness to infinity).

Each connected island is factored by the exact scheme its topology allows.
Serial, branched, and cyclic graphs are private factorization schedules of the
same system, not user-visible solver modes. Existing VBD contact Hessians enter
as a majorizer -- stiffening supported directions without adding contact-graph
edges. Before committing a correction, an island-wide block-Jacobi relaxation
accounts for omitted dynamic-contact overlap, while the unilateral guard keeps
static-contact corrections from consuming unavailable normal gap.
History-bearing rows otherwise retain the step-frozen values owned by local VBD.

This module is private; the only public control is the global-iteration count G.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import warp as wp

from newton._src.core.types import MAXVAL
from newton._src.math import quat_velocity
from newton._src.sim import JointType

from .rigid_vbd_kernels import (
    _NUM_CONTACT_THREADS_PER_BODY,
    _SMALL_ANGLE_EPS,
    _SMALL_LENGTH_EPS,
    _assemble_geometric_cable_kappa_z,
    _cable_bend_twist_delta,
    _cable_bend_twist_jacobian_z_from_measure,
    _compliant_alm_coefficients,
    _evaluate_drive_axis,
    _evaluate_limit_axis,
    _load_solve_weight,
    _material_force_terms,
    _measure_cable_bend_twist_z,
    _resolve_active_drive_row,
    build_joint_projectors,
    compute_kappa,
    compute_kappa_and_jacobian,
    compute_kappa_dot,
    contact_surface_separation,
    evaluate_rigid_contact_from_collision,
)

wp.set_module_options({"enable_backward": False})

_CR_SERIAL_TERMINAL_SIZE = 4
"""Maximum coarse row count finished by one batched serial kernel after CR."""

_CR_PERSISTENT_MAX_ROWS = 128
"""Largest path solved by one synchronized CUDA block instead of level launches.

At this size the saved dependent launches still outweigh single-block occupancy;
larger paths retain the level schedule so multiple blocks can cooperate.
"""

_SPATIAL_GPU_BLOCK_DIM = 32
"""One-warp blocks for register-heavy spatial-matrix kernels."""

_INVERSE_RESIDUAL_TOLERANCE = wp.constant(1.0e-2)
"""Maximum infinity-norm residual accepted from a fast float32 block inverse."""

_JOINT_LIMIT_STEP_FRACTION = wp.constant(0.99)
"""Fraction-to-boundary safety for nonlinear global joint-limit corrections."""


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _synchronize_cr_block(): ...


@wp.func
def _normalized_cable_row(
    error: float,
    previous_error: float,
    material_k: float,
    rho: float,
    damping: float,
    multiplier: float,
    history_force: float,
    history_tangent: float,
    dt: float,
):
    """Return one cable row's tangent and normalized local force defect."""
    s, k_eff, _a = _compliant_alm_coefficients(material_k, rho)
    damping_tangent = damping / dt
    tangent = k_eff + damping_tangent + wp.max(history_tangent, 0.0)
    if tangent <= 0.0:
        return 0.0, 0.0
    force = k_eff * error + s * multiplier + history_force + damping_tangent * (error - previous_error)
    return tangent, force / tangent


@wp.func
def _inverse_spatial(matrix: wp.spatial_matrix):
    """Invert a symmetric positive-definite 6x6 matrix by a 3x3 Schur split."""
    a = wp.mat33(0.0)
    b = wp.mat33(0.0)
    d = wp.mat33(0.0)
    for row in range(3):
        for column in range(3):
            a[row, column] = matrix[row, column]
            b[row, column] = matrix[row, column + 3]
            d[row, column] = matrix[row + 3, column + 3]

    a_inverse = wp.inverse(a)
    a_inverse_b = a_inverse * b
    schur = d - wp.transpose(b) * a_inverse_b
    schur_inverse = wp.inverse(schur)
    upper_right = -(a_inverse_b * schur_inverse)
    lower_left = wp.transpose(upper_right)
    upper_left = a_inverse + a_inverse_b * schur_inverse * wp.transpose(a_inverse_b)

    result = wp.spatial_matrix(0.0)
    for row in range(3):
        for column in range(3):
            result[row, column] = upper_left[row, column]
            result[row, column + 3] = upper_right[row, column]
            result[row + 3, column] = lower_left[row, column]
            result[row + 3, column + 3] = schur_inverse[row, column]
    return result


@wp.func
def _inverse_spatial_pivoted(matrix: wp.spatial_matrix):
    """Invert a general 6x6 frontal block with row partial pivoting."""
    value = matrix
    inverse = wp.identity(6, float)
    for column in range(6):
        pivot = column
        pivot_magnitude = wp.abs(value[column, column])
        for row in range(column + 1, 6):
            magnitude = wp.abs(value[row, column])
            if magnitude > pivot_magnitude:
                pivot = row
                pivot_magnitude = magnitude
        if pivot != column:
            for entry in range(6):
                temporary = value[column, entry]
                value[column, entry] = value[pivot, entry]
                value[pivot, entry] = temporary
                temporary = inverse[column, entry]
                inverse[column, entry] = inverse[pivot, entry]
                inverse[pivot, entry] = temporary

        reciprocal = 1.0 / value[column, column]
        for entry in range(6):
            value[column, entry] = reciprocal * value[column, entry]
            inverse[column, entry] = reciprocal * inverse[column, entry]
        for row in range(6):
            if row != column:
                factor = value[row, column]
                for entry in range(6):
                    value[row, entry] = value[row, entry] - factor * value[column, entry]
                    inverse[row, entry] = inverse[row, entry] - factor * inverse[column, entry]
    return inverse


@wp.func
def _inverse_spatial_robust(matrix: wp.spatial_matrix):
    """Use the fast SPD inverse unless its computed inverse fails a residual check."""
    inverse = _inverse_spatial(matrix)
    maximum_error = 0.0
    for row in range(6):
        for column in range(6):
            value = 0.0
            for entry in range(6):
                value = value + matrix[row, entry] * inverse[entry, column]
            target = 1.0 if row == column else 0.0
            maximum_error = wp.max(maximum_error, wp.abs(value - target))
    if not wp.isfinite(maximum_error) or maximum_error > _INVERSE_RESIDUAL_TOLERANCE:
        inverse = _inverse_spatial_pivoted(matrix)
    return inverse


@wp.func
def _set_spatial_block(matrix: wp.spatial_matrix, row_block: int, column_block: int, value: wp.mat33):
    for row in range(3):
        for column in range(3):
            matrix[row + 3 * row_block, column + 3 * column_block] = value[row, column]
    return matrix


@wp.func
def _spatial_block(matrix: wp.spatial_matrix, row_block: int, column_block: int):
    value = wp.mat33(0.0)
    for row in range(3):
        for column in range(3):
            value[row, column] = matrix[row + 3 * row_block, column + 3 * column_block]
    return value


@wp.func
def _scale_spatial_vector(scale: wp.spatial_vector, value: wp.spatial_vector):
    result = wp.spatial_vector()
    for row in range(6):
        result[row] = scale[row] * value[row]
    return result


@wp.func
def _scale_spatial_matrix(
    left_scale: wp.spatial_vector,
    matrix: wp.spatial_matrix,
    right_scale: wp.spatial_vector,
):
    result = wp.spatial_matrix(0.0)
    for row in range(6):
        for column in range(6):
            result[row, column] = left_scale[row] * matrix[row, column] * right_scale[column]
    return result


@wp.kernel
def clear_structural_contact_objective(
    body_ids: wp.array[wp.int32],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
    body_dynamic_contact_hessian: wp.array[wp.spatial_matrix],
):
    """Clear contact scratch only for bodies represented by the global graph."""
    slot = wp.tid()
    body = body_ids[slot]
    body_forces[body] = wp.vec3(0.0)
    body_torques[body] = wp.vec3(0.0)
    body_hessian_ll[body] = wp.mat33(0.0)
    body_hessian_al[body] = wp.mat33(0.0)
    body_hessian_aa[body] = wp.mat33(0.0)
    body_dynamic_contact_hessian[slot] = wp.spatial_matrix(0.0)


@wp.kernel
def accumulate_structural_body_body_contacts(
    dt: float,
    body_ids: wp.array[wp.int32],
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    friction_epsilon: float,
    contact_penalty_k: wp.array[float],
    contact_normal_rho: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_tangent_rho: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    stab_alpha: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
    body_dynamic_contact_hessian: wp.array[wp.spatial_matrix],
):
    """Accumulate the contact objective used only by a global structural pass."""
    tid = wp.tid()
    body_slot = tid // _NUM_CONTACT_THREADS_PER_BODY
    contact_thread = tid % _NUM_CONTACT_THREADS_PER_BODY
    if body_slot >= body_ids.shape[0]:
        return

    body = body_ids[body_slot]
    if body_inv_mass[body] <= 0.0:
        return

    num_contacts = wp.min(body_contact_counts[body], body_contact_buffer_pre_alloc)
    contact_count = rigid_contact_count[0]
    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)
    dynamic_h_ll_acc = wp.mat33(0.0)
    dynamic_h_al_acc = wp.mat33(0.0)
    dynamic_h_aa_acc = wp.mat33(0.0)

    i = contact_thread
    while i < num_contacts:
        contact = body_contact_indices[body * body_contact_buffer_pre_alloc + i]
        if contact >= contact_count:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        shape0 = rigid_contact_shape0[contact]
        shape1 = rigid_contact_shape1[contact]
        body0 = shape_body[shape0] if shape0 >= 0 else -1
        body1 = shape_body[shape1] if shape1 >= 0 else -1
        if body0 != body and body1 != body:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        point0_local = rigid_contact_point0[contact]
        point1_local = rigid_contact_point1[contact]
        point0_offset = rigid_contact_offset0[contact]
        point1_offset = rigid_contact_offset1[contact]
        normal = rigid_contact_normal[contact]
        point0 = wp.transform_point(body_q[body0], point0_local) if body0 >= 0 else point0_local
        point1 = wp.transform_point(body_q[body1], point1_local) if body1 >= 0 else point1_local
        normal_error = -contact_surface_separation(
            point0,
            point1,
            normal,
            rigid_contact_margin0[contact],
            rigid_contact_margin1[contact],
        )

        multiplier = wp.vec3(0.0)
        multiplier_normal = 0.0
        stabilized_error = normal_error
        tangent_C0 = wp.vec3(0.0)
        normal_weight = _load_solve_weight(
            contact_penalty_k,
            contact_normal_rho,
            contact,
            contact_compliant_alm,
        )
        material_k = contact_material_ke[contact]
        if legacy_hard_contacts == 1 or contact_compliant_alm == 1:
            multiplier = contact_lambda[contact]
            multiplier_normal = wp.dot(multiplier, normal)
            C0 = contact_C0[contact]
            C0_normal = wp.dot(normal, C0)
            stabilized_error = normal_error - stab_alpha * C0_normal
            tangent_C0 = (1.0 - stab_alpha) * (C0 - normal * C0_normal)

        if normal_error <= _SMALL_LENGTH_EPS and multiplier_normal <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        normal_primal_k, multiplier_normal_eff = _material_force_terms(
            normal_weight,
            material_k,
            multiplier_normal,
            contact_compliant_alm,
        )
        if normal_primal_k * stabilized_error + multiplier_normal_eff <= 0.0 and multiplier_normal <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        (
            force0,
            torque0,
            h_ll0,
            h_al0,
            h_aa0,
            force1,
            torque1,
            h_ll1,
            h_al1,
            h_aa1,
        ) = evaluate_rigid_contact_from_collision(
            body0,
            body1,
            body_q,
            body_q_prev,
            body_com,
            point0_local,
            point1_local,
            point0_offset,
            point1_offset,
            normal,
            stabilized_error,
            normal_weight,
            material_k,
            contact_tangent_rho[contact],
            contact_material_kd[contact],
            multiplier,
            contact_material_mu[contact],
            friction_epsilon,
            legacy_hard_contacts,
            contact_compliant_alm,
            dt,
            tangent_C0,
        )

        other_body = body1 if body == body0 else body0
        dynamic_pair = other_body >= 0 and body_inv_mass[other_body] > 0.0
        if body == body0:
            force_acc += force0
            torque_acc += torque0
            h_ll_acc += h_ll0
            h_al_acc += h_al0
            h_aa_acc += h_aa0
            if dynamic_pair:
                dynamic_h_ll_acc += h_ll0
                dynamic_h_al_acc += h_al0
                dynamic_h_aa_acc += h_aa0
        else:
            force_acc += force1
            torque_acc += torque1
            h_ll_acc += h_ll1
            h_al_acc += h_al1
            h_aa_acc += h_aa1
            if dynamic_pair:
                dynamic_h_ll_acc += h_ll1
                dynamic_h_al_acc += h_al1
                dynamic_h_aa_acc += h_aa1
        i += _NUM_CONTACT_THREADS_PER_BODY

    wp.atomic_add(body_forces, body, force_acc)
    wp.atomic_add(body_torques, body, torque_acc)
    wp.atomic_add(body_hessian_ll, body, h_ll_acc)
    wp.atomic_add(body_hessian_al, body, h_al_acc)
    wp.atomic_add(body_hessian_aa, body, h_aa_acc)
    dynamic_hessian = wp.spatial_matrix(0.0)
    for row in range(3):
        for column in range(3):
            dynamic_hessian[row, column] = dynamic_h_ll_acc[row, column]
            dynamic_hessian[row, column + 3] = dynamic_h_al_acc[column, row]
            dynamic_hessian[row + 3, column] = dynamic_h_al_acc[row, column]
            dynamic_hessian[row + 3, column + 3] = dynamic_h_aa_acc[row, column]
    wp.atomic_add(body_dynamic_contact_hessian, body_slot, dynamic_hessian)


@wp.kernel
def classify_global_contact_islands(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    body_slot_by_id: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    island_contact_state: wp.array[wp.int32],
):
    """Mark structural islands touched by active dynamic-dynamic contact."""
    contact = wp.tid()
    if contact >= rigid_contact_count[0]:
        return
    shape0 = rigid_contact_shape0[contact]
    shape1 = rigid_contact_shape1[contact]
    body0 = shape_body[shape0] if shape0 >= 0 else -1
    body1 = shape_body[shape1] if shape1 >= 0 else -1
    dynamic0 = body0 >= 0 and body_inv_mass[body0] > 0.0
    dynamic1 = body1 >= 0 and body_inv_mass[body1] > 0.0
    if not dynamic0 and not dynamic1:
        return

    slot0 = body_slot_by_id[body0] if dynamic0 else -1
    slot1 = body_slot_by_id[body1] if dynamic1 else -1
    island0 = graph_body_island[slot0] if slot0 >= 0 else -1
    island1 = graph_body_island[slot1] if slot1 >= 0 else -1
    normal = rigid_contact_normal[contact]
    point0 = (
        wp.transform_point(body_q[body0], rigid_contact_point0[contact])
        if body0 >= 0
        else rigid_contact_point0[contact]
    )
    point1 = (
        wp.transform_point(body_q[body1], rigid_contact_point1[contact])
        if body1 >= 0
        else rigid_contact_point1[contact]
    )
    separation = wp.dot(normal, point1 - point0) - rigid_contact_margin0[contact] - rigid_contact_margin1[contact]
    if separation > 0.0 and wp.dot(contact_lambda[contact], normal) <= 0.0:
        return
    if dynamic0 and dynamic1:
        if island0 >= 0:
            wp.atomic_min(island_contact_state, island0, -1)
        if island1 >= 0:
            wp.atomic_min(island_contact_state, island1, -1)


@wp.kernel
def build_body_surrogate(
    body_ids: wp.array[wp.int32],
    dt: float,
    body_q: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    contact_hessian_ll: wp.array[wp.mat33],
    contact_hessian_al: wp.array[wp.mat33],
    contact_hessian_aa: wp.array[wp.mat33],
    contact_forces: wp.array[wp.vec3],
    contact_torques: wp.array[wp.vec3],
    dynamic_contact_hessian: wp.array[wp.spatial_matrix],
    graph_body_island: wp.array[wp.int32],
    island_contact_state: wp.array[wp.int32],
    body_matrix_out: wp.array[wp.spatial_matrix],
    body_rhs_out: wp.array[wp.spatial_vector],
):
    """Build the contact-proximal body metric for a structural correction."""
    slot = wp.tid()
    body = body_ids[slot]
    if body_inv_mass[body] <= 0.0:
        body_matrix_out[slot] = wp.spatial_matrix(0.0)
        body_rhs_out[slot] = wp.spatial_vector()
        return

    dt_inv_sq = 1.0 / (dt * dt)
    pose = body_q[body]
    inertial_pose = body_inertia_q[body]
    rotation = wp.transform_get_rotation(pose)
    inertial_rotation = wp.transform_get_rotation(inertial_pose)
    com_local = body_com[body]

    mass_scale = body_mass[body] * dt_inv_sq

    identity = wp.identity(3, float)
    # Dynamic-dynamic contact is an off-topology edge, so use the 2H block
    # majorizer for affected structural islands. Static/kinematic contact is
    # owned by local VBD and uses its exact per-body 1H metric here.
    island = graph_body_island[slot]
    contact_majorizer_scale = 1.0 if island >= 0 and island_contact_state[island] < 0 else 0.0
    dynamic_h_ll = _spatial_block(dynamic_contact_hessian[slot], 0, 0)
    dynamic_h_al = _spatial_block(dynamic_contact_hessian[slot], 1, 0)
    dynamic_h_aa = _spatial_block(dynamic_contact_hessian[slot], 1, 1)
    h_ll = (
        mass_scale * identity
        + (1.0 + contact_majorizer_scale) * contact_hessian_ll[body]
        + (1.0 - contact_majorizer_scale) * dynamic_h_ll
    )
    rotation_matrix = wp.quat_to_matrix(rotation)
    h_aa = (
        dt_inv_sq * rotation_matrix * body_inertia[body] * wp.transpose(rotation_matrix)
        + (1.0 + contact_majorizer_scale) * contact_hessian_aa[body]
        + (1.0 - contact_majorizer_scale) * dynamic_h_aa
    )
    h_al = (1.0 + contact_majorizer_scale) * contact_hessian_al[body] + (1.0 - contact_majorizer_scale) * dynamic_h_al

    com = wp.transform_point(pose, com_local)
    inertial_com = wp.transform_point(inertial_pose, com_local)
    # Use the same current inertial/contact residual as local VBD.  Contact
    # multipliers remain local state; the following local sweep relinearizes
    # and accepts the globally coupled predictor.
    force = mass_scale * (inertial_com - com) + contact_forces[body]
    rotation_delta = wp.quat_inverse(rotation) * inertial_rotation
    if rotation_delta[3] < 0.0:
        rotation_delta = wp.quat(
            -rotation_delta[0],
            -rotation_delta[1],
            -rotation_delta[2],
            -rotation_delta[3],
        )
    axis, angle = wp.quat_to_axis_angle(rotation_delta)
    torque = wp.quat_rotate(rotation, body_inertia[body] * (axis * angle * dt_inv_sq)) + contact_torques[body]

    # Match the local VBD regularization while keeping the compliance solve SPD.
    eps_l = 1.0e-9 * (wp.trace(h_ll) / 3.0 + 1.0)
    eps_a = 1.0e-9 * (wp.trace(h_aa) / 3.0 + 1.0)
    for diagonal_axis in range(3):
        h_ll[diagonal_axis, diagonal_axis] = h_ll[diagonal_axis, diagonal_axis] + eps_l
        h_aa[diagonal_axis, diagonal_axis] = h_aa[diagonal_axis, diagonal_axis] + eps_a

    body_matrix = wp.spatial_matrix(0.0)
    body_matrix = _set_spatial_block(body_matrix, 0, 0, h_ll)
    body_matrix = _set_spatial_block(body_matrix, 0, 1, wp.transpose(h_al))
    body_matrix = _set_spatial_block(body_matrix, 1, 0, h_al)
    body_matrix = _set_spatial_block(body_matrix, 1, 1, h_aa)
    body_matrix_out[slot] = body_matrix
    body_rhs_out[slot] = wp.spatial_vector(force, torque)


@wp.kernel
def invert_body_surrogate_in_place(
    body_ids: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_matrix_and_inverse: wp.array[wp.spatial_matrix],
    body_rhs_and_free: wp.array[wp.spatial_vector],
):
    """Replace each path body's metric and rhs by inverse and free motion."""
    slot = wp.tid()
    body = body_ids[slot]
    if body_inv_mass[body] <= 0.0:
        body_matrix_and_inverse[slot] = wp.spatial_matrix(0.0)
        body_rhs_and_free[slot] = wp.spatial_vector()
        return
    inverse = _inverse_spatial_robust(body_matrix_and_inverse[slot])
    body_matrix_and_inverse[slot] = inverse
    body_rhs_and_free[slot] = inverse * body_rhs_and_free[slot]


@wp.kernel
def linearize_joint_path_rows(
    joint_ids: wp.array[wp.int32],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_cable_rest_kb_local: wp.array[wp.vec3],
    joint_cable_rest_twist: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_constraint_start: wp.array[int],
    joint_material_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_rest_angle: wp.array[float],
    joint_drive_limit_support: wp.array[float],
    joint_drive_lambda: wp.array[float],
    joint_limit_lambda: wp.array[float],
    stab_alpha: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    dt: float,
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    compliance: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    row_active: wp.array[wp.int32],
):
    """Linearize one projected joint row in body tangent coordinates."""
    row = wp.tid()
    joint = joint_ids[row]
    zero_matrix = wp.spatial_matrix(0.0)
    jt = joint_type[joint]
    supported = (
        jt == JointType.CABLE
        or jt == JointType.BALL
        or jt == JointType.FIXED
        or jt == JointType.REVOLUTE
        or jt == JointType.PRISMATIC
        or jt == JointType.D6
    )
    if not supported or not joint_enabled[joint]:
        jacobian_parent[row] = zero_matrix
        jacobian_child[row] = zero_matrix
        compliance[row] = wp.identity(6, float)
        residual[row] = wp.spatial_vector()
        row_active[row] = 0
        return

    parent = joint_parent[joint]
    child = joint_child[joint]
    parent_pose = wp.transform_identity()
    parent_pose_prev = parent_pose
    parent_pose_rest = parent_pose
    if parent >= 0:
        parent_pose = body_q[parent]
        parent_pose_prev = body_q_prev[parent]
        parent_pose_rest = body_q_rest[parent]
    child_pose = body_q[child]
    child_pose_prev = body_q_prev[child]
    child_pose_rest = body_q_rest[child]

    X_wp = parent_pose * joint_X_p[joint]
    X_wc = child_pose * joint_X_c[joint]
    X_wp_prev = parent_pose_prev * joint_X_p[joint]
    X_wc_prev = child_pose_prev * joint_X_c[joint]
    X_wp_rest = parent_pose_rest * joint_X_p[joint]
    X_wc_rest = child_pose_rest * joint_X_c[joint]

    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    x_p_prev = wp.transform_get_translation(X_wp_prev)
    x_c_prev = wp.transform_get_translation(X_wc_prev)
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_wp_prev = wp.transform_get_rotation(X_wp_prev)
    q_wc_prev = wp.transform_get_rotation(X_wc_prev)
    q_wp_rest = wp.transform_get_rotation(X_wp_rest)
    q_wc_rest = wp.transform_get_rotation(X_wc_rest)

    c_start = joint_constraint_start[joint]
    if jt == JointType.CABLE:
        # Cable rows live in their material frame:
        # [shear_x, shear_y, stretch_z, bend_x, bend_y, twist_z].
        stretch_idx = c_start
        shear_idx = c_start + 1
        bend_idx = c_start + 2
        twist_idx = c_start + 3

        q_wp_inv = wp.quat_inverse(q_wp)
        material_from_world = wp.quat_to_matrix(q_wp_inv)
        u = wp.quat_rotate(q_wp_inv, x_c - x_p)
        u_prev = wp.quat_rotate(wp.quat_inverse(q_wp_prev), x_c_prev - x_p_prev)
        C0_lin = joint_C0_lin[joint]
        lambda_lin = joint_lambda_lin[joint]

        shear_tangent, shear_defect_x = _normalized_cable_row(
            u[0] - stab_alpha * C0_lin[0],
            u_prev[0] - stab_alpha * C0_lin[0],
            joint_material_k[shear_idx],
            joint_rho[shear_idx],
            joint_penalty_kd[shear_idx],
            lambda_lin[0],
            0.0,
            0.0,
            dt,
        )
        shear_tangent_y, shear_defect_y = _normalized_cable_row(
            u[1] - stab_alpha * C0_lin[1],
            u_prev[1] - stab_alpha * C0_lin[1],
            joint_material_k[shear_idx],
            joint_rho[shear_idx],
            joint_penalty_kd[shear_idx],
            lambda_lin[1],
            0.0,
            0.0,
            dt,
        )
        stretch_tangent, stretch_defect = _normalized_cable_row(
            u[2] - stab_alpha * C0_lin[2],
            u_prev[2] - stab_alpha * C0_lin[2],
            joint_material_k[stretch_idx],
            joint_rho[stretch_idx],
            joint_penalty_kd[stretch_idx],
            lambda_lin[2],
            0.0,
            0.0,
            dt,
        )
        linear_mask = wp.mat33(0.0)
        linear_compliance = wp.mat33(0.0)
        if shear_tangent > 0.0:
            linear_mask[0, 0] = 1.0
            linear_compliance[0, 0] = 1.0 / shear_tangent
        else:
            linear_compliance[0, 0] = 1.0
        if shear_tangent_y > 0.0:
            linear_mask[1, 1] = 1.0
            linear_compliance[1, 1] = 1.0 / shear_tangent_y
        else:
            linear_compliance[1, 1] = 1.0
        if stretch_tangent > 0.0:
            linear_mask[2, 2] = 1.0
            linear_compliance[2, 2] = 1.0 / stretch_tangent
        else:
            linear_compliance[2, 2] = 1.0

        jp = wp.spatial_matrix(0.0)
        jc = wp.spatial_matrix(0.0)
        linear_jacobian = linear_mask * material_from_world
        parent_com_world = wp.vec3(0.0)
        if parent >= 0:
            parent_com_world = wp.transform_point(parent_pose, body_com[parent])
        child_com_world = wp.transform_point(child_pose, body_com[child])
        # Parent material rotation makes its effective lever arm end at the
        # child anchor, matching the local cable Gauss-Newton Hessian.
        jp = _set_spatial_block(jp, 0, 0, -linear_jacobian)
        jp = _set_spatial_block(jp, 0, 1, linear_jacobian * wp.skew(x_c - parent_com_world))
        jc = _set_spatial_block(jc, 0, 0, linear_jacobian)
        jc = _set_spatial_block(jc, 0, 1, -(linear_jacobian * wp.skew(x_c - child_com_world)))

        measure = _measure_cable_bend_twist_z(q_wp, q_wc)
        kappa = _assemble_geometric_cable_kappa_z(
            q_wp,
            measure.kb_world,
            measure.twist,
            joint_cable_rest_kb_local[joint],
            joint_cable_rest_twist[joint],
        )
        previous_measure = _measure_cable_bend_twist_z(q_wp_prev, q_wc_prev)
        kappa_prev = _assemble_geometric_cable_kappa_z(
            q_wp_prev,
            previous_measure.kb_world,
            previous_measure.twist,
            joint_cable_rest_kb_local[joint],
            joint_cable_rest_twist[joint],
        )
        delta_kappa = _cable_bend_twist_delta(kappa, kappa_prev)
        C0_ang = joint_C0_ang[joint]
        lambda_ang = joint_lambda_ang[joint]
        sigma = joint_sigma_start[joint]
        friction_tangent = joint_C_fric[joint]

        bend_error_x = kappa[0] - stab_alpha * C0_ang[0]
        bend_error_y = kappa[1] - stab_alpha * C0_ang[1]
        twist_error = kappa[2] - stab_alpha * C0_ang[2]
        bend_tangent, bend_defect_x = _normalized_cable_row(
            bend_error_x,
            bend_error_x - delta_kappa[0],
            joint_material_k[bend_idx],
            joint_rho[bend_idx],
            joint_penalty_kd[bend_idx],
            lambda_ang[0],
            sigma[0],
            friction_tangent[0],
            dt,
        )
        bend_tangent_y, bend_defect_y = _normalized_cable_row(
            bend_error_y,
            bend_error_y - delta_kappa[1],
            joint_material_k[bend_idx],
            joint_rho[bend_idx],
            joint_penalty_kd[bend_idx],
            lambda_ang[1],
            sigma[1],
            friction_tangent[1],
            dt,
        )
        twist_tangent, twist_defect = _normalized_cable_row(
            twist_error,
            twist_error - delta_kappa[2],
            joint_material_k[twist_idx],
            joint_rho[twist_idx],
            joint_penalty_kd[twist_idx],
            lambda_ang[2],
            sigma[2],
            friction_tangent[2],
            dt,
        )
        angular_mask = wp.mat33(0.0)
        angular_compliance = wp.mat33(0.0)
        if bend_tangent > 0.0:
            angular_mask[0, 0] = 1.0
            angular_compliance[0, 0] = 1.0 / bend_tangent
        else:
            angular_compliance[0, 0] = 1.0
        if bend_tangent_y > 0.0:
            angular_mask[1, 1] = 1.0
            angular_compliance[1, 1] = 1.0 / bend_tangent_y
        else:
            angular_compliance[1, 1] = 1.0
        if twist_tangent > 0.0:
            angular_mask[2, 2] = 1.0
            angular_compliance[2, 2] = 1.0 / twist_tangent
        else:
            angular_compliance[2, 2] = 1.0

        parent_angular_jacobian = _cable_bend_twist_jacobian_z_from_measure(q_wp, measure, True)
        child_angular_jacobian = _cable_bend_twist_jacobian_z_from_measure(q_wp, measure, False)
        jp = _set_spatial_block(jp, 1, 1, angular_mask * parent_angular_jacobian)
        jc = _set_spatial_block(jc, 1, 1, angular_mask * child_angular_jacobian)

        material_compliance = wp.spatial_matrix(0.0)
        material_compliance = _set_spatial_block(material_compliance, 0, 0, linear_compliance)
        material_compliance = _set_spatial_block(material_compliance, 1, 1, angular_compliance)
        jacobian_parent[row] = jp
        jacobian_child[row] = jc
        compliance[row] = material_compliance
        residual[row] = wp.spatial_vector(
            wp.vec3(shear_defect_x, shear_defect_y, stretch_defect),
            wp.vec3(bend_defect_x, bend_defect_y, twist_defect),
        )
        row_active[row] = (
            1
            if shear_tangent > 0.0
            or shear_tangent_y > 0.0
            or stretch_tangent > 0.0
            or bend_tangent > 0.0
            or bend_tangent_y > 0.0
            or twist_tangent > 0.0
            else 0
        )
        return

    kappa, angular_jacobian_world = compute_kappa_and_jacobian(q_wp, q_wc, q_wp_rest, q_wc_rest)
    kappa_prev = compute_kappa(q_wp_prev, q_wc_prev, q_wp_rest, q_wc_rest)

    lin_count = 0
    ang_count = 0
    qd_start = joint_qd_start[joint]
    if jt == JointType.PRISMATIC:
        lin_count = 1
    elif jt == JointType.REVOLUTE:
        ang_count = 1
    elif jt == JointType.D6:
        lin_count = joint_dof_dim[joint, 0]
        ang_count = joint_dof_dim[joint, 1]
    p_linear, p_angular = build_joint_projectors(jt, joint_axis, qd_start, lin_count, ang_count, q_wp)
    if jt == JointType.BALL:
        p_angular = wp.mat33(0.0)

    stretch_k = wp.max(joint_material_k[c_start], 0.0)
    stretch_rho = wp.max(joint_rho[c_start], 0.0)
    stretch_s, stretch_primal_k, _stretch_a = _compliant_alm_coefficients(stretch_k, stretch_rho)
    stretch_d = wp.max(joint_penalty_kd[c_start], 0.0) / dt
    bend_k = 0.0
    bend_s = 0.0
    bend_primal_k = 0.0
    bend_d = 0.0
    if jt != JointType.BALL:
        bend_k = wp.max(joint_material_k[c_start + 1], 0.0)
        bend_rho = wp.max(joint_rho[c_start + 1], 0.0)
        bend_s, bend_primal_k, _bend_a = _compliant_alm_coefficients(bend_k, bend_rho)
        bend_d = wp.max(joint_penalty_kd[c_start + 1], 0.0) / dt
    stretch_scale = stretch_primal_k + stretch_d
    bend_offset = bend_s * (p_angular * joint_lambda_ang[joint])
    stretch_enabled = stretch_scale > 0.0 and wp.trace(p_linear) > 1.0e-6
    bend_enabled = wp.trace(p_angular) > 1.0e-6 and bend_primal_k + bend_d > 0.0
    jp = wp.spatial_matrix(0.0)
    jc = wp.spatial_matrix(0.0)
    material_compliance = wp.spatial_matrix(0.0)
    c_linear = wp.vec3(0.0)
    c_angular = wp.vec3(0.0)

    if stretch_enabled:
        parent_com_world = wp.vec3(0.0)
        if parent >= 0:
            parent_com_world = wp.transform_point(parent_pose, body_com[parent])
        child_com_world = wp.transform_point(child_pose, body_com[child])
        r_parent = x_p - parent_com_world
        r_child = x_c - child_com_world

        jp = _set_spatial_block(jp, 0, 0, -p_linear)
        jp = _set_spatial_block(jp, 0, 1, p_linear * wp.skew(r_parent))
        jc = _set_spatial_block(jc, 0, 0, p_linear)
        jc = _set_spatial_block(jc, 0, 1, -(p_linear * wp.skew(r_child)))
        stretch_compliance = 1.0 / stretch_scale
        linear_compliance = stretch_compliance * p_linear + (wp.identity(3, float) - p_linear)
        material_compliance = _set_spatial_block(material_compliance, 0, 0, linear_compliance)
        linear_error = p_linear * (x_c - x_p - stab_alpha * joint_C0_lin[joint])
        linear_previous = p_linear * (x_c_prev - x_p_prev)
        # Form the normalized compliant-ALM defect by bounded ratios. This is
        # exactly the force/tangent row used by local VBD, including damping
        # and the current multiplier.
        c_linear = (
            (stretch_primal_k / stretch_scale) * linear_error
            + (stretch_s / stretch_scale) * (p_linear * joint_lambda_lin[joint])
            + (stretch_d / stretch_scale) * (p_linear * (x_c - x_p) - linear_previous)
        )
    else:
        for axis in range(3):
            material_compliance[axis, axis] = 1.0

    if bend_enabled:
        angular_jacobian = p_angular * wp.transpose(angular_jacobian_world)
        jp = _set_spatial_block(jp, 1, 1, -angular_jacobian)
        jc = _set_spatial_block(jc, 1, 1, angular_jacobian)
        angular_error = p_angular * (kappa - stab_alpha * joint_C0_ang[joint])
        angular_rate = p_angular * (kappa - kappa_prev)
        bend_scale = bend_primal_k + bend_d
        angular_compliance = (1.0 / bend_scale) * p_angular + (wp.identity(3, float) - p_angular)
        material_compliance = _set_spatial_block(material_compliance, 1, 1, angular_compliance)
        c_angular = (
            (bend_primal_k / bend_scale) * angular_error
            + (bend_d / bend_scale) * angular_rate
            + bend_offset / bend_scale
        )
    else:
        for axis in range(3):
            material_compliance[axis + 3, axis + 3] = 1.0

    # Free-axis compliance is filled directly below.  Keeping the temporary
    # unit eigenvalue and later applying ``I + (1/tangent - 1) aa^T`` loses
    # ``1/tangent`` to float32 cancellation for near-rigid drives.  Start from
    # only the constrained projector so every active free-axis eigenvalue is
    # written as a positive quantity without subtraction.
    if lin_count > 0:
        constrained_linear_compliance = p_linear
        if stretch_enabled:
            constrained_linear_compliance = (1.0 / stretch_scale) * p_linear
        material_compliance = _set_spatial_block(material_compliance, 0, 0, constrained_linear_compliance)
    if ang_count > 0:
        constrained_angular_compliance = p_angular
        if bend_enabled:
            constrained_angular_compliance = (1.0 / (bend_primal_k + bend_d)) * p_angular
        material_compliance = _set_spatial_block(material_compliance, 1, 1, constrained_angular_compliance)

    # The six-dimensional joint row already has one component for every free
    # joint axis.  Populate those otherwise-neutral components with finite
    # drive or active-limit rows.  This supplies their cross-body tangent to
    # the same KKT solve without adding graph rows or a second solver path.
    target_q_base = joint_target_q_start[joint]
    parent_com_world = wp.vec3(0.0)
    if parent >= 0:
        parent_com_world = wp.transform_point(parent_pose, body_com[parent])
    child_com_world = wp.transform_point(child_pose, body_com[child])
    r_parent = x_p - parent_com_world
    r_child = x_c - child_com_world
    relative_linear = x_c - x_p
    relative_linear_previous = x_c_prev - x_p_prev
    drive_limit_enabled = False

    for free_linear_axis in range(3):
        if free_linear_axis < lin_count:
            dof = qd_start + free_linear_axis
            target_index = target_q_base + free_linear_axis
            axis_world = wp.normalize(wp.quat_rotate(q_wp, joint_axis[dof]))
            coordinate = wp.dot(relative_linear, axis_world)
            coordinate_previous = wp.dot(relative_linear_previous, axis_world)
            target_position = joint_target_q[target_index]
            target_velocity = joint_target_qd[dof]
            drive_k = joint_target_ke[dof]
            drive_d = joint_target_kd[dof]
            lower = joint_limit_lower[dof]
            upper = joint_limit_upper[dof]
            has_drive = drive_k > 0.0 or drive_d > 0.0
            has_limit = joint_limit_ke[dof] > 0.0 and (lower > -MAXVAL or upper < MAXVAL)
            mode, drive_error = _resolve_active_drive_row(
                coordinate,
                target_position,
                lower,
                upper,
                has_drive,
                has_limit,
                1,
            )
            inv_dt = 1.0 / dt
            rate = (coordinate - coordinate_previous) * inv_dt
            force, tangent = _evaluate_drive_axis(
                drive_error,
                rate,
                target_velocity,
                mode,
                drive_k,
                drive_d,
                joint_drive_limit_support[dof],
                joint_drive_lambda[dof],
                inv_dt,
            )
            limit_force, limit_tangent = _evaluate_limit_axis(
                coordinate,
                lower,
                upper,
                has_limit,
                joint_limit_ke[dof],
                joint_limit_kd[dof],
                rate,
                joint_drive_limit_support[dof],
                joint_limit_lambda[dof],
                inv_dt,
            )
            force = force + limit_force
            tangent = tangent + limit_tangent
            normalized_defect = force / tangent if tangent > 0.0 else 0.0
            if tangent > 0.0:
                drive_limit_enabled = True
                axis_projector = wp.outer(axis_world, axis_world)
                jp = _set_spatial_block(jp, 0, 0, _spatial_block(jp, 0, 0) - axis_projector)
                jp = _set_spatial_block(
                    jp,
                    0,
                    1,
                    _spatial_block(jp, 0, 1) + axis_projector * wp.skew(r_parent),
                )
                jc = _set_spatial_block(jc, 0, 0, _spatial_block(jc, 0, 0) + axis_projector)
                jc = _set_spatial_block(
                    jc,
                    0,
                    1,
                    _spatial_block(jc, 0, 1) - axis_projector * wp.skew(r_child),
                )
                linear_compliance = _spatial_block(material_compliance, 0, 0)
                linear_compliance = linear_compliance + (1.0 / tangent) * axis_projector
                material_compliance = _set_spatial_block(material_compliance, 0, 0, linear_compliance)
                c_linear = c_linear + normalized_defect * axis_world
            else:
                linear_compliance = _spatial_block(material_compliance, 0, 0)
                material_compliance = _set_spatial_block(
                    material_compliance,
                    0,
                    0,
                    linear_compliance + wp.outer(axis_world, axis_world),
                )

    if ang_count > 0:
        omega_parent = wp.vec3(0.0)
        if parent >= 0:
            omega_parent = quat_velocity(q_wp, q_wp_prev, dt)
        omega_child = quat_velocity(q_wc, q_wc_prev, dt)
        kappa_rate = compute_kappa_dot(angular_jacobian_world, omega_parent, omega_child)
        for free_angular_axis in range(3):
            if free_angular_axis < ang_count:
                dof = qd_start + lin_count + free_angular_axis
                target_index = target_q_base + lin_count + free_angular_axis
                axis_local = wp.normalize(joint_axis[dof])
                coordinate = wp.dot(kappa, axis_local) + joint_rest_angle[dof]
                target_position = joint_target_q[target_index]
                target_velocity = joint_target_qd[dof]
                drive_k = joint_target_ke[dof]
                drive_d = joint_target_kd[dof]
                lower = joint_limit_lower[dof]
                upper = joint_limit_upper[dof]
                has_drive = drive_k > 0.0 or drive_d > 0.0
                has_limit = joint_limit_ke[dof] > 0.0 and (lower > -MAXVAL or upper < MAXVAL)
                mode, drive_error = _resolve_active_drive_row(
                    coordinate,
                    target_position,
                    lower,
                    upper,
                    has_drive,
                    has_limit,
                    1,
                )
                inv_dt = 1.0 / dt
                rate = wp.dot(kappa_rate, axis_local)
                force, tangent = _evaluate_drive_axis(
                    drive_error,
                    rate,
                    target_velocity,
                    mode,
                    drive_k,
                    drive_d,
                    joint_drive_limit_support[dof],
                    joint_drive_lambda[dof],
                    inv_dt,
                )
                limit_force, limit_tangent = _evaluate_limit_axis(
                    coordinate,
                    lower,
                    upper,
                    has_limit,
                    joint_limit_ke[dof],
                    joint_limit_kd[dof],
                    rate,
                    joint_drive_limit_support[dof],
                    joint_limit_lambda[dof],
                    inv_dt,
                )
                force = force + limit_force
                tangent = tangent + limit_tangent
                normalized_defect = force / tangent if tangent > 0.0 else 0.0
                if tangent > 0.0:
                    drive_limit_enabled = True
                    axis_projector = wp.outer(axis_local, axis_local)
                    angular_jacobian = axis_projector * wp.transpose(angular_jacobian_world)
                    jp = _set_spatial_block(jp, 1, 1, _spatial_block(jp, 1, 1) - angular_jacobian)
                    jc = _set_spatial_block(jc, 1, 1, _spatial_block(jc, 1, 1) + angular_jacobian)
                    angular_compliance = _spatial_block(material_compliance, 1, 1)
                    angular_compliance = angular_compliance + (1.0 / tangent) * axis_projector
                    material_compliance = _set_spatial_block(material_compliance, 1, 1, angular_compliance)
                    c_angular = c_angular + normalized_defect * axis_local
                else:
                    angular_compliance = _spatial_block(material_compliance, 1, 1)
                    material_compliance = _set_spatial_block(
                        material_compliance,
                        1,
                        1,
                        angular_compliance + wp.outer(axis_local, axis_local),
                    )

    jacobian_parent[row] = jp
    jacobian_child[row] = jc
    compliance[row] = material_compliance
    residual[row] = wp.spatial_vector(c_linear, c_angular)
    row_active[row] = 1 if stretch_enabled or bend_enabled or drive_limit_enabled else 0


@wp.kernel
def assemble_joint_path_system(
    row_count: int,
    row_body: wp.array[wp.vec2i],
    body_inverse: wp.array[wp.spatial_matrix],
    body_free: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    compliance: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Assemble one block-tridiagonal constraint-space system per path.

    Each row owns one structural joint. Adjacent path rows are coupled through
    their shared body's inverse surrogate; non-neighbor couplings are absent by
    construction of an eligible path batch.
    """
    row = wp.tid()
    local = row % row_count
    endpoints = row_body[row]
    parent = endpoints[0]
    child = endpoints[1]
    jp = jacobian_parent[row]
    jc = jacobian_child[row]

    diagonal_value = compliance[row]
    rhs_value = residual[row]
    if parent >= 0:
        diagonal_value = diagonal_value + jp * body_inverse[parent] * wp.transpose(jp)
        rhs_value = rhs_value + jp * body_free[parent]
    if child >= 0:
        diagonal_value = diagonal_value + jc * body_inverse[child] * wp.transpose(jc)
        rhs_value = rhs_value + jc * body_free[child]

    lower_value = wp.spatial_matrix(0.0)
    if local > 0:
        previous = row - 1
        previous_endpoints = row_body[previous]
        shared = parent
        if shared < 0 or (shared != previous_endpoints[0] and shared != previous_endpoints[1]):
            shared = child
        current_j = jp if parent == shared else jc
        previous_j = jacobian_parent[previous] if previous_endpoints[0] == shared else jacobian_child[previous]
        lower_value = current_j * body_inverse[shared] * wp.transpose(previous_j)

    upper_value = wp.spatial_matrix(0.0)
    if local + 1 < row_count:
        following = row + 1
        following_endpoints = row_body[following]
        shared = parent
        if shared < 0 or (shared != following_endpoints[0] and shared != following_endpoints[1]):
            shared = child
        current_j = jp if parent == shared else jc
        following_j = jacobian_parent[following] if following_endpoints[0] == shared else jacobian_child[following]
        upper_value = current_j * body_inverse[shared] * wp.transpose(following_j)

    lower[row] = lower_value
    diagonal[row] = diagonal_value
    upper[row] = upper_value
    rhs[row] = rhs_value


@wp.kernel
def invert_cr_eliminated_in_place(
    stride: int,
    row_count: int,
    eliminated_count: int,
    diagonal: wp.array[wp.spatial_matrix],
):
    """Replace eliminated diagonal blocks by their inverses in dead slots."""
    index = wp.tid()
    batch = index // eliminated_count
    local = index - batch * eliminated_count
    row = batch * row_count + stride + 2 * stride * local
    diagonal[row] = _inverse_spatial_robust(diagonal[row])


@wp.func
def _reduce_cr_row(
    base: int,
    local_row: int,
    stride: int,
    row_count: int,
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Reduce one retained CR row after its neighboring pivots are inverted."""
    row = base + local_row
    lower_value = wp.spatial_matrix(0.0)
    diagonal_value = diagonal[row]
    upper_value = wp.spatial_matrix(0.0)
    rhs_value = rhs[row]
    if local_row >= stride:
        eliminated = row - stride
        factor = lower[row] * diagonal[eliminated]
        lower_value = -(factor * lower[eliminated])
        diagonal_value = diagonal_value - factor * upper[eliminated]
        rhs_value = rhs_value - factor * rhs[eliminated]
    if local_row + stride < row_count:
        eliminated = row + stride
        factor = upper[row] * diagonal[eliminated]
        diagonal_value = diagonal_value - factor * lower[eliminated]
        rhs_value = rhs_value - factor * rhs[eliminated]
        if local_row + 2 * stride < row_count:
            upper_value = -(factor * upper[eliminated])

    lower[row] = lower_value
    diagonal[row] = diagonal_value
    upper[row] = upper_value
    rhs[row] = rhs_value


@wp.kernel
def reduce_cr_in_place(
    stride: int,
    row_count: int,
    survivor_count: int,
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Reduce retained CR rows without allocating a second hierarchy."""
    index = wp.tid()
    batch = index // survivor_count
    local = index - batch * survivor_count
    base = batch * row_count
    local_row = 2 * stride * local
    _reduce_cr_row(base, local_row, stride, row_count, lower, diagonal, upper, rhs)


@wp.func
def _solve_strided_cr_coarse_impl(
    base: int,
    stride: int,
    row_count: int,
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Finish one small retained CR system with block Thomas."""
    active_count = (row_count + stride - 1) // stride
    for local in range(1, active_count):
        previous = base + (local - 1) * stride
        row = base + local * stride
        diagonal[previous] = _inverse_spatial_robust(diagonal[previous])
        multiplier = lower[row] * diagonal[previous]
        diagonal[row] = diagonal[row] - multiplier * upper[previous]
        rhs[row] = rhs[row] - multiplier * rhs[previous]

    last = base + (active_count - 1) * stride
    diagonal[last] = _inverse_spatial_robust(diagonal[last])
    rhs[last] = diagonal[last] * rhs[last]
    for reverse_local in range(1, active_count):
        row = base + (active_count - 1 - reverse_local) * stride
        rhs[row] = diagonal[row] * (rhs[row] - upper[row] * rhs[row + stride])


@wp.kernel
def solve_strided_cr_coarse(
    stride: int,
    row_count: int,
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Solve the small retained system in place with block Thomas."""
    base = wp.tid() * row_count
    _solve_strided_cr_coarse_impl(base, stride, row_count, lower, diagonal, upper, rhs)


@wp.func
def _back_substitute_cr_row(
    base: int,
    local_row: int,
    stride: int,
    row_count: int,
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Recover one eliminated CR row from its retained neighbors."""
    row = base + local_row
    value = rhs[row] - lower[row] * rhs[row - stride]
    if local_row + stride < row_count:
        value = value - upper[row] * rhs[row + stride]
    rhs[row] = diagonal[row] * value


@wp.kernel
def back_substitute_cr_in_place(
    stride: int,
    row_count: int,
    eliminated_count: int,
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Recover eliminated rows into rhs, which now stores the solution."""
    index = wp.tid()
    batch = index // eliminated_count
    local = index - batch * eliminated_count
    base = batch * row_count
    local_row = stride + 2 * stride * local
    _back_substitute_cr_row(base, local_row, stride, row_count, lower, diagonal, upper, rhs)


@wp.kernel
def solve_cr_persistent(
    row_count: int,
    terminal_size: int,
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Solve one short path per CUDA block without dependent level launches."""
    tid = wp.tid()
    block_dim = wp.block_dim()
    lane = tid % block_dim
    batch = tid // block_dim
    base = batch * row_count

    # Explicit casts make these loop-carried values dynamic in Warp codegen.
    stride = int(1)
    active_count = int(row_count)
    while active_count > terminal_size:
        eliminated_count = (row_count - stride + 2 * stride - 1) // (2 * stride)
        local = lane
        while local < eliminated_count:
            row = base + stride + 2 * stride * local
            diagonal[row] = _inverse_spatial_robust(diagonal[row])
            local += block_dim
        _synchronize_cr_block()

        survivor_count = (row_count + 2 * stride - 1) // (2 * stride)
        local = lane
        while local < survivor_count:
            _reduce_cr_row(base, 2 * stride * local, stride, row_count, lower, diagonal, upper, rhs)
            local += block_dim
        _synchronize_cr_block()
        stride *= 2
        active_count = survivor_count

    if lane == 0:
        _solve_strided_cr_coarse_impl(base, stride, row_count, lower, diagonal, upper, rhs)
    _synchronize_cr_block()

    stride //= 2
    while stride >= 1:
        eliminated_count = (row_count - stride + 2 * stride - 1) // (2 * stride)
        local = lane
        while local < eliminated_count:
            _back_substitute_cr_row(
                base,
                stride + 2 * stride * local,
                stride,
                row_count,
                lower,
                diagonal,
                upper,
                rhs,
            )
            local += block_dim
        _synchronize_cr_block()
        stride //= 2


@wp.kernel
def initialize_tree_backbone_edges(
    row_count: int,
    backbone_nodes: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling: wp.array[wp.spatial_matrix],
    lower: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
):
    """Gather the two oriented edge blocks of a tree backbone."""
    index = wp.tid()
    batch = index // row_count
    row = index - batch * row_count
    base = batch * row_count
    node = backbone_nodes[index]
    lower_value = wp.spatial_matrix(0.0)
    if row > 0:
        previous = backbone_nodes[base + row - 1]
        if parent_node[node] == previous:
            lower_value = coupling[node]
        else:
            lower_value = wp.transpose(coupling[previous])

    upper_value = wp.spatial_matrix(0.0)
    if row + 1 < row_count:
        following = backbone_nodes[base + row + 1]
        if parent_node[node] == following:
            upper_value = coupling[node]
        else:
            upper_value = wp.transpose(coupling[following])

    lower[index] = lower_value
    upper[index] = upper_value


@wp.kernel
def invert_tree_backbone_cr_eliminated(
    stride: int,
    row_count: int,
    eliminated_count: int,
    backbone_nodes: wp.array[wp.int32],
    diagonal: wp.array[wp.spatial_matrix],
):
    """Invert one CR level directly in the persistent tree-node workspace."""
    index = wp.tid()
    batch = index // eliminated_count
    local = index - batch * eliminated_count
    row = stride + 2 * stride * local
    if local < eliminated_count and row < row_count:
        node = backbone_nodes[batch * row_count + row]
        diagonal[node] = _inverse_spatial_robust(diagonal[node])


@wp.kernel
def reduce_tree_backbone_cr_in_place(
    stride: int,
    row_count: int,
    survivor_count: int,
    closure_count: int,
    backbone_nodes: wp.array[wp.int32],
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
    response_rhs: wp.array[wp.spatial_matrix],
):
    """Reduce the primary and closure-response right-hand sides together."""
    index = wp.tid()
    batch = index // survivor_count
    local = index - batch * survivor_count
    base = batch * row_count
    local_row = 2 * stride * local
    if local >= survivor_count or local_row >= row_count:
        return
    row = base + local_row
    node = backbone_nodes[row]

    lower_value = wp.spatial_matrix(0.0)
    diagonal_value = diagonal[node]
    upper_value = wp.spatial_matrix(0.0)
    rhs_value = rhs[node]
    if local_row >= stride:
        eliminated_row = local_row - stride
        eliminated_index = base + eliminated_row
        eliminated = backbone_nodes[eliminated_index]
        factor = lower[row] * diagonal[eliminated]
        lower_value = -(factor * lower[eliminated_index])
        diagonal_value = diagonal_value - factor * upper[eliminated_index]
        rhs_value = rhs_value - factor * rhs[eliminated]
        for closure in range(closure_count):
            response_index = node * closure_count + closure
            eliminated_response_index = eliminated * closure_count + closure
            response_rhs[response_index] = (
                response_rhs[response_index] - factor * response_rhs[eliminated_response_index]
            )
    if local_row + stride < row_count:
        eliminated_row = local_row + stride
        eliminated_index = base + eliminated_row
        eliminated = backbone_nodes[eliminated_index]
        factor = upper[row] * diagonal[eliminated]
        diagonal_value = diagonal_value - factor * lower[eliminated_index]
        rhs_value = rhs_value - factor * rhs[eliminated]
        for closure in range(closure_count):
            response_index = node * closure_count + closure
            eliminated_response_index = eliminated * closure_count + closure
            response_rhs[response_index] = (
                response_rhs[response_index] - factor * response_rhs[eliminated_response_index]
            )
        if local_row + 2 * stride < row_count:
            upper_value = -(factor * upper[eliminated_index])

    lower[row] = lower_value
    diagonal[node] = diagonal_value
    upper[row] = upper_value
    rhs[node] = rhs_value


@wp.kernel
def solve_tree_backbone_cr_coarse(
    stride: int,
    row_count: int,
    closure_count: int,
    backbone_nodes: wp.array[wp.int32],
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
    response_rhs: wp.array[wp.spatial_matrix],
):
    """Finish the reduced tree backbone with one small block-Thomas solve."""
    base = wp.tid() * row_count
    active_count = (row_count + stride - 1) // stride
    for local in range(1, active_count):
        previous_row = (local - 1) * stride
        row = local * stride
        previous = backbone_nodes[base + previous_row]
        node = backbone_nodes[base + row]
        diagonal[previous] = _inverse_spatial_robust(diagonal[previous])
        multiplier = lower[base + row] * diagonal[previous]
        diagonal[node] = diagonal[node] - multiplier * upper[base + previous_row]
        rhs[node] = rhs[node] - multiplier * rhs[previous]
        for closure in range(closure_count):
            response_index = node * closure_count + closure
            previous_response_index = previous * closure_count + closure
            response_rhs[response_index] = (
                response_rhs[response_index] - multiplier * response_rhs[previous_response_index]
            )

    last_row = (active_count - 1) * stride
    last = backbone_nodes[base + last_row]
    diagonal[last] = _inverse_spatial_robust(diagonal[last])
    rhs[last] = diagonal[last] * rhs[last]
    for closure in range(closure_count):
        response_index = last * closure_count + closure
        response_rhs[response_index] = diagonal[last] * response_rhs[response_index]

    for reverse_local in range(1, active_count):
        row = (active_count - 1 - reverse_local) * stride
        following_row = row + stride
        node = backbone_nodes[base + row]
        following = backbone_nodes[base + following_row]
        rhs[node] = diagonal[node] * (rhs[node] - upper[base + row] * rhs[following])
        for closure in range(closure_count):
            response_index = node * closure_count + closure
            following_response_index = following * closure_count + closure
            response_rhs[response_index] = diagonal[node] * (
                response_rhs[response_index] - upper[base + row] * response_rhs[following_response_index]
            )


@wp.kernel
def back_substitute_tree_backbone_cr_in_place(
    stride: int,
    row_count: int,
    eliminated_count: int,
    closure_count: int,
    backbone_nodes: wp.array[wp.int32],
    lower: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
    response_rhs: wp.array[wp.spatial_matrix],
):
    """Recover one eliminated backbone level for every right-hand side."""
    index = wp.tid()
    batch = index // eliminated_count
    local = index - batch * eliminated_count
    base = batch * row_count
    local_row = stride + 2 * stride * local
    if local >= eliminated_count or local_row >= row_count:
        return
    row = base + local_row
    node = backbone_nodes[row]
    previous = backbone_nodes[base + local_row - stride]
    rhs_value = rhs[node] - lower[row] * rhs[previous]
    if local_row + stride < row_count:
        following = backbone_nodes[base + local_row + stride]
        rhs_value = rhs_value - upper[row] * rhs[following]
    rhs[node] = diagonal[node] * rhs_value

    for closure in range(closure_count):
        response_index = node * closure_count + closure
        previous_response_index = previous * closure_count + closure
        response_value = response_rhs[response_index] - lower[row] * response_rhs[previous_response_index]
        if local_row + stride < row_count:
            following = backbone_nodes[base + local_row + stride]
            following_response_index = following * closure_count + closure
            response_value = response_value - upper[row] * response_rhs[following_response_index]
        response_rhs[response_index] = diagonal[node] * response_value


@wp.kernel
def compute_path_correction(
    row_count: int,
    body_count: int,
    body_ids: wp.array[wp.int32],
    body_slot_by_id: wp.array[wp.int32],
    body_incident_rows: wp.array[wp.vec2i],
    row_active: wp.array[wp.int32],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    multiplier: wp.array[wp.spatial_vector],
    body_inverse: wp.array[wp.spatial_matrix],
    body_free: wp.array[wp.spatial_vector],
    body_inv_mass: wp.array[float],
    body_correction: wp.array[wp.spatial_vector],
):
    """Map solved path multipliers back to one coherent body correction."""
    index = wp.tid()
    batch = index // body_count
    row_start = batch * row_count
    body = body_ids[index]
    slot = body_slot_by_id[body]
    if body_inv_mass[body] <= 0.0:
        return

    reaction = wp.spatial_vector()
    active = 0
    incident_rows = body_incident_rows[index]
    for incident in range(2):
        code = incident_rows[incident]
        if code >= 0:
            row = row_start + code // 2
            jacobian = jacobian_parent[row] if code % 2 == 0 else jacobian_child[row]
            reaction = reaction + wp.transpose(jacobian) * multiplier[row]
            active = active + row_active[row]
    if active == 0:
        body_correction[slot] = wp.spatial_vector()
        return

    body_correction[slot] = body_free[slot] - body_inverse[slot] * reaction


@wp.func
def _corrected_pose(
    pose: wp.transform,
    correction: wp.spatial_vector,
    com_local: wp.vec3,
    alpha: float,
):
    linear = alpha * wp.spatial_top(correction)
    angular = alpha * wp.spatial_bottom(correction)
    rotation = wp.transform_get_rotation(pose)
    com = wp.transform_point(pose, com_local)

    angle = wp.length(angular)
    delta_rotation = wp.quat_identity()
    if angle > _SMALL_ANGLE_EPS:
        delta_rotation = wp.quat_from_axis_angle(angular / angle, angle)
    else:
        half = 0.5 * angular
        delta_rotation = wp.normalize(wp.quat(half[0], half[1], half[2], 1.0))
    rotation_new = wp.normalize(delta_rotation * rotation)
    com_new = com + linear
    position_new = com_new - wp.quat_rotate(rotation_new, com_local)
    return wp.transform(position_new, rotation_new)


@wp.func
def _fraction_to_interval(value: float, value_corrected: float, lower: float, upper: float):
    """Return the largest step in [0, 1] that does not cross a bound."""
    delta = value_corrected - value
    if value < lower:
        if delta <= 0.0:
            return 0.0
        if value_corrected >= lower:
            return _JOINT_LIMIT_STEP_FRACTION * wp.clamp((lower - value) / delta, 0.0, 1.0)
    elif value > upper:
        if delta >= 0.0:
            return 0.0
        if value_corrected <= upper:
            return _JOINT_LIMIT_STEP_FRACTION * wp.clamp((upper - value) / delta, 0.0, 1.0)
    else:
        if value_corrected < lower and delta < 0.0:
            return _JOINT_LIMIT_STEP_FRACTION * wp.clamp((lower - value) / delta, 0.0, 1.0)
        if value_corrected > upper and delta > 0.0:
            return _JOINT_LIMIT_STEP_FRACTION * wp.clamp((upper - value) / delta, 0.0, 1.0)
    return 1.0


@wp.kernel
def limit_dynamic_contact_jacobi_step(
    body_ids: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    island_contact_state: wp.array[wp.int32],
    correction: wp.array[wp.spatial_vector],
    contact_hessian_ll: wp.array[wp.mat33],
    contact_hessian_al: wp.array[wp.mat33],
    contact_hessian_aa: wp.array[wp.mat33],
    dt: float,
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    island_step_scale: wp.array[float],
):
    """Relax a structural correction by its dynamic-contact overlap.

    The global structural islands are solved concurrently.  At an active
    dynamic pair, each endpoint therefore responds as if the other endpoint
    were fixed.  This is block Jacobi on the omitted contact edge.  Weight its
    standard half relaxation by the fraction of the proposed correction's
    local quadratic curvature that comes from those dynamic contacts:

        omega = 1 / (1 + contact_curvature / total_curvature).

    The coefficient is one without dynamic-contact influence and approaches
    one half when contact dominates.  It contains no scene scale or authored
    tolerance, and local VBD relinearizes the nonlinear contact afterward.
    """
    slot = wp.tid()
    island = graph_body_island[slot]
    if island_contact_state[island] >= 0:
        return

    body = body_ids[slot]
    value = correction[slot]
    linear = wp.spatial_top(value)
    angular = wp.spatial_bottom(value)
    contact_curvature = 2.0 * (
        wp.dot(linear, contact_hessian_ll[body] * linear)
        + 2.0 * wp.dot(angular, contact_hessian_al[body] * linear)
        + wp.dot(angular, contact_hessian_aa[body] * angular)
    )
    contact_curvature = wp.max(contact_curvature, 0.0)

    dt_inv_sq = 1.0 / (dt * dt)
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(body_q[body]))
    inertia_world = rotation * body_inertia[body] * wp.transpose(rotation)
    inertial_curvature = dt_inv_sq * (
        body_mass[body] * wp.dot(linear, linear) + wp.dot(angular, inertia_world * angular)
    )
    total_curvature = inertial_curvature + contact_curvature
    if total_curvature > 0.0:
        overlap = wp.clamp(contact_curvature / total_curvature, 0.0, 1.0)
        wp.atomic_min(island_step_scale, island, 1.0 / (1.0 + overlap))


@wp.kernel
def limit_global_joint_limit_step(
    joint_ids: wp.array[wp.int32],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_rest_angle: wp.array[float],
    body_slot: wp.array[int],
    graph_body_island: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    correction: wp.array[wp.spatial_vector],
    island_step_scale: wp.array[float],
):
    """Keep a unilateral joint-limit linearization in its current active set.

    An active finite limit is a one-sided row.  If the global Newton correction
    would cross its boundary, accept only the fraction that reaches the
    boundary.  Corrections that reduce violation without crossing remain
    unrestricted.  The island-wide minimum preserves structural coherence.
    """
    index = wp.tid()
    joint = joint_ids[index]
    jt = joint_type[joint]
    if not joint_enabled[joint] or not (jt == JointType.REVOLUTE or jt == JointType.PRISMATIC or jt == JointType.D6):
        return

    parent = joint_parent[joint]
    child = joint_child[joint]
    child_slot = body_slot[child]
    if child_slot < 0:
        return
    island = graph_body_island[child_slot]

    parent_pose = wp.transform_identity()
    parent_rest = parent_pose
    parent_corrected = parent_pose
    if parent >= 0:
        parent_pose = body_q[parent]
        parent_rest = body_q_rest[parent]
        parent_slot = body_slot[parent]
        parent_corrected = parent_pose
        if parent_slot >= 0:
            parent_corrected = _corrected_pose(
                parent_pose,
                correction[parent_slot],
                body_com[parent],
                1.0,
            )
    child_pose = body_q[child]
    child_corrected = _corrected_pose(child_pose, correction[child_slot], body_com[child], 1.0)
    child_rest = body_q_rest[child]

    X_wp = parent_pose * joint_X_p[joint]
    X_wc = child_pose * joint_X_c[joint]
    X_wp_corrected = parent_corrected * joint_X_p[joint]
    X_wc_corrected = child_corrected * joint_X_c[joint]
    X_wp_rest = parent_rest * joint_X_p[joint]
    X_wc_rest = child_rest * joint_X_c[joint]

    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_wp_corrected = wp.transform_get_rotation(X_wp_corrected)
    q_wc_corrected = wp.transform_get_rotation(X_wc_corrected)
    q_wp_rest = wp.transform_get_rotation(X_wp_rest)
    q_wc_rest = wp.transform_get_rotation(X_wc_rest)

    lin_count = 0
    ang_count = 0
    if jt == JointType.PRISMATIC:
        lin_count = 1
    elif jt == JointType.REVOLUTE:
        ang_count = 1
    elif jt == JointType.D6:
        lin_count = joint_dof_dim[joint, 0]
        ang_count = joint_dof_dim[joint, 1]
    qd_start = joint_qd_start[joint]

    relative_linear = wp.transform_get_translation(X_wc) - wp.transform_get_translation(X_wp)
    relative_linear_corrected = wp.transform_get_translation(X_wc_corrected) - wp.transform_get_translation(
        X_wp_corrected
    )
    for axis_index in range(3):
        if axis_index < lin_count:
            dof = qd_start + axis_index
            if joint_limit_ke[dof] > 0.0:
                axis = wp.normalize(wp.quat_rotate(q_wp, joint_axis[dof]))
                axis_corrected = wp.normalize(wp.quat_rotate(q_wp_corrected, joint_axis[dof]))
                value = wp.dot(relative_linear, axis)
                value_corrected = wp.dot(relative_linear_corrected, axis_corrected)
                lower = joint_limit_lower[dof]
                upper = joint_limit_upper[dof]
                bound = _fraction_to_interval(value, value_corrected, lower, upper)
                wp.atomic_min(island_step_scale, island, bound)

    if ang_count > 0:
        kappa = compute_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        kappa_corrected = compute_kappa(q_wp_corrected, q_wc_corrected, q_wp_rest, q_wc_rest)
        for axis_index in range(3):
            if axis_index < ang_count:
                dof = qd_start + lin_count + axis_index
                if joint_limit_ke[dof] > 0.0:
                    axis = wp.normalize(joint_axis[dof])
                    value = wp.dot(kappa, axis) + joint_rest_angle[dof]
                    value_corrected = wp.dot(kappa_corrected, axis) + joint_rest_angle[dof]
                    lower = joint_limit_lower[dof]
                    upper = joint_limit_upper[dof]
                    bound = _fraction_to_interval(value, value_corrected, lower, upper)
                    wp.atomic_min(island_step_scale, island, bound)


@wp.kernel
def apply_global_correction(
    body_ids: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    correction: wp.array[wp.spatial_vector],
    island_step_scale: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
):
    slot = wp.tid()
    body = body_ids[slot]
    island = graph_body_island[slot]
    scale = island_step_scale[island]
    body_q[body] = _corrected_pose(body_q[body], correction[slot], body_com[body], scale)


@dataclass
class _Path:
    joints: list[int]
    bodies: list[int]
    body_rows: list[list[tuple[int, int]]]


@dataclass
class _Tree:
    joints: list[int]
    bodies: list[int]
    node_body: list[int]
    node_row: list[int]
    parent_node: list[int]
    coupling_side: list[int]
    elimination_levels: list[list[int]]
    root: int


@dataclass
class _ClosedTree:
    """A complete structural island represented by a tree plus closure rows."""

    tree: _Tree
    closure_joints: list[int]
    joints: list[int]
    bodies: list[int]


def _tree_backbone_partition(tree: _Tree) -> tuple[list[int], list[list[int]]]:
    """Split a rooted tree into one long backbone and shallow side branches.

    The backbone joins the deepest two root branches, so it contains the root
    and preserves the existing parent orientation. Side branches can therefore
    be eliminated toward the backbone before a logarithmic CR solve.
    """
    if len(tree.parent_node) <= 1:
        return [tree.root], []

    children: list[list[int]] = [[] for _ in tree.parent_node]
    for node, parent in enumerate(tree.parent_node):
        if parent >= 0:
            children[parent].append(node)

    depth = [-1] * len(tree.parent_node)
    root_branch = [-1] * len(tree.parent_node)
    depth[tree.root] = 0
    traversal = [tree.root]
    for parent in traversal:
        for node in sorted(children[parent]):
            depth[node] = depth[parent] + 1
            root_branch[node] = node if parent == tree.root else root_branch[parent]
            traversal.append(node)

    root_branches: dict[int, tuple[int, int]] = {}
    for node in traversal:
        if node == tree.root:
            continue
        root_child = root_branch[node]
        previous = root_branches.get(root_child)
        if previous is None or depth[node] > previous[0] or (depth[node] == previous[0] and node < previous[1]):
            root_branches[root_child] = (depth[node], node)

    endpoints = sorted(root_branches.values(), key=lambda item: (-item[0], item[1]))[:2]

    def path_to_root(endpoint: int) -> list[int]:
        path = [endpoint]
        while path[-1] != tree.root:
            path.append(tree.parent_node[path[-1]])
        return path

    first = path_to_root(endpoints[0][1]) if endpoints else [tree.root]
    second = path_to_root(endpoints[1][1]) if len(endpoints) > 1 else [tree.root]
    backbone = [*first, *reversed(second[:-1])]
    backbone_set = set(backbone)

    branch_depth = [0] * len(tree.parent_node)
    for node in traversal:
        if node not in backbone_set:
            branch_depth[node] = branch_depth[tree.parent_node[node]] + 1

    levels = [
        sorted(node for node, node_depth in enumerate(branch_depth) if node_depth == level)
        for level in range(max(branch_depth, default=0), 0, -1)
    ]
    return backbone, levels


def _make_device_tree_levels(levels: list[list[int]], parent_node: list[int], device):
    """Encode a host leaf schedule for graph-capturable tree elimination."""
    result = []
    for level in levels:
        leaves = wp.array(level, dtype=wp.int32, device=device)
        if len({parent_node[node] for node in level}) == len(level):
            result.append((leaves, None, None, None, True))
            continue
        grouped: dict[int, list[int]] = defaultdict(list)
        for node in level:
            grouped[parent_node[node]].append(node)
        recipients = sorted(grouped)
        message_nodes: list[int] = []
        offsets = [0]
        for recipient in recipients:
            message_nodes.extend(sorted(grouped[recipient]))
            offsets.append(len(message_nodes))
        result.append(
            (
                leaves,
                wp.array(recipients, dtype=wp.int32, device=device),
                wp.array(offsets, dtype=wp.int32, device=device),
                wp.array(message_nodes, dtype=wp.int32, device=device),
                False,
            )
        )
    return result


def _tree_level_launch_count(levels: list[list[int]], parent_node: list[int]) -> int:
    """Count dependent factor/back-substitution launches for one tree schedule."""
    forward = sum(1 if len({parent_node[node] for node in level}) == len(level) else 2 for level in levels)
    return forward + len(levels)


@dataclass
class _TreeContractionLevel:
    """One immutable exact rake/compress level before device encoding."""

    nodes: list[int]
    neighbors: list[tuple[int, int]]
    edges: list[tuple[int, int]]
    transpose_edges: list[tuple[int, int]]
    generated_edges: list[int]
    recipients: list[int]
    recipient_offsets: list[int]
    message_slots: list[int]


@dataclass
class _TreeContraction:
    """Host schedule for exact independent-node tree contraction."""

    levels: list[_TreeContractionLevel]
    root: int
    edge_count: int


def _tree_contraction_schedule(tree: _Tree) -> _TreeContraction:
    """Build an exact logarithmic-depth rake/compress schedule.

    Every selected node has current degree one or two, and selected nodes are
    independent. Eliminating a degree-two node creates one fill edge between
    its surviving neighbors. The symbolic edge and message layout is fixed at
    construction, so the numeric pass remains graph-capturable.
    """
    node_count = len(tree.parent_node)
    adjacency: list[dict[int, int]] = [{} for _ in range(node_count)]
    edge_u = [-1] * node_count
    edge_v = [-1] * node_count
    for node, parent in enumerate(tree.parent_node):
        if parent < 0:
            continue
        # Original edge ``node`` stores A(node, parent), matching coupling.
        edge_u[node] = node
        edge_v[node] = parent
        adjacency[node][parent] = node
        adjacency[parent][node] = node

    next_edge = node_count
    active = set(range(node_count))
    levels: list[_TreeContractionLevel] = []
    round_index = 0
    while len(active) > 1:
        candidates = [node for node in active if 0 < len(adjacency[node]) <= 2]
        # Rake leaves first. Rotate the deterministic degree-two ordering so a
        # path does not repeatedly favor one end when node identifiers happen
        # to be monotone along it.
        candidates.sort(
            key=lambda node: (
                len(adjacency[node]) != 1,
                (node - round_index) % node_count,
            )
        )
        selected: list[int] = []
        blocked: set[int] = set()
        for node in candidates:
            if node in blocked:
                continue
            selected.append(node)
            blocked.add(node)
            blocked.update(adjacency[node])
        if len(selected) >= len(active):
            selected.pop()
        if not selected:
            raise RuntimeError("Unable to construct an independent tree-contraction level")

        nodes: list[int] = []
        neighbors: list[tuple[int, int]] = []
        edges: list[tuple[int, int]] = []
        transpose_edges: list[tuple[int, int]] = []
        generated_edges: list[int] = []
        grouped_messages: dict[int, list[int]] = defaultdict(list)
        generated: list[tuple[int, int, int]] = []
        for level_index, node in enumerate(sorted(selected)):
            incident = sorted(adjacency[node].items())
            local_neighbors = [neighbor for neighbor, _ in incident]
            local_edges = [edge for _, edge in incident]
            local_transpose = [int(edge_u[edge] != node) for edge in local_edges]
            while len(local_neighbors) < 2:
                local_neighbors.append(-1)
                local_edges.append(-1)
                local_transpose.append(0)

            generated_edge = -1
            if len(incident) == 2:
                generated_edge = next_edge
                next_edge += 1
                first, second = local_neighbors
                edge_u.append(first)
                edge_v.append(second)
                generated.append((first, second, generated_edge))

            nodes.append(node)
            neighbors.append((local_neighbors[0], local_neighbors[1]))
            edges.append((local_edges[0], local_edges[1]))
            transpose_edges.append((local_transpose[0], local_transpose[1]))
            generated_edges.append(generated_edge)
            for local, neighbor in enumerate(local_neighbors):
                if neighbor >= 0:
                    grouped_messages[neighbor].append(2 * level_index + local)

        # Remove the independent nodes, then insert their degree-two fill
        # edges. Every neighbor is a survivor because selected nodes are not
        # adjacent, so no generated edge can reference a dead node.
        for node in selected:
            for neighbor in list(adjacency[node]):
                adjacency[neighbor].pop(node)
            adjacency[node].clear()
        active.difference_update(selected)
        for first, second, edge in generated:
            adjacency[first][second] = edge
            adjacency[second][first] = edge

        recipients = sorted(grouped_messages)
        recipient_offsets = [0]
        message_slots: list[int] = []
        for recipient in recipients:
            message_slots.extend(sorted(grouped_messages[recipient]))
            recipient_offsets.append(len(message_slots))
        levels.append(
            _TreeContractionLevel(
                nodes,
                neighbors,
                edges,
                transpose_edges,
                generated_edges,
                recipients,
                recipient_offsets,
                message_slots,
            )
        )
        round_index += 1

    return _TreeContraction(levels, next(iter(active)), next_edge)


def _make_device_batched_tree_contraction(
    schedule: _TreeContraction,
    batch_count: int,
    node_count: int,
    device,
):
    """Repeat one normalized contraction schedule with compact edge offsets."""
    fill_count = schedule.edge_count - node_count
    result = []
    for level in schedule.levels:
        level_size = len(level.nodes)
        nodes: list[int] = []
        neighbors: list[tuple[int, int]] = []
        edges: list[tuple[int, int]] = []
        transpose_edges: list[tuple[int, int]] = []
        generated_edges: list[int] = []
        recipients: list[int] = []
        recipient_offsets = [0]
        message_slots: list[int] = []

        def global_edge(batch: int, edge: int) -> int:
            if edge < node_count:
                return batch * node_count + edge
            return batch_count * node_count + batch * fill_count + edge - node_count

        for batch in range(batch_count):
            node_offset = batch * node_count
            nodes.extend(node_offset + node for node in level.nodes)
            neighbors.extend(
                tuple(node_offset + neighbor if neighbor >= 0 else -1 for neighbor in pair) for pair in level.neighbors
            )
            edges.extend(tuple(global_edge(batch, edge) if edge >= 0 else -1 for edge in pair) for pair in level.edges)
            transpose_edges.extend(level.transpose_edges)
            generated_edges.extend(global_edge(batch, edge) if edge >= 0 else -1 for edge in level.generated_edges)
            for recipient_index, recipient in enumerate(level.recipients):
                recipients.append(node_offset + recipient)
                begin = level.recipient_offsets[recipient_index]
                end = level.recipient_offsets[recipient_index + 1]
                for slot in level.message_slots[begin:end]:
                    local_index = slot // 2
                    local = slot - 2 * local_index
                    message_slots.append(2 * (batch * level_size + local_index) + local)
                recipient_offsets.append(len(message_slots))
        result.append(
            (
                wp.array(nodes, dtype=wp.int32, device=device),
                wp.array(neighbors, dtype=wp.vec2i, device=device),
                wp.array(edges, dtype=wp.vec2i, device=device),
                wp.array(transpose_edges, dtype=wp.vec2i, device=device),
                wp.array(generated_edges, dtype=wp.int32, device=device),
                wp.array(recipients, dtype=wp.int32, device=device),
                wp.array(recipient_offsets, dtype=wp.int32, device=device),
                wp.array(message_slots, dtype=wp.int32, device=device),
            )
        )
    return result


def _tree_topology_signature(tree: _Tree) -> tuple:
    """Return the normalized symbolic layout used for safe numeric batching."""
    return (
        len(tree.bodies),
        len(tree.joints),
        tuple(tree.parent_node),
        tuple(tree.coupling_side),
    )


def _closed_tree_topology_signature(
    component: _ClosedTree,
    parent: np.ndarray,
    child: np.ndarray,
) -> tuple:
    """Return the normalized spanning-tree and closure endpoint layout."""
    body_node = {body: node for node, body in enumerate(component.tree.bodies)}
    closure_endpoints = tuple(
        (body_node.get(int(parent[joint]), -1), body_node.get(int(child[joint]), -1))
        for joint in component.closure_joints
    )
    return _tree_topology_signature(component.tree), closure_endpoints


def _eligible_structural_joints(model) -> list[int]:
    """Return supported enabled joints whose complete objective is represented."""
    joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int64)
    joint_enabled = np.asarray(model.joint_enabled.numpy(), dtype=bool)
    child = np.asarray(model.joint_child.numpy(), dtype=np.int64)
    supported = {
        int(JointType.CABLE),
        int(JointType.BALL),
        int(JointType.FIXED),
        int(JointType.REVOLUTE),
        int(JointType.PRISMATIC),
        int(JointType.D6),
    }
    result: list[int] = []
    for joint, kind in enumerate(joint_type):
        if kind not in supported or not joint_enabled[joint] or child[joint] < 0:
            continue
        result.append(joint)
    return result


def _build_paths(model, body_inv_mass: np.ndarray) -> list[_Path]:
    """Return supported structural components whose line graph is a path."""
    joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int64)
    joint_enabled = np.asarray(model.joint_enabled.numpy(), dtype=bool)
    parent = np.asarray(model.joint_parent.numpy(), dtype=np.int64)
    child = np.asarray(model.joint_child.numpy(), dtype=np.int64)
    structural_joints = _eligible_structural_joints(model)
    if not structural_joints:
        return []

    body_to_joints: dict[int, list[int]] = defaultdict(list)
    for joint in structural_joints:
        if parent[joint] >= 0:
            body_to_joints[int(parent[joint])].append(joint)
        body_to_joints[int(child[joint])].append(joint)

    all_body_to_joints: dict[int, set[int]] = defaultdict(set)
    for joint, enabled in enumerate(joint_enabled):
        if not enabled or child[joint] < 0:
            continue
        if parent[joint] >= 0:
            all_body_to_joints[int(parent[joint])].add(joint)
        all_body_to_joints[int(child[joint])].add(joint)

    neighbors = {joint: set() for joint in structural_joints}
    for incident in body_to_joints.values():
        for joint in incident:
            neighbors[joint].update(other for other in incident if other != joint)

    paths: list[_Path] = []
    unvisited = set(structural_joints)
    while unvisited:
        seed = min(unvisited)
        component: set[int] = set()
        stack = [seed]
        while stack:
            joint = stack.pop()
            if joint in component:
                continue
            component.add(joint)
            unvisited.discard(joint)
            stack.extend(neighbors[joint] - component)

        # The compliance Schur path is fastest for cable chains, including a
        # world-attachment row at a boundary. General rigid and dynamically
        # mixed chains use the equilibrated primal-dual tree factorization,
        # which remains stable near the hard-joint limit.
        cable_joints = [joint for joint in component if joint_type[joint] == int(JointType.CABLE)]
        non_cable_joints = component - set(cable_joints)
        if non_cable_joints and (not cable_joints or any(parent[joint] >= 0 for joint in non_cable_joints)):
            continue

        component_bodies = {
            body for joint in component for body in (int(parent[joint]), int(child[joint])) if body >= 0
        }
        if not any(body_inv_mass[body] > 0.0 for body in component_bodies):
            continue
        if any(all_body_to_joints[body] - component for body in component_bodies):
            # A path correction must contain the body's complete persistent
            # structural neighborhood. Otherwise it can undo an attachment,
            # drive, or another joint that the KKT system did not assemble.
            continue

        if any(len(neighbors[joint] & component) > 2 for joint in component):
            continue
        edge_count = sum(len(neighbors[joint] & component) for joint in component) // 2
        if edge_count != len(component) - 1:
            continue

        starts = [joint for joint in component if len(neighbors[joint] & component) <= 1]
        if not starts:
            continue
        start = min(starts, key=lambda joint: (parent[joint] >= 0, joint))
        ordered: list[int] = []
        previous = -1
        current = start
        while current >= 0:
            ordered.append(current)
            following = sorted((neighbors[current] & component) - {previous})
            previous, current = current, (following[0] if following else -1)
        if len(ordered) != len(component):
            continue

        valid = True
        for left, right in pairwise(ordered):
            left_bodies = {int(parent[left]), int(child[left])} - {-1}
            right_bodies = {int(parent[right]), int(child[right])} - {-1}
            shared = left_bodies & right_bodies
            if len(shared) != 1:
                valid = False
                break
        if not valid:
            continue

        bodies: list[int] = []
        for joint in ordered:
            for body in (int(parent[joint]), int(child[joint])):
                if body >= 0 and body not in bodies:
                    bodies.append(body)
        body_rows: list[list[tuple[int, int]]] = []
        for body in bodies:
            incident_rows: list[tuple[int, int]] = []
            for row, joint in enumerate(ordered):
                if int(parent[joint]) == body:
                    incident_rows.append((row, 0))
                if int(child[joint]) == body:
                    incident_rows.append((row, 1))
            if len(incident_rows) > 2:
                valid = False
                break
            body_rows.append(incident_rows)
        if valid:
            paths.append(_Path(ordered, bodies, body_rows))
    return paths


def _make_tree(
    joints: list[int],
    parent: np.ndarray,
    child: np.ndarray,
    body_inv_mass: np.ndarray,
) -> _Tree | None:
    """Build the bipartite body-joint tree used by the exact KKT solve."""
    bodies = sorted(
        {
            body
            for joint in joints
            for body in (int(parent[joint]), int(child[joint]))
            if body >= 0 and body_inv_mass[body] > 0.0
        }
    )
    if not bodies:
        return None

    body_node = {body: index for index, body in enumerate(bodies)}
    joint_node = {joint: len(bodies) + row for row, joint in enumerate(joints)}
    node_count = len(bodies) + len(joints)
    adjacency: list[set[int]] = [set() for _ in range(node_count)]
    edge_count = 0
    for joint in joints:
        node = joint_node[joint]
        for body in (int(parent[joint]), int(child[joint])):
            endpoint = body_node.get(body)
            if endpoint is None:
                continue
            adjacency[node].add(endpoint)
            adjacency[endpoint].add(node)
            edge_count += 1
        if not adjacency[node]:
            return None
    if edge_count != node_count - 1:
        return None

    reached = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node] - reached:
            reached.add(neighbor)
            stack.append(neighbor)
    if len(reached) != node_count:
        return None

    # In a tree, a node's eccentricity is the larger distance to the two
    # diameter endpoints. Three linear traversals therefore find the best
    # dynamic-body root; evaluating one BFS per body would be quadratic.
    def distances(start: int) -> list[int]:
        result = [-1] * node_count
        result[start] = 0
        queue = [start]
        for node in queue:
            for neighbor in adjacency[node]:
                if result[neighbor] < 0:
                    result[neighbor] = result[node] + 1
                    queue.append(neighbor)
        return result

    first_distance = distances(0)
    first = max(range(node_count), key=lambda node: (first_distance[node], -node))
    distance_from_first = distances(first)
    second = max(range(node_count), key=lambda node: (distance_from_first[node], -node))
    distance_from_second = distances(second)
    root = min(
        range(len(bodies)),
        key=lambda node: (max(distance_from_first[node], distance_from_second[node]), bodies[node]),
    )
    active = set(range(node_count))
    parent_node = [-1] * node_count
    elimination_levels: list[list[int]] = []
    while len(active) > 1:
        leaves = sorted(node for node in active if node != root and len(adjacency[node] & active) == 1)
        if not leaves:
            return None
        for node in leaves:
            parent_node[node] = next(iter(adjacency[node] & active))
        elimination_levels.append(leaves)
        active.difference_update(leaves)
    if active != {root}:
        return None

    node_body = [*bodies, *([-1] * len(joints))]
    node_row = [*([-1] * len(bodies)), *range(len(joints))]
    coupling_side = [-1] * node_count
    for node in range(node_count):
        if node == root:
            continue
        neighbor = parent_node[node]
        row = node_row[node] if node_row[node] >= 0 else node_row[neighbor]
        body = node_body[node] if node_body[node] >= 0 else node_body[neighbor]
        joint = joints[row]
        coupling_side[node] = 0 if int(parent[joint]) == body else 1

    return _Tree(
        joints,
        bodies,
        node_body,
        node_row,
        parent_node,
        coupling_side,
        elimination_levels,
        root,
    )


def _union_find_root(representative: dict[int, int], body: int) -> int:
    """Return and path-compress one disjoint-set representative."""
    while representative[body] != body:
        representative[body] = representative[representative[body]]
        body = representative[body]
    return body


def _build_trees(
    model,
    path_joint_sets: set[frozenset[int]],
    body_inv_mass: np.ndarray,
) -> list[_Tree]:
    """Return eligible branched structural trees not handled by path CR."""
    joint_enabled = np.asarray(model.joint_enabled.numpy(), dtype=bool)
    parent = np.asarray(model.joint_parent.numpy(), dtype=np.int64)
    child = np.asarray(model.joint_child.numpy(), dtype=np.int64)
    structural_joints = _eligible_structural_joints(model)
    if not structural_joints:
        return []

    body_to_structural: dict[int, list[int]] = defaultdict(list)
    for joint in structural_joints:
        if parent[joint] >= 0:
            body_to_structural[int(parent[joint])].append(joint)
        body_to_structural[int(child[joint])].append(joint)

    all_body_to_joints: dict[int, set[int]] = defaultdict(set)
    for joint, enabled in enumerate(joint_enabled):
        if not enabled or child[joint] < 0:
            continue
        if parent[joint] >= 0:
            all_body_to_joints[int(parent[joint])].add(joint)
        all_body_to_joints[int(child[joint])].add(joint)

    joint_neighbors = {joint: set() for joint in structural_joints}
    for incident in body_to_structural.values():
        for joint in incident:
            joint_neighbors[joint].update(other for other in incident if other != joint)

    trees: list[_Tree] = []
    unvisited = set(structural_joints)
    while unvisited:
        seed = min(unvisited)
        component: set[int] = set()
        stack = [seed]
        while stack:
            joint = stack.pop()
            if joint in component:
                continue
            component.add(joint)
            unvisited.discard(joint)
            stack.extend(joint_neighbors[joint] - component)

        if frozenset(component) in path_joint_sets:
            continue
        component_bodies = {
            body for joint in component for body in (int(parent[joint]), int(child[joint])) if body >= 0
        }
        if any(all_body_to_joints[body] - component for body in component_bodies):
            continue

        tree = _make_tree(sorted(component), parent, child, body_inv_mass)
        if tree is not None:
            trees.append(tree)
    return trees


def _build_closed_trees(
    model,
    handled_joint_sets: set[frozenset[int]],
    body_inv_mass: np.ndarray,
) -> list[_ClosedTree]:
    """Represent complete cyclic islands as a spanning tree plus closure rows."""
    joint_enabled = np.asarray(model.joint_enabled.numpy(), dtype=bool)
    parent = np.asarray(model.joint_parent.numpy(), dtype=np.int64)
    child = np.asarray(model.joint_child.numpy(), dtype=np.int64)
    structural_joints = _eligible_structural_joints(model)
    if not structural_joints:
        return []

    body_to_structural: dict[int, list[int]] = defaultdict(list)
    for joint in structural_joints:
        if parent[joint] >= 0:
            body_to_structural[int(parent[joint])].append(joint)
        body_to_structural[int(child[joint])].append(joint)

    all_body_to_joints: dict[int, set[int]] = defaultdict(set)
    for joint, enabled in enumerate(joint_enabled):
        if not enabled or child[joint] < 0:
            continue
        if parent[joint] >= 0:
            all_body_to_joints[int(parent[joint])].add(joint)
        all_body_to_joints[int(child[joint])].add(joint)

    neighbors = {joint: set() for joint in structural_joints}
    for incident in body_to_structural.values():
        for joint in incident:
            neighbors[joint].update(other for other in incident if other != joint)

    closed_trees: list[_ClosedTree] = []
    unvisited = set(structural_joints)
    while unvisited:
        seed = min(unvisited)
        component: set[int] = set()
        stack = [seed]
        while stack:
            joint = stack.pop()
            if joint in component:
                continue
            component.add(joint)
            unvisited.discard(joint)
            stack.extend(neighbors[joint] - component)

        if frozenset(component) in handled_joint_sets:
            continue
        component_bodies = {
            body for joint in component for body in (int(parent[joint]), int(child[joint])) if body >= 0
        }
        if any(all_body_to_joints[body] - component for body in component_bodies):
            continue
        dynamic_bodies = {body for body in component_bodies if body_inv_mass[body] > 0.0}

        representative = {body: body for body in dynamic_bodies}

        base_joints: list[int] = []
        closure_joints: list[int] = []
        for joint in sorted(component):
            parent_body = int(parent[joint])
            child_body = int(child[joint])
            if parent_body not in dynamic_bodies or child_body not in dynamic_bodies:
                # Static/world attachments are leaves of the bipartite KKT tree.
                base_joints.append(joint)
                continue
            parent_root = _union_find_root(representative, parent_body)
            child_root = _union_find_root(representative, child_body)
            if parent_root != child_root:
                representative[child_root] = parent_root
                base_joints.append(joint)
            else:
                closure_joints.append(joint)

        if not closure_joints:
            continue
        base = _make_tree(base_joints, parent, child, body_inv_mass)
        if base is not None and set(base.bodies) == dynamic_bodies:
            closed_trees.append(_ClosedTree(base, closure_joints, sorted(component), base.bodies))
    return closed_trees


@wp.kernel
def equilibrate_tree_rows(
    compliance: wp.array[wp.spatial_matrix],
    row_scale: wp.array[wp.spatial_vector],
):
    row = wp.tid()
    scale = wp.spatial_vector()
    for axis in range(6):
        scale[axis] = 1.0 / wp.sqrt(wp.max(wp.abs(compliance[row][axis, axis]), 1.0e-30))
    row_scale[row] = scale


@wp.kernel
def equilibrate_tree_bodies(
    body_slots: wp.array[wp.int32],
    body_matrix: wp.array[wp.spatial_matrix],
    body_scale: wp.array[wp.spatial_vector],
):
    body = body_slots[wp.tid()]
    scale = wp.spatial_vector()
    for axis in range(6):
        scale[axis] = 1.0 / wp.sqrt(wp.max(wp.abs(body_matrix[body][axis, axis]), 1.0e-30))
    body_scale[body] = scale


@wp.kernel
def initialize_tree_nodes(
    node_body: wp.array[wp.int32],
    node_row: wp.array[wp.int32],
    body_matrix: wp.array[wp.spatial_matrix],
    body_rhs: wp.array[wp.spatial_vector],
    body_scale: wp.array[wp.spatial_vector],
    compliance: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    diagonal: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    node = wp.tid()
    body = node_body[node]
    if body >= 0:
        scale = body_scale[body]
        diagonal[node] = _scale_spatial_matrix(scale, body_matrix[body], scale)
        rhs[node] = _scale_spatial_vector(scale, body_rhs[body])
    else:
        row = node_row[node]
        scale = row_scale[row]
        diagonal[node] = -_scale_spatial_matrix(scale, compliance[row], scale)
        rhs[node] = -_scale_spatial_vector(scale, residual[row])


@wp.kernel
def initialize_tree_couplings(
    node_body: wp.array[wp.int32],
    node_row: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling_side: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    coupling: wp.array[wp.spatial_matrix],
):
    node = wp.tid()
    parent = parent_node[node]
    if parent < 0:
        coupling[node] = wp.spatial_matrix(0.0)
        return
    row = node_row[node]
    node_is_joint = row >= 0
    if not node_is_joint:
        row = node_row[parent]
    body = node_body[parent] if node_is_joint else node_body[node]
    jacobian = jacobian_parent[row] if coupling_side[node] == 0 else jacobian_child[row]
    value = _scale_spatial_matrix(row_scale[row], jacobian, body_scale[body])
    coupling[node] = value if node_is_joint else wp.transpose(value)


@wp.kernel
def invert_tree_leaves(
    leaves: wp.array[wp.int32],
    diagonal: wp.array[wp.spatial_matrix],
):
    """Factor leaves whose messages share a recipient."""
    node = leaves[wp.tid()]
    diagonal[node] = _inverse_spatial_robust(diagonal[node])


@wp.kernel
def eliminate_tree_unique_leaves(
    leaves: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    diagonal: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
    coupling: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    saved_rhs: wp.array[wp.spatial_vector],
):
    """Eliminate leaves whose recipients are unique within this level."""
    node = leaves[wp.tid()]
    parent = parent_node[node]
    inverse = _inverse_spatial_robust(diagonal[node])
    a = coupling[node]
    saved_inverse[node] = inverse
    saved_rhs[node] = rhs[node]
    diagonal[parent] = diagonal[parent] - wp.transpose(a) * inverse * a
    rhs[parent] = rhs[parent] - wp.transpose(a) * inverse * rhs[node]


@wp.kernel
def accumulate_tree_messages(
    recipients: wp.array[wp.int32],
    recipient_offsets: wp.array[wp.int32],
    message_nodes: wp.array[wp.int32],
    coupling: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    saved_rhs: wp.array[wp.spatial_vector],
    diagonal: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    index = wp.tid()
    recipient = recipients[index]
    diagonal_value = diagonal[recipient]
    rhs_value = rhs[recipient]
    for cursor in range(recipient_offsets[index], recipient_offsets[index + 1]):
        node = message_nodes[cursor]
        a = coupling[node]
        inverse = saved_inverse[node]
        diagonal_value = diagonal_value - wp.transpose(a) * inverse * a
        rhs_value = rhs_value - wp.transpose(a) * inverse * saved_rhs[node]
    diagonal[recipient] = diagonal_value
    rhs[recipient] = rhs_value


@wp.kernel
def solve_tree_roots(
    roots: wp.array[wp.int32],
    diagonal: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
    solution: wp.array[wp.spatial_vector],
):
    """Solve one independent root block per identically scheduled tree."""
    root = roots[wp.tid()]
    solution[root] = _inverse_spatial_robust(diagonal[root]) * rhs[root]


@wp.kernel
def back_substitute_tree_level(
    leaves: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    saved_rhs: wp.array[wp.spatial_vector],
    solution: wp.array[wp.spatial_vector],
):
    node = leaves[wp.tid()]
    parent = parent_node[node]
    solution[node] = saved_inverse[node] * (saved_rhs[node] - coupling[node] * solution[parent])


@wp.func
def _load_tree_contraction_edge(
    edge: int,
    transpose_edge: int,
    original_edge_count: int,
    coupling: wp.array[wp.spatial_matrix],
    fill_edges: wp.array[wp.spatial_matrix],
):
    """Load one original or generated edge in a requested orientation."""
    value = coupling[edge] if edge < original_edge_count else fill_edges[edge - original_edge_count]
    return wp.transpose(value) if transpose_edge != 0 else value


@wp.kernel
def eliminate_tree_contraction_level(
    original_edge_count: int,
    nodes: wp.array[wp.int32],
    neighbors: wp.array[wp.vec2i],
    edges: wp.array[wp.vec2i],
    transpose_edges: wp.array[wp.vec2i],
    generated_edges: wp.array[wp.int32],
    diagonal: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
    coupling: wp.array[wp.spatial_matrix],
    fill_edges: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    saved_rhs: wp.array[wp.spatial_vector],
):
    """Factor independent degree-one/two nodes and form fill edges."""
    index = wp.tid()
    node = nodes[index]
    inverse = _inverse_spatial_robust(diagonal[node])
    node_rhs = rhs[node]
    saved_inverse[node] = inverse
    saved_rhs[node] = node_rhs

    second_neighbor = neighbors[index][1]
    if second_neighbor >= 0:
        first_edge = edges[index][0]
        first_a = _load_tree_contraction_edge(
            first_edge, transpose_edges[index][0], original_edge_count, coupling, fill_edges
        )
        second_edge = edges[index][1]
        second_a = _load_tree_contraction_edge(
            second_edge, transpose_edges[index][1], original_edge_count, coupling, fill_edges
        )
        # The generated edge is oriented neighbor[0] -> neighbor[1].
        fill_edges[generated_edges[index] - original_edge_count] = -(wp.transpose(first_a) * inverse * second_a)


@wp.kernel
def accumulate_tree_contraction_messages(
    original_edge_count: int,
    nodes: wp.array[wp.int32],
    edges: wp.array[wp.vec2i],
    transpose_edges: wp.array[wp.vec2i],
    recipients: wp.array[wp.int32],
    recipient_offsets: wp.array[wp.int32],
    message_slots: wp.array[wp.int32],
    coupling: wp.array[wp.spatial_matrix],
    fill_edges: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    saved_rhs: wp.array[wp.spatial_vector],
    diagonal: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Accumulate exact Schur messages without matrix atomics."""
    index = wp.tid()
    recipient = recipients[index]
    diagonal_value = diagonal[recipient]
    rhs_value = rhs[recipient]
    for cursor in range(recipient_offsets[index], recipient_offsets[index + 1]):
        slot = message_slots[cursor]
        eliminated_index = slot // 2
        local = slot - 2 * eliminated_index
        node = nodes[eliminated_index]
        a = _load_tree_contraction_edge(
            edges[eliminated_index][local],
            transpose_edges[eliminated_index][local],
            original_edge_count,
            coupling,
            fill_edges,
        )
        inverse = saved_inverse[node]
        diagonal_value = diagonal_value - wp.transpose(a) * inverse * a
        rhs_value = rhs_value - wp.transpose(a) * inverse * saved_rhs[node]
    diagonal[recipient] = diagonal_value
    rhs[recipient] = rhs_value


@wp.kernel
def back_substitute_tree_contraction_level(
    original_edge_count: int,
    nodes: wp.array[wp.int32],
    neighbors: wp.array[wp.vec2i],
    edges: wp.array[wp.vec2i],
    transpose_edges: wp.array[wp.vec2i],
    coupling: wp.array[wp.spatial_matrix],
    fill_edges: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    saved_rhs: wp.array[wp.spatial_vector],
    solution: wp.array[wp.spatial_vector],
):
    """Recover one exact rake/compress level in parallel."""
    index = wp.tid()
    node = nodes[index]
    value = saved_rhs[node]

    first_a = _load_tree_contraction_edge(
        edges[index][0], transpose_edges[index][0], original_edge_count, coupling, fill_edges
    )
    value = value - first_a * solution[neighbors[index][0]]

    second_neighbor = neighbors[index][1]
    if second_neighbor >= 0:
        second_a = _load_tree_contraction_edge(
            edges[index][1], transpose_edges[index][1], original_edge_count, coupling, fill_edges
        )
        value = value - second_a * solution[second_neighbor]
    solution[node] = saved_inverse[node] * value


@wp.kernel
def scatter_tree_body_correction_indexed(
    body_nodes: wp.array[wp.int32],
    body_slots: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    solution: wp.array[wp.spatial_vector],
    body_correction: wp.array[wp.spatial_vector],
):
    """Scatter body nodes from multiple flattened tree instances."""
    index = wp.tid()
    node = body_nodes[index]
    slot = body_slots[index]
    correction = _scale_spatial_vector(body_scale[slot], solution[node])
    body_correction[slot] = correction


@wp.kernel
def initialize_closure_response(
    node_body: wp.array[wp.int32],
    tree_node_count: int,
    closure_count: int,
    closure_parent_node: wp.array[wp.int32],
    closure_child_node: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    response_rhs: wp.array[wp.spatial_matrix],
):
    """Build the six tree right-hand sides induced by every closure row."""
    index = wp.tid()
    node = index // closure_count
    batch = node // tree_node_count
    closure = batch * closure_count + index % closure_count
    value = wp.spatial_matrix(0.0)
    if node == closure_parent_node[closure]:
        coupling = _scale_spatial_matrix(row_scale[closure], jacobian_parent[closure], body_scale[node_body[node]])
        value = value + wp.transpose(coupling)
    if node == closure_child_node[closure]:
        coupling = _scale_spatial_matrix(row_scale[closure], jacobian_child[closure], body_scale[node_body[node]])
        value = value + wp.transpose(coupling)
    response_rhs[index] = value


@wp.kernel
def eliminate_tree_response_unique_leaves(
    leaves: wp.array[wp.int32],
    closure_count: int,
    parent_node: wp.array[wp.int32],
    coupling: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    response_rhs: wp.array[wp.spatial_matrix],
):
    index = wp.tid()
    node = leaves[index // closure_count]
    closure = index % closure_count
    parent = parent_node[node]
    response_rhs[parent * closure_count + closure] = response_rhs[parent * closure_count + closure] - (
        wp.transpose(coupling[node]) * saved_inverse[node] * response_rhs[node * closure_count + closure]
    )


@wp.kernel
def accumulate_tree_response_messages(
    recipients: wp.array[wp.int32],
    recipient_offsets: wp.array[wp.int32],
    message_nodes: wp.array[wp.int32],
    closure_count: int,
    coupling: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    response_rhs: wp.array[wp.spatial_matrix],
):
    index = wp.tid()
    recipient_index = index // closure_count
    closure = index % closure_count
    recipient = recipients[recipient_index]
    value = response_rhs[recipient * closure_count + closure]
    for cursor in range(recipient_offsets[recipient_index], recipient_offsets[recipient_index + 1]):
        node = message_nodes[cursor]
        value = value - (
            wp.transpose(coupling[node]) * saved_inverse[node] * response_rhs[node * closure_count + closure]
        )
    response_rhs[recipient * closure_count + closure] = value


@wp.kernel
def back_substitute_tree_response_level(
    leaves: wp.array[wp.int32],
    closure_count: int,
    parent_node: wp.array[wp.int32],
    coupling: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    response_rhs: wp.array[wp.spatial_matrix],
    response_solution: wp.array[wp.spatial_matrix],
):
    index = wp.tid()
    node = leaves[index // closure_count]
    closure = index % closure_count
    parent = parent_node[node]
    response_solution[node * closure_count + closure] = saved_inverse[node] * (
        response_rhs[node * closure_count + closure]
        - coupling[node] * response_solution[parent * closure_count + closure]
    )


@wp.kernel
def assemble_closure_rhs(
    node_body: wp.array[wp.int32],
    closure_parent_node: wp.array[wp.int32],
    closure_child_node: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    tree_solution: wp.array[wp.spatial_vector],
    closure_rhs: wp.array[wp.spatial_vector],
):
    closure = wp.tid()
    rhs = _scale_spatial_vector(row_scale[closure], residual[closure])
    parent_node = closure_parent_node[closure]
    if parent_node >= 0:
        coupling = _scale_spatial_matrix(
            row_scale[closure], jacobian_parent[closure], body_scale[node_body[parent_node]]
        )
        rhs = rhs + coupling * tree_solution[parent_node]
    child_node = closure_child_node[closure]
    if child_node >= 0:
        coupling = _scale_spatial_matrix(row_scale[closure], jacobian_child[closure], body_scale[node_body[child_node]])
        rhs = rhs + coupling * tree_solution[child_node]
    closure_rhs[closure] = rhs


@wp.kernel
def assemble_closure_schur(
    closure_count: int,
    node_body: wp.array[wp.int32],
    closure_parent_node: wp.array[wp.int32],
    closure_child_node: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    compliance: wp.array[wp.spatial_matrix],
    response_solution: wp.array[wp.spatial_matrix],
    schur: wp.array[wp.spatial_matrix],
):
    index = wp.tid()
    row = index // closure_count
    column = index % closure_count
    value = wp.spatial_matrix(0.0)
    if row == column:
        value = _scale_spatial_matrix(row_scale[row], compliance[row], row_scale[row])
    parent_node = closure_parent_node[row]
    if parent_node >= 0:
        coupling = _scale_spatial_matrix(row_scale[row], jacobian_parent[row], body_scale[node_body[parent_node]])
        value = value + coupling * response_solution[parent_node * closure_count + column]
    child_node = closure_child_node[row]
    if child_node >= 0:
        coupling = _scale_spatial_matrix(row_scale[row], jacobian_child[row], body_scale[node_body[child_node]])
        value = value + coupling * response_solution[child_node * closure_count + column]
    schur[index] = value


@wp.kernel
def solve_block_dense_serial(
    block_count: int,
    matrix: wp.array[wp.spatial_matrix],
    rhs_solution: wp.array[wp.spatial_vector],
):
    """Solve one dense SPD block system per batch with Gaussian elimination."""
    batch = wp.tid()
    matrix_base = batch * block_count * block_count
    rhs_base = batch * block_count
    for pivot in range(block_count):
        pivot_index = matrix_base + pivot * block_count + pivot
        pivot_inverse = _inverse_spatial_robust(matrix[pivot_index])
        matrix[pivot_index] = pivot_inverse
        for row in range(pivot + 1, block_count):
            factor = matrix[matrix_base + row * block_count + pivot] * pivot_inverse
            for column in range(pivot + 1, block_count):
                target = matrix_base + row * block_count + column
                matrix[target] = matrix[target] - factor * matrix[matrix_base + pivot * block_count + column]
            rhs_solution[rhs_base + row] = rhs_solution[rhs_base + row] - factor * rhs_solution[rhs_base + pivot]
    for reverse_index in range(block_count):
        row = block_count - 1 - reverse_index
        value = rhs_solution[rhs_base + row]
        for column in range(row + 1, block_count):
            value = value - matrix[matrix_base + row * block_count + column] * rhs_solution[rhs_base + column]
        rhs_solution[rhs_base + row] = matrix[matrix_base + row * block_count + row] * value


@wp.kernel
def invert_block_dense_pivot(
    pivot: int,
    block_count: int,
    matrix: wp.array[wp.spatial_matrix],
):
    """Invert one block pivot in every independent dense system."""
    batch = wp.tid()
    index = batch * block_count * block_count + pivot * block_count + pivot
    matrix[index] = _inverse_spatial_robust(matrix[index])


@wp.kernel
def factor_block_ldlt_pivot_rows(
    pivot: int,
    block_count: int,
    matrix: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Form one block-LDLT column and forward-eliminate its rhs rows."""
    index = wp.tid()
    remaining = block_count - pivot - 1
    batch = index // remaining
    row = pivot + 1 + index - batch * remaining
    if row >= block_count:
        return
    matrix_base = batch * block_count * block_count
    rhs_base = batch * block_count
    multiplier = matrix[matrix_base + row * block_count + pivot] * matrix[matrix_base + pivot * block_count + pivot]
    # The strict upper triangle is scratch once symmetry is used. Store
    # L(row,pivot) transposed by block index, without another factor buffer.
    matrix[matrix_base + pivot * block_count + row] = multiplier
    rhs[rhs_base + row] = rhs[rhs_base + row] - multiplier * rhs[rhs_base + pivot]


@wp.kernel
def update_block_ldlt_lower_triangle(
    pivot: int,
    block_count: int,
    trailing_count: int,
    matrix: wp.array[wp.spatial_matrix],
):
    """Update the symmetric trailing block matrix in parallel."""
    index = wp.tid()
    trailing_size = trailing_count * trailing_count
    batch = index // trailing_size
    local_index = index - batch * trailing_size
    local_row = local_index // trailing_count
    local_column = local_index - local_row * trailing_count
    if local_column > local_row:
        return
    row = pivot + 1 + local_row
    column = pivot + 1 + local_column
    matrix_base = batch * block_count * block_count
    multiplier = matrix[matrix_base + pivot * block_count + row]
    original_column = matrix[matrix_base + column * block_count + pivot]
    target = matrix_base + row * block_count + column
    matrix[target] = matrix[target] - multiplier * wp.transpose(original_column)


@wp.kernel
def solve_block_ldlt_diagonal(
    block_count: int,
    matrix: wp.array[wp.spatial_matrix],
    rhs_solution: wp.array[wp.spatial_vector],
):
    """Apply independent inverse diagonal blocks after forward elimination."""
    index = wp.tid()
    batch = index // block_count
    row = index - batch * block_count
    if row < block_count:
        rhs_solution[index] = matrix[batch * block_count * block_count + row * block_count + row] * rhs_solution[index]


@wp.kernel
def back_substitute_block_ldlt_serial(
    block_count: int,
    matrix: wp.array[wp.spatial_matrix],
    rhs_solution: wp.array[wp.spatial_vector],
):
    """Back-substitute one block-LDLT system per batch."""
    batch = wp.tid()
    matrix_base = batch * block_count * block_count
    rhs_base = batch * block_count
    for reverse_index in range(block_count):
        row = block_count - 1 - reverse_index
        value = rhs_solution[rhs_base + row]
        for column in range(row + 1, block_count):
            multiplier = matrix[matrix_base + row * block_count + column]
            value = value - wp.transpose(multiplier) * rhs_solution[rhs_base + column]
        rhs_solution[rhs_base + row] = value


@wp.kernel
def scatter_closed_tree_body_correction(
    body_nodes: wp.array[wp.int32],
    body_slots: wp.array[wp.int32],
    tree_body_count: int,
    closure_count: int,
    body_scale: wp.array[wp.spatial_vector],
    tree_solution: wp.array[wp.spatial_vector],
    response_solution: wp.array[wp.spatial_matrix],
    closure_multiplier: wp.array[wp.spatial_vector],
    body_correction: wp.array[wp.spatial_vector],
):
    index = wp.tid()
    batch = index // tree_body_count
    node = body_nodes[index]
    slot = body_slots[index]
    correction = tree_solution[node]
    for closure in range(closure_count):
        correction = (
            correction
            - response_solution[node * closure_count + closure] * closure_multiplier[batch * closure_count + closure]
        )
    correction = _scale_spatial_vector(body_scale[slot], correction)
    body_correction[slot] = correction


class _PathBucket:
    """Fixed-size batch of equal-shape joint paths."""

    def __init__(self, paths: list[_Path], device):
        self.device = device
        self.spatial_block_dim = _SPATIAL_GPU_BLOCK_DIM if device.is_cuda else 1
        self.row_count = len(paths[0].joints)
        self.body_count = len(paths[0].bodies)
        self.batch_count = len(paths)
        self.size = self.batch_count * self.row_count
        self.body_size = self.batch_count * self.body_count
        self.use_persistent_cr = device.is_cuda and self.row_count <= _CR_PERSISTENT_MAX_ROWS

        joint_ids: list[int] = []
        body_ids: list[int] = []
        body_incident_rows: list[tuple[int, int]] = []
        for path in paths:
            joint_ids.extend(path.joints)
            body_ids.extend(path.bodies)
            for incident in path.body_rows:
                encoded = [(row << 1) | side for row, side in incident]
                body_incident_rows.append((encoded[0] if encoded else -1, encoded[1] if len(encoded) > 1 else -1))

        self.joint_ids_host = np.asarray(joint_ids, dtype=np.int64)
        self.joint_ids = wp.array(self.joint_ids_host, dtype=wp.int32, device=device)
        self.row_body = None
        self.body_ids_host = np.asarray(body_ids, dtype=np.int32)
        self.body_ids = wp.array(self.body_ids_host, dtype=wp.int32, device=device)
        self.body_incident_rows = wp.array(body_incident_rows, dtype=wp.vec2i, device=device)
        self.jacobian_parent = wp.empty(self.size, dtype=wp.spatial_matrix, device=device)
        self.jacobian_child = wp.empty_like(self.jacobian_parent)
        self.lower = [wp.empty_like(self.jacobian_parent)]
        self.diagonal = [wp.empty_like(self.jacobian_parent)]
        self.upper = [wp.empty_like(self.jacobian_parent)]
        self.rhs = [wp.empty(self.size, dtype=wp.spatial_vector, device=device)]
        self.row_active = wp.zeros(self.size, dtype=wp.int32, device=device)

        # Linearization storage is dead as soon as the path system is
        # assembled. Reuse the diagonal and rhs buffers for compliance and
        # residual, then overwrite them with CR factors and the solution.
        self.compliance = self.diagonal[0]
        self.residual = self.rhs[0]
        self.solution = self.rhs

        self.cr_levels: list[tuple[int, int, int]] = []
        stride = 1
        active_count = self.row_count
        while active_count > _CR_SERIAL_TERMINAL_SIZE:
            eliminated_count = len(range(stride, self.row_count, 2 * stride))
            survivor_count = len(range(0, self.row_count, 2 * stride))
            self.cr_levels.append((stride, eliminated_count, survivor_count))
            stride *= 2
            active_count = survivor_count
        self.terminal_stride = stride

    def bind_body_endpoints(self, parent: np.ndarray, child: np.ndarray, body_slot: dict[int, int]):
        self.row_body = wp.array(
            [
                (body_slot.get(int(parent[joint]), -1), body_slot.get(int(child[joint]), -1))
                for joint in self.joint_ids_host
            ],
            dtype=wp.vec2i,
            device=self.device,
        )

    def solve_rows(self):
        """Solve the block-tridiagonal path in four in-place row buffers."""
        if self.use_persistent_cr:
            wp.launch(
                solve_cr_persistent,
                self.batch_count * self.spatial_block_dim,
                inputs=[self.row_count, _CR_SERIAL_TERMINAL_SIZE],
                outputs=[self.lower[0], self.diagonal[0], self.upper[0], self.rhs[0]],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            return

        for stride, eliminated_count, survivor_count in self.cr_levels:
            wp.launch(
                invert_cr_eliminated_in_place,
                self.batch_count * eliminated_count,
                inputs=[stride, self.row_count, eliminated_count],
                outputs=[self.diagonal[0]],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            wp.launch(
                reduce_cr_in_place,
                self.batch_count * survivor_count,
                inputs=[stride, self.row_count, survivor_count],
                outputs=[self.lower[0], self.diagonal[0], self.upper[0], self.rhs[0]],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        wp.launch(
            solve_strided_cr_coarse,
            self.batch_count,
            inputs=[self.terminal_stride, self.row_count],
            outputs=[self.lower[0], self.diagonal[0], self.upper[0], self.rhs[0]],
            device=self.device,
        )
        for stride, eliminated_count, _ in reversed(self.cr_levels):
            wp.launch(
                back_substitute_cr_in_place,
                self.batch_count * eliminated_count,
                inputs=[stride, self.row_count, eliminated_count],
                outputs=[self.lower[0], self.diagonal[0], self.upper[0], self.rhs[0]],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )


class _TreeBucket:
    """A batch of graph-capturable trees with one exact symbolic schedule."""

    def __init__(self, trees: _Tree | list[_Tree], body_slot: dict[int, int], device):
        if isinstance(trees, _Tree):
            trees = [trees]
        tree = trees[0]
        signature = _tree_topology_signature(tree)
        if any(_tree_topology_signature(candidate) != signature for candidate in trees[1:]):
            raise ValueError("A tree bucket requires one normalized symbolic topology")

        self.device = device
        self.spatial_block_dim = _SPATIAL_GPU_BLOCK_DIM if device.is_cuda else 1
        self.batch_count = len(trees)
        self.tree_row_count = len(tree.joints)
        self.tree_body_count = len(tree.bodies)
        self.tree_node_count = len(tree.node_body)
        self.row_count = self.batch_count * self.tree_row_count
        self.body_count = self.batch_count * self.tree_body_count
        self.node_count = self.batch_count * self.tree_node_count
        self.size = self.row_count
        self.body_size = self.body_count

        joint_ids: list[int] = []
        body_ids: list[int] = []
        node_body: list[int] = []
        body_nodes: list[int] = []
        body_slots: list[int] = []
        node_row: list[int] = []
        parent_node: list[int] = []
        coupling_side: list[int] = []
        roots: list[int] = []
        for batch, candidate in enumerate(trees):
            node_offset = batch * self.tree_node_count
            row_offset = batch * self.tree_row_count
            joint_ids.extend(candidate.joints)
            body_ids.extend(candidate.bodies)
            node_body.extend(body_slot[body] if body >= 0 else -1 for body in candidate.node_body)
            body_nodes.extend(node_offset + node for node in range(self.tree_body_count))
            body_slots.extend(body_slot[body] for body in candidate.bodies)
            node_row.extend(row_offset + row if row >= 0 else -1 for row in candidate.node_row)
            parent_node.extend(node_offset + parent if parent >= 0 else -1 for parent in candidate.parent_node)
            coupling_side.extend(candidate.coupling_side)
            roots.append(node_offset + candidate.root)

        self.root = roots[0]
        self.roots = wp.array(roots, dtype=wp.int32, device=device)
        self.joint_ids_host = np.asarray(joint_ids, dtype=np.int64)
        self.joint_ids = wp.array(self.joint_ids_host, dtype=wp.int32, device=device)
        self.body_ids_host = np.asarray(body_ids, dtype=np.int32)
        self.node_body = wp.array(node_body, dtype=wp.int32, device=device)
        self.body_nodes = wp.array(body_nodes, dtype=wp.int32, device=device)
        # In the common single-tree case body nodes occupy the leading slots,
        # so retain the original zero-copy alias.  Batched trees interleave
        # body and row nodes per instance and therefore need an explicit map.
        self.body_slots = (
            self.node_body if self.batch_count == 1 else wp.array(body_slots, dtype=wp.int32, device=device)
        )
        self.node_row = wp.array(node_row, dtype=wp.int32, device=device)
        self.parent_node = wp.array(parent_node, dtype=wp.int32, device=device)
        self.coupling_side = wp.array(coupling_side, dtype=wp.int32, device=device)
        # Joint Jacobians are dead once node couplings are initialized. Reuse
        # their contiguous workspace for the tree diagonal/inverse factors.
        self.matrix_workspace = wp.zeros(
            max(2 * self.row_count, self.node_count), dtype=wp.spatial_matrix, device=device
        )
        self.jacobian_parent = self.matrix_workspace[: self.row_count]
        self.jacobian_child = self.matrix_workspace[self.row_count : 2 * self.row_count]
        self.diagonal = self.matrix_workspace[: self.node_count]
        self.compliance = wp.zeros(self.row_count, dtype=wp.spatial_matrix, device=device)
        self.row_scale = wp.zeros(self.row_count, dtype=wp.spatial_vector, device=device)
        self.residual = wp.zeros(self.row_count, dtype=wp.spatial_vector, device=device)
        self.row_active = wp.zeros(self.row_count, dtype=wp.int32, device=device)
        self.rhs = wp.zeros(self.node_count, dtype=wp.spatial_vector, device=device)
        self.coupling = wp.zeros(self.node_count, dtype=wp.spatial_matrix, device=device)
        # Eliminated node blocks and rhs values are dead in the forward pass;
        # keep their inverse and later solution in those same node slots.
        self.saved_inverse = self.diagonal
        self.saved_rhs = self.rhs
        self.solution = self.rhs

        levels = [
            [batch * self.tree_node_count + node for batch in range(self.batch_count) for node in level]
            for level in tree.elimination_levels
        ]
        self.levels = _make_device_tree_levels(levels, parent_node, device)
        backbone, branch_levels = _tree_backbone_partition(tree)
        self.backbone_node_count = len(backbone)
        self.backbone_nodes = wp.array(
            [batch * self.tree_node_count + node for batch in range(self.batch_count) for node in backbone],
            dtype=wp.int32,
            device=device,
        )
        batched_branch_levels = [
            [batch * self.tree_node_count + node for batch in range(self.batch_count) for node in level]
            for level in branch_levels
        ]
        self.branch_levels = _make_device_tree_levels(batched_branch_levels, parent_node, device)
        self.backbone_lower = wp.zeros(
            self.batch_count * self.backbone_node_count, dtype=wp.spatial_matrix, device=device
        )
        self.backbone_upper = wp.zeros_like(self.backbone_lower)
        self.backbone_cr_levels: list[tuple[int, int, int]] = []
        stride = 1
        active_count = self.backbone_node_count
        while active_count > _CR_SERIAL_TERMINAL_SIZE:
            eliminated_count = len(range(stride, self.backbone_node_count, 2 * stride))
            survivor_count = len(range(0, self.backbone_node_count, 2 * stride))
            self.backbone_cr_levels.append((stride, eliminated_count, survivor_count))
            stride *= 2
            active_count = survivor_count
        self.backbone_terminal_stride = stride
        level_launches = _tree_level_launch_count(tree.elimination_levels, tree.parent_node) + 1
        backbone_launches = (
            _tree_level_launch_count(branch_levels, tree.parent_node) + 1 + 3 * len(self.backbone_cr_levels) + 1
        )
        contraction = _tree_contraction_schedule(tree)
        contraction_launches = 3 * len(contraction.levels) + 1
        # All schedules are exact. Select once from immutable topology by
        # dependent launch count: backbone CR for long/slender trees, ordinary
        # leaf rake for shallow trees, and full rake/compress for multi-arm
        # trees whose side branches would otherwise serialize.
        self.use_tree_contraction = contraction_launches < min(level_launches, backbone_launches)
        self.use_backbone_cr = not self.use_tree_contraction and backbone_launches < level_launches
        if self.use_tree_contraction:
            self.contraction_levels = _make_device_batched_tree_contraction(
                contraction, self.batch_count, self.tree_node_count, device
            )
            contraction_roots = [batch * self.tree_node_count + contraction.root for batch in range(self.batch_count)]
            self.contraction_root = contraction_roots[0]
            self.contraction_roots = wp.array(contraction_roots, dtype=wp.int32, device=device)
            self.contraction_fill_edges = wp.zeros(
                self.batch_count * (contraction.edge_count - self.tree_node_count),
                dtype=wp.spatial_matrix,
                device=device,
            )
        else:
            self.contraction_levels = []
            self.contraction_root = -1
            self.contraction_roots = None
            self.contraction_fill_edges = None
        # The shared backbone kernels also propagate optional closure-response
        # right-hand sides.  A tree has none; one dummy element keeps the same
        # kernels and graph shape without allocating per-node response storage.
        self.no_closure_response = wp.zeros(1, dtype=wp.spatial_matrix, device=device)

    def initialize_system(self, body_matrix, body_rhs, body_scale):
        """Initialize the equilibrated bipartite KKT nodes and edge blocks."""
        wp.launch(
            equilibrate_tree_bodies,
            self.body_count,
            inputs=[self.body_slots, body_matrix],
            outputs=[body_scale],
            device=self.device,
        )
        wp.launch(
            equilibrate_tree_rows,
            self.row_count,
            inputs=[self.compliance],
            outputs=[self.row_scale],
            device=self.device,
        )
        wp.launch(
            initialize_tree_couplings,
            self.node_count,
            inputs=[
                self.node_body,
                self.node_row,
                self.parent_node,
                self.coupling_side,
                body_scale,
                self.row_scale,
                self.jacobian_parent,
                self.jacobian_child,
            ],
            outputs=[self.coupling],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        wp.launch(
            initialize_tree_nodes,
            self.node_count,
            inputs=[
                self.node_body,
                self.node_row,
                body_matrix,
                body_rhs,
                body_scale,
                self.compliance,
                self.residual,
                self.row_scale,
            ],
            outputs=[self.diagonal, self.rhs],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )

    def eliminate_levels(self, levels):
        """Eliminate a precomputed set of tree leaves toward its survivor."""
        for leaves, recipients, recipient_offsets, message_nodes, recipients_are_unique in levels:
            if recipients_are_unique:
                wp.launch(
                    eliminate_tree_unique_leaves,
                    leaves.shape[0],
                    inputs=[leaves, self.parent_node, self.diagonal, self.rhs, self.coupling],
                    outputs=[self.saved_inverse, self.saved_rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
            else:
                wp.launch(
                    invert_tree_leaves,
                    leaves.shape[0],
                    inputs=[leaves],
                    outputs=[self.saved_inverse],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
                wp.launch(
                    accumulate_tree_messages,
                    recipients.shape[0],
                    inputs=[
                        recipients,
                        recipient_offsets,
                        message_nodes,
                        self.coupling,
                        self.saved_inverse,
                        self.saved_rhs,
                    ],
                    outputs=[self.diagonal, self.rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )

    def back_substitute_levels(self, levels):
        """Recover a precomputed set of eliminated tree leaves."""
        for leaves, _, _, _, _ in reversed(levels):
            wp.launch(
                back_substitute_tree_level,
                leaves.shape[0],
                inputs=[
                    leaves,
                    self.parent_node,
                    self.coupling,
                    self.saved_inverse,
                    self.saved_rhs,
                ],
                outputs=[self.solution],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

    def solve_backbone(self):
        """Solve the surviving exact tree backbone with block CR."""
        wp.launch(
            initialize_tree_backbone_edges,
            self.batch_count * self.backbone_node_count,
            inputs=[self.backbone_node_count, self.backbone_nodes, self.parent_node, self.coupling],
            outputs=[self.backbone_lower, self.backbone_upper],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        for stride, eliminated_count, survivor_count in self.backbone_cr_levels:
            wp.launch(
                invert_tree_backbone_cr_eliminated,
                self.batch_count * eliminated_count,
                inputs=[stride, self.backbone_node_count, eliminated_count, self.backbone_nodes],
                outputs=[self.diagonal],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            wp.launch(
                reduce_tree_backbone_cr_in_place,
                self.batch_count * survivor_count,
                inputs=[stride, self.backbone_node_count, survivor_count, 0, self.backbone_nodes],
                outputs=[
                    self.backbone_lower,
                    self.diagonal,
                    self.backbone_upper,
                    self.rhs,
                    self.no_closure_response,
                ],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        wp.launch(
            solve_tree_backbone_cr_coarse,
            self.batch_count,
            inputs=[self.backbone_terminal_stride, self.backbone_node_count, 0, self.backbone_nodes],
            outputs=[
                self.backbone_lower,
                self.diagonal,
                self.backbone_upper,
                self.rhs,
                self.no_closure_response,
            ],
            device=self.device,
        )
        for stride, eliminated_count, _ in reversed(self.backbone_cr_levels):
            wp.launch(
                back_substitute_tree_backbone_cr_in_place,
                self.batch_count * eliminated_count,
                inputs=[
                    stride,
                    self.backbone_node_count,
                    eliminated_count,
                    0,
                    self.backbone_nodes,
                    self.backbone_lower,
                    self.backbone_upper,
                ],
                outputs=[self.diagonal, self.rhs, self.no_closure_response],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

    def solve_tree_contraction(self):
        """Solve an arbitrary acyclic island by exact rake-and-compress."""
        for (
            nodes,
            neighbors,
            edges,
            transpose_edges,
            generated_edges,
            recipients,
            offsets,
            slots,
        ) in self.contraction_levels:
            wp.launch(
                eliminate_tree_contraction_level,
                nodes.shape[0],
                inputs=[self.node_count, nodes, neighbors, edges, transpose_edges, generated_edges],
                outputs=[
                    self.diagonal,
                    self.rhs,
                    self.coupling,
                    self.contraction_fill_edges,
                    self.saved_inverse,
                    self.saved_rhs,
                ],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            wp.launch(
                accumulate_tree_contraction_messages,
                recipients.shape[0],
                inputs=[
                    self.node_count,
                    nodes,
                    edges,
                    transpose_edges,
                    recipients,
                    offsets,
                    slots,
                    self.coupling,
                    self.contraction_fill_edges,
                ],
                outputs=[self.saved_inverse, self.saved_rhs, self.diagonal, self.rhs],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        wp.launch(
            solve_tree_roots,
            self.batch_count,
            inputs=[self.contraction_roots, self.diagonal, self.rhs],
            outputs=[self.solution],
            device=self.device,
        )
        for nodes, neighbors, edges, transpose_edges, _, _, _, _ in reversed(self.contraction_levels):
            wp.launch(
                back_substitute_tree_contraction_level,
                nodes.shape[0],
                inputs=[
                    self.node_count,
                    nodes,
                    neighbors,
                    edges,
                    transpose_edges,
                    self.coupling,
                    self.contraction_fill_edges,
                ],
                outputs=[self.saved_inverse, self.saved_rhs, self.solution],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

    def solve_tree(self, body_matrix, body_rhs, body_scale, body_correction, *, scatter=True):
        self.initialize_system(body_matrix, body_rhs, body_scale)
        if self.use_tree_contraction:
            self.solve_tree_contraction()
        elif self.use_backbone_cr:
            # Rake only side branches. Solving a long serial part by leaf
            # levels would expose O(n) dependent launches; block CR solves the
            # exact same KKT backbone in O(log n) launch depth.
            self.eliminate_levels(self.branch_levels)
            self.solve_backbone()
            self.back_substitute_levels(self.branch_levels)
        else:
            self.eliminate_levels(self.levels)
            wp.launch(
                solve_tree_roots,
                self.batch_count,
                inputs=[self.roots, self.diagonal, self.rhs],
                outputs=[self.solution],
                device=self.device,
            )
            self.back_substitute_levels(self.levels)

        if scatter:
            wp.launch(
                scatter_tree_body_correction_indexed,
                self.body_count,
                inputs=[self.body_nodes, self.body_slots, body_scale, self.solution],
                outputs=[body_correction],
                device=self.device,
            )


class _ClosedTreeBucket:
    """Batched exact backbone-CR solves completed by closure Schur systems."""

    def __init__(
        self,
        components: _ClosedTree | list[_ClosedTree],
        parent: np.ndarray,
        child: np.ndarray,
        body_slot: dict[int, int],
        device,
    ):
        if isinstance(components, _ClosedTree):
            components = [components]
        component = components[0]
        signature = _closed_tree_topology_signature(component, parent, child)
        if any(_closed_tree_topology_signature(candidate, parent, child) != signature for candidate in components[1:]):
            raise ValueError("A closed-tree bucket requires one normalized symbolic topology")

        self.device = device
        self.tree = _TreeBucket([candidate.tree for candidate in components], body_slot, device)
        self.spatial_block_dim = self.tree.spatial_block_dim
        # Cyclic islands use the same exact tree backbone schedule before the
        # closure Schur solve; share its immutable topology and work buffers.
        self.backbone_node_count = self.tree.backbone_node_count
        self.backbone_nodes = self.tree.backbone_nodes
        self.branch_levels = self.tree.branch_levels
        self.backbone_lower = self.tree.backbone_lower
        self.backbone_upper = self.tree.backbone_upper
        self.backbone_cr_levels = self.tree.backbone_cr_levels
        self.backbone_terminal_stride = self.tree.backbone_terminal_stride
        self.batch_count = len(components)
        self.size = sum(len(candidate.joints) for candidate in components)
        self.body_size = self.tree.body_count
        self.closure_count = len(component.closure_joints)
        self.closure_size = self.batch_count * self.closure_count
        closure_joint_ids: list[int] = []
        closure_parent_node: list[int] = []
        closure_child_node: list[int] = []
        for batch, candidate in enumerate(components):
            node_offset = batch * self.tree.tree_node_count
            body_node = {body: node for node, body in enumerate(candidate.tree.bodies)}
            for joint in candidate.closure_joints:
                closure_joint_ids.append(joint)
                parent_node = body_node.get(int(parent[joint]), -1)
                child_node = body_node.get(int(child[joint]), -1)
                closure_parent_node.append(node_offset + parent_node if parent_node >= 0 else -1)
                closure_child_node.append(node_offset + child_node if child_node >= 0 else -1)
        self.closure_joint_ids = wp.array(closure_joint_ids, dtype=wp.int32, device=device)
        self.closure_parent_node = wp.array(closure_parent_node, dtype=wp.int32, device=device)
        self.closure_child_node = wp.array(closure_child_node, dtype=wp.int32, device=device)

        self.closure_jacobian_parent = wp.zeros(self.closure_size, dtype=wp.spatial_matrix, device=device)
        self.closure_jacobian_child = wp.zeros_like(self.closure_jacobian_parent)
        self.closure_compliance = wp.zeros_like(self.closure_jacobian_parent)
        self.closure_residual = wp.zeros(self.closure_size, dtype=wp.spatial_vector, device=device)
        self.closure_row_active = wp.zeros(self.closure_size, dtype=wp.int32, device=device)
        self.closure_row_scale = wp.zeros(self.closure_size, dtype=wp.spatial_vector, device=device)
        response_size = self.tree.node_count * self.closure_count
        self.response_rhs = wp.zeros(response_size, dtype=wp.spatial_matrix, device=device)
        self.response_solution = self.response_rhs
        self.closure_schur = wp.zeros(
            self.batch_count * self.closure_count * self.closure_count,
            dtype=wp.spatial_matrix,
            device=device,
        )
        self.closure_rhs = wp.zeros(self.closure_size, dtype=wp.spatial_vector, device=device)
        self.closure_multiplier = self.closure_rhs

    def solve_closure_schur(self):
        """Solve the dense closure system, parallelizing independent rows."""
        if self.closure_count <= _CR_SERIAL_TERMINAL_SIZE or not self.device.is_cuda:
            wp.launch(
                solve_block_dense_serial,
                self.batch_count,
                inputs=[self.closure_count],
                outputs=[self.closure_schur, self.closure_multiplier],
                device=self.device,
            )
            return

        for pivot in range(self.closure_count):
            wp.launch(
                invert_block_dense_pivot,
                self.batch_count,
                inputs=[pivot, self.closure_count],
                outputs=[self.closure_schur],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            remaining = self.closure_count - pivot - 1
            if remaining > 0:
                wp.launch(
                    factor_block_ldlt_pivot_rows,
                    self.batch_count * remaining,
                    inputs=[pivot, self.closure_count, self.closure_schur],
                    outputs=[self.closure_multiplier],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
                wp.launch(
                    update_block_ldlt_lower_triangle,
                    self.batch_count * remaining * remaining,
                    inputs=[pivot, self.closure_count, remaining],
                    outputs=[self.closure_schur],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
        wp.launch(
            solve_block_ldlt_diagonal,
            self.closure_size,
            inputs=[self.closure_count, self.closure_schur],
            outputs=[self.closure_multiplier],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        wp.launch(
            back_substitute_block_ldlt_serial,
            self.batch_count,
            inputs=[self.closure_count, self.closure_schur],
            outputs=[self.closure_multiplier],
            device=self.device,
        )

    def solve_tree(self, body_matrix, body_rhs, body_scale, body_correction):
        self.tree.initialize_system(body_matrix, body_rhs, body_scale)
        wp.launch(
            equilibrate_tree_rows,
            self.closure_size,
            inputs=[self.closure_compliance],
            outputs=[self.closure_row_scale],
            device=self.device,
        )
        wp.launch(
            initialize_closure_response,
            self.tree.node_count * self.closure_count,
            inputs=[
                self.tree.node_body,
                self.tree.tree_node_count,
                self.closure_count,
                self.closure_parent_node,
                self.closure_child_node,
                body_scale,
                self.closure_row_scale,
                self.closure_jacobian_parent,
                self.closure_jacobian_child,
            ],
            outputs=[self.response_rhs],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )

        # Eliminate only the shallow side branches. The remaining long tree
        # backbone is block tridiagonal and is solved in logarithmic CR depth.
        for leaves, recipients, recipient_offsets, message_nodes, recipients_are_unique in self.branch_levels:
            if recipients_are_unique:
                wp.launch(
                    eliminate_tree_unique_leaves,
                    leaves.shape[0],
                    inputs=[
                        leaves,
                        self.tree.parent_node,
                        self.tree.diagonal,
                        self.tree.rhs,
                        self.tree.coupling,
                    ],
                    outputs=[self.tree.saved_inverse, self.tree.saved_rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
            else:
                wp.launch(
                    invert_tree_leaves,
                    leaves.shape[0],
                    inputs=[leaves],
                    outputs=[self.tree.saved_inverse],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
                wp.launch(
                    accumulate_tree_messages,
                    recipients.shape[0],
                    inputs=[
                        recipients,
                        recipient_offsets,
                        message_nodes,
                        self.tree.coupling,
                        self.tree.saved_inverse,
                        self.tree.saved_rhs,
                    ],
                    outputs=[self.tree.diagonal, self.tree.rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
            if recipients_are_unique:
                wp.launch(
                    eliminate_tree_response_unique_leaves,
                    leaves.shape[0] * self.closure_count,
                    inputs=[
                        leaves,
                        self.closure_count,
                        self.tree.parent_node,
                        self.tree.coupling,
                        self.tree.saved_inverse,
                    ],
                    outputs=[self.response_rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
            else:
                wp.launch(
                    accumulate_tree_response_messages,
                    recipients.shape[0] * self.closure_count,
                    inputs=[
                        recipients,
                        recipient_offsets,
                        message_nodes,
                        self.closure_count,
                        self.tree.coupling,
                        self.tree.saved_inverse,
                    ],
                    outputs=[self.response_rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )

        wp.launch(
            initialize_tree_backbone_edges,
            self.batch_count * self.backbone_node_count,
            inputs=[self.backbone_node_count, self.backbone_nodes, self.tree.parent_node, self.tree.coupling],
            outputs=[self.backbone_lower, self.backbone_upper],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        for stride, eliminated_count, survivor_count in self.backbone_cr_levels:
            wp.launch(
                invert_tree_backbone_cr_eliminated,
                self.batch_count * eliminated_count,
                inputs=[
                    stride,
                    self.backbone_node_count,
                    eliminated_count,
                    self.backbone_nodes,
                ],
                outputs=[self.tree.diagonal],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            wp.launch(
                reduce_tree_backbone_cr_in_place,
                self.batch_count * survivor_count,
                inputs=[
                    stride,
                    self.backbone_node_count,
                    survivor_count,
                    self.closure_count,
                    self.backbone_nodes,
                ],
                outputs=[
                    self.backbone_lower,
                    self.tree.diagonal,
                    self.backbone_upper,
                    self.tree.rhs,
                    self.response_rhs,
                ],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        wp.launch(
            solve_tree_backbone_cr_coarse,
            self.batch_count,
            inputs=[
                self.backbone_terminal_stride,
                self.backbone_node_count,
                self.closure_count,
                self.backbone_nodes,
            ],
            outputs=[
                self.backbone_lower,
                self.tree.diagonal,
                self.backbone_upper,
                self.tree.rhs,
                self.response_rhs,
            ],
            device=self.device,
        )
        for stride, eliminated_count, _ in reversed(self.backbone_cr_levels):
            wp.launch(
                back_substitute_tree_backbone_cr_in_place,
                self.batch_count * eliminated_count,
                inputs=[
                    stride,
                    self.backbone_node_count,
                    eliminated_count,
                    self.closure_count,
                    self.backbone_nodes,
                    self.backbone_lower,
                    self.backbone_upper,
                ],
                outputs=[self.tree.diagonal, self.tree.rhs, self.response_rhs],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

        for leaves, _, _, _, _ in reversed(self.branch_levels):
            wp.launch(
                back_substitute_tree_level,
                leaves.shape[0],
                inputs=[
                    leaves,
                    self.tree.parent_node,
                    self.tree.coupling,
                    self.tree.saved_inverse,
                    self.tree.saved_rhs,
                ],
                outputs=[self.tree.solution],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            wp.launch(
                back_substitute_tree_response_level,
                leaves.shape[0] * self.closure_count,
                inputs=[
                    leaves,
                    self.closure_count,
                    self.tree.parent_node,
                    self.tree.coupling,
                    self.tree.saved_inverse,
                    self.response_rhs,
                ],
                outputs=[self.response_solution],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

        wp.launch(
            assemble_closure_rhs,
            self.closure_size,
            inputs=[
                self.tree.node_body,
                self.closure_parent_node,
                self.closure_child_node,
                body_scale,
                self.closure_row_scale,
                self.closure_jacobian_parent,
                self.closure_jacobian_child,
                self.closure_residual,
                self.tree.solution,
            ],
            outputs=[self.closure_rhs],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        wp.launch(
            assemble_closure_schur,
            self.batch_count * self.closure_count * self.closure_count,
            inputs=[
                self.closure_count,
                self.tree.node_body,
                self.closure_parent_node,
                self.closure_child_node,
                body_scale,
                self.closure_row_scale,
                self.closure_jacobian_parent,
                self.closure_jacobian_child,
                self.closure_compliance,
                self.response_solution,
            ],
            outputs=[self.closure_schur],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        self.solve_closure_schur()
        wp.launch(
            scatter_closed_tree_body_correction,
            self.tree.body_count,
            inputs=[
                self.tree.body_nodes,
                self.tree.body_slots,
                self.tree.tree_body_count,
                self.closure_count,
                body_scale,
                self.tree.solution,
                self.response_solution,
                self.closure_multiplier,
            ],
            outputs=[body_correction],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )


class StructuralGraphKKT:
    """Graph-capturable compliance-KKT backend for paths, trees, and cyclic graphs."""

    def __init__(self, model, body_inv_mass):
        self.device = model.device
        self.spatial_block_dim = _SPATIAL_GPU_BLOCK_DIM if self.device.is_cuda else 1
        body_inv_mass_host = np.asarray(body_inv_mass.numpy(), dtype=float)
        self.dynamic_body_mask_host = body_inv_mass_host > 0.0
        paths = _build_paths(model, body_inv_mass_host)
        trees = _build_trees(model, {frozenset(path.joints) for path in paths}, body_inv_mass_host)
        handled_joint_sets = {frozenset(component.joints) for component in [*paths, *trees]}
        closed_trees = _build_closed_trees(model, handled_joint_sets, body_inv_mass_host)
        components = [*paths, *trees, *closed_trees]
        body_structural_island = np.full(model.body_count, -1, dtype=np.int32)
        for island, component in enumerate(components):
            body_structural_island[component.bodies] = island
        graph_body_ids = sorted({body for component in components for body in component.bodies})
        graph_body_island = body_structural_island[graph_body_ids]
        body_slot = {body: slot for slot, body in enumerate(graph_body_ids)}
        grouped: dict[tuple[int, int], list[_Path]] = defaultdict(list)
        for path in paths:
            grouped[(len(path.joints), len(path.bodies))].append(path)
        self.path_buckets = [_PathBucket(group, self.device) for group in grouped.values()]
        grouped_trees: dict[tuple, list[_Tree]] = defaultdict(list)
        for tree in trees:
            grouped_trees[_tree_topology_signature(tree)].append(tree)
        self.tree_buckets = [_TreeBucket(group, body_slot, self.device) for group in grouped_trees.values()]
        parent = np.asarray(model.joint_parent.numpy(), dtype=np.int32)
        child = np.asarray(model.joint_child.numpy(), dtype=np.int32)
        grouped_closed_trees: dict[tuple, list[_ClosedTree]] = defaultdict(list)
        for component in closed_trees:
            grouped_closed_trees[_closed_tree_topology_signature(component, parent, child)].append(component)
        self.closed_tree_buckets = [
            _ClosedTreeBucket(group, parent, child, body_slot, self.device) for group in grouped_closed_trees.values()
        ]
        self.buckets = [*self.path_buckets, *self.tree_buckets, *self.closed_tree_buckets]
        for bucket in self.path_buckets:
            bucket.bind_body_endpoints(parent, child, body_slot)

        self.graph_body_count = len(graph_body_ids)
        self.graph_body_ids = wp.array(graph_body_ids, dtype=wp.int32, device=self.device)
        self.graph_body_island = wp.array(graph_body_island, dtype=wp.int32, device=self.device)
        # Per-island contact topology: 1 = no active dynamic-dynamic contact,
        # -1 = active dynamic-dynamic contact.
        self.island_contact_state = wp.ones(len(components), dtype=wp.int32, device=self.device)
        body_slot_by_id = np.full(model.body_count, -1, dtype=np.int32)
        body_slot_by_id[graph_body_ids] = np.arange(len(graph_body_ids), dtype=np.int32)
        graph_joint_ids = sorted({joint for component in components for joint in component.joints})
        self.graph_joint_ids = wp.array(graph_joint_ids, dtype=wp.int32, device=self.device)
        self.body_slot_by_id = wp.array(body_slot_by_id, dtype=wp.int32, device=self.device)
        self.body_correction = wp.zeros(self.graph_body_count, dtype=wp.spatial_vector, device=self.device)
        self.island_step_scale = wp.ones(len(components), dtype=float, device=self.device)
        self.body_matrix = wp.zeros(self.graph_body_count, dtype=wp.spatial_matrix, device=self.device)
        self.body_rhs = wp.zeros(self.graph_body_count, dtype=wp.spatial_vector, device=self.device)
        # Tree equilibration is consumed before its scatter writes the final
        # correction, so both lifetimes share the same compact body buffer.
        self.body_scale = self.body_correction
        # Tree factorization consumes the metric/rhs before path CR. Reuse the
        # same storage in place for path inverse/free motion afterward.
        self.body_inverse = self.body_matrix
        self.body_free = self.body_rhs
        joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int64)
        qd_start = np.asarray(model.joint_qd_start.numpy(), dtype=np.int64)
        dof_dim = np.asarray(model.joint_dof_dim.numpy(), dtype=np.int64)
        limit_ke = np.asarray(model.joint_limit_ke.numpy(), dtype=float)
        limit_lower = np.asarray(model.joint_limit_lower.numpy(), dtype=float)
        limit_upper = np.asarray(model.joint_limit_upper.numpy(), dtype=float)
        self.has_joint_limits = False
        for joint in graph_joint_ids:
            # Cable target coefficients are material parameters, not
            # generalized-coordinate drives. Their constitutive response is
            # already represented by the structural KKT row.
            if joint_type[joint] == int(JointType.CABLE):
                continue
            start = int(qd_start[joint])
            count = int(dof_dim[joint, 0] + dof_dim[joint, 1])
            stop = start + count
            if count > 0 and np.any(
                (limit_ke[start:stop] > 0.0)
                & ((limit_lower[start:stop] > -MAXVAL) | (limit_upper[start:stop] < MAXVAL))
            ):
                self.has_joint_limits = True
                break

    @property
    def active(self) -> bool:
        return bool(self.buckets)

    @property
    def island_count(self) -> int:
        return sum(bucket.batch_count for bucket in self.buckets)

    @property
    def joint_count(self) -> int:
        return sum(bucket.size for bucket in self.buckets)

    def solve(
        self,
        *,
        dt,
        contacts,
        body_q,
        body_inertia_q,
        body_q_prev,
        body_q_rest,
        body_mass,
        body_inv_mass,
        body_inertia,
        body_com,
        contact_hessian_ll,
        contact_hessian_al,
        contact_hessian_aa,
        contact_forces,
        contact_torques,
        dynamic_contact_hessian,
        joint_type,
        joint_enabled,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_cable_rest_kb_local,
        joint_cable_rest_twist,
        joint_qd_start,
        joint_target_q_start,
        joint_dof_dim,
        joint_constraint_start,
        joint_material_k,
        joint_rho,
        joint_penalty_kd,
        joint_target_ke,
        joint_target_kd,
        joint_target_q,
        joint_target_qd,
        joint_limit_lower,
        joint_limit_upper,
        joint_limit_ke,
        joint_limit_kd,
        joint_drive_limit_support,
        joint_drive_lambda,
        joint_limit_lambda,
        joint_lambda_lin,
        joint_lambda_ang,
        joint_C0_lin,
        joint_C0_ang,
        joint_rest_angle,
        joint_sigma_start,
        joint_C_fric,
        stab_alpha,
    ):
        wp.launch(
            build_body_surrogate,
            self.graph_body_ids.shape[0],
            inputs=[
                self.graph_body_ids,
                dt,
                body_q,
                body_inertia_q,
                body_mass,
                body_inv_mass,
                body_inertia,
                body_com,
                contact_hessian_ll,
                contact_hessian_al,
                contact_hessian_aa,
                contact_forces,
                contact_torques,
                dynamic_contact_hessian,
                self.graph_body_island,
                self.island_contact_state,
            ],
            outputs=[self.body_matrix, self.body_rhs],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        for bucket in self.buckets:
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
                    ),
                    (
                        bucket.closure_joint_ids,
                        bucket.closure_size,
                        bucket.closure_jacobian_parent,
                        bucket.closure_jacobian_child,
                        bucket.closure_compliance,
                        bucket.closure_residual,
                        bucket.closure_row_active,
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
                    ),
                )
            )
            for (
                linearization_joint_ids,
                linearization_size,
                jacobian_parent,
                jacobian_child,
                compliance,
                residual,
                row_active,
            ) in linearizations:
                wp.launch(
                    linearize_joint_path_rows,
                    linearization_size,
                    inputs=[
                        linearization_joint_ids,
                        joint_type,
                        joint_enabled,
                        joint_parent,
                        joint_child,
                        joint_X_p,
                        joint_X_c,
                        joint_axis,
                        joint_cable_rest_kb_local,
                        joint_cable_rest_twist,
                        joint_qd_start,
                        joint_target_q_start,
                        joint_dof_dim,
                        joint_constraint_start,
                        joint_material_k,
                        joint_rho,
                        joint_penalty_kd,
                        joint_lambda_lin,
                        joint_lambda_ang,
                        joint_C0_lin,
                        joint_C0_ang,
                        joint_sigma_start,
                        joint_C_fric,
                        joint_target_ke,
                        joint_target_kd,
                        joint_target_q,
                        joint_target_qd,
                        joint_limit_lower,
                        joint_limit_upper,
                        joint_limit_ke,
                        joint_limit_kd,
                        joint_rest_angle,
                        joint_drive_limit_support,
                        joint_drive_lambda,
                        joint_limit_lambda,
                        stab_alpha,
                        body_q,
                        body_q_prev,
                        body_q_rest,
                        body_com,
                        dt,
                    ],
                    outputs=[jacobian_parent, jacobian_child, compliance, residual, row_active],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )

        # Trees consume the body metric directly. Path Schur systems consume
        # its inverse, so solve trees first and then change the shared body
        # buffers to inverse/free-motion form in place.
        self.body_correction.zero_()
        for bucket in self.tree_buckets:
            bucket.solve_tree(self.body_matrix, self.body_rhs, self.body_scale, self.body_correction)
        for bucket in self.closed_tree_buckets:
            bucket.solve_tree(self.body_matrix, self.body_rhs, self.body_scale, self.body_correction)

        if self.path_buckets:
            wp.launch(
                invert_body_surrogate_in_place,
                self.graph_body_count,
                inputs=[self.graph_body_ids, body_inv_mass],
                outputs=[self.body_inverse, self.body_free],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        for bucket in self.path_buckets:
            wp.launch(
                assemble_joint_path_system,
                bucket.size,
                inputs=[
                    bucket.row_count,
                    bucket.row_body,
                    self.body_inverse,
                    self.body_free,
                    bucket.jacobian_parent,
                    bucket.jacobian_child,
                    bucket.compliance,
                    bucket.residual,
                ],
                outputs=[bucket.lower[0], bucket.diagonal[0], bucket.upper[0], bucket.rhs[0]],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            bucket.solve_rows()
            wp.launch(
                compute_path_correction,
                bucket.body_size,
                inputs=[
                    bucket.row_count,
                    bucket.body_count,
                    bucket.body_ids,
                    self.body_slot_by_id,
                    bucket.body_incident_rows,
                    bucket.row_active,
                    bucket.jacobian_parent,
                    bucket.jacobian_child,
                    bucket.solution[0],
                    self.body_inverse,
                    self.body_free,
                    body_inv_mass,
                ],
                outputs=[self.body_correction],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

        # Factor first, then accept one coherent correction per structural
        # island. Dynamic-pair overlap determines a coefficient-free Jacobi
        # relaxation; static/kinematic contact remains in the local VBD solve.
        self.island_step_scale.fill_(1.0)
        if contacts is not None and contacts.rigid_contact_max > 0:
            wp.launch(
                limit_dynamic_contact_jacobi_step,
                self.graph_body_count,
                inputs=[
                    self.graph_body_ids,
                    self.graph_body_island,
                    self.island_contact_state,
                    self.body_correction,
                    contact_hessian_ll,
                    contact_hessian_al,
                    contact_hessian_aa,
                    dt,
                    body_q,
                    body_mass,
                    body_inertia,
                ],
                outputs=[self.island_step_scale],
                device=self.device,
            )
        if self.has_joint_limits:
            wp.launch(
                limit_global_joint_limit_step,
                self.graph_joint_ids.shape[0],
                inputs=[
                    self.graph_joint_ids,
                    joint_type,
                    joint_enabled,
                    joint_parent,
                    joint_child,
                    joint_X_p,
                    joint_X_c,
                    joint_axis,
                    joint_qd_start,
                    joint_dof_dim,
                    joint_limit_lower,
                    joint_limit_upper,
                    joint_limit_ke,
                    joint_rest_angle,
                    self.body_slot_by_id,
                    self.graph_body_island,
                    body_q,
                    body_q_rest,
                    body_com,
                    self.body_correction,
                ],
                outputs=[self.island_step_scale],
                device=self.device,
            )
        wp.launch(
            apply_global_correction,
            self.graph_body_count,
            inputs=[
                self.graph_body_ids,
                self.graph_body_island,
                self.body_correction,
                self.island_step_scale,
                body_com,
            ],
            outputs=[body_q],
            device=self.device,
        )
