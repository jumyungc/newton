# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact float64 atomic-free chain condensation for Cosserat rods coupled to articulated hubs.

Why this route exists
---------------------
In mechanisms where compliant Cosserat cables/tendons wrap around moving links, pulleys, or sliders
(*hubs*), the tree/closed-tree structural route treats dynamic body-body contacts via the block-Jacobi
majorizer ``H_self + H_dynamic = 2 * H_self`` on each body diagonal with zero off-diagonal contact block.
When a phalanx carries ~100 cable contacts at ``k_c ~ 1e8 N/m`` (sum ``~1e10 N/m`` vs link inertia
``m/h^2 ~ 28 N/m``), two pathologies arise without coupled chain-hub condensation:

1. **Co-motion arrest under diagonal majorization:** the link sees ``1e10 N/m`` holding it in place as if
   its own wrapped cable were frozen in space, and each cable segment sees ``2e8 N/m`` holding it against
   the link, requiring many outer global+local sweeps for link and tendon to co-rotate together.
2. **IEEE-754 float32 catastrophic cancellation if subtracting from ``body_matrix``:** in ``float32``,
   ``ulp(1e10) = 596 N/m > 28 N/m``, so storing ``m/h^2 + 2 * sum(K_c)`` in ``float32`` obliterates the
   link's physical inertia completely below machine epsilon before a ``float64`` subtraction can recover it.

What this route does
--------------------
For any structural bucket whose islands consist of unbranched ``JointType.ROD`` chains attached to ``1..HUB_MAX``
hub bodies:

1. **Pure-float64 inertia + stress + Kirchhoff + unilateral active-set contact assembly**
   (:func:`assemble_chain_segments`): builds each segment's ``6 x 6`` diagonal block, rod-neighbour couplings,
   right-hand side, and segment-hub border/diagonal/wrench contributions directly in ``float64`` from body
   inertia, joint stress majorizers, the contact-active Cosserat Kirchhoff director proximal metric
   ``0.25 * L^2 * K_shear``, and unilateral active-set contact Jacobians—never subtracting from ``float32``
   ``body_matrix``;
2. **Parallel ``O(log N)`` block cyclic reduction** (:func:`reduce_chains`, :func:`back_substitute_chains`)
   across ``(segment_count, 1 + 6 * hub_count)`` threads in ``float64``;
3. **100% atomic-free, bit-deterministic register Schur reduction** (:func:`accumulate_chain_schur`) across
   ``(island_count, 6 * HUB_MAX, 1 + 6 * HUB_MAX)`` threads—eliminating millions of contended ``float64``
   global atomics and preserving ``wp.DeterministicMode.RUN_TO_RUN`` compatibility;
4. **Register-blocked ``float64`` Cholesky hub solve & segment back-substitution**
   (:func:`solve_chain_hubs`, :func:`correct_chain_segments`).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.sim import JointType

from .rigid_vbd_kernels import (
    _SMALL_LENGTH_EPS,
    _material_force_terms,
    contact_surface_separation,
    evaluate_rigid_contact_angular_friction,
    evaluate_rigid_contact_normal_sliding,
)

wp.set_module_options({"enable_backward": False})

HUB_MAX = 8
"""Maximum number of hub bodies per island handled by dense ``6 * HUB_MAX`` Schur Cholesky."""

OPEN_CONTACT_WEIGHT = 0.0
"""Optional quadratic weight on open collision-margin gaps (default ``0.0``: open air gaps remain free to close;
rotational whip and active pulley rattle are stabilized physically by the Kirchhoff director proximal metric and
unilateral active-set contact Hessian)."""

ENABLED = True
"""Plan-time switch for ablations against the uncondensed tree/closed-tree route."""

_HUB_DOF = 6 * HUB_MAX
_COLUMNS = 1 + _HUB_DOF


@wp.func
def _inverse_spd6(a: wp.spatial_matrixd):
    """Inverse of a symmetric positive definite ``6 x 6`` matrix by Cholesky; ``NaN`` if not SPD."""
    lower = wp.spatial_matrixd(0.0)
    ok = bool(True)
    for j in range(6):
        pivot = a[j, j]
        for k in range(j):
            pivot = pivot - lower[j, k] * lower[j, k]
        if not (pivot > wp.float64(0.0)):
            ok = False
            pivot = wp.float64(1.0)
        diagonal = wp.sqrt(pivot)
        lower[j, j] = diagonal
        for i in range(j + 1, 6):
            value = a[i, j]
            for k in range(j):
                value = value - lower[i, k] * lower[j, k]
            lower[i, j] = value / diagonal
    lower_inverse = wp.spatial_matrixd(0.0)
    for j in range(6):
        lower_inverse[j, j] = wp.float64(1.0) / lower[j, j]
        for i in range(j + 1, 6):
            value = wp.float64(0.0)
            for k in range(j, i):
                value = value - lower[i, k] * lower_inverse[k, j]
            lower_inverse[i, j] = value / lower[i, i]
    inverse = wp.transpose(lower_inverse) * lower_inverse
    if not ok:
        inverse[0, 0] = wp.sqrt(wp.float64(-1.0))
    return inverse


@wp.func
def _row_weight(compliance: wp.spatial_matrix):
    """Inverse of a joint row's block-diagonal ``(linear 3x3, angular 3x3)`` compliance in ``float64``."""
    linear = wp.mat33d(0.0)
    angular = wp.mat33d(0.0)
    for row in range(3):
        for column in range(3):
            linear[row, column] = wp.float64(compliance[row, column])
            angular[row, column] = wp.float64(compliance[row + 3, column + 3])
    linear = wp.inverse(linear)
    angular = wp.inverse(angular)
    weight = wp.spatial_matrixd(0.0)
    for row in range(3):
        for column in range(3):
            weight[row, column] = linear[row, column]
            weight[row + 3, column + 3] = angular[row, column]
    return weight


@wp.func
def _quat_to_mat33d(q: wp.quat):
    """Orthonormal ``float64`` rotation matrix from a ``float32`` quaternion."""
    x = wp.float64(q[0])
    y = wp.float64(q[1])
    z = wp.float64(q[2])
    w = wp.float64(q[3])
    norm_sq = x * x + y * y + z * z + w * w
    if norm_sq > wp.float64(0.0):
        inv_norm = wp.float64(1.0) / wp.sqrt(norm_sq)
        x = x * inv_norm
        y = y * inv_norm
        z = z * inv_norm
        w = w * inv_norm
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    two = wp.float64(2.0)
    one = wp.float64(1.0)
    return wp.mat33d(
        one - two * (yy + zz),
        two * (xy - wz),
        two * (xz + wy),
        two * (xy + wz),
        one - two * (xx + zz),
        two * (yz - wx),
        two * (xz - wy),
        two * (yz + wx),
        one - two * (xx + yy),
    )


@wp.func
def _transform_point_d(pose: wp.transform, local_point: wp.vec3):
    rot = _quat_to_mat33d(wp.transform_get_rotation(pose))
    pos = wp.vec3d(wp.transform_get_translation(pose))
    return pos + rot * wp.vec3d(local_point)


@wp.func
def _point_stiffness_block_d(k: wp.mat33d, lever_a: wp.vec3d, lever_b: wp.vec3d):
    """``J_a^T K J_b`` for ``J = [I_3, -skew(lever)]`` in ``float64``."""
    skew_a = wp.skew(lever_a)
    skew_b = wp.skew(lever_b)
    linear_angular = -(k * skew_b)
    angular_linear = skew_a * k
    angular_angular = -(skew_a * k * skew_b)
    block = wp.spatial_matrixd(0.0)
    for row in range(3):
        for column in range(3):
            block[row, column] = k[row, column]
            block[row, column + 3] = linear_angular[row, column]
            block[row + 3, column] = angular_linear[row, column]
            block[row + 3, column + 3] = angular_angular[row, column]
    return block


@wp.func
def _contact_pair_block_d(
    stiffness: wp.mat33,
    normal: wp.vec3,
    r_s_a: wp.vec3d,
    r_c_a: wp.vec3d,
    r_s_b: wp.vec3d,
    r_c_b: wp.vec3d,
):
    """Exact ``float64`` pair block ``J_{s,a}^T K_n J_{s,b} + J_{c,a}^T K_t J_{c,b}``."""
    k = wp.mat33d(stiffness)
    n_d = wp.vec3d(normal)
    n_len_sq = wp.dot(n_d, n_d)
    if n_len_sq > wp.float64(0.0):
        n_d = n_d / wp.sqrt(n_len_sq)
    k_n_scalar = wp.dot(n_d, k * n_d)
    k_n = k_n_scalar * wp.outer(n_d, n_d)
    k_t = k - k_n
    return _point_stiffness_block_d(k_n, r_s_a, r_s_b) + _point_stiffness_block_d(k_t, r_c_a, r_c_b)


@wp.func
def _finalize_body_diagonal_d(
    body: int,
    slot: int,
    dt: float,
    pose: wp.transform,
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    contact_block: wp.spatial_matrixd,
    kirchhoff_proximal: wp.float64,
    include_stress: int,
    stress_offsets: wp.array[wp.int32],
    body_stress: wp.array[wp.mat33],
):
    """Assemble ``M_b / dt^2 + contact_block + kirchhoff + eps + stress`` directly in ``float64``."""
    dt_inv = wp.float64(1.0) / wp.float64(dt)
    dt_inv_sq = dt_inv * dt_inv
    rot_d = _quat_to_mat33d(wp.transform_get_rotation(pose))
    mass_d = wp.float64(body_mass[body])
    inertia_d = wp.mat33d(body_inertia[body])
    inert_aa = dt_inv_sq * (rot_d * inertia_d * wp.transpose(rot_d))

    matrix = contact_block
    for axis in range(3):
        matrix[axis, axis] = matrix[axis, axis] + mass_d * dt_inv_sq
    for r in range(3):
        for c in range(3):
            matrix[r + 3, c + 3] = matrix[r + 3, c + 3] + inert_aa[r, c]
    if kirchhoff_proximal > wp.float64(0.0):
        for axis in range(3):
            matrix[axis + 3, axis + 3] = matrix[axis + 3, axis + 3] + kirchhoff_proximal

    trace_ll = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    trace_aa = matrix[3, 3] + matrix[4, 4] + matrix[5, 5]
    eps_l = wp.float64(1.0e-9) * (trace_ll / wp.float64(3.0) + wp.float64(1.0))
    eps_a = wp.float64(1.0e-9) * (trace_aa / wp.float64(3.0) + wp.float64(1.0))
    for axis in range(3):
        matrix[axis, axis] = matrix[axis, axis] + eps_l
        matrix[axis + 3, axis + 3] = matrix[axis + 3, axis + 3] + eps_a

    if include_stress != 0:
        first = stress_offsets[slot]
        last = stress_offsets[slot + 1]
        for entry in range(first, last):
            stress = body_stress[entry]
            for r in range(3):
                for c in range(3):
                    val = stress[r, c]
                    if val != 0.0:
                        matrix[r + 3, c + 3] = matrix[r + 3, c + 3] + wp.float64(val)
    return matrix


@wp.func
def _angular_contact_block_d(k_ang: wp.mat33):
    """Embed a ``3 x 3`` rotational contact metric into the angular corner of a ``6 x 6`` spatial matrix."""
    out = wp.spatial_matrixd(0.0)
    for r in range(3):
        for c in range(3):
            out[r + 3, c + 3] = wp.float64(k_ang[r, c])
    return out


@wp.func
def _body_inertial_rhs_d(
    body: int,
    dt: float,
    pose: wp.transform,
    inertial_pose: wp.transform,
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    use_displacement: int,
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    contact_force: wp.vec3d,
    contact_torque: wp.vec3d,
):
    """Compute inertial + contact objective gradient in ``float64``."""
    dt_inv = wp.float64(1.0) / wp.float64(dt)
    dt_inv_sq = dt_inv * dt_inv
    rotation = wp.transform_get_rotation(pose)
    rot_d = _quat_to_mat33d(rotation)
    mass_d = wp.float64(body_mass[body])
    inertia_d = wp.mat33d(body_inertia[body])
    if use_displacement != 0:
        disp_d = wp.vec3d(body_disp[body])
        disp_ang_d = wp.vec3d(body_disp_ang[body])
        angular_d = rot_d * (inertia_d * (wp.transpose(rot_d) * disp_ang_d))
        force = contact_force - (mass_d * dt_inv_sq) * disp_d
        torque = contact_torque - dt_inv_sq * angular_d
        return wp.spatial_vectord(force[0], force[1], force[2], torque[0], torque[1], torque[2])

    com_local = body_com[body]
    com_cur = _transform_point_d(pose, com_local)
    com_inert = _transform_point_d(inertial_pose, com_local)
    force = (mass_d * dt_inv_sq) * (com_inert - com_cur) + contact_force

    inertial_rotation = wp.transform_get_rotation(inertial_pose)
    rotation_delta = wp.quat_inverse(rotation) * inertial_rotation
    if rotation_delta[3] < 0.0:
        rotation_delta = wp.quat(
            -rotation_delta[0],
            -rotation_delta[1],
            -rotation_delta[2],
            -rotation_delta[3],
        )
    axis, angle = wp.quat_to_axis_angle(rotation_delta)
    torque = rot_d * (inertia_d * (wp.vec3d(axis) * (wp.float64(angle) * dt_inv_sq))) + contact_torque
    return wp.spatial_vectord(force[0], force[1], force[2], torque[0], torque[1], torque[2])


@wp.func
def _contact_levers_d(
    body_idx: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    point_local: wp.vec3,
    offset_local: wp.vec3,
):
    if body_idx < 0:
        return wp.vec3d(point_local), wp.vec3d(point_local + offset_local)
    rot_d = _quat_to_mat33d(wp.transform_get_rotation(body_q[body_idx]))
    com_d = wp.vec3d(body_com[body_idx])
    pt_d = wp.vec3d(point_local)
    off_d = wp.vec3d(offset_local)
    return rot_d * (pt_d - com_d), rot_d * (pt_d + off_d - com_d)


@wp.kernel
def assemble_chain_segments(
    segment_body: wp.array[wp.int32],
    segment_island: wp.array[wp.int32],
    segment_index: wp.array[wp.int32],
    segment_chain_length: wp.array[wp.int32],
    segment_columns: wp.array[wp.int32],
    segment_previous_row: wp.array[wp.int32],
    segment_previous_side: wp.array[wp.int32],
    segment_next_row: wp.array[wp.int32],
    segment_next_side: wp.array[wp.int32],
    segment_link_start: wp.array[wp.int32],
    link_row: wp.array[wp.int32],
    link_side: wp.array[wp.int32],
    link_hub: wp.array[wp.int32],
    body_hub: wp.array[wp.int32],
    body_island: wp.array[wp.int32],
    body_slot_by_id: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    island_contact_state: wp.array[wp.int32],
    segment_kirchhoff_proximal: wp.array[wp.float64],
    include_stress: int,
    stress_offsets: wp.array[wp.int32],
    body_stress: wp.array[wp.mat33],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    compliance: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    row_active: wp.array[wp.int32],
    has_contacts: int,
    open_contact_weight: float,
    dt: float,
    body_q: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    use_displacement: int,
    body_q_prev: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    contact_damping_bidirectional: bool,
    contact_normal_rho: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[wp.vec3],
    contact_tangent_rho: wp.array[wp.mat22],
    contact_torsional_rho: wp.array[float],
    contact_rolling_rho: wp.array[wp.mat22],
    contact_lambda: wp.array[wp.vec3],
    contact_lambda_angular: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    stab_alpha: float,
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_surface_velocity: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    segment_diagonal: wp.array[wp.spatial_matrixd],
    segment_left: wp.array[wp.spatial_matrixd],
    segment_right: wp.array[wp.spatial_matrixd],
    segment_inverse: wp.array[wp.spatial_matrixd],
    segment_rhs: wp.array2d[wp.spatial_vectord],
    segment_border: wp.array2d[wp.spatial_matrixd],
    segment_hub_diagonal: wp.array2d[wp.spatial_matrixd],
    segment_hub_wrench: wp.array2d[wp.spatial_vectord],
    segment_mask: wp.array[wp.int32],
):
    """Assemble one rod segment's pure-``float64`` diagonal, chain couplings, RHS, and hub border."""
    s = wp.tid()
    body = segment_body[s]
    island = segment_island[s]
    columns = segment_columns[s]
    hub_count = (columns - 1) // 6
    slot = body_slot_by_id[body]

    left = wp.spatial_matrixd(0.0)
    right = wp.spatial_matrixd(0.0)
    mask = int(0)
    for hub in range(hub_count):
        segment_border[s, hub] = wp.spatial_matrixd(0.0)
        segment_hub_diagonal[s, hub] = wp.spatial_matrixd(0.0)
        segment_hub_wrench[s, hub] = wp.spatial_vectord(0.0)

    along = wp.vec3(0.0)
    if has_contacts != 0 and open_contact_weight > 0.0:
        before = s
        after = s
        if segment_index[s] > 0:
            before = s - 1
        if segment_index[s] < segment_chain_length[s] - 1:
            after = s + 1
        if before != after:
            body_before = segment_body[before]
            body_after = segment_body[after]
            along = wp.normalize(
                wp.transform_point(body_q[body_after], body_com[body_after])
                - wp.transform_point(body_q[body_before], body_com[body_before])
            )

    contact_block = wp.spatial_matrixd(0.0)
    force_acc = wp.vec3d(0.0)
    torque_acc = wp.vec3d(0.0)

    if has_contacts != 0:
        contact_count = rigid_contact_count[0]
        num_contacts = wp.min(body_contact_counts[body], body_contact_buffer_pre_alloc)
        for i in range(num_contacts):
            contact = body_contact_indices[body * body_contact_buffer_pre_alloc + i]
            if contact >= contact_count:
                continue
            shape0 = rigid_contact_shape0[contact]
            shape1 = rigid_contact_shape1[contact]
            body0 = shape_body[shape0] if shape0 >= 0 else -1
            body1 = shape_body[shape1] if shape1 >= 0 else -1
            if body0 != body and body1 != body:
                continue
            other_body = body1 if body == body0 else body0
            hub = int(-1)
            if other_body >= 0 and body_island[other_body] == island:
                hub = body_hub[other_body]

            point0_local = rigid_contact_point0[contact]
            point1_local = rigid_contact_point1[contact]
            point0_offset = rigid_contact_offset0[contact]
            point1_offset = rigid_contact_offset1[contact]
            normal = rigid_contact_normal[contact]
            point0 = point0_local
            if body0 >= 0:
                point0 = wp.transform_point(body_q[body0], point0_local)
            point1 = point1_local
            if body1 >= 0:
                point1 = wp.transform_point(body_q[body1], point1_local)
            normal_error = -contact_surface_separation(
                point0, point1, normal, rigid_contact_margin0[contact], rigid_contact_margin1[contact]
            )
            normal_weight = contact_normal_rho[contact]
            material_k = contact_material_ke[contact]
            multiplier = contact_lambda[contact]
            multiplier_normal = wp.dot(multiplier, normal)
            C0 = contact_C0[contact]
            C0_normal = wp.dot(normal, C0)
            stabilized_error = normal_error - C0_normal
            tangent_C0 = (1.0 - stab_alpha) * (C0 - normal * C0_normal)
            normal_primal_k, multiplier_normal_eff = _material_force_terms(normal_weight, material_k, multiplier_normal)
            inactive = multiplier_normal <= 0.0 and (
                normal_error <= _SMALL_LENGTH_EPS or normal_primal_k * stabilized_error + multiplier_normal_eff <= 0.0
            )
            if inactive:
                if open_contact_weight > 0.0 and normal_primal_k > 0.0:
                    across = normal - wp.dot(normal, along) * along
                    wall_k = wp.mat33d((open_contact_weight * normal_primal_k) * wp.outer(across, across))
                    mine_local = point0_local if body == body0 else point1_local
                    mine_offset = point0_offset if body == body0 else point1_offset
                    _, wall_lever = _contact_levers_d(body, body_q, body_com, mine_local, mine_offset)
                    wall_self = _point_stiffness_block_d(wall_k, wall_lever, wall_lever)
                    if hub >= 0:
                        contact_block = contact_block + wall_self
                        hub_local = point1_local if body == body0 else point0_local
                        hub_offset = point1_offset if body == body0 else point0_offset
                        _, wall_hub_lever = _contact_levers_d(other_body, body_q, body_com, hub_local, hub_offset)
                        segment_hub_diagonal[s, hub] = segment_hub_diagonal[s, hub] + _point_stiffness_block_d(
                            wall_k, wall_hub_lever, wall_hub_lever
                        )
                        segment_border[s, hub] = segment_border[s, hub] - _point_stiffness_block_d(
                            wall_k, wall_lever, wall_hub_lever
                        )
                        mask = mask | (1 << hub)
                    else:
                        scale = (
                            wp.float64(2.0)
                            if (other_body >= 0 and body_inv_mass[other_body] > 0.0)
                            else wp.float64(1.0)
                        )
                        contact_block = contact_block + scale * wall_self
                continue

            surface_velocity = wp.vec3(0.0)
            if rigid_contact_surface_velocity:
                surface_velocity = rigid_contact_surface_velocity[contact]
            friction_mu = contact_material_mu[contact]
            (
                force0,
                torque0,
                stiffness,
                _h_al0,
                _h_aa0,
                force1,
                torque1,
                _h_ll1,
                _h_al1,
                _h_aa1,
            ) = evaluate_rigid_contact_normal_sliding(
                body0,
                body1,
                body_q,
                body_q_prev,
                body_com,
                point0_local,
                point1_local,
                surface_velocity,
                point0_offset,
                point1_offset,
                normal,
                stabilized_error,
                normal_weight,
                material_k,
                contact_tangent_rho[contact],
                contact_material_kd[contact],
                multiplier,
                friction_mu[0],
                dt,
                tangent_C0,
                contact_damping_bidirectional,
                1,
            )
            mu_torsional = friction_mu[1]
            mu_rolling = friction_mu[2]
            normal_load = wp.max(normal_primal_k * stabilized_error + multiplier_normal_eff, 0.0)
            has_angular_friction = normal_load > 0.0 and (mu_torsional > 0.0 or mu_rolling > 0.0)
            ang_block = wp.spatial_matrixd(0.0)
            if has_angular_friction:
                pure_torque_b, k_angular = evaluate_rigid_contact_angular_friction(
                    body0,
                    body1,
                    body_q,
                    body_q_prev,
                    normal,
                    normal_load,
                    mu_torsional,
                    mu_rolling,
                    contact_torsional_rho[contact],
                    contact_rolling_rho[contact],
                    contact_lambda_angular[contact],
                )
                torque0 -= pure_torque_b
                torque1 += pure_torque_b
                ang_block = _angular_contact_block_d(k_angular)

            r_s_0, r_c_0 = _contact_levers_d(body0, body_q, body_com, point0_local, point0_offset)
            r_s_1, r_c_1 = _contact_levers_d(body1, body_q, body_com, point1_local, point1_offset)

            if body == body0:
                my_f = wp.vec3d(force0)
                my_t = wp.vec3d(torque0)
                other_f = wp.vec3d(force1)
                other_t = wp.vec3d(torque1)
                r_s_me, r_c_me = r_s_0, r_c_0
                r_s_other, r_c_other = r_s_1, r_c_1
            else:
                my_f = wp.vec3d(force1)
                my_t = wp.vec3d(torque1)
                other_f = wp.vec3d(force0)
                other_t = wp.vec3d(torque0)
                r_s_me, r_c_me = r_s_1, r_c_1
                r_s_other, r_c_other = r_s_0, r_c_0

            force_acc = force_acc + my_f
            torque_acc = torque_acc + my_t
            mine_block = _contact_pair_block_d(stiffness, normal, r_s_me, r_c_me, r_s_me, r_c_me) + ang_block

            if hub >= 0:
                # Coupled segment-hub contact: +1x on segment, +1x on hub, -1x off-diagonal border.
                contact_block = contact_block + mine_block
                hub_block = (
                    _contact_pair_block_d(stiffness, normal, r_s_other, r_c_other, r_s_other, r_c_other) + ang_block
                )
                cross_block = _contact_pair_block_d(stiffness, normal, r_s_me, r_c_me, r_s_other, r_c_other) + ang_block
                segment_hub_diagonal[s, hub] = segment_hub_diagonal[s, hub] + hub_block
                segment_border[s, hub] = segment_border[s, hub] - cross_block
                segment_hub_wrench[s, hub] = segment_hub_wrench[s, hub] + wp.spatial_vectord(
                    other_f[0], other_f[1], other_f[2], other_t[0], other_t[1], other_t[2]
                )
                mask = mask | (1 << hub)
            else:
                # Uncoupled contact (world, other island, or segment-segment): majorizer weight 1x or 2x.
                scale = wp.float64(2.0) if (other_body >= 0 and body_inv_mass[other_body] > 0.0) else wp.float64(1.0)
                contact_block = contact_block + scale * mine_block

    kirchhoff = wp.float64(0.0)
    if has_contacts != 0 and island_contact_state[graph_body_island[slot]] == -1:
        kirchhoff = segment_kirchhoff_proximal[s]

    pose = body_q[body]
    diagonal = _finalize_body_diagonal_d(
        body,
        slot,
        dt,
        pose,
        body_mass,
        body_inertia,
        contact_block,
        kirchhoff,
        include_stress,
        stress_offsets,
        body_stress,
    )
    rhs = _body_inertial_rhs_d(
        body,
        dt,
        pose,
        body_inertia_q[body],
        body_disp,
        body_disp_ang,
        use_displacement,
        body_mass,
        body_inertia,
        body_com,
        force_acc,
        torque_acc,
    )

    # Incident rod joints along the chain.
    for which in range(2):
        row = segment_previous_row[s]
        side = segment_previous_side[s]
        if which == 1:
            row = segment_next_row[s]
            side = segment_next_side[s]
        if row < 0 or row_active[row] == 0:
            continue
        weight = _row_weight(compliance[row])
        mine = wp.spatial_matrixd(jacobian_parent[row])
        other = wp.spatial_matrixd(jacobian_child[row])
        if side == 1:
            mine = wp.spatial_matrixd(jacobian_child[row])
            other = wp.spatial_matrixd(jacobian_parent[row])
        mine_weighted = wp.transpose(mine) * weight
        diagonal = diagonal + mine_weighted * mine
        rhs = rhs - mine_weighted * wp.spatial_vectord(residual[row])
        if which == 0:
            left = mine_weighted * other
        else:
            right = mine_weighted * other

    # Attachment joints from this segment to hubs or ground.
    for entry in range(segment_link_start[s], segment_link_start[s + 1]):
        row = link_row[entry]
        if row_active[row] == 0:
            continue
        weight = _row_weight(compliance[row])
        mine = wp.spatial_matrixd(jacobian_parent[row])
        other = wp.spatial_matrixd(jacobian_child[row])
        if link_side[entry] == 1:
            mine = wp.spatial_matrixd(jacobian_child[row])
            other = wp.spatial_matrixd(jacobian_parent[row])
        mine_weighted = wp.transpose(mine) * weight
        diagonal = diagonal + mine_weighted * mine
        rhs = rhs - mine_weighted * wp.spatial_vectord(residual[row])
        hub = link_hub[entry]
        if hub >= 0:
            segment_border[s, hub] = segment_border[s, hub] + mine_weighted * other
            mask = mask | (1 << hub)

    segment_diagonal[s] = diagonal
    segment_left[s] = left
    segment_right[s] = right
    if segment_index[s] % 2 == 1 or segment_chain_length[s] == 1:
        segment_inverse[s] = _inverse_spd6(diagonal)
    segment_rhs[s, 0] = rhs
    for hub in range(hub_count):
        border = segment_border[s, hub]
        for j in range(6):
            segment_rhs[s, 1 + 6 * hub + j] = wp.spatial_vectord(
                border[0, j], border[1, j], border[2, j], border[3, j], border[4, j], border[5, j]
            )
    segment_mask[s] = mask


@wp.kernel
def reduce_chains(
    stride: int,
    segment_index: wp.array[wp.int32],
    segment_chain_length: wp.array[wp.int32],
    segment_columns: wp.array[wp.int32],
    segment_diagonal: wp.array[wp.spatial_matrixd],
    segment_left: wp.array[wp.spatial_matrixd],
    segment_right: wp.array[wp.spatial_matrixd],
    segment_inverse: wp.array[wp.spatial_matrixd],
    segment_rhs: wp.array2d[wp.spatial_vectord],
):
    """One parallel level of ``float64`` block cyclic reduction across all chains."""
    s, column = wp.tid()
    index = segment_index[s]
    length = segment_chain_length[s]
    if stride >= length or index % (2 * stride) != 0:
        return
    has_left = index > 0
    has_right = index + stride < length
    if column == _COLUMNS:
        diagonal = segment_diagonal[s]
        new_left = wp.spatial_matrixd(0.0)
        new_right = wp.spatial_matrixd(0.0)
        if has_left:
            e = s - stride
            toward = segment_right[e]
            reduced = wp.transpose(toward) * segment_inverse[e]
            diagonal = diagonal - reduced * toward
            new_left = -(reduced * segment_left[e])
        if has_right:
            e = s + stride
            toward = segment_left[e]
            reduced = wp.transpose(toward) * segment_inverse[e]
            diagonal = diagonal - reduced * toward
            if index + 2 * stride < length:
                new_right = -(reduced * segment_right[e])
        segment_diagonal[s] = diagonal
        segment_left[s] = new_left
        segment_right[s] = new_right
        if index % (4 * stride) == 2 * stride or (index == 0 and 2 * stride >= length):
            segment_inverse[s] = _inverse_spd6(diagonal)
        return
    if column >= segment_columns[s]:
        return
    rhs = segment_rhs[s, column]
    if has_left:
        e = s - stride
        rhs = rhs - wp.transpose(segment_right[e]) * (segment_inverse[e] * segment_rhs[e, column])
    if has_right:
        e = s + stride
        rhs = rhs - wp.transpose(segment_left[e]) * (segment_inverse[e] * segment_rhs[e, column])
    segment_rhs[s, column] = rhs


@wp.kernel
def back_substitute_chains(
    stride: int,
    segment_index: wp.array[wp.int32],
    segment_chain_length: wp.array[wp.int32],
    segment_chain_top: wp.array[wp.int32],
    segment_columns: wp.array[wp.int32],
    segment_left: wp.array[wp.spatial_matrixd],
    segment_right: wp.array[wp.spatial_matrixd],
    segment_inverse: wp.array[wp.spatial_matrixd],
    segment_rhs: wp.array2d[wp.spatial_vectord],
    segment_solution: wp.array2d[wp.spatial_vectord],
):
    """One parallel level of back-substitution of :func:`reduce_chains`."""
    s, column = wp.tid()
    if column >= segment_columns[s]:
        return
    index = segment_index[s]
    length = segment_chain_length[s]
    top = segment_chain_top[s]
    if index == 0:
        if stride == top or (length == 1 and stride == 1):
            segment_solution[s, column] = segment_inverse[s] * segment_rhs[s, column]
        return
    if stride >= length or index % (2 * stride) != stride:
        return
    rhs = segment_rhs[s, column]
    e = s - stride
    before = segment_solution[e, column]
    if index == stride and stride == top:
        before = segment_inverse[e] * segment_rhs[e, column]
    rhs = rhs - segment_left[s] * before
    if index + stride < length:
        rhs = rhs - segment_right[s] * segment_solution[s + stride, column]
    segment_solution[s, column] = segment_inverse[s] * rhs


@wp.kernel
def accumulate_chain_schur(
    island_segment_start: wp.array[wp.int32],
    island_hub_start: wp.array[wp.int32],
    segment_mask: wp.array[wp.int32],
    segment_border: wp.array2d[wp.spatial_matrixd],
    segment_hub_diagonal: wp.array2d[wp.spatial_matrixd],
    segment_hub_wrench: wp.array2d[wp.spatial_vectord],
    segment_solution: wp.array2d[wp.spatial_vectord],
    schur: wp.array3d[wp.float64],
    schur_rhs: wp.array2d[wp.float64],
):
    """Atomic-free, bit-deterministic register Schur reduction; thread ``(island, hub_dof_r, column)``."""
    e, r, column = wp.tid()
    hub_count = island_hub_start[e + 1] - island_hub_start[e]
    dof_count = 6 * hub_count
    if r >= dof_count or column >= 1 + dof_count:
        return

    hub = r // 6
    i = r - 6 * hub
    seg_begin = island_segment_start[e]
    seg_end = island_segment_start[e + 1]
    acc = wp.float64(0.0)

    if column == 0:
        for s in range(seg_begin, seg_end):
            if ((segment_mask[s] >> hub) & 1) == 0:
                continue
            border = segment_border[s, hub]
            sol = segment_solution[s, 0]
            dot = (
                border[0, i] * sol[0]
                + border[1, i] * sol[1]
                + border[2, i] * sol[2]
                + border[3, i] * sol[3]
                + border[4, i] * sol[4]
                + border[5, i] * sol[5]
            )
            acc = acc + segment_hub_wrench[s, hub][i] - dot
        schur_rhs[e, r] = acc
    else:
        c = column - 1
        col_hub = c // 6
        j = c - 6 * col_hub
        same_hub = int(hub == col_hub)
        for s in range(seg_begin, seg_end):
            if ((segment_mask[s] >> hub) & 1) == 0:
                continue
            border = segment_border[s, hub]
            sol = segment_solution[s, column]
            dot = (
                border[0, i] * sol[0]
                + border[1, i] * sol[1]
                + border[2, i] * sol[2]
                + border[3, i] * sol[3]
                + border[4, i] * sol[4]
                + border[5, i] * sol[5]
            )
            if same_hub != 0:
                acc = acc + segment_hub_diagonal[s, hub][i, j] - dot
            else:
                acc = acc - dot
        schur[e, r, c] = acc


@wp.kernel
def solve_chain_hubs(
    island_hub_start: wp.array[wp.int32],
    island_row_start: wp.array[wp.int32],
    hub_body: wp.array[wp.int32],
    hub_row: wp.array[wp.int32],
    hub_row_parent: wp.array[wp.int32],
    hub_row_child: wp.array[wp.int32],
    body_island: wp.array[wp.int32],
    body_is_segment: wp.array[wp.int32],
    body_slot_by_id: wp.array[wp.int32],
    include_stress: int,
    stress_offsets: wp.array[wp.int32],
    body_stress: wp.array[wp.mat33],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    compliance: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    row_active: wp.array[wp.int32],
    has_contacts: int,
    dt: float,
    body_q: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    use_displacement: int,
    body_q_prev: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    contact_damping_bidirectional: bool,
    contact_normal_rho: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[wp.vec3],
    contact_tangent_rho: wp.array[wp.mat22],
    contact_torsional_rho: wp.array[float],
    contact_rolling_rho: wp.array[wp.mat22],
    contact_lambda: wp.array[wp.vec3],
    contact_lambda_angular: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    stab_alpha: float,
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_surface_velocity: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    schur: wp.array3d[wp.float64],
    schur_rhs: wp.array2d[wp.float64],
    keep_hub_base: int,
    hub_base_out: wp.array2d[wp.spatial_matrixd],
    hub_rhs_out: wp.array2d[wp.spatial_vectord],
    hub_block: wp.array3d[wp.spatial_matrixd],
    hub_vector: wp.array2d[wp.spatial_vectord],
    hub_solution: wp.array2d[wp.float64],
    body_correction: wp.array[wp.spatial_vector],
):
    """Assemble and solve the ``6 * hub_count`` condensed hub system of island ``e`` in ``float64``."""
    e = wp.tid()
    first_hub = island_hub_start[e]
    hub_count = island_hub_start[e + 1] - first_hub

    for i in range(hub_count):
        body = hub_body[first_hub + i]
        slot = body_slot_by_id[body]
        ext_contact_block = wp.spatial_matrixd(0.0)
        ext_force = wp.vec3d(0.0)
        ext_torque = wp.vec3d(0.0)

        if has_contacts != 0:
            contact_count = rigid_contact_count[0]
            num_contacts = wp.min(body_contact_counts[body], body_contact_buffer_pre_alloc)
            for m in range(num_contacts):
                contact = body_contact_indices[body * body_contact_buffer_pre_alloc + m]
                if contact >= contact_count:
                    continue
                shape0 = rigid_contact_shape0[contact]
                shape1 = rigid_contact_shape1[contact]
                body0 = shape_body[shape0] if shape0 >= 0 else -1
                body1 = shape_body[shape1] if shape1 >= 0 else -1
                if body0 != body and body1 != body:
                    continue
                other_body = body1 if body == body0 else body0
                if other_body >= 0 and body_island[other_body] == e and body_is_segment[other_body] != 0:
                    continue

                point0_local = rigid_contact_point0[contact]
                point1_local = rigid_contact_point1[contact]
                point0_offset = rigid_contact_offset0[contact]
                point1_offset = rigid_contact_offset1[contact]
                normal = rigid_contact_normal[contact]
                point0 = point0_local if body0 < 0 else wp.transform_point(body_q[body0], point0_local)
                point1 = point1_local if body1 < 0 else wp.transform_point(body_q[body1], point1_local)
                normal_error = -contact_surface_separation(
                    point0, point1, normal, rigid_contact_margin0[contact], rigid_contact_margin1[contact]
                )
                normal_weight = contact_normal_rho[contact]
                material_k = contact_material_ke[contact]
                multiplier = contact_lambda[contact]
                multiplier_normal = wp.dot(multiplier, normal)
                C0 = contact_C0[contact]
                C0_normal = wp.dot(normal, C0)
                stabilized_error = normal_error - C0_normal
                tangent_C0 = (1.0 - stab_alpha) * (C0 - normal * C0_normal)
                normal_primal_k, multiplier_normal_eff = _material_force_terms(
                    normal_weight, material_k, multiplier_normal
                )
                if multiplier_normal <= 0.0 and (
                    normal_error <= _SMALL_LENGTH_EPS
                    or normal_primal_k * stabilized_error + multiplier_normal_eff <= 0.0
                ):
                    continue
                surface_velocity = wp.vec3(0.0)
                if rigid_contact_surface_velocity:
                    surface_velocity = rigid_contact_surface_velocity[contact]
                friction_mu = contact_material_mu[contact]
                (
                    force0,
                    torque0,
                    stiffness,
                    _h_al0,
                    _h_aa0,
                    force1,
                    torque1,
                    _h_ll1,
                    _h_al1,
                    _h_aa1,
                ) = evaluate_rigid_contact_normal_sliding(
                    body0,
                    body1,
                    body_q,
                    body_q_prev,
                    body_com,
                    point0_local,
                    point1_local,
                    surface_velocity,
                    point0_offset,
                    point1_offset,
                    normal,
                    stabilized_error,
                    normal_weight,
                    material_k,
                    contact_tangent_rho[contact],
                    contact_material_kd[contact],
                    multiplier,
                    friction_mu[0],
                    dt,
                    tangent_C0,
                    contact_damping_bidirectional,
                    1,
                )
                mu_torsional = friction_mu[1]
                mu_rolling = friction_mu[2]
                normal_load = wp.max(normal_primal_k * stabilized_error + multiplier_normal_eff, 0.0)
                ang_block = wp.spatial_matrixd(0.0)
                if normal_load > 0.0 and (mu_torsional > 0.0 or mu_rolling > 0.0):
                    pure_torque_b, k_angular = evaluate_rigid_contact_angular_friction(
                        body0,
                        body1,
                        body_q,
                        body_q_prev,
                        normal,
                        normal_load,
                        mu_torsional,
                        mu_rolling,
                        contact_torsional_rho[contact],
                        contact_rolling_rho[contact],
                        contact_lambda_angular[contact],
                    )
                    torque0 -= pure_torque_b
                    torque1 += pure_torque_b
                    ang_block = _angular_contact_block_d(k_angular)
                if body == body0:
                    ext_force = ext_force + wp.vec3d(force0)
                    ext_torque = ext_torque + wp.vec3d(torque0)
                    r_s_me, r_c_me = _contact_levers_d(body0, body_q, body_com, point0_local, point0_offset)
                else:
                    ext_force = ext_force + wp.vec3d(force1)
                    ext_torque = ext_torque + wp.vec3d(torque1)
                    r_s_me, r_c_me = _contact_levers_d(body1, body_q, body_com, point1_local, point1_offset)
                mine_block = _contact_pair_block_d(stiffness, normal, r_s_me, r_c_me, r_s_me, r_c_me) + ang_block
                scale = wp.float64(2.0) if (other_body >= 0 and body_inv_mass[other_body] > 0.0) else wp.float64(1.0)
                ext_contact_block = ext_contact_block + scale * mine_block

        pose = body_q[body]
        base_block = _finalize_body_diagonal_d(
            body,
            slot,
            dt,
            pose,
            body_mass,
            body_inertia,
            ext_contact_block,
            wp.float64(0.0),
            include_stress,
            stress_offsets,
            body_stress,
        )
        base_rhs = _body_inertial_rhs_d(
            body,
            dt,
            pose,
            body_inertia_q[body],
            body_disp,
            body_disp_ang,
            use_displacement,
            body_mass,
            body_inertia,
            body_com,
            ext_force,
            ext_torque,
        )
        if keep_hub_base != 0:
            hub_base_out[e, i] = base_block
            hub_rhs_out[e, i] = base_rhs

        block = base_block
        vector = base_rhs
        for r in range(6):
            vector[r] = vector[r] + schur_rhs[e, 6 * i + r]
            for c in range(6):
                block[r, c] = block[r, c] + schur[e, 6 * i + r, 6 * i + c]
        hub_block[e, i, i] = block
        hub_vector[e, i] = vector
        for j in range(i):
            below = wp.spatial_matrixd(0.0)
            for r in range(6):
                for c in range(6):
                    below[r, c] = schur[e, 6 * i + r, 6 * j + c]
            hub_block[e, i, j] = below

    for entry in range(island_row_start[e], island_row_start[e + 1]):
        row = hub_row[entry]
        if row_active[row] == 0:
            continue
        parent = hub_row_parent[entry]
        child = hub_row_child[entry]
        weight = _row_weight(compliance[row])
        jacobian_p = wp.spatial_matrixd(jacobian_parent[row])
        jacobian_c = wp.spatial_matrixd(jacobian_child[row])
        row_residual = wp.spatial_vectord(residual[row])
        if parent >= 0:
            weighted = wp.transpose(jacobian_p) * weight
            hub_block[e, parent, parent] = hub_block[e, parent, parent] + weighted * jacobian_p
            hub_vector[e, parent] = hub_vector[e, parent] - weighted * row_residual
            if child >= 0:
                cross = weighted * jacobian_c
                if parent > child:
                    hub_block[e, parent, child] = hub_block[e, parent, child] + cross
                else:
                    hub_block[e, child, parent] = hub_block[e, child, parent] + wp.transpose(cross)
        if child >= 0:
            weighted = wp.transpose(jacobian_c) * weight
            hub_block[e, child, child] = hub_block[e, child, child] + weighted * jacobian_c
            hub_vector[e, child] = hub_vector[e, child] - weighted * row_residual

    for j in range(hub_count):
        inverse = _inverse_spd6(hub_block[e, j, j])
        hub_block[e, j, j] = inverse
        pivot_vector = hub_vector[e, j]
        for i in range(j + 1, hub_count):
            reduced = hub_block[e, i, j] * inverse
            hub_vector[e, i] = hub_vector[e, i] - reduced * pivot_vector
            for k in range(j + 1, i + 1):
                hub_block[e, i, k] = hub_block[e, i, k] - reduced * wp.transpose(hub_block[e, k, j])
    for reverse in range(hub_count):
        j = hub_count - 1 - reverse
        value = hub_vector[e, j]
        for i in range(j + 1, hub_count):
            value = value - wp.transpose(hub_block[e, i, j]) * hub_vector[e, i]
        value = hub_block[e, j, j] * value
        hub_vector[e, j] = value
        slot = body_slot_by_id[hub_body[first_hub + j]]
        correction = wp.spatial_vector()
        for r in range(6):
            hub_solution[e, 6 * j + r] = value[r]
            correction[r] = float(value[r])
        body_correction[slot] = correction


@wp.kernel
def correct_chain_segments(
    segment_body: wp.array[wp.int32],
    segment_island: wp.array[wp.int32],
    segment_columns: wp.array[wp.int32],
    body_slot_by_id: wp.array[wp.int32],
    segment_solution: wp.array2d[wp.spatial_vectord],
    hub_solution: wp.array2d[wp.float64],
    body_correction: wp.array[wp.spatial_vector],
):
    """Recover each rod segment correction ``z - Z * delta_hub``."""
    s = wp.tid()
    e = segment_island[s]
    value = segment_solution[s, 0]
    for column in range(1, segment_columns[s]):
        value = value - hub_solution[e, column - 1] * segment_solution[s, column]
    correction = wp.spatial_vector()
    for i in range(6):
        correction[i] = float(value[i])
    body_correction[body_slot_by_id[segment_body[s]]] = correction


def island_in_scope(tree, joint_type, joint_parent, joint_child, body_inv_mass) -> bool:
    """Return True iff one structural island qualifies for :class:`ChainCondensation`."""
    if not ENABLED:
        return False
    rod = int(JointType.ROD)
    attachment_types = (int(JointType.BALL), int(JointType.FIXED))
    free = int(JointType.FREE)

    bodies = [body for body in tree.bodies if body >= 0 and body_inv_mass[body] > 0.0]
    if not bodies:
        return False
    joints_of = {body: [] for body in bodies}
    for joint in tree.joints:
        for end in (int(joint_parent[joint]), int(joint_child[joint])):
            if end in joints_of:
                joints_of[end].append(joint)
    segments = {
        body
        for body in bodies
        if any(joint_type[j] == rod for j in joints_of[body])
        and all(joint_type[j] == rod or joint_type[j] in attachment_types for j in joints_of[body])
    }
    hubs = [body for body in bodies if body not in segments]
    if not segments or not hubs or len(hubs) > HUB_MAX:
        return False
    neighbours = {body: [] for body in segments}
    for joint in tree.joints:
        kind = int(joint_type[joint])
        if kind == free:
            continue
        parent, child = int(joint_parent[joint]), int(joint_child[joint])
        ends = [end for end in (parent, child) if end in segments]
        if len(ends) == 2:
            if kind != rod:
                return False
            neighbours[parent].append(child)
            neighbours[child].append(parent)
    if any(len(near) > 2 for near in neighbours.values()):
        return False
    visited = set()
    for start in sorted(segments):
        if start in visited or len(neighbours[start]) > 1:
            continue
        cur = start
        visited.add(cur)
        while True:
            nxt = [b for b in neighbours[cur] if b not in visited]
            if not nxt:
                break
            cur = nxt[0]
            visited.add(cur)
    return len(visited) == len(segments)


class ChainCondensation:
    """Device plan and exact ``float64`` atomic-free solve for coupled rod-chain + hub islands."""

    keep_assembly = False
    """Set in unit tests to capture exact ``float64`` pre-reduction blocks on ``self.assembly``."""

    @staticmethod
    def plan(buckets, joint_type, joint_parent, joint_child, body_inv_mass, body_count, device):
        """Plan one shared chain-condensation route across all qualifying buckets of a solver."""
        offsets = [None] * len(buckets)
        if not ENABLED:
            return None, offsets
        rod = int(JointType.ROD)
        attachment_types = (int(JointType.BALL), int(JointType.FIXED))
        free = int(JointType.FREE)

        def dynamic(body):
            return body >= 0 and body_inv_mass[body] > 0.0

        names = (
            "segment_body",
            "segment_island",
            "segment_index",
            "segment_chain_length",
            "segment_chain_top",
            "segment_columns",
            "segment_previous_row",
            "segment_previous_side",
            "segment_next_row",
            "segment_next_side",
            "link_row",
            "link_side",
            "link_hub",
            "hub_body",
            "hub_row",
            "hub_row_parent",
            "hub_row_child",
        )
        total = {name: [] for name in names}
        link_start, island_segment_start, island_hub_start, island_row_start = [0], [0], [0], [0]
        body_hub = np.full(body_count, -1, dtype=np.int32)
        body_island = np.full(body_count, -1, dtype=np.int32)
        body_is_segment = np.zeros(body_count, dtype=np.int32)
        chain_count = island_count = row_count = 0
        row_joint = []

        for bucket_index, (islands, row_joints) in enumerate(buckets):
            rows = {int(joint): row_count + row for row, joint in enumerate(row_joints)}
            found = {name: [] for name in names}
            found_links, found_seg_starts, found_hub_starts, found_row_starts = [], [], [], []
            hub_of, island_of, seg_bodies = {}, {}, []
            chains = 0
            in_scope = True
            for local_island, tree in enumerate(islands):
                island = island_count + local_island
                bodies = [body for body in tree.bodies if dynamic(body)]
                joints_of = {body: [] for body in bodies}
                for joint in tree.joints:
                    for end in (int(joint_parent[joint]), int(joint_child[joint])):
                        if end in joints_of:
                            joints_of[end].append(joint)
                segments = {
                    body
                    for body in bodies
                    if any(joint_type[j] == rod for j in joints_of[body])
                    and all(joint_type[j] == rod or joint_type[j] in attachment_types for j in joints_of[body])
                }
                hubs = [body for body in bodies if body not in segments]
                if not segments or not hubs or len(hubs) > HUB_MAX:
                    in_scope = False
                    break
                neighbours = {body: [] for body in segments}
                links = {body: [] for body in segments}
                for joint in tree.joints:
                    kind = int(joint_type[joint])
                    if kind == free:
                        continue
                    parent, child = int(joint_parent[joint]), int(joint_child[joint])
                    ends = [end for end in (parent, child) if end in segments]
                    if len(ends) == 2:
                        if kind != rod:
                            in_scope = False
                            break
                        neighbours[parent].append((child, joint))
                        neighbours[child].append((parent, joint))
                    elif len(ends) == 1:
                        links[ends[0]].append(joint)
                if not in_scope or any(len(near) > 2 for near in neighbours.values()):
                    in_scope = False
                    break
                for index, body in enumerate(hubs):
                    hub_of[body] = index
                    island_of[body] = island
                found["hub_body"].extend(hubs)
                found_hub_starts.append(len(found["hub_body"]))

                visited = set()
                for start in sorted(segments):
                    if start in visited or len(neighbours[start]) > 1:
                        continue
                    chain, joints = [start], []
                    visited.add(start)
                    while True:
                        step = [(body, joint) for body, joint in neighbours[chain[-1]] if body not in visited]
                        if not step:
                            break
                        chain.append(step[0][0])
                        joints.append(step[0][1])
                        visited.add(step[0][0])
                    top = 1
                    while 2 * top < len(chain):
                        top *= 2
                    for index, body in enumerate(chain):
                        island_of[body] = island
                        seg_bodies.append(body)
                        found["segment_body"].append(body)
                        found["segment_island"].append(island)
                        found["segment_index"].append(index)
                        found["segment_chain_length"].append(len(chain))
                        found["segment_chain_top"].append(top if len(chain) > 1 else 0)
                        found["segment_columns"].append(1 + 6 * len(hubs))
                        before = joints[index - 1] if index > 0 else -1
                        after = joints[index] if index < len(joints) else -1
                        found["segment_previous_row"].append(rows[before] if before >= 0 else -1)
                        found["segment_previous_side"].append(int(before >= 0 and int(joint_child[before]) == body))
                        found["segment_next_row"].append(rows[after] if after >= 0 else -1)
                        found["segment_next_side"].append(int(after >= 0 and int(joint_child[after]) == body))
                        for joint in links[body]:
                            is_child = int(joint_child[joint]) == body
                            other = int(joint_parent[joint]) if is_child else int(joint_child[joint])
                            found["link_row"].append(rows[joint])
                            found["link_side"].append(int(is_child))
                            found["link_hub"].append(hub_of[other] if dynamic(other) else -1)
                        found_links.append(len(found["link_row"]))
                    chains += 1
                if len(visited) != len(segments):
                    in_scope = False
                    break
                found_seg_starts.append(len(found["segment_body"]))

                for joint in tree.joints:
                    if int(joint_type[joint]) == free:
                        continue
                    parent, child = int(joint_parent[joint]), int(joint_child[joint])
                    hub_parent = hub_of[parent] if parent in joints_of and parent not in segments else -1
                    hub_child = hub_of[child] if child in joints_of and child not in segments else -1
                    if hub_parent >= 0 or hub_child >= 0:
                        found["hub_row"].append(rows[joint])
                        found["hub_row_parent"].append(hub_parent)
                        found["hub_row_child"].append(hub_child)
                found_row_starts.append(len(found["hub_row"]))
            if not in_scope:
                continue

            offsets[bucket_index] = row_count
            row_count += len(row_joints)
            row_joint.extend(int(joint) for joint in row_joints)
            island_count += len(islands)
            chain_count += chains
            link_start.extend(len(total["link_row"]) + value for value in found_links)
            island_segment_start.extend(len(total["segment_body"]) + value for value in found_seg_starts)
            island_hub_start.extend(len(total["hub_body"]) + value for value in found_hub_starts)
            island_row_start.extend(len(total["hub_row"]) + value for value in found_row_starts)
            for name in names:
                total[name].extend(found[name])
            for body, index in hub_of.items():
                body_hub[body] = index
            for body, island in island_of.items():
                body_island[body] = island
            for body in seg_bodies:
                body_is_segment[body] = 1

        if island_count == 0:
            return None, offsets
        route = ChainCondensation()
        route.device = device
        route.island_count = island_count
        route.chain_count = chain_count
        route.segment_count = len(total["segment_body"])
        route.hub_count = len(total["hub_body"])
        route.row_count = row_count
        route.row_joint_host = np.asarray(row_joint, dtype=np.int32)
        route.hub_body_host = np.asarray(total["hub_body"], dtype=np.int32)
        route.segment_body_host = np.asarray(total["segment_body"], dtype=np.int32)
        route.strides = []
        stride = 1
        while stride < max(total["segment_chain_length"]):
            route.strides.append(stride)
            stride *= 2

        def ints(values):
            return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

        for name in names:
            setattr(route, name, ints(total[name]))
        route.segment_link_start = ints(link_start)
        route.island_segment_start = ints(island_segment_start)
        route.island_hub_start = ints(island_hub_start)
        route.island_row_start = ints(island_row_start)
        route.body_hub = ints(body_hub)
        route.body_island = ints(body_island)
        route.body_is_segment = ints(body_is_segment)
        route.segment_kirchhoff_proximal = wp.zeros(route.segment_count, dtype=wp.float64, device=device)

        route.jacobian_parent = wp.zeros(row_count, dtype=wp.spatial_matrix, device=device)
        route.jacobian_child = wp.zeros(row_count, dtype=wp.spatial_matrix, device=device)
        route.compliance = wp.zeros(row_count, dtype=wp.spatial_matrix, device=device)
        route.residual = wp.zeros(row_count, dtype=wp.spatial_vector, device=device)
        route.row_active = wp.zeros(row_count, dtype=wp.int32, device=device)
        count = route.segment_count
        route.segment_diagonal = wp.zeros(count, dtype=wp.spatial_matrixd, device=device)
        route.segment_left = wp.zeros(count, dtype=wp.spatial_matrixd, device=device)
        route.segment_right = wp.zeros(count, dtype=wp.spatial_matrixd, device=device)
        route.segment_inverse = wp.zeros(count, dtype=wp.spatial_matrixd, device=device)
        route.segment_rhs = wp.zeros((count, _COLUMNS), dtype=wp.spatial_vectord, device=device)
        route.segment_solution = wp.zeros((count, _COLUMNS), dtype=wp.spatial_vectord, device=device)
        route.segment_border = wp.zeros((count, HUB_MAX), dtype=wp.spatial_matrixd, device=device)
        route.segment_hub_diagonal = wp.zeros((count, HUB_MAX), dtype=wp.spatial_matrixd, device=device)
        route.segment_hub_wrench = wp.zeros((count, HUB_MAX), dtype=wp.spatial_vectord, device=device)
        route.segment_mask = wp.zeros(count, dtype=wp.int32, device=device)
        route.schur = wp.zeros((island_count, _HUB_DOF, _HUB_DOF), dtype=wp.float64, device=device)
        route.schur_rhs = wp.zeros((island_count, _HUB_DOF), dtype=wp.float64, device=device)
        route.hub_block = wp.zeros((island_count, HUB_MAX, HUB_MAX), dtype=wp.spatial_matrixd, device=device)
        route.hub_vector = wp.zeros((island_count, HUB_MAX), dtype=wp.spatial_vectord, device=device)
        route.hub_solution = wp.zeros((island_count, _HUB_DOF), dtype=wp.float64, device=device)
        route._hub_base_debug = None
        route._hub_rhs_debug = None
        return route, offsets

    def set_segment_kirchhoff_proximal(self, kirchhoff_by_body: dict[int, float]) -> None:
        """Populate per-segment Kirchhoff director proximal stiffness ``0.25 * L^2 * K_shear`` in ``float64``."""
        values = np.array(
            [float(kirchhoff_by_body.get(int(body), 0.0)) for body in self.segment_body_host],
            dtype=np.float64,
        )
        self.segment_kirchhoff_proximal.assign(values)

    def rows(self, start, count):
        """Views of ``count`` joint rows from ``start`` for unified bucket linearization."""
        end = start + count
        return (
            self.jacobian_parent[start:end],
            self.jacobian_child[start:end],
            self.compliance[start:end],
            self.residual[start:end],
            self.row_active[start:end],
        )

    def solve(
        self,
        *,
        body_slot_by_id,
        graph_body_island,
        island_contact_state,
        include_stress,
        stress_offsets,
        body_stress,
        body_correction,
        dt,
        body_q,
        body_inertia_q,
        body_disp,
        body_disp_ang,
        use_displacement,
        body_q_prev,
        body_mass,
        body_inv_mass,
        body_inertia,
        body_com,
        contacts,
        shape_body,
        body_contact_buffer_size,
        body_contact_counts,
        body_contact_indices,
        contact_damping_bidirectional,
        contact_normal_rho,
        contact_material_ke,
        contact_material_kd,
        contact_material_mu,
        contact_tangent_rho,
        contact_torsional_rho,
        contact_rolling_rho,
        contact_lambda,
        contact_lambda_angular,
        contact_C0,
        contact_stab_alpha,
    ):
        """Solve all chain-condensation islands in ``float64`` without global atomics."""
        has_contacts = int(contacts is not None and contacts.rigid_contact_max > 0 and contact_normal_rho is not None)
        device = self.device
        keep_debug = int(bool(self.keep_assembly))
        if keep_debug and self._hub_base_debug is None:
            self._hub_base_debug = wp.zeros((self.island_count, HUB_MAX), dtype=wp.spatial_matrixd, device=device)
            self._hub_rhs_debug = wp.zeros((self.island_count, HUB_MAX), dtype=wp.spatial_vectord, device=device)

        wp.launch(
            assemble_chain_segments,
            self.segment_count,
            inputs=[
                self.segment_body,
                self.segment_island,
                self.segment_index,
                self.segment_chain_length,
                self.segment_columns,
                self.segment_previous_row,
                self.segment_previous_side,
                self.segment_next_row,
                self.segment_next_side,
                self.segment_link_start,
                self.link_row,
                self.link_side,
                self.link_hub,
                self.body_hub,
                self.body_island,
                body_slot_by_id,
                graph_body_island,
                island_contact_state,
                self.segment_kirchhoff_proximal,
                int(include_stress),
                stress_offsets,
                body_stress,
                self.jacobian_parent,
                self.jacobian_child,
                self.compliance,
                self.residual,
                self.row_active,
                has_contacts,
                float(OPEN_CONTACT_WEIGHT),
                dt,
                body_q,
                body_inertia_q,
                body_disp,
                body_disp_ang,
                int(use_displacement),
                body_q_prev,
                body_mass,
                body_inv_mass,
                body_inertia,
                body_com,
                bool(contact_damping_bidirectional),
                contact_normal_rho if has_contacts else None,
                contact_material_ke if has_contacts else None,
                contact_material_kd if has_contacts else None,
                contact_material_mu if has_contacts else None,
                contact_tangent_rho if has_contacts else None,
                contact_torsional_rho if has_contacts else None,
                contact_rolling_rho if has_contacts else None,
                contact_lambda if has_contacts else None,
                contact_lambda_angular if has_contacts else None,
                contact_C0 if has_contacts else None,
                float(contact_stab_alpha),
                contacts.rigid_contact_count if has_contacts else None,
                contacts.rigid_contact_shape0 if has_contacts else None,
                contacts.rigid_contact_shape1 if has_contacts else None,
                contacts.rigid_contact_point0 if has_contacts else None,
                contacts.rigid_contact_point1 if has_contacts else None,
                contacts.rigid_contact_surface_velocity if has_contacts else None,
                contacts.rigid_contact_offset0 if has_contacts else None,
                contacts.rigid_contact_offset1 if has_contacts else None,
                contacts.rigid_contact_normal if has_contacts else None,
                contacts.rigid_contact_margin0 if has_contacts else None,
                contacts.rigid_contact_margin1 if has_contacts else None,
                shape_body,
                body_contact_buffer_size,
                body_contact_counts,
                body_contact_indices,
            ],
            outputs=[
                self.segment_diagonal,
                self.segment_left,
                self.segment_right,
                self.segment_inverse,
                self.segment_rhs,
                self.segment_border,
                self.segment_hub_diagonal,
                self.segment_hub_wrench,
                self.segment_mask,
            ],
            device=device,
        )
        if keep_debug:
            self.assembly = {
                "diagonal": self.segment_diagonal.numpy().copy(),
                "left": self.segment_left.numpy().copy(),
                "right": self.segment_right.numpy().copy(),
                "rhs": self.segment_rhs.numpy()[:, 0].copy(),
                "border": self.segment_border.numpy().copy(),
                "hub_diagonal": self.segment_hub_diagonal.numpy().copy(),
                "hub_wrench": self.segment_hub_wrench.numpy().copy(),
                "mask": self.segment_mask.numpy().copy(),
            }
        for stride in self.strides:
            wp.launch(
                reduce_chains,
                (self.segment_count, _COLUMNS + 1),
                inputs=[stride, self.segment_index, self.segment_chain_length, self.segment_columns],
                outputs=[
                    self.segment_diagonal,
                    self.segment_left,
                    self.segment_right,
                    self.segment_inverse,
                    self.segment_rhs,
                ],
                device=device,
            )
        for stride in reversed(self.strides or [1]):
            wp.launch(
                back_substitute_chains,
                (self.segment_count, _COLUMNS),
                inputs=[
                    stride,
                    self.segment_index,
                    self.segment_chain_length,
                    self.segment_chain_top,
                    self.segment_columns,
                    self.segment_left,
                    self.segment_right,
                    self.segment_inverse,
                    self.segment_rhs,
                ],
                outputs=[self.segment_solution],
                device=device,
            )
        wp.launch(
            accumulate_chain_schur,
            (self.island_count, _HUB_DOF, _COLUMNS),
            inputs=[
                self.island_segment_start,
                self.island_hub_start,
                self.segment_mask,
                self.segment_border,
                self.segment_hub_diagonal,
                self.segment_hub_wrench,
                self.segment_solution,
            ],
            outputs=[self.schur, self.schur_rhs],
            device=device,
        )
        wp.launch(
            solve_chain_hubs,
            self.island_count,
            inputs=[
                self.island_hub_start,
                self.island_row_start,
                self.hub_body,
                self.hub_row,
                self.hub_row_parent,
                self.hub_row_child,
                self.body_island,
                self.body_is_segment,
                body_slot_by_id,
                int(include_stress),
                stress_offsets,
                body_stress,
                self.jacobian_parent,
                self.jacobian_child,
                self.compliance,
                self.residual,
                self.row_active,
                has_contacts,
                dt,
                body_q,
                body_inertia_q,
                body_disp,
                body_disp_ang,
                int(use_displacement),
                body_q_prev,
                body_mass,
                body_inv_mass,
                body_inertia,
                body_com,
                bool(contact_damping_bidirectional),
                contact_normal_rho if has_contacts else None,
                contact_material_ke if has_contacts else None,
                contact_material_kd if has_contacts else None,
                contact_material_mu if has_contacts else None,
                contact_tangent_rho if has_contacts else None,
                contact_torsional_rho if has_contacts else None,
                contact_rolling_rho if has_contacts else None,
                contact_lambda if has_contacts else None,
                contact_lambda_angular if has_contacts else None,
                contact_C0 if has_contacts else None,
                float(contact_stab_alpha),
                contacts.rigid_contact_count if has_contacts else None,
                contacts.rigid_contact_shape0 if has_contacts else None,
                contacts.rigid_contact_shape1 if has_contacts else None,
                contacts.rigid_contact_point0 if has_contacts else None,
                contacts.rigid_contact_point1 if has_contacts else None,
                contacts.rigid_contact_surface_velocity if has_contacts else None,
                contacts.rigid_contact_offset0 if has_contacts else None,
                contacts.rigid_contact_offset1 if has_contacts else None,
                contacts.rigid_contact_normal if has_contacts else None,
                contacts.rigid_contact_margin0 if has_contacts else None,
                contacts.rigid_contact_margin1 if has_contacts else None,
                shape_body,
                body_contact_buffer_size,
                body_contact_counts,
                body_contact_indices,
                self.schur,
                self.schur_rhs,
                keep_debug,
            ],
            outputs=[
                self._hub_base_debug,
                self._hub_rhs_debug,
                self.hub_block,
                self.hub_vector,
                self.hub_solution,
                body_correction,
            ],
            device=device,
        )
        wp.launch(
            correct_chain_segments,
            self.segment_count,
            inputs=[
                self.segment_body,
                self.segment_island,
                self.segment_columns,
                body_slot_by_id,
                self.segment_solution,
                self.hub_solution,
            ],
            outputs=[body_correction],
            device=device,
        )
        if keep_debug:
            self.assembly["hub_base"] = self._hub_base_debug.numpy().copy()
            self.assembly["hub_rhs"] = self._hub_rhs_debug.numpy().copy()
            self.assembly["correction"] = body_correction.numpy().copy()
