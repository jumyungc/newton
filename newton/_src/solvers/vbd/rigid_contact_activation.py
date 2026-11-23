# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional, force-free normal metrics for contacts crossed by an actual body trial.

The trial includes joints and other forces, unlike the inertial target alone.
This is a finite-budget stabilization, not a nonlinear convergence guarantee.
"""

import functools

import warp as wp

from ...geometry.sdf_planar_recovery import MeshRecoveryData
from .rigid_vbd_kernels import (
    _SMALL_LENGTH_EPS,
    _body_step_motion,
    _contact_lever_blocks,
    _first_order_lever_displacement,
    _normal_material,
    _recovery_slot,
    _rigid_contact_normal_law,
)


@functools.cache
def create_accumulate_inactive_contact_metric(threads_per_body: int):
    """Build a strided gather which adds only normal Hessian blocks, never forces.

    A body whose trial crosses any inactive row is flagged in ``body_activated``.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def accumulate_inactive_contact_metric(
        dt: float,
        body_ids: wp.array[int],
        body_q_prev: wp.array[wp.transform],
        body_q: wp.array[wp.transform],
        body_q_trial: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        contact_rho: wp.array[float],
        contact_ke: wp.array[float],
        contact_kd: wp.array[float],
        contact_lambda: wp.array[wp.vec3],
        contact_C0: wp.array[wp.vec3],
        recovery: MeshRecoveryData,
        contact_generation: wp.array[int],
        contact_damping_bidirectional: bool,
        contact_count: wp.array[int],
        shape0: wp.array[int],
        shape1: wp.array[int],
        normal: wp.array[wp.vec3],
        contact_lever_s0: wp.array[wp.vec3],
        contact_lever_s1: wp.array[wp.vec3],
        contact_start_penetration: wp.array[float],
        capacity: int,
        body_contact_counts: wp.array[int],
        body_contact_indices: wp.array[int],
        body_contact_bodies: wp.array[wp.vec3i],
        body_h_ll: wp.array[wp.mat33],
        body_h_al: wp.array[wp.mat33],
        body_h_aa: wp.array[wp.mat33],
        body_activated: wp.array[wp.int32],
    ):
        tid = wp.tid()
        body = body_ids[tid // wp.static(threads_per_body)]
        lane = tid % wp.static(threads_per_body)
        # Contacts are listed only under dynamic bodies: an immovable body has a zero count.
        count = wp.min(body_contact_counts[body], capacity)
        if lane >= count:
            return

        translation, rotation, jacobian_t = _body_step_motion(body, body_q, body_q_prev, body_com)
        translation_trial, rotation_trial, _jacobian_t_trial = _body_step_motion(
            body, body_q_trial, body_q_prev, body_com
        )
        h_ll = wp.mat33(0.0)
        h_al = wp.mat33(0.0)
        h_aa = wp.mat33(0.0)
        has_inactive = int(0)
        i = lane
        while i < count:
            # The list entry carries the contact's two bodies and the same-color flag.
            slot = body * capacity + i
            row = body_contact_indices[slot]
            bodies = body_contact_bodies[slot]
            i += wp.static(threads_per_body)
            if row >= contact_count[0]:
                continue
            s0 = shape0[row]
            s1 = shape1[row]
            b0 = bodies[0]
            b1 = bodies[1]
            n = normal[row]
            rec_slot = _recovery_slot(recovery, contact_generation, s0, s1, b0, b1, body_q, n)
            if (b0 != body and b1 != body) or rec_slot < -1:
                continue
            self_is_a = b0 == body
            other = b1 if self_is_a else b0
            lever_self = contact_lever_s0[row] if self_is_a else contact_lever_s1[row]
            lever_other = contact_lever_s1[row] if self_is_a else contact_lever_s0[row]
            # Penetration grows with the approach of endpoint 0 toward endpoint 1 along the normal.
            approach_sign = 1.0 if self_is_a else -1.0
            translation_other, rotation_other, _jacobian_t_other = _body_step_motion(
                other, body_q, body_q_prev, body_com
            )
            start_penetration = contact_start_penetration[row]
            C = start_penetration + approach_sign * wp.dot(
                n,
                _first_order_lever_displacement(translation, rotation, lever_self)
                - _first_order_lever_displacement(translation_other, rotation_other, lever_other),
            )
            lam_vec = contact_lambda[row]
            lam = wp.dot(lam_vec, n)
            shift = wp.dot(contact_C0[row], n)
            kd = contact_kd[row]
            previous_penetration = float(0.0)
            if kd > 0.0:
                previous_penetration = start_penetration - shift
            ke_now = _normal_material(contact_ke[row], rec_slot)

            # Rows the body solve already carries need no guard: mirror its activity rule and normal law.
            if not (C <= _SMALL_LENGTH_EPS and lam <= 0.0):
                _metric_now, _elastic_now, force_now, _lam_n_now, _lam_eff_now = _rigid_contact_normal_law(
                    C - shift,
                    previous_penetration,
                    contact_rho[row],
                    ke_now,
                    kd,
                    lam_vec,
                    n,
                    dt,
                    contact_damping_bidirectional,
                )
                if force_now > 0.0:
                    continue

            rec_slot_trial = _recovery_slot(recovery, contact_generation, s0, s1, b0, b1, body_q_trial, n)
            if rec_slot_trial < -1:
                continue
            ke_trial = _normal_material(contact_ke[row], rec_slot_trial)
            translation_other_trial, rotation_other_trial, _jacobian_t_other_trial = _body_step_motion(
                other, body_q_trial, body_q_prev, body_com
            )
            C_trial = start_penetration + approach_sign * wp.dot(
                n,
                _first_order_lever_displacement(translation_trial, rotation_trial, lever_self)
                - _first_order_lever_displacement(translation_other_trial, rotation_other_trial, lever_other),
            )
            if C_trial <= _SMALL_LENGTH_EPS and lam <= 0.0:
                continue
            metric, _elastic, force, _lam_n, _lam_eff = _rigid_contact_normal_law(
                C_trial - shift,
                previous_penetration,
                contact_rho[row],
                ke_trial,
                kd,
                lam_vec,
                n,
                dt,
                contact_damping_bidirectional,
            )
            if force <= 0.0 or metric <= 0.0:
                continue
            if contact_damping_bidirectional:
                # The explicit velocity damper does not enter the elastic ALM multiplier.
                # Its metric is active even at zero normal velocity.
                metric += wp.max(kd, 0.0) / dt
            if bodies[2] == 1:
                # Same bound as the body solve for two dynamic bodies of one color.
                metric *= 2.0
            K = metric * wp.outer(n, n)
            _torque, H_al, H_aa = _contact_lever_blocks(
                wp.vec3(0.0), wp.vec3(0.0), K, wp.mat33(0.0), lever_self, wp.vec3(0.0), jacobian_t
            )
            h_ll += K
            h_al += H_al
            h_aa += H_aa
            has_inactive = 1

        if has_inactive != 0:
            wp.atomic_add(body_h_ll, body, h_ll)
            wp.atomic_add(body_h_al, body, h_al)
            wp.atomic_add(body_h_aa, body, h_aa)
            body_activated[body] = 1

    return accumulate_inactive_contact_metric
