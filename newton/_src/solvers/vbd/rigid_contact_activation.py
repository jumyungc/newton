# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional, force-free normal metrics for contacts crossed by an actual body trial.

The trial includes joints and other forces, unlike the inertial target alone.
This is a finite-budget stabilization, not a nonlinear convergence guarantee.
"""

import functools

import warp as wp

from .rigid_vbd_kernels import (
    _SMALL_LENGTH_EPS,
    _body_step_motion,
    _compliant_contact_normal_response,
    _contact_lever_blocks,
    _contact_step_levers,
    _lever_displacement,
    _material_force_terms,
    _step_start_penetration,
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
        body_inv_mass: wp.array[float],
        body_colors: wp.array[int],
        contact_rho: wp.array[float],
        contact_ke: wp.array[float],
        contact_kd: wp.array[float],
        contact_lambda: wp.array[wp.vec3],
        contact_C0: wp.array[wp.vec3],
        alpha: float,
        contact_damping_bidirectional: bool,
        contact_count: wp.array[int],
        shape0: wp.array[int],
        shape1: wp.array[int],
        point0: wp.array[wp.vec3],
        point1: wp.array[wp.vec3],
        normal: wp.array[wp.vec3],
        margin0: wp.array[float],
        margin1: wp.array[float],
        shape_body: wp.array[int],
        capacity: int,
        body_contact_counts: wp.array[int],
        body_contact_indices: wp.array[int],
        body_h_ll: wp.array[wp.mat33],
        body_h_al: wp.array[wp.mat33],
        body_h_aa: wp.array[wp.mat33],
        body_activated: wp.array[wp.int32],
    ):
        tid = wp.tid()
        body = body_ids[tid // wp.static(threads_per_body)]
        lane = tid % wp.static(threads_per_body)
        count = wp.min(body_contact_counts[body], capacity)
        if body_inv_mass[body] == 0.0 or lane >= count:
            return

        X_prev, com_disp, rotation, jacobian_t = _body_step_motion(body, body_q, body_q_prev, body_com)
        _X_trial, com_disp_trial, rotation_trial, _J_trial = _body_step_motion(
            body, body_q_trial, body_q_prev, body_com
        )
        h_ll = wp.mat33(0.0)
        h_al = wp.mat33(0.0)
        h_aa = wp.mat33(0.0)
        has_inactive = int(0)
        i = lane
        while i < count:
            row = body_contact_indices[body * capacity + i]
            i += wp.static(threads_per_body)
            if row >= contact_count[0]:
                continue
            s0 = shape0[row]
            s1 = shape1[row]
            if s0 < 0 or s1 < 0:
                continue
            b0 = shape_body[s0]
            b1 = shape_body[s1]
            if b0 != body and b1 != body:
                continue
            self_is_a = b0 == body
            other = b1 if self_is_a else b0
            X_other_prev, com_disp_other, rotation_other, _J_other = _body_step_motion(
                other, body_q, body_q_prev, body_com
            )
            pt_a = point0[row]
            pt_b = point1[row]
            lever_self, _ = _contact_step_levers(body, X_prev, body_com, pt_a if self_is_a else pt_b, wp.vec3(0.0))
            lever_other, _ = _contact_step_levers(
                other, X_other_prev, body_com, pt_b if self_is_a else pt_a, wp.vec3(0.0)
            )
            dx = _lever_displacement(com_disp, rotation, lever_self)
            dy = _lever_displacement(com_disp_other, rotation_other, lever_other)
            n = normal[row]
            Cp = _step_start_penetration(
                b0,
                X_prev if self_is_a else X_other_prev,
                pt_a,
                b1,
                X_other_prev if self_is_a else X_prev,
                pt_b,
                n,
                margin0[row],
                margin1[row],
            )
            C = Cp + wp.dot(n, (dx - dy) if self_is_a else (dy - dx))
            lam = wp.dot(contact_lambda[row], n)
            shift = alpha * wp.dot(contact_C0[row], n)
            k, lam_eff = _material_force_terms(contact_rho[row], contact_ke[row], lam)
            # Mirror the existing compliant normal-force activation, including positive warm multipliers.
            if not ((C <= _SMALL_LENGTH_EPS and lam <= 0.0) or k * (C - shift) + lam_eff <= 0.0):
                continue
            if k <= 0.0:
                continue

            _X_other_trial, ct_disp_other, rt_other, _Jt_other = _body_step_motion(
                other, body_q_trial, body_q_prev, body_com
            )
            dxt = _lever_displacement(com_disp_trial, rotation_trial, lever_self)
            dyt = _lever_displacement(ct_disp_other, rt_other, lever_other)
            Ct = Cp + wp.dot(n, (dxt - dyt) if self_is_a else (dyt - dxt))
            if (Ct <= _SMALL_LENGTH_EPS and lam <= 0.0) or k * (Ct - shift) + lam_eff <= 0.0:
                continue
            force = k * (Ct - shift) + lam_eff
            metric = k
            if contact_damping_bidirectional:
                # The explicit velocity damper does not enter the elastic ALM multiplier.
                # Its metric is active even at zero normal velocity.
                metric += wp.max(contact_kd[row], 0.0) / dt
            else:
                force, _elastic, metric = _compliant_contact_normal_response(
                    Ct - shift, Cp - shift, contact_ke[row], contact_kd[row], contact_rho[row], lam, dt
                )
            if force <= 0.0 or metric <= 0.0:
                continue
            if b0 >= 0 and b1 >= 0:
                if body_inv_mass[b0] > 0.0 and body_inv_mass[b1] > 0.0 and body_colors[b0] == body_colors[b1]:
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
