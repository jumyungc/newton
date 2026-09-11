# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Projected Coulomb channels with metrics from directional compliance."""

import warp as wp


@wp.func
def _friction_disk_multiplier_double(d: wp.vec2d, z: wp.vec2d):
    """Keep the secular equation representable outside float32's exponent range."""
    delta = wp.vec2d(wp.abs(z[0]) - d[0], wp.abs(z[1]) - d[1])
    lo = wp.max(wp.float64(0.0), wp.max(delta[0], delta[1]))
    hi = wp.max(wp.float64(0.0), wp.length(z) - d[0])
    eta = lo
    for _ in range(512):
        if hi - lo <= wp.float64(2.220446049250313e-16) * (hi + d[0]):
            eta = wp.float64(0.5) * (lo + hi)
            break
        den = d + wp.vec2d(eta)
        y = wp.vec2d(z[0] / den[0], z[1] / den[1])
        residual = wp.float64(0.0)
        if wp.abs(y[0]) >= wp.abs(y[1]):
            residual = ((delta[0] - eta) / den[0]) * ((wp.abs(z[0]) + d[0] + eta) / den[0]) + y[1] * y[1]
        else:
            residual = ((delta[1] - eta) / den[1]) * ((wp.abs(z[1]) + d[1] + eta) / den[1]) + y[0] * y[0]
        if residual >= wp.float64(0.0):
            lo = eta
        else:
            hi = eta
        derivative = y[0] * y[0] / den[0] + y[1] * y[1] / den[1]
        norm = wp.sqrt(wp.max(wp.float64(0.0), wp.float64(1.0) + residual))
        step = residual * norm * norm / ((norm + wp.float64(1.0)) * derivative)
        proposal = eta + step
        if wp.abs(step) <= wp.float64(2.220446049250313e-16) * (eta + d[0]) or proposal == eta:
            break
        if proposal < lo or proposal > hi:
            proposal = wp.float64(0.5) * (lo + hi)
        if proposal == eta:
            break
        eta = proposal
    return eta


@wp.func
def _friction_disk_multiplier(d: wp.vec2d, z: wp.vec2d):
    """Solve the disk secular equation with a cancellation-safe residual."""
    # Float32's smallest normal value bounds the reciprocal and derivative.
    if d[0] < wp.float64(1.1754943508222875e-38):
        return _friction_disk_multiplier_double(d, z)
    df = wp.vec2(d)
    zf = wp.vec2(z)
    delta = wp.vec2d(wp.abs(z[0]) - d[0], wp.abs(z[1]) - d[1])
    lo = wp.max(wp.float64(0.0), wp.max(delta[0], delta[1]))
    hi = wp.max(wp.float64(0.0), wp.length(z) - d[0])
    eta = wp.float32(lo)
    for _ in range(64):
        if hi - lo <= wp.float64(1.1920928955078125e-7) * (hi + d[0]):
            eta = wp.float32(wp.float64(0.5) * (lo + hi))
            break
        den = df + wp.vec2(eta)
        y = wp.vec2(zf[0] / den[0], zf[1] / den[1])
        residual_d = wp.float64(0.0)
        # Retain the difference near the sphere in double precision. The
        # remaining products and Newton derivative need float32 accuracy.
        if wp.abs(y[0]) >= wp.abs(y[1]):
            factor = ((wp.abs(zf[0]) + df[0] + eta) / den[0]) / den[0]
            residual_d = (delta[0] - wp.float64(eta)) * wp.float64(factor) + wp.float64(y[1] * y[1])
        else:
            factor = ((wp.abs(zf[1]) + df[1] + eta) / den[1]) / den[1]
            residual_d = (delta[1] - wp.float64(eta)) * wp.float64(factor) + wp.float64(y[0] * y[0])
        if residual_d >= wp.float64(0.0):
            lo = wp.float64(eta)
        else:
            hi = wp.float64(eta)
        derivative = y[0] * y[0] / den[0] + y[1] * y[1] / den[1]
        residual = wp.float32(residual_d)
        norm = wp.sqrt(wp.max(0.0, 1.0 + residual))
        step = residual * norm * norm / ((norm + 1.0) * derivative)
        proposal = eta + step
        if wp.abs(step) <= 1.1920928955078125e-7 * (eta + df[0]) or proposal == eta:
            break
        if wp.float64(proposal) < lo or wp.float64(proposal) > hi:
            proposal = wp.float32(wp.float64(0.5) * (lo + hi))
        if proposal == eta:
            break
        eta = proposal
    return wp.float64(eta)


@wp.func
def _project_friction_interval(displacement: float, lambda_old: float, mobility: float, bound: float):
    """Return the interval force and an upper bound on its displacement derivative."""
    force = float(0.0)
    metric = float(0.0)
    if bound > 0.0 and mobility > 0.0:
        rhs = wp.float64(displacement) + wp.float64(mobility) * wp.float64(lambda_old)
        if wp.abs(rhs) <= wp.float64(mobility) * wp.float64(bound):
            force = wp.float32(rhs / wp.float64(mobility))
            metric = 1.0 / mobility
        else:
            force = bound
            if rhs < wp.float64(0.0):
                force = -bound
            metric = wp.float32(wp.float64(bound) / wp.abs(rhs))
    return force, metric


@wp.func
def _project_friction_disk(displacement: wp.vec2, lambda_old: wp.vec2, mobility: wp.mat22, bound: float):
    """Return the metric projection onto a Coulomb disk and its primal metric.

    With ``W = mobility`` and ``b = displacement + W * lambda_old``, solve
    ``min |lambda| <= bound: lambda.T * W * lambda / 2 - b.T * lambda``.
    The returned metric is ``Q = (W + eta I)^-1``, where eta is the disk's
    nonnegative KKT multiplier. It equals the exact derivative in stick and
    bounds the positive-semidefinite derivative from above in slip.

    Mobility must be positive definite for an active channel. Zero mobility
    disables the channel. Double intermediates preserve the determinant and
    rotated force components when the supplied float32 matrix is anisotropic;
    they cannot recover definiteness already lost when that matrix was built.
    """
    force = wp.vec2(0.0)
    metric = wp.mat22(0.0)
    if bound > 0.0:
        # Diagonal stick and isotropic slip have exact elementary formulas.
        if mobility[0, 1] == 0.0 and mobility[0, 0] > 0.0 and mobility[1, 1] > 0.0:
            trial = lambda_old + wp.vec2(displacement[0] / mobility[0, 0], displacement[1] / mobility[1, 1])
            trial_norm = wp.length(trial / bound)
            # A reduced normal load can leave history outside the new disk;
            # cancellation then needs double precision relative to that bound.
            history_inside = wp.length(lambda_old / bound) <= 1.0
            if trial_norm <= 1.0 and history_inside:
                return trial, wp.mat22(1.0 / mobility[0, 0], 0.0, 0.0, 1.0 / mobility[1, 1])
            active_axis = int(-1)
            if displacement[0] == 0.0 and lambda_old[0] == 0.0:
                active_axis = 1
            elif displacement[1] == 0.0 and lambda_old[1] == 0.0:
                active_axis = 0
            if active_axis >= 0:
                # A genuinely inactive axis reduces this disk to an interval.
                active_mobility = wp.float64(mobility[active_axis, active_axis])
                active_trial = (
                    wp.float64(lambda_old[active_axis]) + wp.float64(displacement[active_axis]) / active_mobility
                )
                active_eta = wp.max(
                    wp.float64(0.0), (wp.abs(active_trial) / wp.float64(bound) - wp.float64(1.0)) * active_mobility
                )
                force[active_axis] = wp.float32(wp.clamp(active_trial, -wp.float64(bound), wp.float64(bound)))
                active_q0 = wp.float32(wp.float64(1.0) / (wp.float64(mobility[0, 0]) + active_eta))
                active_q1 = wp.float32(wp.float64(1.0) / (wp.float64(mobility[1, 1]) + active_eta))
                return force, wp.mat22(active_q0, 0.0, 0.0, active_q1)
            if mobility[0, 0] == mobility[1, 1] and wp.isfinite(trial_norm) and history_inside:
                ratio = 1.0 / trial_norm
                return ratio * trial, wp.mat22(ratio / mobility[0, 0], 0.0, 0.0, ratio / mobility[1, 1])
        w = wp.mat22d(mobility)
        w_scale = wp.max(wp.max(wp.abs(w[0, 0]), wp.abs(w[1, 1])), wp.abs(w[0, 1]))
        if w_scale > wp.float64(0.0):
            inverse_w_scale = wp.float64(1.0) / w_scale
            a = w[0, 0] * inverse_w_scale
            b = w[0, 1] * inverse_w_scale
            c = w[1, 1] * inverse_w_scale
            determinant = a * c - b * b
            if a > wp.float64(0.0) and determinant > wp.float64(0.0):
                inverse = wp.mat22d(c, -b, -b, a) * (inverse_w_scale / determinant)
                trial_d = wp.vec2d(lambda_old) + inverse * wp.vec2d(displacement)
                if wp.length(trial_d) <= wp.float64(bound):
                    return wp.vec2(trial_d), wp.mat22(inverse)
                discriminant = wp.sqrt((a - c) * (a - c) + wp.float64(4.0) * b * b)
                high = wp.float64(0.5) * (a + c + discriminant)
                low = determinant / high
                v = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
                if b != wp.float64(0.0):
                    if a >= c:
                        v = wp.normalize(wp.vec2d(high - c, b))
                    else:
                        v = wp.normalize(wp.vec2d(b, high - a))
                elif c > a:
                    v = wp.vec2d(wp.float64(0.0), wp.float64(1.0))
                u = wp.vec2d(-v[1], v[0])
                rhs = wp.vec2d(displacement) * inverse_w_scale + (w * inverse_w_scale) * wp.vec2d(lambda_old)
                z = wp.vec2d(wp.dot(u, rhs), wp.dot(v, rhs)) * (wp.float64(1.0) / wp.float64(bound))
                scale = wp.max(high, wp.length(z))
                inverse_scale = wp.float64(1.0) / scale
                d = wp.vec2d(low, high) * inverse_scale
                z = z * inverse_scale
                eta = wp.float64(0.0)
                if high == low:
                    eta = wp.max(wp.float64(0.0), wp.length(z) - d[0])
                else:
                    eta = _friction_disk_multiplier(d, z)
                inverse_den_0 = wp.float64(1.0) / (d[0] + eta)
                inverse_den_1 = wp.float64(1.0) / (d[1] + eta)
                q0 = inverse_den_0 * inverse_scale * inverse_w_scale
                q1 = inverse_den_1 * inverse_scale * inverse_w_scale
                y = wp.vec2d(z[0] * inverse_den_0, z[1] * inverse_den_1)
                force = wp.vec2(wp.float64(bound) * (y[0] * u + y[1] * v))
                metric = wp.mat22(q0 * wp.outer(u, u) + q1 * wp.outer(v, v))
    return force, metric
