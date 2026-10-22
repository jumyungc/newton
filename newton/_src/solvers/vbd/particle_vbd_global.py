# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal local-global tetrahedral particle accelerator for VBD.

Uses the backward-Euler stable Neo-Hookean elastic and Green--Saint-Venant
strain-rate damping forces of local particle VBD in a Newton--Krylov correction.
Scope: active/inactive tetrahedral particles and frictionless, undamped contact
with static rigid shapes in a single world, excluding feature-filtered mesh
contacts. This experimental primal displacement solve is not the rigid-joint KKT
system. A finite iteration budget does not guarantee nonlinear convergence.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, cg

wp.set_module_options({"enable_backward": False})


@wp.func
def grad(B: wp.mat33, k: int):
    g = wp.vec3(B[0, 0], B[0, 1], B[0, 2])
    if k == 0:
        g = -wp.vec3(B[0, 0] + B[1, 0] + B[2, 0], B[0, 1] + B[1, 1] + B[2, 1], B[0, 2] + B[1, 2] + B[2, 2])
    elif k == 2:
        g = wp.vec3(B[1, 0], B[1, 1], B[1, 2])
    elif k == 3:
        g = wp.vec3(B[2, 0], B[2, 1], B[2, 2])
    return g


@wp.func
def cof(F: wp.mat33):
    return wp.matrix_from_cols(wp.cross(F[:, 1], F[:, 2]), wp.cross(F[:, 2], F[:, 0]), wp.cross(F[:, 0], F[:, 1]))


@wp.func
def dcof(F: wp.mat33, D: wp.mat33):
    return wp.matrix_from_cols(
        wp.cross(D[:, 1], F[:, 2]) + wp.cross(F[:, 1], D[:, 2]),
        wp.cross(D[:, 2], F[:, 0]) + wp.cross(F[:, 2], D[:, 0]),
        wp.cross(D[:, 0], F[:, 1]) + wp.cross(F[:, 0], D[:, 1]),
    )


@wp.func
def tet_tangent(
    F: wp.mat33,
    C: wp.mat33,
    E: wp.mat33,
    D: wp.mat33,
    mu: float,
    lam: float,
    kd: float,
    h: float,
    full: int,
):
    """Apply the exact elastic and Green--Saint-Venant damping tangent (or its PSD GN part)."""
    Q = mu * D + lam * wp.ddot(C, D) * C
    if full != 0:
        Q += (lam * (wp.determinant(F) - 1.0) - mu) * dcof(F, D) + (2.0 * kd / h) * D * E
    Q += 2.0 * kd / h * F * (wp.transpose(D) * F + wp.transpose(F) * D)
    return Q


@wp.kernel
def constant_velocity_initial_iterate(
    previous: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    mass: wp.array[float],
    flags: wp.array[int],
    h: float,
    reference: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
):
    i = wp.tid()
    if flags[i] & 1 != 0 and mass[i] > 0.0:
        q[i] = previous[i] + h * velocity[i]
        displacement[i] = q[i] - reference[i]


@wp.kernel
def initialize(
    q: wp.array[wp.vec3],
    inertia: wp.array[wp.vec3],
    mass: wp.array[float],
    flags: wp.array[int],
    inv_h2: float,
    rhs: wp.array[wp.vec3],
    diag: wp.array[wp.mat33],
):
    i = wp.tid()
    rhs[i] = mass[i] * inv_h2 * (inertia[i] - q[i])
    diag[i] = mass[i] * inv_h2 * wp.identity(3, float)
    if flags[i] & 1 == 0 or mass[i] == 0.0:
        rhs[i] = wp.vec3(0.0)
        diag[i] = wp.identity(3, float)


@wp.kernel
def assemble_tets(
    q: wp.array[wp.vec3],
    old: wp.array[wp.vec3],
    ids: wp.array2d[int],
    B: wp.array[wp.mat33],
    mat: wp.array2d[float],
    flags: wp.array[int],
    mass: wp.array[float],
    h: float,
    rhs: wp.array[wp.vec3],
    diag: wp.array[wp.mat33],
    Fs: wp.array[wp.mat33],
    Cs: wp.array[wp.mat33],
    Es: wp.array[wp.mat33],
):
    t = wp.tid()
    a, b, c, d = ids[t, 0], ids[t, 1], ids[t, 2], ids[t, 3]
    F = wp.matrix_from_cols(q[b] - q[a], q[c] - q[a], q[d] - q[a]) * B[t]
    F0 = wp.matrix_from_cols(old[b] - old[a], old[c] - old[a], old[d] - old[a]) * B[t]
    C = cof(F)
    E = wp.transpose(F) * F - wp.transpose(F0) * F0
    mu = mat[t, 0]
    lam = mat[t, 1] + mu
    kd = mat[t, 2]
    V = 1.0 / (6.0 * wp.determinant(B[t]))
    P = mu * F + (lam * (wp.determinant(F) - 1.0) - mu) * C + 2.0 * kd / h * F * E
    Fs[t] = F
    Cs[t] = C
    Es[t] = E
    for k in range(4):
        i = ids[t, k]
        if flags[i] & 1 != 0 and mass[i] > 0.0:
            g = grad(B[t], k)
            cg = C * g
            fg = F * g
            H = mu * wp.dot(g, g) * wp.identity(3, float) + lam * wp.outer(cg, cg)
            H += 2.0 * kd / h * (wp.outer(fg, fg) + wp.dot(g, g) * F * wp.transpose(F))
            wp.atomic_add(rhs, i, -V * P * g)
            wp.atomic_add(diag, i, V * H)


@wp.func
def contact_point(q: wp.array[wp.vec3], ids: wp.vec3i, bary: wp.vec3):
    x = wp.vec3(0.0)
    for k in range(3):
        if ids[k] >= 0:
            x += bary[k] * q[ids[k]]
    return x


@wp.func
def contact_radius(radius: wp.array[float], ids: wp.vec3i):
    r = radius[ids[0]]
    for k in range(1, 3):
        if ids[k] >= 0:
            r = wp.max(r, radius[ids[k]])
    return r


@wp.kernel
def assemble_contacts(
    q: wp.array[wp.vec3],
    ids: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    count: wp.array[int],
    shapes: wp.array[int],
    normals: wp.array[wp.vec3],
    point: wp.array[wp.vec3],
    radius: wp.array[float],
    margin: wp.array[float],
    ke: wp.array[float],
    flags: wp.array[int],
    mass: wp.array[float],
    rhs: wp.array[wp.vec3],
    diag: wp.array[wp.mat33],
    active_k: wp.array[float],
):
    r = wp.tid()
    active_k[r] = 0.0
    if r < count[0]:
        n = normals[r]
        pen = (
            contact_radius(radius, ids[r]) + margin[shapes[r]] - wp.dot(n, contact_point(q, ids[r], bary[r]) - point[r])
        )
        if pen > 0.0:
            active_k[r] = ke[r]
            for k in range(3):
                i = ids[r][k]
                if i >= 0:
                    if flags[i] & 1 != 0 and mass[i] > 0.0:
                        w = bary[r][k]
                        wp.atomic_add(rhs, i, ke[r] * pen * w * n)
                        wp.atomic_add(diag, i, ke[r] * w * w * wp.outer(n, n))


@wp.kernel
def invert_blocks(diag: wp.array[wp.mat33], inverse: wp.array[wp.mat33]):
    i = wp.tid()
    value = wp.inverse(diag[i])
    valid = bool(True)
    for row in range(3):
        valid = valid and value[row, row] > 0.0
        for col in range(3):
            valid = valid and wp.isfinite(value[row, col])
    if not valid:
        # A representable SPD block can overflow/underflow the float32
        # determinant. Recover its scale without regularizing the operator;
        # preserve the original arithmetic whenever the inverse is usable.
        scale = wp.max(wp.abs(diag[i][0, 0]), wp.max(wp.abs(diag[i][1, 1]), wp.abs(diag[i][2, 2])))
        if scale > 0.0 and wp.isfinite(scale):
            value = wp.inverse(diag[i] / scale) / scale
    inverse[i] = value


@wp.kernel
def start_mv(x: wp.array[wp.vec3], mass: wp.array[float], flags: wp.array[int], inv_h2: float, out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = mass[i] * inv_h2 * x[i]
    if flags[i] & 1 == 0 or mass[i] == 0.0:
        out[i] = x[i]


@wp.func
def free_v(i: int, x: wp.array[wp.vec3], flags: wp.array[int], mass: wp.array[float]):
    v = wp.vec3(0.0)
    if flags[i] & 1 != 0 and mass[i] > 0.0:
        v = x[i]
    return v


@wp.kernel
def tet_mv(
    x: wp.array[wp.vec3],
    ids: wp.array2d[int],
    B: wp.array[wp.mat33],
    mat: wp.array2d[float],
    Fs: wp.array[wp.mat33],
    Cs: wp.array[wp.mat33],
    Es: wp.array[wp.mat33],
    flags: wp.array[int],
    mass: wp.array[float],
    h: float,
    full: int,
    out: wp.array[wp.vec3],
):
    t = wp.tid()
    F, C, E = Fs[t], Cs[t], Es[t]
    u0 = free_v(ids[t, 0], x, flags, mass)
    D = (
        wp.matrix_from_cols(
            free_v(ids[t, 1], x, flags, mass) - u0,
            free_v(ids[t, 2], x, flags, mass) - u0,
            free_v(ids[t, 3], x, flags, mass) - u0,
        )
        * B[t]
    )
    mu = mat[t, 0]
    lam = mat[t, 1] + mu
    Q = tet_tangent(F, C, E, D, mu, lam, mat[t, 2], h, full)
    V = 1.0 / (6.0 * wp.determinant(B[t]))
    for k in range(4):
        i = ids[t, k]
        if flags[i] & 1 != 0 and mass[i] > 0.0:
            wp.atomic_add(out, i, V * Q * grad(B[t], k))


@wp.kernel
def contact_mv(
    x: wp.array[wp.vec3],
    ids: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    count: wp.array[int],
    normals: wp.array[wp.vec3],
    active_k: wp.array[float],
    flags: wp.array[int],
    mass: wp.array[float],
    out: wp.array[wp.vec3],
):
    r = wp.tid()
    if r < count[0] and active_k[r] > 0.0:
        dx = wp.vec3(0.0)
        for k in range(3):
            i = ids[r][k]
            if i >= 0:
                dx += bary[r][k] * free_v(i, x, flags, mass)
        f = active_k[r] * wp.dot(normals[r], dx) * normals[r]
        for k in range(3):
            i = ids[r][k]
            if i >= 0:
                if flags[i] & 1 != 0 and mass[i] > 0.0:
                    wp.atomic_add(out, i, bary[r][k] * f)


@wp.kernel
def combine(v: wp.array[wp.vec3], y: wp.array[wp.vec3], z: wp.array[wp.vec3], alpha: float, beta: float):
    i = wp.tid()
    val = alpha * v[i]
    if beta != 0.0:
        val += beta * y[i]
    z[i] = val


@wp.kernel
def precond(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
    Hinv: wp.array[wp.mat33],
    alpha: float,
    beta: float,
):
    i = wp.tid()
    val = alpha * Hinv[i] * x[i]
    if beta != 0.0:
        val += beta * y[i]
    z[i] = val


@wp.func
def trial(q: wp.array[wp.vec3], delta: wp.array[wp.vec3], i: int, alpha: float):
    return wp.vec3d(q[i] + alpha * delta[i])


@wp.func
def tet_energy(
    a: wp.vec3d,
    b: wp.vec3d,
    c: wp.vec3d,
    d: wp.vec3d,
    F0: wp.mat33d,
    B: wp.mat33d,
    mu: wp.float64,
    lam: wp.float64,
    kd: wp.float64,
    h: wp.float64,
):
    F = wp.matrix_from_cols(b - a, c - a, d - a) * B
    J = wp.determinant(F)
    E = wp.transpose(F) * F - wp.transpose(F0) * F0
    V = wp.float64(1.0) / (wp.float64(6.0) * wp.determinant(B))
    # Drop the state-independent mu**2/(2*lam) term of the shifted determinant
    # energy; energy differences are unchanged and zero elasticity stays finite.
    return V * (
        wp.float64(0.5) * mu * wp.ddot(F, F)
        + wp.float64(0.5) * lam * (J - wp.float64(1.0)) ** wp.float64(2.0)
        - mu * (J - wp.float64(1.0))
        + kd / (wp.float64(2.0) * h) * wp.ddot(E, E)
    )


@wp.kernel
def tet_merit(
    q: wp.array[wp.vec3],
    old: wp.array[wp.vec3],
    delta: wp.array[wp.vec3],
    ids: wp.array2d[int],
    Bs: wp.array[wp.mat33],
    mat: wp.array2d[float],
    h: float,
    changes: wp.array[wp.float64],
):
    t, j = wp.tid()
    alpha = wp.pow(0.5, float(j))
    a, b, c, d = ids[t, 0], ids[t, 1], ids[t, 2], ids[t, 3]
    B = wp.mat33d(Bs[t])
    F0 = (
        wp.matrix_from_cols(
            wp.vec3d(old[b]) - wp.vec3d(old[a]),
            wp.vec3d(old[c]) - wp.vec3d(old[a]),
            wp.vec3d(old[d]) - wp.vec3d(old[a]),
        )
        * B
    )
    mu, lam, kd = wp.float64(mat[t, 0]), wp.float64(mat[t, 1] + mat[t, 0]), wp.float64(mat[t, 2])
    e0 = tet_energy(wp.vec3d(q[a]), wp.vec3d(q[b]), wp.vec3d(q[c]), wp.vec3d(q[d]), F0, B, mu, lam, kd, wp.float64(h))
    e1 = tet_energy(
        trial(q, delta, a, alpha),
        trial(q, delta, b, alpha),
        trial(q, delta, c, alpha),
        trial(q, delta, d, alpha),
        F0,
        B,
        mu,
        lam,
        kd,
        wp.float64(h),
    )
    wp.atomic_add(changes, j, e1 - e0)


@wp.kernel
def inertia_merit(
    q: wp.array[wp.vec3],
    inertia: wp.array[wp.vec3],
    delta: wp.array[wp.vec3],
    mass: wp.array[float],
    flags: wp.array[int],
    rhs: wp.array[wp.vec3],
    h: float,
    changes: wp.array[wp.float64],
    slope: wp.array[wp.float64],
):
    i, j = wp.tid()
    if flags[i] & 1 != 0 and mass[i] > 0.0:
        alpha = wp.pow(0.5, float(j))
        a = wp.vec3d(q[i]) - wp.vec3d(inertia[i])
        b = trial(q, delta, i, alpha) - wp.vec3d(inertia[i])
        wp.atomic_add(
            changes,
            j,
            wp.float64(0.5) * wp.float64(mass[i]) / (wp.float64(h) * wp.float64(h)) * (wp.dot(b, b) - wp.dot(a, a)),
        )
        if j == 0:
            wp.atomic_add(slope, 0, -wp.dot(wp.vec3d(rhs[i]), wp.vec3d(delta[i])))


@wp.kernel
def contact_merit(
    q: wp.array[wp.vec3],
    delta: wp.array[wp.vec3],
    ids: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    count: wp.array[int],
    normals: wp.array[wp.vec3],
    point: wp.array[wp.vec3],
    radius: wp.array[float],
    margin: wp.array[float],
    shapes: wp.array[int],
    ke: wp.array[float],
    changes: wp.array[wp.float64],
):
    r, j = wp.tid()
    if r < count[0]:
        alpha = wp.pow(0.5, float(j))
        a, b = wp.vec3d(0.0), wp.vec3d(0.0)
        for k in range(3):
            i = ids[r][k]
            if i >= 0:
                a += wp.float64(bary[r][k]) * wp.vec3d(q[i])
                b += wp.float64(bary[r][k]) * trial(q, delta, i, alpha)
        rad = wp.float64(contact_radius(radius, ids[r]) + margin[shapes[r]])
        n, pt = wp.vec3d(normals[r]), wp.vec3d(point[r])
        da = wp.max(wp.float64(0.0), rad - wp.dot(n, a - pt))
        db = wp.max(wp.float64(0.0), rad - wp.dot(n, b - pt))
        wp.atomic_add(changes, j, wp.float64(0.5) * wp.float64(ke[r]) * (db * db - da * da))


@wp.kernel
def choose_step(
    changes: wp.array[wp.float64], slope: wp.array[wp.float64], alpha: wp.array[float], stats: wp.array[int]
):
    chosen = float(0.0)
    if wp.isfinite(slope[0]) and slope[0] < wp.float64(0.0):
        for j in range(8):
            a = wp.pow(0.5, float(j))
            if chosen == 0.0 and wp.isfinite(changes[j]) and changes[j] <= wp.float64(1.0e-4 * a) * slope[0]:
                chosen = a
    alpha[0] = chosen
    wp.atomic_add(stats, 0, 1)
    if chosen > 0.0:
        wp.atomic_add(stats, 1, 1)


@wp.kernel
def check_full_trial(changes: wp.array[wp.float64], slope: wp.array[wp.float64], pending: wp.array[int]):
    pending[0] = int(
        wp.isfinite(slope[0])
        and slope[0] < wp.float64(0.0)
        and (not wp.isfinite(changes[0]) or changes[0] > wp.float64(1.0e-4) * slope[0])
    )


@wp.kernel
def check_model_validity_fallback(
    changes: wp.array[wp.float64],
    slope: wp.array[wp.float64],
    rho_min: float,
    flag: wp.array[int],
):
    """Trigger PSD Gauss--Newton fallback when the full Newton direction fails Armijo model validity."""
    s = slope[0]
    ch = changes[0]
    ok = int(0)
    if wp.isfinite(s) and s < wp.float64(0.0) and wp.isfinite(ch):
        if ch <= wp.float64(0.5 * rho_min) * s:
            ok = 1
    flag[0] = 1 - ok


@wp.kernel
def count_one(counter: wp.array[int]):
    wp.atomic_add(counter, 0, 1)


@wp.kernel
def apply_step(
    q: wp.array[wp.vec3],
    delta: wp.array[wp.vec3],
    flags: wp.array[int],
    mass: wp.array[float],
    alpha: wp.array[float],
    reference: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
):
    i = wp.tid()
    if flags[i] & 1 != 0 and mass[i] > 0.0 and alpha[0] > 0.0:
        q[i] = q[i] + alpha[0] * delta[i]
        displacement[i] = q[i] - reference[i]


# --- Affine Coarse Space Preconditioner Kernels ---
@wp.kernel
def factor_coarse(
    ac: wp.array3d[wp.float64],
    chol: wp.array3d[wp.float64],
    valid: wp.array[int],
    stats: wp.array[int],
    ranks: wp.array[int],
):
    component = wp.tid()
    n = 3 * ranks[component]
    ok = int(n > 0)
    scale = wp.float64(0.0)
    for i in range(n):
        scale = wp.max(scale, wp.abs(ac[component, i, i]))
    for i in range(n):
        for j in range(i + 1):
            v = wp.float64(0.5) * (ac[component, i, j] + ac[component, j, i])
            for k in range(j):
                v -= chol[component, i, k] * chol[component, j, k]
            if i == j:
                if not wp.isfinite(v) or v <= wp.float64(1.0e-12) * scale:
                    ok = 0
                    v = wp.float64(1.0)
                chol[component, i, j] = wp.sqrt(v)
            else:
                chol[component, i, j] = v / chol[component, j, j]
    valid[component] = ok
    wp.atomic_add(stats, 0, 1)
    wp.atomic_add(stats, 1, ok)


@wp.kernel
def invert_coarse(
    chol: wp.array3d[wp.float64], inverse: wp.array3d[wp.float64], valid: wp.array[int], ranks: wp.array[int]
):
    component, col = wp.tid()
    n = 3 * ranks[component]
    for i in range(inverse.shape[1]):
        inverse[component, i, col] = wp.float64(0.0)
    if valid[component] != 0 and col < n:
        for i in range(n):
            v = wp.float64(0.0)
            if i == col:
                v = wp.float64(1.0)
            for j in range(i):
                v -= chol[component, i, j] * inverse[component, j, col]
            inverse[component, i, col] = v / chol[component, i, i]
        for j in range(n):
            i = n - 1 - j
            v = inverse[component, i, col]
            for k in range(i + 1, n):
                v -= chol[component, k, i] * inverse[component, k, col]
            inverse[component, i, col] = v / chol[component, i, i]


@wp.kernel
def restrict_vector(phi: wp.array2d[float], components: wp.array[int], x: wp.array[wp.vec3], r: wp.array2d[wp.float64]):
    i, k = wp.tid()
    for axis in range(3):
        wp.atomic_add(r, components[i], 3 * k + axis, wp.float64(phi[i, k]) * wp.float64(x[i][axis]))


@wp.kernel
def apply_coarse_inverse(inverse: wp.array3d[wp.float64], r: wp.array2d[wp.float64], u: wp.array2d[wp.float64], n: int):
    component, i = wp.tid()
    v = wp.float64(0.0)
    for j in range(n):
        v += wp.float64(0.5) * (inverse[component, i, j] + inverse[component, j, i]) * r[component, j]
    u[component, i] = v


@wp.kernel
def prolong(
    phi: wp.array2d[float],
    u: wp.array2d[wp.float64],
    components: wp.array[int],
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
    blocks: wp.array[wp.mat33],
    alpha: float,
    beta: float,
    additive: int,
):
    i = wp.tid()
    v = wp.vec3d(0.0)
    for k in range(phi.shape[1]):
        for axis in range(3):
            v[axis] += wp.float64(phi[i, k]) * u[components[i], 3 * k + axis]
    result = wp.vec3(v)
    if additive != 0:
        result += blocks[i] * x[i]
    result *= alpha
    if beta != 0.0:
        result += beta * y[i]
    z[i] = result


@wp.kernel
def refresh_basis(source: wp.array2d[float], flags: wp.array[int], mass: wp.array[float], phi: wp.array2d[float]):
    i, k = wp.tid()
    value = float(0.0)
    if flags[i] & 1 != 0 and mass[i] > 0.0:
        value = source[i, k]
    phi[i, k] = value


@wp.func
def mode(phi: wp.array2d[float], i: int, col: int):
    v = wp.vec3(0.0)
    if i >= 0:
        v[col % 3] = phi[i, col // 3]
    return v


@wp.kernel
def start_coarse(phi: wp.array2d[float], mass: wp.array[float], inv_h2: float, ax: wp.array2d[wp.vec3]):
    col, i = wp.tid()
    ax[col, i] = mass[i] * inv_h2 * mode(phi, i, col)


@wp.kernel
def tets_coarse(
    phi: wp.array2d[float],
    ids: wp.array2d[int],
    B: wp.array[wp.mat33],
    mat: wp.array2d[float],
    Fs: wp.array[wp.mat33],
    Cs: wp.array[wp.mat33],
    Es: wp.array[wp.mat33],
    flags: wp.array[int],
    mass: wp.array[float],
    h: float,
    full: int,
    ax: wp.array2d[wp.vec3],
):
    col, t = wp.tid()
    F, C, E = Fs[t], Cs[t], Es[t]
    u0 = mode(phi, ids[t, 0], col)
    D = (
        wp.matrix_from_cols(
            mode(phi, ids[t, 1], col) - u0, mode(phi, ids[t, 2], col) - u0, mode(phi, ids[t, 3], col) - u0
        )
        * B[t]
    )
    mu = mat[t, 0]
    lam = mat[t, 1] + mu
    Q = tet_tangent(F, C, E, D, mu, lam, mat[t, 2], h, full)
    V = 1.0 / (6.0 * wp.determinant(B[t]))
    for k in range(4):
        i = ids[t, k]
        if flags[i] & 1 != 0 and mass[i] > 0.0:
            wp.atomic_add(ax, col, i, V * Q * grad(B[t], k))


@wp.kernel
def contacts_coarse(
    phi: wp.array2d[float],
    ids: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    count: wp.array[int],
    normals: wp.array[wp.vec3],
    active_k: wp.array[float],
    flags: wp.array[int],
    mass: wp.array[float],
    ax: wp.array2d[wp.vec3],
):
    col, r = wp.tid()
    if r < count[0] and active_k[r] > 0.0:
        dx = wp.vec3(0.0)
        for k in range(3):
            dx += bary[r][k] * mode(phi, ids[r][k], col)
        f = active_k[r] * wp.dot(normals[r], dx) * normals[r]
        for k in range(3):
            i = ids[r][k]
            if i >= 0:
                if flags[i] & 1 != 0 and mass[i] > 0.0:
                    wp.atomic_add(ax, col, i, bary[r][k] * f)


@wp.kernel
def restrict_coarse(
    phi: wp.array2d[float], components: wp.array[int], ax: wp.array2d[wp.vec3], ac: wp.array3d[wp.float64]
):
    col, i, k = wp.tid()
    for axis in range(3):
        wp.atomic_add(ac, components[i], 3 * k + axis, col, wp.float64(phi[i, k]) * wp.float64(ax[col, i][axis]))


class AffineCoarsePreconditioner:
    """Additive two-level preconditioner B^-1 + Z (Z^T A Z)^-1 Z^T on the fine operator.

    Rest-space affine modes are rank-revealed per connected component at
    construction. Prescribed particles are eliminated again at every setup.
    Cholesky failure disables only that component coarse term, not its
    base block-Jacobi preconditioner.
    """

    def __init__(self, backend: ParticleGlobalVBD):
        self.s = backend
        m = backend.model
        q = m.particle_q.numpy().astype(np.float64)
        free = ((m.particle_flags.numpy() & 1) != 0) & (m.particle_mass.numpy() > 0)
        if not np.any(free):
            raise ValueError("Affine preconditioning requires at least one free massive particle.")
        parent = np.arange(len(q))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for indices in (m.tet_indices, m.tri_indices):
            if indices is not None:
                for stencil in indices.numpy():
                    first = root(int(stencil[0]))
                    for i in stencil[1:]:
                        parent[root(int(i))] = first
        _, components = np.unique([root(i) for i in range(len(q))], return_inverse=True)
        self.component_count = int(components.max()) + 1
        ranks = np.zeros(self.component_count, dtype=np.int32)
        phi = np.zeros((len(q), 4), dtype=np.float32)
        order = np.argsort(components, kind="stable")
        offsets = np.searchsorted(components[order], np.arange(self.component_count + 1))
        for component in range(self.component_count):
            selected = order[offsets[component] : offsets[component + 1]]
            selected = selected[free[selected]]
            if len(selected) == 0:
                continue
            local = q[selected]
            length = max(float(np.linalg.norm(np.ptp(local, axis=0))), np.finfo(float).eps)
            basis = np.column_stack((np.ones(len(local)), (local - local.mean(axis=0)) / length))
            u, singular, _ = np.linalg.svd(basis, full_matrices=False)
            rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
            ranks[component] = rank
            phi[selected, :rank] = u[:, :rank]
        rank_max = int(ranks.max())
        phi = phi[:, :rank_max].copy()
        self.n, self.k = len(q), 3 * rank_max
        self.components = wp.array(components.astype(np.int32), dtype=int, device=backend.device)
        self.ranks = wp.array(ranks, dtype=int, device=backend.device)
        self.phi_source = wp.array(phi, dtype=float, device=backend.device)
        self.phi = wp.clone(self.phi_source)
        self.batched_ax = wp.zeros((self.k, self.n), dtype=wp.vec3, device=backend.device)
        self.ac = wp.zeros((self.component_count, self.k, self.k), dtype=wp.float64, device=backend.device)
        self.chol = wp.zeros_like(self.ac)
        self.inverse = wp.zeros_like(self.ac)
        self.r = wp.zeros((self.component_count, self.k), dtype=wp.float64, device=backend.device)
        self.u = wp.zeros_like(self.r)
        self.valid = wp.zeros(self.component_count, dtype=int, device=backend.device)
        self.stats = wp.zeros(2, dtype=int, device=backend.device)
        backend._global_cg._M = LinearOperator((3 * self.n, 3 * self.n), wp.mat33, backend.device, self.apply)

    def setup(self):
        s, m = self.s, self.s.model
        wp.launch(
            refresh_basis,
            self.phi.shape,
            [self.phi_source, m.particle_flags, m.particle_mass, self.phi],
            device=s.device,
        )
        self.ac.zero_()
        wp.launch(
            start_coarse,
            (self.k, self.n),
            [self.phi, m.particle_mass, 1.0 / s._global_h**2, self.batched_ax],
            device=s.device,
        )
        wp.launch(
            tets_coarse,
            (self.k, m.tet_count),
            [
                self.phi,
                m.tet_indices,
                m.tet_poses,
                m.tet_materials,
                s._global_F,
                s._global_C,
                s._global_E,
                m.particle_flags,
                m.particle_mass,
                s._global_h,
                s._global_full,
                self.batched_ax,
            ],
            device=s.device,
        )
        c = s._global_contacts
        if c is not None and c.soft_contact_max > 0:
            wp.launch(
                contacts_coarse,
                (self.k, c.soft_contact_max),
                [
                    self.phi,
                    c.soft_contact_indices,
                    c.soft_contact_barycentric,
                    c.soft_contact_count,
                    c.soft_contact_normal,
                    s._global_contact_k,
                    m.particle_flags,
                    m.particle_mass,
                    self.batched_ax,
                ],
                device=s.device,
            )
        wp.launch(
            restrict_coarse,
            (self.k, self.n, self.k // 3),
            [self.phi, self.components, self.batched_ax, self.ac],
            device=s.device,
        )
        wp.launch(
            factor_coarse,
            self.component_count,
            [self.ac, self.chol, self.valid, self.stats, self.ranks],
            device=s.device,
        )
        wp.launch(
            invert_coarse,
            (self.component_count, self.k),
            [self.chol, self.inverse, self.valid, self.ranks],
            device=s.device,
        )

    def apply(self, x, y, z, alpha, beta):
        self.r.zero_()
        wp.launch(restrict_vector, (self.n, self.k // 3), [self.phi, self.components, x, self.r], device=self.s.device)
        wp.launch(
            apply_coarse_inverse,
            (self.component_count, self.k),
            [self.inverse, self.r, self.u, self.k],
            device=self.s.device,
        )
        wp.launch(
            prolong,
            self.n,
            [self.phi, self.u, self.components, x, y, z, self.s._global_inverse, alpha, beta, 1],
            device=self.s.device,
        )


class ParticleGlobalVBD:
    """Private tetrahedral Krylov backend; SolverVBD exposes only its stage count.

    The remaining constructor options are internal experiment controls, not stable
    public API. Keep defaults here so the public wrapper cannot silently diverge.
    """

    def __init__(
        self,
        solver,
        *,
        global_stages: int = 1,
        linear_iterations: int = 32,
        curvature: str = "full",
        preconditioner: str = "affine",
        initial_guess: str = "constant-velocity",
        global_placement: str = "penultimate",
        rho_min: float = 0.0,
    ):
        if preconditioner not in ("block-jacobi", "affine"):
            raise ValueError("preconditioner must be block-jacobi or affine.")
        if curvature not in ("full", "gn"):
            raise ValueError("curvature must be full or gn.")
        if initial_guess not in ("inertial", "constant-velocity"):
            raise ValueError("initial_guess must be inertial or constant-velocity.")
        if global_placement not in ("interleaved", "penultimate"):
            raise ValueError("global_placement must be interleaved or penultimate.")
        if (
            isinstance(linear_iterations, bool)
            or not isinstance(linear_iterations, (int, np.integer))
            or linear_iterations < 1
        ):
            raise ValueError("linear_iterations must be a positive integer.")
        if isinstance(global_stages, bool) or not isinstance(global_stages, (int, np.integer)) or global_stages < 1:
            raise ValueError("global_stages must be a positive integer.")
        if not np.isfinite(rho_min) or not 0.0 <= rho_min <= 1.0:
            raise ValueError("rho_min must be finite and in [0, 1].")

        self.solver = solver
        self.model = model = solver.model
        self.device = solver.device
        self.validate_model()

        n, nt = model.particle_count, model.tet_count
        if solver.iterations < 2 or global_stages < 1 or global_stages >= solver.iterations:
            raise ValueError("Require tets, at least two local sweeps, and 1 <= global_stages < iterations.")

        self.initial_guess = initial_guess
        self.global_placement = global_placement
        if global_placement == "penultimate" and global_stages == 1:
            self.global_indices = frozenset((solver.iterations - 2,))
        else:
            self.global_indices = frozenset(
                np.linspace(0, max(0, solver.iterations - 2), global_stages, dtype=int).tolist()
            )

        self._global_full = int(curvature == "full")
        self._configured_full = self._global_full
        self._rho_min = float(rho_min)
        self._global_h = 1.0
        self._global_contacts = None
        initial_contact_cap = int(solver.body_particle_contact_penalty_k.shape[0])
        self._global_contact_k = (
            wp.zeros(initial_contact_cap, dtype=float, device=self.device) if initial_contact_cap > 0 else None
        )

        def arr(count, dtype):
            return wp.zeros(count, dtype=dtype, device=self.device)

        self._global_rhs = arr(n, wp.vec3)
        self._global_delta = arr(n, wp.vec3)
        self._global_diag = arr(n, wp.mat33)
        self._global_inverse = arr(n, wp.mat33)
        self._global_F = arr(nt, wp.mat33)
        self._global_C = arr(nt, wp.mat33)
        self._global_E = arr(nt, wp.mat33)
        self._global_Ax = arr(n, wp.vec3)
        self._global_energy = arr(8, wp.float64)
        self._global_slope = arr(1, wp.float64)
        self._global_alpha = arr(1, float)
        self._global_trial_pending = arr(1, int)
        self._global_fallback = arr(1, int)
        self.fallback_count = arr(1, int)
        self._global_lazy_merit = True
        self.global_stats = arr(2, int)

        A = LinearOperator((3 * n, 3 * n), wp.mat33, self.device, self._global_matvec)
        M = LinearOperator((3 * n, 3 * n), wp.mat33, self.device, self._global_precond)
        self._global_cg = cg(
            A,
            self._global_rhs,
            self._global_delta,
            M=M,
            tol=1.0e-4,
            atol=1.0e-6,
            maxiter=linear_iterations,
            check_every=0,
            run=False,
        )
        self._global_coarse = None
        if preconditioner == "affine":
            self._global_coarse = AffineCoarsePreconditioner(self)

    def validate_model(self) -> None:
        """Validate the supported material/contact domain without replacing device scratch."""
        solver, model = self.solver, self.model
        if model.tet_count == 0:
            raise ValueError("particle_global_iterations requires tetrahedra.")
        materials = model.tet_materials.numpy()
        mu, lam, damping = materials[:, 0], materials[:, 0] + materials[:, 1], materials[:, 2]
        if (
            not np.isfinite(materials).all()
            or np.any(mu < 0.0)
            or np.any(lam < 0.0)
            or np.any(damping < 0.0)
            or np.any((mu > 0.0) & (lam < 1.0e-6))
        ):
            raise ValueError(
                "particle_global_iterations requires finite, nonnegative shear, shifted-volume and damping "
                "coefficients, with lambda + mu >= 1e-6 when mu > 0; smaller shifted-volume coefficients "
                "use a different clamped constitutive law in local VBD."
            )
        if model.world_count > 1:
            raise ValueError("particle_global_iterations does not yet support multiple worlds.")
        if model.body_count or model.spring_count or model.tendon_count:
            raise ValueError("particle_global_iterations requires tet particles and static rigid shapes.")
        if any(
            array is not None and np.any(array.numpy() != 0)
            for array in (model.tri_materials, model.edge_bending_properties)
        ):
            raise ValueError("Surface membrane/bending is not supported by particle_global_iterations.")
        if (
            model.soft_contact_mu != 0
            or model.soft_contact_kd != 0
            or (model.shape_material_kd is not None and np.any(model.shape_material_kd.numpy() != 0))
        ):
            raise ValueError("particle_global_iterations requires frictionless, undamped soft contacts.")
        if solver.particle_enable_self_contact or solver.rigid_soft_enable_dat:
            raise ValueError("Self-contact and DAT are not supported by particle_global_iterations.")
        if solver.rigid_soft_contact_use_log_barrier:
            raise ValueError("Log-barrier contact is not supported by particle_global_iterations.")

    def ensure_contact_capacity(self, soft_contact_max: int) -> None:
        if soft_contact_max <= 0:
            return
        if self._global_contact_k is None or self._global_contact_k.shape[0] < soft_contact_max:
            current = 0 if self._global_contact_k is None else self._global_contact_k.shape[0]
            self.solver._raise_if_capturing_resize("particle global contact state", current, soft_contact_max)
            self._global_contact_k = wp.zeros(soft_contact_max, dtype=float, device=self.device)

    def validate_step(self, contacts) -> None:
        """Reject unimplemented consumer semantics before the step mutates particle state."""
        if contacts is not None and contacts._soft_contact_mesh_features is not None:
            raise NotImplementedError(
                "particle_global_iterations does not support feature-filtered mesh contacts; "
                "use the local particle solver for this contact representation."
            )
        if self._configured_full and self.device.is_capturing and not wp.is_conditional_graph_supported():
            raise RuntimeError(
                "Full-curvature particle global capture requires CUDA conditional graphs for its fallback; "
                "use eager execution on devices without conditional graph support."
            )
        if contacts is not None:
            self.ensure_contact_capacity(contacts.soft_contact_max)

    def apply_initial_guess(self, state_in, dt: float) -> None:
        if self.initial_guess != "constant-velocity":
            return
        m = self.model
        s = self.solver
        wp.launch(
            constant_velocity_initial_iterate,
            m.particle_count,
            [
                s.particle_q_prev,
                state_in.particle_qd,
                m.particle_mass,
                m.particle_flags,
                dt,
                s.pos_prev_collision_detection,
                state_in.particle_q,
                s.particle_displacements,
            ],
            device=self.device,
        )

    def _global_matvec(self, x, y, z, alpha, beta):
        m = self.model
        h = self._global_h
        c = self._global_contacts
        item_size = wp.types.type_size_in_bytes(wp.vec3)
        disjoint = x.ptr + x.size * item_size <= z.ptr or z.ptr + z.size * item_size <= x.ptr
        direct = alpha == 1.0 and beta == 0.0 and disjoint
        out = z if direct else self._global_Ax
        wp.launch(
            start_mv,
            m.particle_count,
            [x, m.particle_mass, m.particle_flags, 1.0 / (h * h), out],
            device=self.device,
        )
        wp.launch(
            tet_mv,
            m.tet_count,
            [
                x,
                m.tet_indices,
                m.tet_poses,
                m.tet_materials,
                self._global_F,
                self._global_C,
                self._global_E,
                m.particle_flags,
                m.particle_mass,
                h,
                self._global_full,
                out,
            ],
            device=self.device,
        )
        if c is not None and c.soft_contact_max > 0:
            wp.launch(
                contact_mv,
                c.soft_contact_max,
                [
                    x,
                    c.soft_contact_indices,
                    c.soft_contact_barycentric,
                    c.soft_contact_count,
                    c.soft_contact_normal,
                    self._global_contact_k,
                    m.particle_flags,
                    m.particle_mass,
                    out,
                ],
                device=self.device,
            )
        if not direct:
            wp.launch(combine, m.particle_count, [out, y, z, alpha, beta], device=self.device)

    def _global_precond(self, x, y, z, alpha, beta):
        wp.launch(precond, self.model.particle_count, [x, y, z, self._global_inverse, alpha, beta], device=self.device)

    def solve(self, state_in, contacts, dt: float) -> None:
        m = self.model
        s = self.solver
        q = state_in.particle_q
        self._global_h = dt
        self._global_contacts = contacts
        self._global_full = self._configured_full
        if contacts is not None and contacts.soft_contact_max > 0:
            self.ensure_contact_capacity(contacts.soft_contact_max)
        wp.launch(
            initialize,
            m.particle_count,
            [q, s.inertia, m.particle_mass, m.particle_flags, 1.0 / (dt * dt), self._global_rhs, self._global_diag],
            device=self.device,
        )
        wp.launch(
            assemble_tets,
            m.tet_count,
            [
                q,
                s.particle_q_prev,
                m.tet_indices,
                m.tet_poses,
                m.tet_materials,
                m.particle_flags,
                m.particle_mass,
                dt,
                self._global_rhs,
                self._global_diag,
                self._global_F,
                self._global_C,
                self._global_E,
            ],
            device=self.device,
        )
        if contacts is not None and contacts.soft_contact_max > 0:
            wp.launch(
                assemble_contacts,
                contacts.soft_contact_max,
                [
                    q,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_count,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_body_pos,
                    m.particle_radius,
                    m.shape_margin,
                    s.body_particle_contact_penalty_k,
                    m.particle_flags,
                    m.particle_mass,
                    self._global_rhs,
                    self._global_diag,
                    self._global_contact_k,
                ],
                device=self.device,
            )
        wp.launch(invert_blocks, m.particle_count, [self._global_diag, self._global_inverse], device=self.device)
        self._global_delta.zero_()
        if self._global_coarse is not None:
            self._global_coarse.setup()
        self._global_cg()
        self._accept_global_step(q, contacts, dt)

    def _accept_global_step(self, q, c, h: float) -> None:
        m = self.model
        s = self.solver

        def evaluate_trials(trials):
            self._global_energy.zero_()
            self._global_slope.zero_()
            wp.launch(
                tet_merit,
                (m.tet_count, trials),
                [
                    q,
                    s.particle_q_prev,
                    self._global_delta,
                    m.tet_indices,
                    m.tet_poses,
                    m.tet_materials,
                    h,
                    self._global_energy,
                ],
                device=self.device,
            )
            wp.launch(
                inertia_merit,
                (m.particle_count, trials),
                [
                    q,
                    s.inertia,
                    self._global_delta,
                    m.particle_mass,
                    m.particle_flags,
                    self._global_rhs,
                    h,
                    self._global_energy,
                    self._global_slope,
                ],
                device=self.device,
            )
            if c is not None and c.soft_contact_max > 0:
                wp.launch(
                    contact_merit,
                    (c.soft_contact_max, trials),
                    [
                        q,
                        self._global_delta,
                        c.soft_contact_indices,
                        c.soft_contact_barycentric,
                        c.soft_contact_count,
                        c.soft_contact_normal,
                        c.soft_contact_body_pos,
                        m.particle_radius,
                        m.shape_margin,
                        c.soft_contact_shape,
                        s.body_particle_contact_penalty_k,
                        self._global_energy,
                    ],
                    device=self.device,
                )

        lazy = self._global_lazy_merit and self.device.is_capturing and wp.is_conditional_graph_supported()
        trials = 1 if lazy else 8
        evaluate_trials(trials)

        if self._configured_full != 0:
            wp.launch(
                check_model_validity_fallback,
                1,
                [self._global_energy, self._global_slope, self._rho_min, self._global_fallback],
                device=self.device,
            )

            def fallback_to_gn():
                self._global_full = 0
                if self._global_coarse is not None:
                    self._global_coarse.setup()
                self._global_delta.zero_()
                self._global_cg()
                self._global_full = self._configured_full
                wp.launch(count_one, 1, [self.fallback_count], device=self.device)
                evaluate_trials(trials)

            if self.device.is_capturing and wp.is_conditional_graph_supported():
                wp.capture_if(self._global_fallback, fallback_to_gn)
            elif not self.device.is_capturing and int(self._global_fallback.numpy()[0]):
                fallback_to_gn()

        if lazy:
            wp.launch(
                check_full_trial,
                1,
                [self._global_energy, self._global_slope, self._global_trial_pending],
                device=self.device,
            )
            wp.capture_if(self._global_trial_pending, evaluate_trials, trials=8)
        wp.launch(
            choose_step,
            1,
            [self._global_energy, self._global_slope, self._global_alpha, self.global_stats],
            device=self.device,
        )
        wp.launch(
            apply_step,
            m.particle_count,
            [
                q,
                self._global_delta,
                m.particle_flags,
                m.particle_mass,
                self._global_alpha,
                s.pos_prev_collision_detection,
                s.particle_displacements,
            ],
            device=self.device,
        )
