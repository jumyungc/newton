"""Chain-condensed contact prototype for one full-dt cable linearization.

The central object in this module is the *stretch-constrained* primal inverse

    S(r) = d,  where [H  A.T] [d ] = [r]
                         [A  -C ] [ls]   [0].

``C`` is zero for hard/inextensible stretch and is the authored stretch
compliance for finite stretch.  Contact never sees an unconstrained cable
inverse followed by a stretch repair: its Delassus operator is exactly

    W x = J S(J.T x) + Cc x.

For the small CPU research cases in scope here, frictionless Signorini is
solved by an exhaustive active set and high-friction sticking is solved as an
equality only after the resulting force is certified inside the authored
Coulomb cone.  Low/moderate-friction blocks use the non-associated De Saxce
natural map; certified high-mu blocks are Schur-condensed before that solve,
then the recovered *full* law is checked again.  Unsupported sizes or failed
certificates raise.  The public ``delassus_linear_operator`` keeps the same
construction matrix-free for a future iterative contact solve.

This is deliberately a one-linearization prototype, not a pile solver or a
production integrator.  ``solve_one_linearized_frame`` predicts exactly one
full ``dt``, applies one coupled SQP direction to a private state, and never
mutates its caller.  It performs no hidden time substeps and fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, splu

from bench.global_cable import codex_al_bgn as al
from bench.global_cable import codex_bgn_contact_kkt as contact_v1
from bench.global_cable import codex_bgn_global as bgn
from bench.global_cable import codex_desaxce as desaxce
from bench.global_cable import codex_rc_forest_newton as rc


SCHEMA = "codex-chain-condensed-de-saxce-newton/v0-linear"
FINITE_STRETCH = "finite_authored_compliance"
HARD_STRETCH = "hard_inextensible"
FRICTIONLESS = "frictionless_signorini"
STICKING = "certified_high_mu_sticking"
DESAXCE = "desaxce_coulomb"
CONTACT_MODES = (FRICTIONLESS, STICKING, DESAXCE)


class CoupledSolveError(RuntimeError):
    """The coupled system could not be certified; no caller state is valid."""


class ContactModeNotApplicable(CoupledSolveError):
    """The requested exact small-contact mode is not safely applicable."""


def owner_guard(where: str) -> str:
    """Share the coordinator's exact, current owner guard."""
    return al.owner_guard(f"CCDSN {where}")


def _array_digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_sparse(value: sparse.spmatrix, name: str) -> sparse.csc_matrix:
    matrix = value.astype(np.float64).tocsc(copy=True)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{name} must be a finite rank-two sparse matrix")
    return matrix


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if out.shape != (size,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite with shape ({size},)")
    return out


def _nonnegative_vector(value: Any, size: int, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        out = np.full(size, float(raw), dtype=np.float64)
    else:
        out = raw.reshape(-1).copy()
    if out.shape != (size,) or not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError(f"{name} must be finite, nonnegative, and scalar or length {size}")
    return out


class ChainKKTInverse:
    """Reusable exact sparse factorization of the stretch saddle system."""

    def __init__(
        self,
        H: sparse.spmatrix,
        A: sparse.spmatrix,
        stretch_compliance: Any,
        *,
        symmetry_tolerance: float = 2.0e-11,
    ) -> None:
        self.H = _finite_sparse(H, "H")
        if self.H.shape[0] != self.H.shape[1] or self.H.shape[0] == 0:
            raise ValueError("H must be nonempty and square")
        asymmetry = self.H - self.H.T
        asymmetry_inf = float(np.max(np.abs(asymmetry.data), initial=0.0))
        scale = max(1.0, float(np.max(np.abs(self.H.data), initial=0.0)))
        if asymmetry_inf > symmetry_tolerance * scale:
            raise ValueError(f"H is not symmetric: relative asymmetry {asymmetry_inf / scale}")
        self.H = (0.5 * (self.H + self.H.T)).tocsc()
        self.A = _finite_sparse(A, "A").tocsr()
        if self.A.shape[1] != self.H.shape[0]:
            raise ValueError("A column count must equal H dimension")
        self.n = self.H.shape[0]
        self.m = self.A.shape[0]
        self.compliance = _nonnegative_vector(
            stretch_compliance, self.m, "stretch_compliance"
        )
        lower = -sparse.diags(self.compliance, format="csc")
        self.K = sparse.bmat(
            [[self.H, self.A.T], [self.A, lower]], format="csc"
        )
        try:
            self.factor = splu(self.K, permc_spec="COLAMD")
        except Exception as exc:
            raise CoupledSolveError(f"stretch KKT factorization failed: {exc}") from exc
        self.factor_solve_calls = 0

    def solve(
        self, primal_rhs: Any, constraint_rhs: Any | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        rhs = np.asarray(primal_rhs, dtype=np.float64)
        one = rhs.ndim == 1
        if one:
            rhs = rhs.reshape(-1, 1)
        if rhs.ndim != 2 or rhs.shape[0] != self.n or not np.all(np.isfinite(rhs)):
            raise ValueError("primal_rhs has invalid shape or non-finite values")
        columns = rhs.shape[1]
        if constraint_rhs is None:
            lower = np.zeros((self.m, columns), dtype=np.float64)
        else:
            lower = np.asarray(constraint_rhs, dtype=np.float64)
            if lower.ndim == 1:
                lower = lower.reshape(-1, 1)
            if lower.shape == (self.m, 1) and columns > 1:
                lower = np.repeat(lower, columns, axis=1)
            if lower.shape != (self.m, columns) or not np.all(np.isfinite(lower)):
                raise ValueError("constraint_rhs has invalid shape or non-finite values")
        joined = np.vstack((rhs, lower))
        try:
            solved = np.asarray(self.factor.solve(joined), dtype=np.float64)
        except Exception as exc:
            raise CoupledSolveError(f"stretch KKT solve failed: {exc}") from exc
        self.factor_solve_calls += 1
        if not np.all(np.isfinite(solved)):
            raise CoupledSolveError("stretch KKT solve returned non-finite values")
        residual = self.K @ solved - joined
        residual_scale = max(1.0, float(np.max(np.abs(joined), initial=0.0)))
        if float(np.max(np.abs(residual), initial=0.0)) > 2.0e-8 * residual_scale:
            raise CoupledSolveError("stretch KKT factorization failed its residual gate")
        d, multiplier = solved[: self.n], solved[self.n :]
        if one:
            return d[:, 0], multiplier[:, 0]
        return d, multiplier

    def free(self, rhs: Any, stretch_offset: Any) -> tuple[np.ndarray, np.ndarray]:
        offset = _vector(stretch_offset, self.m, "stretch_offset")
        return self.solve(rhs, -offset)

    def response(self, primal_rhs: Any) -> tuple[np.ndarray, np.ndarray]:
        return self.solve(primal_rhs, None)


def delassus_linear_operator(
    inverse: ChainKKTInverse,
    J: sparse.spmatrix,
    contact_compliance: Any = 0.0,
) -> LinearOperator:
    """Return matrix-free ``x -> J S(J.T x) + Cc x``."""
    jacobian = _finite_sparse(J, "J").tocsr()
    if jacobian.shape[1] != inverse.n:
        raise ValueError("J column count must equal the primal dimension")
    compliance = _nonnegative_vector(
        contact_compliance, jacobian.shape[0], "contact_compliance"
    )

    def matvec(x: np.ndarray) -> np.ndarray:
        value = _vector(x, jacobian.shape[0], "Delassus vector")
        response, _ = inverse.response(jacobian.T @ value)
        return np.asarray(jacobian @ response, dtype=np.float64) + compliance * value

    def matmat(x: np.ndarray) -> np.ndarray:
        value = np.asarray(x, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != jacobian.shape[0] or not np.all(np.isfinite(value)):
            raise ValueError("Delassus matrix argument has invalid shape")
        response, _ = inverse.response(jacobian.T @ value)
        return np.asarray(jacobian @ response, dtype=np.float64) + compliance[:, None] * value

    return LinearOperator(
        (jacobian.shape[0], jacobian.shape[0]),
        matvec=matvec,
        matmat=matmat,
        dtype=np.float64,
    )


def _enumerate_frictionless_lcp(
    W: np.ndarray,
    b: np.ndarray,
    *,
    contact_limit: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    count = len(b)
    if count > contact_limit:
        raise ContactModeNotApplicable(
            f"{count} frictionless contacts exceed exact active-set limit {contact_limit}"
        )
    for mask in range(1 << count):
        active = [i for i in range(count) if mask & (1 << i)]
        lam = np.zeros(count, dtype=np.float64)
        if active:
            block = W[np.ix_(active, active)]
            try:
                lam[active] = np.linalg.solve(block, -b[active])
            except np.linalg.LinAlgError:
                continue
        gap = b + W @ lam
        inactive = [i for i in range(count) if i not in active]
        if (
            np.all(lam[active] >= -tolerance)
            and np.all(np.abs(gap[active]) <= tolerance)
            and np.all(gap[inactive] >= -tolerance)
        ):
            lam[np.abs(lam) <= tolerance] = 0.0
            return lam, gap, active
    raise CoupledSolveError("no certifiable frictionless active set exists")


def _cone_violation(force: np.ndarray, mu: np.ndarray) -> float:
    if len(force) == 0:
        return 0.0
    block = np.asarray(force, dtype=np.float64).reshape(-1, 3)
    normal = block[:, 0]
    tangent = np.linalg.norm(block[:, 1:], axis=1)
    return float(max(
        np.max(np.maximum(-normal, 0.0), initial=0.0),
        np.max(np.maximum(tangent - mu * normal, 0.0), initial=0.0),
    ))


def _enumerate_sticking_sets(
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    *,
    contact_limit: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    count = len(mu)
    if count > contact_limit:
        raise ContactModeNotApplicable(
            f"{count} sticking contacts exceed exact active-set limit {contact_limit}"
        )
    for mask in range(1 << count):
        active_contacts = [i for i in range(count) if mask & (1 << i)]
        active_rows = [3 * i + j for i in active_contacts for j in range(3)]
        force = np.zeros(3 * count, dtype=np.float64)
        if active_rows:
            block = W[np.ix_(active_rows, active_rows)]
            try:
                force[active_rows] = np.linalg.solve(block, -b[active_rows])
            except np.linalg.LinAlgError:
                continue
        residual = b + W @ force
        inactive = [i for i in range(count) if i not in active_contacts]
        active_force = force.reshape(-1, 3)[active_contacts]
        cone_ok = _cone_violation(active_force, mu[active_contacts]) <= tolerance
        if (
            cone_ok
            and np.all(np.abs(residual[active_rows]) <= tolerance)
            and np.all(residual.reshape(-1, 3)[inactive, 0] >= -tolerance)
        ):
            force[np.abs(force) <= tolerance] = 0.0
            return force, residual, active_contacts
    raise ContactModeNotApplicable(
        "no sticking subset satisfies unilateral normal and authored cone gates"
    )


def _contact_block_rows(contacts: list[int] | np.ndarray) -> np.ndarray:
    value = np.asarray(contacts, dtype=np.int64).reshape(-1)
    if len(value) == 0:
        return np.empty(0, dtype=np.int64)
    return (3 * value[:, None] + np.arange(3)[None, :]).reshape(-1)


def _desaxce_full_certificates(
    impulse: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
) -> dict[str, Any]:
    """Evaluate the full non-associated law after any stick condensation."""
    impulse = np.asarray(impulse, dtype=np.float64).reshape(-1)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    velocity = b + W @ impulse
    corrected, _ = desaxce.desaxce_correction_with_jacobian(velocity, mu)
    spectral_radius = max(float(np.linalg.eigvalsh(W)[-1]), 1.0e-12)
    gamma = 1.0 / spectral_radius
    residual, aux = desaxce.natural_map(impulse, W, b, mu, gamma)
    scale = max(
        1.0,
        float(np.max(np.abs(impulse), initial=0.0)),
        gamma * float(np.max(np.abs(corrected), initial=0.0)),
    )
    blocks = impulse.reshape(-1, 3)
    velocity_blocks = velocity.reshape(-1, 3)
    corrected_blocks = corrected.reshape(-1, 3)
    complementarity = float(
        np.max(np.abs(np.sum(blocks * corrected_blocks, axis=1)), initial=0.0)
    )
    friction_work = float(np.sum(blocks[:, 1:] * velocity_blocks[:, 1:]))
    return {
        "velocity": velocity,
        "corrected_velocity": corrected,
        "gamma": gamma,
        "natural_map_inf": float(np.max(np.abs(residual), initial=0.0)),
        "natural_map_relative_inf": float(
            np.max(np.abs(residual), initial=0.0) / scale
        ),
        "primal_cone_violation": desaxce.cone_violation(impulse, mu),
        "dual_cone_violation": desaxce.dual_cone_violation(corrected, mu),
        "complementarity_inf": complementarity,
        "friction_work": friction_work,
        "regions": list(aux["regions"]),
    }


def _solve_desaxce_with_stick_condensation(
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    *,
    stick_mu: float,
    contact_limit: int,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    """Condense certified high-mu blocks and solve the rest with De Saxce.

    Every subset of the bounded high-friction contacts is considered.  A
    candidate is accepted only if the recovered *full* impulse/velocity pair
    satisfies the natural map, primal cone, dual cone, and complementarity.
    Thus a high-mu equality is a fast path only when it is also a valid
    non-associated Coulomb solution.
    """
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    contacts = len(mu)
    if contacts > contact_limit:
        raise ContactModeNotApplicable(
            f"{contacts} contacts exceed De Saxce direct limit {contact_limit}"
        )
    high = np.flatnonzero(mu >= stick_mu)
    low = np.flatnonzero(mu < stick_mu)
    low_rows = _contact_block_rows(low)
    rejection: list[str] = []
    for mask in range(1 << len(high)):
        active_high = np.asarray(
            [int(high[i]) for i in range(len(high)) if mask & (1 << i)],
            dtype=np.int64,
        )
        high_rows = _contact_block_rows(active_high)
        try:
            if len(high_rows):
                Wss = W[np.ix_(high_rows, high_rows)]
                solved_b = np.linalg.solve(Wss, b[high_rows])
                if len(low_rows):
                    Wsl = W[np.ix_(high_rows, low_rows)]
                    Wls = Wsl.T
                    solved_cross = np.linalg.solve(Wss, Wsl)
                    reduced_W = W[np.ix_(low_rows, low_rows)] - Wls @ solved_cross
                    reduced_b = b[low_rows] - Wls @ solved_b
                else:
                    Wsl = np.empty((len(high_rows), 0), dtype=np.float64)
                    reduced_W = np.empty((0, 0), dtype=np.float64)
                    reduced_b = np.empty(0, dtype=np.float64)
            else:
                Wss = np.empty((0, 0), dtype=np.float64)
                Wsl = np.empty((0, len(low_rows)), dtype=np.float64)
                solved_b = np.empty(0, dtype=np.float64)
                reduced_W = W[np.ix_(low_rows, low_rows)]
                reduced_b = b[low_rows]

            if len(low):
                reduced = desaxce.solve_desaxce_dense(
                    reduced_W,
                    reduced_b,
                    mu[low],
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    direct_contact_limit=contact_limit,
                    fail_closed=True,
                )
                low_impulse = reduced.impulse
                reduced_info = reduced.info
            else:
                low_impulse = np.empty(0, dtype=np.float64)
                reduced_info = {
                    "method": "no moderate-friction blocks after stick condensation",
                    "contacts": 0,
                    "iterations": 0,
                    "converged": True,
                    "natural_map_inf": 0.0,
                    "natural_map_relative_inf": 0.0,
                }

            impulse = np.zeros(3 * contacts, dtype=np.float64)
            impulse[low_rows] = low_impulse
            if len(high_rows):
                impulse[high_rows] = -np.linalg.solve(
                    Wss, b[high_rows] + Wsl @ low_impulse
                )
            certificates = _desaxce_full_certificates(impulse, W, b, mu)
            full_scale = max(1.0, float(np.max(np.abs(impulse), initial=0.0)))
            gate = max(20.0 * tolerance * full_scale, 2.0e-9 * full_scale)
            high_force = impulse.reshape(-1, 3)[active_high]
            high_cone = _cone_violation(high_force, mu[active_high])
            if (
                certificates["natural_map_inf"] <= gate
                and certificates["primal_cone_violation"] <= gate
                and certificates["dual_cone_violation"] <= gate
                and certificates["complementarity_inf"] <= gate
                and high_cone <= gate
                and certificates["friction_work"] <= gate
            ):
                velocity = certificates["velocity"].reshape(-1, 3)
                active = np.flatnonzero(
                    impulse.reshape(-1, 3)[:, 0] > gate
                ).astype(int).tolist()
                sliding = [
                    int(contact) for contact in low
                    if impulse.reshape(-1, 3)[contact, 0] > gate
                    and np.linalg.norm(velocity[contact, 1:]) > gate
                ]
                info = {
                    "method": "high-mu stick Schur condensation + dense De Saxce",
                    "sticking_contact_indices": active_high.astype(int).tolist(),
                    "moderate_friction_contact_indices": low.astype(int).tolist(),
                    "sliding_contact_indices": sliding,
                    "reduced_desaxce": reduced_info,
                    **certificates,
                }
                return impulse, active, info
            rejection.append(
                f"mask={mask}: natural={certificates['natural_map_inf']:.3e}, "
                f"primal={certificates['primal_cone_violation']:.3e}, "
                f"dual={certificates['dual_cone_violation']:.3e}, "
                f"comp={certificates['complementarity_inf']:.3e}"
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            rejection.append(f"mask={mask}: {type(exc).__name__}: {exc}")
    tail = "; ".join(rejection[-4:])
    raise ContactModeNotApplicable(
        "no high-mu active set produced a certified full De Saxce law"
        + (f": {tail}" if tail else "")
    )


@dataclass(frozen=True)
class CoupledResult:
    direction: np.ndarray
    stretch_force_N: np.ndarray
    contact_force_N: np.ndarray
    free_direction: np.ndarray
    info: dict[str, Any]


def solve_coupled_quadratic(
    H: sparse.spmatrix,
    rhs: Any,
    A: sparse.spmatrix,
    stretch_offset: Any,
    stretch_compliance: Any,
    J: sparse.spmatrix,
    contact_offset: Any,
    *,
    contact_mode: str,
    contact_compliance: Any = 1.0e-9,
    friction_mu: Any | None = None,
    stick_mu: float = 1.0e3,
    exact_contact_limit: int = 10,
    certificate_tolerance: float = 2.0e-9,
    desaxce_tolerance: float = 1.0e-10,
    desaxce_max_iterations: int = 80,
    dt: float,
) -> CoupledResult:
    """Solve stretch and small-contact KKT conditions simultaneously."""
    owner_guard("coupled quadratic start")
    if contact_mode not in CONTACT_MODES:
        raise ContactModeNotApplicable(f"unsupported contact mode {contact_mode!r}")
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be finite and positive")
    if not (math.isfinite(certificate_tolerance) and certificate_tolerance > 0.0):
        raise ValueError("certificate_tolerance must be finite and positive")
    inverse = ChainKKTInverse(H, A, stretch_compliance)
    right = _vector(rhs, inverse.n, "rhs")
    offset = _vector(stretch_offset, inverse.m, "stretch_offset")
    jacobian = _finite_sparse(J, "J").tocsr()
    if jacobian.shape[1] != inverse.n:
        raise ValueError("J column count must equal H dimension")
    b = _vector(contact_offset, jacobian.shape[0], "contact_offset")
    free, free_stretch = inverse.free(right, offset)
    response, response_stretch = inverse.response(jacobian.T.toarray())
    W0 = np.asarray(jacobian @ response, dtype=np.float64)
    W0 = 0.5 * (W0 + W0.T)
    law_info: dict[str, Any] = {}

    if contact_mode == FRICTIONLESS:
        compliance = _nonnegative_vector(
            contact_compliance, jacobian.shape[0], "contact_compliance"
        )
        W = W0 + np.diag(compliance)
        endpoint = b + np.asarray(jacobian @ free, dtype=np.float64)
        force, complementarity_gap, active = _enumerate_frictionless_lcp(
            W, endpoint, contact_limit=exact_contact_limit,
            tolerance=certificate_tolerance,
        )
        mu = np.empty(0, dtype=np.float64)
        cone_violation = 0.0
    else:
        if jacobian.shape[0] % 3:
            raise ValueError("frictional J rows must be normal/tangent triplets")
        count = jacobian.shape[0] // 3
        if friction_mu is None:
            raise ValueError("frictional modes require friction_mu")
        mu = _vector(friction_mu, count, "friction_mu")
        raw_compliance = np.asarray(contact_compliance, dtype=np.float64)
        if raw_compliance.ndim == 1 and raw_compliance.size == count:
            compliance = np.repeat(raw_compliance, 3)
        else:
            compliance = _nonnegative_vector(
                contact_compliance, jacobian.shape[0], "contact_compliance"
            )
        W = W0 + np.diag(compliance)
        endpoint = b + np.asarray(jacobian @ free, dtype=np.float64)
        if contact_mode == STICKING:
            if np.any(mu < stick_mu):
                raise ContactModeNotApplicable(
                    "sticking mode requires every supplied contact to meet stick_mu"
                )
            force, complementarity_gap, active = _enumerate_sticking_sets(
                W, endpoint, mu, contact_limit=exact_contact_limit,
                tolerance=certificate_tolerance,
            )
            cone_violation = _cone_violation(force, mu)
            law_info = _desaxce_full_certificates(force, W, endpoint, mu)
            law_info.update({
                "method": "certified high-mu sticking active set",
                "sticking_contact_indices": list(active),
                "moderate_friction_contact_indices": [],
                "sliding_contact_indices": [],
            })
        else:
            if not (
                math.isfinite(desaxce_tolerance) and desaxce_tolerance > 0.0
                and desaxce_max_iterations > 0
            ):
                raise ValueError("invalid De Saxce tolerance or iteration budget")
            force, active, law_info = _solve_desaxce_with_stick_condensation(
                W,
                endpoint,
                mu,
                stick_mu=stick_mu,
                contact_limit=exact_contact_limit,
                tolerance=desaxce_tolerance,
                max_iterations=desaxce_max_iterations,
            )
            complementarity_gap = law_info["corrected_velocity"]
            cone_violation = float(law_info["primal_cone_violation"])

    direction = free + response @ force
    stretch_force = free_stretch + response_stretch @ force
    stationarity = (
        inverse.H @ direction + inverse.A.T @ stretch_force
        - right - jacobian.T @ force
    )
    stretch_residual = offset + inverse.A @ direction - inverse.compliance * stretch_force
    physical_contact = b + np.asarray(jacobian @ direction, dtype=np.float64)
    compliant_contact = physical_contact + compliance * force
    scale = max(
        1.0,
        float(np.max(np.abs(right), initial=0.0)),
        float(np.max(np.abs(force), initial=0.0)),
        float(np.max(np.abs(stretch_force), initial=0.0)),
    )
    stationarity_inf = float(np.max(np.abs(stationarity), initial=0.0))
    stretch_inf = float(np.max(np.abs(stretch_residual), initial=0.0))
    if contact_mode == FRICTIONLESS:
        certified_rows = list(active)
    elif contact_mode == STICKING:
        certified_rows = [3 * i + j for i in active for j in range(3)]
    else:
        # Sliding tangential velocity is intentionally nonzero.  Its normal
        # contact velocity, not all three components, is the direct gap gate;
        # the full friction law is certified by the natural map below.
        certified_rows = [3 * i for i in active]
    contact_inf = (
        float(np.max(np.abs(compliant_contact[certified_rows]), initial=0.0))
        if certified_rows else 0.0
    )
    natural_map_inf = float(law_info.get("natural_map_inf", 0.0))
    primal_cone = float(law_info.get("primal_cone_violation", cone_violation))
    dual_cone = float(law_info.get("dual_cone_violation", 0.0))
    complementarity_inf = float(law_info.get("complementarity_inf", 0.0))
    friction_work = float(law_info.get("friction_work", 0.0))
    physical_friction_work = (
        float(np.sum(
            force.reshape(-1, 3)[:, 1:]
            * physical_contact.reshape(-1, 3)[:, 1:]
        ))
        if contact_mode != FRICTIONLESS else 0.0
    )
    if (
        stationarity_inf > 20.0 * certificate_tolerance * scale
        or stretch_inf > 20.0 * certificate_tolerance * scale
        or contact_inf > 20.0 * certificate_tolerance * scale
        or cone_violation > 20.0 * certificate_tolerance * scale
        or natural_map_inf > max(
            20.0 * desaxce_tolerance * scale,
            20.0 * certificate_tolerance * scale,
        )
        or primal_cone > 20.0 * certificate_tolerance * scale
        or dual_cone > 20.0 * certificate_tolerance * scale
        or complementarity_inf > 20.0 * certificate_tolerance * scale
        or friction_work > 20.0 * certificate_tolerance * scale
    ):
        raise CoupledSolveError("coupled result failed the final KKT certificate")
    owner_end = owner_guard("coupled quadratic end")
    return CoupledResult(
        direction=np.asarray(direction, dtype=np.float64),
        stretch_force_N=np.asarray(stretch_force, dtype=np.float64),
        contact_force_N=np.asarray(force, dtype=np.float64),
        free_direction=np.asarray(free, dtype=np.float64),
        info={
            "schema": SCHEMA,
            "owner_line_end": owner_end,
            "contact_mode": contact_mode,
            "active_contact_indices": active,
            "stretch_constraint_count": inverse.m,
            "contact_count": (
                jacobian.shape[0] if contact_mode == FRICTIONLESS
                else jacobian.shape[0] // 3
            ),
            "stretch_mode": (
                HARD_STRETCH if np.all(inverse.compliance == 0.0)
                else FINITE_STRETCH
            ),
            "matrix_free_delassus_available": True,
            "dense_delassus_used_only_for_exact_small_active_set": True,
            "factor_solve_calls": inverse.factor_solve_calls,
            "stationarity_inf": stationarity_inf,
            "stretch_residual_inf_m": stretch_inf,
            "active_contact_residual_inf_m": contact_inf,
            "cone_violation_N": cone_violation,
            "natural_map_inf": natural_map_inf,
            "natural_map_relative_inf": float(
                law_info.get("natural_map_relative_inf", 0.0)
            ),
            "primal_cone_violation_N": primal_cone,
            "dual_cone_violation_m": dual_cone,
            "complementarity_inf_Nm": complementarity_inf,
            "friction_work_Nm": friction_work,
            "physical_friction_work_Nm": physical_friction_work,
            "friction_law": law_info,
            "contact_complementarity_gap": complementarity_gap,
            "physical_contact_value": physical_contact,
            "compliant_contact_value": compliant_contact,
            "physical_interval_dt": float(dt),
            "internal_time_substeps": 0,
            "linearized_full_dt_solve": True,
            "claim_boundary": (
                "exact small CPU linearized KKT solve; frictionless, certified "
                "high-mu sticking, or bounded dense non-associated De Saxce; "
                "no CCD, nonlinear convergence, large pile, GPU, or production claim"
            ),
        },
    )


@dataclass(frozen=True)
class PreparedChainLinearization:
    H_backbone: sparse.csc_matrix
    rhs_backbone: np.ndarray
    A: sparse.csr_matrix
    stretch_offset: np.ndarray
    stretch_compliance: np.ndarray
    full_system: bgn.BlockSystem
    stats: dict[str, Any]


def stretch_jacobian(
    plan: bgn.PreparedBlockPlan, pose: rc.PoseState
) -> tuple[sparse.csr_matrix, np.ndarray]:
    jb = bgn._joint_batch(plan, pose)
    joint_count = len(plan.data.joint_ids)
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for local in range(joint_count):
        for slot, block in (
            (int(plan.joint_parent_slot[local]), jb["Jpt"][local]),
            (int(plan.joint_child_slot[local]), jb["Jct"][local]),
        ):
            if slot < 0:
                continue
            row_parts.append(np.repeat(3 * local + np.arange(3), 6))
            col_parts.append(np.tile(6 * slot + np.arange(6), 3))
            value_parts.append(np.asarray(block, dtype=np.float64).reshape(-1))
    if value_parts:
        A = sparse.coo_matrix(
            (np.concatenate(value_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
            shape=(3 * joint_count, 6 * plan.n),
        ).tocsr()
        A.sum_duplicates()
    else:
        A = sparse.csr_matrix((0, 6 * plan.n), dtype=np.float64)
    return A, np.asarray(jb["C"], dtype=np.float64).reshape(-1)


def assemble_prepared_chain_linearization(
    plan: bgn.PreparedBlockPlan,
    pose: rc.PoseState,
    predicted: rc.PoseState,
    previous_raw: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    stretch_mode: str,
) -> PreparedChainLinearization:
    """Lift authored finite stretch out of BGN into an explicit chain KKT."""
    if stretch_mode not in (FINITE_STRETCH, HARD_STRETCH):
        raise ValueError(f"unknown stretch mode {stretch_mode!r}")
    full = bgn.assemble_blocks(plan, pose, predicted, previous_raw)
    H_full = bgn.sparse_matrix(plan, full).tocsc()
    A, C = stretch_jacobian(plan, pose)
    joint_count = len(plan.data.joint_ids)
    if joint_count == 0:
        raise ValueError("CCDSN prepared path requires a cable chain")
    previous_C = np.stack([
        np.asarray(previous_raw[int(joint)][0], dtype=np.float64)
        for joint in plan.data.joint_ids
    ]).reshape(-1)
    ke = np.asarray(plan.data.stretch_ke, dtype=np.float64)
    kd_dt = np.asarray(plan.data.stretch_kd, dtype=np.float64) / plan.dt
    coefficient = ke + kd_dt
    if np.any(~np.isfinite(coefficient)) or np.any(coefficient <= 0.0):
        raise ValueError("authored stretch coefficient must be finite and positive")
    repeated = np.repeat(coefficient, 3)
    y = repeated * C - np.repeat(kd_dt, 3) * previous_C
    H_stretch = A.T @ sparse.diags(repeated, format="csc") @ A
    rhs_stretch = -np.asarray(A.T @ y, dtype=np.float64).reshape(-1)
    H_backbone = (H_full - H_stretch).tocsc()
    H_backbone = (0.5 * (H_backbone + H_backbone.T)).tocsc()
    H_backbone.eliminate_zeros()
    rhs_backbone = np.asarray(full.rhs, dtype=np.float64).reshape(-1) - rhs_stretch
    if stretch_mode == FINITE_STRETCH:
        stretch_offset = y / repeated
        stretch_compliance = 1.0 / repeated
    else:
        stretch_offset = C
        stretch_compliance = np.zeros_like(C)
    return PreparedChainLinearization(
        H_backbone=H_backbone,
        rhs_backbone=rhs_backbone,
        A=A,
        stretch_offset=stretch_offset,
        stretch_compliance=stretch_compliance,
        full_system=full,
        stats={
            "stretch_mode": stretch_mode,
            "joint_count": joint_count,
            "full_hessian_nnz": int(H_full.nnz),
            "backbone_hessian_nnz": int(H_backbone.nnz),
            "stretch_jacobian_nnz": int(A.nnz),
        },
    )


def solve_prepared_linearization(
    plan: bgn.PreparedBlockPlan,
    pose: rc.PoseState,
    predicted: rc.PoseState,
    previous_raw: dict[int, tuple[np.ndarray, np.ndarray]],
    rows: contact_v1.ContactRows,
    *,
    stretch_mode: str,
    contact_mode: str,
    contact_compliance: float = 1.0e-9,
    exact_contact_limit: int = 10,
    stick_mu: float = 1.0e3,
    desaxce_tolerance: float = 1.0e-10,
    desaxce_max_iterations: int = 80,
) -> CoupledResult:
    prepared = assemble_prepared_chain_linearization(
        plan, pose, predicted, previous_raw, stretch_mode=stretch_mode
    )
    if rows.J.shape != (3 * rows.count, 6 * plan.n):
        raise ValueError("contact rows do not match prepared cable dimensions")
    if contact_mode == FRICTIONLESS:
        normal_rows = np.arange(0, 3 * rows.count, 3)
        J = rows.J[normal_rows]
        contact_offset = np.asarray(rows.b0, dtype=np.float64)[normal_rows]
        mu = None
    elif contact_mode in (STICKING, DESAXCE):
        J = rows.J
        contact_offset = rows.b0
        mu = rows.mu
    else:
        raise ContactModeNotApplicable(f"unsupported contact mode {contact_mode!r}")
    result = solve_coupled_quadratic(
        prepared.H_backbone,
        prepared.rhs_backbone,
        prepared.A,
        prepared.stretch_offset,
        prepared.stretch_compliance,
        J,
        contact_offset,
        contact_mode=contact_mode,
        contact_compliance=contact_compliance,
        friction_mu=mu,
        stick_mu=stick_mu,
        exact_contact_limit=exact_contact_limit,
        desaxce_tolerance=desaxce_tolerance,
        desaxce_max_iterations=desaxce_max_iterations,
        dt=plan.dt,
    )
    result.info["prepared_chain"] = prepared.stats
    return result


def solve_one_linearized_frame(
    scene: Any,
    rows: contact_v1.ContactRows,
    *,
    dt: float,
    plan: bgn.PreparedBlockPlan | None = None,
    stretch_mode: str = HARD_STRETCH,
    contact_mode: str = FRICTIONLESS,
    contact_compliance: float = 1.0e-9,
    exact_contact_limit: int = 10,
    stick_mu: float = 1.0e3,
    desaxce_tolerance: float = 1.0e-10,
    desaxce_max_iterations: int = 80,
) -> tuple[Any, dict[str, Any]]:
    """Predict one dt, solve one coupled linearization, return private state."""
    owner_start = owner_guard("one-frame start")
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be finite and positive")
    plan = bgn.prepare(scene, dt) if plan is None else plan
    plan.validate(scene, dt, full=True)
    caller_arrays = tuple(value.copy() for value in al._state_arrays(scene.state_0))
    caller_digest = _array_digest(*caller_arrays)
    qd_snapshot = rc._as_numpy(scene.state_0.body_qd, np.float32)
    original, predicted = bgn._predict_pose_fast(plan.data, scene.state_0, dt)
    previous_raw = bgn.raw_by_joint(plan, original)
    result = solve_prepared_linearization(
        plan,
        predicted,
        predicted,
        previous_raw,
        rows,
        stretch_mode=stretch_mode,
        contact_mode=contact_mode,
        contact_compliance=contact_compliance,
        exact_contact_limit=exact_contact_limit,
        stick_mu=stick_mu,
        desaxce_tolerance=desaxce_tolerance,
        desaxce_max_iterations=desaxce_max_iterations,
    )
    final_pose = bgn._retract_fast(plan.data, predicted, result.direction, 1.0)
    private = scene.model.state()
    private.assign(scene.state_0)
    al._commit_pose(plan, private, original, final_pose, qd_snapshot)
    after_arrays = al._state_arrays(scene.state_0)
    if (
        _array_digest(*after_arrays) != caller_digest
        or not all(np.array_equal(a, b) for a, b in zip(caller_arrays, after_arrays))
    ):
        raise CoupledSolveError("CCDSN mutated its caller")
    if not rc.state_finite(private):
        raise CoupledSolveError("CCDSN private result is non-finite")
    owner_end = owner_guard("one-frame end")
    metadata = {
        "schema": SCHEMA,
        "owner_line_start": owner_start,
        "owner_line_end": owner_end,
        "caller_state_sha256": caller_digest,
        "caller_state_unchanged": True,
        "physical_advances": 1,
        "predictor_evaluations": 1,
        "internal_time_substeps": 0,
        "nonlinear_outer_iterations": 1,
        "dt": float(dt),
        "coupled": result.info,
        "claim_boundary": (
            "one full-dt predictor plus one coupled linearized correction on a "
            "private CPU state; caller supplies already-linearized contact rows; "
            "no collision discovery/CCD or nonlinear convergence claim"
        ),
    }
    return private, metadata
