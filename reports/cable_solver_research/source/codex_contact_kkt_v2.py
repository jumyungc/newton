"""Contact KKT v2: exact sticking fast path plus associated-cone fallback.

The v1 research solver uses projected FISTA on a product Coulomb cone.  That
is a useful general baseline but is needlessly slow when the authored mixed
friction coefficient makes a contact numerically sticking.  This module adds
an exact active-set solve for that regime:

    (J H^-1 J^T + c I) lambda = -(b + J H^-1 rhs)

with row equilibration, normal-force active-set pruning, and an explicit cone
admissibility check.  Contacts that are not proven sticking fall back to the
v1 cone QP; nothing is silently clamped or reclassified.

This is add-only research code.  It advances one physical dt exactly once and
does not mutate its caller.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import solve as dense_solve
from scipy.sparse.linalg import splu
import warp as wp

from bench.global_cable import codex_bgn_contact_kkt as v1
from bench.global_cable import codex_bgn_global as bgn
from bench.global_cable import codex_rc_portfolio as pf


SCHEMA = "codex-coupled-cable-contact-kkt/v2"
EXPECTED_OWNER = (
    "OWNER: Team Codex-Remote — 2026-07-05T11:40Z — "
    "coupled long-chain/contact solver invention round"
)
DEFAULT_STICK_MU = 1.0e3
DEFAULT_DIRECT_CONTACT_LIMIT = 256
CONTACT_LINE_SEARCH_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
CONTACT_PENETRATION_GATE_M = 5.0e-6
CONTACT_MIN_POSITIVE_ALPHA = 2.0 ** -20
CONTACT_BISECTION_ITERS = 12


class StickyNotApplicable(RuntimeError):
    """The active contacts are not certifiably inside the sticking regime."""


def _adaptive_collision_safe_search(
    evaluate: Callable[[float], dict[str, Any]],
    *,
    penetration_target_m: float = CONTACT_PENETRATION_GATE_M,
    initial_alphas: tuple[float, ...] = CONTACT_LINE_SEARCH_ALPHAS,
    min_positive_alpha: float = CONTACT_MIN_POSITIVE_ALPHA,
    bisection_iters: int = CONTACT_BISECTION_ITERS,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    """Find the largest sampled endpoint that satisfies the hard gap gate.

    The fixed v2 schedule stopped at alpha=1/32.  That can reject an otherwise
    useful Newton direction merely because its safe interval is narrower.  We
    retain those historical trial points, continue geometric backtracking to
    ``min_positive_alpha``, and then bisect the first unsafe/safe bracket.

    Every accepted endpoint is re-collided by ``evaluate`` and independently
    certified against the existing 5 um gate.  This certifies the endpoint,
    not the continuous swept path; proper CCD/OGC is still required for that.
    If even alpha=0 is outside the gate, no alpha can be *certified* by
    backtracking from the current pose.  In that recovery-only case the least
    penetrating finite sample is returned with ``endpoint_safe=False`` so the
    unchanged frame pass gate still fails closed.
    """
    if not (math.isfinite(penetration_target_m) and penetration_target_m >= 0.0):
        raise ValueError("penetration target must be finite and nonnegative")
    if penetration_target_m > CONTACT_PENETRATION_GATE_M:
        raise ValueError("penetration target may not relax the 5 um frame gate")
    if not (
        math.isfinite(min_positive_alpha)
        and 0.0 < min_positive_alpha < 1.0
    ):
        raise ValueError("minimum positive alpha must lie strictly inside (0, 1)")
    if bisection_iters < 0:
        raise ValueError("bisection iteration count must be nonnegative")
    if not initial_alphas:
        raise ValueError("at least one initial alpha is required")
    seeds = tuple(float(alpha) for alpha in initial_alphas)
    if any(
        not math.isfinite(alpha) or not (0.0 < alpha <= 1.0)
        for alpha in seeds
    ):
        raise ValueError("initial alphas must be finite and lie inside (0, 1]")
    if any(left <= right for left, right in zip(seeds, seeds[1:])):
        raise ValueError("initial alphas must be strictly decreasing")

    evaluated: list[dict[str, Any]] = []
    cache: dict[float, dict[str, Any]] = {}

    def sample(alpha: float) -> dict[str, Any]:
        alpha = float(alpha)
        if alpha in cache:
            return cache[alpha]
        trial = dict(evaluate(alpha))
        penetration = float(trial["maximum_penetration_m"])
        objective = float(trial["cable_objective"])
        finite = bool(
            trial.get("finite", True)
            and math.isfinite(penetration)
            and math.isfinite(objective)
        )
        trial.update({
            "alpha": alpha,
            "finite": finite,
            "maximum_penetration_m": penetration,
            "cable_objective": objective,
            "safe_for_penetration_gate": bool(
                finite and penetration <= penetration_target_m
            ),
        })
        cache[alpha] = trial
        evaluated.append(trial)
        return trial

    safe: dict[str, Any] | None = None
    unsafe_upper: dict[str, Any] | None = None
    for alpha in seeds:
        trial = sample(alpha)
        if trial["safe_for_penetration_gate"]:
            safe = trial
            break
        unsafe_upper = trial

    # Preserve the historical schedule, then continue far enough to resolve a
    # narrow feasible interval instead of silently treating 1/32 as zero.
    alpha = seeds[-1]
    while safe is None and alpha > min_positive_alpha:
        next_alpha = max(min_positive_alpha, 0.5 * alpha)
        if next_alpha == alpha:
            break
        trial = sample(next_alpha)
        if trial["safe_for_penetration_gate"]:
            safe = trial
            break
        unsafe_upper = trial
        alpha = next_alpha

    zero = None
    if safe is None:
        zero = sample(0.0)
        if zero["safe_for_penetration_gate"]:
            safe = zero

    refinement_count = 0
    # The lower endpoint is certified safe and the upper endpoint is sampled
    # unsafe.  Bisection only moves the accepted endpoint after recollision.
    if safe is not None and unsafe_upper is not None and safe["alpha"] < unsafe_upper["alpha"]:
        low = safe
        high = unsafe_upper
        for _ in range(bisection_iters):
            midpoint = 0.5 * (float(low["alpha"]) + float(high["alpha"]))
            if midpoint == low["alpha"] or midpoint == high["alpha"]:
                break
            trial = sample(midpoint)
            refinement_count += 1
            if trial["safe_for_penetration_gate"]:
                low = trial
            else:
                high = trial
        safe = low

    finite_trials = [trial for trial in evaluated if trial["finite"]]
    if safe is not None:
        selected = safe
        status = (
            "full_step_safe" if safe["alpha"] == 1.0 else
            "zero_step_only" if safe["alpha"] == 0.0 else
            "backtracked_endpoint_safe"
        )
    elif finite_trials:
        # Recovery only.  This keeps legacy behavior available, while the
        # explicit unsafe status and unchanged frame gate prevent promotion.
        selected = min(finite_trials, key=lambda trial: (
            trial["maximum_penetration_m"],
            trial["cable_objective"],
            -trial["alpha"],
        ))
        status = "no_safe_endpoint"
    else:
        selected = None
        status = "no_finite_endpoint"

    public_trials = [{
        "alpha": trial["alpha"],
        "finite": trial["finite"],
        "maximum_penetration_m": trial["maximum_penetration_m"],
        "active_rows": trial.get("active_rows"),
        "cable_objective": trial["cable_objective"],
        "safe_for_penetration_gate": trial["safe_for_penetration_gate"],
    } for trial in evaluated]
    diagnostics = {
        "status": status,
        "penetration_target_m": penetration_target_m,
        "endpoint_safe": bool(
            selected is not None and selected["safe_for_penetration_gate"]
        ),
        "selected_alpha": None if selected is None else selected["alpha"],
        "selected_maximum_penetration_m": (
            None if selected is None else selected["maximum_penetration_m"]
        ),
        "evaluations": len(evaluated),
        "bisection_iterations": refinement_count,
        "minimum_positive_alpha": min_positive_alpha,
        "continuous_path_certified": False,
        "claim_boundary": (
            "collision is re-evaluated at every accepted endpoint; this is "
            "not continuous collision detection and does not certify the "
            "swept path"
        ),
    }
    return selected, public_trials, diagnostics


def owner_guard(where: str) -> str:
    line = v1.DOC.read_text(encoding="utf-8").splitlines()[0]
    if line != EXPECTED_OWNER:
        raise RuntimeError(f"owner mismatch before {where}: {line!r}")
    return line


def _candidate_indices(factor: Any, flat_rhs: np.ndarray,
                       rows: v1.ContactRows) -> tuple[np.ndarray, np.ndarray]:
    free = factor.solve(flat_rhs)
    endpoint = rows.b0 + rows.J @ free
    candidate = np.flatnonzero(endpoint[0::3] <= v1.ACTIVE_GAP_M)
    return free, candidate


def _empty_result(H: sparse.csc_matrix, flat_rhs: np.ndarray,
                  free: np.ndarray, rows: v1.ContactRows) -> v1.DualResult:
    stationarity = H @ free - flat_rhs
    return v1.DualResult(np.empty(0), free, free, {
        "solver_mode": "no_active_contact",
        "reported_contact_count": rows.reported_count,
        "linearized_contact_count": 0,
        "dual_iterations": 0,
        "stationarity_inf": float(np.max(np.abs(stationarity), initial=0.0)),
        "projected_kkt_inf": 0.0,
        "cone_violation": 0.0,
        "converged": True,
        "normal_force_max_N": 0.0,
    })


def solve_sticking_active_set(
    H: sparse.csc_matrix,
    rhs: np.ndarray,
    rows: v1.ContactRows,
    *,
    compliance: float = v1.CONTACT_COMPLIANCE_M_PER_N,
    stick_mu: float = DEFAULT_STICK_MU,
    direct_contact_limit: int = DEFAULT_DIRECT_CONTACT_LIMIT,
    normal_tolerance: float = 1.0e-9,
    cone_tolerance: float = 1.0e-8,
    residual_tolerance: float = 5.0e-9,
) -> v1.DualResult:
    """Solve contacts as sticking equalities when that mode is certified.

    The solve is dense only in contact space and is therefore intentionally
    bounded.  A future large-pile path uses the same operator matrix-free.
    """
    if not (math.isfinite(compliance) and compliance > 0.0):
        raise ValueError("compliance must be finite and positive")
    if not (math.isfinite(stick_mu) and stick_mu >= 0.0):
        raise ValueError("stick_mu must be finite and nonnegative")
    H = H.tocsc()
    flat_rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    factor = splu(H, permc_spec="COLAMD")
    free, candidate = _candidate_indices(factor, flat_rhs, rows)
    if len(candidate) == 0:
        return _empty_result(H, flat_rhs, free, rows)
    if len(candidate) > direct_contact_limit:
        raise StickyNotApplicable(
            f"{len(candidate)} contacts exceed direct limit {direct_contact_limit}"
        )
    if np.any(rows.mu[candidate] < stick_mu):
        raise StickyNotApplicable("not every active contact is high-friction")

    active_indices = candidate.copy()
    dropped: list[int] = []
    condition = math.inf
    row_scale_min = math.inf
    row_scale_max = 0.0
    solve_residual = math.inf
    active = None
    lam = np.empty(0)
    W = np.empty((0, 0))
    b = np.empty(0)
    max_active_set_passes = len(active_indices) + 1
    for active_pass in range(max_active_set_passes):
        if len(active_indices) == 0:
            return _empty_result(H, flat_rhs, free, rows)
        active = v1.select_rows(rows, active_indices)
        response = factor.solve(active.J.T.toarray())
        W = np.asarray(active.J @ response, dtype=np.float64)
        W = 0.5 * (W + W.T)
        W.flat[:: len(W) + 1] += compliance
        b = np.asarray(active.b0 + active.J @ free, dtype=np.float64)
        diagonal = np.diag(W)
        if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)):
            raise RuntimeError("non-finite sticking Delassus system")
        if np.any(diagonal <= 0.0):
            raise RuntimeError("non-positive sticking Delassus diagonal")
        scale = np.sqrt(diagonal)
        row_scale_min = min(row_scale_min, float(np.min(scale)))
        row_scale_max = max(row_scale_max, float(np.max(scale)))
        scaled = W / scale[:, None] / scale[None, :]
        scaled_rhs = -b / scale
        try:
            z = dense_solve(
                scaled, scaled_rhs, assume_a="sym", check_finite=False
            )
        except Exception:
            z, *_ = np.linalg.lstsq(scaled, scaled_rhs, rcond=1.0e-12)
        lam = z / scale
        solve_residual = float(
            np.max(np.abs(W @ lam + b), initial=0.0)
        )
        condition = float(np.linalg.cond(scaled))
        normal = lam[0::3]
        negative = np.flatnonzero(normal < -normal_tolerance)
        if len(negative) == 0:
            break
        dropped.extend(int(active_indices[index]) for index in negative)
        active_indices = np.delete(active_indices, negative)
    else:  # pragma: no cover - finite active set must terminate
        raise RuntimeError("sticking normal active set did not terminate")

    assert active is not None
    lam3 = lam.reshape(-1, 3)
    tangent_norm = np.linalg.norm(lam3[:, 1:], axis=1)
    cone_margin = active.mu * lam3[:, 0] - tangent_norm
    if np.any(lam3[:, 0] < -normal_tolerance):
        raise StickyNotApplicable("negative normal force after active-set solve")
    if np.any(cone_margin < -cone_tolerance):
        raise StickyNotApplicable("unconstrained equality impulse is outside Coulomb cone")
    scale_residual = max(1.0, float(np.max(np.abs(b), initial=0.0)))
    if solve_residual > residual_tolerance * scale_residual:
        raise RuntimeError(
            f"sticking equality residual too large: {solve_residual}"
        )

    direction = factor.solve(flat_rhs + active.J.T @ lam)
    stationarity = H @ direction - flat_rhs - active.J.T @ lam
    physical = np.asarray(active.b0 + active.J @ direction, dtype=np.float64)
    compliant = physical + compliance * lam
    normal_force = lam3[:, 0]
    normal_gap = physical[0::3]
    tangent_slip = physical.reshape(-1, 3)[:, 1:]
    hard_comp = normal_force * normal_gap
    compliant_comp = normal_force * compliant[0::3]
    return v1.DualResult(lam, direction, free, {
        "solver_mode": "direct_sticking_active_set",
        "reported_contact_count": rows.reported_count,
        "linearized_contact_count": active.count,
        "culled_positive_gap_count": int(rows.count - len(candidate)),
        "dropped_negative_normal_contacts": sorted(dropped),
        "active_set_passes": active_pass + 1,
        "dual_iterations": active_pass + 1,
        "dual_objective": float(0.5 * lam @ (W @ lam) + b @ lam),
        "estimated_lipschitz": float(np.linalg.eigvalsh(W)[-1]),
        "equilibrated_condition_number": condition,
        "row_scale_min": row_scale_min,
        "row_scale_max": row_scale_max,
        "projected_kkt_inf": solve_residual,
        "equality_residual_inf": solve_residual,
        "cone_violation": v1.cone_violation(lam, active.mu),
        "minimum_stick_cone_margin_N": float(np.min(cone_margin)),
        "stationarity_inf": float(np.max(np.abs(stationarity), initial=0.0)),
        "linearized_minimum_hard_gap_m": float(np.min(normal_gap)),
        "linearized_maximum_penetration_m": float(
            max(0.0, -float(np.min(normal_gap)))
        ),
        "hard_complementarity_max_Nm": float(
            np.max(np.abs(hard_comp), initial=0.0)
        ),
        "compliant_complementarity_max_Nm": float(
            np.max(np.abs(compliant_comp), initial=0.0)
        ),
        "contact_compliance_m_per_N": compliance,
        "normal_force_min_N": float(np.min(normal_force, initial=0.0)),
        "normal_force_max_N": float(np.max(normal_force, initial=0.0)),
        "friction_mu_min": float(np.min(active.mu, initial=0.0)),
        "friction_mu_median": float(np.median(active.mu)),
        "friction_mu_max": float(np.max(active.mu, initial=0.0)),
        "effectively_sticky_mu_rows_ge_1e3": int(
            np.count_nonzero(active.mu >= DEFAULT_STICK_MU)
        ),
        "friction_work_J": float(np.sum(lam3[:, 1:] * tangent_slip)),
        "normal_contact_work_J": float(np.sum(normal_force * normal_gap)),
        "converged": True,
        "selected_contact_indices": active_indices.tolist(),
        "claim_boundary": (
            "exact equality solve only after the unconstrained impulse is "
            "verified inside the authored Coulomb cone"
        ),
    })


def _dense_cone_qp(
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    *,
    max_iters: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Projected accelerated solve with cone-preserving block equilibration."""
    contacts = len(mu)
    if contacts == 0:
        return np.empty(0), {
            "iterations": 0, "projected_kkt_inf": 0.0, "converged": True,
        }
    W = 0.5 * (np.asarray(W, dtype=np.float64) + np.asarray(W).T)
    b = np.asarray(b, dtype=np.float64)
    block_scale = np.empty(contacts, dtype=np.float64)
    for contact in range(contacts):
        block = W[3 * contact:3 * contact + 3, 3 * contact:3 * contact + 3]
        block_scale[contact] = math.sqrt(max(float(np.max(np.diag(block))), 1.0e-18))
    scale = np.repeat(block_scale, 3)
    Ws = W / scale[:, None] / scale[None, :]
    bs = b / scale
    eigenvalues = np.linalg.eigvalsh(Ws)
    lipschitz = max(float(eigenvalues[-1]), 1.0e-12)
    z = np.zeros(len(b), dtype=np.float64)
    accelerated = z.copy()
    momentum = 1.0
    objective = 0.0
    projected = math.inf
    history: list[dict[str, float]] = []
    converged = False
    for iteration in range(max_iters):
        gradient = Ws @ accelerated + bs
        proposal = v1.project_coulomb(
            accelerated - gradient / lipschitz, mu
        )
        proposal_objective = (
            0.5 * float(proposal @ (Ws @ proposal)) + float(bs @ proposal)
        )
        if iteration and proposal_objective > objective + 1.0e-13 * max(1.0, abs(objective)):
            accelerated = z.copy()
            momentum = 1.0
            gradient = Ws @ accelerated + bs
            proposal = v1.project_coulomb(
                accelerated - gradient / lipschitz, mu
            )
            proposal_objective = (
                0.5 * float(proposal @ (Ws @ proposal)) + float(bs @ proposal)
            )
        proposal_gradient = Ws @ proposal + bs
        mapping = lipschitz * (
            proposal - v1.project_coulomb(
                proposal - proposal_gradient / lipschitz, mu
            )
        )
        projected = float(np.max(np.abs(mapping), initial=0.0))
        if iteration < 4 or (iteration + 1) % 10 == 0:
            history.append({
                "iteration": float(iteration + 1),
                "objective": proposal_objective,
                "projected_kkt_inf_scaled": projected,
            })
        old_momentum = momentum
        momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        accelerated = proposal + (
            (old_momentum - 1.0) / momentum
        ) * (proposal - z)
        z = proposal
        objective = proposal_objective
        if projected <= tolerance * max(1.0, float(np.max(np.abs(bs), initial=0.0))):
            converged = True
            break
    lam = z / scale
    original_gradient = W @ lam + b
    # Report the original-dimensional fixed-point mapping with a safe exact
    # spectral step.  This is comparable to v1's projected KKT quantity.
    original_lipschitz = max(float(np.linalg.eigvalsh(W)[-1]), 1.0e-12)
    original_mapping = original_lipschitz * (
        lam - v1.project_coulomb(
            lam - original_gradient / original_lipschitz, mu
        )
    )
    return lam, {
        "iterations": iteration + 1,
        "projected_kkt_inf_scaled": projected,
        "projected_kkt_inf": float(np.max(np.abs(original_mapping), initial=0.0)),
        "lipschitz_scaled": lipschitz,
        "condition_scaled": float(eigenvalues[-1] / max(eigenvalues[0], 1.0e-18)),
        "block_scale_min": float(np.min(block_scale)),
        "block_scale_max": float(np.max(block_scale)),
        "history": history,
        "converged": converged,
    }


def solve_mixed_stick_slide_active_set(
    H: sparse.csc_matrix,
    rhs: np.ndarray,
    rows: v1.ContactRows,
    *,
    compliance: float = v1.CONTACT_COMPLIANCE_M_PER_N,
    stick_mu: float = DEFAULT_STICK_MU,
    direct_contact_limit: int = DEFAULT_DIRECT_CONTACT_LIMIT,
    max_iters: int = 400,
    tolerance: float = 5.0e-9,
    normal_tolerance: float = 1.0e-9,
    cone_tolerance: float = 1.0e-8,
) -> v1.DualResult:
    """Condense certified sticking rows, solve only the sliding cone Schur."""
    H = H.tocsc()
    flat_rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    factor = splu(H, permc_spec="COLAMD")
    free, candidate = _candidate_indices(factor, flat_rhs, rows)
    if len(candidate) == 0:
        return _empty_result(H, flat_rhs, free, rows)
    if len(candidate) > direct_contact_limit:
        raise StickyNotApplicable(
            f"{len(candidate)} contacts exceed mixed direct limit {direct_contact_limit}"
        )
    if not np.any(rows.mu[candidate] >= stick_mu):
        raise StickyNotApplicable("mixed condensation has no sticking contacts")

    active_indices = candidate.copy()
    dropped: list[int] = []
    cone_info: dict[str, Any] = {}
    active = None
    lam = np.empty(0)
    b = np.empty(0)
    W = np.empty((0, 0))
    high_contacts = np.empty(0, dtype=np.int64)
    low_contacts = np.empty(0, dtype=np.int64)
    for active_pass in range(len(active_indices) + 1):
        if len(active_indices) == 0:
            return _empty_result(H, flat_rhs, free, rows)
        active = v1.select_rows(rows, active_indices)
        response = factor.solve(active.J.T.toarray())
        W = np.asarray(active.J @ response, dtype=np.float64)
        W = 0.5 * (W + W.T)
        W.flat[:: len(W) + 1] += compliance
        b = np.asarray(active.b0 + active.J @ free, dtype=np.float64)
        high_contacts = np.flatnonzero(active.mu >= stick_mu)
        low_contacts = np.flatnonzero(active.mu < stick_mu)
        high_rows = (
            3 * high_contacts[:, None] + np.arange(3)[None, :]
        ).reshape(-1)
        low_rows = (
            3 * low_contacts[:, None] + np.arange(3)[None, :]
        ).reshape(-1)
        Wss = W[np.ix_(high_rows, high_rows)]
        bs = b[high_rows]
        if len(low_rows):
            Wsc = W[np.ix_(high_rows, low_rows)]
            Wcs = Wsc.T
            Wcc = W[np.ix_(low_rows, low_rows)]
            solved_bs = dense_solve(Wss, bs, assume_a="sym", check_finite=False)
            solved_cross = dense_solve(
                Wss, Wsc, assume_a="sym", check_finite=False
            )
            reduced_W = Wcc - Wcs @ solved_cross
            reduced_b = b[low_rows] - Wcs @ solved_bs
            low_lambda, cone_info = _dense_cone_qp(
                reduced_W, reduced_b, active.mu[low_contacts],
                max_iters=max_iters, tolerance=tolerance,
            )
            high_lambda = -dense_solve(
                Wss, bs + Wsc @ low_lambda,
                assume_a="sym", check_finite=False,
            )
        else:
            low_lambda = np.empty(0)
            high_lambda = -dense_solve(
                Wss, bs, assume_a="sym", check_finite=False
            )
            cone_info = {
                "iterations": 0, "projected_kkt_inf": 0.0,
                "projected_kkt_inf_scaled": 0.0, "converged": True,
                "history": [],
            }
        lam = np.zeros(len(b), dtype=np.float64)
        lam[high_rows] = high_lambda
        lam[low_rows] = low_lambda
        negative_high = high_contacts[
            np.flatnonzero(high_lambda.reshape(-1, 3)[:, 0] < -normal_tolerance)
        ]
        if len(negative_high) == 0:
            break
        drop_positions = np.asarray(negative_high, dtype=np.int64)
        dropped.extend(int(active_indices[index]) for index in drop_positions)
        active_indices = np.delete(active_indices, drop_positions)
    else:  # pragma: no cover
        raise RuntimeError("mixed sticking active set did not terminate")

    assert active is not None
    lam3 = lam.reshape(-1, 3)
    high_lambda3 = lam3[high_contacts]
    high_margin = (
        active.mu[high_contacts] * high_lambda3[:, 0]
        - np.linalg.norm(high_lambda3[:, 1:], axis=1)
    )
    if np.any(high_margin < -cone_tolerance):
        raise StickyNotApplicable("condensed sticking impulse left authored cone")
    if not cone_info.get("converged", False):
        raise StickyNotApplicable("reduced sliding cone solve did not converge")

    direction = factor.solve(flat_rhs + active.J.T @ lam)
    stationarity = H @ direction - flat_rhs - active.J.T @ lam
    physical = np.asarray(active.b0 + active.J @ direction, dtype=np.float64)
    compliant = physical + compliance * lam
    full_gradient = W @ lam + b
    sticky_residual = float(
        np.max(np.abs(full_gradient.reshape(-1, 3)[high_contacts]), initial=0.0)
    )
    projected_kkt = max(
        sticky_residual, float(cone_info["projected_kkt_inf"])
    )
    normal_force = lam3[:, 0]
    normal_gap = physical[0::3]
    tangent_slip = physical.reshape(-1, 3)[:, 1:]
    return v1.DualResult(lam, direction, free, {
        "solver_mode": "mixed_stick_condensation_plus_dense_cone",
        "reported_contact_count": rows.reported_count,
        "linearized_contact_count": active.count,
        "culled_positive_gap_count": int(rows.count - len(candidate)),
        "sticking_contact_count": int(len(high_contacts)),
        "sliding_candidate_contact_count": int(len(low_contacts)),
        "dropped_negative_normal_contacts": sorted(dropped),
        "active_set_passes": active_pass + 1,
        "dual_iterations": int(cone_info["iterations"]),
        "dual_objective": float(0.5 * lam @ (W @ lam) + b @ lam),
        "projected_kkt_inf": projected_kkt,
        "sticky_equality_residual_inf": sticky_residual,
        "reduced_cone": cone_info,
        "cone_violation": v1.cone_violation(lam, active.mu),
        "minimum_stick_cone_margin_N": float(np.min(high_margin)),
        "stationarity_inf": float(np.max(np.abs(stationarity), initial=0.0)),
        "linearized_minimum_hard_gap_m": float(np.min(normal_gap)),
        "linearized_maximum_penetration_m": float(
            max(0.0, -float(np.min(normal_gap)))
        ),
        "hard_complementarity_max_Nm": float(
            np.max(np.abs(normal_force * normal_gap), initial=0.0)
        ),
        "compliant_complementarity_max_Nm": float(
            np.max(np.abs(normal_force * compliant[0::3]), initial=0.0)
        ),
        "contact_compliance_m_per_N": compliance,
        "normal_force_min_N": float(np.min(normal_force, initial=0.0)),
        "normal_force_max_N": float(np.max(normal_force, initial=0.0)),
        "friction_mu_min": float(np.min(active.mu, initial=0.0)),
        "friction_mu_median": float(np.median(active.mu)),
        "friction_mu_max": float(np.max(active.mu, initial=0.0)),
        "effectively_sticky_mu_rows_ge_1e3": int(len(high_contacts)),
        "friction_work_J": float(np.sum(lam3[:, 1:] * tangent_slip)),
        "normal_contact_work_J": float(np.sum(normal_force * normal_gap)),
        "converged": True,
        "selected_contact_indices": active_indices.tolist(),
        "claim_boundary": (
            "high-mu equalities are Schur-condensed only after force/cone "
            "verification; remaining contacts retain associated-cone QP "
            "pending the De Saxce residual"
        ),
    })


def solve_dual_hybrid(
    H: sparse.csc_matrix,
    rhs: np.ndarray,
    rows: v1.ContactRows,
    *,
    compliance: float = v1.CONTACT_COMPLIANCE_M_PER_N,
    max_iters: int = v1.DEFAULT_DUAL_ITERS,
    stick_mu: float = DEFAULT_STICK_MU,
    direct_contact_limit: int = DEFAULT_DIRECT_CONTACT_LIMIT,
) -> v1.DualResult:
    """Use certified direct sticking when possible, else the v1 cone QP."""
    try:
        return solve_sticking_active_set(
            H, rhs, rows, compliance=compliance, stick_mu=stick_mu,
            direct_contact_limit=direct_contact_limit,
        )
    except StickyNotApplicable as exc:
        sticky_rejection = str(exc)
    try:
        result = solve_mixed_stick_slide_active_set(
            H, rhs, rows, compliance=compliance, stick_mu=stick_mu,
            direct_contact_limit=direct_contact_limit,
            max_iters=max(max_iters, 400),
        )
        result.info["all_stick_rejection"] = sticky_rejection
        return result
    except StickyNotApplicable as mixed_exc:
        result = v1.solve_dual_cone_qp(
            H, rhs, rows, compliance=compliance, max_iters=max_iters
        )
        result.info["solver_mode"] = "associated_cone_fista_fallback"
        result.info["sticky_rejection"] = sticky_rejection
        result.info["mixed_rejection"] = str(mixed_exc)
        return result


def solve_contact_frame(
    scene: Any,
    dt: float,
    *,
    plan: bgn.PreparedBlockPlan | None = None,
    max_outer_iters: int = v1.DEFAULT_OUTER_ITERS,
    max_dual_iters: int = v1.DEFAULT_DUAL_ITERS,
    compliance: float = v1.CONTACT_COMPLIANCE_M_PER_N,
    stick_mu: float = DEFAULT_STICK_MU,
    direct_contact_limit: int = DEFAULT_DIRECT_CONTACT_LIMIT,
) -> dict[str, Any]:
    """Advance one dt with SQP and the certified hybrid contact solve."""
    owner_guard("contact KKT v2 frame")
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be finite and positive")
    if scene.collision_pipeline is None:
        raise ValueError("contact solver requires a collision pipeline")
    source_state = scene.state_0
    q0 = v1._as_numpy(source_state.body_q, np.float32).copy()
    qd0 = v1._as_numpy(source_state.body_qd, np.float32).copy()
    f0 = (v1._as_numpy(source_state.body_f, np.float32).copy()
          if getattr(source_state, "body_f", None) is not None else None)
    caller_hash_before = v1.state_sha256(source_state)
    started = time.perf_counter_ns()
    plan = v1.prepare_contact(scene, dt) if plan is None else plan
    v1._validate_plan(plan, scene, dt)
    original, predicted = bgn._predict_pose_fast(plan.data, source_state, dt)
    previous_raw = bgn.raw_by_joint(plan, original) if len(plan.data.joint_ids) else {}
    pose = predicted.copy()
    scratch = scene.model.state()
    scratch.assign(source_state)
    contacts = scene.model.contacts(collision_pipeline=scene.collision_pipeline)
    iterations: list[dict[str, Any]] = []
    final_dual: v1.DualResult | None = None
    converged = False

    for outer in range(max_outer_iters):
        assembly_started = time.perf_counter_ns()
        system = v1._assemble(plan, pose, predicted, previous_raw)
        H = bgn.sparse_matrix(plan, system).tocsc()
        assembly_ms = (time.perf_counter_ns() - assembly_started) * 1.0e-6
        rows = v1.collision_rows(scene, plan, pose, original, scratch, contacts)
        before = v1._contact_metrics(rows)
        solve_started = time.perf_counter_ns()
        dual = solve_dual_hybrid(
            H, system.rhs, rows, compliance=compliance,
            max_iters=max_dual_iters, stick_mu=stick_mu,
            direct_contact_limit=direct_contact_limit,
        )
        solve_ms = (time.perf_counter_ns() - solve_started) * 1.0e-6
        direction = dual.direction.reshape(plan.n, 6)
        raw_translation_inf = float(np.max(np.abs(direction[:, :3]), initial=0.0))
        raw_rotation_inf = float(np.max(np.abs(direction[:, 3:]), initial=0.0))
        if not np.all(np.isfinite(direction)):
            raise RuntimeError("non-finite contact SQP direction")
        # Contact sets are discontinuous.  A direction that exactly resolves
        # the current rows may create new rows at alpha=1, so recollide every
        # trial.  Adaptive backtracking continues below the historical 1/32
        # floor and only labels an endpoint safe after it meets the unchanged
        # 5 um frame gate.
        def evaluate_trial(alpha: float) -> dict[str, Any]:
            trial_pose = bgn._retract_fast(plan.data, pose, direction, alpha)
            trial_rows = v1.collision_rows(
                scene, plan, trial_pose, original, scratch, contacts
            )
            trial_contact = v1._contact_metrics(trial_rows)
            trial_objective, trial_structural = v1._objective(
                plan, trial_pose, predicted, previous_raw
            )
            finite_trial = bool(
                math.isfinite(trial_objective)
                and math.isfinite(trial_contact["maximum_penetration_m"])
                and not trial_contact["overflow"]
            )
            return {
                "alpha": alpha,
                "finite": finite_trial,
                "maximum_penetration_m": trial_contact["maximum_penetration_m"],
                "active_rows": trial_contact["active_rows"],
                "cable_objective": float(trial_objective),
                "pose": trial_pose,
                "rows": trial_rows,
                "contact": trial_contact,
                "structural": trial_structural,
            }

        selected, trials, globalization = _adaptive_collision_safe_search(
            evaluate_trial,
            penetration_target_m=CONTACT_PENETRATION_GATE_M,
        )
        if selected is None:
            raise RuntimeError("contact-aware line search found no finite trial")
        alpha = float(selected["alpha"])
        pose = selected["pose"]
        after_rows = selected["rows"]
        after = selected["contact"]
        objective_after = float(selected["cable_objective"])
        structural_after = selected["structural"]
        translation_inf = alpha * raw_translation_inf
        rotation_inf = alpha * raw_rotation_inf
        iterations.append({
            "outer_iteration": outer,
            "cable_objective_before": float(system.objective),
            "cable_objective_after": float(objective_after),
            "contact_before": before,
            "contact_after": after,
            "translation_direction_inf_m": translation_inf,
            "rotation_direction_inf_rad": rotation_inf,
            "raw_translation_direction_inf_m": raw_translation_inf,
            "raw_rotation_direction_inf_rad": raw_rotation_inf,
            "selected_alpha": alpha,
            "contact_line_search": trials,
            "collision_safe_globalization": globalization,
            "dual": dual.info,
            "structural_after": structural_after,
            "assembly_ms": assembly_ms,
            "dual_solve_ms": solve_ms,
        })
        final_dual = dual
        if (
            translation_inf <= 2.0e-8
            and rotation_inf <= 2.0e-8
            and after["maximum_penetration_m"] <= max(
                5.0e-7, 4.0 * compliance * dual.info.get("normal_force_max_N", 0.0)
            )
            and dual.info["cone_violation"] <= 2.0e-7
            and dual.info["projected_kkt_inf"] <= 2.0e-5
        ):
            converged = True
            break

    if final_dual is None:
        raise RuntimeError("contact SQP ran no iterations")
    final_rows = v1.collision_rows(scene, plan, pose, original, scratch, contacts)
    final_contact = v1._contact_metrics(final_rows)
    body_q = bgn._body_q_from_pose_fast(plan.data, pose)
    body_qd = qd0.astype(np.float64)
    bodies = np.asarray(plan.data.dynamic_bodies, dtype=np.int64)
    body_qd[bodies, :3] = (pose.p_com[bodies] - original.p_com[bodies]) / dt
    delta_q = pf._q_normalize_batch(pf._q_mul_batch(
        pose.q[bodies], pf._q_conj_batch(original.q[bodies])
    ))
    body_qd[bodies, 3:] = pf._q_log_batch(delta_q) / dt
    output = scene.model.state()
    output.assign(source_state)
    output.body_q.assign(body_q.astype(np.float32))
    output.body_qd.assign(body_qd.astype(np.float32))
    wp.synchronize_device(scene.model.device)

    assigned = bgn._pose_from_state_fast(plan.data, output)
    reconstructed_linear = (assigned.p_com[bodies] - original.p_com[bodies]) / dt
    assigned_delta_q = pf._q_normalize_batch(pf._q_mul_batch(
        assigned.q[bodies], pf._q_conj_batch(original.q[bodies])
    ))
    reconstructed_angular = pf._q_log_batch(assigned_delta_q) / dt
    committed_qd = v1._as_numpy(output.body_qd, np.float64)[bodies]
    velocity_error = max(
        float(np.max(np.abs(committed_qd[:, :3] - reconstructed_linear), initial=0.0)),
        float(np.max(np.abs(committed_qd[:, 3:] - reconstructed_angular), initial=0.0)),
    )
    caller_hash_after = v1.state_sha256(source_state)
    caller_unchanged = bool(
        caller_hash_before == caller_hash_after
        and np.array_equal(q0, v1._as_numpy(source_state.body_q, np.float32))
        and np.array_equal(qd0, v1._as_numpy(source_state.body_qd, np.float32))
        and (f0 is None or np.array_equal(f0, v1._as_numpy(source_state.body_f, np.float32)))
    )
    finite = bool(
        np.all(np.isfinite(v1._as_numpy(output.body_q, np.float64)))
        and np.all(np.isfinite(v1._as_numpy(output.body_qd, np.float64)))
    )
    quaternion_error = float(np.max(np.abs(
        np.linalg.norm(v1._as_numpy(output.body_q, np.float64)[:, 3:7], axis=1) - 1.0
    ), initial=0.0))
    pass_gate = bool(
        caller_unchanged and finite and quaternion_error <= 3.0e-5
        and velocity_error <= 5.0e-5 and not final_contact["overflow"]
        and final_contact["maximum_penetration_m"] <= max(
            5.0e-6, 4.0 * compliance * final_dual.info.get("normal_force_max_N", 0.0)
        )
        and final_dual.info["stationarity_inf"] <= 1.0e-6
        and final_dual.info["cone_violation"] <= 1.0e-6
        and final_dual.info["projected_kkt_inf"] <= 2.0e-5
        and bool(final_dual.info["converged"])
    )
    return {
        "schema": SCHEMA,
        "state": output,
        "pass": pass_gate,
        "converged": converged,
        "method": "compact cable SQP + certified sticking active set / cone fallback",
        "time_advance_count": 1,
        "outer_iterations_are_time_substeps": False,
        "dt": float(dt),
        "contact_compliance_m_per_N": compliance,
        "iterations": iterations,
        "final_contact": final_contact,
        "final_kkt": final_dual.info,
        "caller_unchanged": caller_unchanged,
        "caller_state_sha256": caller_hash_before,
        "returned_state_sha256": v1.state_sha256(output),
        "velocity_reconstruction_inf": velocity_error,
        "state_finite": finite,
        "quaternion_norm_max_error": quaternion_error,
        "elapsed_ms": (time.perf_counter_ns() - started) * 1.0e-6,
        "claim_boundary": (
            "sticking is used only when the exact equality impulse lies inside "
            "the authored Coulomb cone; other contacts retain the v1 associated "
            "cone fallback pending the non-associated De Saxce path"
        ),
    }
