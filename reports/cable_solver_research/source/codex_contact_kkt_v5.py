"""Contact KKT v5: certificate-aware De Saxce refinement.

v4 exposed a scale mismatch: a normalized natural-map residual can pass while
an absolute cone-law certificate fails for a large impulse.  v5 retains every
v3 geometry and v4 physical-law decision, but tightens the natural-map solve
geometrically until both the requested normalized residual and all absolute
1e-8 certificates pass.  Tolerances only tighten; they are never relaxed.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
from scipy import sparse

from bench.global_cable import codex_bgn_contact_kkt as v1
from bench.global_cable import codex_contact_kkt_v2 as v2
from bench.global_cable import codex_contact_kkt_v3 as v3
from bench.global_cable import codex_contact_kkt_v4 as v4
from bench.global_cable import codex_desaxce as desaxce


SCHEMA = "codex-coupled-cable-contact-kkt/v5"
ABSOLUTE_CERTIFICATE_TOLERANCE = 1.0e-8
REFINEMENT_TOLERANCES = (5.0e-9, 5.0e-10, 5.0e-11, 5.0e-12, 5.0e-13, 5.0e-14)

StickyNotApplicable = v2.StickyNotApplicable
solve_sticking_active_set = v2.solve_sticking_active_set


def owner_guard(where: str) -> str:
    return v4.owner_guard(where)


def _law_audit(
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    impulse: np.ndarray,
    gamma: float,
    normalized_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    residual, aux = desaxce.natural_map(impulse, W, b, mu, gamma)
    velocity = np.asarray(aux["velocity"], dtype=np.float64)
    corrected = np.asarray(aux["corrected_velocity"], dtype=np.float64)
    scale = max(
        1.0,
        float(np.max(np.abs(impulse), initial=0.0)),
        gamma * float(np.max(np.abs(corrected), initial=0.0)),
    )
    natural_inf = float(np.max(np.abs(residual), initial=0.0))
    relative = natural_inf / scale
    impulse3 = np.asarray(impulse, dtype=np.float64).reshape(-1, 3)
    velocity3 = velocity.reshape(-1, 3)
    corrected3 = corrected.reshape(-1, 3)
    primal = desaxce.cone_violation(impulse, mu)
    dual = desaxce.dual_cone_violation(corrected, mu)
    complementarity = float(np.max(
        np.abs(np.sum(impulse3 * corrected3, axis=1)), initial=0.0
    ))
    dissipation = float(np.sum(impulse3[:, 1:] * velocity3[:, 1:]))
    checks = {
        "normalized_natural_map": relative <= normalized_tolerance,
        "absolute_primal_cone": primal <= absolute_tolerance,
        "absolute_dual_cone": dual <= absolute_tolerance,
        "absolute_complementarity": complementarity <= absolute_tolerance,
        "absolute_dissipation": dissipation <= absolute_tolerance,
    }
    return {
        "natural_map_inf": natural_inf,
        "natural_map_relative_inf": relative,
        "primal_cone_violation": primal,
        "dual_cone_violation": dual,
        "complementarity_inf": complementarity,
        "friction_dissipation": dissipation,
        "normalized_tolerance": normalized_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "velocity": velocity,
    }


def solve_desaxce_certificate_aware(
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    *,
    initial_impulse: np.ndarray | None = None,
    gamma: float | None = None,
    tolerance: float = v4.DEFAULT_NATURAL_MAP_TOLERANCE,
    max_iterations: int = 80,
    max_line_search: int = 24,
    direct_contact_limit: int = desaxce.DEFAULT_DIRECT_CONTACT_LIMIT,
    fail_closed: bool = True,
    absolute_tolerance: float = ABSOLUTE_CERTIFICATE_TOLERANCE,
) -> desaxce.DeSaxceResult:
    """Tighten natural-map accuracy until all original-unit laws pass."""
    requested_tolerance = float(tolerance)
    schedule = [
        value for value in REFINEMENT_TOLERANCES
        if value <= requested_tolerance
    ]
    if not schedule or schedule[0] != requested_tolerance:
        schedule.insert(0, requested_tolerance)
    # Strictly decreasing, without duplicate user/default values.
    schedule = list(dict.fromkeys(schedule))
    attempts: list[dict[str, Any]] = []
    impulse = initial_impulse
    last_result: desaxce.DeSaxceResult | None = None
    last_audit: dict[str, Any] | None = None

    for refinement, solve_tolerance in enumerate(schedule):
        result = desaxce.solve_desaxce_dense(
            W,
            b,
            mu,
            initial_impulse=impulse,
            gamma=gamma,
            tolerance=solve_tolerance,
            max_iterations=max_iterations,
            max_line_search=max_line_search,
            direct_contact_limit=direct_contact_limit,
            fail_closed=True,
        )
        audit = _law_audit(
            W,
            b,
            mu,
            result.impulse,
            float(result.info["gamma"]),
            requested_tolerance,
            absolute_tolerance,
        )
        attempts.append({
            "refinement": refinement,
            "solve_tolerance": solve_tolerance,
            **{key: value for key, value in audit.items() if key != "velocity"},
        })
        last_result = result
        last_audit = audit
        if audit["pass"]:
            info = dict(result.info)
            info.update({
                "method": (
                    "dense De Saxce natural-map semismooth Newton with "
                    "absolute-law refinement"
                ),
                "natural_map_inf": audit["natural_map_inf"],
                "natural_map_relative_inf": audit[
                    "natural_map_relative_inf"
                ],
                "cone_violation": audit["primal_cone_violation"],
                "dual_cone_violation": audit["dual_cone_violation"],
                "complementarity_inf": audit["complementarity_inf"],
                "friction_work": audit["friction_dissipation"],
                "absolute_certificate_tolerance": absolute_tolerance,
                "certificate_refinements": refinement,
                "certificate_attempts": attempts,
                "certificate_pass": True,
                "final_cone_projection_used": False,
            })
            return desaxce.DeSaxceResult(
                result.impulse,
                audit["velocity"],
                info,
            )

        # A projection is only a candidate.  Accept it solely after the full
        # natural map and every original-unit law are re-evaluated.
        projected, _, _ = desaxce.project_coulomb_with_jacobian(
            result.impulse, mu
        )
        projected_audit = _law_audit(
            W,
            b,
            mu,
            projected,
            float(result.info["gamma"]),
            requested_tolerance,
            absolute_tolerance,
        )
        attempts[-1]["projected_candidate"] = {
            key: value for key, value in projected_audit.items()
            if key != "velocity"
        }
        if projected_audit["pass"]:
            info = dict(result.info)
            info.update({
                "method": (
                    "dense De Saxce natural-map semismooth Newton with "
                    "fully re-audited final cone projection"
                ),
                "natural_map_inf": projected_audit["natural_map_inf"],
                "natural_map_relative_inf": projected_audit[
                    "natural_map_relative_inf"
                ],
                "cone_violation": projected_audit[
                    "primal_cone_violation"
                ],
                "dual_cone_violation": projected_audit[
                    "dual_cone_violation"
                ],
                "complementarity_inf": projected_audit[
                    "complementarity_inf"
                ],
                "friction_work": projected_audit[
                    "friction_dissipation"
                ],
                "absolute_certificate_tolerance": absolute_tolerance,
                "certificate_refinements": refinement,
                "certificate_attempts": attempts,
                "certificate_pass": True,
                "final_cone_projection_used": True,
            })
            return desaxce.DeSaxceResult(
                projected,
                projected_audit["velocity"],
                info,
            )
        impulse = result.impulse

    assert last_result is not None and last_audit is not None
    failed = [
        name for name, passed in last_audit["checks"].items() if not passed
    ]
    final_metrics = {
        "natural_map_relative_inf": last_audit[
            "natural_map_relative_inf"
        ],
        "primal_cone_violation": last_audit["primal_cone_violation"],
        "dual_cone_violation": last_audit["dual_cone_violation"],
        "complementarity_inf": last_audit["complementarity_inf"],
        "friction_dissipation": last_audit["friction_dissipation"],
    }
    message = (
        "certificate-aware De Saxce refinement exhausted: "
        + ", ".join(failed)
        + f"; final={final_metrics}"
    )
    if fail_closed:
        raise RuntimeError(message)
    info = dict(last_result.info)
    info.update({
        "certificate_pass": False,
        "certificate_attempts": attempts,
        "certificate_failure": message,
    })
    return desaxce.DeSaxceResult(
        last_result.impulse, last_audit["velocity"], info
    )


class _DeSaxceFacade:
    @staticmethod
    def solve_desaxce_dense(*args: Any, **kwargs: Any) -> Any:
        return solve_desaxce_certificate_aware(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(desaxce, name)


def _bound_v4_condensed() -> Any:
    globals_copy = dict(v4.solve_stick_condensed_desaxce.__globals__)
    globals_copy["desaxce"] = _DeSaxceFacade()
    bound = types.FunctionType(
        v4.solve_stick_condensed_desaxce.__code__,
        globals_copy,
        name="solve_stick_condensed_desaxce_v5_bound",
        argdefs=v4.solve_stick_condensed_desaxce.__defaults__,
        closure=v4.solve_stick_condensed_desaxce.__closure__,
    )
    bound.__kwdefaults__ = dict(
        v4.solve_stick_condensed_desaxce.__kwdefaults__ or {}
    )
    return bound


def solve_stick_condensed_desaxce(*args: Any, **kwargs: Any) -> v1.DualResult:
    return _bound_v4_condensed()(*args, **kwargs)


def solve_dual_hybrid(
    H: sparse.csc_matrix,
    rhs: np.ndarray,
    rows: v1.ContactRows,
    *,
    compliance: float = v1.CONTACT_COMPLIANCE_M_PER_N,
    max_iters: int = v1.DEFAULT_DUAL_ITERS,
    stick_mu: float = v2.DEFAULT_STICK_MU,
    direct_contact_limit: int = v2.DEFAULT_DIRECT_CONTACT_LIMIT,
) -> v1.DualResult:
    try:
        return solve_sticking_active_set(
            H,
            rhs,
            rows,
            compliance=compliance,
            stick_mu=stick_mu,
            direct_contact_limit=direct_contact_limit,
        )
    except StickyNotApplicable as exc:
        all_stick_rejection = str(exc)
    result = solve_stick_condensed_desaxce(
        H,
        rhs,
        rows,
        compliance=compliance,
        stick_mu=stick_mu,
        direct_contact_limit=direct_contact_limit,
        max_iters=max(max_iters, 400),
        tolerance=v4.DEFAULT_NATURAL_MAP_TOLERANCE,
    )
    result.info["all_stick_rejection"] = all_stick_rejection
    return result


def _bound_v3_frame() -> Any:
    globals_copy = dict(v3.solve_contact_frame.__globals__)
    globals_copy["solve_dual_hybrid"] = solve_dual_hybrid
    bound = types.FunctionType(
        v3.solve_contact_frame.__code__,
        globals_copy,
        name="solve_contact_frame_v5_bound",
        argdefs=v3.solve_contact_frame.__defaults__,
        closure=v3.solve_contact_frame.__closure__,
    )
    bound.__kwdefaults__ = dict(v3.solve_contact_frame.__kwdefaults__ or {})
    return bound


def solve_contact_frame(scene: Any, dt: float, **kwargs: Any) -> dict[str, Any]:
    owner_guard("contact KKT v5 frame")
    result = _bound_v3_frame()(scene, dt, **kwargs)
    result["schema"] = SCHEMA
    result["method"] = (
        "v3 float32-commit-aware full-dt SQP + certified high-mu sticking + "
        "certificate-aware stick-condensed De Saxce friction"
    )
    result["claim_boundary"] += (
        "; De Saxce accuracy tightens until normalized natural-map and all "
        "absolute 1e-8 law certificates pass, with no associated fallback"
    )
    return result
