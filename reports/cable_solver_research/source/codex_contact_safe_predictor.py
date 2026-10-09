"""Endpoint-safe initial predictor for the coupled cable/contact SQP.

Newton's native ``CollisionPipeline`` exposes discrete collision queries and
speculative per-shape margin/gap support, but no general rigid-body TOI or
continuous collision query.  The current pile has zero margins and gaps.  A
free-flight predicted pose can therefore begin SQP already penetrating even
when the accepted state at the beginning of the physical step is safe.

This add-only module brackets the original and free-flight predicted poses,
re-collides every sampled endpoint, and returns a safe SQP *initial guess*.
The original free-flight prediction is returned separately and byte-for-byte
unchanged; it remains the inertial target in the one-dt objective.  Clamping
the initial guess is consequently a nonlinear globalization choice, not a
fractional time advance.

The certificate is deliberately narrow: it proves only that the selected
endpoint meets the existing 5 um penetration gate.  It does not prove that
the swept path is collision-free, and bisection finds the largest safe point
only under a locally monotone safe/unsafe predicate along this interpolation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from bench.global_cable import codex_bgn_contact_kkt as v1
from bench.global_cable import codex_bgn_global as bgn
from bench.global_cable import codex_rc_forest_newton as rc
from bench.global_cable import codex_rc_portfolio as pf


SCHEMA = "codex-contact-safe-predictor/v1"
EXPECTED_OWNER = (
    "OWNER: Team Codex-Remote — 2026-07-05T11:40Z — "
    "coupled long-chain/contact solver invention round"
)
PENETRATION_GATE_M = 5.0e-6
DEFAULT_BISECTION_ITERS = 24


@dataclass(frozen=True)
class SafeAlphaDecision:
    status: str
    accepted: bool
    endpoint_safe: bool
    alpha: float
    maximum_penetration_m: float
    penetration_target_m: float
    evaluations: tuple[dict[str, Any], ...]
    monotonicity_assumed: bool
    continuous_path_certified: bool = False

    def info(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "accepted": self.accepted,
            "endpoint_safe": self.endpoint_safe,
            "alpha": self.alpha,
            "maximum_penetration_m": self.maximum_penetration_m,
            "penetration_target_m": self.penetration_target_m,
            "evaluations": list(self.evaluations),
            "monotonicity_assumed": self.monotonicity_assumed,
            "continuous_path_certified": self.continuous_path_certified,
            "claim_boundary": (
                "selected endpoint was re-collided; no TOI/CCD query was "
                "available, so the swept path is not certified"
            ),
        }


@dataclass
class SafePredictorResult:
    physical_target: rc.PoseState
    initial_guess: rc.PoseState
    decision: SafeAlphaDecision
    caller_unchanged: bool
    physical_target_unchanged: bool
    commit_aware_endpoint_certification: bool = False
    time_advance_count: int = 0
    physical_dt_fraction: float = 1.0

    def info(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "decision": self.decision.info(),
            "caller_unchanged": self.caller_unchanged,
            "physical_target_unchanged": self.physical_target_unchanged,
            "commit_aware_endpoint_certification": (
                self.commit_aware_endpoint_certification
            ),
            "predictor_time_advance_count": self.time_advance_count,
            "physical_dt_fraction": self.physical_dt_fraction,
            "one_dt_semantics": (
                "initial_guess may be clamped, but physical_target is the "
                "unchanged full-dt free-flight prediction"
            ),
        }


def owner_guard(where: str) -> str:
    line = v1.DOC.read_text(encoding="utf-8").splitlines()[0]
    if line != EXPECTED_OWNER:
        raise RuntimeError(f"owner mismatch before {where}: {line!r}")
    return line


def find_safe_predictor_alpha(
    evaluate: Callable[[float], dict[str, Any]],
    *,
    penetration_target_m: float = PENETRATION_GATE_M,
    bisection_iters: int = DEFAULT_BISECTION_ITERS,
) -> SafeAlphaDecision:
    """Bracket a safe original endpoint and an unsafe predicted endpoint.

    ``evaluate(alpha)`` must perform a fresh discrete collision query and
    return ``maximum_penetration_m`` plus an optional ``overflow`` flag.
    """
    if not (
        math.isfinite(penetration_target_m)
        and 0.0 <= penetration_target_m <= PENETRATION_GATE_M
    ):
        raise ValueError("penetration target must be finite and may not relax 5 um")
    if bisection_iters < 0:
        raise ValueError("bisection iteration count must be nonnegative")

    samples: list[dict[str, Any]] = []
    cache: dict[float, dict[str, Any]] = {}

    def sample(alpha: float) -> dict[str, Any]:
        alpha = float(alpha)
        if alpha in cache:
            return cache[alpha]
        raw = dict(evaluate(alpha))
        penetration = float(raw["maximum_penetration_m"])
        overflow = bool(raw.get("overflow", False))
        finite = bool(math.isfinite(penetration) and not overflow)
        record = {
            "alpha": alpha,
            "maximum_penetration_m": penetration,
            "overflow": overflow,
            "finite": finite,
            "endpoint_safe": bool(
                finite and penetration <= penetration_target_m
            ),
        }
        cache[alpha] = record
        samples.append(record)
        return record

    original = sample(0.0)
    if not original["endpoint_safe"]:
        return SafeAlphaDecision(
            status="original_endpoint_unsafe",
            accepted=False,
            endpoint_safe=False,
            alpha=0.0,
            maximum_penetration_m=original["maximum_penetration_m"],
            penetration_target_m=penetration_target_m,
            evaluations=tuple(samples),
            monotonicity_assumed=False,
        )

    predicted = sample(1.0)
    if predicted["endpoint_safe"]:
        return SafeAlphaDecision(
            status="full_prediction_safe",
            accepted=True,
            endpoint_safe=True,
            alpha=1.0,
            maximum_penetration_m=predicted["maximum_penetration_m"],
            penetration_target_m=penetration_target_m,
            evaluations=tuple(samples),
            monotonicity_assumed=False,
        )

    low = original
    high = predicted
    for _ in range(bisection_iters):
        midpoint = 0.5 * (float(low["alpha"]) + float(high["alpha"]))
        if midpoint == low["alpha"] or midpoint == high["alpha"]:
            break
        trial = sample(midpoint)
        if trial["endpoint_safe"]:
            low = trial
        else:
            high = trial

    # ``low`` is updated only after a fresh query passes the unchanged gate.
    return SafeAlphaDecision(
        status="prediction_clamped_to_safe_endpoint",
        accepted=True,
        endpoint_safe=True,
        alpha=float(low["alpha"]),
        maximum_penetration_m=float(low["maximum_penetration_m"]),
        penetration_target_m=penetration_target_m,
        evaluations=tuple(samples),
        monotonicity_assumed=True,
    )


def prediction_direction(
    data: rc.ModelData,
    original: rc.PoseState,
    predicted: rc.PoseState,
) -> np.ndarray:
    """Return the left-trivialized SE(3) increment original -> predicted."""
    bodies = np.asarray(data.dynamic_bodies, dtype=np.int64)
    direction = np.empty((len(bodies), 6), dtype=np.float64)
    direction[:, :3] = (
        np.asarray(predicted.p_com, dtype=np.float64)[bodies]
        - np.asarray(original.p_com, dtype=np.float64)[bodies]
    )
    delta_q = pf._q_normalize_batch(pf._q_mul_batch(
        np.asarray(predicted.q, dtype=np.float64)[bodies],
        pf._q_conj_batch(np.asarray(original.q, dtype=np.float64)[bodies]),
    ))
    direction[:, 3:] = pf._q_log_batch(delta_q)
    if not np.all(np.isfinite(direction)):
        raise RuntimeError("non-finite free-flight prediction increment")
    return direction


def clamp_predicted_base(
    scene: Any,
    plan: bgn.PreparedBlockPlan,
    original: rc.PoseState,
    predicted: rc.PoseState,
    *,
    penetration_target_m: float = PENETRATION_GATE_M,
    bisection_iters: int = DEFAULT_BISECTION_ITERS,
    pose_projector: Callable[[rc.PoseState], rc.PoseState] | None = None,
) -> SafePredictorResult:
    """Return a safe SQP initial guess and unchanged full-dt physical target.

    This function does not mutate ``scene.state_0``, ``original``, or
    ``predicted``.  A caller must continue to assemble inertia against
    ``result.physical_target``; using ``initial_guess`` as the target would
    change the dynamics and is outside this API's claim.
    """
    owner_guard("contact safe predictor")
    if scene.collision_pipeline is None:
        raise ValueError("safe predictor requires a collision pipeline")
    v1._validate_plan(plan, scene, plan.dt)

    caller_hash_before = v1.state_sha256(scene.state_0)
    original_p_before = np.asarray(original.p_com).copy()
    original_q_before = np.asarray(original.q).copy()
    predicted_p_before = np.asarray(predicted.p_com).copy()
    predicted_q_before = np.asarray(predicted.q).copy()
    physical_target = predicted.copy()
    direction = prediction_direction(plan.data, original, predicted)

    scratch = scene.model.state()
    scratch.assign(scene.state_0)
    contacts = scene.model.contacts(collision_pipeline=scene.collision_pipeline)
    pose_cache: dict[float, rc.PoseState] = {}

    def evaluate(alpha: float) -> dict[str, Any]:
        pose = bgn._retract_fast(plan.data, original, direction, alpha)
        if pose_projector is not None:
            pose = pose_projector(pose)
        pose_cache[float(alpha)] = pose
        rows = v1.collision_rows(
            scene, plan, pose, original, scratch, contacts
        )
        metrics = v1._contact_metrics(rows)
        return {
            "maximum_penetration_m": metrics["maximum_penetration_m"],
            "overflow": metrics["overflow"],
        }

    decision = find_safe_predictor_alpha(
        evaluate,
        penetration_target_m=penetration_target_m,
        bisection_iters=bisection_iters,
    )
    initial_guess = pose_cache[decision.alpha].copy()

    caller_unchanged = bool(
        caller_hash_before == v1.state_sha256(scene.state_0)
        and np.array_equal(original_p_before, np.asarray(original.p_com))
        and np.array_equal(original_q_before, np.asarray(original.q))
        and np.array_equal(predicted_p_before, np.asarray(predicted.p_com))
        and np.array_equal(predicted_q_before, np.asarray(predicted.q))
    )
    target_unchanged = bool(
        np.array_equal(predicted_p_before, physical_target.p_com)
        and np.array_equal(predicted_q_before, physical_target.q)
    )
    if not caller_unchanged or not target_unchanged:
        raise RuntimeError("safe predictor mutated caller or physical target")
    return SafePredictorResult(
        physical_target=physical_target,
        initial_guess=initial_guess,
        decision=decision,
        caller_unchanged=caller_unchanged,
        physical_target_unchanged=target_unchanged,
        commit_aware_endpoint_certification=pose_projector is not None,
    )
