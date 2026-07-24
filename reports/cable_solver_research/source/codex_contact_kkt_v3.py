"""Contact KKT v3: safe SQP initial guess with unchanged full-dt target.

This module preserves v2's certified sticking/mixed-cone algebra and every
numerical gate.  Its only algorithmic change is to keep two poses distinct:

* ``physical_target`` is the unchanged free-flight prediction for one full dt
  and is used by every inertia assembly and merit evaluation;
* ``initial_guess`` is endpoint-clamped, when necessary, before the first SQP
  linearization so the nonlinear iteration does not start from an already
  unsafe predicted pose.

The predictor and line search re-collide discrete endpoints.  They are not
continuous collision detection and make no swept-path claim.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import warp as wp

from bench.global_cable import codex_bgn_contact_kkt as v1
from bench.global_cable import codex_bgn_global as bgn
from bench.global_cable import codex_contact_kkt_v2 as v2
from bench.global_cable import codex_contact_safe_predictor as predictor
from bench.global_cable import codex_rc_forest_newton as rc
from bench.global_cable import codex_rc_portfolio as pf


SCHEMA = "codex-coupled-cable-contact-kkt/v3"
PREDICTOR_HEADROOM_TARGET_M = 2.5e-6
PUBLIC_SOLVER_PENETRATION_GATE_M = v2.CONTACT_PENETRATION_GATE_M

# The algebra is exactly the v2 implementation; expose these names so focused
# tests and callers do not need to reach through two modules.
solve_sticking_active_set = v2.solve_sticking_active_set
solve_mixed_stick_slide_active_set = v2.solve_mixed_stick_slide_active_set
solve_dual_hybrid = v2.solve_dual_hybrid
StickyNotApplicable = v2.StickyNotApplicable


def owner_guard(where: str) -> str:
    return v2.owner_guard(where)


def _committed_within_internal_headroom(contact: dict[str, Any]) -> bool:
    """Strict internal certificate; deliberately tighter than public gate."""
    return bool(
        not contact["overflow"]
        and math.isfinite(float(contact["maximum_penetration_m"]))
        and float(contact["maximum_penetration_m"])
        <= PREDICTOR_HEADROOM_TARGET_M
    )


def _float32_commit_roundtrip(
    data: rc.ModelData,
    pose: rc.PoseState,
) -> tuple[rc.PoseState, np.ndarray]:
    """Project a pose through the exact float32 body_q commit representation."""
    body_q = bgn._body_q_from_pose_fast(data, pose).astype(np.float32)
    body_q64 = body_q.astype(np.float64)
    q = pf._q_normalize_batch(body_q64[:, 3:7])
    p_com = body_q64[:, :3] + pf._q_rotate_batch(
        q, np.asarray(data.body_com, dtype=np.float64)
    )
    committed = rc.PoseState(p_com=p_com, q=q)
    return committed, body_q


def _assemble_against_physical_target(
    plan: bgn.PreparedBlockPlan,
    pose: Any,
    safe_prediction: predictor.SafePredictorResult,
    previous_raw: dict[int, tuple[np.ndarray, np.ndarray]],
) -> Any:
    """Assemble only against the unchanged one-dt inertial target."""
    return v1._assemble(
        plan, pose, safe_prediction.physical_target, previous_raw
    )


def _objective_against_physical_target(
    plan: bgn.PreparedBlockPlan,
    pose: Any,
    safe_prediction: predictor.SafePredictorResult,
    previous_raw: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, dict[str, Any]]:
    """Evaluate merit only against the unchanged one-dt inertial target."""
    return v1._objective(
        plan, pose, safe_prediction.physical_target, previous_raw
    )


def solve_contact_frame(
    scene: Any,
    dt: float,
    *,
    plan: bgn.PreparedBlockPlan | None = None,
    max_outer_iters: int = v1.DEFAULT_OUTER_ITERS,
    max_dual_iters: int = v1.DEFAULT_DUAL_ITERS,
    compliance: float = v1.CONTACT_COMPLIANCE_M_PER_N,
    stick_mu: float = v2.DEFAULT_STICK_MU,
    direct_contact_limit: int = v2.DEFAULT_DIRECT_CONTACT_LIMIT,
) -> dict[str, Any]:
    """Advance one dt with a safe initial guess and the unchanged target."""
    owner_guard("contact KKT v3 frame")
    started = time.perf_counter_ns()
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

    plan = v1.prepare_contact(scene, dt) if plan is None else plan
    v1._validate_plan(plan, scene, dt)
    original, full_dt_target = bgn._predict_pose_fast(plan.data, source_state, dt)
    predictor_started = time.perf_counter_ns()
    safe_prediction = predictor.clamp_predicted_base(
        scene, plan, original, full_dt_target,
        penetration_target_m=PREDICTOR_HEADROOM_TARGET_M,
        pose_projector=lambda pose: _float32_commit_roundtrip(
            plan.data, pose
        )[0],
    )
    predictor_elapsed_ms = (
        time.perf_counter_ns() - predictor_started
    ) * 1.0e-6
    if not (
        safe_prediction.decision.accepted
        and safe_prediction.decision.endpoint_safe
        and safe_prediction.caller_unchanged
        and safe_prediction.physical_target_unchanged
        and safe_prediction.commit_aware_endpoint_certification
        and safe_prediction.time_advance_count == 0
        and safe_prediction.physical_dt_fraction == 1.0
    ):
        raise RuntimeError(
            "safe predictor could not provide a certified one-dt initial guess: "
            f"{safe_prediction.decision.status}; penetration="
            f"{safe_prediction.decision.maximum_penetration_m:.17g} m, "
            f"internal_target={PREDICTOR_HEADROOM_TARGET_M:.17g} m, "
            f"public_gate={PUBLIC_SOLVER_PENETRATION_GATE_M:.17g} m"
        )

    previous_raw = (
        bgn.raw_by_joint(plan, original) if len(plan.data.joint_ids) else {}
    )
    pose = safe_prediction.initial_guess.copy()
    scratch = scene.model.state()
    scratch.assign(source_state)
    contacts = scene.model.contacts(collision_pipeline=scene.collision_pipeline)
    iterations: list[dict[str, Any]] = []
    final_dual: v1.DualResult | None = None
    final_body_q: np.ndarray | None = None
    converged = False

    for outer in range(max_outer_iters):
        assembly_started = time.perf_counter_ns()
        system = _assemble_against_physical_target(
            plan, pose, safe_prediction, previous_raw
        )
        H = bgn.sparse_matrix(plan, system).tocsc()
        assembly_ms = (time.perf_counter_ns() - assembly_started) * 1.0e-6
        rows = v1.collision_rows(
            scene, plan, pose, original, scratch, contacts
        )
        before = v1._contact_metrics(rows)
        solve_started = time.perf_counter_ns()
        dual = solve_dual_hybrid(
            H, system.rhs, rows, compliance=compliance,
            max_iters=max_dual_iters, stick_mu=stick_mu,
            direct_contact_limit=direct_contact_limit,
        )
        solve_ms = (time.perf_counter_ns() - solve_started) * 1.0e-6
        direction = dual.direction.reshape(plan.n, 6)
        raw_translation_inf = float(
            np.max(np.abs(direction[:, :3]), initial=0.0)
        )
        raw_rotation_inf = float(
            np.max(np.abs(direction[:, 3:]), initial=0.0)
        )
        if not np.all(np.isfinite(direction)):
            raise RuntimeError("non-finite contact SQP direction")

        def evaluate_trial(alpha: float) -> dict[str, Any]:
            raw_trial_pose = bgn._retract_fast(
                plan.data, pose, direction, alpha
            )
            trial_pose, trial_body_q = _float32_commit_roundtrip(
                plan.data, raw_trial_pose
            )
            trial_rows = v1.collision_rows(
                scene, plan, trial_pose, original, scratch, contacts
            )
            trial_contact = v1._contact_metrics(trial_rows)
            trial_objective, trial_structural = (
                _objective_against_physical_target(
                    plan, trial_pose, safe_prediction, previous_raw
                )
            )
            finite_trial = bool(
                math.isfinite(trial_objective)
                and math.isfinite(trial_contact["maximum_penetration_m"])
                and not trial_contact["overflow"]
            )
            return {
                "alpha": alpha,
                "finite": finite_trial,
                "maximum_penetration_m": trial_contact[
                    "maximum_penetration_m"
                ],
                "active_rows": trial_contact["active_rows"],
                "cable_objective": float(trial_objective),
                "pose": trial_pose,
                "body_q": trial_body_q,
                "rows": trial_rows,
                "contact": trial_contact,
                "structural": trial_structural,
            }

        selected, trials, globalization = (
            v2._adaptive_collision_safe_search(
                evaluate_trial,
                penetration_target_m=PREDICTOR_HEADROOM_TARGET_M,
            )
        )
        if selected is None:
            raise RuntimeError(
                "contact-aware line search found no finite trial"
            )
        alpha = float(selected["alpha"])
        pose = selected["pose"]
        final_body_q = selected["body_q"]
        after = selected["contact"]
        objective_after = float(selected["cable_objective"])
        structural_after = selected["structural"]
        translation_inf = alpha * raw_translation_inf
        rotation_inf = alpha * raw_rotation_inf
        iterations.append({
            "outer_iteration": outer,
            "cable_objective_before": float(system.objective),
            "cable_objective_after": objective_after,
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
                5.0e-7,
                4.0 * compliance
                * dual.info.get("normal_force_max_N", 0.0),
            )
            and dual.info["cone_violation"] <= 2.0e-7
            and dual.info["projected_kkt_inf"] <= 2.0e-5
        ):
            converged = True
            break

    if final_dual is None:
        raise RuntimeError("contact SQP ran no iterations")
    if final_body_q is None:
        raise RuntimeError("contact SQP selected no commit representation")
    final_rows = v1.collision_rows(
        scene, plan, pose, original, scratch, contacts
    )
    final_contact = v1._contact_metrics(final_rows)
    body_qd = qd0.astype(np.float64)
    bodies = np.asarray(plan.data.dynamic_bodies, dtype=np.int64)
    body_qd[bodies, :3] = (
        pose.p_com[bodies] - original.p_com[bodies]
    ) / dt
    delta_q = pf._q_normalize_batch(pf._q_mul_batch(
        pose.q[bodies], pf._q_conj_batch(original.q[bodies])
    ))
    body_qd[bodies, 3:] = pf._q_log_batch(delta_q) / dt
    output = scene.model.state()
    output.assign(source_state)
    output.body_q.assign(final_body_q)
    output.body_qd.assign(body_qd.astype(np.float32))
    wp.synchronize_device(scene.model.device)

    assigned = bgn._pose_from_state_fast(plan.data, output)
    committed_rows = v1.collision_rows(
        scene, plan, assigned, original, scratch, contacts
    )
    committed_contact = v1._contact_metrics(committed_rows)
    reconstructed_linear = (
        assigned.p_com[bodies] - original.p_com[bodies]
    ) / dt
    assigned_delta_q = pf._q_normalize_batch(pf._q_mul_batch(
        assigned.q[bodies], pf._q_conj_batch(original.q[bodies])
    ))
    reconstructed_angular = pf._q_log_batch(assigned_delta_q) / dt
    committed_qd = v1._as_numpy(output.body_qd, np.float64)[bodies]
    velocity_error = max(
        float(np.max(np.abs(
            committed_qd[:, :3] - reconstructed_linear
        ), initial=0.0)),
        float(np.max(np.abs(
            committed_qd[:, 3:] - reconstructed_angular
        ), initial=0.0)),
    )
    caller_hash_after = v1.state_sha256(source_state)
    caller_unchanged = bool(
        caller_hash_before == caller_hash_after
        and np.array_equal(
            q0, v1._as_numpy(source_state.body_q, np.float32)
        )
        and np.array_equal(
            qd0, v1._as_numpy(source_state.body_qd, np.float32)
        )
        and (
            f0 is None
            or np.array_equal(
                f0, v1._as_numpy(source_state.body_f, np.float32)
            )
        )
    )
    finite = bool(
        np.all(np.isfinite(v1._as_numpy(output.body_q, np.float64)))
        and np.all(np.isfinite(v1._as_numpy(output.body_qd, np.float64)))
    )
    quaternion_error = float(np.max(np.abs(
        np.linalg.norm(
            v1._as_numpy(output.body_q, np.float64)[:, 3:7], axis=1
        ) - 1.0
    ), initial=0.0))
    predictor_pass = bool(
        safe_prediction.decision.endpoint_safe
        and safe_prediction.caller_unchanged
        and safe_prediction.physical_target_unchanged
        and safe_prediction.commit_aware_endpoint_certification
        and safe_prediction.time_advance_count == 0
        and safe_prediction.physical_dt_fraction == 1.0
    )
    committed_headroom_pass = _committed_within_internal_headroom(
        committed_contact
    )
    # These are v2's gates verbatim, plus the predictor's non-mutation and
    # one-dt certificate.  No tolerance is relaxed.
    pass_gate = bool(
        predictor_pass and committed_headroom_pass
        and caller_unchanged and finite
        and quaternion_error <= 3.0e-5
        and velocity_error <= 5.0e-5
        and not final_contact["overflow"]
        and final_contact["maximum_penetration_m"] <= max(
            5.0e-6,
            4.0 * compliance
            * final_dual.info.get("normal_force_max_N", 0.0),
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
        "method": (
            "full-dt target cable SQP + endpoint-safe initial predictor + "
            "v2 certified sticking/cone algebra"
        ),
        "time_advance_count": 1,
        "outer_iterations_are_time_substeps": False,
        "dt": float(dt),
        "contact_compliance_m_per_N": compliance,
        "collision_gate_manifest": {
            "predictor_internal_target_m": PREDICTOR_HEADROOM_TARGET_M,
            "accepted_sqp_endpoint_internal_target_m": (
                PREDICTOR_HEADROOM_TARGET_M
            ),
            "committed_state_internal_target_m": (
                PREDICTOR_HEADROOM_TARGET_M
            ),
            "public_solver_penetration_gate_m": (
                PUBLIC_SOLVER_PENETRATION_GATE_M
            ),
            "public_gate_relaxed": False,
            "float32_commit_aware_certification": True,
        },
        "predictor": safe_prediction.info(),
        "predictor_elapsed_ms": predictor_elapsed_ms,
        "iterations": iterations,
        "final_contact": final_contact,
        "committed_contact": committed_contact,
        "final_kkt": final_dual.info,
        "caller_unchanged": caller_unchanged,
        "caller_state_sha256": caller_hash_before,
        "returned_state_sha256": v1.state_sha256(output),
        "velocity_reconstruction_inf": velocity_error,
        "state_finite": finite,
        "quaternion_norm_max_error": quaternion_error,
        "elapsed_ms": (time.perf_counter_ns() - started) * 1.0e-6,
        "claim_boundary": (
            "the initial guess and every accepted correction endpoint are "
            "projected through and certified against the exact float32 body_q "
            "that is committed, while inertia always uses the unchanged "
            "full-dt target; no TOI/CCD or swept-path certificate is claimed"
        ),
    }
