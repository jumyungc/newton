#!/usr/bin/env python3
"""Residual-adaptive hard-KKT N=512 trajectory gate.

Each physical step predicts exactly one ``dt`` and restores hard stretch
feasibility once.  It starts with two nonlinear SQP iterations, then continues
the *same implicit solve* to K5 and at most K12 only when the independently
recomputed original nonlinear stationarity residual approaches the unchanged
5e-5 acceptance gate.  Nonlinear iterations are never time substeps.

The immutable same-semantics N=128 parity artifact is a prerequisite.  N=512
runs 600 physical steps first and may continue to frame 193 only if every
frozen feasibility, nonlinear/linear KKT, finite-state, and one-dt gate passes.
N=1024 is never executed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import platform
import socket
import time
import uuid
from typing import Any, Callable

import numpy as np
import warp as wp

from bench.global_cable import codex_al_bgn as al
from bench.global_cable import codex_al_bgn_long_benchmark as lb
from bench.global_cable import codex_bgn_global as bgn
from bench.global_cable import codex_hard_kkt_bgn as hard
from bench.global_cable import codex_hard_kkt_parity as parity
from bench.global_cable import codex_hard_kkt_trajectory as traj
from bench.global_cable import codex_rc_forest_newton as rc
from bench.global_cable import scenes


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/_workspace/codex_hard_kkt_adaptive_n512"
SCHEMA = "codex-hard-kkt-residual-adaptive-n512/v1"
SOLVER_SCHEMA = "codex-hard-kkt-residual-adaptive-frame/v1"
DT = 1.0 / 600.0
RENDER_STRIDE = 10
GATE_STEPS = 600
FULL_STEPS = 1930
ITERATION_BUDGETS = (2, 5, 12)
ESCALATION_TRIGGER_RATIO = 0.8
ESCALATION_TRIGGER = traj.STATIONARITY_TOL * ESCALATION_TRIGGER_RATIO

PARITY_RESULT = (
    ROOT
    / "bench/_workspace/codex_hard_kkt_parity"
    / "run_20260705T122437Z_1928744/result.json"
)
PARITY_RESULT_SHA256 = "febfce298a81e358accc318a142b85ca1608c92fa39d836147cea32433b878ff"


def owner_guard(where: str) -> str:
    return al.owner_guard(where)


def sha256_file(path: Path) -> str:
    return al.sha256_file(path)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def next_budget(
    used: int,
    stationarity_relative: float,
    *,
    finite: bool = True,
    constraint_max_m: float = 0.0,
    linear_kkt: float = 0.0,
) -> int | None:
    """Return the next frozen budget when the original certificate is near/failing."""

    if used not in ITERATION_BUDGETS:
        raise ValueError(f"used budget must be one of {ITERATION_BUDGETS}")
    if used == ITERATION_BUDGETS[-1]:
        return None
    needs_more = bool(
        not finite
        or stationarity_relative >= ESCALATION_TRIGGER
        or constraint_max_m > traj.FEASIBILITY_TOL_M
        or linear_kkt > traj.LINEAR_KKT_TOL
    )
    if not needs_more:
        return None
    return ITERATION_BUDGETS[ITERATION_BUDGETS.index(used) + 1]


def timing_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total_ms": float(sum(values)),
        "p50_ms": percentile(values, 50.0),
        "p95_ms": percentile(values, 95.0),
        "max_ms": max(values, default=0.0),
    }


def _certificate_record(budget: int, certificate: dict[str, Any]) -> dict[str, Any]:
    linear = certificate["linear_original_kkt"]["original_kkt_residual"]
    upcoming = next_budget(
        budget,
        certificate["stationarity_relative"],
        finite=certificate["finite"],
        constraint_max_m=certificate["constraint"]["max_m"],
        linear_kkt=linear,
    )
    return {
        "budget": budget,
        "finite": bool(certificate["finite"]),
        "constraint_max_m": float(certificate["constraint"]["max_m"]),
        "stationarity_relative": float(certificate["stationarity_relative"]),
        "linear_original_kkt_residual": float(linear),
        "escalation_trigger": float(ESCALATION_TRIGGER),
        "next_budget": upcoming,
    }


def _run_adaptive_inplace(
    plan: bgn.PreparedBlockPlan,
    scene: Any,
    *,
    feasibility_tolerance_m: float,
    stationarity_tolerance: float,
) -> dict[str, Any]:
    """One predictor and one physical dt with residual-adaptive SQP continuation."""

    state = scene.state_0
    q_snapshot = rc._as_numpy(state.body_q, np.float32)
    qd_snapshot = rc._as_numpy(state.body_qd, np.float32)
    body_f = getattr(state, "body_f", None)
    f_snapshot = rc._as_numpy(body_f, np.float32) if body_f is not None else None
    committed = False
    try:
        original, predicted = bgn._predict_pose_fast(plan.data, state, plan.dt)
        previous_raw = bgn.raw_by_joint(plan, original)
        input_C = bgn._joint_batch(plan, original)["C"]
        predicted_C = bgn._joint_batch(plan, predicted)["C"]
        bend_before = bgn._vectorized_structural_certificate(plan, predicted)["bend"]
        pose, restoration = hard.restore_translation(plan, predicted)
        bend_restored = bgn._vectorized_structural_certificate(plan, pose)["bend"]
        if bend_restored != bend_before:
            raise RuntimeError("translation restoration changed bend metrics")

        iterations: list[dict[str, Any]] = []
        checkpoints: list[dict[str, Any]] = []
        accepted = 0
        stop_reason = "maximum_budget"
        final: dict[str, Any] | None = None
        for iteration in range(ITERATION_BUDGETS[-1]):
            system = bgn.assemble_blocks(plan, pose, predicted, previous_raw)
            H = bgn.sparse_matrix(plan, system).tocsc()
            J, C = hard.stretch_jacobian(plan, pose)
            direction, multiplier, linear = hard.solve_scaled_kkt(
                H, J, system.rhs.reshape(-1), -C.reshape(-1)
            )
            direction, trust = rc.trust_scale(plan.data, direction)
            gradient = -system.rhs.reshape(-1)
            predicted_reduction = float(
                -(gradient @ direction) - 0.5 * direction @ (H @ direction)
            )
            current_gap = hard._constraint_metrics(C)
            selected: dict[str, Any] | None = None
            if predicted_reduction > 0.0 and np.all(np.isfinite(direction)):
                for alpha in bgn.ALPHAS:
                    raw_trial = bgn._retract_fast(plan.data, pose, direction, alpha)
                    trial, trial_restoration = hard.restore_translation(plan, raw_trial)
                    trial_objective, trial_stats = bgn.objective_only(
                        plan, trial, predicted, previous_raw, with_stats=True
                    )
                    trial_C = bgn._joint_batch(plan, trial)["C"]
                    trial_gap = hard._constraint_metrics(trial_C)
                    if (
                        math.isfinite(trial_objective)
                        and trial_objective
                        < system.objective - bgn.ARMIJO * alpha * predicted_reduction
                        and trial_gap["max_m"] <= feasibility_tolerance_m
                    ):
                        pose = trial
                        accepted += 1
                        selected = {
                            "alpha": float(alpha),
                            "objective_after": float(trial_objective),
                            "actual_reduction": float(system.objective - trial_objective),
                            "constraint_after": trial_gap,
                            "structural_after": trial_stats,
                            "trial_restoration": trial_restoration,
                        }
                        break
            blocks = direction.reshape(plan.n, 6)
            iterations.append(
                {
                    "iteration": iteration,
                    "objective_before": float(system.objective),
                    "constraint_before": current_gap,
                    "predicted_reduction": predicted_reduction,
                    "direction_translation_max_m": float(
                        np.max(np.linalg.norm(blocks[:, :3], axis=1), initial=0.0)
                    ),
                    "direction_rotation_max_rad": float(
                        np.max(np.linalg.norm(blocks[:, 3:], axis=1), initial=0.0)
                    ),
                    "trust_region": trust,
                    "linear_kkt": linear,
                    "accepted": selected is not None,
                    "selected": selected,
                }
            )
            used = iteration + 1
            stalled = selected is None or (
                selected["actual_reduction"]
                <= 1.0e-12 * max(abs(system.objective), 1.0)
            )
            at_budget = used in ITERATION_BUDGETS
            if at_budget or stalled:
                final = hard.hard_kkt_certificate(plan, pose, predicted, previous_raw)
                checkpoint_budget = used if used in ITERATION_BUDGETS else ITERATION_BUDGETS[0]
                if used not in ITERATION_BUDGETS:
                    record = _certificate_record(checkpoint_budget, final)
                    record["iterations_executed"] = used
                else:
                    record = _certificate_record(used, final)
                checkpoints.append(record)
                if stalled:
                    stop_reason = "line_search_or_objective_stalled"
                    break
                upcoming = record["next_budget"]
                if upcoming is None:
                    stop_reason = (
                        "certificate_below_80pct_trigger"
                        if used < ITERATION_BUDGETS[-1]
                        else "maximum_budget"
                    )
                    break

        if final is None:
            final = hard.hard_kkt_certificate(plan, pose, predicted, previous_raw)
        used_iterations = len(iterations)
        used_budget = next(
            (budget for budget in ITERATION_BUDGETS if used_iterations <= budget),
            ITERATION_BUDGETS[-1],
        )
        converged = bool(
            final["finite"]
            and final["constraint"]["max_m"] <= feasibility_tolerance_m
            and final["stationarity_relative"] <= stationarity_tolerance
            and final["linear_original_kkt"]["original_kkt_residual"]
            <= hard.LINEAR_KKT_TOL
        )

        al._commit_pose(plan, state, original, pose, qd_snapshot)
        assigned_pose = bgn._pose_from_state_fast(plan.data, state)
        assigned = hard._constraint_metrics(bgn._joint_batch(plan, assigned_pose)["C"])
        if not rc.state_finite(state):
            raise RuntimeError("adaptive hard KKT assigned non-finite state")
        if body_f is not None and not np.array_equal(
            rc._as_numpy(body_f, np.float32), f_snapshot
        ):
            raise RuntimeError("adaptive hard KKT mutated body forces")
        committed = True
        return {
            "schema": SOLVER_SCHEMA,
            "mode": "hard_equality_kkt_residual_adaptive",
            "track": "Track B: hard/inextensible stretch",
            "backend": "one restore + one-predictor sparse hard-KKT SQP with residual continuation",
            "physical_advances": 1,
            "predictor_evaluations": 1,
            "internal_time_substeps": 0,
            "dt": float(plan.dt),
            "iteration_budgets": list(ITERATION_BUDGETS),
            "iterations_executed": used_iterations,
            "budget_used": used_budget,
            "accepted_steps": accepted,
            "input_constraint": hard._constraint_metrics(input_C),
            "predicted_constraint": hard._constraint_metrics(predicted_C),
            "restoration": restoration,
            "bend_before_restoration": bend_before,
            "bend_after_restoration": bend_restored,
            "iterations": iterations,
            "adaptive_checkpoints": checkpoints,
            "adaptive_stop_reason": stop_reason,
            "escalated_to_k5": used_budget >= 5,
            "escalated_to_k12": used_budget >= 12,
            "final_hard_kkt_preassign": final,
            "assigned_constraint": assigned,
            "converged": converged,
            "pass": converged,
            "contact_supported": False,
        }
    finally:
        if not committed:
            state.body_q.assign(q_snapshot)
            state.body_qd.assign(qd_snapshot)
            if body_f is not None:
                body_f.assign(f_snapshot)


def solve_frame_adaptive(
    scene: Any,
    plan: bgn.PreparedBlockPlan,
    *,
    dt: float,
) -> tuple[Any, dict[str, Any]]:
    owner_start = owner_guard("adaptive hard KKT solve start")
    plan.validate(scene, dt, full=True)
    caller_before = tuple(value.copy() for value in al._state_arrays(scene.state_0))
    caller_hash = al.state_sha256(scene.state_0)
    setup_started = time.perf_counter_ns()
    private_state = scene.model.state()
    private_state.assign(scene.state_0)
    private_scene = replace(scene, state_0=private_state)
    wp.synchronize()
    setup_ms = (time.perf_counter_ns() - setup_started) * 1.0e-6
    solve_started = time.perf_counter_ns()
    method = _run_adaptive_inplace(
        plan,
        private_scene,
        feasibility_tolerance_m=traj.FEASIBILITY_TOL_M,
        stationarity_tolerance=traj.STATIONARITY_TOL,
    )
    wp.synchronize()
    solve_ms = (time.perf_counter_ns() - solve_started) * 1.0e-6
    if not al._arrays_equal(caller_before, scene.state_0):
        raise RuntimeError("adaptive hard KKT mutated caller state")
    owner_end = owner_guard("adaptive hard KKT solve end")
    return private_state, {
        "schema": SOLVER_SCHEMA,
        "owner_line_start": owner_start,
        "owner_line_end": owner_end,
        "caller_state_sha256": caller_hash,
        "returned_state_sha256": al.state_sha256(private_state),
        "caller_state_unchanged": True,
        "setup_ms_descriptive": float(setup_ms),
        "solve_ms_descriptive": float(solve_ms),
        "stage_ms_descriptive": float(setup_ms + solve_ms),
        "method": method,
    }


def run_adaptive(
    factory: Callable[[], Any],
    *,
    physical_steps: int,
    initial_arrays: dict[str, np.ndarray] | None = None,
    progress_label: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    owner_guard(f"adaptive N512 trajectory {progress_label}")
    scene = factory()
    if initial_arrays is not None:
        scene.state_0.body_q.assign(initial_arrays["body_q"])
        scene.state_0.body_qd.assign(initial_arrays["body_qd"])
        if "body_f" in initial_arrays:
            scene.state_0.body_f.assign(initial_arrays["body_f"])
    plan = hard.prepare(scene, DT)
    initial = traj._record_state(plan, scene.state_0)
    rendered = [initial]
    per_step_metrics: list[dict[str, Any]] = []
    setup_ms: list[float] = []
    solve_ms: list[float] = []
    inclusive_ms: list[float] = []
    by_budget_ms: dict[int, list[float]] = {2: [], 5: [], 12: []}
    budget_counts = {2: 0, 5: 0, 12: 0}
    work = 0.0
    failure = None
    completed = 0
    worst = {
        "constraint_max_m": 0.0,
        "stationarity_relative": 0.0,
        "linear_original_kkt_residual": 0.0,
        "translation_restoration_max_m": 0.0,
        "physical_advances_min": 1,
        "physical_advances_max": 1,
        "predictor_evaluations_min": 1,
        "predictor_evaluations_max": 1,
        "internal_time_substeps_max": 0,
        "caller_state_unchanged_all": True,
    }
    for step in range(1, physical_steps + 1):
        before = scene.model.state()
        before.assign(scene.state_0)
        state, metadata = solve_frame_adaptive(scene, plan, dt=DT)
        method = metadata["method"]
        certificate = method["final_hard_kkt_preassign"]
        metrics = lb.component_metrics(plan, state)
        per_step_metrics.append(metrics)
        setup_ms.append(metadata["setup_ms_descriptive"])
        solve_ms.append(metadata["solve_ms_descriptive"])
        inclusive_ms.append(metadata["stage_ms_descriptive"])
        budget = int(method["budget_used"])
        budget_counts[budget] += 1
        by_budget_ms[budget].append(metadata["stage_ms_descriptive"])
        work += traj.gravity_work(plan, before, state)
        worst["constraint_max_m"] = max(
            worst["constraint_max_m"], certificate["constraint"]["max_m"]
        )
        worst["stationarity_relative"] = max(
            worst["stationarity_relative"], certificate["stationarity_relative"]
        )
        worst["linear_original_kkt_residual"] = max(
            worst["linear_original_kkt_residual"],
            certificate["linear_original_kkt"]["original_kkt_residual"],
        )
        worst["translation_restoration_max_m"] = max(
            worst["translation_restoration_max_m"],
            method["restoration"]["translation_correction_max_m"],
        )
        for key in ("physical_advances", "predictor_evaluations"):
            worst[f"{key}_min"] = min(worst[f"{key}_min"], method[key])
            worst[f"{key}_max"] = max(worst[f"{key}_max"], method[key])
        worst["internal_time_substeps_max"] = max(
            worst["internal_time_substeps_max"], method["internal_time_substeps"]
        )
        worst["caller_state_unchanged_all"] = bool(
            worst["caller_state_unchanged_all"] and metadata["caller_state_unchanged"]
        )
        step_pass = bool(
            method["pass"]
            and metrics["state_finite"]
            and metrics["quaternion_norm_max_error"] <= 2.0e-5
            and metadata["caller_state_unchanged"]
            and method["physical_advances"] == 1
            and method["predictor_evaluations"] == 1
            and method["internal_time_substeps"] == 0
        )
        if not step_pass:
            failure = {
                "substep": step,
                "time_s": step * DT,
                "method_pass": method["pass"],
                "budget_used": budget,
                "adaptive_checkpoints": method["adaptive_checkpoints"],
                "metrics": metrics,
                "certificate": certificate,
            }
            print(json.dumps({"event": "failure", "phase": progress_label,
                              "step": step, "budget": budget,
                              "stationarity": certificate["stationarity_relative"]}),
                  flush=True)
            break
        scene.state_0.assign(state)
        completed = step
        if step % RENDER_STRIDE == 0:
            rendered.append(traj._record_state(plan, scene.state_0))
        if step % 60 == 0:
            print(json.dumps({"event": "progress", "phase": progress_label,
                              "step": step, "requested": physical_steps,
                              "budget_counts": budget_counts,
                              "worst_stationarity": worst["stationarity_relative"]}),
                  flush=True)

    final_energy = traj.energy_terms(plan, scene.state_0)
    summary = traj._trajectory_summary(
        method="hard_equality_kkt_residual_adaptive_k2_k5_k12",
        semantics="Track B hard/inextensible; K2 then same-solve K5/K12 residual continuation",
        requested_frames=physical_steps // RENDER_STRIDE,
        completed_substeps=completed,
        rendered=rendered,
        per_step_metrics=per_step_metrics,
        stage_ms=inclusive_ms,
        setup_ms=setup_ms,
        solve_ms=solve_ms,
        gravity_work_J=work,
        initial_energy=initial["energy"],
        final_energy=final_energy,
        failure=failure,
        hard_kkt=worst,
    )
    summary["requested_substeps"] = physical_steps
    summary["completed_substeps"] = completed
    summary["duration_s"] = completed * DT
    summary["pass"] = bool(failure is None and completed == physical_steps)
    summary["residual_adaptation"] = {
        "iteration_budgets": list(ITERATION_BUDGETS),
        "stationarity_gate": traj.STATIONARITY_TOL,
        "escalation_trigger_ratio": ESCALATION_TRIGGER_RATIO,
        "escalation_trigger": ESCALATION_TRIGGER,
        "budget_counts": {f"k{key}": value for key, value in budget_counts.items()},
        "budget_rates": {
            f"k{key}": value / max(len(inclusive_ms), 1)
            for key, value in budget_counts.items()
        },
        "escalation_rate_to_k5_or_more": (
            budget_counts[5] + budget_counts[12]
        ) / max(len(inclusive_ms), 1),
        "escalation_rate_to_k12": budget_counts[12] / max(len(inclusive_ms), 1),
        "inclusive_timing_all": timing_stats(inclusive_ms),
        "inclusive_timing_by_budget": {
            f"k{key}": timing_stats(values) for key, values in by_budget_ms.items()
        },
        "setup_timing": timing_stats(setup_ms),
        "solve_timing": timing_stats(solve_ms),
    }
    snapshots = {
        "body_q": np.stack([x["body_q"] for x in rendered]),
        "body_qd": np.stack([x["body_qd"] for x in rendered]),
        "render_time_s": np.arange(len(rendered), dtype=np.float64)
        * RENDER_STRIDE * DT,
        "gap_max_m": np.array([x["metrics"]["gap_3d_m"]["max"] for x in rendered]),
        "gap_rms_m": np.array([x["metrics"]["gap_3d_m"]["rms"] for x in rendered]),
        "bend_max_rad": np.array(
            [x["metrics"]["bend_rest_angle_rad"]["max"] for x in rendered]
        ),
        "mechanical_total_J": np.array(
            [x["energy"]["mechanical_total_J"] for x in rendered]
        ),
    }
    final_arrays = {
        "body_q": rc._as_numpy(scene.state_0.body_q, np.float32),
        "body_qd": rc._as_numpy(scene.state_0.body_qd, np.float32),
    }
    body_f = getattr(scene.state_0, "body_f", None)
    if body_f is not None:
        final_arrays["body_f"] = rc._as_numpy(body_f, np.float32)
    return summary, snapshots, final_arrays


def phase_gate(summary: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    kkt = summary["hard_kkt_worst"]
    gates = {
        "complete": bool(summary["pass"]),
        "hard_feasibility": bool(kkt["constraint_max_m"] <= traj.FEASIBILITY_TOL_M),
        "hard_stationarity": bool(kkt["stationarity_relative"] <= traj.STATIONARITY_TOL),
        "linear_original_kkt": bool(
            kkt["linear_original_kkt_residual"] <= traj.LINEAR_KKT_TOL
        ),
        "finite_state": bool(
            summary["state_finite_all_steps"]
            and summary["quaternion_norm_max_error"] <= 2.0e-5
        ),
        "one_dt": bool(
            kkt["physical_advances_min"] == 1
            and kkt["physical_advances_max"] == 1
            and kkt["predictor_evaluations_min"] == 1
            and kkt["predictor_evaluations_max"] == 1
            and kkt["internal_time_substeps_max"] == 0
            and kkt["caller_state_unchanged_all"]
        ),
    }
    return bool(all(gates.values())), gates


def _write_result(run_dir: Path, payload: dict[str, Any], snapshots: dict[str, np.ndarray]) -> Path:
    snapshot_path = run_dir / "n512_adaptive_render_states.npz"
    parity._write_npz(snapshot_path, snapshots)
    payload["snapshot_file"] = {
        "path": str(snapshot_path.relative_to(ROOT)),
        "sha256": sha256_file(snapshot_path),
        "bytes": snapshot_path.stat().st_size,
    }
    result = run_dir / "result.json"
    encoded = (
        json.dumps(al._strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = result.with_name(f".{result.name}.{uuid.uuid4().hex}.tmp")
    owner_guard("adaptive N512 result write")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    owner_guard("adaptive N512 result publish")
    os.replace(temporary, result)
    return result


def run() -> tuple[Path, dict[str, Any]]:
    owner_start = owner_guard("adaptive N512 campaign start")
    if sha256_file(PARITY_RESULT) != PARITY_RESULT_SHA256:
        raise RuntimeError("frozen same-semantics parity artifact changed")
    parity_payload = json.loads(PARITY_RESULT.read_text(encoding="utf-8"))
    if not parity_payload["n128"]["pass"]:
        raise RuntimeError("frozen N128 same-semantics parity did not pass")

    phase1, snap1, final1 = run_adaptive(
        scenes.long_cantilever_512_bend,
        physical_steps=GATE_STEPS,
        progress_label="n512_gate_600",
    )
    gate1, gates1 = phase_gate(phase1)
    phase2 = None
    gates2 = None
    gate2 = False
    snapshots = snap1
    if gate1:
        phase2, snap2, _ = run_adaptive(
            scenes.long_cantilever_512_bend,
            physical_steps=FULL_STEPS - GATE_STEPS,
            initial_arrays=final1,
            progress_label="n512_continue_to_frame193",
        )
        gate2, gates2 = phase_gate(phase2)
        snapshots = {}
        for key in snap1:
            tail = snap2[key][1:]
            if key == "render_time_s":
                tail = tail + GATE_STEPS * DT
            snapshots[key] = np.concatenate([snap1[key], tail], axis=0)
    full_gate = bool(gate1 and gate2)
    owner_end = owner_guard("adaptive N512 campaign end")
    payload = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "device": str(wp.get_device()),
        "owner_line_start": owner_start,
        "owner_line_end": owner_end,
        "sources": {
            "codex_hard_kkt_adaptive_n512.py": sha256_file(Path(__file__).resolve()),
            "codex_hard_kkt_bgn.py": sha256_file(Path(hard.__file__).resolve()),
            "codex_hard_kkt_parity.py": sha256_file(Path(parity.__file__).resolve()),
        },
        "frozen_parity_artifact": str(PARITY_RESULT.relative_to(ROOT)),
        "frozen_parity_artifact_sha256": PARITY_RESULT_SHA256,
        "thresholds_unchanged": {
            "hard_feasibility_m": traj.FEASIBILITY_TOL_M,
            "hard_stationarity": traj.STATIONARITY_TOL,
            "linear_original_kkt": traj.LINEAR_KKT_TOL,
            "quaternion_norm_max_error": 2.0e-5,
        },
        "adaptation": {
            "budgets": list(ITERATION_BUDGETS),
            "escalation_trigger_ratio": ESCALATION_TRIGGER_RATIO,
            "escalation_trigger": ESCALATION_TRIGGER,
            "rule": (
                "K2 then continue the same predictor/restored implicit solve to K5/K12 "
                "when original nonlinear stationarity >= 0.8*5e-5 or another frozen "
                "certificate gate fails"
            ),
        },
        "n512": {
            "phase1_600_steps": {"summary": phase1, "gates": gates1, "pass": gate1},
            "phase2_to_frame193": (
                {"summary": phase2, "gates": gates2, "pass": gate2}
                if phase2 is not None else None
            ),
            "full_gate": full_gate,
            "finite_context": traj.source_bound_n512_baselines(),
            "context_note": "finite rows have different authored-compliance semantics",
        },
        "n1024_executed": False,
        "n1024_admission_gate": full_gate,
        "claim_boundary": (
            "N512 no-contact hard/inextensible residual-adaptive scale trajectory; "
            "diagnostic timing; no contact, N1024, production, novelty, or universal-winner claim"
        ),
    }
    run_dir = OUT / f"run_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}"
    owner_guard("adaptive N512 output directory")
    run_dir.mkdir(parents=True, exist_ok=False)
    result = _write_result(run_dir, payload, snapshots)
    return result, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("only --run is supported")
    print(json.dumps({"event": "start", "pid": os.getpid(), "n1024": False}), flush=True)
    path, payload = run()
    print(json.dumps({
        "event": "complete",
        "result": str(path),
        "result_sha256": sha256_file(path),
        "n512_phase1_gate": payload["n512"]["phase1_600_steps"]["pass"],
        "n512_full_gate": payload["n512"]["full_gate"],
        "n1024_executed": payload["n1024_executed"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
