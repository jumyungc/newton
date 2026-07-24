"""Infeasible-start hard-equality SQP/KKT cable research solver.

This add-only candidate addresses a failure observed when a finite-penalty
trajectory with centimetre-scale joint gaps is switched to hard cable
semantics.  A penalty method first spends iterations fighting the incompatible
state and can reduce an L2 merit while leaving the worst joint untouched.

The candidate instead performs two explicit, audited operations over one
physical interval:

1. project the predicted pose onto ``C(q)=0`` with a translation-only sparse
   minimum-norm restoration (orientations and therefore bend angles are not
   changed by restoration); and
2. minimize the same full-dt inertia + authored stretch/bend/damping energy on
   that manifold using scaled sparse equality-constrained SQP systems

       [ H  J' ] [d]      [ -g ]
       [ J   0 ] [l]  =   [ -C ].

The original, unscaled KKT equations are independently residual-certified.
This is explicit Track B hard/inextensible semantics.  Restoration and SQP
iterations are nonlinear solver work, not time substeps; velocity is rebuilt
once from the caller state over exactly one ``dt``.  Contact is unsupported and
fails closed until a contact/cutset Schur implementation is verified.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import time
import uuid
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, splu
import warp as wp

from bench.global_cable import codex_al_bgn as al
from bench.global_cable import codex_bgn_global as bgn
from bench.global_cable import codex_rc_forest_newton as rc


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "bench" / "global_cable" / "cable_research.md"
OUT = ROOT / "bench" / "_workspace" / "codex_hard_kkt_bgn"
EXPECTED_OWNER = al.EXPECTED_OWNER
SCHEMA = "codex-hard-equality-kkt-bgn/v1"
DEFAULT_MAX_ITERS = 5
DEFAULT_FEASIBILITY_TOL_M = 1.0e-4
DEFAULT_STATIONARITY_TOL = 5.0e-5
LINEAR_KKT_TOL = 1.0e-10
SEGMENT_LENGTH_M = 0.04


def owner_guard(where: str) -> str:
    return al.owner_guard(where)


def sha256_file(path: Path) -> str:
    return al.sha256_file(path)


def prepare(scene: Any, dt: float) -> bgn.PreparedBlockPlan:
    owner_guard("hard KKT prepare")
    reason = bgn.support_reason(scene)
    if reason is not None:
        raise ValueError(f"hard KKT unsupported scene: {reason}")
    return bgn.prepare(scene, dt)


def stretch_jacobian(
    plan: bgn.PreparedBlockPlan,
    pose: rc.PoseState,
    *,
    translation_only: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return exact stretch rows and raw anchor gaps at ``pose``."""

    jb = bgn._joint_batch(plan, pose)
    Bp = np.asarray(jb["Jpt"], dtype=np.float64)
    Bc = np.asarray(jb["Jct"], dtype=np.float64)
    C = np.asarray(jb["C"], dtype=np.float64)
    m = len(C)
    block_width = 3 if translation_only else 6
    ncol = block_width * plan.n
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for j in range(m):
        for slot, block in (
            (int(plan.joint_parent_slot[j]), Bp[j]),
            (int(plan.joint_child_slot[j]), Bc[j]),
        ):
            if slot < 0:
                continue
            use = block[:, :3] if translation_only else block
            for r in range(3):
                for c in range(block_width):
                    value = float(use[r, c])
                    if value != 0.0:
                        rows.append(3 * j + r)
                        cols.append(block_width * slot + c)
                        values.append(value)
    J = sparse.coo_matrix(
        (values, (rows, cols)), shape=(3 * m, ncol), dtype=np.float64
    ).tocsr()
    if not np.all(np.isfinite(J.data)) or not np.all(np.isfinite(C)):
        raise RuntimeError("non-finite hard stretch linearization")
    return J, C


def restore_translation(
    plan: bgn.PreparedBlockPlan,
    pose: rc.PoseState,
) -> tuple[rc.PoseState, dict[str, Any]]:
    """Restore C=0 with translations only; rotations are bit-identical."""

    J, C = stretch_jacobian(plan, pose, translation_only=True)
    rhs = -C.reshape(-1)
    started = time.perf_counter_ns()
    dispatch: str
    if J.shape[0] == J.shape[1]:
        factor = splu(J.tocsc(), permc_spec="COLAMD")
        correction = factor.solve(rhs)
        dispatch = "square sparse LU translation restoration"
        iterations = 1
    else:
        solved = lsqr(J, rhs, atol=1.0e-13, btol=1.0e-13, conlim=1.0e12,
                      iter_lim=max(200, 4 * min(J.shape)))
        correction = solved[0]
        dispatch = "rectangular LSQR minimum-norm translation restoration"
        iterations = int(solved[2])
    elapsed_ms = (time.perf_counter_ns() - started) * 1.0e-6
    if not np.all(np.isfinite(correction)):
        raise RuntimeError("non-finite translation restoration")
    linear_residual = J @ correction - rhs
    out = pose.copy()
    bodies = np.asarray(plan.data.dynamic_bodies, dtype=np.int64)
    delta = correction.reshape(plan.n, 3)
    q_before = np.asarray(out.q, dtype=np.float64).copy()
    out.p_com[bodies] += delta
    if not np.array_equal(np.asarray(out.q), q_before):
        raise RuntimeError("translation restoration changed orientation")
    _, after = stretch_jacobian(plan, out, translation_only=True)
    before_norm = np.linalg.norm(C, axis=1)
    after_norm = np.linalg.norm(after, axis=1)
    scale = max(float(np.max(np.abs(rhs), initial=0.0)), 1.0)
    return out, {
        "dispatch": dispatch,
        "iterations": iterations,
        "elapsed_ms_descriptive": float(elapsed_ms),
        "matrix_shape": [int(J.shape[0]), int(J.shape[1])],
        "nnz": int(J.nnz),
        "linear_residual_inf": float(np.max(np.abs(linear_residual), initial=0.0)),
        "linear_residual_relative": float(
            np.max(np.abs(linear_residual), initial=0.0) / scale
        ),
        "gap_before_max_m": float(np.max(before_norm, initial=0.0)),
        "gap_before_rms_m": float(np.sqrt(np.mean(before_norm * before_norm))),
        "gap_after_max_m": float(np.max(after_norm, initial=0.0)),
        "gap_after_rms_m": float(np.sqrt(np.mean(after_norm * after_norm))),
        "translation_correction_max_m": float(
            np.max(np.linalg.norm(delta, axis=1), initial=0.0)
        ),
        "translation_correction_rms_m": float(
            np.sqrt(np.mean(np.sum(delta * delta, axis=1)))
        ),
        "orientation_bit_identical": True,
    }


def solve_scaled_kkt(
    H: sparse.spmatrix,
    J: sparse.spmatrix,
    primal_rhs: np.ndarray,
    constraint_rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve a row/column-equilibrated KKT system and certify the original."""

    started = time.perf_counter_ns()
    H0 = sparse.csc_matrix(H, dtype=np.float64)
    J0 = sparse.csr_matrix(J, dtype=np.float64)
    primal_rhs = np.asarray(primal_rhs, dtype=np.float64).reshape(-1)
    constraint_rhs = np.asarray(constraint_rhs, dtype=np.float64).reshape(-1)
    if H0.shape[0] != H0.shape[1] or J0.shape[1] != H0.shape[0]:
        raise ValueError("incompatible hard KKT shapes")
    if primal_rhs.shape != (H0.shape[0],) or constraint_rhs.shape != (J0.shape[0],):
        raise ValueError("incompatible hard KKT right-hand side")
    if not (
        np.all(np.isfinite(H0.data))
        and np.all(np.isfinite(J0.data))
        and np.all(np.isfinite(primal_rhs))
        and np.all(np.isfinite(constraint_rhs))
    ):
        raise ValueError("non-finite hard KKT input")

    diagonal = np.abs(H0.diagonal())
    floor = max(float(np.max(diagonal, initial=0.0)) * 1.0e-12, 1.0e-18)
    sx = 1.0 / np.sqrt(np.maximum(diagonal, floor))
    Dx = sparse.diags(sx)
    Hs = (Dx @ H0 @ Dx).tocsc()
    Js0 = (J0 @ Dx).tocsr()
    row_norm = np.sqrt(np.asarray(Js0.multiply(Js0).sum(axis=1)).reshape(-1))
    sc = 1.0 / np.maximum(row_norm, 1.0e-15)
    Dc = sparse.diags(sc)
    Js = (Dc @ Js0).tocsc()
    zero = sparse.csc_matrix((J0.shape[0], J0.shape[0]), dtype=np.float64)
    K = sparse.bmat([[Hs, Js.T], [Js, zero]], format="csc")
    scaled_rhs = np.concatenate([sx * primal_rhs, sc * constraint_rhs])
    factor = splu(K, permc_spec="COLAMD")
    y = factor.solve(scaled_rhs)
    direction = sx * y[: H0.shape[0]]
    multiplier = sc * y[H0.shape[0] :]
    if not np.all(np.isfinite(direction)) or not np.all(np.isfinite(multiplier)):
        raise RuntimeError("non-finite hard KKT solution")

    stationarity = H0 @ direction + J0.T @ multiplier - primal_rhs
    feasibility = J0 @ direction - constraint_rhs
    pscale = max(
        1.0,
        float(np.max(np.abs(H0 @ direction), initial=0.0)),
        float(np.max(np.abs(J0.T @ multiplier), initial=0.0)),
        float(np.max(np.abs(primal_rhs), initial=0.0)),
    )
    cscale = max(
        1.0,
        float(np.max(np.abs(J0 @ direction), initial=0.0)),
        float(np.max(np.abs(constraint_rhs), initial=0.0)),
    )
    pivot = np.abs(factor.U.diagonal())
    info = {
        "backend": "symmetrically scaled sparse hard KKT -> FP64 splu",
        "matrix_shape": [int(K.shape[0]), int(K.shape[1])],
        "primal_dofs": int(H0.shape[0]),
        "constraint_rows": int(J0.shape[0]),
        "nnz": int(K.nnz),
        "elapsed_ms_descriptive": (time.perf_counter_ns() - started) * 1.0e-6,
        "original_stationarity_inf": float(
            np.max(np.abs(stationarity), initial=0.0)
        ),
        "original_stationarity_relative": float(
            np.max(np.abs(stationarity), initial=0.0) / pscale
        ),
        "original_feasibility_inf": float(
            np.max(np.abs(feasibility), initial=0.0)
        ),
        "original_feasibility_relative": float(
            np.max(np.abs(feasibility), initial=0.0) / cscale
        ),
        "original_kkt_residual": float(
            max(
                np.max(np.abs(stationarity), initial=0.0) / pscale,
                np.max(np.abs(feasibility), initial=0.0) / cscale,
            )
        ),
        "pivot_ratio_scaled": float(
            np.min(pivot, initial=math.inf)
            / max(float(np.max(pivot, initial=0.0)), 1.0e-300)
        ),
        "primal_scale_min": float(np.min(sx, initial=math.inf)),
        "primal_scale_max": float(np.max(sx, initial=0.0)),
        "constraint_scale_min": float(np.min(sc, initial=math.inf)),
        "constraint_scale_max": float(np.max(sc, initial=0.0)),
    }
    if info["original_kkt_residual"] > LINEAR_KKT_TOL:
        raise RuntimeError(f"original hard KKT residual rejected: {info}")
    return direction, multiplier, info


def _constraint_metrics(C: np.ndarray) -> dict[str, float]:
    return al._constraint_metrics(C)


def hard_kkt_certificate(
    plan: bgn.PreparedBlockPlan,
    pose: rc.PoseState,
    predicted: rc.PoseState,
    previous_raw: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    system = bgn.assemble_blocks(plan, pose, predicted, previous_raw)
    H = bgn.sparse_matrix(plan, system).tocsc()
    J, C = stretch_jacobian(plan, pose)
    direction, multiplier, linear = solve_scaled_kkt(
        H, J, system.rhs.reshape(-1), -C.reshape(-1)
    )
    gradient = -system.rhs.reshape(-1)
    stationarity = gradient + J.T @ multiplier
    multiplier_term = J.T @ multiplier
    scale = max(
        1.0,
        float(np.max(np.abs(gradient), initial=0.0)),
        float(np.max(np.abs(multiplier_term), initial=0.0)),
    )
    blocks = direction.reshape(plan.n, 6)
    return {
        "constraint": _constraint_metrics(C),
        "stationarity_inf": float(np.max(np.abs(stationarity), initial=0.0)),
        "stationarity_relative": float(
            np.max(np.abs(stationarity), initial=0.0) / scale
        ),
        "newton_correction_translation_max_m": float(
            np.max(np.linalg.norm(blocks[:, :3], axis=1), initial=0.0)
        ),
        "newton_correction_rotation_max_rad": float(
            np.max(np.linalg.norm(blocks[:, 3:], axis=1), initial=0.0)
        ),
        "multiplier_max_N": float(
            np.max(np.linalg.norm(multiplier.reshape(-1, 3), axis=1), initial=0.0)
        ),
        "linear_original_kkt": linear,
        "finite": bool(
            np.all(np.isfinite(stationarity))
            and np.all(np.isfinite(C))
            and np.all(np.isfinite(direction))
        ),
    }


def _run_inplace(
    plan: bgn.PreparedBlockPlan,
    scene: Any,
    *,
    max_iters: int,
    feasibility_tolerance_m: float,
    stationarity_tolerance: float,
    require_converged: bool,
) -> dict[str, Any]:
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if feasibility_tolerance_m <= 0.0 or not math.isfinite(feasibility_tolerance_m):
        raise ValueError("invalid feasibility tolerance")
    if stationarity_tolerance <= 0.0 or not math.isfinite(stationarity_tolerance):
        raise ValueError("invalid stationarity tolerance")
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
        pose, restoration = restore_translation(plan, predicted)
        bend_restored = bgn._vectorized_structural_certificate(plan, pose)["bend"]
        if bend_restored != bend_before:
            raise RuntimeError("translation restoration changed bend metrics")

        iterations: list[dict[str, Any]] = []
        accepted = 0
        for iteration in range(max_iters):
            system = bgn.assemble_blocks(plan, pose, predicted, previous_raw)
            H = bgn.sparse_matrix(plan, system).tocsc()
            J, C = stretch_jacobian(plan, pose)
            direction, multiplier, linear = solve_scaled_kkt(
                H, J, system.rhs.reshape(-1), -C.reshape(-1)
            )
            direction, trust = rc.trust_scale(plan.data, direction)
            gradient = -system.rhs.reshape(-1)
            predicted_reduction = float(
                -(gradient @ direction) - 0.5 * direction @ (H @ direction)
            )
            current_gap = _constraint_metrics(C)
            selected: dict[str, Any] | None = None
            if predicted_reduction > 0.0 and np.all(np.isfinite(direction)):
                for alpha in bgn.ALPHAS:
                    raw_trial = bgn._retract_fast(plan.data, pose, direction, alpha)
                    # Retraction is nonlinear, so a tangent step is feasible
                    # only to first order.  Restore translations after every
                    # trial before applying the hard feasibility filter.
                    trial, trial_restoration = restore_translation(plan, raw_trial)
                    trial_objective, trial_stats = bgn.objective_only(
                        plan, trial, predicted, previous_raw, with_stats=True
                    )
                    trial_C = bgn._joint_batch(plan, trial)["C"]
                    trial_gap = _constraint_metrics(trial_C)
                    # A filter: objective must decrease and hard feasibility
                    # must remain inside the frozen gate.  No penalty merit can
                    # trade a bad worst gap for lower summed energy.
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
            if selected is None:
                break
            if selected["actual_reduction"] <= 1.0e-12 * max(abs(system.objective), 1.0):
                break

        final = hard_kkt_certificate(plan, pose, predicted, previous_raw)
        converged = bool(
            final["finite"]
            and final["constraint"]["max_m"] <= feasibility_tolerance_m
            and final["stationarity_relative"] <= stationarity_tolerance
            and final["linear_original_kkt"]["original_kkt_residual"]
            <= LINEAR_KKT_TOL
        )
        if require_converged and not converged:
            raise RuntimeError(
                "hard KKT certificate failed: "
                f"gap={final['constraint']['max_m']}, "
                f"stationarity={final['stationarity_relative']}, "
                f"linear={final['linear_original_kkt']['original_kkt_residual']}"
            )

        al._commit_pose(plan, state, original, pose, qd_snapshot)
        assigned_pose = bgn._pose_from_state_fast(plan.data, state)
        assigned = _constraint_metrics(bgn._joint_batch(plan, assigned_pose)["C"])
        if not rc.state_finite(state):
            raise RuntimeError("hard KKT assigned non-finite state")
        if require_converged and assigned["max_m"] > feasibility_tolerance_m:
            raise RuntimeError("hard KKT float32 assignment violates feasibility gate")
        if body_f is not None and not np.array_equal(rc._as_numpy(body_f, np.float32), f_snapshot):
            raise RuntimeError("hard KKT mutated body forces")
        committed = True
        return {
            "schema": SCHEMA,
            "mode": "hard_equality_kkt",
            "track": "Track B: hard/inextensible stretch with infeasible-start restoration",
            "backend": "translation feasibility restoration + scaled sparse FP64 equality KKT SQP",
            "physical_advances": 1,
            "predictor_evaluations": 1,
            "internal_time_substeps": 0,
            "dt": float(plan.dt),
            "max_iterations": int(max_iters),
            "accepted_steps": int(accepted),
            "input_constraint": _constraint_metrics(input_C),
            "predicted_constraint": _constraint_metrics(predicted_C),
            "restoration": restoration,
            "bend_before_restoration": bend_before,
            "bend_after_restoration": bend_restored,
            "iterations": iterations,
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


def solve_frame(
    scene: Any,
    plan: bgn.PreparedBlockPlan | None = None,
    *,
    dt: float,
    max_iters: int = DEFAULT_MAX_ITERS,
    feasibility_tolerance_m: float = DEFAULT_FEASIBILITY_TOL_M,
    stationarity_tolerance: float = DEFAULT_STATIONARITY_TOL,
    require_converged: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Return a private one-dt hard-KKT candidate and audit metadata."""

    owner_start = owner_guard("hard KKT solve start")
    reason = bgn.support_reason(scene)
    if reason is not None:
        raise ValueError(f"hard KKT unsupported scene: {reason}")
    if plan is None:
        plan = bgn.prepare(scene, dt)
    plan.validate(scene, dt, full=True)
    caller_before = tuple(value.copy() for value in al._state_arrays(scene.state_0))
    caller_hash = al.state_sha256(scene.state_0)

    import warp as wp

    setup_started = time.perf_counter_ns()
    private_state = scene.model.state()
    private_state.assign(scene.state_0)
    private_scene = replace(scene, state_0=private_state)
    wp.synchronize()
    setup_ms = (time.perf_counter_ns() - setup_started) * 1.0e-6
    solve_started = time.perf_counter_ns()
    method = _run_inplace(
        plan,
        private_scene,
        max_iters=max_iters,
        feasibility_tolerance_m=feasibility_tolerance_m,
        stationarity_tolerance=stationarity_tolerance,
        require_converged=require_converged,
    )
    wp.synchronize()
    solve_ms = (time.perf_counter_ns() - solve_started) * 1.0e-6
    if not al._arrays_equal(caller_before, scene.state_0):
        raise RuntimeError("hard KKT mutated caller state")
    owner_end = owner_guard("hard KKT solve end")
    return private_state, {
        "schema": SCHEMA,
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


def _long_n512() -> tuple[Path, dict[str, Any]]:
    """Run only the frozen N512 checkpoint gate; never starts N1024."""

    owner_start = owner_guard("hard KKT N512 benchmark start")
    from bench.global_cable import codex_al_bgn_long_benchmark as lb
    from bench.global_cable import scenes

    frozen_result = (
        ROOT
        / "bench/_workspace/codex_al_bgn_long/run_20260705T120146Z_1910667/result.json"
    )
    frozen_sha = "389bd49f33234c26018e02ff3b87d8d6af587c549ed1438606239510e7e270fb"
    if sha256_file(frozen_result) != frozen_sha:
        raise RuntimeError("frozen AL long result changed")
    frozen = json.loads(frozen_result.read_text(encoding="utf-8"))
    prior = frozen["results"][0]
    checkpoint_path = frozen_result.parent / "checkpoint_long_cantilever_512_bend.npz"
    checkpoint_sha = frozen["checkpoint_files"][checkpoint_path.name]["sha256"]
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("frozen N512 checkpoint changed")
    arrays = np.load(checkpoint_path)
    scene = scenes.long_cantilever_512_bend()
    scene.state_0.body_q.assign(arrays["body_q"])
    scene.state_0.body_qd.assign(arrays["body_qd"])
    if "body_f" in arrays.files:
        scene.state_0.body_f.assign(arrays["body_f"])
    plan = prepare(scene, lb.DT)
    caller_hash = al.state_sha256(scene.state_0)
    finite_gap = float(prior["finite_reference_max_gap_m"])
    finite_bend = float(prior["finite_reference_max_bend_rad"])
    bend_limit = float(prior["bend_gate_limit_rad"])
    rows = []
    for budget in (2, 5, 12):
        owner_guard(f"hard KKT N512 K={budget}")
        candidate = f"hard_kkt_k{budget}"
        try:
            state, metadata = solve_frame(
                scene,
                plan,
                dt=lb.DT,
                max_iters=budget,
                feasibility_tolerance_m=DEFAULT_FEASIBILITY_TOL_M,
                stationarity_tolerance=DEFAULT_STATIONARITY_TOL,
                require_converged=False,
            )
            metrics = lb.component_metrics(plan, state)
            method = metadata["method"]
            kkt = method["final_hard_kkt_preassign"]
            gap = metrics["gap_3d_m"]["max"]
            bend = metrics["bend_rest_angle_rad"]["max"]
            reduction = finite_gap / max(gap, 1.0e-300)
            hard_kkt_gate = bool(
                kkt["finite"]
                and kkt["constraint"]["max_m"] <= DEFAULT_FEASIBILITY_TOL_M
                and kkt["stationarity_relative"] <= DEFAULT_STATIONARITY_TOL
                and kkt["linear_original_kkt"]["original_kkt_residual"]
                <= LINEAR_KKT_TOL
            )
            state_gate = bool(
                metrics["state_finite"]
                and metrics["quaternion_norm_max_error"] <= 2.0e-5
                and metadata["caller_state_unchanged"]
                and method["physical_advances"] == 1
                and method["predictor_evaluations"] == 1
                and method["internal_time_substeps"] == 0
            )
            row = {
                "candidate": candidate,
                "mode": method["mode"],
                "track": method["track"],
                "max_iters": budget,
                "accepted_steps": method["accepted_steps"],
                "metrics": metrics,
                "restoration": method["restoration"],
                "final_hard_kkt": kkt,
                "caller_state_unchanged": metadata["caller_state_unchanged"],
                "physical_advances": method["physical_advances"],
                "predictor_evaluations": method["predictor_evaluations"],
                "internal_time_substeps": method["internal_time_substeps"],
                "setup_ms_descriptive": metadata["setup_ms_descriptive"],
                "solve_ms_descriptive": metadata["solve_ms_descriptive"],
                "stage_ms_descriptive": metadata["stage_ms_descriptive"],
                "returned_state_sha256": metadata["returned_state_sha256"],
                "max_gap_reduction_vs_finite_k12": float(reduction),
                "bend_limit_rad": bend_limit,
                "bend_gate": bool(bend <= bend_limit),
                "hard_kkt_gate": hard_kkt_gate,
                "state_and_one_dt_gate": state_gate,
                "promotion_gate": bool(
                    reduction >= lb.PROMOTION_REDUCTION
                    and bend <= bend_limit
                    and hard_kkt_gate
                    and state_gate
                ),
                "pass": bool(method["pass"]),
            }
        except Exception as exc:
            row = {
                "candidate": candidate,
                "mode": "hard_equality_kkt",
                "track": "Track B: hard/inextensible stretch with infeasible-start restoration",
                "max_iters": budget,
                "exception": repr(exc)[:1000],
                "promotion_gate": False,
                "pass": False,
            }
        rows.append(row)
        if al.state_sha256(scene.state_0) != caller_hash:
            raise RuntimeError("hard KKT benchmark mutated checkpoint caller")

    eligible = [row for row in rows if row.get("promotion_gate", False)]
    selected = min(
        eligible,
        key=lambda row: (
            row["stage_ms_descriptive"],
            row["metrics"]["gap_3d_m"]["max"],
        ),
        default=None,
    )
    owner_end = owner_guard("hard KKT N512 benchmark end")
    payload = {
        "schema": "codex-hard-equality-kkt-n512/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "owner_line_start": owner_start,
        "owner_line_end": owner_end,
        "device": str(wp.get_device()),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "al_source_sha256": sha256_file(Path(al.__file__).resolve()),
        "bgn_source_sha256": sha256_file(Path(bgn.__file__).resolve()),
        "frozen_al_result": str(frozen_result.relative_to(ROOT)),
        "frozen_al_result_sha256": frozen_sha,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_metrics": prior["checkpoint_metrics"],
        "finite_reference_max_gap_m": finite_gap,
        "finite_reference_max_bend_rad": finite_bend,
        "bend_gate_limit_rad": bend_limit,
        "promotion_max_gap_reduction_required": lb.PROMOTION_REDUCTION,
        "hard_kkt_stationarity_limit": DEFAULT_STATIONARITY_TOL,
        "hard_kkt_linear_residual_limit": LINEAR_KKT_TOL,
        "rows": rows,
        "eligible_count": len(eligible),
        "selected": (
            None
            if selected is None
            else {
                "candidate": selected["candidate"],
                "max_iters": selected["max_iters"],
                "max_gap_m": selected["metrics"]["gap_3d_m"]["max"],
                "rms_gap_m": selected["metrics"]["gap_3d_m"]["rms"],
                "max_gap_reduction_vs_finite_k12": selected[
                    "max_gap_reduction_vs_finite_k12"
                ],
                "max_bend_rad": selected["metrics"]["bend_rest_angle_rad"]["max"],
                "stationarity_relative": selected["final_hard_kkt"][
                    "stationarity_relative"
                ],
                "linear_original_kkt_residual": selected["final_hard_kkt"][
                    "linear_original_kkt"
                ]["original_kkt_residual"],
                "stage_ms_descriptive": selected["stage_ms_descriptive"],
            }
        ),
        "n1024_admission_gate": bool(selected is not None),
        "n1024_executed": False,
        "claim_boundary": (
            "one-process, one-dt N512 infeasible-start hard-equality formulation funnel; "
            "Track B; diagnostic setup-inclusive timing; no N1024, trajectory, contact, "
            "GPU speed, production, novelty, or universal winner claim"
        ),
    }
    run_dir = OUT / f"run_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}"
    owner_guard("hard KKT N512 output directory")
    run_dir.mkdir(parents=True, exist_ok=False)
    result = run_dir / "result.json"
    encoded = (
        json.dumps(al._strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = result.with_name(f".{result.name}.{uuid.uuid4().hex}.tmp")
    owner_guard("hard KKT N512 result write")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    owner_guard("hard KKT N512 result publish")
    os.replace(temporary, result)
    return result, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n512", action="store_true")
    args = parser.parse_args(argv)
    if not args.n512:
        parser.error("only --n512 is supported")
    path, payload = _long_n512()
    print(
        json.dumps(
            {
                "result": str(path),
                "eligible_count": payload["eligible_count"],
                "n1024_admission_gate": payload["n1024_admission_gate"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["eligible_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
