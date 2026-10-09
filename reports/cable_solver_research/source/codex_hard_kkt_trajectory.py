"""Hard-equality cable trajectories from the feasible authored rest state.

This is the scientifically matched follow-up to the preserved infeasible-start
negative.  It never switches a soft, stretched trajectory to hard semantics.
N=128 runs first for 60 rendered frames (600 production substeps).  Only if
hard feasibility, bend-reference, original nonlinear KKT, finite-state,
one-dt, and >=4x gap-reduction gates pass is N=512 run from its own feasible
rest state through the exact round-4 frame-193 comparison time.

Every hard step uses one physical ``dt=1/600``.  Requested VBD2/10/80 and
finite compact BGN are explicit finite-compliance baselines on N=128.  N=512
finite rows are source-bound round-4 evidence.  Timings are single-process
diagnostics, and hard rows are explicit Track B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import time
import uuid
from typing import Any, Callable

import newton
import numpy as np
import warp as wp

from bench.global_cable import codex_al_bgn as al
from bench.global_cable import codex_al_bgn_long_benchmark as lb
from bench.global_cable import codex_bgn_global as bgn
from bench.global_cable import codex_hard_kkt_bgn as hard
from bench.global_cable import codex_rc_forest_newton as rc
from bench.global_cable import scenes


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "_workspace" / "codex_hard_kkt_trajectory"
SCHEMA = "codex-hard-equality-feasible-trajectory/v1"
DT = 1.0 / 600.0
SUBSTEPS = 10
N128_FRAMES = 60
N512_FRAMES = 193
HARD_MAX_ITERS = 2
FEASIBILITY_TOL_M = hard.DEFAULT_FEASIBILITY_TOL_M
STATIONARITY_TOL = hard.DEFAULT_STATIONARITY_TOL
LINEAR_KKT_TOL = hard.LINEAR_KKT_TOL
REDUCTION_GATE = 4.0
BEND_RELATIVE_GATE = 0.05
BEND_ABSOLUTE_ALLOWANCE_RAD = 1.0e-3


def owner_guard(where: str) -> str:
    return al.owner_guard(where)


def sha256_file(path: Path) -> str:
    return al.sha256_file(path)


def energy_terms(plan: bgn.PreparedBlockPlan, state: Any) -> dict[str, float]:
    """Mechanical-energy components; damping is reported through balance loss."""

    data = plan.data
    pose = bgn._pose_from_state_fast(data, state)
    qd = rc._as_numpy(state.body_qd, np.float64)
    bodies = np.asarray(data.dynamic_bodies, dtype=np.int64)
    mass = np.asarray(data.body_mass, dtype=np.float64)[bodies]
    linear_ke = 0.5 * float(np.sum(mass[:, None] * qd[bodies, :3] ** 2))
    rotations = bgn.pf._q_matrix_batch(np.asarray(pose.q)[bodies])
    inertia = np.asarray(data.body_inertia, dtype=np.float64)[bodies]
    world_inertia = rotations @ inertia @ np.transpose(rotations, (0, 2, 1))
    omega = qd[bodies, 3:]
    angular_ke = 0.5 * float(
        np.sum(omega * np.einsum("nij,nj->ni", world_inertia, omega))
    )
    gravity = rc._as_numpy(data.model.gravity, np.float64).reshape(-1, 3)
    worlds = np.maximum(np.asarray(data.body_world, dtype=np.int64)[bodies], 0)
    gravitational = -float(
        np.sum(mass * np.einsum("ni,ni->n", gravity[worlds], pose.p_com[bodies]))
    )
    jb = bgn._joint_batch(plan, pose)
    stretch = 0.5 * float(
        np.sum(np.asarray(data.stretch_ke)[:, None] * jb["C"] ** 2)
    )
    bend = 0.5 * float(
        np.sum(np.asarray(data.bend_ke)[:, None] * jb["phi"] ** 2)
    )
    total = linear_ke + angular_ke + gravitational + stretch + bend
    return {
        "kinetic_linear_J": linear_ke,
        "kinetic_angular_J": angular_ke,
        "potential_gravity_J": gravitational,
        "elastic_stretch_J": stretch,
        "elastic_bend_J": bend,
        "mechanical_total_J": total,
    }


def gravity_work(plan: bgn.PreparedBlockPlan, before: Any, after: Any) -> float:
    data = plan.data
    a = bgn._pose_from_state_fast(data, before)
    b = bgn._pose_from_state_fast(data, after)
    bodies = np.asarray(data.dynamic_bodies, dtype=np.int64)
    mass = np.asarray(data.body_mass, dtype=np.float64)[bodies]
    gravity = rc._as_numpy(data.model.gravity, np.float64).reshape(-1, 3)
    worlds = np.maximum(np.asarray(data.body_world, dtype=np.int64)[bodies], 0)
    return float(
        np.sum(
            mass
            * np.einsum(
                "ni,ni->n", gravity[worlds], b.p_com[bodies] - a.p_com[bodies]
            )
        )
    )


def _record_state(plan: bgn.PreparedBlockPlan, state: Any) -> dict[str, Any]:
    metrics = lb.component_metrics(plan, state)
    return {
        "body_q": rc._as_numpy(state.body_q, np.float32),
        "body_qd": rc._as_numpy(state.body_qd, np.float32),
        "metrics": metrics,
        "energy": energy_terms(plan, state),
    }


def _trajectory_summary(
    *,
    method: str,
    semantics: str,
    requested_frames: int,
    completed_substeps: int,
    rendered: list[dict[str, Any]],
    per_step_metrics: list[dict[str, Any]],
    stage_ms: list[float],
    setup_ms: list[float],
    solve_ms: list[float],
    gravity_work_J: float,
    initial_energy: dict[str, float],
    final_energy: dict[str, float],
    failure: dict[str, Any] | None,
    hard_kkt: dict[str, float] | None = None,
) -> dict[str, Any]:
    gaps_max = np.array([x["gap_3d_m"]["max"] for x in per_step_metrics])
    gaps_rms = np.array([x["gap_3d_m"]["rms"] for x in per_step_metrics])
    axial_max = np.array([x["axial_abs_m"]["max"] for x in per_step_metrics])
    axial_rms = np.array([x["axial_abs_m"]["rms"] for x in per_step_metrics])
    transverse_max = np.array([x["transverse_m"]["max"] for x in per_step_metrics])
    transverse_rms = np.array([x["transverse_m"]["rms"] for x in per_step_metrics])
    bend_max = np.array([x["bend_rest_angle_rad"]["max"] for x in per_step_metrics])
    bend_rms = np.array([x["bend_rest_angle_rad"]["rms"] for x in per_step_metrics])

    def aggregate(maximum: np.ndarray, rms: np.ndarray) -> dict[str, float]:
        return {
            "trajectory_max": float(np.max(maximum, initial=0.0)),
            "trajectory_rms_of_joint_rms": float(
                np.sqrt(np.mean(rms * rms)) if len(rms) else 0.0
            ),
            "final_max": float(maximum[-1]) if len(maximum) else 0.0,
            "final_joint_rms": float(rms[-1]) if len(rms) else 0.0,
        }

    times = np.asarray(stage_ms, dtype=np.float64)
    energy_change = (
        final_energy["mechanical_total_J"] - initial_energy["mechanical_total_J"]
    )
    row: dict[str, Any] = {
        "method": method,
        "semantics": semantics,
        "requested_render_frames": requested_frames,
        "requested_substeps": requested_frames * SUBSTEPS,
        "completed_substeps": completed_substeps,
        "completed_render_frames": len(rendered) - 1,
        "duration_s": completed_substeps * DT,
        "gap_3d_m": aggregate(gaps_max, gaps_rms),
        "axial_abs_m": aggregate(axial_max, axial_rms),
        "transverse_m": aggregate(transverse_max, transverse_rms),
        "bend_rest_angle_rad": aggregate(bend_max, bend_rms),
        "state_finite_all_steps": bool(
            all(x["state_finite"] for x in per_step_metrics)
        ),
        "quaternion_norm_max_error": float(
            max((x["quaternion_norm_max_error"] for x in per_step_metrics), default=0.0)
        ),
        "timing_descriptive": {
            "setup_ms_total": float(np.sum(setup_ms)),
            "solve_ms_total": float(np.sum(solve_ms)),
            "stage_ms_total": float(np.sum(times)),
            "stage_ms_p50": float(np.percentile(times, 50)) if len(times) else 0.0,
            "stage_ms_p95": float(np.percentile(times, 95)) if len(times) else 0.0,
        },
        "energy_work": {
            "initial": initial_energy,
            "final": final_energy,
            "mechanical_energy_change_J": float(energy_change),
            "gravity_work_J": float(gravity_work_J),
            "change_minus_gravity_work_J": float(energy_change - gravity_work_J),
            "note": (
                "negative balance includes authored damping and implicit-integration loss; "
                "this is a diagnostic, not a conservation claim"
            ),
        },
        "failure": failure,
        "pass": bool(
            failure is None
            and completed_substeps == requested_frames * SUBSTEPS
            and all(x["state_finite"] for x in per_step_metrics)
        ),
    }
    if hard_kkt is not None:
        row["hard_kkt_worst"] = hard_kkt
    return row


def run_hard_trajectory(factory: Callable[[], Any], frames: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    owner_guard(f"hard feasible trajectory start frames={frames}")
    scene = factory()
    plan = hard.prepare(scene, DT)
    initial = _record_state(plan, scene.state_0)
    rendered = [initial]
    per_step_metrics: list[dict[str, Any]] = []
    setup_ms: list[float] = []
    solve_ms: list[float] = []
    stage_ms: list[float] = []
    work = 0.0
    failure = None
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
    completed = 0
    for step in range(1, frames * SUBSTEPS + 1):
        before = scene.model.state()
        before.assign(scene.state_0)
        state, meta = hard.solve_frame(
            scene,
            plan,
            dt=DT,
            max_iters=HARD_MAX_ITERS,
            feasibility_tolerance_m=FEASIBILITY_TOL_M,
            stationarity_tolerance=STATIONARITY_TOL,
            require_converged=False,
        )
        method = meta["method"]
        cert = method["final_hard_kkt_preassign"]
        metric = lb.component_metrics(plan, state)
        per_step_metrics.append(metric)
        setup_ms.append(meta["setup_ms_descriptive"])
        solve_ms.append(meta["solve_ms_descriptive"])
        stage_ms.append(meta["stage_ms_descriptive"])
        work += gravity_work(plan, before, state)
        worst["constraint_max_m"] = max(
            worst["constraint_max_m"], cert["constraint"]["max_m"]
        )
        worst["stationarity_relative"] = max(
            worst["stationarity_relative"], cert["stationarity_relative"]
        )
        worst["linear_original_kkt_residual"] = max(
            worst["linear_original_kkt_residual"],
            cert["linear_original_kkt"]["original_kkt_residual"],
        )
        worst["translation_restoration_max_m"] = max(
            worst["translation_restoration_max_m"],
            method["restoration"]["translation_correction_max_m"],
        )
        worst["physical_advances_min"] = min(
            worst["physical_advances_min"], method["physical_advances"]
        )
        worst["physical_advances_max"] = max(
            worst["physical_advances_max"], method["physical_advances"]
        )
        worst["predictor_evaluations_min"] = min(
            worst["predictor_evaluations_min"], method["predictor_evaluations"]
        )
        worst["predictor_evaluations_max"] = max(
            worst["predictor_evaluations_max"], method["predictor_evaluations"]
        )
        worst["internal_time_substeps_max"] = max(
            worst["internal_time_substeps_max"], method["internal_time_substeps"]
        )
        worst["caller_state_unchanged_all"] = bool(
            worst["caller_state_unchanged_all"] and meta["caller_state_unchanged"]
        )
        step_pass = bool(
            method["pass"]
            and metric["state_finite"]
            and metric["quaternion_norm_max_error"] <= 2.0e-5
            and meta["caller_state_unchanged"]
            and method["physical_advances"] == 1
            and method["predictor_evaluations"] == 1
            and method["internal_time_substeps"] == 0
        )
        if not step_pass:
            failure = {
                "substep": step,
                "time_s": step * DT,
                "method_pass": method["pass"],
                "metrics": metric,
                "certificate": cert,
            }
            break
        scene.state_0.assign(state)
        completed = step
        if step % SUBSTEPS == 0:
            rendered.append(_record_state(plan, scene.state_0))

    final_energy = energy_terms(plan, scene.state_0)
    summary = _trajectory_summary(
        method="hard_equality_kkt_k2",
        semantics="Track B hard/inextensible stretch from feasible authored rest state",
        requested_frames=frames,
        completed_substeps=completed,
        rendered=rendered,
        per_step_metrics=per_step_metrics,
        stage_ms=stage_ms,
        setup_ms=setup_ms,
        solve_ms=solve_ms,
        gravity_work_J=work,
        initial_energy=initial["energy"],
        final_energy=final_energy,
        failure=failure,
        hard_kkt=worst,
    )
    snapshots = {
        "body_q": np.stack([x["body_q"] for x in rendered]),
        "body_qd": np.stack([x["body_qd"] for x in rendered]),
        "render_time_s": np.arange(len(rendered), dtype=np.float64) / 60.0,
        "gap_max_m": np.array(
            [x["metrics"]["gap_3d_m"]["max"] for x in rendered], dtype=np.float64
        ),
        "gap_rms_m": np.array(
            [x["metrics"]["gap_3d_m"]["rms"] for x in rendered], dtype=np.float64
        ),
        "axial_max_m": np.array(
            [x["metrics"]["axial_abs_m"]["max"] for x in rendered], dtype=np.float64
        ),
        "transverse_max_m": np.array(
            [x["metrics"]["transverse_m"]["max"] for x in rendered], dtype=np.float64
        ),
        "bend_max_rad": np.array(
            [x["metrics"]["bend_rest_angle_rad"]["max"] for x in rendered],
            dtype=np.float64,
        ),
        "mechanical_total_J": np.array(
            [x["energy"]["mechanical_total_J"] for x in rendered], dtype=np.float64
        ),
    }
    return summary, snapshots


def run_vbd_trajectory(iterations: int, frames: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    owner_guard(f"N128 VBD{iterations} trajectory")
    scene = scenes.horiz_cantilever_stretch_bend()
    plan = bgn.prepare(scene, DT)
    solver = newton.solvers.SolverVBD(
        scene.model,
        iterations=iterations,
        rigid_body_contact_buffer_size=1024,
        rigid_contact_history=True,
    )
    state_in = scene.model.state()
    state_in.assign(scene.state_0)
    state_out = scene.model.state()
    control = scene.model.control()
    initial = _record_state(plan, state_in)
    rendered = [initial]
    metrics = []
    times = []
    work = 0.0
    failure = None
    for step in range(1, frames * SUBSTEPS + 1):
        before = scene.model.state()
        before.assign(state_in)
        started = time.perf_counter_ns()
        solver.step(state_in, state_out, control, None, DT)
        wp.synchronize()
        times.append((time.perf_counter_ns() - started) * 1.0e-6)
        metric = lb.component_metrics(plan, state_out)
        metrics.append(metric)
        work += gravity_work(plan, before, state_out)
        if not metric["state_finite"]:
            failure = {"substep": step, "reason": "non_finite_state"}
            break
        state_in, state_out = state_out, state_in
        if step % SUBSTEPS == 0:
            rendered.append(_record_state(plan, state_in))
    completed = len(metrics)
    final = energy_terms(plan, state_in)
    summary = _trajectory_summary(
        method=f"requested_vbd_{iterations}",
        semantics="authored finite stretch/bend penalty; requested local VBD baseline",
        requested_frames=frames,
        completed_substeps=completed,
        rendered=rendered,
        per_step_metrics=metrics,
        stage_ms=times,
        setup_ms=[0.0] * len(times),
        solve_ms=times,
        gravity_work_J=work,
        initial_energy=initial["energy"],
        final_energy=final,
        failure=failure,
    )
    snapshots = {
        "body_q": np.stack([x["body_q"] for x in rendered]),
        "body_qd": np.stack([x["body_qd"] for x in rendered]),
        "render_time_s": np.arange(len(rendered), dtype=np.float64) / 60.0,
        "gap_max_m": np.array([x["metrics"]["gap_3d_m"]["max"] for x in rendered]),
        "bend_max_rad": np.array(
            [x["metrics"]["bend_rest_angle_rad"]["max"] for x in rendered]
        ),
    }
    return summary, snapshots


def run_finite_bgn_trajectory(frames: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    owner_guard("N128 finite compact BGN trajectory")
    scene = scenes.horiz_cantilever_stretch_bend()
    plan = al.prepare(scene, DT)
    initial = _record_state(plan, scene.state_0)
    rendered = [initial]
    metrics = []
    setup_ms = []
    solve_ms = []
    stage_ms = []
    work = 0.0
    failure = None
    for step in range(1, frames * SUBSTEPS + 1):
        before = scene.model.state()
        before.assign(scene.state_0)
        state, meta = al.solve_frame(
            scene,
            plan,
            dt=DT,
            mode=al.FINITE_MODE,
            finite_max_iters=5,
            adaptive_lm=True,
            full_oracle_validation=False,
        )
        metric = lb.component_metrics(plan, state)
        metrics.append(metric)
        setup_ms.append(meta["setup_ms_descriptive"])
        solve_ms.append(meta["solve_ms_descriptive"])
        stage_ms.append(meta["stage_ms_descriptive"])
        work += gravity_work(plan, before, state)
        if not meta["method"]["pass"] or not metric["state_finite"]:
            failure = {
                "substep": step,
                "finite_frame_pass": meta["method"]["pass"],
                "metrics": metric,
            }
            break
        scene.state_0.assign(state)
        if step % SUBSTEPS == 0:
            rendered.append(_record_state(plan, scene.state_0))
    completed = len(metrics)
    final = energy_terms(plan, scene.state_0)
    summary = _trajectory_summary(
        method="finite_compact_bgn_k5",
        semantics="same authored finite compliance; compact global Track-B solver path",
        requested_frames=frames,
        completed_substeps=completed,
        rendered=rendered,
        per_step_metrics=metrics,
        stage_ms=stage_ms,
        setup_ms=setup_ms,
        solve_ms=solve_ms,
        gravity_work_J=work,
        initial_energy=initial["energy"],
        final_energy=final,
        failure=failure,
    )
    snapshots = {
        "body_q": np.stack([x["body_q"] for x in rendered]),
        "body_qd": np.stack([x["body_qd"] for x in rendered]),
        "render_time_s": np.arange(len(rendered), dtype=np.float64) / 60.0,
        "gap_max_m": np.array([x["metrics"]["gap_3d_m"]["max"] for x in rendered]),
        "bend_max_rad": np.array(
            [x["metrics"]["bend_rest_angle_rad"]["max"] for x in rendered]
        ),
    }
    return summary, snapshots


def source_bound_n512_baselines() -> list[dict[str, Any]]:
    manifest = json.loads(lb.ROUND4_MANIFEST.read_text(encoding="utf-8"))
    scene = manifest["scenes"]["long_cantilever_512_bend"]
    rows = []
    for method in ("vbd_2", "vbd_10", "vbd_80", "global_bgn"):
        record = scene["methods"][method]
        path = lb.ROUND4_ROOT / "long_cantilever_512_bend" / f"traj_{method}.npz"
        if sha256_file(path) != record["npz_sha256"]:
            raise RuntimeError(f"round-4 N512 {method} hash mismatch")
        with np.load(path) as arrays:
            stretch = np.asarray(arrays["stretch"][: N512_FRAMES + 1], dtype=np.float64)
        rows.append(
            {
                "method": method,
                "semantics": (
                    "authored finite penalty requested VBD"
                    if method.startswith("vbd")
                    else "same authored finite compliance; compact global Track-B path"
                ),
                "source_bound": True,
                "npz_sha256": record["npz_sha256"],
                "duration_s": N512_FRAMES / 60.0,
                "gap_3d_m": {
                    "trajectory_max": float(np.max(stretch, initial=0.0)),
                    "trajectory_rms_of_joint_rms": float(
                        np.sqrt(np.mean(np.mean(stretch * stretch, axis=1)))
                    ),
                    "final_max": float(np.max(stretch[-1], initial=0.0)),
                    "final_joint_rms": float(np.sqrt(np.mean(stretch[-1] ** 2))),
                },
                "substep_ms_p50_descriptive": record["metrics"]["substep_ms_p50"],
                "limitation": "round-4 NPZ stores positions/gaps, not orientation or energy",
            }
        )
    return rows


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    owner_guard(f"trajectory snapshot write {path.name}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    owner_guard(f"trajectory snapshot publish {path.name}")
    os.replace(temporary, path)


def run() -> tuple[Path, dict[str, Any]]:
    owner_start = owner_guard("feasible hard trajectory start")
    hard128, snapshots_hard128 = run_hard_trajectory(
        scenes.horiz_cantilever_stretch_bend, N128_FRAMES
    )
    vbd_rows = []
    vbd_snapshots = {}
    for iterations in (2, 10, 80):
        row, snapshots = run_vbd_trajectory(iterations, N128_FRAMES)
        vbd_rows.append(row)
        vbd_snapshots[f"n128_vbd{iterations}"] = snapshots
    finite128, snapshots_finite128 = run_finite_bgn_trajectory(N128_FRAMES)

    references = [*vbd_rows, finite128]
    reference_gap = min(row["gap_3d_m"]["trajectory_max"] for row in references)
    bend_reference = min(
        row["bend_rest_angle_rad"]["trajectory_max"] for row in references
    )
    bend_limit = bend_reference + max(
        BEND_RELATIVE_GATE * bend_reference, BEND_ABSOLUTE_ALLOWANCE_RAD
    )
    hard_gap = hard128["gap_3d_m"]["trajectory_max"]
    kkt = hard128["hard_kkt_worst"]
    gate128 = {
        "complete": bool(hard128["pass"]),
        "gap_reduction_vs_best_finite": float(reference_gap / max(hard_gap, 1.0e-300)),
        "gap_gate": bool(reference_gap / max(hard_gap, 1.0e-300) >= REDUCTION_GATE),
        "bend_reference_rad": float(bend_reference),
        "bend_limit_rad": float(bend_limit),
        "bend_gate": bool(hard128["bend_rest_angle_rad"]["trajectory_max"] <= bend_limit),
        "hard_feasibility_gate": bool(kkt["constraint_max_m"] <= FEASIBILITY_TOL_M),
        "hard_stationarity_gate": bool(kkt["stationarity_relative"] <= STATIONARITY_TOL),
        "linear_original_kkt_gate": bool(
            kkt["linear_original_kkt_residual"] <= LINEAR_KKT_TOL
        ),
        "finite_state_gate": bool(
            hard128["state_finite_all_steps"]
            and hard128["quaternion_norm_max_error"] <= 2.0e-5
        ),
        "one_dt_gate": bool(
            kkt["physical_advances_min"] == 1
            and kkt["physical_advances_max"] == 1
            and kkt["predictor_evaluations_min"] == 1
            and kkt["predictor_evaluations_max"] == 1
            and kkt["internal_time_substeps_max"] == 0
            and kkt["caller_state_unchanged_all"]
        ),
    }
    gate128["pass"] = all(
        value for key, value in gate128.items() if key.endswith("gate") or key == "complete"
    )

    hard512 = None
    snapshots_hard512 = None
    gate512 = None
    baseline512 = source_bound_n512_baselines()
    if gate128["pass"]:
        hard512, snapshots_hard512 = run_hard_trajectory(
            scenes.long_cantilever_512_bend, N512_FRAMES
        )
        reference512 = min(x["gap_3d_m"]["trajectory_max"] for x in baseline512)
        # Bound VBD80 full-state checkpoint supplies the matched bend metric.
        prior = json.loads(
            (
                ROOT
                / "bench/_workspace/codex_al_bgn_long/run_20260705T120146Z_1910667/result.json"
            ).read_text(encoding="utf-8")
        )["results"][0]
        bend_ref512 = prior["checkpoint_metrics"]["bend_rest_angle_rad"]["max"]
        bend_limit512 = bend_ref512 + max(
            BEND_RELATIVE_GATE * bend_ref512, BEND_ABSOLUTE_ALLOWANCE_RAD
        )
        hk = hard512["hard_kkt_worst"]
        gate512 = {
            "complete": bool(hard512["pass"]),
            "gap_reduction_vs_best_finite": float(
                reference512 / max(hard512["gap_3d_m"]["trajectory_max"], 1.0e-300)
            ),
            "gap_gate": bool(
                reference512 / max(hard512["gap_3d_m"]["trajectory_max"], 1.0e-300)
                >= REDUCTION_GATE
            ),
            "bend_reference_rad": float(bend_ref512),
            "bend_limit_rad": float(bend_limit512),
            "bend_gate": bool(
                hard512["bend_rest_angle_rad"]["trajectory_max"] <= bend_limit512
            ),
            "hard_feasibility_gate": bool(hk["constraint_max_m"] <= FEASIBILITY_TOL_M),
            "hard_stationarity_gate": bool(hk["stationarity_relative"] <= STATIONARITY_TOL),
            "linear_original_kkt_gate": bool(
                hk["linear_original_kkt_residual"] <= LINEAR_KKT_TOL
            ),
            "finite_state_gate": bool(
                hard512["state_finite_all_steps"]
                and hard512["quaternion_norm_max_error"] <= 2.0e-5
            ),
            "one_dt_gate": bool(
                hk["physical_advances_min"] == 1
                and hk["physical_advances_max"] == 1
                and hk["predictor_evaluations_min"] == 1
                and hk["predictor_evaluations_max"] == 1
                and hk["internal_time_substeps_max"] == 0
                and hk["caller_state_unchanged_all"]
            ),
        }
        gate512["pass"] = all(
            value
            for key, value in gate512.items()
            if key.endswith("gate") or key == "complete"
        )

    owner_end = owner_guard("feasible hard trajectory end")
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
            "codex_hard_kkt_trajectory.py": sha256_file(Path(__file__).resolve()),
            "codex_hard_kkt_bgn.py": sha256_file(Path(hard.__file__).resolve()),
            "codex_al_bgn.py": sha256_file(Path(al.__file__).resolve()),
            "codex_bgn_global.py": sha256_file(Path(bgn.__file__).resolve()),
            "scenes.py": sha256_file(Path(scenes.__file__).resolve()),
        },
        "frozen": {
            "dt": DT,
            "substeps_per_render": SUBSTEPS,
            "hard_max_iters": HARD_MAX_ITERS,
            "n128_render_frames": N128_FRAMES,
            "n512_render_frames": N512_FRAMES,
            "feasibility_tolerance_m": FEASIBILITY_TOL_M,
            "stationarity_tolerance": STATIONARITY_TOL,
            "linear_kkt_tolerance": LINEAR_KKT_TOL,
            "gap_reduction_gate": REDUCTION_GATE,
            "bend_relative_gate": BEND_RELATIVE_GATE,
            "bend_absolute_allowance_rad": BEND_ABSOLUTE_ALLOWANCE_RAD,
        },
        "n128": {
            "hard": hard128,
            "finite_baselines": references,
            "gate": gate128,
        },
        "n512": {
            "executed": hard512 is not None,
            "hard": hard512,
            "source_bound_finite_baselines": baseline512,
            "gate": gate512,
        },
        "n1024_executed": False,
        "n1024_admission_gate": bool(gate512 is not None and gate512["pass"]),
        "claim_boundary": (
            "feasible-rest Track-B hard trajectory; N128 gate then N512 to matched frame193; "
            "single-process diagnostic timing; N512 finite baselines source-bound; no contact, "
            "N1024, production, novelty, or universal winner claim"
        ),
    }
    run_dir = OUT / f"run_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}"
    owner_guard("feasible trajectory output directory")
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot_sets = {
        "n128_hard": snapshots_hard128,
        "n128_finite_bgn": snapshots_finite128,
        **vbd_snapshots,
    }
    if snapshots_hard512 is not None:
        snapshot_sets["n512_hard"] = snapshots_hard512
    files = {}
    for name, arrays in snapshot_sets.items():
        path = run_dir / f"{name}_render_states.npz"
        _write_npz(path, arrays)
        files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    payload["snapshot_files"] = files
    result = run_dir / "result.json"
    encoded = (
        json.dumps(al._strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = result.with_name(f".{result.name}.{uuid.uuid4().hex}.tmp")
    owner_guard("feasible trajectory result write")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    owner_guard("feasible trajectory result publish")
    os.replace(temporary, result)
    return result, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("only --run is supported")
    path, payload = run()
    print(
        json.dumps(
            {
                "result": str(path),
                "n128_pass": payload["n128"]["gate"]["pass"],
                "n512_executed": payload["n512"]["executed"],
                "n512_pass": (
                    None if payload["n512"]["gate"] is None else payload["n512"]["gate"]["pass"]
                ),
                "n1024_admission_gate": payload["n1024_admission_gate"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["n128"]["gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
