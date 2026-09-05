"""Same-semantics hard-KKT trajectory parity and scale gate.

The prior raw-bend-vs-finite negative is preserved.  This lane asks the
scientifically correct question: does the fast hard K2 trajectory match a
more converged hard K12 trajectory?  N=128 K2/K5 are compared to K12 using
rest-curvature, full state, energy/work, original nonlinear KKT, finite-state,
and one-dt gates.  Only then may N=512 run 600 physical steps; only a stable
600-step phase may continue to the exact frame-193 horizon.  N=1024 is never
run here.
"""

from __future__ import annotations

import argparse
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
from bench.global_cable import codex_hard_kkt_trajectory as traj
from bench.global_cable import codex_rc_forest_newton as rc
from bench.global_cable import scenes


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "_workspace" / "codex_hard_kkt_parity"
SCHEMA = "codex-hard-kkt-same-semantics-parity/v1"
DT = 1.0 / 600.0
SUBSTEPS = 10
N128_FRAMES = 60
N512_GATE_FRAMES = 60
N512_FULL_FRAMES = 193
CURVATURE_REL_TOL = 0.01
CURVATURE_ABS_TOL_RAD = 1.0e-3
FULL_STATE_REL_L2_TOL = 1.0e-3
POSITION_RMS_TOL_M = 1.0e-3
ROTATION_RMS_TOL_RAD = 1.0e-3
VELOCITY_REL_L2_TOL = 1.0e-2
ENERGY_REL_TOL = 0.01
WORK_REL_TOL = 0.01

K2_ARTIFACT = (
    ROOT
    / "bench/_workspace/codex_hard_kkt_trajectory/run_20260705T121845Z_1924599/result.json"
)
K2_ARTIFACT_SHA256 = "597305e713f3f0ac24b3829126d3d9c2fd61168c530acf35c8a0f68120e3be56"


def owner_guard(where: str) -> str:
    return al.owner_guard(where)


def sha256_file(path: Path) -> str:
    return al.sha256_file(path)


def snapshot_parity(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, float]:
    qa = np.asarray(candidate["body_q"], dtype=np.float64)
    qb = np.asarray(reference["body_q"], dtype=np.float64)
    va = np.asarray(candidate["body_qd"], dtype=np.float64)
    vb = np.asarray(reference["body_qd"], dtype=np.float64)
    if qa.shape != qb.shape or va.shape != vb.shape:
        raise ValueError(f"snapshot shape mismatch: {qa.shape}/{qb.shape}, {va.shape}/{vb.shape}")
    position = qa[..., :3] - qb[..., :3]
    q1 = qa[..., 3:7]
    q2 = qb[..., 3:7]
    dots = np.abs(np.sum(q1 * q2, axis=-1))
    rotation = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    velocity = va - vb
    aligned_q2 = q2.copy()
    signs = np.where(np.sum(q1 * q2, axis=-1) < 0.0, -1.0, 1.0)
    aligned_q2 *= signs[..., None]
    state_a = np.concatenate([qa[..., :3], q1, va], axis=-1)
    state_b = np.concatenate([qb[..., :3], aligned_q2, vb], axis=-1)
    difference = state_a - state_b
    return {
        "full_state_relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(state_b), 1.0e-300)
        ),
        "position_max_m": float(np.max(np.linalg.norm(position, axis=-1), initial=0.0)),
        "position_rms_m": float(np.sqrt(np.mean(position * position))),
        "rotation_max_rad": float(np.max(rotation, initial=0.0)),
        "rotation_rms_rad": float(np.sqrt(np.mean(rotation * rotation))),
        "velocity_max": float(np.max(np.linalg.norm(velocity, axis=-1), initial=0.0)),
        "velocity_relative_l2": float(
            np.linalg.norm(velocity) / max(np.linalg.norm(vb), 1.0e-300)
        ),
    }


def parity_gate(
    candidate_summary: dict[str, Any],
    candidate_snapshots: dict[str, np.ndarray],
    reference_summary: dict[str, Any],
    reference_snapshots: dict[str, np.ndarray],
) -> dict[str, Any]:
    bend_candidate = candidate_summary["bend_rest_angle_rad"]
    bend_reference = reference_summary["bend_rest_angle_rad"]
    bend_differences = {
        key: abs(float(bend_candidate[key]) - float(bend_reference[key]))
        for key in (
            "trajectory_max",
            "trajectory_rms_of_joint_rms",
            "final_max",
            "final_joint_rms",
        )
    }
    bend_relative = {
        key: value / max(abs(float(bend_reference[key])), 1.0e-12)
        for key, value in bend_differences.items()
    }
    state = snapshot_parity(candidate_snapshots, reference_snapshots)
    ec = candidate_summary["energy_work"]
    er = reference_summary["energy_work"]
    final_energy_diff = abs(
        ec["final"]["mechanical_total_J"] - er["final"]["mechanical_total_J"]
    )
    energy_scale = max(abs(er["final"]["mechanical_total_J"]), 1.0)
    work_diff = abs(ec["gravity_work_J"] - er["gravity_work_J"])
    work_scale = max(abs(er["gravity_work_J"]), 1.0)
    kkt = candidate_summary["hard_kkt_worst"]
    gates = {
        "complete_gate": bool(candidate_summary["pass"]),
        "curvature_absolute_gate": bool(
            max(bend_differences.values(), default=0.0) <= CURVATURE_ABS_TOL_RAD
        ),
        "curvature_relative_gate": bool(
            max(bend_relative.values(), default=0.0) <= CURVATURE_REL_TOL
        ),
        "full_state_relative_gate": bool(
            state["full_state_relative_l2"] <= FULL_STATE_REL_L2_TOL
        ),
        "position_gate": bool(state["position_rms_m"] <= POSITION_RMS_TOL_M),
        "rotation_gate": bool(state["rotation_rms_rad"] <= ROTATION_RMS_TOL_RAD),
        "velocity_gate": bool(state["velocity_relative_l2"] <= VELOCITY_REL_L2_TOL),
        "energy_gate": bool(final_energy_diff / energy_scale <= ENERGY_REL_TOL),
        "work_gate": bool(work_diff / work_scale <= WORK_REL_TOL),
        "hard_feasibility_gate": bool(
            kkt["constraint_max_m"] <= traj.FEASIBILITY_TOL_M
        ),
        "hard_stationarity_gate": bool(
            kkt["stationarity_relative"] <= traj.STATIONARITY_TOL
        ),
        "linear_original_kkt_gate": bool(
            kkt["linear_original_kkt_residual"] <= traj.LINEAR_KKT_TOL
        ),
        "finite_state_gate": bool(
            candidate_summary["state_finite_all_steps"]
            and candidate_summary["quaternion_norm_max_error"] <= 2.0e-5
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
    return {
        "bend_absolute_differences_rad": bend_differences,
        "bend_relative_differences": bend_relative,
        "full_state": state,
        "final_energy_absolute_difference_J": float(final_energy_diff),
        "final_energy_relative_difference": float(final_energy_diff / energy_scale),
        "gravity_work_absolute_difference_J": float(work_diff),
        "gravity_work_relative_difference": float(work_diff / work_scale),
        **gates,
        "pass": all(gates.values()),
    }


def run_hard(
    factory: Callable[[], Any],
    *,
    max_iters: int,
    physical_steps: int,
    dt: float = DT,
    render_stride: int = SUBSTEPS,
    initial_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    owner_guard(f"hard parity trajectory K={max_iters} steps={physical_steps} dt={dt}")
    scene = factory()
    if initial_arrays is not None:
        scene.state_0.body_q.assign(initial_arrays["body_q"])
        scene.state_0.body_qd.assign(initial_arrays["body_qd"])
        if "body_f" in initial_arrays:
            scene.state_0.body_f.assign(initial_arrays["body_f"])
    plan = hard.prepare(scene, dt)
    initial = traj._record_state(plan, scene.state_0)
    rendered = [initial]
    per_step_metrics = []
    setup_ms = []
    solve_ms = []
    stage_ms = []
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
        state, metadata = hard.solve_frame(
            scene,
            plan,
            dt=dt,
            max_iters=max_iters,
            feasibility_tolerance_m=traj.FEASIBILITY_TOL_M,
            stationarity_tolerance=traj.STATIONARITY_TOL,
            require_converged=False,
        )
        method = metadata["method"]
        certificate = method["final_hard_kkt_preassign"]
        metrics = lb.component_metrics(plan, state)
        per_step_metrics.append(metrics)
        setup_ms.append(metadata["setup_ms_descriptive"])
        solve_ms.append(metadata["solve_ms_descriptive"])
        stage_ms.append(metadata["stage_ms_descriptive"])
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
                "time_s": step * dt,
                "method_pass": method["pass"],
                "metrics": metrics,
                "certificate": certificate,
            }
            break
        scene.state_0.assign(state)
        completed = step
        if step % render_stride == 0:
            rendered.append(traj._record_state(plan, scene.state_0))
    final_energy = traj.energy_terms(plan, scene.state_0)
    frames_requested = physical_steps // render_stride
    summary = traj._trajectory_summary(
        method=f"hard_equality_kkt_k{max_iters}",
        semantics="Track B hard/inextensible stretch from feasible hard state",
        requested_frames=frames_requested,
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
    # _trajectory_summary assumes production 10 substeps for its requested
    # count/duration; override for dt-refinement and continuation phases.
    summary["requested_substeps"] = physical_steps
    summary["completed_substeps"] = completed
    summary["duration_s"] = completed * dt
    summary["pass"] = bool(
        failure is None
        and completed == physical_steps
        and summary["state_finite_all_steps"]
    )
    snapshots = {
        "body_q": np.stack([x["body_q"] for x in rendered]),
        "body_qd": np.stack([x["body_qd"] for x in rendered]),
        "render_time_s": np.arange(len(rendered), dtype=np.float64)
        * render_stride
        * dt,
        "gap_max_m": np.array(
            [x["metrics"]["gap_3d_m"]["max"] for x in rendered]
        ),
        "gap_rms_m": np.array(
            [x["metrics"]["gap_3d_m"]["rms"] for x in rendered]
        ),
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


def _load_k2() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if sha256_file(K2_ARTIFACT) != K2_ARTIFACT_SHA256:
        raise RuntimeError("frozen N128 K2 artifact changed")
    payload = json.loads(K2_ARTIFACT.read_text(encoding="utf-8"))
    record = payload["snapshot_files"]["n128_hard_render_states.npz"]
    path = K2_ARTIFACT.parent / "n128_hard_render_states.npz"
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError("frozen N128 K2 snapshots changed")
    with np.load(path) as arrays:
        snapshots = {key: np.asarray(arrays[key]).copy() for key in arrays.files}
    return payload["n128"]["hard"], snapshots


def dt_refinement_n64() -> dict[str, Any]:
    """Small 0.1-s dt/dt2 hard trajectory comparison."""

    def factory():
        return scenes._make_cantilever(
            64, "hard_dt_n64", "Hard KKT dt refinement N64", bend_ke=5.0, num_frames=1
        )

    coarse, coarse_snap, _ = run_hard(
        factory, max_iters=2, physical_steps=60, dt=1.0 / 600.0, render_stride=10
    )
    fine, fine_snap, _ = run_hard(
        factory, max_iters=2, physical_steps=120, dt=1.0 / 1200.0, render_stride=20
    )
    state = snapshot_parity(
        {key: value[-1:] for key, value in coarse_snap.items() if key in {"body_q", "body_qd"}},
        {key: value[-1:] for key, value in fine_snap.items() if key in {"body_q", "body_qd"}},
    )
    return {
        "physical_horizon_s": 0.1,
        "coarse": coarse,
        "fine": fine,
        "final_state_coarse_vs_fine": state,
        "pass": bool(coarse["pass"] and fine["pass"]),
        "claim_boundary": "descriptive first-order timestep refinement; no convergence-order claim",
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    owner_guard(f"hard parity snapshot write {path.name}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    owner_guard(f"hard parity snapshot publish {path.name}")
    os.replace(temporary, path)


def run() -> tuple[Path, dict[str, Any]]:
    owner_start = owner_guard("hard parity campaign start")
    k2_summary, k2_snapshots = _load_k2()
    k5_summary, k5_snapshots, _ = run_hard(
        scenes.horiz_cantilever_stretch_bend,
        max_iters=5,
        physical_steps=N128_FRAMES * SUBSTEPS,
    )
    k12_summary, k12_snapshots, _ = run_hard(
        scenes.horiz_cantilever_stretch_bend,
        max_iters=12,
        physical_steps=N128_FRAMES * SUBSTEPS,
    )
    k2_gate = parity_gate(k2_summary, k2_snapshots, k12_summary, k12_snapshots)
    k5_gate = parity_gate(k5_summary, k5_snapshots, k12_summary, k12_snapshots)
    n128_pass = bool(k2_gate["pass"] and k5_gate["pass"] and k12_summary["pass"])

    n512_phase1 = None
    n512_phase2 = None
    n512_snapshots = None
    n512_gate = None
    if n128_pass:
        phase1, snap1, final1 = run_hard(
            scenes.long_cantilever_512_bend,
            max_iters=2,
            physical_steps=N512_GATE_FRAMES * SUBSTEPS,
        )
        k = phase1["hard_kkt_worst"]
        phase1_gate = bool(
            phase1["pass"]
            and k["constraint_max_m"] <= traj.FEASIBILITY_TOL_M
            and k["stationarity_relative"] <= traj.STATIONARITY_TOL
            and k["linear_original_kkt_residual"] <= traj.LINEAR_KKT_TOL
            and phase1["state_finite_all_steps"]
            and phase1["quaternion_norm_max_error"] <= 2.0e-5
        )
        n512_phase1 = {"summary": phase1, "gate": phase1_gate}
        if phase1_gate:
            remaining = (N512_FULL_FRAMES - N512_GATE_FRAMES) * SUBSTEPS
            phase2, snap2, _ = run_hard(
                scenes.long_cantilever_512_bend,
                max_iters=2,
                physical_steps=remaining,
                initial_arrays=final1,
            )
            k2 = phase2["hard_kkt_worst"]
            phase2_gate = bool(
                phase2["pass"]
                and k2["constraint_max_m"] <= traj.FEASIBILITY_TOL_M
                and k2["stationarity_relative"] <= traj.STATIONARITY_TOL
                and k2["linear_original_kkt_residual"] <= traj.LINEAR_KKT_TOL
                and phase2["state_finite_all_steps"]
                and phase2["quaternion_norm_max_error"] <= 2.0e-5
            )
            n512_phase2 = {"summary": phase2, "gate": phase2_gate}
            n512_gate = bool(phase1_gate and phase2_gate)
            n512_snapshots = {
                key: np.concatenate([snap1[key], snap2[key][1:]], axis=0)
                for key in snap1
            }
            # Phase-2 time is local; shift to full trajectory time.
            n512_snapshots["render_time_s"][len(snap1["render_time_s"]) :] += (
                N512_GATE_FRAMES / 60.0
            )
        else:
            n512_gate = False
            n512_snapshots = snap1

    refinement = dt_refinement_n64()
    owner_end = owner_guard("hard parity campaign end")
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
            "codex_hard_kkt_parity.py": sha256_file(Path(__file__).resolve()),
            "codex_hard_kkt_bgn.py": sha256_file(Path(hard.__file__).resolve()),
            "codex_hard_kkt_trajectory.py": sha256_file(Path(traj.__file__).resolve()),
        },
        "frozen_k2_artifact": str(K2_ARTIFACT.relative_to(ROOT)),
        "frozen_k2_artifact_sha256": K2_ARTIFACT_SHA256,
        "thresholds": {
            "curvature_relative": CURVATURE_REL_TOL,
            "curvature_absolute_rad": CURVATURE_ABS_TOL_RAD,
            "full_state_relative_l2": FULL_STATE_REL_L2_TOL,
            "position_rms_m": POSITION_RMS_TOL_M,
            "rotation_rms_rad": ROTATION_RMS_TOL_RAD,
            "velocity_relative_l2": VELOCITY_REL_L2_TOL,
            "energy_relative": ENERGY_REL_TOL,
            "work_relative": WORK_REL_TOL,
            "hard_feasibility_m": traj.FEASIBILITY_TOL_M,
            "hard_stationarity": traj.STATIONARITY_TOL,
            "linear_original_kkt": traj.LINEAR_KKT_TOL,
        },
        "n128": {
            "k2": k2_summary,
            "k5": k5_summary,
            "k12_reference": k12_summary,
            "k2_vs_k12": k2_gate,
            "k5_vs_k12": k5_gate,
            "pass": n128_pass,
            "finite_vbd_context": json.loads(K2_ARTIFACT.read_text(encoding="utf-8"))[
                "n128"
            ]["finite_baselines"],
            "context_note": "different finite-compliance material context; never bend truth",
        },
        "n512": {
            "executed": n512_phase1 is not None,
            "phase1_600_steps": n512_phase1,
            "phase2_to_frame193": n512_phase2,
            "full_gate": n512_gate,
            "finite_context": traj.source_bound_n512_baselines(),
            "context_note": "different finite-compliance material context; never bend truth",
        },
        "n64_dt_refinement": refinement,
        "n1024_executed": False,
        "n1024_admission_gate": bool(n512_gate),
        "claim_boundary": (
            "same-semantics hard K2/K5 vs hard K12 parity; N512 gated 600-step then "
            "frame193 scale trajectory; finite/VBD rows context only; diagnostic timing; "
            "no contact, N1024, production, novelty, or universal winner claim"
        ),
    }
    run_dir = OUT / f"run_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}"
    owner_guard("hard parity output directory")
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshots = {
        "n128_k5": k5_snapshots,
        "n128_k12": k12_snapshots,
    }
    if n512_snapshots is not None:
        snapshots["n512_k2"] = n512_snapshots
    files = {}
    for name, arrays in snapshots.items():
        path = run_dir / f"{name}_render_states.npz"
        _write_npz(path, arrays)
        files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    payload["snapshot_files"] = files
    result = run_dir / "result.json"
    encoded = (
        json.dumps(al._strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = result.with_name(f".{result.name}.{uuid.uuid4().hex}.tmp")
    owner_guard("hard parity result write")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    owner_guard("hard parity result publish")
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
                "n128_pass": payload["n128"]["pass"],
                "n512_executed": payload["n512"]["executed"],
                "n512_full_gate": payload["n512"]["full_gate"],
                "n1024_admission_gate": payload["n1024_admission_gate"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["n128"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
