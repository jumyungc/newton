#!/usr/bin/env python3
"""Preregistered K24-only extension from the hash-bound N512 step-1200 state.

This candidate changes one thing: the residual-adaptive same-implicit-solve
budget ladder becomes K2 -> K5 -> K12 -> K24.  It replays from the immutable
step-1200 snapshot, must pass global step 1205 under every unchanged gate, and
only then may continue to frame 193.  There is no K32, tolerance change, time
substepping, contact claim, or N1024 run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import socket
import time
import uuid
from typing import Any, Callable, Iterator

import numpy as np
import warp as wp

from bench.global_cable import codex_al_bgn as al
from bench.global_cable import codex_al_bgn_long_benchmark as lb
from bench.global_cable import codex_hard_kkt_adaptive_n512 as adaptive
from bench.global_cable import codex_hard_kkt_bgn as hard
from bench.global_cable import codex_hard_kkt_parity as parity
from bench.global_cable import codex_hard_kkt_trajectory as traj
from bench.global_cable import codex_rc_forest_newton as rc
from bench.global_cable import scenes


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "bench/global_cable/cable_research.md"
OUT = ROOT / "bench/_workspace/codex_hard_kkt_k24_probe"
SCHEMA = "codex-hard-kkt-preregistered-k24-probe/v1"
DT = 1.0 / 600.0
RENDER_STRIDE = 10
SOURCE_STEP = 1200
PROBE_STEP = 1205
FULL_STEP = 1930
K24_BUDGETS = (2, 5, 12, 24)

PREREGISTERED_DOC_SHA256 = (
    "3520bc1b1010a8a2f9903eac6cc98a91af994b7f39b8cda65f3400747d188fc6"
)
SOURCE_RESULT = (
    ROOT
    / "bench/_workspace/codex_hard_kkt_adaptive_n512"
    / "run_20260705T123956Z_1940700/result.json"
)
SOURCE_RESULT_SHA256 = (
    "988f83b688350d5f561e79e96e862eb7be583eb0e1e0461a594a4178cd0562d7"
)
SOURCE_SNAPSHOT = SOURCE_RESULT.parent / "n512_adaptive_render_states.npz"
SOURCE_SNAPSHOT_SHA256 = (
    "4081fe1536571b225881c6d6d373ba0cbd5b461115a02d92ee58d2bd53dc7bb3"
)
BASE_ADAPTIVE_SOURCE_SHA256 = (
    "cbac9b4130cbc67fad04d43d3a6513d367c93e2006a2e8b7875934526ec2c61c"
)


def owner_guard(where: str) -> str:
    return al.owner_guard(where)


def sha256_file(path: Path) -> str:
    return al.sha256_file(path)


@contextmanager
def k24_budget_scope() -> Iterator[None]:
    """Process-local extension of the already-tested adaptive solver ladder."""

    original = adaptive.ITERATION_BUDGETS
    if original != (2, 5, 12):
        raise RuntimeError(f"unexpected base adaptive budgets: {original}")
    adaptive.ITERATION_BUDGETS = K24_BUDGETS
    try:
        yield
    finally:
        adaptive.ITERATION_BUDGETS = original


def load_bound_step1200() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if sha256_file(SOURCE_RESULT) != SOURCE_RESULT_SHA256:
        raise RuntimeError("frozen adaptive result changed")
    if sha256_file(SOURCE_SNAPSHOT) != SOURCE_SNAPSHOT_SHA256:
        raise RuntimeError("frozen adaptive snapshots changed")
    with np.load(SOURCE_SNAPSHOT, allow_pickle=False) as arrays:
        prefix = {key: np.asarray(arrays[key]).copy() for key in arrays.files}
    expected_indices = np.arange(SOURCE_STEP // RENDER_STRIDE + 1, dtype=np.int64)
    expected_times = expected_indices / 60.0
    if prefix["body_q"].shape != (121, 512, 7):
        raise RuntimeError(f"unexpected bound body_q shape {prefix['body_q'].shape}")
    if prefix["body_qd"].shape != (121, 512, 6):
        raise RuntimeError(f"unexpected bound body_qd shape {prefix['body_qd'].shape}")
    stored_times = np.asarray(prefix["render_time_s"], dtype=np.float64)
    stored_indices = np.rint(stored_times * 60.0).astype(np.int64)
    if not np.array_equal(stored_indices, expected_indices):
        raise RuntimeError("bound step-1200 frame-index mismatch")
    # The two immutable trajectory phases formed their time grids through
    # algebraically equivalent FP64 expressions.  Preserve exact integer frame
    # identity and accept only one local ULP of representation difference.
    cadence_atol = float(np.spacing(np.max(np.abs(expected_times), initial=1.0)))
    if not np.allclose(stored_times, expected_times, rtol=0.0, atol=cadence_atol):
        raise RuntimeError("bound step-1200 cadence exceeds one-ULP allowance")
    initial = {
        "body_q": np.asarray(prefix["body_q"][-1], dtype=np.float32).copy(),
        "body_qd": np.asarray(prefix["body_qd"][-1], dtype=np.float32).copy(),
    }
    return prefix, initial


def global_render_steps(start_step: int, physical_steps: int) -> list[int]:
    return [
        step
        for step in range(start_step + 1, start_step + physical_steps + 1)
        if step % RENDER_STRIDE == 0
    ]


def run_segment(
    factory: Callable[[], Any],
    *,
    global_start_step: int,
    physical_steps: int,
    initial_arrays: dict[str, np.ndarray],
    progress_label: str,
    retain_all_step_audits: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    owner_guard(f"K24 segment {progress_label}")
    scene = factory()
    scene.state_0.body_q.assign(initial_arrays["body_q"])
    scene.state_0.body_qd.assign(initial_arrays["body_qd"])
    if "body_f" in initial_arrays:
        scene.state_0.body_f.assign(initial_arrays["body_f"])
    plan = hard.prepare(scene, DT)
    initial = traj._record_state(plan, scene.state_0)
    rendered: list[tuple[int, dict[str, Any]]] = [(global_start_step, initial)]
    per_step_metrics: list[dict[str, Any]] = []
    setup_ms: list[float] = []
    solve_ms: list[float] = []
    inclusive_ms: list[float] = []
    by_budget_ms: dict[int, list[float]] = {key: [] for key in K24_BUDGETS}
    budget_counts = {key: 0 for key in K24_BUDGETS}
    step_audits: list[dict[str, Any]] = []
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
    for local_step in range(1, physical_steps + 1):
        global_step = global_start_step + local_step
        before = scene.model.state()
        before.assign(scene.state_0)
        state, metadata = adaptive.solve_frame_adaptive(scene, plan, dt=DT)
        method = metadata["method"]
        certificate = method["final_hard_kkt_preassign"]
        metrics = lb.component_metrics(plan, state)
        per_step_metrics.append(metrics)
        setup_ms.append(metadata["setup_ms_descriptive"])
        solve_ms.append(metadata["solve_ms_descriptive"])
        inclusive_ms.append(metadata["stage_ms_descriptive"])
        budget = int(method["budget_used"])
        if budget not in budget_counts:
            raise RuntimeError(f"unregistered adaptive budget {budget}")
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
        audit = {
            "global_step": global_step,
            "time_s": global_step * DT,
            "budget_used": budget,
            "iterations_executed": method["iterations_executed"],
            "accepted_steps": method["accepted_steps"],
            "adaptive_checkpoints": method["adaptive_checkpoints"],
            "adaptive_stop_reason": method["adaptive_stop_reason"],
            "final_hard_kkt_preassign": certificate,
            "assigned_constraint": method["assigned_constraint"],
            "metrics": metrics,
            "inclusive_ms_descriptive": metadata["stage_ms_descriptive"],
        }
        if retain_all_step_audits:
            step_audits.append(audit)
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
            failure = {**audit, "method_pass": bool(method["pass"])}
            if not retain_all_step_audits:
                step_audits.append(audit)
            print(json.dumps({
                "event": "failure", "phase": progress_label,
                "global_step": global_step, "budget": budget,
                "stationarity": certificate["stationarity_relative"],
            }), flush=True)
            break
        scene.state_0.assign(state)
        completed = local_step
        if global_step % RENDER_STRIDE == 0:
            rendered.append((global_step, traj._record_state(plan, scene.state_0)))
        if local_step % 60 == 0 or local_step == physical_steps:
            print(json.dumps({
                "event": "progress", "phase": progress_label,
                "global_step": global_step,
                "local_step": local_step, "requested": physical_steps,
                "budget_counts": budget_counts,
                "worst_stationarity": worst["stationarity_relative"],
            }), flush=True)

    final_energy = traj.energy_terms(plan, scene.state_0)
    summary = traj._trajectory_summary(
        method="hard_equality_kkt_residual_adaptive_k2_k5_k12_k24",
        semantics="Track B hard/inextensible; same-solve residual continuation, K24-only extension",
        requested_frames=physical_steps // RENDER_STRIDE,
        completed_substeps=completed,
        rendered=[record for _, record in rendered],
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
    summary["global_start_step"] = global_start_step
    summary["global_completed_step"] = global_start_step + completed
    summary["global_requested_end_step"] = global_start_step + physical_steps
    summary["requested_substeps"] = physical_steps
    summary["completed_substeps"] = completed
    summary["duration_s"] = completed * DT
    summary["pass"] = bool(failure is None and completed == physical_steps)
    count = max(len(inclusive_ms), 1)
    summary["residual_adaptation"] = {
        "iteration_budgets": list(K24_BUDGETS),
        "stationarity_gate": traj.STATIONARITY_TOL,
        "escalation_trigger": adaptive.ESCALATION_TRIGGER,
        "budget_counts": {f"k{key}": value for key, value in budget_counts.items()},
        "budget_rates": {
            f"k{key}": value / count for key, value in budget_counts.items()
        },
        "k24_rate": budget_counts[24] / count,
        "inclusive_timing_all": adaptive.timing_stats(inclusive_ms),
        "inclusive_timing_by_budget": {
            f"k{key}": adaptive.timing_stats(values)
            for key, values in by_budget_ms.items()
        },
        "setup_timing": adaptive.timing_stats(setup_ms),
        "solve_timing": adaptive.timing_stats(solve_ms),
    }
    summary["retained_step_audits"] = step_audits
    snapshots = {
        "body_q": np.stack([record["body_q"] for _, record in rendered]),
        "body_qd": np.stack([record["body_qd"] for _, record in rendered]),
        "render_time_s": np.array([step * DT for step, _ in rendered]),
        "gap_max_m": np.array(
            [record["metrics"]["gap_3d_m"]["max"] for _, record in rendered]
        ),
        "gap_rms_m": np.array(
            [record["metrics"]["gap_3d_m"]["rms"] for _, record in rendered]
        ),
        "bend_max_rad": np.array(
            [record["metrics"]["bend_rest_angle_rad"]["max"] for _, record in rendered]
        ),
        "mechanical_total_J": np.array(
            [record["energy"]["mechanical_total_J"] for _, record in rendered]
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
    return adaptive.phase_gate(summary)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    parity._write_npz(path, arrays)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def run() -> tuple[Path, dict[str, Any]]:
    owner_start = owner_guard("preregistered K24 campaign start")
    if sha256_file(DOC) != PREREGISTERED_DOC_SHA256:
        raise RuntimeError("research doc changed after K24 preregistration")
    if sha256_file(Path(adaptive.__file__).resolve()) != BASE_ADAPTIVE_SOURCE_SHA256:
        raise RuntimeError("base adaptive solver source changed")
    prefix, step1200 = load_bound_step1200()

    with k24_budget_scope():
        probe, probe_snapshots, step1205 = run_segment(
            scenes.long_cantilever_512_bend,
            global_start_step=SOURCE_STEP,
            physical_steps=PROBE_STEP - SOURCE_STEP,
            initial_arrays=step1200,
            progress_label="preregistered_step1205_probe",
            retain_all_step_audits=True,
        )
        probe_pass, probe_gates = phase_gate(probe)
        continuation = None
        continuation_gates = None
        continuation_pass = False
        continuation_snapshots = None
        if probe_pass:
            continuation, continuation_snapshots, _ = run_segment(
                scenes.long_cantilever_512_bend,
                global_start_step=PROBE_STEP,
                physical_steps=FULL_STEP - PROBE_STEP,
                initial_arrays=step1205,
                progress_label="preregistered_continue_to_frame193",
                retain_all_step_audits=False,
            )
            continuation_pass, continuation_gates = phase_gate(continuation)

    full_gate = bool(probe_pass and continuation_pass)
    owner_end = owner_guard("preregistered K24 campaign end")
    if sha256_file(DOC) != PREREGISTERED_DOC_SHA256:
        raise RuntimeError("research doc changed during K24 campaign")

    run_dir = OUT / f"run_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}"
    owner_guard("K24 immutable output directory")
    run_dir.mkdir(parents=True, exist_ok=False)
    step1205_arrays = {
        "body_q": step1205["body_q"],
        "body_qd": step1205["body_qd"],
        "global_step": np.array([PROBE_STEP], dtype=np.int64),
        "time_s": np.array([PROBE_STEP * DT], dtype=np.float64),
    }
    files = {
        "step1205_probe_state": _write_npz(
            run_dir / "step1205_probe_state.npz", step1205_arrays
        )
    }
    if full_gate and continuation_snapshots is not None:
        full_snapshots = {}
        for key in prefix:
            tail = continuation_snapshots[key]
            # Continuation includes off-cadence step1205 as its first record.
            tail = tail[1:]
            full_snapshots[key] = np.concatenate([prefix[key], tail], axis=0)
        expected_time = np.arange(FULL_STEP // RENDER_STRIDE + 1) / 60.0
        if not np.array_equal(full_snapshots["render_time_s"], expected_time):
            raise RuntimeError("combined frame193 render cadence mismatch")
        files["full_render_states"] = _write_npz(
            run_dir / "n512_k24_full_render_states.npz", full_snapshots
        )
    else:
        files["probe_render_states"] = _write_npz(
            run_dir / "n512_k24_probe_render_states.npz", probe_snapshots
        )

    payload = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "device": str(wp.get_device()),
        "owner_line_start": owner_start,
        "owner_line_end": owner_end,
        "preregistered_doc_sha256": PREREGISTERED_DOC_SHA256,
        "source_binding": {
            "adaptive_result": str(SOURCE_RESULT.relative_to(ROOT)),
            "adaptive_result_sha256": SOURCE_RESULT_SHA256,
            "step1200_snapshots": str(SOURCE_SNAPSHOT.relative_to(ROOT)),
            "step1200_snapshots_sha256": SOURCE_SNAPSHOT_SHA256,
            "step1200_index": 120,
            "step1200_time_s": 2.0,
        },
        "sources": {
            "codex_hard_kkt_k24_probe.py": sha256_file(Path(__file__).resolve()),
            "codex_hard_kkt_adaptive_n512.py": sha256_file(
                Path(adaptive.__file__).resolve()
            ),
            "codex_hard_kkt_bgn.py": sha256_file(Path(hard.__file__).resolve()),
        },
        "thresholds_unchanged": {
            "hard_feasibility_m": traj.FEASIBILITY_TOL_M,
            "hard_stationarity": traj.STATIONARITY_TOL,
            "linear_original_kkt": traj.LINEAR_KKT_TOL,
            "quaternion_norm_max_error": 2.0e-5,
            "escalation_trigger": adaptive.ESCALATION_TRIGGER,
        },
        "adaptation": {
            "budgets": list(K24_BUDGETS),
            "only_new_budget": 24,
            "same_predictor_and_implicit_solve": True,
            "internal_time_substeps": 0,
            "k32_allowed": False,
            "tolerance_change_allowed": False,
        },
        "n512": {
            "step1205_probe": {
                "summary": probe, "gates": probe_gates, "pass": probe_pass
            },
            "continuation_to_frame193": (
                {
                    "summary": continuation,
                    "gates": continuation_gates,
                    "pass": continuation_pass,
                }
                if continuation is not None else None
            ),
            "full_gate": full_gate,
        },
        "files": files,
        "n1024_executed": False,
        "n1024_admission_gate": full_gate,
        "claim_boundary": (
            "preregistered hash-bound N512 no-contact K24-only budget extension; "
            "unchanged hard gates; no K32, tolerance change, contact, N1024, "
            "production, novelty, or universal-winner claim"
        ),
    }
    result = run_dir / "result.json"
    encoded = (
        json.dumps(al._strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = result.with_name(f".{result.name}.{uuid.uuid4().hex}.tmp")
    owner_guard("K24 result write")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    owner_guard("K24 result publish")
    os.replace(temporary, result)
    return result, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("only --run is supported")
    print(json.dumps({
        "event": "start", "pid": os.getpid(), "source_step": SOURCE_STEP,
        "probe_step": PROBE_STEP, "budgets": K24_BUDGETS, "n1024": False,
    }), flush=True)
    result, payload = run()
    print(json.dumps({
        "event": "complete", "result": str(result),
        "result_sha256": sha256_file(result),
        "step1205_probe_pass": payload["n512"]["step1205_probe"]["pass"],
        "frame193_full_gate": payload["n512"]["full_gate"],
        "n1024_executed": payload["n1024_executed"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
