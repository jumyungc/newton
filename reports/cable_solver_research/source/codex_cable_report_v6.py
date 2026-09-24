#!/usr/bin/env python3
"""Build the concise v6 cable-solver results review.

The builder is deliberately add-only and publication-free.  It consumes
hash-pinned results and three source-bound videos, writes one new exact-set
stage, fully decodes every copied video, and never invokes Git or touches a
Pages worktree.  Internal research discussion is not an input.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import imageio_ffmpeg


SCHEMA = "codex-cable-results-review/v6"
MANIFEST_SCHEMA = "codex-cable-results-review-manifest/v6"
QA_SCHEMA = "codex-cable-results-review-qa/v6"
EXPECTED_OWNER = (
    "OWNER: Team Codex-Remote — 2026-07-05T11:40Z — "
    "coupled long-chain/contact solver invention round"
)
ARCHIVE_URL = (
    "https://github.com/jumyungc/newton/tree/"
    "76483b1df266bf15ade2eb3397005cbe630d6fec/"
    "reports/cable_solver_research"
)

EXPECTED_HASH = {
    "legacy_manifest": "e92ec18f60ba78a38dbba62c20e21396149f8e830b2ae7bfb6e83c86d84bd517",
    "legacy_report": "27c22e17ee85b9542fef15858eca86a09e2cfb6d9ea0a4ba223d2205cfaa3724",
    "smooth_report": "98c61e348234b3b2cfb0b1f1f1d63a8ea8144cd1f4a05951bb18419112828460",
    "dense_robustness": "789055c31ef4bb35b87265e87700255cf36a481075c57da5af899f61013ad83f",
    "dense_visual_manifest": "c8553c992363ad69f29f820d44401e06dfc0bcbcc5699fa1e63471730384d8ea",
    "dense_video": "788508ab54f1893ba2be723f8e12169b8ea4698e773a23c3242e5d2bf9e0d338",
    "dense_poster": "d84b7a8e576c0cf55641a3165008f530ff0f7c9bd4229f5aedcd7e78f21ac3d8",
    "coverage": "557339dd10bbd26b30ac9e3276e2c7400b5eed61f4697119326fd52f1a18a535",
    "coverage_render": "8a891a390424dba2380332f1b56cddc5cf4fda215cf5fbdbaa303b214d99ee7f",
    "y_video": "4c02c0a4a9860730e161a350d0e2267f5334c5545bb17187999c5876bc717750",
    "y_poster": "689840dc2d4f09d48bfbe24847d02b749df87885172f5c10021829eb9c66eafc",
    "hard_result": "597305e713f3f0ac24b3829126d3d9c2fd61168c530acf35c8a0f68120e3be56",
    "hard_parity": "febfce298a81e358accc318a142b85ca1608c92fa39d836147cea32433b878ff",
    "hard_scale": "988f83b688350d5f561e79e96e862eb7be583eb0e1e0461a594a4178cd0562d7",
    "hard_k24": "6a19861f2110df1c73becfbb9ef75ef074db9cd80a367f2fb25629fa661c8f31",
    "hard_visual_manifest": "e9d1853d1ff7423ae619809c8510e76b0edf7ebdd11c3668193f83dc30e9938c",
    "hard_video": "2f89d0fcf515775fae0058e8df37f790bd6eff6d95c55df350c6d8affdc714d9",
    "hard_poster": "9b414ed15d745027c79bf30036da9ea8e6e8f554687302c46c517df34bdb0f31",
    "ccdsn": "7982dc5bd88e3b21ea53746152a6da38b29bde76bfefc032307b910e21c91f38",
    "contact_v2": "bd30d94f6de25a4c85120e892f799099b05c90a40eae69015d8692d47dc8ec8c",
    "contact_v3": "03d93f47ec4d1f97d70371fa558ad517154d2b986f7f97aeadcc9e3e68650060",
    "contact_v5": "640cf7fc826ae1dcda3aae38cb598a0f20cda06467d22dd14b9a5f1732238941",
}


class ReportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ReportError(f"non-finite JSON constant {value}")


def assert_finite(value: Any, where: str = "root") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{where}: non-finite float")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{where}[{index}]")


def strict_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot load strict JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: top level is not an object")
    assert_finite(value, str(path))
    return value


def stable_json(value: Any) -> bytes:
    assert_finite(value)
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def verify_hash(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    observed = sha256_file(path)
    require(observed == expected, f"{label} SHA mismatch: {observed}")


def guard_owner(root: Path, where: str) -> str:
    doc = root / "bench/global_cable/cable_research.md"
    line = doc.read_text(encoding="utf-8").splitlines()[0]
    require(line == EXPECTED_OWNER, f"owner mismatch before {where}: {line!r}")
    return line


def _source(root: Path, name: str, expected: str) -> Path:
    path = root / "bench/global_cable" / name
    verify_hash(path, expected, f"source {name}")
    return path


def default_paths(root: Path) -> dict[str, Path]:
    workspace = root / "bench/_workspace"
    legacy = workspace / "codex_cable_report_v5/stage_20260705T025500Z"
    return {
        "legacy_stage": legacy,
        "coverage": workspace / "fable_coverage/run_20260705T055525Z_1875833/coverage_manifest.json",
        "coverage_render": workspace / "fable_coverage/render_20260705/coverage_render_record.json",
        "y_video": workspace / "fable_coverage/render_20260705/y127_default_2x2.mp4",
        "y_poster": workspace / "fable_coverage/render_20260705/y127_default_poster.png",
        "hard_result": workspace / "codex_hard_kkt_trajectory/run_20260705T121845Z_1924599/result.json",
        "hard_parity": workspace / "codex_hard_kkt_parity/run_20260705T122437Z_1928744/result.json",
        "hard_scale": workspace / "codex_hard_kkt_adaptive_n512/run_20260705T123956Z_1940700/result.json",
        "hard_k24": workspace / "codex_hard_kkt_k24_probe/run_20260705T124923Z_1950768/result.json",
        "ccdsn": workspace / "codex_ccdsn_reference/run_20260705T121754Z_1925101/result.json",
        "contact_v2": workspace / "codex_contact_kkt_v2_pile/smoke_20260705T120615Z_1914274.json",
        "contact_v3": workspace / "codex_contact_kkt_v3_pile/smoke_20260705T122926Z_1934367.json",
        "contact_v5": workspace / "codex_contact_kkt_v5_pile/smoke_20260705T124305Z_1945429.json",
    }


def _validate_legacy(paths: Mapping[str, Path]) -> dict[str, Any]:
    stage = paths["legacy_stage"]
    verify_hash(stage / "manifest.json", EXPECTED_HASH["legacy_manifest"], "v5 manifest")
    verify_hash(stage / "report.json", EXPECTED_HASH["legacy_report"], "v5 report")
    verify_hash(stage / "data/smooth/base_report_v4.json", EXPECTED_HASH["smooth_report"], "smooth report")
    verify_hash(stage / "data/contact/dense_robustness.json", EXPECTED_HASH["dense_robustness"], "dense robustness")
    verify_hash(stage / "data/visual/dense_manifest.json", EXPECTED_HASH["dense_visual_manifest"], "dense visual manifest")
    verify_hash(stage / "media/dense/dense_contact_comparison.mp4", EXPECTED_HASH["dense_video"], "dense video")
    verify_hash(stage / "media/dense/dense_contact_comparison_poster.png", EXPECTED_HASH["dense_poster"], "dense poster")
    report = strict_load(stage / "report.json")
    dense = strict_load(stage / "data/contact/dense_robustness.json")
    visual = strict_load(stage / "data/visual/dense_manifest.json")
    require(report.get("schema") == "codex-cable-results-review/v5", "v5 report schema changed")
    summary = report.get("summary", {})
    require(summary.get("smooth_cases_pass") == summary.get("smooth_cases_executed") == 20,
            "smooth 20/20 claim changed")
    require(summary.get("largest_smooth_dynamic") == 8191, "largest smooth case changed")
    speeds = report.get("speed", [])
    require(len(speeds) == 5 and all(row.get("samples_each") == 50 for row in speeds),
            "matched smooth speed evidence changed")
    require(dense.get("all_policy_repeats_gate_pass") is True, "dense repeat gate changed")
    require(dense.get("selected_robust_dense_default") == "soft_contact", "dense default changed")
    for key in ("alpha000_nohistory_latest", "soft_contact"):
        row = dense["summaries"][key]
        require(row["gate_pass_count"] == row["repeat_count"] == 5, f"{key}: not 5/5")
    require(visual.get("schema") == "codex-bgn-dense-contact-visual-comparison/v1",
            "dense visual schema changed")
    require(visual.get("status") == "VALID_EXACT_DENSE_CONTACT_VISUAL_EVIDENCE",
            "dense visual status changed")
    require(all(value is True for value in visual.get("quality_gates", {}).values()),
            "dense visual quality gate failed")
    return {"report": report, "dense": dense, "dense_visual": visual}


def _validate_coverage(paths: Mapping[str, Path]) -> dict[str, Any]:
    verify_hash(paths["coverage"], EXPECTED_HASH["coverage"], "coverage campaign")
    verify_hash(paths["coverage_render"], EXPECTED_HASH["coverage_render"], "coverage render")
    verify_hash(paths["y_video"], EXPECTED_HASH["y_video"], "Y127 video")
    verify_hash(paths["y_poster"], EXPECTED_HASH["y_poster"], "Y127 poster")
    campaign = strict_load(paths["coverage"])
    render = strict_load(paths["coverage_render"])
    require(campaign.get("schema") == "fable-coverage-campaign/v1", "coverage schema changed")
    require(render.get("schema") == "fable-coverage-render/v1", "coverage render schema changed")
    require(render.get("coverage_manifest_sha256") == EXPECTED_HASH["coverage"],
            "coverage/render binding changed")
    y = render.get("scenes", {}).get("y127_default", {})
    require(y.get("video_sha256") == EXPECTED_HASH["y_video"], "Y127 video record changed")
    require(y.get("poster_sha256") == EXPECTED_HASH["y_poster"], "Y127 poster record changed")
    require(y.get("frames_decoded") == 301, "Y127 decoded frame count changed")
    return {"coverage": campaign, "coverage_render": render}


def _validate_hard(paths: Mapping[str, Path]) -> dict[str, Any]:
    verify_hash(paths["hard_result"], EXPECTED_HASH["hard_result"], "hard trajectory")
    verify_hash(paths["hard_parity"], EXPECTED_HASH["hard_parity"], "hard parity")
    scale_hash = str(paths.get("hard_scale_sha256", EXPECTED_HASH["hard_scale"]))
    verify_hash(paths["hard_scale"], scale_hash, "hard N512 scale boundary")
    k24_hash = str(paths.get("hard_k24_sha256", EXPECTED_HASH["hard_k24"]))
    verify_hash(paths["hard_k24"], k24_hash, "hard N512 K24 probe")
    verify_hash(paths["hard_visual_manifest"], EXPECTED_HASH["hard_visual_manifest"], "hard visual manifest")
    verify_hash(paths["hard_video"], EXPECTED_HASH["hard_video"], "hard video")
    verify_hash(paths["hard_poster"], EXPECTED_HASH["hard_poster"], "hard poster")
    trajectory = strict_load(paths["hard_result"])
    parity = strict_load(paths["hard_parity"])
    scale = strict_load(paths["hard_scale"])
    k24 = strict_load(paths["hard_k24"])
    visual = strict_load(paths["hard_visual_manifest"])
    require(trajectory.get("schema") == "codex-hard-equality-feasible-trajectory/v1",
            "hard trajectory schema changed")
    require(parity.get("schema") == "codex-hard-kkt-same-semantics-parity/v1",
            "hard parity schema changed")
    require(trajectory["n128"]["hard"]["pass"] is True, "N128 hard trajectory failed")
    require(trajectory["n128"]["hard"]["completed_substeps"] == 600,
            "N128 hard trajectory incomplete")
    require(trajectory["n128"]["gate"]["pass"] is False,
            "cross-semantics N128 gate disclosure changed")
    require(parity["n128"]["pass"] is True and parity["n128"]["k2_vs_k12"]["pass"] is True,
            "same-hard N128 parity failed")
    require(parity["n64_dt_refinement"]["pass"] is True, "hard dt refinement failed")
    require(parity.get("n1024_executed") is False, "N1024 scope changed")
    require(scale.get("schema") == "codex-hard-kkt-residual-adaptive-n512/v1",
            "hard N512 scale schema changed")
    phase1 = scale["n512"]["phase1_600_steps"]
    phase2 = scale["n512"]["phase2_to_frame193"]
    require(phase1["pass"] is True and phase1["summary"]["completed_substeps"] == 600,
            "hard N512 phase1 boundary changed")
    require(scale["n512"]["full_gate"] is False and phase2["pass"] is False,
            "hard N512 continuation disclosure changed")
    require(phase2["summary"]["completed_substeps"] == 604 and
            phase2["summary"]["failure"]["substep"] == 605,
            "hard N512 failure step changed")
    require(phase2["summary"]["failure"]["budget_used"] == 12,
            "hard N512 terminal budget changed")
    require(phase2["summary"]["failure"]["certificate"]["stationarity_relative"] > 5e-5,
            "hard N512 failure certificate changed")
    require(k24.get("schema") == "codex-hard-kkt-preregistered-k24-probe/v1",
            "hard N512 K24 schema changed")
    require(k24["n512"]["step1205_probe"]["pass"] is True and
            k24["n512"]["step1205_probe"]["summary"]["global_completed_step"] == 1205,
            "hard N512 K24 admission changed")
    k24_continuation = k24["n512"]["continuation_to_frame193"]
    require(k24["n512"]["full_gate"] is False and k24_continuation["pass"] is False,
            "hard N512 K24 full-gate disclosure changed")
    require(k24_continuation["summary"]["failure"]["global_step"] == 1209 and
            k24_continuation["summary"]["failure"]["budget_used"] == 24,
            "hard N512 K24 failure boundary changed")
    require(k24_continuation["summary"]["failure"]["final_hard_kkt_preassign"]
            ["stationarity_relative"] > 5e-5,
            "hard N512 K24 terminal certificate changed")
    require(k24.get("n1024_executed") is False, "hard N512 K24 scope changed")
    require(visual.get("schema") == "codex-hard-kkt-four-panel-review/v1",
            "hard visual schema changed")
    require(visual.get("source_result", {}).get("sha256") == EXPECTED_HASH["hard_result"],
            "hard visual/result binding changed")
    require(visual.get("media_qa", {}).get("all_pass") is True,
            "hard visual quality gate failed")
    outputs = visual.get("outputs", {})
    require(outputs.get("video", {}).get("sha256") == EXPECTED_HASH["hard_video"],
            "hard video record changed")
    require(outputs.get("poster", {}).get("sha256") == EXPECTED_HASH["hard_poster"],
            "hard poster record changed")
    sources = parity["sources"]
    _source(paths["root"], "codex_hard_kkt_bgn.py", sources["codex_hard_kkt_bgn.py"])
    _source(paths["root"], "codex_hard_kkt_parity.py", sources["codex_hard_kkt_parity.py"])
    _source(paths["root"], "codex_hard_kkt_trajectory.py", sources["codex_hard_kkt_trajectory.py"])
    for name, expected in scale["sources"].items():
        _source(paths["root"], name, expected)
    for name, expected in k24["sources"].items():
        _source(paths["root"], name, expected)
    _source(paths["root"], "codex_hard_kkt_review_render.py",
            visual["source_script"]["sha256"])
    return {"hard": trajectory, "parity": parity, "hard_scale": scale,
            "hard_k24": k24, "hard_visual": visual}


def _validate_frontier(paths: Mapping[str, Path]) -> dict[str, Any]:
    for key in ("ccdsn", "contact_v2", "contact_v3"):
        verify_hash(paths[key], EXPECTED_HASH[key], key)
    v5_hash = str(paths.get("contact_v5_sha256", EXPECTED_HASH["contact_v5"]))
    verify_hash(paths["contact_v5"], v5_hash, "contact v5")
    ccdsn = strict_load(paths["ccdsn"])
    v2 = strict_load(paths["contact_v2"])
    v3 = strict_load(paths["contact_v3"])
    v5 = strict_load(paths["contact_v5"])
    require(ccdsn.get("schema") == "codex-ccdsn-small-reference/v1" and ccdsn.get("pass") is True,
            "CCDSN small oracle failed")
    require(all(value is True for value in ccdsn.get("gates", {}).values()),
            "CCDSN oracle gate failed")
    require(v2.get("schema") == "codex-contact-kkt-v2-pile-lockstep/v1" and v2.get("status") == "FAIL",
            "contact v2 negative changed")
    require(v2["result"]["completed_physical_steps"] == 33 and
            v2["result"]["requested_physical_steps"] == 60,
            "contact v2 partial boundary changed")
    require(v2["result"]["samples"][-2]["physical_step"] == 32 and
            v2["result"]["samples"][-2]["candidate"]["pass"] is True and
            v2["result"]["samples"][-1]["physical_step"] == 33 and
            v2["result"]["samples"][-1]["candidate"]["pass"] is False,
            "contact v2 certified/failure-step boundary changed")
    require(v3.get("schema") == "codex-contact-kkt-v3-pile-lockstep/v1" and v3.get("status") == "FAIL",
            "contact v3 negative changed")
    require(v3["result"]["certified_physical_steps"] == 45 and
            v3["result"]["requested_physical_steps"] == 60,
            "contact v3 partial boundary changed")
    require(v5.get("schema") == "codex-contact-kkt-v5-pile-lockstep/v1" and
            v5.get("status") == "PASS", "contact v5 full result changed")
    v5_result = v5["result"]
    require(v5_result["completed_physical_steps"] ==
            v5_result["certified_physical_steps"] ==
            v5_result["requested_physical_steps"] == 60,
            "contact v5 trajectory is not 60/60 certified")
    require(v5_result["candidate"]["pass"] is True and v5_result["failure"] is None,
            "contact v5 candidate no longer passes")
    require(v5_result["case"]["body_count"] == 48, "contact v5 scope changed")
    require(all(v5_result["candidate"]["execution_certificates"].values()),
            "contact v5 execution certificate failed")
    final_kkt = v5_result["samples"][-1]["candidate"]["final_kkt"]
    require(final_kkt["desaxce"]["certificate_pass"] is True and
            final_kkt["desaxce_primal_cone_violation"] == 0.0 and
            final_kkt["stationarity_inf"] < 1e-8,
            "contact v5 final exact-law certificate failed")
    root = paths["root"]
    for name, expected in (
        ("codex_ccdsn_reference_benchmark.py", ccdsn["source_sha256"]),
        ("codex_ccdsn.py", ccdsn["solver_sha256"]),
        ("codex_desaxce.py", ccdsn["desaxce_sha256"]),
        ("test_codex_ccdsn.py", ccdsn["test_sha256"]),
        ("codex_contact_kkt_v2_pile.py", v2["source_sha256"]),
        ("codex_contact_kkt_v2.py", v2["solver_sha256"]),
        ("test_codex_contact_kkt_v2.py", v2["solver_test_sha256"]),
        ("codex_contact_kkt_v3_pile.py", v3["source_sha256"]),
        ("codex_contact_kkt_v3.py", v3["solver_sha256"]),
        ("test_codex_contact_kkt_v3.py", v3["solver_test_sha256"]),
        ("codex_contact_safe_predictor.py", v3["predictor_sha256"]),
        ("test_codex_contact_safe_predictor.py", v3["predictor_test_sha256"]),
        ("codex_contact_kkt_v5_pile.py", v5["source_sha256"]),
        ("codex_contact_kkt_v5.py", v5["solver_sha256"]),
        ("test_codex_contact_kkt_v5.py", v5["solver_test_sha256"]),
    ):
        _source(root, name, expected)
    return {"ccdsn": ccdsn, "contact_v2": v2, "contact_v3": v3,
            "contact_v5": v5}


def validate_inputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    values.update(_validate_legacy(paths))
    values.update(_validate_coverage(paths))
    values.update(_validate_hard(paths))
    values.update(_validate_frontier(paths))
    return values


def _row_from_dense(report: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    row = next((item for item in report["dense"] if item["label"] == label), None)
    require(row is not None, f"dense row missing: {label}")
    return row


def make_report(values: Mapping[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    legacy = values["report"]
    hard = values["hard"]["n128"]["hard"]
    parity = values["parity"]["n128"]
    k2k12 = parity["k2_vs_k12"]
    analytic = values["ccdsn"]["analytic_sliding"]
    mixed = values["ccdsn"]["mixed_stick_slide"]
    v2 = values["contact_v2"]["result"]
    v3 = values["contact_v3"]["result"]
    v5 = values["contact_v5"]["result"]
    v5_candidate = v5["candidate"]
    v5_baseline = v5["vbd10"]
    v5_kkt = v5["samples"][-1]["candidate"]["final_kkt"]
    scale = values["hard_scale"]["n512"]
    scale_phase1 = scale["phase1_600_steps"]["summary"]
    scale_phase2 = scale["phase2_to_frame193"]["summary"]
    k24 = values["hard_k24"]["n512"]
    k24_admission = k24["step1205_probe"]["summary"]
    k24_continuation = k24["continuation_to_frame193"]["summary"]
    old = _row_from_dense(legacy, "hard/default canonical")
    hard_dense = _row_from_dense(legacy, "hard alpha0 / history off / latest")
    soft_dense = _row_from_dense(legacy, "penalty-only / history off")
    speeds = legacy["speed"]
    finite_bgn = next(row for row in values["hard"]["n128"]["finite_baselines"]
                      if row["method"] == "finite_compact_bgn_k5")
    vbd10 = next(row for row in values["hard"]["n128"]["finite_baselines"]
                 if row["method"] == "requested_vbd_10")
    return {
        "schema": SCHEMA,
        "publication_ready": False,
        "immutable_stage": True,
        "universal_solver_established": False,
        "verdict": "BEST VERIFIED RESULT IS A ROUTER, NOT ONE UNIVERSAL SOLVER",
        "summary": {
            "visible_videos": 3,
            "hard_n128_substeps": hard["completed_substeps"],
            "hard_n128_peak_axial_m": hard["axial_abs_m"]["trajectory_max"],
            "hard_k2_vs_k12_full_state_relative_l2": k2k12["full_state"]["full_state_relative_l2"],
            "hard_k2_vs_k12_position_rms_m": k2k12["full_state"]["position_rms_m"],
            "soft_prepared_speedup_min": min(row["prepared_global_speedup"] for row in speeds),
            "soft_prepared_speedup_max": max(row["prepared_global_speedup"] for row in speeds),
            "soft_smooth_passes": legacy["summary"]["smooth_cases_pass"],
            "soft_smooth_executed": legacy["summary"]["smooth_cases_executed"],
            "largest_soft_smooth_dynamic": legacy["summary"]["largest_smooth_dynamic"],
            "dense_selected_repeats": soft_dense["repeats"],
            "ccdsn_small_oracle_pass": values["ccdsn"]["pass"],
            "contact_v3_certified_steps": v3["certified_physical_steps"],
            "contact_v3_requested_steps": v3["requested_physical_steps"],
            "unified_contact_v5_pass": v5_candidate["pass"],
            "unified_contact_v5_steps": v5["certified_physical_steps"],
            "hard_n512_phase1_pass": scale["phase1_600_steps"]["pass"],
            "hard_n512_full_gate": scale["full_gate"],
        },
        "routing": [
            {
                "regime": "Hard/inextensible, no contact (N128 evidence)",
                "route": "hard equality KKT K2",
                "measured": "600/600 steps; 0.477 µm peak axial error; same-hard K12 parity passes",
                "boundary": "accuracy specialist; 47.25 ms/substep p50 CPU diagnostic; no contact or N1024 claim",
            },
            {
                "regime": "Authored finite-compliance smooth cable",
                "route": "prepared compact BGN",
                "measured": "3.065–5.046× over five matched targets; 20/20 broad executions",
                "boundary": "soft-material semantics; matched speed evidence is five Y systems",
            },
            {
                "regime": "7,680-body dense contact",
                "route": "VBD10 penalty-only, contact history off",
                "measured": "5/5 gate passes; 6.853 mm worst penetration; 0.463 mm worst stretch",
                "boundary": "selected scene specialist; discrete contact-mode change; not bitwise repeatable",
            },
            {
                "regime": "Exact coupled stretch + Coulomb law",
                "route": "contact KKT v5 + CCDSN oracle",
                "measured": "48-body endpoint-safe smoke passes 60/60; exact-law certificates pass",
                "boundary": "34× slower than VBD10; no swept CCD, dense pile, or scale claim",
            },
        ],
        "hard_n128": {
            "hard": hard,
            "finite_bgn": finite_bgn,
            "finite_vbd10": vbd10,
            "cross_semantics_gate_pass": values["hard"]["n128"]["gate"]["pass"],
            "same_hard": {
                "k2_vs_k12": k2k12,
                "k5_vs_k12": parity["k5_vs_k12"],
                "pass": parity["pass"],
            },
        },
        "hard_n512_scale_boundary": {
            "phase1_pass": scale["phase1_600_steps"]["pass"],
            "full_gate": scale["full_gate"],
            "phase1_steps": scale_phase1["completed_substeps"],
            "phase1_peak_gap_m": scale_phase1["gap_3d_m"]["trajectory_max"],
            "phase1_p50_ms": scale_phase1["timing_descriptive"]["stage_ms_p50"],
            "phase1_p95_ms": scale_phase1["timing_descriptive"]["stage_ms_p95"],
            "phase1_budget_counts": scale_phase1["residual_adaptation"]["budget_counts"],
            "continuation_completed_steps": scale_phase2["completed_substeps"],
            "total_completed_steps": scale_phase1["completed_substeps"] + scale_phase2["completed_substeps"],
            "failed_total_step": scale_phase1["completed_substeps"] + scale_phase2["failure"]["substep"],
            "total_completed_time_s": scale_phase1["duration_s"] + scale_phase2["duration_s"],
            "terminal_budget": scale_phase2["failure"]["budget_used"],
            "terminal_stationarity": scale_phase2["failure"]["certificate"]["stationarity_relative"],
            "stationarity_limit": scale_phase2["residual_adaptation"]["stationarity_gate"],
            "k24_step1205_pass": k24["step1205_probe"]["pass"],
            "k24_step1205_stationarity": k24_admission["hard_kkt_worst"]["stationarity_relative"],
            "k24_full_gate": k24["full_gate"],
            "k24_failure_global_step": k24_continuation["failure"]["global_step"],
            "k24_failure_budget": k24_continuation["failure"]["budget_used"],
            "k24_failure_accepted_attempts": k24_continuation["failure"]["accepted_steps"],
            "k24_failure_stationarity": (
                k24_continuation["failure"]["final_hard_kkt_preassign"]
                ["stationarity_relative"]
            ),
        },
        "soft_smooth": {
            "matched_speed": speeds,
            "broad_pass": legacy["summary"]["smooth_cases_pass"],
            "broad_total": legacy["summary"]["smooth_cases_executed"],
            "largest_dynamic_bodies": legacy["summary"]["largest_smooth_dynamic"],
            "visual_claim": "qualitative Track-B trajectory; overlays are not canonical R",
        },
        "dense_contact": [old, hard_dense, soft_dense],
        "ccdsn_oracle": {
            "pass": values["ccdsn"]["pass"],
            "claim_boundary": values["ccdsn"]["claim_boundary"],
            "analytic": {
                "maximum_force_error_N": analytic["maximum_force_error_N"],
                "natural_map_inf": analytic["info"]["natural_map_inf"],
                "stationarity_inf": analytic["info"]["stationarity_inf"],
                "stretch_residual_inf_m": analytic["info"]["stretch_residual_inf_m"],
                "friction_work_Nm": analytic["info"]["friction_work_Nm"],
            },
            "mixed": {
                "natural_map_inf": mixed["info"]["natural_map_inf"],
                "stationarity_inf": mixed["info"]["stationarity_inf"],
                "stretch_residual_inf_m": mixed["info"]["stretch_residual_inf_m"],
                "friction_work_Nm": mixed["info"]["friction_work_Nm"],
                "sticking_contacts": len(mixed["info"]["friction_law"]["sticking_contact_indices"]),
                "sliding_contacts": len(mixed["info"]["friction_law"]["sliding_contact_indices"]),
            },
        },
        "unified_contact_v5": {
            "pass": v5_candidate["pass"],
            "body_count": v5["case"]["body_count"],
            "certified_steps": v5["certified_physical_steps"],
            "requested_steps": v5["requested_physical_steps"],
            "peak_penetration_m": v5_candidate["peak_penetration_m"],
            "final_stretch_m": v5_candidate["final_cable"]["stretch_max_m"],
            "vbd10_peak_penetration_m": v5_baseline["peak_penetration_m"],
            "vbd10_final_stretch_m": v5_baseline["final_cable"]["stretch_max_m"],
            "penetration_improvement": (
                v5_baseline["peak_penetration_m"] / v5_candidate["peak_penetration_m"]
            ),
            "stretch_improvement": (
                v5_baseline["final_cable"]["stretch_max_m"] /
                v5_candidate["final_cable"]["stretch_max_m"]
            ),
            "setup_inclusive_ms": v5_candidate["timing"]["setup_inclusive_ms"],
            "vbd10_setup_inclusive_ms": v5_baseline["timing"]["setup_inclusive_ms"],
            "slowdown": (
                v5_candidate["timing"]["setup_inclusive_ms"] /
                v5_baseline["timing"]["setup_inclusive_ms"]
            ),
            "final_exact_law": {
                "natural_map_relative_inf": v5_kkt["desaxce_natural_map_relative_inf"],
                "primal_cone_violation": v5_kkt["desaxce_primal_cone_violation"],
                "dual_cone_violation": v5_kkt["desaxce"]["dual_cone_violation"],
                "complementarity_inf": v5_kkt["desaxce"]["complementarity_inf"],
                "stationarity_inf": v5_kkt["stationarity_inf"],
            },
            "continuous_path_certified": False,
            "claim_boundary": values["contact_v5"]["result"]["claim_boundary"],
        },
        "contact_negatives": [
            {
                "version": "v2",
                "status": values["contact_v2"]["status"],
                "certified_steps": v2["samples"][-2]["physical_step"],
                "failure_attempted_step": v2["samples"][-1]["physical_step"],
                "requested": v2["requested_physical_steps"],
                "peak_penetration_m": v2["candidate"]["peak_penetration_m"],
                "peak_stretch_m": v2["candidate"]["final_cable"]["stretch_max_m"],
                "solver_ms": v2["candidate"]["elapsed_ms"],
                "failure": "solver gate failure at attempted step 33",
            },
            {
                "version": "v3",
                "status": values["contact_v3"]["status"],
                "certified_steps": v3["certified_physical_steps"],
                "failure_attempted_step": v3["failure"]["attempted_physical_step"],
                "requested": v3["requested_physical_steps"],
                "peak_penetration_m": v3["candidate"]["peak_penetration_m"],
                "peak_stretch_m": v3["candidate"]["final_cable"]["stretch_max_m"],
                "solver_ms": v3["candidate"]["timing"]["physical_step_solver_ms"],
                "failure": f"{v3['failure']['kind']} at attempted step {v3['failure']['attempted_physical_step']}",
            },
        ],
        "archive": {
            "label": "full previous v5 report and evidence",
            "url": ARCHIVE_URL,
            "immutable_commit": "76483b1df266bf15ade2eb3397005cbe630d6fec",
        },
        "claim_boundaries": [
            "Hard and finite rows use different material semantics; their visual gap contrast is not a same-physics accuracy ranking.",
            "The hard N128 result is accurate but not fast: timing is a single-process CPU diagnostic and no contact is present.",
            "N512 passes its first 600 steps; K24 rescues step 1205 but fails closed at step 1209, so it remains a scale boundary.",
            "The soft smooth speed claim is inherited from five matched Y targets; broad rows are accuracy audits.",
            "The dense contact result is one authored scene under frozen 12/10-mm gates and is not bitwise repeatable.",
            "Unified contact v5 passes only a 48-body, 60-step endpoint-safe smoke and is 34× slower; no swept path is certified.",
            "CCDSN is exact-law on small dense oracles; v2 and v3 remain failed partial trajectories.",
            "No K32 or N1024 probe was run; no universal, production, GPU, long-chain-contact, novelty, or paper-best claim is made.",
        ],
        "evidence": evidence,
    }


def _table(headers: list[str], rows: Iterable[list[str]], label: str) -> str:
    head = "".join(f'<th scope="col">{escape(item)}</th>' for item in headers)
    body = "".join(
        "<tr>" + "".join(
            f'<td data-label="{escape(headers[index])}">{item}</td>'
            for index, item in enumerate(row)
        ) + "</tr>" for row in rows
    )
    return (
        f'<div class="table-scroll" role="region" aria-label="{escape(label)}" tabindex="0">'
        f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def _badge(passed: bool, text: str | None = None) -> str:
    label = text or ("PASS" if passed else "FAIL")
    return f'<span class="badge {"pass" if passed else "fail"}">{escape(label)}</span>'


def _metric(value: float, unit: str = "") -> str:
    if value == 0.0:
        return f"0 {unit}".strip()
    return f"{value:.3e} {unit}".strip()


def render_html(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    routing = _table(
        ["Regime", "Verified route", "Measured evidence", "Boundary"],
        [[escape(row["regime"]), f'<strong>{escape(row["route"])}</strong>',
          escape(row["measured"]), escape(row["boundary"])] for row in report["routing"]],
        "Verified solver routing",
    )
    hard = report["hard_n128"]
    hard_rows = [
        ["Hard KKT K2", "inextensible Track B",
         f'{1e6*hard["hard"]["axial_abs_m"]["trajectory_max"]:.3f} µm',
         f'{1e6*hard["hard"]["gap_3d_m"]["trajectory_max"]:.3f} µm',
         f'{hard["hard"]["timing_descriptive"]["stage_ms_p50"]:.2f} ms', _badge(True)],
        ["Compact BGN K5", "authored finite compliance",
         f'{1e3*hard["finite_bgn"]["axial_abs_m"]["trajectory_max"]:.3f} mm',
         f'{1e3*hard["finite_bgn"]["gap_3d_m"]["trajectory_max"]:.3f} mm',
         f'{hard["finite_bgn"]["timing_descriptive"]["stage_ms_p50"]:.2f} ms', _badge(True, "complete")],
        ["Requested VBD10", "authored finite compliance",
         f'{1e3*hard["finite_vbd10"]["axial_abs_m"]["trajectory_max"]:.3f} mm',
         f'{1e3*hard["finite_vbd10"]["gap_3d_m"]["trajectory_max"]:.3f} mm',
         f'{hard["finite_vbd10"]["timing_descriptive"]["stage_ms_p50"]:.2f} ms', _badge(True, "complete")],
    ]
    hard_table = _table(
        ["Method", "Material semantics", "Peak axial error", "Peak 3-D gap", "p50/substep", "run"],
        hard_rows, "N128 hard and finite trajectories",
    )
    parity = hard["same_hard"]["k2_vs_k12"]
    parity_table = _table(
        ["Same-hard comparison", "value", "frozen limit", "gate"],
        [
            ["full-state relative L2", f'{parity["full_state"]["full_state_relative_l2"]:.3e}', "1.0e-3", _badge(parity["full_state_relative_gate"])],
            ["position RMS", f'{1e9*parity["full_state"]["position_rms_m"]:.2f} nm', "1.0 mm", _badge(parity["position_gate"])],
            ["bend RMS difference", f'{parity["bend_absolute_differences_rad"]["final_joint_rms"]:.3e} rad', "1.0e-3 rad", _badge(parity["curvature_absolute_gate"])],
            ["final energy relative difference", f'{parity["final_energy_relative_difference"]:.3e}', "1.0e-2", _badge(parity["energy_gate"])],
        ],
        "N128 same-hard parity",
    )
    scale = report["hard_n512_scale_boundary"]
    scale_table = _table(
        ["N512 interval", "status", "gap / certificate", "timing", "adaptive budgets"],
        [
            ["first 600 steps", _badge(scale["phase1_pass"], "600/600 PASS"),
             f'{1e6*scale["phase1_peak_gap_m"]:.3f} µm peak gap',
             f'{scale["phase1_p50_ms"]:.2f} / {scale["phase1_p95_ms"]:.2f} ms p50/p95',
             f'K2 {scale["phase1_budget_counts"]["k2"]} · K5 {scale["phase1_budget_counts"]["k5"]} · K12 {scale["phase1_budget_counts"]["k12"]}'],
            ["K2/K5/K12 continuation", _badge(False, f'FAIL at total step {scale["failed_total_step"]}'),
             f'{scale["terminal_stationarity"]:.4e} stationarity > {scale["stationarity_limit"]:.1e}',
             f'{scale["total_completed_time_s"]:.4f} s simulated through step {scale["total_completed_steps"]}',
             f'K{scale["terminal_budget"]} exhausted'],
            ["preregistered K24 probe", _badge(False, f'1205 PASS; FAIL at {scale["k24_failure_global_step"]}'),
             f'{scale["k24_step1205_stationarity"]:.4e} at 1205; {scale["k24_failure_stationarity"]:.4e} terminal',
             "frame 193 not reached",
             f'K24 path stalled after {scale["k24_failure_accepted_attempts"]} accepted attempts; no K32'],
        ],
        "N512 hard scale boundary",
    )
    speed = _table(
        ["Matched soft target", "objective R", "prepared speedup", "cold speedup", "samples/method"],
        [[escape(row["label"]), f'{row["objective_r"]:.4f}',
          f'<strong>{row["prepared_global_speedup"]:.3f}×</strong>',
          f'{row["cold_global_speedup"]:.3f}×', str(row["samples_each"])]
         for row in report["soft_smooth"]["matched_speed"]],
        "Matched finite-compliance smooth targets",
    )
    dense = _table(
        ["7,680-body contact route", "scope", "worst penetration", "worst stretch", "peak contacts", "repeat gate"],
        [[f'<strong>{escape(row["label"])}</strong>', escape(row["scope"]),
          f'{1e3*row["penetration_m"]:.3f} mm', f'{1e3*row["stretch_m"]:.3f} mm',
          f'{int(round(row["peak_contacts"])):,}', _badge(row["pass"], row["repeats"])]
         for row in report["dense_contact"]],
        "Dense contact policies",
    )
    oracle = report["ccdsn_oracle"]
    oracle_table = _table(
        ["CCDSN small oracle", "natural map", "stationarity", "stretch residual", "friction work", "gate"],
        [
            ["analytic sliding", _metric(oracle["analytic"]["natural_map_inf"]),
             _metric(oracle["analytic"]["stationarity_inf"]),
             _metric(oracle["analytic"]["stretch_residual_inf_m"], "m"),
             f'{oracle["analytic"]["friction_work_Nm"]:.3f} N·m', _badge(oracle["pass"])],
            ["mixed stick + slide", _metric(oracle["mixed"]["natural_map_inf"]),
             _metric(oracle["mixed"]["stationarity_inf"]),
             _metric(oracle["mixed"]["stretch_residual_inf_m"], "m"),
             f'{oracle["mixed"]["friction_work_Nm"]:.3f} N·m', _badge(oracle["pass"])],
        ],
        "CCDSN exact-law oracle",
    )
    unified = report["unified_contact_v5"]
    law = unified["final_exact_law"]
    unified_table = _table(
        ["48-body contact smoke", "trajectory", "penetration", "final stretch", "setup-inclusive", "result"],
        [
            ["contact KKT v5", f'{unified["certified_steps"]}/{unified["requested_steps"]} certified',
             f'{1e6*unified["peak_penetration_m"]:.3f} µm',
             f'{1e6*unified["final_stretch_m"]:.3f} µm',
             f'{unified["setup_inclusive_ms"]:.1f} ms', _badge(unified["pass"])],
            ["production VBD10", "60/60 executed",
             f'{1e3*unified["vbd10_peak_penetration_m"]:.3f} mm',
             f'{1e3*unified["vbd10_final_stretch_m"]:.3f} mm',
             f'{unified["vbd10_setup_inclusive_ms"]:.1f} ms', _badge(False, "penetration FAIL")],
        ],
        "Unified contact v5 result",
    )
    law_table = _table(
        ["Final v5 exact-law certificate", "value", "gate"],
        [
            ["normalized natural map", f'{law["natural_map_relative_inf"]:.3e}', _badge(law["natural_map_relative_inf"] < 1e-8)],
            ["primal / dual cone", f'{law["primal_cone_violation"]:.3e} / {law["dual_cone_violation"]:.3e}', _badge(max(law["primal_cone_violation"], law["dual_cone_violation"]) < 1e-8)],
            ["complementarity", f'{law["complementarity_inf"]:.3e}', _badge(law["complementarity_inf"] < 1e-8)],
            ["stationarity", f'{law["stationarity_inf"]:.3e}', _badge(law["stationarity_inf"] < 1e-8)],
        ],
        "Unified contact exact-law certificate",
    )
    negatives = _table(
        ["Pile prototype", "certified before failure", "failure attempt", "peak penetration", "peak stretch", "solver time", "result"],
        [[escape(row["version"]), f'{row["certified_steps"]} certified',
          f'step {row["failure_attempted_step"]} of {row["requested"]}',
          f'{1e6*row["peak_penetration_m"]:.3f} µm', f'{1e6*row["peak_stretch_m"]:.3f} µm',
          f'{row["solver_ms"]:.1f} ms', _badge(False, row["failure"])]
         for row in report["contact_negatives"]],
        "Contact KKT negative trajectories",
    )
    limits = "".join(f"<li>{escape(item)}</li>" for item in report["claim_boundaries"])
    evidence = "".join(
        f'<li><a href="{escape(row["path"])}">{escape(row["label"])}</a>'
        f'<code>{escape(row["sha256"][:12])}…</code></li>' for row in report["evidence"]
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>Cable solver research — verified frontier</title>
<link rel="stylesheet" href="styles.css"></head><body>
<header class="hero"><div class="shell"><p class="eyebrow">CABLE SOLVER RESEARCH / VERIFIED FRONTIER</p>
<h1>Best verified result:<br><span>a router, not one universal solver.</span></h1>
<p class="lead">Hard N128 stretch is now genuinely near-inextensible and same-hard reference checked. Soft smooth cables remain fast with compact BGN. The 7,680-body pile still needs its measured contact specialist. Unified exact-law contact now passes a small endpoint-safe smoke, but it is 34× slower and not dense or swept-path certified.</p>
<div class="metrics"><div><strong>{1e6*summary["hard_n128_peak_axial_m"]:.3f} µm</strong><span>N128 peak hard axial error, 600 steps</span></div>
<div><strong>{summary["soft_prepared_speedup_min"]:.3f}–{summary["soft_prepared_speedup_max"]:.3f}×</strong><span>matched soft prepared speedup</span></div>
<div><strong>{summary["soft_smooth_passes"]}/{summary["soft_smooth_executed"]}</strong><span>executed soft smooth audits pass</span></div>
<div><strong>60/60</strong><span>small unified contact v5 PASS</span></div></div></div></header>
<main class="shell"><nav aria-label="Report sections"><a href="#verdict">Verdict</a><a href="#watch">Watch</a><a href="#hard">Hard N128</a><a href="#portfolio">Portfolio</a><a href="#frontier">Exact law</a><a href="#evidence">Evidence</a></nav>
<section id="verdict"><div class="section-head"><p>01 / VERDICT</p><h2>Four verified lanes; no universal champion</h2></div>{routing}
<div class="notice"><strong>Read this before comparing rows.</strong><ul>{limits}</ul></div></section>
<section id="watch"><div class="section-head"><p>02 / THREE SOURCE-BOUND VIDEOS</p><h2>One visual for each established lane</h2></div>
<article class="video-card primary"><div><p class="kicker">NEW / HARD STRETCH</p><h3>N128: hard equality against finite-compliance methods</h3><p>Four synchronized panels. Colour encodes absolute joint gap. The hard and finite rows intentionally use different material semantics; this shows the Track-B inextensible mode, not a same-physics ranking.</p></div>
<video controls preload="metadata" poster="media/hard/n128_hard_vs_finite_poster.png"><source src="media/hard/n128_hard_vs_finite.mp4" type="video/mp4"></video></article>
<div class="video-grid"><article class="video-card"><p class="kicker">SOFT SMOOTH</p><h3>Y127: VBD2 / VBD10 / VBD80 / global</h3><video controls preload="metadata" poster="media/smooth/y127_poster.png"><source src="media/smooth/y127.mp4" type="video/mp4"></video><p>Qualitative Track-B trajectory with frozen routing overlays; the overlay is not canonical objective R.</p></article>
<article class="video-card"><p class="kicker">DENSE CONTACT</p><h3>7,680 bodies: old failure vs passing policies</h3><video controls preload="metadata" poster="media/contact/dense_poster.png"><source src="media/contact/dense.mp4" type="video/mp4"></video><p>Same initial state, camera, two-second horizon, and frozen 12/10-mm gates. Green policies pass this scene.</p></article></div></section>
<section id="hard"><div class="section-head"><p>03 / HARD, INEXTENSIBLE TRACK B</p><h2>N128 accuracy is real; speed is not solved</h2></div>
<p>The hard KKT trajectory completes 600 physical substeps with one advance per call and a 0.477-µm peak axial error. Finite BGN and VBD rows below retain the authored soft stretch energy, so millimetre gaps are expected material response. Their timings are context only.</p>{hard_table}
<h3>Same-hard convergence check</h3><p>K2 is compared to the same hard formulation at K12. Every frozen state, curvature, energy, work, feasibility, stationarity, and one-dt gate passes.</p>{parity_table}
<h3>N512 scale boundary</h3><p>Residual adaptation makes the first 600-step interval valid. K2/K5/K12 first stops at step 1205; the preregistered K24 extension admits that step, then fails the same unchanged stationarity gate at step 1209. Frame 193 is not reached.</p>{scale_table}
<div class="callout"><strong>Honest outcome:</strong> N128 K2 is the accuracy specialist at 47.25 ms/substep p50 on this CPU diagnostic. Requested finite-compliance VBD10 is 2.46 ms/substep p50, but it solves a different material problem. N512 is partial evidence; N1024 hard was not admitted.</div></section>
<section id="portfolio"><div class="section-head"><p>04 / EXISTING VERIFIED PORTFOLIO</p><h2>Soft smooth speed and dense-contact survival</h2></div>
<h3>Finite-compliance smooth cables</h3><p>Five matched Y targets use 50 samples per method. Prepared compact BGN is 3.065–5.046× faster at the matched target. The wider audit records 20/20 executed passes through 8,191 dynamic bodies; broad rows are accuracy audits, not matched speed measurements.</p>{speed}
<h3>Dense contact</h3><p>The selected robust route is penalty-only VBD10 with contact history disabled. The hard alpha-zero/history-off alternative also passes, but its worst penetration uses 94.6% of the 12-mm gate. Neither passing policy is bitwise repeatable.</p>{dense}</section>
<section id="frontier"><div class="section-head"><p>05 / COUPLED EXACT-LAW FRONTIER</p><h2>Small unified contact passes; scale remains open</h2></div>
<h3>Full 48-body endpoint-safe smoke</h3><p>Contact KKT v5 completes and certifies all 60 physical steps with both ground and cable–cable contact. It improves peak penetration by {unified["penetration_improvement"]:.0f}× and final stretch by {unified["stretch_improvement"]:.0f}× over VBD10 on this smoke, while satisfying the exact non-associated Coulomb certificates.</p>{unified_table}{law_table}
<div class="notice"><strong>Speed and geometry boundary.</strong> Setup-inclusive v5 time is {unified["setup_inclusive_ms"]:.1f} ms versus {unified["vbd10_setup_inclusive_ms"]:.1f} ms ({unified["slowdown"]:.1f}× slower). The line search certifies accepted endpoints only; it is not CCD and does not certify swept paths. This is 48 bodies, not the 7,680-body dense pile.</div>
<h3>Exact-law unit oracle</h3><p>CCDSN combines hard stretch response, high-friction sticking condensation, and a dense De Saxcé natural-map solve. The analytic sliding and mixed stick/slide oracles satisfy all frozen 1e-9 gates, including primal/dual cones, complementarity, stationarity, stretch residual, and non-positive friction work.</p>{oracle_table}
<h3>Earlier full-trajectory negatives</h3><p>V2 and v3 remain useful failure evidence, not near-passes. V2 certifies 32 steps and fails at attempted step 33. V3 certifies 45 steps and fails its solver gate at attempted step 46. Low penetration on a partial trajectory does not establish a solver.</p>{negatives}</section>
<section id="evidence"><div class="section-head"><p>06 / EVIDENCE</p><h2>Small public surface; exact source binding</h2></div>
<p>Every displayed number and video is bound to strict finite JSON or a source manifest. The stage uses an exact-set SHA-256 manifest, checks every internal link, and fully decodes all three videos.</p>
<p class="links"><a href="report.json">machine-readable v6 report</a><a href="manifest.json">exact-set manifest</a><a href="{escape(report["archive"]["url"])}">immutable full v5 archive</a></p>
<details><summary>Evidence and source snapshots</summary><ul class="evidence-list">{evidence}</ul></details></section>
</main><footer><div class="shell">Verified frontier · measured specialists · universal solver not established</div></footer></body></html>'''


CSS = r'''
:root{--bg:#07111b;--panel:#0d1b29;--panel2:#102337;--line:#294158;--text:#f3f7fb;--muted:#aec0d1;--cyan:#66dcff;--green:#65e1a9;--red:#ff7c89;--amber:#ffd074;--max:1180px}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:linear-gradient(180deg,#06101a,#091521 30%,#07111b);color:var(--text);font:16px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}.shell{width:min(calc(100% - 40px),var(--max));margin:auto}.hero{padding:64px 0 42px;border-bottom:1px solid var(--line);background:radial-gradient(circle at 82% 0,#1b4664 0,transparent 43%)}.eyebrow,.section-head>p,.kicker{margin:0;color:var(--cyan);font:750 12px/1.2 ui-monospace,SFMono-Regular,monospace;letter-spacing:.14em}h1{max-width:1000px;margin:14px 0 20px;font-size:clamp(42px,6vw,74px);line-height:1;letter-spacing:-.05em}h1 span{color:#a9cae7}.lead{max-width:920px;color:#c8d7e5;font-size:20px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:30px}.metrics>div,.callout{padding:15px;border:1px solid var(--line);border-radius:12px;background:#0c1a28d9}.metrics strong{display:block;color:var(--green);font-size:25px;line-height:1.2}.metrics span{display:block;margin-top:5px;color:var(--muted);font-size:13px}nav{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:8px 19px;padding:14px 0;border-bottom:1px solid var(--line);background:#07111bf2;backdrop-filter:blur(10px)}a{color:#78ddff;text-decoration:none}a:hover,a:focus-visible{text-decoration:underline}section{padding:48px 0;border-bottom:1px solid #1e3448}.section-head h2{margin:8px 0 20px;font-size:clamp(29px,4vw,44px);line-height:1.08;letter-spacing:-.035em}h3{margin:28px 0 10px;font-size:21px}.notice{margin:20px 0;padding:15px 18px;border-left:4px solid var(--amber);border-radius:0 10px 10px 0;background:#251e12}.notice ul{margin:8px 0;padding-left:22px}.table-scroll{max-width:100%;margin:16px 0 28px;overflow:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel);scrollbar-gutter:stable}.table-scroll:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}table{width:100%;min-width:780px;border-collapse:collapse}th,td{padding:11px 13px;border-bottom:1px solid #223a51;text-align:left;vertical-align:top}th{color:#bfd0df;font-size:13px}tbody tr:last-child td{border-bottom:0}.badge{display:inline-block;max-width:230px;padding:3px 8px;border-radius:999px;font-size:12px;font-weight:800;line-height:1.35}.badge.pass{color:#062016;background:var(--green)}.badge.fail{color:#2c090d;background:var(--red)}.video-card{min-width:0;margin:16px 0;padding:14px;border:1px solid var(--line);border-radius:14px;background:var(--panel)}.video-card.primary{display:grid;grid-template-columns:minmax(240px,.7fr) minmax(0,1.3fr);gap:18px;align-items:center}.video-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.video-card h3{margin:7px 0 10px}.video-card p{color:var(--muted)}video{display:block;width:100%;height:auto;border-radius:9px;background:#02070b}.callout{margin:18px 0;border-color:#2c7055}.links{display:flex;flex-wrap:wrap;gap:8px 20px}.evidence-list{columns:2;gap:26px;padding-left:20px}.evidence-list li{break-inside:avoid;margin:8px 0}.evidence-list a,.evidence-list code{display:block;overflow-wrap:anywhere;word-break:break-word}.evidence-list code{color:var(--muted);font-size:12px}details{padding:12px 16px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}summary{cursor:pointer;font-weight:800}footer{padding:28px 0;color:var(--muted)}
@media(max-width:900px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.video-card.primary{grid-template-columns:1fr}.video-grid{grid-template-columns:1fr}.evidence-list{columns:1}.hero{padding-top:46px}}
@media(max-width:520px){.shell{width:min(calc(100% - 22px),var(--max))}.metrics{grid-template-columns:1fr}h1{font-size:39px}.lead{font-size:17px}section{padding:38px 0}.video-card{padding:9px}.table-scroll{margin-left:0;margin-right:0}th,td{padding:9px 10px}.badge{white-space:normal}nav{position:static}}
'''


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.video_tags = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "video":
            self.video_tags += 1
        key = "href" if tag == "a" else "src" if tag in ("source", "video") else None
        if key and values.get(key):
            self.links.append(str(values[key]))


def safe_relative(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc or value.startswith("#"):
        return None
    path = PurePosixPath(parsed.path)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe link {value}")
    return path.as_posix()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def copy_verified(source: Path, destination: Path, expected: str | None = None) -> None:
    require(source.is_file(), f"source missing: {source}")
    if expected is not None:
        verify_hash(source, expected, str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    require(destination.stat().st_size == source.stat().st_size, f"copy size mismatch: {destination}")
    require(sha256_file(destination) == sha256_file(source), f"copy hash mismatch: {destination}")


def qa(stage: Path, decode: bool = True) -> dict[str, Any]:
    manifest = strict_load(stage / "manifest.json")
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema mismatch")
    listed = manifest.get("files", [])
    expected = {row["path"] for row in listed} | {"manifest.json"}
    actual = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
    require(expected == actual, f"exact set mismatch missing={expected-actual}, extra={actual-expected}")
    for row in listed:
        path = stage / row["path"]
        require(path.stat().st_size == row["bytes"], f"bytes mismatch {row['path']}")
        verify_hash(path, row["sha256"], row["path"])
    for path in stage.rglob("*.json"):
        strict_load(path)
    html = (stage / "index.html").read_text(encoding="utf-8")
    lower = html.lower()
    forbidden = ("cable_research.md", "research handoff", "owner: team", "next experiments")
    require(not any(token in lower for token in forbidden), "internal discussion leaked into report")
    require("a router, not one universal solver" in lower, "visible verdict missing")
    require("60/60" in html and "34× slower" in html, "visible contact v5 outcome missing")
    require("32 certified" in html and "step 33 of 60" in html and
            "45 certified" in html and "step 46 of 60" in html,
            "visible contact negatives missing or ambiguous")
    require("K24 path stalled after 18 accepted attempts; no K32" in html,
            "visible K24 stop reason missing")
    parser = LinkParser()
    parser.feed(html)
    require(parser.video_tags == 3, f"expected 3 visible videos, found {parser.video_tags}")
    checked = 0
    for link in parser.links:
        relative = safe_relative(link)
        if relative:
            require((stage / relative).is_file(), f"broken internal link: {link}")
            checked += 1
    require(ARCHIVE_URL in html and "76483b1df266bf15ade2eb3397005cbe630d6fec" in html,
            "immutable archive link missing")
    css = (stage / "styles.css").read_text(encoding="utf-8")
    require("@media(max-width:900px)" in css and "@media(max-width:520px)" in css,
            "responsive breakpoints missing")
    require(".video-card.primary{grid-template-columns:1fr}" in css and
            ".video-grid{grid-template-columns:1fr}" in css,
            "mobile video layout missing")
    require("overflow-x:hidden" not in css, "page-level clipping is forbidden")
    videos = sorted(stage.rglob("*.mp4"))
    require(len(videos) == 3, f"stage must contain exactly 3 videos, found {len(videos)}")
    decoded = 0
    if decode:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        for video in videos:
            result = subprocess.run(
                [ffmpeg, "-v", "error", "-i", str(video), "-map", "0:v:0", "-f", "null", "-"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            require(result.returncode == 0 and not result.stderr.strip(),
                    f"full decode failed: {video}: {result.stderr[-500:]}")
            decoded += 1
    report = strict_load(stage / "report.json")
    require(report["summary"]["visible_videos"] == 3, "report video contract changed")
    require(report["universal_solver_established"] is False, "universal claim boundary changed")
    return {
        "schema": QA_SCHEMA,
        "pass": True,
        "files": len(actual),
        "links_checked": checked,
        "visible_videos": parser.video_tags,
        "videos_full_decoded": decoded,
        "strict_json": True,
        "exact_set": True,
        "internal_discussion_absent": True,
        "responsive_static_contract": True,
        "immutable_archive_link": True,
        "publication_ready": False,
    }


def _paths_from_args(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    paths = {key: value.resolve() for key, value in default_paths(root).items()}
    if args.hard_scale_result is not None:
        paths["hard_scale"] = args.hard_scale_result.resolve()
    if args.hard_k24_result is not None:
        paths["hard_k24"] = args.hard_k24_result.resolve()
    if args.contact_v5 is not None:
        paths["contact_v5"] = args.contact_v5.resolve()
    paths.update({
        "root": root,
        "hard_visual_manifest": args.hard_visual_manifest.resolve(),
        "hard_video": args.hard_video.resolve(),
        "hard_poster": args.hard_poster.resolve(),
        "hard_scale_sha256": args.hard_scale_sha256.lower(),
        "hard_k24_sha256": args.hard_k24_sha256.lower(),
        "contact_v5_sha256": args.contact_v5_sha256.lower(),
    })
    for key in ("hard_scale_sha256", "hard_k24_sha256", "contact_v5_sha256"):
        value = paths[key]
        require(len(value) == 64 and all(character in "0123456789abcdef" for character in value),
                f"invalid {key}")
    return paths


def build(args: argparse.Namespace) -> Path:
    paths = _paths_from_args(args)
    root = paths["root"]
    output = args.output.resolve()
    require(not output.exists(), f"add-only output exists: {output}")
    guard_owner(root, "v6 report build start")
    values = validate_inputs(paths)
    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    require(not temporary.exists(), f"temporary output exists: {temporary}")
    temporary.mkdir(parents=True)
    evidence: list[dict[str, Any]] = []

    def add(source: Path, relative: str, label: str, expected: str | None = None) -> None:
        destination = temporary / relative
        copy_verified(source, destination, expected)
        evidence.append({"label": label, **file_record(destination, temporary)})

    try:
        legacy = paths["legacy_stage"]
        add(legacy / "report.json", "data/legacy/v5_report.json", "v5 results archive index", EXPECTED_HASH["legacy_report"])
        add(legacy / "manifest.json", "data/legacy/v5_manifest.json", "v5 exact-set manifest", EXPECTED_HASH["legacy_manifest"])
        add(legacy / "data/smooth/base_report_v4.json", "data/smooth/base_report.json", "soft smooth portfolio result", EXPECTED_HASH["smooth_report"])
        add(paths["coverage"], "data/smooth/coverage_manifest.json", "smooth coverage campaign", EXPECTED_HASH["coverage"])
        add(paths["coverage_render"], "data/smooth/y127_render_manifest.json", "Y127 visual provenance", EXPECTED_HASH["coverage_render"])
        add(legacy / "data/contact/dense_robustness.json", "data/contact/dense_robustness.json", "7,680-body robustness result", EXPECTED_HASH["dense_robustness"])
        add(legacy / "data/visual/dense_manifest.json", "data/contact/dense_visual_manifest.json", "dense visual provenance", EXPECTED_HASH["dense_visual_manifest"])
        add(paths["hard_result"], "data/hard/n128_trajectory.json", "N128 hard trajectory", EXPECTED_HASH["hard_result"])
        add(paths["hard_parity"], "data/hard/n128_same_hard_parity.json", "N128 same-hard parity", EXPECTED_HASH["hard_parity"])
        add(paths["hard_scale"], "data/hard/n512_adaptive_scale_boundary.json",
            "N512 adaptive scale boundary", str(paths["hard_scale_sha256"]))
        add(paths["hard_k24"], "data/hard/n512_preregistered_k24_probe.json",
            "N512 preregistered K24 probe", str(paths["hard_k24_sha256"]))
        add(paths["hard_visual_manifest"], "data/hard/visual_manifest.json", "N128 hard visual provenance", EXPECTED_HASH["hard_visual_manifest"])
        add(paths["ccdsn"], "data/ccdsn/exact_law_oracle.json", "CCDSN exact-law oracle", EXPECTED_HASH["ccdsn"])
        add(paths["contact_v2"], "data/contact/kkt_v2_negative.json", "contact KKT v2 negative", EXPECTED_HASH["contact_v2"])
        add(paths["contact_v3"], "data/contact/kkt_v3_negative.json", "contact KKT v3 negative", EXPECTED_HASH["contact_v3"])
        add(paths["contact_v5"], "data/contact/kkt_v5_small_full_pass.json",
            "contact KKT v5 small full pass", str(paths["contact_v5_sha256"]))

        add(paths["hard_video"], "media/hard/n128_hard_vs_finite.mp4", "N128 hard comparison video", EXPECTED_HASH["hard_video"])
        add(paths["hard_poster"], "media/hard/n128_hard_vs_finite_poster.png", "N128 hard comparison poster", EXPECTED_HASH["hard_poster"])
        add(paths["y_video"], "media/smooth/y127.mp4", "Y127 smooth comparison video", EXPECTED_HASH["y_video"])
        add(paths["y_poster"], "media/smooth/y127_poster.png", "Y127 smooth comparison poster", EXPECTED_HASH["y_poster"])
        add(legacy / "media/dense/dense_contact_comparison.mp4", "media/contact/dense.mp4", "dense contact comparison video", EXPECTED_HASH["dense_video"])
        add(legacy / "media/dense/dense_contact_comparison_poster.png", "media/contact/dense_poster.png", "dense contact comparison poster", EXPECTED_HASH["dense_poster"])

        source_names = {
            "codex_hard_kkt_bgn.py", "codex_hard_kkt_trajectory.py", "codex_hard_kkt_parity.py",
            "codex_hard_kkt_adaptive_n512.py",
            "codex_hard_kkt_k24_probe.py",
            "codex_hard_kkt_review_render.py", "codex_ccdsn_reference_benchmark.py", "codex_ccdsn.py",
            "codex_desaxce.py", "test_codex_ccdsn.py", "codex_contact_kkt_v2.py",
            "codex_contact_kkt_v2_pile.py", "test_codex_contact_kkt_v2.py", "codex_contact_kkt_v3.py",
            "codex_contact_kkt_v3_pile.py", "test_codex_contact_kkt_v3.py",
            "codex_contact_safe_predictor.py", "test_codex_contact_safe_predictor.py",
            "codex_contact_kkt_v5.py", "codex_contact_kkt_v5_pile.py", "test_codex_contact_kkt_v5.py",
            "codex_cable_report_v6.py", "test_codex_cable_report_v6.py",
        }
        for name in sorted(source_names):
            add(root / "bench/global_cable" / name, f"source/{name}", f"source {name}")

        contract = {
            "schema": "codex-cable-results-review-build-contract/v6",
            "add_only": True,
            "git_invoked": False,
            "pages_worktree_touched": False,
            "publication_performed": False,
            "visible_video_limit": 3,
            "strict_json_required": True,
            "full_media_decode_required": True,
            "archive_url": ARCHIVE_URL,
            "input_sha256": {
                **EXPECTED_HASH,
                "hard_scale": paths["hard_scale_sha256"],
                "hard_k24": paths["hard_k24_sha256"],
                "contact_v5": paths["contact_v5_sha256"],
            },
        }
        (temporary / "build_contract.json").write_bytes(stable_json(contract))
        evidence.append({"label": "v6 build contract", **file_record(temporary / "build_contract.json", temporary)})
        report = make_report(values, evidence)
        (temporary / "report.json").write_bytes(stable_json(report))
        (temporary / "index.html").write_text(render_html(report), encoding="utf-8", newline="\n")
        (temporary / "styles.css").write_text(CSS.strip() + "\n", encoding="utf-8", newline="\n")
        guard_owner(root, "v6 exact-set manifest")
        files = [file_record(path, temporary) for path in sorted(
            (item for item in temporary.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(temporary).as_posix(),
        )]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "publication_ready": False,
            "immutable_stage": True,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verdict": report["verdict"],
            "visible_videos": 3,
            "exact_set_rule": "actual files equal manifest.json plus files; bytes and SHA-256 exact",
            "files": files,
        }
        (temporary / "manifest.json").write_bytes(stable_json(manifest))
        result = qa(temporary, decode=True)
        require(result["pass"] is True, "staged QA failed")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return output
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="build one new immutable local stage")
    build_parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--hard-visual-manifest", type=Path, required=True)
    build_parser.add_argument("--hard-video", type=Path, required=True)
    build_parser.add_argument("--hard-poster", type=Path, required=True)
    build_parser.add_argument("--hard-scale-result", type=Path)
    build_parser.add_argument("--hard-scale-sha256", default=EXPECTED_HASH["hard_scale"])
    build_parser.add_argument("--hard-k24-result", type=Path)
    build_parser.add_argument("--hard-k24-sha256", default=EXPECTED_HASH["hard_k24"])
    build_parser.add_argument("--contact-v5", type=Path)
    build_parser.add_argument("--contact-v5-sha256", default=EXPECTED_HASH["contact_v5"])
    qa_parser = sub.add_parser("qa", help="re-verify an immutable stage")
    qa_parser.add_argument("--stage", type=Path, required=True)
    qa_parser.add_argument("--skip-video-decode", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "build":
            output = build(args)
            print(f"stage={output}")
            print(f"manifest_sha256={sha256_file(output / 'manifest.json')}")
            print(json.dumps(qa(output, decode=False), sort_keys=True))
        else:
            print(json.dumps(qa(args.stage.resolve(), decode=not args.skip_video_decode), sort_keys=True))
        return 0
    except (ReportError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
