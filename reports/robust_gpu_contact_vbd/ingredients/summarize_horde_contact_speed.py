# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Package focused Horde contact-speed sweeps for the public report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _twist_summary(cases: list[dict]) -> dict:
    medians = [case["simulation_frame_time_median_ms"] for case in cases]
    p95s = [case["simulation_frame_time_p95_ms"] for case in cases]
    penetrations = [case["maximum_fresh_penetration_m"] for case in cases]
    return {
        "runs": len(cases),
        "median_frame_time_range_ms": [min(medians), max(medians)],
        "median_of_run_medians_ms": statistics.median(medians),
        "p95_frame_time_range_ms": [min(p95s), max(p95s)],
        "maximum_fresh_penetration_m": max(penetrations),
        "total_replayed_frames": sum(case["replayed_frames"] for case in cases),
        "total_replays": sum(case["total_replays"] for case in cases),
        "query_debt_frames": sum(case["query_integrity_debt_frames"] for case in cases),
        "solver_list_debt_frames": sum(
            case["solver_contact_list_integrity_debt_frames"] for case in cases
        ),
        "replay_cap_debt_frames": sum(case["replay_cap_debt_frames"] for case in cases),
    }


def _pile_case(payload: dict) -> dict:
    return payload["staged_cable_pile"]


def _pile_summary(cases: list[dict]) -> dict:
    medians = [case["median_frame_time_ms"] for case in cases]
    p95s = [case["p95_frame_time_ms"] for case in cases]
    penetrations = [case["maximum_fresh_penetration_m"] for case in cases]
    return {
        "runs": len(cases),
        "median_frame_time_range_ms": [min(medians), max(medians)],
        "median_of_run_medians_ms": statistics.median(medians),
        "p95_frame_time_range_ms": [min(p95s), max(p95s)],
        "maximum_fresh_penetration_m": max(penetrations),
        "replayed_frames_range": [
            min(case["replayed_frames"] for case in cases),
            max(case["replayed_frames"] for case in cases),
        ],
        "query_debt_frames": sum(case["query_integrity_debt_frames"] for case in cases),
        "solver_list_debt_frames": sum(
            case["solver_list_integrity_debt_frames"] for case in cases
        ),
        "replay_cap_debt_frames": sum(case["replay_cap_debt_frames"] for case in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    twist_names = {
        "target_20mm": ["speed_target_20mm.json"],
        "target_30mm": [
            "speed_target_30mm.json",
            "speed_target_30mm_r2.json",
            "speed_target_30mm_r3.json",
        ],
        "target_40mm": [
            "speed_target_40mm.json",
            "speed_target_40mm_r2.json",
            "speed_target_40mm_r3.json",
        ],
        "target_50mm": [
            "speed_target_50mm.json",
            "speed_target_50mm_r2.json",
            "speed_target_50mm_r3.json",
        ],
        "target_100mm": ["speed_target_100mm.json"],
    }
    twist_cases = {
        group: [_load(args.input_dir / name) for name in names]
        for group, names in twist_names.items()
    }
    negative_twist_names = (
        "speed_target_40mm_bucket5.json",
        "speed_target_40mm_refresh2.json",
        "speed_target_50mm_iter4.json",
        "speed_target_50mm_rho10.json",
        "speed_target_50mm_rho7p5.json",
    )
    negative_twist_cases = {
        Path(name).stem: _load(args.input_dir / name) for name in negative_twist_names
    }

    pile_names = {
        "baseline_10_steps": "pile_baseline10.json",
        "balanced_5_steps": "pile_optimized5_canonical.json",
        "aggressive_4_steps": "pile_aggressive4.json",
    }
    pile_cases = {
        group: _pile_case(_load(args.input_dir / name)) for group, name in pile_names.items()
    }
    pile_five_repeats = [
        _pile_case(_load(args.input_dir / name))
        for name in (
            "pile_optimized5_r1.json",
            "pile_optimized5_r2.json",
            "pile_optimized5_r3.json",
        )
    ]

    old_twist = _load(args.assets / "contact_replay_final.json")
    always_fine = _load(args.assets / "contact_always_fine_final.json")
    balanced_twist = _twist_summary(twist_cases["target_30mm"])
    balanced_pile = _pile_summary(pile_five_repeats)
    report = {
        "hardware": "NVIDIA L40",
        "scope": "focused Horde speed sweeps; no unit tests",
        "twist": {
            "automatic_balanced_default": _load(args.input_dir / "speed_auto_balanced.json"),
            "previous_10mm_validated": old_twist,
            "always_fine_1mm": always_fine,
            "sweep_summaries": {
                group: _twist_summary(cases) for group, cases in twist_cases.items()
            },
            "balanced_30mm_speedup_vs_previous_median": (
                old_twist["simulation_frame_time_median_ms"]
                / balanced_twist["median_of_run_medians_ms"]
            ),
            "raw_sweep_cases": twist_cases,
            "rejected_or_aggressive_ablations": negative_twist_cases,
        },
        "pile": {
            "canonical_cases": pile_cases,
            "balanced_5_step_summary": balanced_pile,
            "balanced_5_step_speedup_vs_10_step_median": (
                pile_cases["baseline_10_steps"]["median_frame_time_ms"]
                / balanced_pile["median_of_run_medians_ms"]
            ),
            "raw_5_step_repeats": pile_five_repeats,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
