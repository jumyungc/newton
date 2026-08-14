# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record matched CUDA-graph costs for the contact report on Horde.

This is a focused benchmark, not a unit test.  Each comparison resets to the
same initial state and times a complete captured collision+solve replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import warp as wp

from newton.tests import benchmark_acm_event_strategy as benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    wp.init()
    with wp.ScopedDevice(args.device):
        dense = [
            benchmark.run_dense_sphere_chain(
                device=args.device,
                body_count=16,
                dt=0.03,
                iterations=5,
                event_projection=event_projection,
                event_waves=1,
                closure_iterations=20,
                closure_relaxation=1.0,
                capture_graph=True,
                mass_ratio=1.0,
                active_set_iterations=4,
                timing_repeats=args.repeats,
            )
            for event_projection in (False, True)
        ]
        batch = [
            benchmark.run_batch_timing(
                device=args.device,
                dt=0.03,
                iterations=2,
                event_projection=event_projection,
                body_count=1024,
                repeats=args.repeats,
                capture_graph=True,
                event_waves=1,
                closure_iterations=0,
            )
            for event_projection in (False, True)
        ]

    result = {
        "device": args.device,
        "scope": "complete captured collision+solve replay; focused benchmark; no unit tests",
        "repeats": args.repeats,
        "dense_16_equal_mass": dense,
        "independent_sphere_plane_1024": batch,
    }
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
