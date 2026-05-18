#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record MP4 clips for the ALM pyramid comparison report."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import warp as wp

from newton.examples.cable.example_box_pyramid import DEFAULT_PYRAMID_VARIANT, Example, PYRAMID_VARIANTS
from newton.viewer import ViewerGL


ASSET_NAMES = {
    "1": "v01_new_alm_finite_large_gap.mp4",
    "2": "v02_new_alm_inf_large_gap.mp4",
    "3": "v03_old_soft_large_gap.mp4",
    "4": "v04_new_alm_finite_small_gap.mp4",
    "5": "v05_old_hard_matched_rho_large_gap.mp4",
    "6": "v06_old_hard_ramped_large_gap.mp4",
}


def _record_variant(
    version: str,
    output_path: Path,
    *,
    device: str,
    frames: int,
    fps: int,
    width: int,
    height: int,
    iterations: int,
):
    wp.set_device(device)
    viewer = ViewerGL(width=width, height=height, headless=True)
    args = SimpleNamespace(iterations=iterations, pyramid_version=version)
    example = Example(viewer, args)
    if hasattr(viewer, "_paused"):
        viewer._paused = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for _frame in range(frames):
            example.step()
            example.render()
            frame = viewer.get_frame(render_ui=False).numpy()
            proc.stdin.write(frame.tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        rc = proc.wait()
        viewer.close()

    if rc != 0:
        raise RuntimeError(f"ffmpeg failed for version {version} with exit code {rc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--versions",
        nargs="+",
        default=[DEFAULT_PYRAMID_VARIANT],
        choices=sorted(PYRAMID_VARIANTS),
        help="Pyramid versions to record.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "assets")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--single-version-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_version_worker:
        version = args.versions[0]
        output_path = args.output_dir / ASSET_NAMES[version]
        _record_variant(
            version,
            output_path,
            device=args.device,
            frames=args.frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
            iterations=args.iterations,
        )
        return

    for version in args.versions:
        output_path = args.output_dir / ASSET_NAMES[version]
        name = PYRAMID_VARIANTS[version]["name"]
        print(f"[record] v{version} {name} -> {output_path}")
        cmd = [
            sys.executable,
            __file__,
            "--single-version-worker",
            "--versions",
            version,
            "--output-dir",
            str(args.output_dir),
            "--device",
            args.device,
            "--frames",
            str(args.frames),
            "--fps",
            str(args.fps),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--iterations",
            str(args.iterations),
        ]
        env = os.environ.copy()
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
