#!/usr/bin/env python3
"""Render immutable, synchronized N=128 hard/finite cable review evidence.

The four panels use the same camera, world bounds, source cadence, and
absolute joint-anchor-gap colour scale.  This is visual evidence, not timing
evidence.  Hard equality and finite compliance are deliberately labelled as
different physical semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "bench/global_cable/cable_research.md"
EXPECTED_OWNER = (
    "OWNER: Team Codex-Remote — 2026-07-05T11:40Z — "
    "coupled long-chain/contact solver invention round"
)
SOURCE_ROOT = (
    ROOT
    / "bench/_workspace/codex_hard_kkt_trajectory"
    / "run_20260705T121845Z_1924599"
)
SOURCE_RESULT = SOURCE_ROOT / "result.json"
OUT_ROOT = ROOT / "bench/_workspace/codex_hard_kkt_review"

SCHEMA = "codex-hard-kkt-four-panel-review/v1"
WIDTH = 1920
HEIGHT = 1080
DPI = 100
VIDEO_FPS = 60
EXPECTED_FRAMES = 61
DT = 1.0 / 600.0
SEGMENT_LENGTH_M = 0.04
GAP_VMAX_MM = 8.0

METHODS = (
    {
        "key": "hard_k2",
        "title": "Hard equality K2",
        "subtitle": "inextensible Track B",
        "file": "n128_hard_render_states.npz",
        "accent": "#54e1c2",
    },
    {
        "key": "finite_bgn",
        "title": "Compact global BGN K5",
        "subtitle": "authored finite compliance",
        "file": "n128_finite_bgn_render_states.npz",
        "accent": "#85b9ff",
    },
    {
        "key": "vbd10",
        "title": "Requested VBD10",
        "subtitle": "authored finite compliance",
        "file": "n128_vbd10_render_states.npz",
        "accent": "#ffbe66",
    },
    {
        "key": "vbd80",
        "title": "Requested VBD80",
        "subtitle": "authored finite compliance",
        "file": "n128_vbd80_render_states.npz",
        "accent": "#ef8fff",
    },
)


class RenderError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise RenderError(message)


def owner_guard(where: str) -> str:
    first = DOC.read_text(encoding="utf-8").splitlines()[0]
    if first != EXPECTED_OWNER:
        raise RenderError(f"owner guard rejected {where}: {first!r}")
    return first


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def format_gap(value_m: float) -> str:
    value_mm = float(value_m) * 1.0e3
    if value_mm < 0.01:
        return f"{value_m * 1.0e6:.3f} µm"
    return f"{value_mm:.3f} mm"


def padded_bounds(values: np.ndarray, fraction: float = 0.055) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    require(values.size > 0 and bool(np.all(np.isfinite(values))), "invalid plot bounds")
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = max(hi - lo, 0.25)
    return lo - fraction * span, hi + fraction * span


def load_and_extract() -> tuple[list[dict[str, Any]], tuple[float, float], tuple[float, float]]:
    """Load bound snapshots and reconstruct exact per-joint gaps."""

    from bench.global_cable import codex_al_bgn as al
    from bench.global_cable import codex_bgn_global as bgn
    from bench.global_cable import codex_rc_forest_newton as rc
    from bench.global_cable import scenes

    owner_guard("build exact render metric plan")
    scene = scenes.horiz_cantilever_stretch_bend()
    plan = al.prepare(scene, DT)
    body_com = np.asarray(plan.data.body_com, dtype=np.float64)
    child = np.asarray(plan.data.child, dtype=np.int64)
    require(len(child) == 127, f"expected 127 cable joints, got {len(child)}")

    records: list[dict[str, Any]] = []
    all_x: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    for spec in METHODS:
        path = SOURCE_ROOT / str(spec["file"])
        require(path.is_file(), f"missing source snapshot {path}")
        with np.load(path, allow_pickle=False) as arrays:
            body_q = np.asarray(arrays["body_q"], dtype=np.float64)
            render_time_s = np.asarray(arrays["render_time_s"], dtype=np.float64)
            stored_gap_max = np.asarray(arrays["gap_max_m"], dtype=np.float64)
            bend_max = np.asarray(arrays["bend_max_rad"], dtype=np.float64)
        require(body_q.shape == (EXPECTED_FRAMES, 128, 7), f"bad {path.name} body_q")
        require(render_time_s.shape == (EXPECTED_FRAMES,), f"bad {path.name} time")
        require(np.array_equal(render_time_s, np.arange(EXPECTED_FRAMES) / 60.0),
                f"bad {path.name} source cadence")

        centers = np.empty((EXPECTED_FRAMES, 128, 3), dtype=np.float64)
        rod_segments = np.empty((EXPECTED_FRAMES, 128, 2, 2), dtype=np.float64)
        joint_gaps = np.empty((EXPECTED_FRAMES, 127), dtype=np.float64)
        body_gap = np.empty((EXPECTED_FRAMES, 128), dtype=np.float64)
        for frame, raw in enumerate(body_q):
            q = bgn.pf._q_normalize_batch(raw[:, 3:7])
            p_com = raw[:, :3] + bgn.pf._q_rotate_batch(q, body_com)
            pose = rc.PoseState(p_com=p_com, q=q)
            gap = np.linalg.norm(bgn._joint_batch(plan, pose)["C"], axis=1)
            require(abs(float(np.max(gap)) - float(stored_gap_max[frame])) <= 1.0e-15,
                    f"{path.name} frame {frame} exact gap reconstruction mismatch")
            local_half = np.zeros((128, 3), dtype=np.float64)
            local_half[:, 0] = 0.5 * SEGMENT_LENGTH_M
            half = bgn.pf._q_rotate_batch(q, local_half)
            endpoints_a = p_com - half
            endpoints_b = p_com + half
            centers[frame] = p_com
            rod_segments[frame, :, 0, 0] = endpoints_a[:, 0]
            rod_segments[frame, :, 0, 1] = endpoints_a[:, 2]
            rod_segments[frame, :, 1, 0] = endpoints_b[:, 0]
            rod_segments[frame, :, 1, 1] = endpoints_b[:, 2]
            joint_gaps[frame] = gap
            body_gap[frame, 0] = gap[0]
            body_gap[frame, child] = gap

        all_x.append(rod_segments[..., 0].reshape(-1))
        all_z.append(rod_segments[..., 1].reshape(-1))
        records.append(
            {
                **spec,
                "path": path,
                "source_sha256": sha256_file(path),
                "body_q": body_q,
                "render_time_s": render_time_s,
                "stored_gap_max_m": stored_gap_max,
                "bend_max_rad": bend_max,
                "centers": centers,
                "rod_segments": rod_segments,
                "joint_gaps_m": joint_gaps,
                "body_gap_m": body_gap,
                "trajectory_gap_max_m": float(np.max(stored_gap_max)),
            }
        )
    return records, padded_bounds(np.concatenate(all_x)), padded_bounds(np.concatenate(all_z))


def render_frame(
    records: list[dict[str, Any]],
    frame: int,
    xlim: tuple[float, float],
    zlim: tuple[float, float],
) -> np.ndarray:
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor="#071019")
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.0, 1.0, 0.035),
        left=0.042, right=0.963, top=0.875, bottom=0.105,
        hspace=0.20, wspace=0.10,
    )
    norm = Normalize(vmin=0.0, vmax=GAP_VMAX_MM, clip=True)
    cmap = plt.get_cmap("turbo")
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    for axis, record in zip(axes, records, strict=True):
        axis.set_facecolor("#0b1722")
        centers = record["centers"][frame]
        rods = record["rod_segments"][frame]
        body_gap_mm = record["body_gap_m"][frame] * 1.0e3
        axis.plot(centers[:, 0], centers[:, 2], color="#b9c7d5", alpha=0.22,
                  linewidth=1.0, zorder=1)
        collection = LineCollection(rods, cmap=cmap, norm=norm, linewidths=4.0,
                                    capstyle="round", zorder=3)
        collection.set_array(body_gap_mm)
        axis.add_collection(collection)
        axis.scatter([centers[0, 0]], [centers[0, 2]], marker="s", s=34,
                     facecolor="#ffffff", edgecolor="#071019", linewidth=0.8,
                     zorder=5)
        axis.set_xlim(*xlim)
        axis.set_ylim(*zlim)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#294052", linewidth=0.6, alpha=0.45)
        axis.tick_params(colors="#9db0bf", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#294052")
        axis.set_title(str(record["title"]), loc="left", color=str(record["accent"]),
                       fontsize=14, fontweight="bold", pad=10)
        axis.text(0.982, 0.972, str(record["subtitle"]), transform=axis.transAxes,
                  ha="right", va="top", color="#b5c4d0", fontsize=8.5,
                  zorder=10)
        current = float(record["stored_gap_max_m"][frame])
        peak = float(np.max(record["stored_gap_max_m"][: frame + 1]))
        bend = float(record["bend_max_rad"][frame])
        axis.text(
            0.018, 0.965,
            f"max gap now  {format_gap(current)}\n"
            f"peak to time {format_gap(peak)}\n"
            f"max bend     {bend:.3f} rad",
            transform=axis.transAxes, va="top", ha="left", family="monospace",
            fontsize=9.5, color="#edf5fb",
            bbox={"facecolor": "#071019", "alpha": 0.86, "edgecolor": "#355169",
                  "boxstyle": "round,pad=0.45"},
            zorder=10,
        )
        axis.set_xlabel("world x (m)", color="#9db0bf", fontsize=8)
        axis.set_ylabel("world z (m)", color="#9db0bf", fontsize=8)

    color_axis = fig.add_subplot(grid[:, 2])
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(scalar, cax=color_axis)
    cbar.set_label("preceding joint anchor gap (mm)\nfixed absolute 0–8 mm scale",
                   color="#d9e6ef", fontsize=10, labelpad=12)
    cbar.ax.tick_params(colors="#bacbd8", labelsize=9)
    cbar.outline.set_edgecolor("#355169")

    time_s = float(records[0]["render_time_s"][frame])
    fig.suptitle("N=128 cantilever · hard equality versus finite-compliance solvers",
                 x=0.045, y=0.955, ha="left", color="#f2f7fa", fontsize=23,
                 fontweight="bold")
    fig.text(0.958, 0.951, f"t = {time_s:0.3f} s  ·  source {frame:02d}/60",
             ha="right", va="top", color="#54e1c2", fontsize=13,
             family="monospace", fontweight="bold")
    fig.text(
        0.5, 0.042,
        "same authored rest state · fixed camera · exact 3D joint-anchor gap · "
        "10 sequential dt=1/600 substeps per source frame · NO CONTACT",
        ha="center", color="#d7e3eb", fontsize=10.5,
    )
    fig.text(
        0.5, 0.018,
        "Hard equality and finite compliance are different material semantics; shape is qualitative. "
        "Colour and overlays are quantitative.",
        ha="center", color="#ffbe66", fontsize=9.5, fontweight="bold",
    )
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
    plt.close(fig)
    require(image.shape == (HEIGHT, WIDTH, 3), f"bad rendered frame shape {image.shape}")
    require(float(image.std()) > 8.0, "rendered frame is blank")
    return image


def encode(
    records: list[dict[str, Any]],
    run_dir: Path,
    xlim: tuple[float, float],
    zlim: tuple[float, float],
) -> tuple[Path, Path, str]:
    video = run_dir / "n128_hard_vs_finite_2x2.mp4"
    poster = run_dir / "n128_hard_vs_finite_2x2_poster.png"
    temporary = run_dir / f".{video.stem}.{uuid.uuid4().hex}.tmp.mp4"
    owner_guard("open N128 four-panel encoder")
    writer = imageio_ffmpeg.write_frames(
        str(temporary), (WIDTH, HEIGHT), fps=VIDEO_FPS, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2,
        ffmpeg_log_level="warning",
        output_params=["-preset", "medium", "-crf", "18", "-g", str(VIDEO_FPS),
                       "-movflags", "+faststart"],
    )
    writer.send(None)
    raw_digest = hashlib.sha256()
    final_frame: np.ndarray | None = None
    try:
        for frame in range(EXPECTED_FRAMES):
            owner_guard(f"render N128 four-panel frame {frame}")
            image = render_frame(records, frame, xlim, zlim)
            raw_digest.update(image.tobytes())
            writer.send(image.tobytes())
            final_frame = image
    finally:
        writer.close()
    require(final_frame is not None, "no frames encoded")
    owner_guard("publish N128 four-panel video")
    os.replace(temporary, video)
    owner_guard("publish N128 four-panel poster")
    Image.fromarray(final_frame).save(poster, format="PNG", compress_level=6)
    return video, poster, raw_digest.hexdigest()


def media_qa(video: Path, poster: Path) -> dict[str, Any]:
    owner_guard("full N128 media QA")
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(reader)
    decoded_digest = hashlib.sha256()
    frame_stats: list[dict[str, float]] = []
    decoded_count = 0
    for payload in reader:
        array = np.frombuffer(payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        decoded_digest.update(payload)
        frame_stats.append({"mean": float(array.mean()), "std": float(array.std())})
        decoded_count += 1
    reader.close()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-v", "error", "-i", str(video), "-map", "0:v:0",
               "-f", "null", "-"]
    owner_guard("ffmpeg full decode N128 media")
    decoded = subprocess.run(command, text=True, capture_output=True, check=False)
    with Image.open(poster) as image:
        poster_size = image.size
    checks = {
        "count_api_frames": int(count) == EXPECTED_FRAMES,
        "full_reader_frames": decoded_count == EXPECTED_FRAMES,
        "dimensions": tuple(metadata["size"]) == (WIDTH, HEIGHT),
        "fps": abs(float(metadata["fps"]) - VIDEO_FPS) <= 1.0e-9,
        "duration": abs(float(seconds) - EXPECTED_FRAMES / VIDEO_FPS) <= 0.04,
        "codec_h264": str(metadata.get("codec", "")).lower() == "h264",
        "pixel_format_yuv420p": str(metadata.get("pix_fmt", "")).startswith("yuv420p"),
        "all_frames_nonblank": len(frame_stats) == EXPECTED_FRAMES
        and all(row["std"] > 8.0 for row in frame_stats),
        "poster_dimensions": poster_size == (WIDTH, HEIGHT),
        "ffmpeg_full_decode_clean": decoded.returncode == 0 and not decoded.stderr.strip(),
    }
    return {
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "frame_count": int(count),
        "reader_frame_count": decoded_count,
        "duration_s": float(seconds),
        "metadata": metadata,
        "decoded_rgb_stream_sha256": decoded_digest.hexdigest(),
        "frame_statistics": {
            "minimum_std": min(row["std"] for row in frame_stats),
            "maximum_std": max(row["std"] for row in frame_stats),
            "first": frame_stats[0],
            "middle": frame_stats[len(frame_stats) // 2],
            "last": frame_stats[-1],
        },
        "full_decode_command": shlex.join(command),
        "full_decode_stderr": decoded.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    owner = owner_guard("start N128 four-panel review")
    require(SOURCE_RESULT.is_file(), f"missing source result {SOURCE_RESULT}")
    frozen_inputs = {path: sha256_file(path) for path in [SOURCE_RESULT] + [
        SOURCE_ROOT / str(spec["file"]) for spec in METHODS
    ]}
    records, xlim, zlim = load_and_extract()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_root / f"run_{stamp}_{os.getpid()}"
    owner_guard("create immutable N128 review directory")
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    video, poster, raw_sha = encode(records, run_dir, xlim, zlim)
    qa = media_qa(video, poster)
    require(qa["all_pass"], f"media QA failed: {qa['checks']}")
    require(all(sha256_file(path) == digest for path, digest in frozen_inputs.items()),
            "bound inputs changed during rendering")
    manifest = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "owner": owner,
        "claim_boundary": (
            "synchronized N128 no-contact qualitative trajectory plus quantitative "
            "absolute joint-gap colour/overlays; hard and finite rows have different "
            "material semantics; never timing, contact, production, novelty, or universal-winner evidence"
        ),
        "render": {
            "width": WIDTH, "height": HEIGHT, "fps": VIDEO_FPS,
            "frames": EXPECTED_FRAMES, "source_cadence_hz": 60,
            "source_duration_s": 1.0, "encoded_duration_s": EXPECTED_FRAMES / VIDEO_FPS,
            "camera": {"projection": "orthographic x-z", "xlim_m": xlim, "zlim_m": zlim},
            "joint_gap_colour_scale_mm": {"minimum": 0.0, "maximum": GAP_VMAX_MM,
                                            "mapping": "turbo, clipped"},
            "raw_preencode_rgb_stream_sha256": raw_sha,
            "elapsed_s_descriptive": time.perf_counter() - started,
        },
        "methods": [
            {
                "key": row["key"], "title": row["title"], "semantics": row["subtitle"],
                "trajectory_gap_max_m": row["trajectory_gap_max_m"],
                "source": artifact_record(row["path"]),
            }
            for row in records
        ],
        "source_result": artifact_record(SOURCE_RESULT),
        "source_script": artifact_record(Path(__file__).resolve()),
        "outputs": {"video": artifact_record(video), "poster": artifact_record(poster)},
        "media_qa": qa,
    }
    manifest_path = run_dir / "render_manifest.json"
    owner_guard("write N128 review manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": str(manifest_path), "result_sha256": sha256_file(manifest_path),
        "video": str(video), "video_sha256": sha256_file(video),
        "poster": str(poster), "poster_sha256": sha256_file(poster),
        "qa_pass": qa["all_pass"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
