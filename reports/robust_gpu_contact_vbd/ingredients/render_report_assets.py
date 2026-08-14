# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the static plots and MP4 videos from Horde-recorded traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


INK = "#17202a"
MUTED = "#5f6f82"
LINE = "#d9e0e8"
ACCENT = "#1f6f8b"
GOOD = "#16845b"
WARN = "#c57b00"
BAD = "#c43d4b"
PANEL = "#f7f9fb"
CABLE_COLORS = ("#36c5f0", "#9b7ede", "#ffb45c")
WIDTH = 1280
HEIGHT = 720


def _style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.grid(color=LINE, linewidth=0.7, alpha=0.7)


def _figure_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3])


def _clear_figure_text(fig) -> None:
    for artist in tuple(fig.texts):
        artist.remove()


def _write_video(path: Path, frame_callback, frame_count: int, poster_path: Path, poster_frame: int) -> None:
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (WIDTH, HEIGHT),
        fps=30,
        quality=7,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    poster = None
    try:
        for frame_index in range(frame_count):
            frame = frame_callback(frame_index)
            writer.send(frame.tobytes())
            if frame_index == poster_frame:
                poster = frame.copy()
    finally:
        writer.close()
    if poster is None:
        poster = frame_callback(frame_count - 1)
    Image.fromarray(poster).save(poster_path, quality=91, optimize=True)


def _sphere_position(progress: float, initial_z: float, final_z: float, event: bool, result: dict) -> float:
    if not event:
        return initial_z + progress * (final_z - initial_z)
    event_fraction = result["start_gap_m"] / result["potential_displacement_m"]
    if progress <= event_fraction:
        return initial_z - result["potential_displacement_m"] * progress
    return final_z


def render_fast_impact(assets: Path, traces, results: dict) -> None:
    sphere_results = results["sphere_plane_10000_m_s"]
    ordinary = next(item for item in sphere_results if item["method"] == "ordinary_vbd")
    event = next(item for item in sphere_results if item["method"] == "event")
    initial_z = float(traces["sphere_vbd_initial_pose"][2])
    final_z = (
        float(traces["sphere_vbd_final_pose"][2]),
        float(traces["sphere_event_final_pose"][2]),
    )
    radius = 0.05
    motion_frames = 105
    frame_count = 135

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")

    def draw(frame_index: int) -> np.ndarray:
        _clear_figure_text(fig)
        if frame_index < motion_frames:
            u = frame_index / (motion_frames - 1)
            progress = u**3
        else:
            progress = 1.0
        for ax, title, result, z_end, is_event in zip(
            axes,
            ("Ordinary one-step VBD", "Queried event + direct impulse"),
            (ordinary, event),
            final_z,
            (False, True),
            strict=True,
        ):
            ax.clear()
            _style_axis(ax)
            ax.axhspan(-0.20, 0.0, color="#fdecee", alpha=0.9)
            ax.axhline(0.0, color=INK, linewidth=2.0)
            ax.text(-0.235, 0.015, "contact plane", color=MUTED, fontsize=9)
            z = _sphere_position(progress, initial_z, z_end, is_event, result)
            if z >= -0.14:
                circle = Circle((0.0, z), radius, facecolor=GOOD if is_event else BAD, edgecolor="white", linewidth=1.5)
                ax.add_patch(circle)
            else:
                ax.annotate(
                    "",
                    xy=(0.0, -0.16),
                    xytext=(0.0, -0.08),
                    arrowprops={"arrowstyle": "-|>", "color": BAD, "linewidth": 3},
                )
                ax.text(0.0, -0.055, f"off-screen: z = {z:.2f} m", ha="center", color=BAD, fontsize=10, weight="bold")
            ax.set_xlim(-0.25, 0.25)
            ax.set_ylim(-0.20, 0.62)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_ylabel("height z [m]", color=MUTED)
            ax.set_title(title, color=INK, fontsize=14, weight="bold", pad=12)
            status = (
                f"final gap {result['final_signed_gap_m'] * 1e6:+.1f} µm\n"
                f"final vₙ {result['final_normal_velocity_m_s']:+.3g} m/s"
            )
            ax.text(
                0.03,
                0.96,
                status,
                transform=ax.transAxes,
                va="top",
                color=GOOD if is_event else BAD,
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": LINE, "alpha": 0.95},
            )
        physical_us = 0.03 * progress * 1.0e6
        fig.suptitle(
            "10,000 m/s sphere → plane, one 30 ms frame",
            x=0.5,
            y=0.988,
            fontsize=20,
            color=INK,
            weight="bold",
        )
        fig.text(
            0.5,
            0.935,
            f"Large-step endpoint visualization · slowed near impact · frame time {physical_us:,.0f} µs",
            ha="center",
            color=MUTED,
            fontsize=11,
        )
        fig.text(
            0.5,
            0.025,
            "Both methods received the same queried sphere–plane pair. Red is penetration; green is feasible space.",
            ha="center",
            color=MUTED,
            fontsize=10,
        )
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.82, wspace=0.20)
        return _figure_rgb(fig)

    _write_video(
        assets / "fast_impact_comparison.mp4",
        draw,
        frame_count,
        assets / "fast_impact_poster.jpg",
        frame_count - 1,
    )
    plt.close(fig)


def render_dense_chain(assets: Path, traces, results: dict) -> None:
    dense_results = results["dense_equal_mass_chain"]
    ordinary = next(item for item in dense_results if item["method"] == "ordinary_vbd")
    event = next(item for item in dense_results if item["method"] == "event_pcg")
    initial = traces["dense_vbd_initial_pose"][:, 0]
    endpoints = (traces["dense_vbd_final_pose"][:, 0], traces["dense_event_final_pose"][:, 0])
    velocities = (traces["dense_vbd_final_velocity"][:, 0], traces["dense_event_final_velocity"][:, 0])
    radius = 0.05
    motion_frames = 90
    frame_count = 120
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")

    def draw(frame_index: int) -> np.ndarray:
        _clear_figure_text(fig)
        progress = min(1.0, frame_index / (motion_frames - 1))
        progress = 0.5 - 0.5 * np.cos(np.pi * progress)
        for ax, title, endpoint, velocity, result, color in zip(
            axes,
            ("Ordinary one-step VBD", "Event-time dense normal PCG"),
            endpoints,
            velocities,
            (ordinary, event),
            (BAD, GOOD),
            strict=True,
        ):
            ax.clear()
            _style_axis(ax)
            x = initial + progress * (endpoint - initial)
            ax.axhline(0.0, color=LINE, linewidth=1.0)
            for index, position in enumerate(x):
                circle = Circle((float(position), 0.0), radius, facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.95)
                ax.add_patch(circle)
                if progress > 0.94:
                    ax.text(position, -0.10, str(index), ha="center", color=MUTED, fontsize=7)
            ax.set_xlim(-1.35, 2.35)
            ax.set_ylim(-0.18, 0.18)
            ax.set_yticks([])
            ax.set_xlabel("x [m]", color=MUTED)
            ax.set_title(title, loc="left", color=INK, fontsize=13, weight="bold")
            ax.text(
                0.99,
                0.84,
                f"max |v-v*| = {result['maximum_common_velocity_error_m_s']:.3g} m/s\n"
                f"gap = [{result['minimum_gap_m']:.2e}, {result['maximum_gap_m']:.2e}] m\n"
                f"momentum error = {result['momentum_error_kg_m_s']:.3g}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color=color,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": LINE, "alpha": 0.94},
            )
        fig.suptitle(
            "Dense 16-sphere simultaneous impact",
            x=0.5,
            y=0.97,
            fontsize=20,
            color=INK,
            weight="bold",
        )
        fig.text(
            0.5,
            0.925,
            "Initial velocity 20 m/s on body 0 · endpoint positions interpolated only for viewing",
            ha="center",
            color=MUTED,
            fontsize=11,
        )
        fig.subplots_adjust(left=0.065, right=0.98, bottom=0.09, top=0.87, hspace=0.48)
        return _figure_rgb(fig)

    _write_video(
        assets / "dense_chain_comparison.mp4",
        draw,
        frame_count,
        assets / "dense_chain_poster.jpg",
        frame_count - 1,
    )
    plt.close(fig)


def _configure_3d_axis(ax, title: str) -> None:
    ax.set_facecolor("white")
    ax.set_xlim(-4.6, 2.0)
    ax.set_ylim(-5.0, 7.5)
    ax.set_zlim(-0.2, 5.0)
    ax.set_box_aspect((6.6, 12.5, 5.2))
    ax.view_init(elev=27, azim=-56)
    ax.set_xlabel("x [m]", color=MUTED, labelpad=6)
    ax.set_ylabel("y [m]", color=MUTED, labelpad=7)
    ax.set_zlabel("z [m]", color=MUTED, labelpad=5)
    ax.tick_params(colors=MUTED, labelsize=7, pad=1)
    ax.set_title(title, color=INK, fontsize=13, weight="bold", pad=10)
    ground_x, ground_y = np.meshgrid(np.linspace(-4.6, 2.0, 2), np.linspace(-5.0, 7.5, 2))
    ax.plot_surface(ground_x, ground_y, np.zeros_like(ground_x), color="#e8edf3", alpha=0.42, shade=False)


def render_cable_comparison(
    assets: Path,
    traces,
    left_prefix: str,
    right_prefix: str,
    left_title: str,
    right_title: str,
    filename: str,
    poster: str,
) -> None:
    left_positions = traces[f"{left_prefix}_positions"]
    right_positions = traces[f"{right_prefix}_positions"]
    frame_count = min(left_positions.shape[0], right_positions.shape[0])
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")
    axes = (fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d"))

    def draw(frame_index: int) -> np.ndarray:
        _clear_figure_text(fig)
        for ax, prefix, positions, title in zip(
            axes,
            (left_prefix, right_prefix),
            (left_positions, right_positions),
            (left_title, right_title),
            strict=True,
        ):
            ax.clear()
            _configure_3d_axis(ax, title)
            current = positions[frame_index]
            for cable_index, cable in enumerate(current):
                color = CABLE_COLORS[cable_index]
                ax.plot(cable[:, 0], cable[:, 1], cable[:, 2], color=color, linewidth=2.3)
                ax.scatter(cable[::4, 0], cable[::4, 1], cable[::4, 2], color=color, s=7, depthshade=False)
            if frame_index:
                gap = float(traces[f"{prefix}_fresh_gap_m"][frame_index - 1])
                substeps = int(traces[f"{prefix}_selected_substeps"][frame_index - 1])
                missing = int(traces[f"{prefix}_missing_pairs"][frame_index - 1])
            else:
                gap, substeps, missing = 0.0, 0, 0
            penetration_mm = max(0.0, -gap) * 1000.0
            status_color = GOOD if penetration_mm < 1.0 else (WARN if penetration_mm < 5.0 else BAD)
            ax.text2D(
                0.03,
                0.95,
                f"fresh penetration {penetration_mm:.3f} mm\nsubsteps {substeps} · newly missed pairs {missing}",
                transform=ax.transAxes,
                va="top",
                color=status_color,
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": LINE, "alpha": 0.94},
            )
        fig.suptitle(
            "20 rad/s cable twist: actual Horde frame states",
            x=0.5,
            y=0.988,
            fontsize=20,
            color=INK,
            weight="bold",
        )
        fig.text(
            0.5,
            0.94,
            f"simulation frame {frame_index:03d} / {frame_count - 1:03d}",
            ha="center",
            color=MUTED,
            fontsize=11,
        )
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.015, top=0.84, wspace=-0.02)
        return _figure_rgb(fig)

    _write_video(assets / filename, draw, frame_count, assets / poster, frame_count - 1)
    plt.close(fig)


def render_plots(assets: Path, traces, results: dict, costs: dict) -> None:
    dense = results["dense_equal_mass_chain"]
    ordinary = next(item for item in dense if item["method"] == "ordinary_vbd")
    event = next(item for item in dense if item["method"] == "event_pcg")
    body = np.arange(16)
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=140)
    _style_axis(ax)
    ax.plot(body, traces["dense_vbd_final_velocity"][:, 0], "o-", color=BAD, label="ordinary VBD")
    ax.plot(body, traces["dense_event_final_velocity"][:, 0], "o-", color=GOOD, label="event + PCG")
    ax.axhline(event["analytic_common_velocity_m_s"], color=ACCENT, linestyle="--", label="analytic inelastic velocity")
    ax.set_xlabel("sphere index")
    ax.set_ylabel("final x velocity [m/s]")
    ax.set_title("Dense impact velocity closure", color=INK, fontsize=16, weight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(assets / "dense_velocity_closure.png", bbox_inches="tight")
    plt.close(fig)

    baseline = results["cable_twist_unconditioned"]
    conditioned = results["cable_twist_conditioned"]
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=140)
    _style_axis(ax)
    x = [baseline["median_frame_time_ms"]] + [item["median_frame_time_ms"] for item in conditioned]
    y = [baseline["maximum_fresh_penetration_m"] * 1000.0] + [
        item["maximum_fresh_penetration_m"] * 1000.0 for item in conditioned
    ]
    labels = ["ρ=0, target 10 mm"] + [
        f"ρ=5, target {item['target_surface_motion_m'] * 1000:.0f} mm" for item in conditioned
    ]
    colors = [BAD, GOOD, GOOD, GOOD]
    for px, py, label, color in zip(x, y, labels, colors, strict=True):
        ax.scatter(px, py, s=80, color=color, zorder=3)
        ax.annotate(label, (px, py), xytext=(8, 7), textcoords="offset points", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("median frame time [ms]")
    ax.set_ylabel("worst fresh penetration [mm, log scale]")
    ax.set_title("Cable contact accuracy / cost tradeoff", color=INK, fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(assets / "cable_accuracy_cost.png", bbox_inches="tight")
    plt.close(fig)

    dense_cost = costs["dense_16_equal_mass"]
    batch_cost = costs["independent_sphere_plane_1024"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), dpi=140)
    fig.patch.set_facecolor("white")
    comparisons = (
        (
            axes[0],
            "16-body coupled dense impact",
            [dense_cost[0]["median_graph_gpu_ms"], dense_cost[1]["median_graph_gpu_ms"]],
        ),
        (
            axes[1],
            "1,024 independent sphere-plane impacts",
            [batch_cost[0]["median_solve_gpu_ms"], batch_cost[1]["median_solve_gpu_ms"]],
        ),
    )
    for ax, title, values in comparisons:
        _style_axis(ax)
        bars = ax.bar(
            ("ordinary VBD", "routed contact"),
            values,
            color=(BAD, GOOD),
            width=0.62,
        )
        ax.set_title(title, color=INK, fontsize=13, weight="bold", pad=12)
        ax.set_ylabel("complete CUDA-graph replay [ms]")
        ax.set_ylim(0.0, max(values) * 1.35)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(values) * 0.035,
                f"{value:.3f} ms",
                ha="center",
                color=INK,
                fontsize=10,
                weight="bold",
            )
        ax.text(
            0.5,
            0.91,
            f"{values[1] / values[0]:.2f}× cost",
            transform=ax.transAxes,
            ha="center",
            color=MUTED,
            fontsize=10,
        )
    fig.suptitle(
        "Matched Horde L40 cost: collision + solve, 100 captured replays",
        color=INK,
        fontsize=17,
        weight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(assets / "captured_gpu_cost.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    with (args.assets / "horde_results.json").open(encoding="utf-8") as source:
        results = json.load(source)
    with (args.assets / "horde_cost_results.json").open(encoding="utf-8") as source:
        costs = json.load(source)
    traces = np.load(args.assets / "horde_traces.npz")

    if not args.plots_only:
        render_fast_impact(args.assets, traces, results)
        render_dense_chain(args.assets, traces, results)
        render_cable_comparison(
            args.assets,
            traces,
            "cable_default_10mm",
            "cable_conditioned_10mm",
            "Unconditioned · target 10 mm",
            "Conditioned ρ=5 · target 10 mm",
            "cable_conditioning_comparison.mp4",
            "cable_conditioning_poster.jpg",
        )
        render_cable_comparison(
            args.assets,
            traces,
            "cable_conditioned_10mm",
            "cable_conditioned_1mm",
            "Fast · target 10 mm",
            "High robustness · target 1 mm",
            "cable_accuracy_comparison.mp4",
            "cable_accuracy_poster.jpg",
        )
    render_plots(args.assets, traces, results, costs)


if __name__ == "__main__":
    main()
