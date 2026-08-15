# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render final-policy cable-pile traces with Newton ViewerGL on Horde."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import render_horde_newton_videos as common
import warp as wp
from newton.examples.cable.example_cable_pile import Example


PALETTE = (
    (54, 174, 240),
    (174, 125, 238),
    (255, 177, 83),
    (45, 207, 137),
    (244, 104, 133),
)


def _configure_model(viewer, scenario: str):
    example = Example(
        viewer,
        layers=5,
        lanes_per_layer=4,
        num_elements=40,
        drop_mode=scenario,
        drop_height=5.0,
        staged_base_height=0.65,
        staged_layer_spacing=0.52,
        release_interval_frames=90,
        capture_graph=False,
    )
    shape_body = example.model.shape_body.numpy()
    colors = example.model.shape_color.numpy()
    for bodies, color in zip(example.layer_bodies, PALETTE, strict=True):
        colors[np.isin(shape_body, np.asarray(bodies, dtype=np.int32))] = (
            np.asarray(color, dtype=np.float32) / 255.0
        )
    example.model.shape_color.assign(colors)
    viewer.set_model(example.model)
    return example


def _render_case(*, assets: Path, traces, results: dict, scenario: str) -> None:
    viewer = common._viewer(common.WIDTH, 520)
    example = _configure_model(viewer, scenario)
    state = example.model.state()
    poses = traces[f"{scenario}_poses"]
    sim_frames = traces[f"{scenario}_frames"]
    substeps = traces[f"{scenario}_substeps"]
    replays = traces[f"{scenario}_replays"]
    penetration = traces[f"{scenario}_penetrations"]
    metrics = results[scenario]
    title = (
        "Cable pile dropped from 5 m · 800 coupled rigid bodies"
        if scenario == "high-drop"
        else "Five-layer cable pile · separate releases · 800 bodies"
    )
    subtitle = (
        "final rho=1 policy · bounded swept admission · contact + joint certificate · CUDA-graph replay"
    )
    initial_hold = 30
    final_hold = 60
    state_indices = [0] * initial_hold + list(range(len(poses))) + [len(poses) - 1] * final_hold

    def frames():
        for output_index, pose_index in enumerate(state_indices):
            state.body_q.assign(poses[pose_index])
            frame = int(sim_frames[pose_index])
            center_z = float(np.mean(poses[pose_index, :, 2]))
            viewer.set_camera(wp.vec3(3.15, -4.55, center_z + 1.85), 0.0, 0.0)
            viewer.camera.look_at(wp.vec3(0.0, 0.0, max(0.18, center_z)))
            scene = common._render(viewer, state, max(frame, 0) / 60.0)
            canvas = Image.new("RGB", (common.WIDTH, common.HEIGHT), common.BG)
            canvas.paste(Image.fromarray(scene), (0, 103))
            draw = ImageDraw.Draw(canvas)
            common._header(draw, title, subtitle)
            pen_mm = float(penetration[pose_index]) * 1000.0
            draw.rounded_rectangle((30, 500, 715, 607), 11, fill=common.PANEL)
            if frame < 0:
                status = "Initial state · simulation begins after this hold"
            elif scenario == "staged":
                active = min(5, 1 + frame // 90)
                status = f"simulation frame {frame:03d}/{metrics['frames'] - 1:03d} · active layers {active}/5"
            else:
                status = f"simulation frame {frame:03d}/{metrics['frames'] - 1:03d} · complete pile in free fall / contact"
            draw.text((50, 516), status, font=common.FONTS["panel"], fill=common.INK)
            draw.text(
                (50, 557),
                f"fresh penetration {pen_mm:.3f} mm · accepted physical steps {int(substeps[pose_index])}\n"
                f"exact replays this frame {int(replays[pose_index])}",
                font=common.FONTS["small"],
                fill=common.GOOD if pen_mm <= 1.0 else common.BAD,
                spacing=5,
            )
            draw.rounded_rectangle((760, 500, 1250, 607), 11, fill=common.PANEL)
            draw.text((780, 516), "Complete Horde result", font=common.FONTS["panel"], fill=common.INK)
            draw.text(
                (780, 557),
                f"worst contact {metrics['maximum_fresh_penetration_m'] * 1000.0:.3f} mm · "
                f"joint {metrics['maximum_joint_translation_residual_m'] * 1000.0:.3f} mm\n"
                f"median {metrics['median_frame_time_ms']:.2f} ms · all certificate debt 0",
                font=common.FONTS["small"],
                fill=common.GOOD,
                spacing=5,
            )
            common._footer(
                draw,
                "Simulation and Newton ViewerGL rendering ran on Horde L40 · displayed poses are sampled every four 60 Hz frames",
            )
            yield np.asarray(canvas)

    stem = "cable_pile_highdrop_800" if scenario == "high-drop" else "cable_pile_staged_800"
    common._write_video(
        assets / f"{stem}.mp4",
        frames(),
        poster=assets / f"{stem}_poster.jpg",
        poster_index=initial_hold + (18 if scenario == "high-drop" else min(102, len(poses) - 1)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    traces = np.load(args.traces)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    with wp.ScopedDevice("cuda:0"):
        _render_case(assets=args.assets, traces=traces, results=results, scenario="high-drop")
        _render_case(assets=args.assets, traces=traces, results=results, scenario="staged")


if __name__ == "__main__":
    main()
