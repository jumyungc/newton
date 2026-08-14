# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the report's comparison videos with Newton ViewerGL on Horde.

The impact clips replay exact Horde solver endpoints in Newton geometry.  The
cable clips rerun every physical frame on Horde and capture Newton's GPU
framebuffer.  Pillow is used only for labels/compositing and FFmpeg only for
encoding; neither changes simulation state.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pyglet
from PIL import Image, ImageDraw, ImageFont

pyglet.options["headless"] = True

import warp as wp

import newton
from newton.examples.cable.example_cable_twist import Example
from newton.tests import benchmark_cable_twist_contact_strategy as cable_benchmark
from newton.viewer import ViewerGL


WIDTH = 1280
HEIGHT = 720
FPS = 30
INK = (235, 241, 248)
MUTED = (171, 184, 199)
GOOD = (45, 207, 137)
WARN = (255, 177, 67)
BAD = (244, 84, 105)
BLUE = (54, 174, 240)
PURPLE = (174, 125, 238)
ORANGE = (255, 177, 83)
BG = (12, 18, 27)
PANEL = (22, 31, 44)


def _font(size: int, bold: bool = False):
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONTS = {
    "title": _font(31, True),
    "subtitle": _font(18),
    "panel": _font(19, True),
    "metric": _font(16),
    "small": _font(14),
}


def _rgb(color: tuple[int, int, int]) -> wp.vec3:
    return wp.vec3(*(component / 255.0 for component in color))


def _ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _write_video(path: Path, frames, *, poster: Path, poster_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (WIDTH, HEIGHT),
        fps=FPS,
        quality=8,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    selected = None
    try:
        for index, frame in enumerate(frames):
            pixels = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
            writer.send(pixels.tobytes())
            if index == poster_index:
                selected = pixels.copy()
    finally:
        writer.close()
    if selected is None:
        raise RuntimeError(f"poster frame {poster_index} was not generated for {path}")
    Image.fromarray(selected).save(poster, quality=92, optimize=True)


def _viewer(width: int, height: int) -> ViewerGL:
    viewer = ViewerGL(width=width, height=height, headless=True)
    viewer.show_ui = False
    return viewer


def _render(viewer: ViewerGL, state, time_s: float, arrows=None) -> np.ndarray:
    viewer.begin_frame(time_s)
    viewer.log_state(state)
    if arrows is None:
        viewer.log_arrows("/velocity", None, None, None)
    else:
        starts, ends, colors = arrows
        viewer.log_arrows(
            "/velocity",
            wp.array(starts, dtype=wp.vec3, device=viewer.device),
            wp.array(ends, dtype=wp.vec3, device=viewer.device),
            wp.array(colors, dtype=wp.vec3, device=viewer.device),
        )
    viewer.end_frame()
    return viewer.get_frame(render_ui=False).numpy()


def _header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.rectangle((0, 0, WIDTH, 103), fill=BG)
    draw.text((36, 17), title, font=FONTS["title"], fill=INK)
    draw.text((37, 61), subtitle, font=FONTS["subtitle"], fill=MUTED)


def _footer(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.rectangle((0, 623, WIDTH, HEIGHT), fill=BG)
    draw.text((36, 648), text, font=FONTS["small"], fill=MUTED)


def render_fast_impact(assets: Path, traces, results: dict, viewer: ViewerGL) -> None:
    values = results["sphere_plane_10000_m_s"]
    ordinary = next(item for item in values if item["method"] == "ordinary_vbd")
    event = next(item for item in values if item["method"] == "event")
    initial_z = float(traces["sphere_vbd_initial_pose"][2])
    radius = 0.05

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    bodies = []
    for x, color in ((-0.62, BAD), (0.62, GOOD)):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, initial_z)))
        builder.add_shape_sphere(body, radius=radius, color=_rgb(color))
        bodies.append(body)
    builder.add_ground_plane()
    model = builder.finalize()
    state = model.state()
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(1.55, -2.8, 1.15), 0.0, 0.0)
    viewer.camera.look_at(wp.vec3(0.0, 0.0, 0.20))

    def frames():
        count = 180
        for index in range(count):
            if index < 30:
                u = 0.0
            elif index < 105:
                u = _ease((index - 30) / 74.0)
            else:
                u = 1.0
            contact_z = radius + event["final_signed_gap_m"]
            right_z = initial_z + u * (contact_z - initial_z)
            # The tunneled endpoint is far outside the camera.  Stop the visible
            # replay below the floor, while the label reports the exact endpoint.
            left_visible_end = -0.32
            left_z = initial_z + u * (left_visible_end - initial_z)
            pose = np.zeros((2, 7), dtype=np.float32)
            pose[:, 6] = 1.0
            pose[0, :3] = (-0.62, 0.0, left_z)
            pose[1, :3] = (0.62, 0.0, right_z)
            state.body_q.assign(pose)
            arrows = None
            if index >= 105:
                arrows = (
                    np.asarray(((-0.62, 0.0, 0.13),), dtype=np.float32),
                    np.asarray(((-0.62, 0.0, -0.12),), dtype=np.float32),
                    np.asarray((_rgb(BAD),), dtype=np.float32),
                )
            scene = _render(viewer, state, index / FPS, arrows)
            canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
            canvas.paste(Image.fromarray(scene), (0, 103))
            draw = ImageDraw.Draw(canvas)
            _header(
                draw,
                "Fast queried impact · 10,000 m/s sphere → plane",
                "One 30 ms solver step · slowed endpoint replay · same queried pair",
            )
            draw.text((144, 116), "ordinary one-step VBD", font=FONTS["panel"], fill=BAD)
            draw.text((774, 116), "event-time direct impulse", font=FONTS["panel"], fill=GOOD)
            if index >= 105:
                draw.line((548, 338, 548, 431), fill=BAD, width=5)
                draw.polygon(((536, 420), (560, 420), (548, 441)), fill=BAD)
                draw.text((374, 448), "tunneled far below the view", font=FONTS["small"], fill=BAD)
            draw.rounded_rectangle((65, 526, 605, 613), 12, fill=PANEL)
            draw.rounded_rectangle((675, 526, 1215, 613), 12, fill=PANEL)
            draw.text(
                (88, 543),
                f"endpoint gap  {ordinary['final_signed_gap_m']:+.3f} m\n"
                f"endpoint normal velocity  {ordinary['final_normal_velocity_m_s']:+.3f} m/s",
                font=FONTS["metric"],
                fill=BAD,
                spacing=8,
            )
            draw.text(
                (698, 543),
                f"endpoint gap  {event['final_signed_gap_m'] * 1e6:+.2f} µm\n"
                f"endpoint normal velocity  {event['final_normal_velocity_m_s']:+.3g} m/s",
                font=FONTS["metric"],
                fill=GOOD,
                spacing=8,
            )
            _footer(draw, "Newton ViewerGL framebuffer captured on Horde L40 · intermediate motion is explanatory, endpoints are exact")
            yield np.asarray(canvas)

    _write_video(
        assets / "fast_impact_comparison.mp4",
        frames(),
        poster=assets / "fast_impact_poster.jpg",
        poster_index=150,
    )


def render_dense_chain(assets: Path, traces, results: dict, viewer: ViewerGL) -> None:
    values = results["dense_equal_mass_chain"]
    ordinary = next(item for item in values if item["method"] == "ordinary_vbd")
    event = next(item for item in values if item["method"] == "event_pcg")
    initial = traces["dense_vbd_initial_pose"][:, 0].astype(np.float32)
    final_ordinary = traces["dense_vbd_final_pose"][:, 0].astype(np.float32)
    final_event = traces["dense_event_final_pose"][:, 0].astype(np.float32)
    velocity_ordinary = traces["dense_vbd_final_velocity"][:, 0].astype(np.float32)
    velocity_event = traces["dense_event_final_velocity"][:, 0].astype(np.float32)
    radius = 0.06

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for row, color in ((0, BAD), (1, GOOD)):
        row_z = 0.55 if row == 0 else 0.16
        for x in initial:
            body = builder.add_body(xform=wp.transform(wp.vec3(float(x), 0.0, row_z)))
            builder.add_shape_sphere(body, radius=radius, color=_rgb(color))
    builder.add_ground_plane()
    model = builder.finalize()
    state = model.state()
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(0.75, -3.7, 1.0), 0.0, 0.0)
    viewer.camera.look_at(wp.vec3(0.68, 0.0, 0.32))

    def frames():
        count = 210
        for index in range(count):
            if index < 45:
                u = 0.0
            elif index < 135:
                u = _ease((index - 45) / 89.0)
            else:
                u = 1.0
            x0 = initial + u * (final_ordinary - initial)
            x1 = initial + u * (final_event - initial)
            pose = np.zeros((32, 7), dtype=np.float32)
            pose[:, 6] = 1.0
            pose[:16, 0] = x0
            pose[:16, 2] = 0.55
            pose[16:, 0] = x1
            pose[16:, 2] = 0.16
            state.body_q.assign(pose)

            starts = []
            ends = []
            colors = []
            current_v0 = (1.0 - u) * np.r_[20.0, np.zeros(15)] + u * velocity_ordinary
            current_v1 = (1.0 - u) * np.r_[20.0, np.zeros(15)] + u * velocity_event
            for xs, z, velocities, color in (
                (x0, 0.55, current_v0, BAD),
                (x1, 0.16, current_v1, GOOD),
            ):
                scale = max(1.0, float(np.max(np.abs(velocities))))
                for x, velocity in zip(xs, velocities, strict=True):
                    if abs(float(velocity)) < 1.0e-4:
                        continue
                    length = 0.08 + 0.26 * math.sqrt(abs(float(velocity)) / scale)
                    starts.append((float(x), 0.0, z + 0.075))
                    ends.append((float(x + math.copysign(length, float(velocity))), 0.0, z + 0.075))
                    colors.append(tuple(component / 255.0 for component in color))
            arrows = (
                np.asarray(starts, dtype=np.float32),
                np.asarray(ends, dtype=np.float32),
                np.asarray(colors, dtype=np.float32),
            )
            scene = _render(viewer, state, index / FPS, arrows)
            canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
            canvas.paste(Image.fromarray(scene), (0, 103))
            draw = ImageDraw.Draw(canvas)
            _header(
                draw,
                "Dense simultaneous impact · 16 touching spheres",
                "body 0 enters at 20 m/s · arrows show velocity direction · endpoint morph is deliberately slow",
            )
            draw.rounded_rectangle((30, 118, 510, 199), 10, fill=PANEL)
            draw.text((48, 130), "ordinary VBD", font=FONTS["panel"], fill=BAD)
            draw.text(
                (48, 162),
                f"max |v−v*| {ordinary['maximum_common_velocity_error_m_s']:.2f} m/s · gap [{ordinary['minimum_gap_m']:.2f}, {ordinary['maximum_gap_m']:.2f}] m",
                font=FONTS["small"],
                fill=BAD,
            )
            draw.rounded_rectangle((30, 450, 570, 531), 10, fill=PANEL)
            draw.text((48, 462), "event-time dense normal PCG", font=FONTS["panel"], fill=GOOD)
            draw.text(
                (48, 494),
                f"max |v−v*| {event['maximum_common_velocity_error_m_s']:.2e} m/s · gap [{event['minimum_gap_m']:.2e}, {event['maximum_gap_m']:.2e}] m",
                font=FONTS["small"],
                fill=GOOD,
            )
            _footer(
                draw,
                "Newton ViewerGL on Horde L40 · final states are exact · event PCG solves the coupled impact island, not 15 isolated pairs",
            )
            yield np.asarray(canvas)

    _write_video(
        assets / "dense_chain_comparison.mp4",
        frames(),
        poster=assets / "dense_chain_poster.jpg",
        poster_index=175,
    )


def _configure_cable_example(
    *,
    viewer: ViewerGL,
    target_surface_motion: float,
    inertial_contact_ratio: float,
    twist_rate: float,
):
    example = Example(viewer, argparse.Namespace())
    example.sim_substeps = 10
    example.sim_dt = example.frame_dt / example.sim_substeps
    example.update_step_interval = 10
    example.first_twist_rates.assign(
        np.full(example.kinematic_bodies.shape[0], twist_rate, dtype=np.float32)
    )
    if inertial_contact_ratio:
        example.collision_pipeline = newton.CollisionPipeline(
            example.model,
            contact_matching="disabled",
        )
        example.contacts = example.collision_pipeline.contacts()
        example.solver = newton.solvers.SolverVBD(
            example.model,
            iterations=5,
            rigid_avbd_contact_alpha=0.0,
            rigid_contact_history=False,
            rigid_contact_inertial_stiffness_ratio=inertial_contact_ratio,
            rigid_contact_feasibility_projection=True,
        )
    else:
        example.solver.rigid_contact_feasibility_projection = True

    collision_radii = cable_benchmark._body_collision_radii(example.model)
    speed_np = np.zeros(example.model.body_count, dtype=np.float32)
    kinematic_ids = example.kinematic_bodies.numpy()
    speed_np[kinematic_ids] = abs(twist_rate) * collision_radii[kinematic_ids]
    speed = wp.array(speed_np, dtype=float, device=example.model.device)
    example.collision_pipeline.collide(example.state_0, example.contacts, dt=example.frame_dt)
    example.solver.update_rigid_contact_island_schedule(
        example.state_0,
        example.contacts,
        example.frame_dt,
        target_surface_motion,
        minimum_substeps=10,
        maximum_substeps=320,
        body_surface_speed_override=speed,
    )
    adaptive_args = argparse.Namespace(
        target_surface_motion=target_surface_motion,
        twist_rate=twist_rate,
    )
    example.graph, required_substeps = cable_benchmark._capture_adaptive_contact_substeps(
        example,
        adaptive_args,
    )

    shape_body = example.model.shape_body.numpy()
    colors = example.model.shape_color.numpy()
    for cable_bodies, color in zip(
        example.cable_bodies_list,
        (BLUE, PURPLE, ORANGE),
        strict=True,
    ):
        mask = np.isin(shape_body, np.asarray(cable_bodies, dtype=np.int32))
        colors[mask] = np.asarray(color, dtype=np.float32) / 255.0
    example.model.shape_color.assign(colors)

    viewer.set_camera(wp.vec3(7.8, -12.6, 8.1), 0.0, 0.0)
    viewer.camera.look_at(wp.vec3(-0.7, 0.7, 1.0))
    return example, required_substeps


def _render_cable_case(
    *,
    viewer: ViewerGL,
    work_dir: Path,
    key: str,
    frames: int,
    target_surface_motion: float,
    inertial_contact_ratio: float,
    twist_rate: float,
) -> dict:
    frame_dir = work_dir / key
    frame_dir.mkdir(parents=True, exist_ok=True)
    example, required_substeps = _configure_cable_example(
        viewer=viewer,
        target_surface_motion=target_surface_motion,
        inertial_contact_ratio=inertial_contact_ratio,
        twist_rate=twist_rate,
    )
    verification_pipeline = newton.CollisionPipeline(example.model)
    verification_contacts = verification_pipeline.contacts()
    fresh_gap = []
    missing_pairs = []
    substeps = []

    def save_frame(index: int) -> None:
        viewer.begin_frame(example.sim_time)
        viewer.log_state(example.state_0)
        viewer.end_frame()
        pixels = viewer.get_frame(render_ui=False).numpy()
        Image.fromarray(pixels).save(frame_dir / f"{index:04d}.jpg", quality=92, optimize=True)

    save_frame(0)
    for index in range(1, frames + 1):
        example.step()
        retained = cable_benchmark._contact_pairs(example.contacts)
        verification_pipeline.collide(example.state_0, verification_contacts)
        fresh = cable_benchmark._contact_pairs(verification_contacts)
        gap, _ = cable_benchmark._minimum_fresh_gap(
            example.model,
            example.state_0,
            verification_contacts,
        )
        fresh_gap.append(float(gap))
        missing_pairs.append(len(fresh - retained))
        substeps.append(int(required_substeps.numpy()[0]))
        save_frame(index)

    return {
        "key": key,
        "frames": frames,
        "target_surface_motion_m": target_surface_motion,
        "inertial_contact_stiffness_ratio": inertial_contact_ratio,
        "twist_rate_rad_s": twist_rate,
        "fresh_gap_m": fresh_gap,
        "missing_pairs": missing_pairs,
        "selected_substeps": substeps,
        "maximum_fresh_penetration_m": max(0.0, -min(fresh_gap)),
        "max_missing_pairs_per_frame": max(missing_pairs),
    }


def _compose_cable_video(
    *,
    assets: Path,
    work_dir: Path,
    left: dict,
    right: dict,
    left_title: str,
    right_title: str,
    filename: str,
    poster: str,
) -> None:
    state_count = min(left["frames"], right["frames"]) + 1

    def frames():
        for state_index in range(state_count):
            left_scene = Image.open(work_dir / left["key"] / f"{state_index:04d}.jpg").convert("RGB")
            right_scene = Image.open(work_dir / right["key"] / f"{state_index:04d}.jpg").convert("RGB")
            for _duplicate in range(2):  # 60 Hz states at 30 fps => 4× slow motion, about 8 seconds.
                canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
                canvas.paste(left_scene, (0, 103))
                canvas.paste(right_scene, (640, 103))
                draw = ImageDraw.Draw(canvas)
                _header(
                    draw,
                    "Jointed cable contact · coupled VBD owns the routing decision",
                    f"20 rad/s twist · actual Horde frame {state_index:03d}/{state_count - 1:03d} · 4× slow motion",
                )
                draw.text((26, 117), left_title, font=FONTS["panel"], fill=INK)
                draw.text((666, 117), right_title, font=FONTS["panel"], fill=INK)
                if state_index == 0:
                    left_pen = right_pen = 0.0
                    left_sub = right_sub = 0
                    left_missing = right_missing = 0
                else:
                    offset = state_index - 1
                    left_pen = max(0.0, -left["fresh_gap_m"][offset]) * 1000.0
                    right_pen = max(0.0, -right["fresh_gap_m"][offset]) * 1000.0
                    left_sub = left["selected_substeps"][offset]
                    right_sub = right["selected_substeps"][offset]
                    left_missing = left["missing_pairs"][offset]
                    right_missing = right["missing_pairs"][offset]
                draw.rounded_rectangle((18, 547, 622, 617), 10, fill=PANEL)
                draw.rounded_rectangle((658, 547, 1262, 617), 10, fill=PANEL)
                draw.text(
                    (35, 561),
                    f"fresh penetration {left_pen:.3f} mm · physical substeps {left_sub}\n"
                    f"fresh pairs absent from retained set {left_missing}",
                    font=FONTS["small"],
                    fill=GOOD if left_pen < 1.0 else WARN if left_pen < 5.0 else BAD,
                    spacing=6,
                )
                draw.text(
                    (675, 561),
                    f"fresh penetration {right_pen:.3f} mm · physical substeps {right_sub}\n"
                    f"fresh pairs absent from retained set {right_missing}",
                    font=FONTS["small"],
                    fill=GOOD if right_pen < 1.0 else WARN if right_pen < 5.0 else BAD,
                    spacing=6,
                )
                _footer(
                    draw,
                    "Simulation + Newton ViewerGL capture ran on Horde L40 · event impulses stay gated off for jointed islands",
                )
                yield np.asarray(canvas)

    _write_video(
        assets / filename,
        frames(),
        poster=assets / poster,
        poster_index=2 * state_count - 12,
    )


def render_cables(assets: Path, work_dir: Path, frames: int) -> dict:
    viewer = _viewer(640, 520)
    cases = {
        "unconditioned_10mm": _render_cable_case(
            viewer=viewer,
            work_dir=work_dir,
            key="unconditioned_10mm",
            frames=frames,
            target_surface_motion=10.0e-3,
            inertial_contact_ratio=0.0,
            twist_rate=20.0,
        ),
        "conditioned_10mm": _render_cable_case(
            viewer=viewer,
            work_dir=work_dir,
            key="conditioned_10mm",
            frames=frames,
            target_surface_motion=10.0e-3,
            inertial_contact_ratio=5.0,
            twist_rate=20.0,
        ),
        "conditioned_1mm": _render_cable_case(
            viewer=viewer,
            work_dir=work_dir,
            key="conditioned_1mm",
            frames=frames,
            target_surface_motion=1.0e-3,
            inertial_contact_ratio=5.0,
            twist_rate=20.0,
        ),
    }
    _compose_cable_video(
        assets=assets,
        work_dir=work_dir,
        left=cases["unconditioned_10mm"],
        right=cases["conditioned_10mm"],
        left_title="unconditioned contact · target 10 mm",
        right_title="inertially conditioned · ρ=5 · target 10 mm",
        filename="cable_conditioning_comparison.mp4",
        poster="cable_conditioning_poster.jpg",
    )
    _compose_cable_video(
        assets=assets,
        work_dir=work_dir,
        left=cases["conditioned_10mm"],
        right=cases["conditioned_1mm"],
        left_title="fast routing · target 10 mm",
        right_title="robust routing · target 1 mm",
        filename="cable_accuracy_comparison.mp4",
        poster="cable_accuracy_poster.jpg",
    )
    viewer.close()
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--only",
        choices=("impacts", "cables"),
        default="impacts",
    )
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    with (args.assets / "horde_results.json").open(encoding="utf-8") as source:
        results = json.load(source)
    traces = np.load(args.assets / "horde_traces.npz")

    with wp.ScopedDevice("cuda:0"):
        if args.only == "impacts":
            viewer = _viewer(WIDTH, 520)
            render_fast_impact(args.assets, traces, results, viewer)
            render_dense_chain(args.assets, traces, results, viewer)
            viewer.close()
        if args.only == "cables":
            cable_results = render_cables(args.assets, args.work_dir, args.frames)
            with (args.assets / "horde_render_results.json").open("w", encoding="utf-8") as output:
                json.dump(cable_results, output, indent=2, sort_keys=True)
                output.write("\n")


if __name__ == "__main__":
    main()
