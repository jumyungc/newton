# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate VBD per-joint reaction readout validation data and plots."""

from __future__ import annotations

import json
import multiprocessing
import shutil
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.cable.example_vbd_joint_reaction_closed_cable import (
    PULL_FORCE,
    Example as ClosedCableExample,
)
from newton.examples.cable.example_vbd_joint_reaction_four_bar import (
    DRIVE_TORQUE,
    Example as FourBarExample,
    fourbar_geometry,
    link_pose,
)
from newton.viewer import ViewerGL

REPORT_DIR = Path(__file__).resolve().parent
ASSET_DIR = REPORT_DIR / "assets"
DATA_DIR = REPORT_DIR / "data"
STEPS = 180
DT = 1.0 / 240.0
FRAMES = 120
FRAME_DT = 1.0 / 60.0
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540
VIDEO_FPS = 60
VIDEO_SECONDS = 4


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "font.size": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    return plt


def build_fourbar_model(device: str = "cpu"):
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f", "vbd:joint_reaction_f")
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 1000.0

    (point_a, point_b, point_c, point_d), (a_link, b_link, c_link) = fourbar_geometry()
    crank = builder.add_link(xform=link_pose(point_a, point_b))
    coupler = builder.add_link(xform=link_pose(point_b, point_c))
    rocker = builder.add_link(xform=link_pose(point_d, point_c))
    builder.add_shape_box(crank, hx=0.5 * a_link, hy=0.02, hz=0.02, cfg=cfg)
    builder.add_shape_box(coupler, hx=0.5 * b_link, hy=0.02, hz=0.02, cfg=cfg)
    builder.add_shape_box(rocker, hx=0.5 * c_link, hy=0.02, hz=0.02, cfg=cfg)

    j_crank = builder.add_joint_revolute(
        parent=-1,
        child=crank,
        parent_xform=wp.transform(wp.vec3(*point_a), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * a_link, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        label="ground_crank",
    )
    j_coupler = builder.add_joint_revolute(
        parent=crank,
        child=coupler,
        parent_xform=wp.transform(wp.vec3(0.5 * a_link, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * b_link, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        label="crank_coupler",
    )
    j_rocker = builder.add_joint_revolute(
        parent=coupler,
        child=rocker,
        parent_xform=wp.transform(wp.vec3(0.5 * b_link, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5 * c_link, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        label="coupler_rocker",
    )
    j_loop = builder.add_joint_revolute(
        parent=-1,
        child=rocker,
        parent_xform=wp.transform(wp.vec3(*point_d), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * c_link, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        label="loop_ground_rocker",
    )
    builder.add_articulation([j_crank, j_coupler, j_rocker])
    builder.joint_articulation[j_loop] = -1
    builder.color()

    model = builder.finalize(device=device)
    return model, {"ground_crank": j_crank, "coupler_rocker": j_rocker, "loop_ground_rocker": j_loop}, rocker


def simulate(device: str = "cpu") -> dict[str, np.ndarray | float]:
    model, joints, rocker = build_fourbar_model(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=32)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_f[int(model.joint_qd_start.numpy()[joints["ground_crank"]])] = DRIVE_TORQUE
    control.joint_f.assign(joint_f)

    time = []
    tree_force = []
    loop_force = []
    body_force = []
    summed_force = []
    sum_error = []

    for step in range(STEPS):
        solver.step(state_0, state_1, control, None, DT)
        joint_reaction = state_1.vbd.joint_reaction_f.numpy()
        body_parent_f = state_1.body_parent_f.numpy()
        joint_child = model.joint_child.numpy()

        summed = np.zeros_like(body_parent_f)
        for joint_index, child in enumerate(joint_child):
            if child >= 0:
                summed[child] += joint_reaction[joint_index]

        time.append((step + 1) * DT)
        tree_force.append(float(np.linalg.norm(joint_reaction[joints["coupler_rocker"], :3])))
        loop_force.append(float(np.linalg.norm(joint_reaction[joints["loop_ground_rocker"], :3])))
        body_force.append(float(np.linalg.norm(body_parent_f[rocker, :3])))
        summed_force.append(float(np.linalg.norm(summed[rocker, :3])))
        sum_error.append(float(np.max(np.linalg.norm(body_parent_f - summed, axis=1))))
        state_0, state_1 = state_1, state_0

    return {
        "time": np.asarray(time),
        "tree_force": np.asarray(tree_force),
        "loop_force": np.asarray(loop_force),
        "body_force": np.asarray(body_force),
        "summed_force": np.asarray(summed_force),
        "sum_error": np.asarray(sum_error),
        "max_sum_error": float(np.max(sum_error)),
        "final_tree_force": float(tree_force[-1]),
        "final_loop_force": float(loop_force[-1]),
        "final_body_force": float(body_force[-1]),
    }


class _NullViewer:
    def set_model(self, model):
        pass


class _Args:
    pass


def simulate_closed_cable() -> dict[str, np.ndarray | float]:
    with wp.ScopedDevice("cpu"):
        example = ClosedCableExample(_NullViewer(), _Args())

        time = []
        mean_reaction = []
        max_reaction = []
        loop_reaction = []
        loaded_body_reaction = []
        mean_tension = []
        max_tension = []
        sum_error = []

        for step in range(FRAMES):
            example.simulate()
            time.append((step + 1) * FRAME_DT)
            mean_reaction.append(example.mean_reaction)
            max_reaction.append(example.max_reaction)
            loop_reaction.append(example.loop_reaction)
            loaded_body_reaction.append(example.loaded_body_reaction)
            mean_tension.append(example.mean_tension)
            max_tension.append(example.max_tension)
            sum_error.append(example.sum_error)

    return {
        "time": np.asarray(time),
        "mean_reaction": np.asarray(mean_reaction),
        "max_reaction": np.asarray(max_reaction),
        "loop_reaction": np.asarray(loop_reaction),
        "loaded_body_reaction": np.asarray(loaded_body_reaction),
        "mean_tension": np.asarray(mean_tension),
        "max_tension": np.asarray(max_tension),
        "sum_error": np.asarray(sum_error),
        "max_sum_error": float(np.max(sum_error)),
        "final_mean_reaction": float(mean_reaction[-1]),
        "final_max_reaction": float(max_reaction[-1]),
        "final_loop_reaction": float(loop_reaction[-1]),
        "final_loaded_body_reaction": float(loaded_body_reaction[-1]),
        "final_mean_tension": float(mean_tension[-1]),
        "final_max_tension": float(max_tension[-1]),
    }


def save_data(data: dict[str, np.ndarray | float], closed_cable: dict[str, np.ndarray | float]) -> dict[str, float]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / "fourbar_joint_reaction.npz", **data)
    np.savez(DATA_DIR / "closed_cable_joint_reaction.npz", **closed_cable)

    summary = {
        "max_sum_error_n": float(data["max_sum_error"]),
        "final_tree_force_n": float(data["final_tree_force"]),
        "final_loop_force_n": float(data["final_loop_force"]),
        "final_body_force_n": float(data["final_body_force"]),
        "closed_cable_max_sum_error_n": float(closed_cable["max_sum_error"]),
        "closed_cable_final_mean_reaction_n": float(closed_cable["final_mean_reaction"]),
        "closed_cable_final_max_reaction_n": float(closed_cable["final_max_reaction"]),
        "closed_cable_final_loop_reaction_n": float(closed_cable["final_loop_reaction"]),
        "closed_cable_final_loaded_body_reaction_n": float(closed_cable["final_loaded_body_reaction"]),
        "closed_cable_final_mean_tension_n": float(closed_cable["final_mean_tension"]),
        "closed_cable_final_max_tension_n": float(closed_cable["final_max_tension"]),
        "closed_cable_mean_reaction_tension_error_n": abs(
            float(closed_cable["final_mean_reaction"]) - float(closed_cable["final_mean_tension"])
        ),
        "closed_cable_applied_pull_n": PULL_FORCE,
    }
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def plot_data(data: dict[str, np.ndarray | float], closed_cable: dict[str, np.ndarray | float]) -> None:
    plt = _get_pyplot()
    colors = {
        "joint_a": "#2563a6",
        "joint_b": "#7c3aed",
        "body_net": "#2f7d55",
        "tension": "#b45309",
        "check": "#64748b",
    }

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True)
    axes[0].plot(
        data["time"],
        data["tree_force"],
        linewidth=2.4,
        color=colors["joint_a"],
        label="orange-green internal joint reaction",
    )
    axes[0].plot(
        data["time"],
        data["loop_force"],
        linewidth=2.4,
        color=colors["joint_b"],
        label="gray-green closure joint reaction",
    )
    axes[0].set_title("Closed four-bar: individual joint-reaction magnitudes")
    axes[0].set_ylabel("per-joint force magnitude [N]")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend()

    axes[1].plot(
        data["time"],
        data["body_force"],
        linewidth=2.4,
        color=colors["body_net"],
        label="green link net load",
    )
    axes[1].plot(
        data["time"],
        data["summed_force"],
        "--",
        linewidth=2.2,
        color=colors["check"],
        label="vector sum of the two green-link joint reactions",
    )
    axes[1].set_title(
        f"Consistency check: green net load matches vector-summed joint reactions, max error {data['max_sum_error']:.1e} N"
    )
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("net force magnitude [N]")
    axes[1].grid(True, alpha=0.28)
    axes[1].legend()
    fig.savefig(ASSET_DIR / "fourbar_joint_reactions.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True)
    axes[0].plot(
        closed_cable["time"],
        closed_cable["mean_reaction"],
        linewidth=2.4,
        color=colors["joint_a"],
        label="mean per-joint cable reaction",
    )
    axes[0].plot(
        closed_cable["time"],
        closed_cable["mean_tension"],
        "--",
        linewidth=2.2,
        color=colors["tension"],
        label="mean cable tension",
    )
    axes[0].set_title("Closed cable: reaction readout matches cable tension")
    axes[0].set_ylabel("magnitude [N]")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend()

    axes[1].plot(
        closed_cable["time"],
        closed_cable["loop_reaction"],
        linewidth=2.4,
        color=colors["joint_b"],
        label="purple closure-joint reaction",
    )
    axes[1].plot(
        closed_cable["time"],
        closed_cable["loaded_body_reaction"],
        linewidth=2.4,
        color=colors["body_net"],
        label="max red-segment net joint load",
    )
    axes[1].set_title(f"Closed-loop load checks: max body-sum error {closed_cable['max_sum_error']:.1e} N")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("magnitude [N]")
    axes[1].grid(True, alpha=0.28)
    axes[1].legend()
    fig.savefig(ASSET_DIR / "closed_cable_joint_reactions.png", dpi=170)
    plt.close(fig)

def _open_video_writer(path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
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
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _render_example_video(example_cls, output_path: Path) -> None:
    frame_count = VIDEO_FPS * VIDEO_SECONDS
    with wp.ScopedDevice("cuda:0"):
        viewer = ViewerGL(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, headless=True, plot_history_size=frame_count)
        writer = _open_video_writer(output_path, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)
        if writer.stdin is None:
            raise RuntimeError("ffmpeg stdin pipe was not opened")

        frame_buffer = None
        try:
            viewer.show_ui = False
            example = example_cls(viewer, _Args())
            for _frame in range(frame_count):
                example.step()
                example.render()
                frame_buffer = viewer.get_frame(frame_buffer, render_ui=False)
                writer.stdin.write(np.ascontiguousarray(frame_buffer.numpy()).tobytes())
        finally:
            if writer.stdin is not None:
                writer.stdin.close()
            return_code = writer.wait()
            viewer.close()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg exited with status {return_code}")


def _render_video_worker(kind: str) -> None:
    if kind == "four_bar":
        _render_example_video(FourBarExample, ASSET_DIR / "vbd_joint_reaction_four_bar.mp4")
    elif kind == "closed_cable":
        _render_example_video(ClosedCableExample, ASSET_DIR / "vbd_joint_reaction_closed_cable.mp4")
    else:
        raise ValueError(f"Unknown video kind: {kind}")


def render_videos() -> None:
    if not wp.is_cuda_available():
        print("Skipping videos: ViewerGL video rendering requires CUDA.")
        return
    if shutil.which("ffmpeg") is None:
        print("Skipping videos: ffmpeg was not found.")
        return

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    for kind in ("four_bar", "closed_cable"):
        process = ctx.Process(target=_render_video_worker, args=(kind,))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{kind} video rendering failed with exit code {process.exitcode}.")


def write_html(summary: dict[str, float]) -> None:
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>VBD Joint Reaction Readout</title>
    <style>
      :root {{
        color-scheme: light;
        --ink: #17202a;
        --muted: #607086;
        --line: #d8e0e8;
        --panel: #f7f9fc;
        --panel-strong: #eef3f7;
        --blue: #2563a6;
        --purple: #7c3aed;
        --green: #2f7d55;
        --orange: #b45309;
        --max: 1080px;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: #ffffff;
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1.5;
      }}
      a {{ color: var(--blue); }}
      main, .header-inner {{
        width: min(var(--max), calc(100vw - 32px));
        margin: 0 auto;
      }}
      header {{
        border-bottom: 1px solid var(--line);
        background: #fbfcfd;
      }}
      .header-inner {{ padding: 38px 0 30px; }}
      h1, h2, h3 {{ margin: 0; line-height: 1.12; letter-spacing: 0; }}
      h1 {{ max-width: 880px; font-size: clamp(2.1rem, 4vw, 3.6rem); }}
      h2 {{ margin-bottom: 12px; font-size: 1.58rem; }}
      h3 {{ font-size: 1.08rem; }}
      p {{ margin: 0; }}
      code {{
        padding: 0.08rem 0.28rem;
        border: 1px solid #dce6ef;
        border-radius: 4px;
        background: #eef3f7;
        font-size: 0.93em;
      }}
      .eyebrow {{
        margin-bottom: 10px;
        color: var(--blue);
        font-size: 0.82rem;
        font-weight: 760;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}
      .report-nav {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 18px;
      }}
      .button {{
        display: inline-flex;
        align-items: center;
        min-height: 34px;
        padding: 7px 12px;
        border: 1px solid #bfd2e7;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.78);
        color: var(--blue);
        font-weight: 650;
        text-decoration: none;
      }}
      .subtitle {{
        max-width: 940px;
        margin-top: 14px;
        color: var(--muted);
        font-size: 1.06rem;
      }}
      .card .subtitle {{
        margin-top: 8px;
        font-size: 0.92rem;
      }}
      main {{ padding: 34px 0 48px; }}
      section {{ margin-bottom: 38px; }}
      .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
      .media-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
      .cards {{ margin-top: 18px; }}
      .card, .note, .command-table {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
      }}
      .card {{ padding: 14px 16px; }}
      .label {{
        display: block;
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 760;
        text-transform: uppercase;
      }}
      .value {{
        display: block;
        margin-top: 6px;
        font-size: 1.25rem;
        font-weight: 760;
      }}
      .note {{
        margin-top: 14px;
        padding: 14px 16px;
        color: var(--muted);
      }}
      .figure {{
        margin-top: 16px;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #ffffff;
      }}
      .figure img, .figure video {{
        display: block;
        width: 100%;
        background: #ffffff;
      }}
      .figure-body {{ padding: 14px 16px; }}
      .figure-body p {{ margin-top: 6px; color: var(--muted); }}
      .legend {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
      }}
      .tag {{
        display: inline-flex;
        align-items: center;
        min-height: 26px;
        padding: 4px 8px;
        border-radius: 6px;
        background: var(--panel-strong);
        color: #334155;
        font-size: 0.82rem;
        font-weight: 650;
      }}
      .blue {{ color: var(--blue); }}
      .purple {{ color: var(--purple); }}
      .green {{ color: var(--green); }}
      .orange {{ color: var(--orange); }}
      table {{ width: 100%; border-collapse: collapse; }}
      th, td {{ padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
      th {{ background: var(--panel); }}
      tr:last-child td {{ border-bottom: 0; }}
      @media (max-width: 820px) {{
        .grid {{ grid-template-columns: 1fr 1fr; }}
        .media-grid {{ grid-template-columns: 1fr; }}
      }}
      @media (max-width: 560px) {{
        .grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <header>
      <div class="header-inner">
        <nav class="report-nav" aria-label="report navigation">
          <a class="button" href="../vbd_validation/index.html">VBD validation overview</a>
        </nav>
        <p class="eyebrow">Newton VBD validation</p>
        <h1>Per-joint reaction readout in closed systems</h1>
        <p class="subtitle">A closed four-bar linkage where two joints act on the green right link. The report checks that
        <code>state.vbd.joint_reaction_f</code> keeps those joint loads separated while their sum matches
        <code>state.body_parent_f</code>. A second scene applies equal and opposite pulls to two highlighted segments of a
        standalone closed cable loop. The purple marker is the loop-closing joint; the plot compares
        per-joint reactions against the cable-specific
        <code>state.vbd.cable_tension</code> readout.</p>
        <div class="grid cards">
          <div class="card"><span class="label">Four-bar sum error</span><span class="value">{summary["max_sum_error_n"]:.2e} N</span></div>
          <div class="card"><span class="label">Cable sum error</span><span class="value">{summary["closed_cable_max_sum_error_n"]:.2e} N</span></div>
          <div class="card"><span class="label">Reaction vs tension</span><span class="value">{summary["closed_cable_mean_reaction_tension_error_n"]:.2e} N</span></div>
          <div class="card"><span class="label">Applied cable pull</span><span class="value">{summary["closed_cable_applied_pull_n"]:.3f} N</span></div>
        </div>
      </div>
    </header>
    <main>
      <section>
        <h2>Readout Contract</h2>
        <div class="grid">
          <div class="card">
            <span class="label">Per-joint</span>
            <span class="value">joint_reaction_f</span>
            <p class="subtitle">One reaction wrench per joint, including loop-closing joints that cannot be represented by a tree-only parent chain.</p>
          </div>
          <div class="card">
            <span class="label">Body-level</span>
            <span class="value">net joint load</span>
            <p class="subtitle">A body-indexed readout stored in <code>state.body_parent_f</code>. In closed loops it is the merged load on the body, not a substitute for per-joint reporting.</p>
          </div>
          <div class="card">
            <span class="label">Cable check</span>
            <span class="value">cable_tension</span>
            <p class="subtitle">A scalar cable-specific tension readout used here as an independent check against the axial part of the per-joint reaction.</p>
          </div>
          <div class="card">
            <span class="label">Reference frame</span>
            <span class="value">world force</span>
            <p class="subtitle">The plots show force magnitudes from the linear component of the reported world-frame reaction wrenches.</p>
          </div>
        </div>
      </section>

      <section>
        <h2>Newton Rendered Videos</h2>
        <div class="media-grid">
          <article class="figure">
            <video controls muted loop playsinline preload="metadata">
              <source src="assets/vbd_joint_reaction_four_bar.mp4" type="video/mp4" />
            </video>
            <div class="figure-body">
              <h3>Closed four-bar</h3>
              <p>Blue, orange, and green links form a closed linkage. The clean plots below show the two separate joint-reaction magnitudes and the green-link vector-sum check.</p>
            </div>
          </article>
          <article class="figure">
            <video controls muted loop playsinline preload="metadata">
              <source src="assets/vbd_joint_reaction_closed_cable.mp4" type="video/mp4" />
            </video>
            <div class="figure-body">
              <h3>Closed cable loop</h3>
              <p>Two red cable segments are pulled in opposite directions. The purple marker identifies the loop-closing cable joint used by the closure-joint reaction plot.</p>
            </div>
          </article>
        </div>
      </section>

      <section>
        <h2>Closed Four-Bar</h2>
        <div class="grid cards">
          <div class="card"><span class="label">Orange-green joint</span><span class="value">{summary["final_tree_force_n"]:.3f} N</span></div>
          <div class="card"><span class="label">Gray-green closure</span><span class="value">{summary["final_loop_force_n"]:.3f} N</span></div>
          <div class="card"><span class="label">Green body net</span><span class="value">{summary["final_body_force_n"]:.3f} N</span></div>
          <div class="card"><span class="label">Sum check</span><span class="value">{summary["max_sum_error_n"]:.2e} N</span></div>
        </div>
        <p class="note">The upper plot is a separation view: two different joints act on the green link, and the plotted values are individual force magnitudes. They are not meant to add as scalars. The lower plot is the consistency check: the green link net load, read from <code>state.body_parent_f</code>, lands on top of the vector sum of the two green-link per-joint reactions.</p>
        <div class="legend">
          <span class="tag"><span class="blue">blue</span>&nbsp;internal joint reaction</span>
          <span class="tag"><span class="purple">purple</span>&nbsp;closure joint reaction</span>
          <span class="tag"><span class="green">green</span>&nbsp;body-level net load</span>
        </div>
        <div class="figure">
          <img src="assets/fourbar_joint_reactions.png" alt="Four-bar joint reaction force plot" />
        </div>
      </section>
      <section>
        <h2>Closed Cable Loop</h2>
        <div class="grid cards">
          <div class="card"><span class="label">Mean per-joint reaction</span><span class="value">{summary["closed_cable_final_mean_reaction_n"]:.3f} N</span></div>
          <div class="card"><span class="label">Mean cable tension</span><span class="value">{summary["closed_cable_final_mean_tension_n"]:.3f} N</span></div>
          <div class="card"><span class="label">Purple closure joint</span><span class="value">{summary["closed_cable_final_loop_reaction_n"]:.3f} N</span></div>
          <div class="card"><span class="label">Max red body load</span><span class="value">{summary["closed_cable_final_loaded_body_reaction_n"]:.3f} N</span></div>
        </div>
        <p class="note">The cable loop is pulled outward at the two red segments with equal and opposite forces, so the net external force on the whole loop is zero. The purple marker identifies the loop-closing joint. The mean per-joint reaction and mean cable tension agree to {summary["closed_cable_mean_reaction_tension_error_n"]:.2e} N; the max red-body load is the larger net joint-load magnitude over the two loaded red segments, read from <code>state.body_parent_f</code>.</p>
        <div class="legend">
          <span class="tag"><span class="blue">blue</span>&nbsp;mean per-joint reaction</span>
          <span class="tag"><span class="orange">orange</span>&nbsp;mean cable tension</span>
          <span class="tag"><span class="purple">purple</span>&nbsp;marked closure joint</span>
          <span class="tag"><span class="green">green</span>&nbsp;max red-segment body load</span>
        </div>
        <div class="figure">
          <img src="assets/closed_cable_joint_reactions.png" alt="Closed cable joint reaction and tension plot" />
        </div>
      </section>
      <section>
        <h2>Reproduce</h2>
        <div class="command-table">
          <table>
            <thead>
              <tr><th>Scene</th><th>Command</th><th>What to inspect</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>Closed four-bar</td>
                <td><code>uv run -m newton.examples vbd_joint_reaction_four_bar</code></td>
                <td>Orange-green and gray-green reactions stay separated; the green link net load is their body-level sum.</td>
              </tr>
              <tr>
                <td>Closed cable loop</td>
                <td><code>uv run -m newton.examples vbd_joint_reaction_closed_cable</code></td>
                <td>Red loaded segments, purple closure joint, reaction-tension agreement, and max red-segment net joint load.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>
  </body>
</html>
"""
    (REPORT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    data = simulate()
    closed_cable = simulate_closed_cable()
    summary = save_data(data, closed_cable)
    plot_data(data, closed_cable)
    render_videos()
    write_html(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
