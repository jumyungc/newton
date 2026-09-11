# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate VBD joint-friction validation data and plots."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerGL

REPORT_DIR = Path(__file__).resolve().parent
ASSET_DIR = REPORT_DIR / "assets"
DATA_DIR = REPORT_DIR / "data"
DT = 1.0 / 240.0
FRICTIONS = (0.0, 0.05, 0.2, 0.5, 1.5)
ROD_MASS = 1.0
ROD_LENGTH = 1.0
ROD_RADIUS = 0.04
I_COM = (1.0 / 12.0) * ROD_MASS * ROD_LENGTH * ROD_LENGTH
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540
VIDEO_FPS = 60
VIDEO_SECONDS = 5
VIDEO_SUBSTEPS = 10


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


def build_revolute(mu: float, qd0: float = 0.0, inertia: float = 1.0) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=wp.mat33(inertia, 0.0, 0.0, 0.0, inertia, 0.0, 0.0, 0.0, inertia),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        axis=2,
        parent=-1,
        child=body,
        target_ke=0.0,
        target_kd=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        armature=0.0,
        friction=mu,
    )
    builder.joint_qd[0] = qd0
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize()


def build_prismatic(mu: float, qd0: float = 0.0, mass: float = 1.0) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=mass,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_prismatic(
        axis=2,
        parent=-1,
        child=body,
        target_ke=0.0,
        target_kd=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        armature=0.0,
        friction=mu,
    )
    builder.joint_qd[0] = qd0
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize()


def build_static_stick(mu: float = 1.0, drive_torque: float = 0.5, drive_ke: float = 100.0) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        axis=2,
        parent=-1,
        child=body,
        target_ke=drive_ke,
        target_kd=0.1,
        target_pos=drive_torque / drive_ke,
        limit_ke=0.0,
        limit_kd=0.0,
        armature=0.0,
        friction=mu,
    )
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize()


def read_joint_state(model: newton.Model, state: newton.State) -> tuple[float, float]:
    joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=state.body_q.device)
    joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=state.body_q.device)
    newton.eval_ik(model, state, joint_q, joint_qd)
    return float(joint_q.numpy()[0]), float(joint_qd.numpy()[0])


def simulate(model: newton.Model, steps: int, control: newton.Control | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solver = newton.solvers.SolverVBD(model, iterations=12, rigid_avbd_beta=1.0e5)
    state_0 = model.state()
    state_1 = model.state()
    if control is None:
        control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    time = []
    q = []
    qd = []
    for step in range(steps):
        qi, qdi = read_joint_state(model, state_0)
        time.append(step * DT)
        q.append(qi)
        qd.append(qdi)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return np.asarray(time), np.asarray(q), np.asarray(qd)


def run_cases() -> dict[str, dict[str, np.ndarray | float]]:
    coast_steps = int(round(1.0 / DT))
    time_r, _q_r, qd_r = simulate(build_revolute(mu=1.0, qd0=2.0), coast_steps)
    time_p, _q_p, qd_p = simulate(build_prismatic(mu=1.0, qd0=2.0), coast_steps)

    slide_model = build_revolute(mu=0.5, qd0=0.0)
    slide_control = slide_model.control()
    slide_control.joint_f = wp.array([2.0], dtype=float)
    time_s, _q_s, qd_s = simulate(slide_model, int(round(0.6 / DT)), slide_control)

    time_stick, q_stick, qd_stick = simulate(build_static_stick(), int(round(0.6 / DT)))

    energy_time, _q_e, qd_e = simulate(build_revolute(mu=2.0, qd0=2.0), int(round(0.8 / DT)))
    energy = 0.5 * qd_e * qd_e

    return {
        "revolute_coast": {
            "time": time_r,
            "qd": qd_r,
            "reference_qd": 2.0 - time_r,
            "max_error": float(np.max(np.abs(qd_r - (2.0 - time_r)))),
        },
        "prismatic_coast": {
            "time": time_p,
            "qd": qd_p,
            "reference_qd": 2.0 - time_p,
            "max_error": float(np.max(np.abs(qd_p - (2.0 - time_p)))),
        },
        "sliding_drive": {
            "time": time_s,
            "qd": qd_s,
            "reference_qd": 1.5 * time_s,
            "max_error": float(np.max(np.abs(qd_s - 1.5 * time_s))),
        },
        "static_stick": {
            "time": time_stick,
            "q": q_stick,
            "qd": qd_stick,
            "max_abs_q": float(np.max(np.abs(q_stick))),
            "max_abs_qd": float(np.max(np.abs(qd_stick))),
        },
        "energy": {
            "time": energy_time,
            "energy": energy,
            "initial": float(energy[0]),
            "final": float(energy[-1]),
            "drop_fraction": float(1.0 - energy[-1] / energy[0]),
        },
    }


def save_data(cases: dict[str, dict[str, np.ndarray | float]]) -> dict[str, float]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in cases.items():
        np.savez(DATA_DIR / f"{name}.npz", **data)

    summary = {
        "revolute_coast_max_error_rad_s": float(cases["revolute_coast"]["max_error"]),
        "prismatic_coast_max_error_m_s": float(cases["prismatic_coast"]["max_error"]),
        "sliding_drive_max_error_rad_s": float(cases["sliding_drive"]["max_error"]),
        "static_stick_max_abs_q_rad": float(cases["static_stick"]["max_abs_q"]),
        "static_stick_max_abs_qd_rad_s": float(cases["static_stick"]["max_abs_qd"]),
        "energy_drop_fraction": float(cases["energy"]["drop_fraction"]),
    }
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def plot_cases(cases: dict[str, dict[str, np.ndarray | float]]) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), constrained_layout=True)

    for key, label, ax in (
        ("revolute_coast", "revolute", axes[0, 0]),
        ("prismatic_coast", "prismatic", axes[0, 1]),
    ):
        data = cases[key]
        ax.plot(data["time"], data["qd"], linewidth=2.4, label="VBD")
        ax.plot(data["time"], data["reference_qd"], "--", linewidth=2.2, label="analytic")
        ax.set_title(f"{label.capitalize()} coast-down")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("velocity")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best")

    data = cases["sliding_drive"]
    axes[1, 0].plot(data["time"], data["qd"], linewidth=2.4, label="VBD")
    axes[1, 0].plot(data["time"], data["reference_qd"], "--", linewidth=2.2, label="analytic")
    axes[1, 0].set_title("Super-threshold drive")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("angular velocity [rad/s]")
    axes[1, 0].grid(True, alpha=0.28)
    axes[1, 0].legend(loc="best")

    energy = cases["energy"]
    axes[1, 1].plot(energy["time"], energy["energy"], linewidth=2.4, color="#2f7d55")
    axes[1, 1].set_title("Friction dissipates kinetic energy")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("kinetic energy [J]")
    axes[1, 1].grid(True, alpha=0.28)

    fig.savefig(ASSET_DIR / "joint_friction_validation.png", dpi=180)
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


def build_video_scene(device: str = "cuda:0") -> newton.Model:
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
    colors = (
        (0.18, 0.45, 0.74),
        (0.28, 0.62, 0.38),
        (0.82, 0.55, 0.20),
        (0.76, 0.34, 0.22),
        (0.50, 0.38, 0.70),
    )
    marker_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)

    for i, mu in enumerate(FRICTIONS):
        x_offset = (i - (len(FRICTIONS) - 1) * 0.5) * 1.4
        builder.add_shape_sphere(
            -1,
            xform=wp.transform(wp.vec3(x_offset, 0.0, 2.0), wp.quat_identity()),
            radius=0.05,
            cfg=marker_cfg,
            color=(0.86, 0.88, 0.90),
        )

        link = builder.add_link(
            xform=wp.transform(wp.vec3(x_offset, 0.0, 2.0), wp.quat_identity()),
            mass=ROD_MASS,
            com=wp.vec3(ROD_LENGTH / 2.0, 0.0, 0.0),
            inertia=wp.mat33(I_COM, 0.0, 0.0, 0.0, I_COM, 0.0, 0.0, 0.0, I_COM),
            lock_inertia=True,
        )
        builder.add_shape_capsule(
            body=link,
            radius=ROD_RADIUS,
            half_height=ROD_LENGTH / 2.0,
            xform=wp.transform(wp.vec3(ROD_LENGTH / 2.0, 0.0, 0.0), wp.quat_identity()),
            cfg=marker_cfg,
            color=colors[i],
        )
        joint = builder.add_joint_revolute(
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent=-1,
            child=link,
            parent_xform=wp.transform(wp.vec3(x_offset, 0.0, 2.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            target_ke=0.0,
            target_kd=0.0,
            limit_ke=0.0,
            limit_kd=0.0,
            armature=0.0,
            friction=mu,
        )
        builder.add_articulation([joint], label=f"pendulum_mu_{mu:.2f}")

    builder.color()
    return builder.finalize(device=device)


def render_video() -> None:
    if not wp.is_cuda_available():
        print("Skipping joint-friction video: ViewerGL video rendering requires CUDA.")
        return
    if shutil.which("ffmpeg") is None:
        print("Skipping joint-friction video: ffmpeg was not found.")
        return

    device = "cuda:0"
    model = build_video_scene(device)
    solver = newton.solvers.SolverVBD(model, iterations=2)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    viewer = ViewerGL(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, headless=True)
    writer = _open_video_writer(ASSET_DIR / "joint_friction_pendulums.mp4", VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)
    if writer.stdin is None:
        raise RuntimeError("ffmpeg stdin pipe was not opened")

    frame_buffer = None
    frame_dt = 1.0 / VIDEO_FPS
    sim_dt = frame_dt / VIDEO_SUBSTEPS
    try:
        viewer.show_ui = False
        viewer.set_model(model)
        viewer.set_camera(wp.vec3(0.0, -7.0, 2.0), pitch=-5.0, yaw=90.0)
        viewer.camera.look_at(wp.vec3(0.0, 0.0, 1.8))

        for frame in range(VIDEO_FPS * VIDEO_SECONDS):
            for _substep in range(VIDEO_SUBSTEPS):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, sim_dt)
                state_0, state_1 = state_1, state_0

            viewer.begin_frame((frame + 1) * frame_dt)
            viewer.log_state(state_0)
            viewer.end_frame()
            frame_buffer = viewer.get_frame(frame_buffer, render_ui=False)
            writer.stdin.write(np.ascontiguousarray(frame_buffer.numpy()).tobytes())
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
        viewer.close()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")


def write_index(summary: dict[str, float]) -> None:
    html = f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>VBD Joint Friction Validation</title>
    <style>
      :root {{
        color-scheme: light;
        --ink: #17202a;
        --muted: #607086;
        --line: #d8e0e8;
        --panel: #f6f8fb;
        --accent: #1f6f8b;
        --accent-2: #6f8f2f;
        --max: 1180px;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: #ffffff;
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
        line-height: 1.5;
      }}
      a {{ color: var(--accent); }}
      code {{
        padding: 0.08rem 0.28rem;
        border: 1px solid #dce6ef;
        border-radius: 4px;
        background: #eef3f7;
        font-size: 0.93em;
      }}
      header {{ border-bottom: 1px solid var(--line); background: #fbfcfd; }}
      main, .header-inner {{ width: min(var(--max), calc(100vw - 32px)); margin: 0 auto; }}
      .header-inner {{ padding: 38px 0 30px; }}
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
        color: var(--accent);
        font-weight: 650;
        text-decoration: none;
      }}
      .eyebrow {{
        margin: 0 0 10px;
        color: var(--accent);
        font-size: 0.82rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}
      h1, h2 {{ margin: 0; line-height: 1.12; letter-spacing: 0; }}
      h1 {{ max-width: 900px; font-size: clamp(2.2rem, 4vw, 3.6rem); }}
      h2 {{ margin-bottom: 14px; font-size: 1.55rem; }}
      p {{ margin: 0; }}
      .subtitle {{ max-width: 920px; margin-top: 14px; color: var(--muted); font-size: 1.08rem; }}
      main {{ padding: 34px 0 48px; }}
      section {{ margin-bottom: 38px; }}
      .summary {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-top: 24px;
      }}
      .card, .note, .figure, .example-table {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #ffffff;
      }}
      .card {{ padding: 14px 16px; }}
      .card .label {{ display: block; color: var(--muted); font-size: 0.78rem; font-weight: 750; text-transform: uppercase; }}
      .card .value {{ display: block; margin-top: 5px; font-size: 1.16rem; font-weight: 760; }}
      .figure {{ overflow: hidden; }}
      .figure img, .figure video {{ display: block; width: 100%; background: var(--panel); }}
      .note {{ padding: 16px 18px; background: var(--panel); color: var(--muted); }}
      table {{ width: 100%; border-collapse: collapse; }}
      th, td {{ padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
      th {{ background: var(--panel); color: var(--ink); }}
      tr:last-child td {{ border-bottom: 0; }}
      @media (max-width: 880px) {{ .summary {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <header>
      <div class=\"header-inner\">
        <nav class=\"report-nav\" aria-label=\"report navigation\">
          <a class=\"button\" href=\"../vbd_validation/index.html\">VBD validation overview</a>
        </nav>
        <p class=\"eyebrow\">Newton VBD</p>
        <h1>Joint friction validation</h1>
        <p class=\"subtitle\">
          Validates per-DOF Coulomb joint friction in SolverVBD with revolute and prismatic coast-down,
          subthreshold stick, super-threshold sliding, and energy dissipation checks.
        </p>
        <div class=\"summary\" aria-label=\"summary\">
          <div class=\"card\"><span class=\"label\">Revolute coast error</span><span class=\"value\">{summary["revolute_coast_max_error_rad_s"]:.3f} rad/s</span></div>
          <div class=\"card\"><span class=\"label\">Prismatic coast error</span><span class=\"value\">{summary["prismatic_coast_max_error_m_s"]:.3f} m/s</span></div>
          <div class=\"card\"><span class=\"label\">Sliding drive error</span><span class=\"value\">{summary["sliding_drive_max_error_rad_s"]:.3f} rad/s</span></div>
          <div class=\"card\"><span class=\"label\">Energy drop</span><span class=\"value\">{100.0 * summary["energy_drop_fraction"]:.1f}%</span></div>
        </div>
      </div>
    </header>
    <main>
      <section>
        <h2>Newton Rendered Video</h2>
        <div class=\"figure\">
          <video controls muted loop playsinline preload=\"metadata\">
            <source src=\"assets/joint_friction_pendulums.mp4\" type=\"video/mp4\" />
          </video>
        </div>
      </section>
      <section>
        <h2>Validation Plots</h2>
        <div class=\"figure\">
          <img src=\"assets/joint_friction_validation.png\" alt=\"Joint friction validation plots\" />
        </div>
      </section>
      <section>
        <h2>Standalone Example</h2>
        <div class=\"example-table\">
          <table>
            <thead><tr><th>Feature</th><th>Run Command</th><th>What To Check</th></tr></thead>
            <tbody>
              <tr>
                <td><code>Model.joint_friction</code></td>
                <td><code>uv run -m newton.examples basic_joint_friction</code></td>
                <td>Five pendulums with different friction torques are compared against an RK4 reference.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>
  </body>
</html>
"""
    (REPORT_DIR / "index.html").write_text(html)


def main() -> None:
    cases = run_cases()
    summary = save_data(cases)
    plot_cases(cases)
    render_video()
    write_index(summary)
    print(f"Wrote {REPORT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
