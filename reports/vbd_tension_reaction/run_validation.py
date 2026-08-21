# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate VBD joint reaction validation data and plots."""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerGL

REPORT_DIR = Path(__file__).resolve().parent
ASSET_DIR = REPORT_DIR / "assets"
DATA_DIR = REPORT_DIR / "data"
BASE_ITERATIONS = 4
BASE_FRAME_DT = 1.0 / 60.0
BASE_SUBSTEPS = 4
BASE_RIGID_JOINT_LINEAR_KE = 1.0e4
BASE_RIGID_JOINT_ANGULAR_KE = 1.0e5
REPORT_SECONDS = 4
ITERATION_SWEEP = (2, 4, 8, 16)
SUBSTEP_SWEEP = (1, 2, 4, 8)
VIDEO_HEIGHT = 540
VIDEO_SCENE_WIDTH = 960
VIDEO_PLOT_WIDTH = 640
VIDEO_FPS = 60
VIDEO_SECONDS = REPORT_SECONDS
VIDEO_SUBSTEPS = BASE_SUBSTEPS


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


def _quat_y(theta: float) -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta)


def _rot_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _transform_rotation_matrix(transform_row: np.ndarray) -> np.ndarray:
    quat = wp.quat(*transform_row[3:7].tolist())
    return np.array(wp.quat_to_matrix(quat), dtype=float).reshape(3, 3)


def _make_solver(model: newton.Model, iterations: int, hard_joints: bool) -> newton.solvers.SolverVBD:
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_joint_linear_ke=BASE_RIGID_JOINT_LINEAR_KE,
        rigid_joint_angular_ke=BASE_RIGID_JOINT_ANGULAR_KE,
    )
    if not hard_joints:
        for joint_index in range(model.joint_count):
            solver.set_joint_constraint_mode(joint_index, hard=False)
    return solver


def _configure_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)


@lru_cache
def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def build_single_pendulum(device: str, theta: float = 0.55, length: float = 1.0) -> tuple[newton.Model, int]:
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")

    q = _quat_y(theta)
    com = _rot_y(theta) @ np.array([0.0, 0.0, -length])
    link = builder.add_link(xform=wp.transform(wp.vec3(*com), q))
    builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, length), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device), link


def build_double_pendulum(
    device: str,
    theta_0: float = 0.45,
    theta_1: float = -0.35,
    length: float = 1.0,
) -> tuple[newton.Model, tuple[int, int]]:
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")

    half = 0.5 * length
    q0 = _quat_y(theta_0)
    q1 = _quat_y(theta_1)
    r0 = _rot_y(theta_0)
    r1 = _rot_y(theta_1)

    com0 = r0 @ np.array([0.0, 0.0, -half])
    pivot1 = r0 @ np.array([0.0, 0.0, -length])
    com1 = pivot1 + r1 @ np.array([0.0, 0.0, -half])

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 1000.0

    link0 = builder.add_link(xform=wp.transform(wp.vec3(*com0), q0))
    builder.add_shape_box(link0, hx=0.04, hy=0.04, hz=0.04, cfg=cfg)
    link1 = builder.add_link(xform=wp.transform(wp.vec3(*com1), q1))
    builder.add_shape_box(link1, hx=0.04, hy=0.04, hz=0.04, cfg=cfg)

    joint0 = builder.add_joint_revolute(
        parent=-1,
        child=link0,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, half), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    joint1 = builder.add_joint_revolute(
        parent=link0,
        child=link1,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -half), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, half), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    builder.add_articulation([joint0, joint1])
    builder.color()
    return builder.finalize(device=device), (link0, link1)


def build_single_pendulum_video_scene(
    device: str, theta: float = 0.55, length: float = 1.0
) -> tuple[newton.Model, int]:
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")
    q = _quat_y(theta)
    half = 0.5 * length
    com = _rot_y(theta) @ np.array([0.0, 0.0, -half])

    cfg = newton.ModelBuilder.ShapeConfig()
    marker_cfg = newton.ModelBuilder.ShapeConfig()
    marker_cfg.collision_group = 0
    builder.add_shape_sphere(
        -1,
        xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
        radius=0.055,
        cfg=marker_cfg,
        color=(0.9, 0.9, 0.9),
    )
    link = builder.add_link(xform=wp.transform(wp.vec3(*com), q))
    builder.add_shape_box(link, hx=0.04, hy=0.04, hz=half, cfg=cfg, color=(0.13, 0.47, 0.74))
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, half), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device), link


def build_double_pendulum_video_scene(
    device: str,
    theta_0: float = 0.45,
    theta_1: float = -0.35,
    length: float = 1.0,
) -> tuple[newton.Model, tuple[int, int]]:
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")
    half = 0.5 * length
    q0 = _quat_y(theta_0)
    q1 = _quat_y(theta_1)
    r0 = _rot_y(theta_0)
    r1 = _rot_y(theta_1)

    com0 = r0 @ np.array([0.0, 0.0, -half])
    pivot1 = r0 @ np.array([0.0, 0.0, -length])
    com1 = pivot1 + r1 @ np.array([0.0, 0.0, -half])

    cfg0 = newton.ModelBuilder.ShapeConfig()
    cfg1 = newton.ModelBuilder.ShapeConfig()
    marker_cfg = newton.ModelBuilder.ShapeConfig()
    marker_cfg.collision_group = 0
    builder.add_shape_sphere(
        -1,
        xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
        radius=0.055,
        cfg=marker_cfg,
        color=(0.9, 0.9, 0.9),
    )
    link0 = builder.add_link(xform=wp.transform(wp.vec3(*com0), q0))
    builder.add_shape_box(link0, hx=0.04, hy=0.04, hz=half, cfg=cfg0, color=(0.13, 0.47, 0.74))
    link1 = builder.add_link(xform=wp.transform(wp.vec3(*com1), q1))
    builder.add_shape_box(link1, hx=0.04, hy=0.04, hz=half, cfg=cfg1, color=(0.78, 0.31, 0.16))

    joint0 = builder.add_joint_revolute(
        parent=-1,
        child=link0,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, half), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    joint1 = builder.add_joint_revolute(
        parent=link0,
        child=link1,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -half), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, half), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    builder.add_articulation([joint0, joint1])
    builder.color()
    return builder.finalize(device=device), (link0, link1)


def simulate_single(
    device: str = "cpu",
    seconds: int = REPORT_SECONDS,
    frame_dt: float = BASE_FRAME_DT,
    substeps: int = BASE_SUBSTEPS,
    iterations: int = BASE_ITERATIONS,
    hard_joints: bool = True,
) -> dict[str, np.ndarray]:
    length = 1.0
    model, link = build_single_pendulum(device=device, length=length)
    solver = _make_solver(model, iterations, hard_joints)
    state0 = model.state()
    state1 = model.state()
    control = model.control()

    mass = float(model.body_mass.numpy()[link])
    gravity = model.gravity.numpy()[0].astype(float)
    arm_local = np.array([0.0, 0.0, length], dtype=float)

    time = []
    force_vbd = []
    force_balance = []
    torque_vbd = []
    torque_balance = []
    dt = frame_dt / substeps
    steps = int(round(seconds / frame_dt)) * substeps

    for step in range(steps):
        vel_before = state0.body_qd.numpy()[link, :3].astype(float)
        solver.step(state0, state1, control, None, dt)
        body_q = state1.body_q.numpy()[link]
        vel_after = state1.body_qd.numpy()[link, :3].astype(float)
        acceleration = (vel_after - vel_before) / dt

        reported = state1.body_parent_f.numpy()[link].astype(float)
        expected_force = mass * (acceleration - gravity)
        arm_world = _transform_rotation_matrix(body_q) @ arm_local
        expected_torque = np.cross(arm_world, expected_force)

        time.append((step + 1) * dt)
        force_vbd.append(reported[:3])
        force_balance.append(expected_force)
        torque_vbd.append(reported[3:])
        torque_balance.append(expected_torque)

        state0, state1 = state1, state0

    return {
        "time": np.asarray(time),
        "force_vbd": np.asarray(force_vbd),
        "force_balance": np.asarray(force_balance),
        "torque_vbd": np.asarray(torque_vbd),
        "torque_balance": np.asarray(torque_balance),
    }


def simulate_double(
    device: str = "cpu",
    seconds: int = REPORT_SECONDS,
    frame_dt: float = BASE_FRAME_DT,
    substeps: int = BASE_SUBSTEPS,
    iterations: int = BASE_ITERATIONS,
    hard_joints: bool = True,
) -> dict[str, np.ndarray]:
    length = 1.0
    half = 0.5 * length
    model, links = build_double_pendulum(device=device, length=length)
    solver = _make_solver(model, iterations, hard_joints)
    state0 = model.state()
    state1 = model.state()
    control = model.control()

    masses = model.body_mass.numpy().astype(float)
    gravity = model.gravity.numpy()[0].astype(float)
    arm_local = np.array([0.0, 0.0, half], dtype=float)

    time = []
    force_vbd = []
    force_balance = []
    torque_vbd = []
    torque_balance = []
    dt = frame_dt / substeps
    steps = int(round(seconds / frame_dt)) * substeps

    for step in range(steps):
        vel_before = state0.body_qd.numpy()[list(links), :3].astype(float)
        solver.step(state0, state1, control, None, dt)
        body_q = state1.body_q.numpy()[list(links)]
        vel_after = state1.body_qd.numpy()[list(links), :3].astype(float)
        acceleration = (vel_after - vel_before) / dt

        reported = state1.body_parent_f.numpy()[list(links)].astype(float)
        expected = np.zeros((2, 3), dtype=float)
        expected[1] = masses[links[1]] * (acceleration[1] - gravity)
        expected[0] = masses[links[0]] * (acceleration[0] - gravity) + expected[1]

        expected_torque = np.zeros((2, 3), dtype=float)
        for row in range(2):
            arm_world = _transform_rotation_matrix(body_q[row]) @ arm_local
            expected_torque[row] = np.cross(arm_world, expected[row])

        time.append((step + 1) * dt)
        force_vbd.append(reported[:, :3])
        force_balance.append(expected)
        torque_vbd.append(reported[:, 3:])
        torque_balance.append(expected_torque)

        state0, state1 = state1, state0

    return {
        "time": np.asarray(time),
        "force_vbd": np.asarray(force_vbd),
        "force_balance": np.asarray(force_balance),
        "torque_vbd": np.asarray(torque_vbd),
        "torque_balance": np.asarray(torque_balance),
    }


def summarize(data: dict[str, np.ndarray]) -> dict[str, float]:
    force_error = np.linalg.norm(data["force_vbd"] - data["force_balance"], axis=-1)
    force_reference = np.linalg.norm(data["force_balance"], axis=-1)
    force_scale = np.maximum(force_reference, 1.0)
    torque_error = np.linalg.norm(data["torque_vbd"] - data["torque_balance"], axis=-1)
    torque_reference = np.linalg.norm(data["torque_balance"], axis=-1)
    torque_scale = np.maximum(torque_reference, 1.0)
    force_reference_rms = float(np.sqrt(np.mean(force_reference**2)))
    torque_reference_rms = float(np.sqrt(np.mean(torque_reference**2)))
    rms_force_error = float(np.sqrt(np.mean(force_error**2)))
    rms_torque_error = float(np.sqrt(np.mean(torque_error**2)))
    return {
        "max_force_error_n": float(np.max(force_error)),
        "rms_force_error_n": rms_force_error,
        "rms_force_reference_n": force_reference_rms,
        "rms_force_error_percent_of_reference": float(100.0 * rms_force_error / max(force_reference_rms, 1.0e-12)),
        "max_force_relative_error": float(np.max(force_error / force_scale)),
        "max_torque_error_nm": float(np.max(torque_error)),
        "rms_torque_error_nm": rms_torque_error,
        "rms_torque_reference_nm": torque_reference_rms,
        "rms_torque_error_percent_of_reference": float(100.0 * rms_torque_error / max(torque_reference_rms, 1.0e-12)),
        "max_torque_relative_error": float(np.max(torque_error / torque_scale)),
    }


def summarize_joints(data: dict[str, np.ndarray], labels: tuple[str, ...]) -> list[dict[str, float | str]]:
    force_error = np.linalg.norm(data["force_vbd"] - data["force_balance"], axis=-1)
    force_reference = np.linalg.norm(data["force_balance"], axis=-1)
    torque_error = np.linalg.norm(data["torque_vbd"] - data["torque_balance"], axis=-1)
    torque_reference = np.linalg.norm(data["torque_balance"], axis=-1)

    if force_error.ndim == 1:
        force_error = force_error[:, None]
        force_reference = force_reference[:, None]
        torque_error = torque_error[:, None]
        torque_reference = torque_reference[:, None]

    rows: list[dict[str, float | str]] = []
    for joint_index, label in enumerate(labels):
        force_ref_rms = float(np.sqrt(np.mean(force_reference[:, joint_index] ** 2)))
        torque_ref_rms = float(np.sqrt(np.mean(torque_reference[:, joint_index] ** 2)))
        force_err_rms = float(np.sqrt(np.mean(force_error[:, joint_index] ** 2)))
        torque_err_rms = float(np.sqrt(np.mean(torque_error[:, joint_index] ** 2)))
        rows.append(
            {
                "label": label,
                "rms_force_reference_n": force_ref_rms,
                "rms_force_error_n": force_err_rms,
                "rms_force_error_percent_of_reference": 100.0 * force_err_rms / max(force_ref_rms, 1.0e-12),
                "rms_torque_reference_nm": torque_ref_rms,
                "rms_torque_error_nm": torque_err_rms,
                "rms_torque_error_percent_of_reference": 100.0 * torque_err_rms / max(torque_ref_rms, 1.0e-12),
            }
        )
    return rows


def load_metadata(device: str = "cpu") -> dict[str, dict[str, float]]:
    single_model, single_link = build_single_pendulum(device=device)
    double_model, double_links = build_double_pendulum(device=device)

    single_mass = float(single_model.body_mass.numpy()[single_link])
    single_g = float(np.linalg.norm(single_model.gravity.numpy()[0]))
    double_masses = double_model.body_mass.numpy().astype(float)
    double_g = float(np.linalg.norm(double_model.gravity.numpy()[0]))
    double_link_mass = float(double_masses[double_links[0]])

    return {
        "single_pendulum": {
            "body_mass_kg": single_mass,
            "body_weight_n": single_mass * single_g,
        },
        "double_pendulum": {
            "link_mass_kg": double_link_mass,
            "link_weight_n": double_link_mass * double_g,
            "root_subtree_weight_n": float(np.sum(double_masses[list(double_links)]) * double_g),
        },
    }


def _add_error_note(ax, error: np.ndarray, reference: np.ndarray, unit: str, loc: str = "upper right") -> None:
    rms_error = float(np.sqrt(np.mean(error**2)))
    rms_reference = float(np.sqrt(np.mean(reference**2)))
    percent = 100.0 * rms_error / max(rms_reference, 1.0e-12)
    x = 0.98 if "right" in loc else 0.02
    ha = "right" if "right" in loc else "left"
    y = 0.93 if "upper" in loc else 0.07
    va = "top" if "upper" in loc else "bottom"
    ax.text(
        x,
        y,
        f"RMS {rms_error:.3g} {unit} ({percent:.2g}% of {rms_reference:.3g} {unit})",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#d9e0e8", "alpha": 0.9},
    )


def plot_single(data: dict[str, np.ndarray]) -> None:
    plt = _get_pyplot()
    t = data["time"]
    error = np.linalg.norm(data["force_vbd"] - data["force_balance"], axis=1)
    torque_error = np.linalg.norm(data["torque_vbd"] - data["torque_balance"], axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(12.0, 14.0), sharex=True)
    for idx, label in enumerate(("Fx", "Fy", "Fz")):
        axes[0].plot(t, data["force_vbd"][:, idx], label=f"{label} VBD", linewidth=2.0)
        axes[0].plot(t, data["force_balance"][:, idx], "--", label=f"{label} reference", linewidth=1.8)
    _configure_axes(axes[0], "Single pendulum joint force", "force [N]")
    axes[0].legend(ncol=3)

    axes[1].plot(t, data["torque_vbd"][:, 1], label="Ty VBD", linewidth=2.0)
    axes[1].plot(t, data["torque_balance"][:, 1], "--", label="Ty reference", linewidth=1.8)
    _configure_axes(axes[1], "Single pendulum joint torque about COM", "torque [N m]")
    axes[1].legend()

    axes[2].plot(t, error, linewidth=2.0)
    _configure_axes(axes[2], "Force reference residual", "force error [N]")
    _add_error_note(axes[2], error, np.linalg.norm(data["force_balance"], axis=1), "N")

    axes[3].plot(t, torque_error, linewidth=2.0, color="tab:orange")
    _configure_axes(axes[3], "Torque reference residual", "torque error [N m]")
    _add_error_note(axes[3], torque_error, np.linalg.norm(data["torque_balance"], axis=1), "N m")

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "single_pendulum_reaction.png", dpi=180)
    plt.close(fig)


def plot_double(data: dict[str, np.ndarray]) -> None:
    plt = _get_pyplot()
    t = data["time"]
    force_error = np.linalg.norm(data["force_vbd"] - data["force_balance"], axis=2)
    torque_error = np.linalg.norm(data["torque_vbd"] - data["torque_balance"], axis=2)

    fig, axes = plt.subplots(4, 2, figsize=(14.0, 16.0), sharex=True)
    names = ("root joint", "middle joint")
    for body in range(2):
        ax = axes[0, body]
        for idx, label in enumerate(("Fx", "Fy", "Fz")):
            ax.plot(t, data["force_vbd"][:, body, idx], label=f"{label} VBD", linewidth=2.0)
            ax.plot(t, data["force_balance"][:, body, idx], "--", label=f"{label} reference", linewidth=1.8)
        _configure_axes(ax, f"Double pendulum {names[body]} force", "force [N]")
        ax.legend(ncol=2, loc="lower right")

        ax = axes[1, body]
        ax.plot(t, data["torque_vbd"][:, body, 1], label="Ty VBD", linewidth=2.0)
        ax.plot(t, data["torque_balance"][:, body, 1], "--", label="Ty reference", linewidth=1.8)
        _configure_axes(ax, f"Double pendulum {names[body]} torque about COM", "torque [N m]")
        ax.legend(loc="lower right")

        ax = axes[2, body]
        ax.plot(t, force_error[:, body], linewidth=2.0)
        _configure_axes(ax, f"Double pendulum {names[body]} force reference residual", "force error [N]")
        _add_error_note(
            ax,
            force_error[:, body],
            np.linalg.norm(data["force_balance"][:, body], axis=1),
            "N",
        )

        ax = axes[3, body]
        ax.plot(t, torque_error[:, body], linewidth=2.0, color="tab:orange")
        _configure_axes(ax, f"Double pendulum {names[body]} torque reference residual", "torque error [N m]")
        _add_error_note(
            ax,
            torque_error[:, body],
            np.linalg.norm(data["torque_balance"][:, body], axis=1),
            "N m",
        )

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "double_pendulum_reaction.png", dpi=180)
    plt.close(fig)


def run_budget_sweeps() -> dict[str, dict[str, list[dict[str, float | int]]]]:
    cases = {
        "single_pendulum": simulate_single,
        "double_pendulum": simulate_double,
    }
    sweeps: dict[str, dict[str, list[dict[str, float | int]]]] = {
        "iteration_sweep": {},
        "substep_sweep": {},
    }

    for case_name, simulate in cases.items():
        sweeps["iteration_sweep"][case_name] = []
        for iterations in ITERATION_SWEEP:
            summary = summarize(simulate(iterations=iterations, substeps=BASE_SUBSTEPS, frame_dt=BASE_FRAME_DT))
            sweeps["iteration_sweep"][case_name].append({"iterations": iterations, **summary})

        sweeps["substep_sweep"][case_name] = []
        for substeps in SUBSTEP_SWEEP:
            summary = summarize(simulate(iterations=BASE_ITERATIONS, substeps=substeps, frame_dt=BASE_FRAME_DT))
            sweeps["substep_sweep"][case_name].append({"substeps": substeps, **summary})

    return sweeps


def plot_budget_sweeps(sweeps: dict[str, dict[str, list[dict[str, float | int]]]]) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    case_labels = {
        "single_pendulum": "single",
        "double_pendulum": "double",
    }

    def plot_metric(ax, sweep_name: str, x_key: str, y_key: str, title: str, ylabel: str) -> None:
        for case_name, rows in sweeps[sweep_name].items():
            x = [row[x_key] for row in rows]
            y = [row[y_key] for row in rows]
            ax.plot(x, y, marker="o", linewidth=2.0, label=case_labels[case_name])
        ax.set_title(title)
        ax.set_xlabel(x_key)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28)
        ax.legend()

    plot_metric(
        axes[0, 0],
        "iteration_sweep",
        "iterations",
        "rms_force_error_n",
        f"Iteration sweep at {BASE_SUBSTEPS} substeps: RMS force residual",
        "force error [N]",
    )
    plot_metric(
        axes[0, 1],
        "iteration_sweep",
        "iterations",
        "rms_torque_error_nm",
        f"Iteration sweep at {BASE_SUBSTEPS} substeps: RMS torque residual",
        "torque error [N m]",
    )
    plot_metric(
        axes[1, 0],
        "substep_sweep",
        "substeps",
        "rms_force_error_n",
        f"Substep sweep at {BASE_ITERATIONS} iterations: RMS force residual",
        "force error [N]",
    )
    plot_metric(
        axes[1, 1],
        "substep_sweep",
        "substeps",
        "rms_torque_error_nm",
        f"Substep sweep at {BASE_ITERATIONS} iterations: RMS torque residual",
        "torque error [N m]",
    )

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "budget_sweeps.png", dpi=180)
    plt.close(fig)


def save_npz(name: str, data: dict[str, np.ndarray]) -> None:
    np.savez(DATA_DIR / f"{name}.npz", **data)


def _build_video_case(case: str, device: str) -> tuple[newton.Model, tuple[int, ...], np.ndarray, wp.vec3, int]:
    if case == "single":
        model, link = build_single_pendulum_video_scene(device=device)
        return model, (link,), np.array([0.0, 0.0, 0.5], dtype=float), wp.vec3(2.8, -4.8, 1.35), BASE_ITERATIONS
    if case == "double":
        model, links = build_double_pendulum_video_scene(device=device)
        return model, links, np.array([0.0, 0.0, 0.5], dtype=float), wp.vec3(3.4, -5.4, 1.45), BASE_ITERATIONS
    raise ValueError(f"Unknown video case: {case}")


def _simulate_video_reaction_errors(
    case: str,
    device: str = "cuda:0",
    fps: int = VIDEO_FPS,
    seconds: int = VIDEO_SECONDS,
    substeps: int = VIDEO_SUBSTEPS,
) -> dict[str, np.ndarray]:
    model, links, arm_local, _camera_pos, iterations = _build_video_case(case, device)
    solver = _make_solver(model, iterations, hard_joints=True)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    masses = model.body_mass.numpy().astype(float)
    gravity = model.gravity.numpy()[0].astype(float)
    link_indices = list(links)
    dt = 1.0 / (fps * substeps)
    steps = fps * seconds * substeps

    time = []
    force_vbd = []
    force_balance = []
    torque_vbd = []
    torque_balance = []

    for step in range(steps):
        vel_before = state_0.body_qd.numpy()[link_indices, :3].astype(float)
        solver.step(state_0, state_1, control, None, dt)
        body_q = state_1.body_q.numpy()[link_indices]
        vel_after = state_1.body_qd.numpy()[link_indices, :3].astype(float)
        acceleration = (vel_after - vel_before) / dt
        reported = state_1.body_parent_f.numpy()[link_indices].astype(float)

        expected = np.zeros((len(links), 3), dtype=float)
        if case == "single":
            expected[0] = masses[links[0]] * (acceleration[0] - gravity)
        else:
            expected[1] = masses[links[1]] * (acceleration[1] - gravity)
            expected[0] = masses[links[0]] * (acceleration[0] - gravity) + expected[1]

        expected_torque = np.zeros((len(links), 3), dtype=float)
        for row in range(len(links)):
            arm_world = _transform_rotation_matrix(body_q[row]) @ arm_local
            expected_torque[row] = np.cross(arm_world, expected[row])

        time.append((step + 1) * dt)
        force_vbd.append(reported[:, :3])
        force_balance.append(expected)
        torque_vbd.append(reported[:, 3:])
        torque_balance.append(expected_torque)

        state_0, state_1 = state_1, state_0

    return {
        "time": np.asarray(time),
        "force_vbd": np.asarray(force_vbd),
        "force_balance": np.asarray(force_balance),
        "torque_vbd": np.asarray(torque_vbd),
        "torque_balance": np.asarray(torque_balance),
    }


def _draw_polyline(
    draw,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int,
    dashed: bool = False,
) -> None:
    if len(points) <= 1:
        return
    if not dashed:
        draw.line(points, fill=color, width=width)
        return

    for i in range(len(points) - 1):
        if i % 2 == 0:
            draw.line((points[i], points[i + 1]), fill=color, width=width)


def _series_bounds(*series: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(item, dtype=float).reshape(-1) for item in series])
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    span = y_max - y_min
    if span < 1.0e-8:
        pad = max(abs(y_min), 1.0) * 0.08
    else:
        pad = span * 0.08
    return y_min - pad, y_max + pad


def _fmt_metric(value: float) -> str:
    return f"{value:.2g}"


def _draw_load_axis(
    draw,
    box: tuple[int, int, int, int],
    title: str,
    time: np.ndarray,
    vbd_values: np.ndarray,
    reference_values: np.ndarray,
    error_values: np.ndarray,
    current_index: int,
    ylabel: str,
    unit: str,
) -> None:
    left, top, right, bottom = box
    plot_left = left + 58
    plot_top = top + 38
    plot_right = right - 14
    plot_bottom = bottom - 34
    x_max = float(time[-1])
    y_min, y_max = _series_bounds(vbd_values, reference_values)
    y_span = max(y_max - y_min, 1.0e-8)
    rms_error = float(np.sqrt(np.mean(error_values**2)))
    rms_reference = float(np.sqrt(np.mean(reference_values**2)))

    draw.text((left, top), title, fill=(23, 32, 42), font=_font(15, bold=True))
    draw.text((plot_left, plot_top - 20), ylabel, fill=(70, 82, 96), font=_font(12))
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(190, 200, 212), width=1)

    for frac in (0.0, 0.5, 1.0):
        x = plot_left + frac * (plot_right - plot_left)
        y = plot_bottom - frac * (plot_bottom - plot_top)
        y_value = y_min + frac * y_span
        draw.line((x, plot_top, x, plot_bottom), fill=(225, 230, 236), width=1)
        draw.line((plot_left, y, plot_right, y), fill=(225, 230, 236), width=1)
        draw.text((x - 12, plot_bottom + 8), f"{frac * x_max:.1f}", fill=(70, 82, 96), font=_font(11))
        draw.text((left + 8, y - 7), f"{y_value:.2g}", fill=(70, 82, 96), font=_font(11))

    def points(data: np.ndarray, end: int) -> list[tuple[int, int]]:
        pts = []
        for t_value, value in zip(time[: end + 1], data[: end + 1], strict=True):
            x = plot_left + int((float(t_value) / x_max) * (plot_right - plot_left))
            y = plot_bottom - int(((float(value) - y_min) / y_span) * (plot_bottom - plot_top))
            pts.append((x, y))
        return pts

    vbd_color = (31, 119, 180)
    reference_color = (255, 127, 14)
    faded_vbd = tuple(min(255, int(0.45 * channel + 0.55 * 255)) for channel in vbd_color)
    faded_ref = tuple(min(255, int(0.45 * channel + 0.55 * 255)) for channel in reference_color)
    _draw_polyline(draw, points(vbd_values, len(vbd_values) - 1), faded_vbd, 2)
    _draw_polyline(draw, points(reference_values, len(reference_values) - 1), faded_ref, 2, dashed=True)
    _draw_polyline(draw, points(vbd_values, current_index), vbd_color, 3)
    _draw_polyline(draw, points(reference_values, current_index), reference_color, 3, dashed=True)

    current_x = plot_left + int((float(time[current_index]) / x_max) * (plot_right - plot_left))
    draw.line((current_x, plot_top, current_x, plot_bottom), fill=(44, 62, 80), width=2)
    legend_y = plot_top + 8
    draw.line((plot_right - 150, legend_y + 7, plot_right - 122, legend_y + 7), fill=vbd_color, width=3)
    draw.text((plot_right - 116, legend_y), "VBD", fill=(44, 62, 80), font=_font(11))
    _draw_polyline(
        draw,
        [(plot_right - 150, legend_y + 24), (plot_right - 122, legend_y + 24)],
        reference_color,
        3,
        dashed=True,
    )
    draw.text((plot_right - 116, legend_y + 17), "reference", fill=(44, 62, 80), font=_font(11))
    draw.text(
        (plot_left + 6, bottom - 19),
        f"load {_fmt_metric(rms_reference)} {unit}; err {_fmt_metric(rms_error)} {unit}",
        fill=(70, 82, 96),
        font=_font(10),
    )


def _error_panel_frame(
    case: str,
    data: dict[str, np.ndarray],
    frame: int,
    width: int = VIDEO_PLOT_WIDTH,
    height: int = VIDEO_HEIGHT,
    substeps: int = VIDEO_SUBSTEPS,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    time = data["time"]
    force_error = np.linalg.norm(data["force_vbd"] - data["force_balance"], axis=2)
    torque_y_error = np.abs(data["torque_vbd"][:, :, 1] - data["torque_balance"][:, :, 1])
    force_vbd = np.linalg.norm(data["force_vbd"], axis=2)
    force_reference = np.linalg.norm(data["force_balance"], axis=2)
    torque_vbd = data["torque_vbd"][:, :, 1]
    torque_reference = data["torque_balance"][:, :, 1]
    current_index = min((frame + 1) * substeps - 1, len(time) - 1)

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "Reaction loads", fill=(23, 32, 42), font=_font(22, bold=True))
    draw.text(
        (24, 48),
        f"t = {time[current_index]:.2f}s",
        fill=(70, 82, 96),
        font=_font(14, bold=True),
    )

    if case == "single":
        _draw_load_axis(
            draw,
            (18, 86, width - 18, 304),
            "single |F|",
            time,
            force_vbd[:, 0],
            force_reference[:, 0],
            force_error[:, 0],
            current_index,
            "y: ||F|| [N]",
            "N",
        )
        _draw_load_axis(
            draw,
            (18, 314, width - 18, height - 18),
            "single Ty",
            time,
            torque_vbd[:, 0],
            torque_reference[:, 0],
            torque_y_error[:, 0],
            current_index,
            "y: Ty [N m]",
            "N m",
        )
    else:
        _draw_load_axis(
            draw,
            (18, 86, width // 2 - 8, 304),
            "root |F|",
            time,
            force_vbd[:, 0],
            force_reference[:, 0],
            force_error[:, 0],
            current_index,
            "y: ||F|| [N]",
            "N",
        )
        _draw_load_axis(
            draw,
            (width // 2 + 8, 86, width - 18, 304),
            "root Ty",
            time,
            torque_vbd[:, 0],
            torque_reference[:, 0],
            torque_y_error[:, 0],
            current_index,
            "y: Ty [N m]",
            "N m",
        )
        _draw_load_axis(
            draw,
            (18, 314, width // 2 - 8, height - 18),
            "middle |F|",
            time,
            force_vbd[:, 1],
            force_reference[:, 1],
            force_error[:, 1],
            current_index,
            "y: ||F|| [N]",
            "N",
        )
        _draw_load_axis(
            draw,
            (width // 2 + 8, 314, width - 18, height - 18),
            "middle Ty",
            time,
            torque_vbd[:, 1],
            torque_reference[:, 1],
            torque_y_error[:, 1],
            current_index,
            "y: Ty [N m]",
            "N m",
        )

    return np.asarray(image, dtype=np.uint8)


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


def _render_video_case(
    case: str,
    output_path: Path,
    viewer,
    error_data: dict[str, np.ndarray],
    scene_width: int = VIDEO_SCENE_WIDTH,
    plot_width: int = VIDEO_PLOT_WIDTH,
    height: int = VIDEO_HEIGHT,
    fps: int = VIDEO_FPS,
) -> None:
    device = "cuda:0"
    model, _links, _arm_local, camera_pos, iterations = _build_video_case(case, device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_joint_linear_ke=BASE_RIGID_JOINT_LINEAR_KE,
        rigid_joint_angular_ke=BASE_RIGID_JOINT_ANGULAR_KE,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    viewer.set_model(model)
    viewer.set_camera(camera_pos, pitch=-12.0, yaw=32.0)
    viewer.camera.look_at(wp.vec3(0.0, 0.0, -0.6))

    writer = _open_video_writer(output_path, width=scene_width + plot_width, height=height, fps=fps)
    if writer.stdin is None:
        raise RuntimeError("ffmpeg stdin pipe was not opened")

    frame_dt = 1.0 / fps
    sim_dt = frame_dt / VIDEO_SUBSTEPS
    frame_buffer = None
    for frame in range(fps * VIDEO_SECONDS):
        for _substep in range(VIDEO_SUBSTEPS):
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0

        viewer.begin_frame((frame + 1) * frame_dt)
        viewer.log_state(state_0)
        viewer.end_frame()
        frame_buffer = viewer.get_frame(frame_buffer)
        scene = frame_buffer.numpy()
        panel = _error_panel_frame(case, error_data, frame, width=plot_width, height=height)
        writer.stdin.write(np.ascontiguousarray(np.concatenate((scene, panel), axis=1)).tobytes())

    writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")


def render_videos() -> None:
    if not wp.is_cuda_available():
        print("Skipping videos: ViewerGL video rendering requires CUDA.")
        return
    if shutil.which("ffmpeg") is None:
        print("Skipping videos: ffmpeg was not found.")
        return

    single_errors = _simulate_video_reaction_errors("single")
    double_errors = _simulate_video_reaction_errors("double")

    viewer = ViewerGL(width=VIDEO_SCENE_WIDTH, height=VIDEO_HEIGHT, headless=True)
    try:
        viewer.show_ui = False
        _render_video_case("single", ASSET_DIR / "single_pendulum_newton.mp4", viewer, single_errors)
        _render_video_case("double", ASSET_DIR / "double_pendulum_newton.mp4", viewer, double_errors)
    finally:
        viewer.close()


def main() -> None:
    wp.init()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    single = simulate_single()
    double = simulate_double()
    soft_single = simulate_single(hard_joints=False)
    soft_double = simulate_double(hard_joints=False)

    plot_single(single)
    plot_double(double)
    save_npz("single_pendulum_reaction", single)
    save_npz("double_pendulum_reaction", double)
    save_npz("single_pendulum_reaction_soft", soft_single)
    save_npz("double_pendulum_reaction_soft", soft_double)
    sweeps = run_budget_sweeps()
    plot_budget_sweeps(sweeps)
    render_videos()

    summary = {
        "budget": {
            "iterations": BASE_ITERATIONS,
            "frame_dt": BASE_FRAME_DT,
            "substeps": BASE_SUBSTEPS,
            "seconds": REPORT_SECONDS,
            "rigid_joint_linear_ke": BASE_RIGID_JOINT_LINEAR_KE,
            "rigid_joint_angular_ke": BASE_RIGID_JOINT_ANGULAR_KE,
            "rigid_avbd_beta": 0.0,
        },
        "single_pendulum": summarize(single),
        "double_pendulum": summarize(double),
        "soft_single_pendulum": summarize(soft_single),
        "soft_double_pendulum": summarize(soft_double),
        "joint_scales": {
            "single_pendulum": summarize_joints(single, ("single joint",)),
            "double_pendulum": summarize_joints(double, ("root joint", "middle joint")),
        },
        "load_metadata": load_metadata(),
        "sweeps": sweeps,
    }
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (DATA_DIR / "budget_sweeps.json").write_text(json.dumps(sweeps, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
