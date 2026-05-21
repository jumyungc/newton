# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate VBD cable tension validation data and plots."""

from __future__ import annotations

import json
import multiprocessing
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

GRAVITY = 9.81
LOAD_STEPS = 160
LOAD_DT = 1.0 / 300.0
LOAD_ITERATIONS = 32
WRAP_STEPS = 600
WRAP_DT = 1.0 / 600.0
WRAP_ITERATIONS = 32
CAPSTAN_STEPS = 900
CAPSTAN_DT = 1.0 / 600.0
CAPSTAN_ITERATIONS = 32
VIDEO_WIDTH = 1280
VIDEO_SCENE_WIDTH = 800
VIDEO_PLOT_WIDTH = VIDEO_WIDTH - VIDEO_SCENE_WIDTH
VIDEO_HEIGHT = 720
VIDEO_FPS = 60
VIDEO_SECONDS = 4
VIDEO_HANGING_SUBSTEPS = 5
VIDEO_PULLEY_SUBSTEPS = 10
VIDEO_CAPSTAN_SUBSTEPS = 10
CABLE_COLOR = (0.92, 0.72, 0.16)


@wp.kernel
def _set_body_linear_force(
    body: int,
    force: wp.vec3,
    body_f: wp.array[wp.spatial_vector],
):
    body_f[body] = wp.spatial_vector(force, wp.vec3(0.0))


@wp.kernel
def _apply_endpoint_loads(
    first_body: int,
    last_body: int,
    left_load: float,
    right_load: float,
    left_tangent: wp.vec3,
    right_tangent: wp.vec3,
    body_f: wp.array[wp.spatial_vector],
):
    body_f[first_body] = wp.spatial_vector(left_load * left_tangent, wp.vec3(0.0))
    body_f[last_body] = wp.spatial_vector(right_load * right_tangent, wp.vec3(0.0))


@wp.kernel
def _update_cable_line(
    body: int,
    body_q: wp.array[wp.transform],
    line_start: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
    line_start[0] = wp.vec3(0.0, 0.0, 0.0)
    line_end[0] = wp.transform_get_translation(body_q[body])


@wp.kernel
def _update_body_chain_lines(
    body_ids: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    line_start: wp.array[wp.vec3],
    line_end: wp.array[wp.vec3],
):
    segment = wp.tid()
    line_start[segment] = wp.transform_get_translation(body_q[body_ids[segment]])
    line_end[segment] = wp.transform_get_translation(body_q[body_ids[segment + 1]])


@wp.kernel
def _update_body_points(
    body_ids: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    points: wp.array[wp.vec3],
):
    body = wp.tid()
    points[body] = wp.transform_get_translation(body_q[body_ids[body]])


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "font.size": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    return plt


def _as_np(points: list[wp.vec3]) -> np.ndarray:
    return np.array([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=float)


def _set_rod_shape_color(builder: newton.ModelBuilder, bodies: list[int], color: tuple[float, float, float]) -> None:
    body_set = set(bodies)
    for shape_index, shape_body in enumerate(builder.shape_body):
        if shape_body in body_set:
            builder.shape_color[shape_index] = color


def build_hanging_cable(device: str) -> tuple[newton.Model, int, int]:
    builder = newton.ModelBuilder(gravity=-GRAVITY, up_axis=newton.Axis.Z)
    builder.request_state_attributes("vbd:cable_tension")

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 1000.0

    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -1.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg, color=(0.16, 0.45, 0.72))
    joint = builder.add_joint_cable(
        parent=-1,
        child=body,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        stretch_stiffness=1.0e5,
        stretch_damping=1.0e-2,
        label="support_cable",
    )
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device), body, joint


def simulate_hanging_load(applied_downward_force: float, device: str = "cpu") -> dict[str, np.ndarray | float]:
    model, body, joint = build_hanging_cable(device)
    solver = newton.solvers.SolverVBD(model, iterations=LOAD_ITERATIONS)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    mass = float(model.body_mass.numpy()[body])
    expected = mass * GRAVITY + applied_downward_force
    time = []
    tension = []

    for step in range(LOAD_STEPS):
        state_0.clear_forces()
        if applied_downward_force > 0.0:
            body_f = state_0.body_f.numpy()
            body_f[body, :3] = (0.0, 0.0, -applied_downward_force)
            state_0.body_f.assign(body_f)

        solver.step(state_0, state_1, control, None, LOAD_DT)
        time.append((step + 1) * LOAD_DT)
        tension.append(float(state_1.vbd.cable_tension.numpy()[joint]))
        state_0, state_1 = state_1, state_0

    return {
        "time": np.asarray(time),
        "tension": np.asarray(tension),
        "expected": float(expected),
        "mass": float(mass),
        "applied_downward_force": float(applied_downward_force),
    }


def build_wrapped_cable(mu: float, device: str = "cpu") -> tuple[newton.Model, list[int], list[int], np.ndarray]:
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    builder.request_state_attributes("vbd:cable_tension")

    cable_cfg = newton.ModelBuilder.ShapeConfig()
    cable_cfg.density = 1000.0
    cable_cfg.ke = 1.0e4
    cable_cfg.kd = 0.0
    cable_cfg.mu = mu
    cable_cfg.gap = 0.02

    pulley_cfg = newton.ModelBuilder.ShapeConfig()
    pulley_cfg.density = 0.0
    pulley_cfg.ke = 1.0e4
    pulley_cfg.kd = 0.0
    pulley_cfg.mu = mu
    pulley_cfg.gap = 0.02

    pulley_radius = 0.40
    cable_radius = 0.035
    center = np.array([0.0, 0.0, 1.0], dtype=float)
    cyl_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * np.pi)
    builder.add_shape_cylinder(
        -1,
        xform=wp.transform(wp.vec3(*center), cyl_q),
        radius=pulley_radius,
        half_height=0.25,
        cfg=pulley_cfg,
        color=(0.25, 0.27, 0.30),
        label="pulley",
    )

    centerline_radius = pulley_radius + 0.75 * cable_radius
    angles = np.linspace(np.deg2rad(210.0), np.deg2rad(-30.0), 25)
    points = [
        wp.vec3(
            float(centerline_radius * np.cos(angle)),
            0.0,
            float(center[2] + centerline_radius * np.sin(angle)),
        )
        for angle in angles
    ]
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(points)
    bodies, joints = builder.add_rod(
        points,
        quaternions,
        radius=cable_radius,
        cfg=cable_cfg,
        stretch_stiffness=1.0e4,
        stretch_damping=0.0,
        bend_stiffness=0.0,
        bend_damping=0.0,
        label=f"wrap_mu_{mu:g}",
    )
    _set_rod_shape_color(builder, bodies, CABLE_COLOR)
    builder.color()
    return builder.finalize(device=device), bodies, joints, _as_np(points)


def simulate_wrapped_cable(mu: float, device: str = "cpu") -> dict[str, np.ndarray | float]:
    model, bodies, joints, points = build_wrapped_cable(mu, device)
    solver = newton.solvers.SolverVBD(model, iterations=WRAP_ITERATIONS, rigid_contact_hard=True)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    left_tangent = points[0] - points[1]
    left_tangent /= np.linalg.norm(left_tangent)
    right_tangent = points[-1] - points[-2]
    right_tangent /= np.linalg.norm(right_tangent)
    left_load = 20.0
    right_load = 20.0

    for _step in range(WRAP_STEPS):
        state_0.clear_forces()
        body_f = state_0.body_f.numpy()
        body_f[bodies[0], :3] = left_load * left_tangent
        body_f[bodies[-1], :3] = right_load * right_tangent
        state_0.body_f.assign(body_f)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, WRAP_DT)
        state_0, state_1 = state_1, state_0

    tensions = state_0.vbd.cable_tension.numpy()[joints].astype(float)
    wrap_angle = np.deg2rad(240.0)
    capstan_ratio = float(np.exp(mu * wrap_angle))
    return {
        "mu": float(mu),
        "joint_index": np.arange(len(joints), dtype=int),
        "tension": tensions,
        "points": points,
        "left_load": float(left_load),
        "right_load": float(right_load),
        "wrap_angle": float(wrap_angle),
        "capstan_ratio_bound": capstan_ratio,
        "measured_ratio": float(np.max(tensions) / max(np.min(tensions), 1.0e-12)),
    }


def _capstan_path(pulley_radius: float, cable_radius: float) -> tuple[list[wp.vec3], dict[str, slice]]:
    centerline_radius = pulley_radius + 0.75 * cable_radius
    left_center = np.array([-0.65, 0.0, 1.0], dtype=float)
    right_center = np.array([0.65, 0.0, 1.0], dtype=float)

    points: list[wp.vec3] = []

    for z in np.linspace(0.25, left_center[2], 7):
        points.append(wp.vec3(float(left_center[0] - centerline_radius), 0.0, float(z)))

    for angle in np.linspace(np.pi, 0.5 * np.pi, 9)[1:]:
        points.append(
            wp.vec3(
                float(left_center[0] + centerline_radius * np.cos(angle)),
                0.0,
                float(left_center[2] + centerline_radius * np.sin(angle)),
            )
        )
    middle_start = len(points) - 1

    for x in np.linspace(left_center[0], right_center[0], 13)[1:]:
        points.append(wp.vec3(float(x), 0.0, float(left_center[2] + centerline_radius)))

    for angle in np.linspace(0.5 * np.pi, 0.0, 9)[1:]:
        points.append(
            wp.vec3(
                float(right_center[0] + centerline_radius * np.cos(angle)),
                0.0,
                float(right_center[2] + centerline_radius * np.sin(angle)),
            )
        )

    for z in np.linspace(right_center[2], 0.25, 7)[1:]:
        points.append(wp.vec3(float(right_center[0] + centerline_radius), 0.0, float(z)))

    regions = {
        "left": slice(0, 1),
        "middle": slice(middle_start + 6, middle_start + 7),
        "right": slice(-1, None),
    }
    return points, regions


def build_capstan_pulley_system(
    mu: float = 0.20,
    device: str = "cpu",
) -> tuple[newton.Model, list[int], list[int], np.ndarray, dict[str, slice]]:
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    builder.request_state_attributes("vbd:cable_tension")

    cable_radius = 0.035
    pulley_radius = 0.32

    cable_cfg = newton.ModelBuilder.ShapeConfig()
    cable_cfg.density = 1000.0
    cable_cfg.ke = 1.0e4
    cable_cfg.kd = 0.0
    cable_cfg.mu = mu
    cable_cfg.gap = 0.02

    pulley_cfg = newton.ModelBuilder.ShapeConfig()
    pulley_cfg.density = 0.0
    pulley_cfg.ke = 1.0e4
    pulley_cfg.kd = 0.0
    pulley_cfg.mu = mu
    pulley_cfg.gap = 0.02

    cylinder_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * np.pi)
    for center_x in (-0.65, 0.65):
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(center_x, 0.0, 1.0), cylinder_rotation),
            radius=pulley_radius,
            half_height=0.25,
            cfg=pulley_cfg,
            color=(0.25, 0.27, 0.30),
            label="capstan_pulley",
        )

    points, regions = _capstan_path(pulley_radius, cable_radius)
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(points)
    bodies, joints = builder.add_rod(
        points,
        quaternions,
        radius=cable_radius,
        cfg=cable_cfg,
        stretch_stiffness=1.0e4,
        stretch_damping=0.0,
        bend_stiffness=0.0,
        bend_damping=0.0,
        label="capstan_two_pulley_cable",
    )
    _set_rod_shape_color(builder, bodies, CABLE_COLOR)
    builder.color()
    return builder.finalize(device=device), bodies, joints, _as_np(points), regions


def capstan_reference(mu: float = 0.20, wrap_angle: float = 0.5 * np.pi) -> dict[str, float | list[float] | list[str]]:
    left_mass = 1.0
    per_pulley_ratio = float(np.exp(mu * wrap_angle))
    left_tension = left_mass * GRAVITY
    middle_tension = left_tension * per_pulley_ratio
    right_tension = middle_tension * per_pulley_ratio
    right_mass = right_tension / GRAVITY
    return {
        "mu": float(mu),
        "wrap_angle": float(wrap_angle),
        "per_pulley_ratio": per_pulley_ratio,
        "left_mass": left_mass,
        "right_mass": right_mass,
        "labels": ["left mass side", "between pulleys", "right mass side"],
        "tensions": [left_tension, middle_tension, right_tension],
    }


def simulate_capstan_pulley_system(mu: float = 0.20, device: str = "cpu") -> dict[str, np.ndarray | float | list[str]]:
    model, bodies, joints, points, regions = build_capstan_pulley_system(mu, device)
    solver = newton.solvers.SolverVBD(model, iterations=CAPSTAN_ITERATIONS, rigid_contact_hard=True)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    left_wrap_angle = 0.5 * np.pi
    right_wrap_angle = 0.5 * np.pi
    left_load = GRAVITY
    middle_reference = left_load * np.exp(mu * left_wrap_angle)
    right_load = middle_reference * np.exp(mu * right_wrap_angle)

    left_tangent = points[0] - points[1]
    left_tangent /= np.linalg.norm(left_tangent)
    right_tangent = points[-1] - points[-2]
    right_tangent /= np.linalg.norm(right_tangent)

    for _step in range(CAPSTAN_STEPS):
        state_0.clear_forces()
        body_f = state_0.body_f.numpy()
        body_f[bodies[0], :3] = left_load * left_tangent
        body_f[bodies[-1], :3] = right_load * right_tangent
        state_0.body_f.assign(body_f)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, CAPSTAN_DT)
        state_0, state_1 = state_1, state_0

    tension_profile = state_0.vbd.cable_tension.numpy()[joints].astype(float)
    middle_span_tension = float(np.mean(tension_profile[regions["middle"]]))
    diagnostic = np.asarray([left_load, middle_span_tension, right_load], dtype=float)
    reference = np.asarray([left_load, middle_reference, right_load], dtype=float)
    return {
        "mu": float(mu),
        "left_wrap_angle": float(left_wrap_angle),
        "right_wrap_angle": float(right_wrap_angle),
        "labels": ["left mass side", "between pulleys", "right mass side"],
        "reference_tension": reference,
        "diagnostic_tension": diagnostic,
        "middle_span_tension": middle_span_tension,
        "tension_profile": tension_profile,
        "points": points,
        "left_mass": float(left_load / GRAVITY),
        "right_mass": float(right_load / GRAVITY),
    }


def summarize_load_case(data: dict[str, np.ndarray | float]) -> dict[str, float]:
    tension = np.asarray(data["tension"], dtype=float)
    expected = float(data["expected"])
    tail = tension[-40:]
    return {
        "expected_n": expected,
        "final_n": float(tension[-1]),
        "tail_mean_n": float(np.mean(tail)),
        "tail_abs_error_n": float(abs(np.mean(tail) - expected)),
        "tail_relative_error": float(abs(np.mean(tail) - expected) / expected),
    }


def plot_load_cases(cases: dict[str, dict[str, np.ndarray | float]]) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True)

    for label, data in cases.items():
        t = np.asarray(data["time"], dtype=float)
        tension = np.asarray(data["tension"], dtype=float)
        expected = float(data["expected"])
        axes[0].plot(t, tension, linewidth=2.4, label=f"{label} VBD")
        axes[0].axhline(expected, linestyle="--", linewidth=1.8, label=f"{label} reference")
        axes[1].plot(t, tension - expected, linewidth=2.2, label=label)

    axes[0].set_title("Hanging cable tension")
    axes[0].set_ylabel("tension [N]")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(ncol=2)

    axes[1].set_title("Tension error")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("VBD - reference [N]")
    axes[1].grid(True, alpha=0.28)
    axes[1].legend()

    fig.savefig(ASSET_DIR / "hanging_load_tension.png", dpi=180)
    plt.close(fig)


def plot_wrapped_profiles(profiles: list[dict[str, np.ndarray | float]]) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), constrained_layout=True)

    for profile in profiles:
        x = np.asarray(profile["joint_index"], dtype=float)
        tension = np.asarray(profile["tension"], dtype=float)
        mu = float(profile["mu"])
        axes[0].plot(x, tension, marker="o", linewidth=2.2, markersize=4.8, label=f"mu={mu:g}")

    axes[0].set_title("VBD cable tension around one pulley")
    axes[0].set_xlabel("cable joint index along wrap")
    axes[0].set_ylabel("stretch tension [N]")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend()

    points = np.asarray(profiles[-1]["points"], dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    axes[1].fill(0.40 * np.cos(theta), 1.0 + 0.40 * np.sin(theta), color="#d8dee8", label="pulley")
    axes[1].plot(
        points[:, 0], points[:, 2], "-o", color="#2563a6", linewidth=2.2, markersize=3.8, label="cable centerline"
    )
    axes[1].set_title("Pulley contact diagnostic geometry")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("z [m]")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    fig.savefig(ASSET_DIR / "single_pulley_tension_gradient.png", dpi=180)
    plt.close(fig)


def plot_capstan_reference(reference: dict[str, float | list[float] | list[str]]) -> None:
    plt = _get_pyplot()
    labels = list(reference["labels"])
    tensions = np.asarray(reference["tensions"], dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.78)
    bars = ax.bar(labels, tensions, color=("#2d6a9f", "#7b9d31", "#b35d32"), width=0.62)
    for bar, value in zip(bars, tensions, strict=True):
        ax.text(bar.get_x() + bar.get_width() * 0.5, value + 0.35, f"{value:.1f} N", ha="center", va="bottom")

    ratio = float(reference["per_pulley_ratio"])
    mu = float(reference["mu"])
    wrap_angle = float(reference["wrap_angle"])
    fig.suptitle("Ideal Capstan Tension Reference", y=0.97, fontsize=20)
    fig.text(
        0.5,
        0.90,
        f"Per-pulley ratio exp(mu theta) = {ratio:.2f}; mu={mu:.2f}, theta={wrap_angle / np.pi:.1f} pi",
        ha="center",
        va="center",
        fontsize=13,
    )
    ax.set_ylabel("ideal tension [N]")
    ax.set_ylim(0.0, float(np.max(tensions)) * 1.24)
    ax.grid(True, axis="y", alpha=0.28)
    fig.savefig(ASSET_DIR / "capstan_reference_pulley_system.png", dpi=180)
    plt.close(fig)


def plot_capstan_simulation(capstan: dict[str, np.ndarray | float | list[str]]) -> None:
    plt = _get_pyplot()
    labels = list(capstan["labels"])
    reference = np.asarray(capstan["reference_tension"], dtype=float)
    diagnostic = np.asarray(capstan["diagnostic_tension"], dtype=float)
    points = np.asarray(capstan["points"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - 0.5 * width, reference, width=width, label="capstan reference", color="#7b9d31")
    axes[0].bar(x + 0.5 * width, diagnostic, width=width, label="simulation diagnostic", color="#2d6a9f")
    for idx, value in enumerate(reference):
        axes[0].text(idx - 0.5 * width, value + 0.45, f"{value:.1f}", ha="center", va="bottom", fontsize=11)
    for idx, value in enumerate(diagnostic):
        axes[0].text(idx + 0.5 * width, value + 0.45, f"{value:.1f}", ha="center", va="bottom", fontsize=11)
    axes[0].set_title("Two-pulley capstan diagnostic")
    axes[0].set_xticks(x, labels, rotation=10)
    axes[0].set_ylabel("tension [N]")
    axes[0].set_ylim(0.0, float(max(np.max(reference), np.max(diagnostic))) * 1.22)
    axes[0].grid(True, axis="y", alpha=0.28)
    axes[0].legend(loc="upper left")

    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    for center_x in (-0.65, 0.65):
        axes[1].fill(
            center_x + 0.32 * np.cos(theta),
            1.0 + 0.32 * np.sin(theta),
            color="#d8dee8",
            alpha=0.95,
        )
    axes[1].plot(points[:, 0], points[:, 2], "-o", color="#2563a6", linewidth=2.2, markersize=3.4)
    axes[1].set_title("Simulated two-pulley geometry")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("z [m]")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.28)

    fig.savefig(ASSET_DIR / "capstan_simulated_pulley_system.png", dpi=180)
    plt.close(fig)


def save_npz(name: str, data: dict[str, np.ndarray | float]) -> None:
    arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    scalars = {key: value for key, value in data.items() if not isinstance(value, np.ndarray)}
    np.savez(DATA_DIR / f"{name}.npz", **arrays)
    with (DATA_DIR / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(scalars, f, indent=2)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _plot_points(x_values: np.ndarray, y_values: np.ndarray, rect: tuple[int, int, int, int], xlim, ylim):
    left, top, right, bottom = rect
    x0, x1 = xlim
    y0, y1 = ylim
    if len(x_values) == 0:
        return []

    xs = left + (x_values - x0) / (x1 - x0) * (right - left)
    ys = bottom - (np.clip(y_values, y0, y1) - y0) / (y1 - y0) * (bottom - top)
    valid = np.isfinite(xs) & np.isfinite(ys)
    return [(float(x), float(y)) for x, y in zip(xs[valid], ys[valid], strict=False)]


def _draw_axis(
    draw,
    rect: tuple[int, int, int, int],
    title: str,
    time: np.ndarray,
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    ylim: tuple[float, float],
    xlim: tuple[float, float] = (0.0, float(VIDEO_SECONDS)),
) -> None:
    left, top, right, bottom = rect
    draw.rectangle((left, top, right, bottom), fill=(255, 255, 255), outline=(210, 218, 228), width=1)
    draw.text((left, top - 30), title, fill=(24, 34, 46), font=_font(17, bold=True))

    for i in range(5):
        x = left + i * (right - left) / 4
        y = top + i * (bottom - top) / 4
        draw.line((x, top, x, bottom), fill=(232, 236, 242), width=1)
        draw.line((left, y, right, y), fill=(232, 236, 242), width=1)

    for _label, values, color in series:
        points = _plot_points(time, values, rect, xlim, ylim)
        if len(points) > 1:
            draw.line(points, fill=color, width=3)

    y_ticks = np.linspace(ylim[0], ylim[1], 5)
    y_label_format = "{:.2f}" if abs(ylim[1] - ylim[0]) <= 1.0 else "{:.2g}"
    for i, value in enumerate(y_ticks):
        y = bottom - i * (bottom - top) / 4
        draw.text((8, y - 8), y_label_format.format(value), fill=(78, 90, 105), font=_font(12))

    for i, value in enumerate(np.linspace(xlim[0], xlim[1], 5)):
        x = left + i * (right - left) / 4
        draw.text((x - 10, bottom + 8), f"{value:.0f}", fill=(78, 90, 105), font=_font(12))

    legend_x = left
    legend_y = bottom + 34
    for label, _values, color in series:
        draw.line((legend_x, legend_y, legend_x + 26, legend_y), fill=color, width=4)
        draw.text((legend_x + 34, legend_y - 8), label, fill=(24, 34, 46), font=_font(13))
        legend_x += 150


def _draw_metric(draw, origin: tuple[int, int], label: str, value: str, color: tuple[int, int, int]) -> None:
    x, y = origin
    draw.text((x, y), label, fill=(91, 104, 120), font=_font(12, bold=True))
    draw.text((x, y + 17), value, fill=color, font=_font(22, bold=True))


def _hanging_panel_frame(time: list[float], tension: list[float], expected: float) -> np.ndarray:
    from PIL import Image, ImageDraw

    time_np = np.asarray(time, dtype=float)
    tension_np = np.asarray(tension, dtype=float)
    expected_np = np.full_like(tension_np, expected)
    error = float(tension_np[-1] - expected)

    image = Image.new("RGB", (VIDEO_PLOT_WIDTH, VIDEO_HEIGHT), (250, 252, 255))
    draw = ImageDraw.Draw(image)
    draw.text((26, 22), "Hanging Cable", fill=(18, 27, 39), font=_font(24, bold=True))
    draw.text((26, 54), f"t = {time_np[-1]:.2f}s", fill=(91, 104, 120), font=_font(14, bold=True))
    _draw_metric(draw, (26, 82), "tension", f"{tension_np[-1]:.3f} N", (37, 99, 166))
    _draw_metric(draw, (194, 82), "reference", f"{expected:.3f} N", (117, 143, 49))
    _draw_metric(draw, (342, 82), "error", f"{error:+.3f} N", (179, 93, 50))

    _draw_axis(
        draw,
        (58, 188, VIDEO_PLOT_WIDTH - 24, 615),
        "Tension [N]",
        time_np,
        [
            ("VBD", tension_np, (37, 99, 166)),
            ("reference", expected_np, (117, 143, 49)),
        ],
        (0.0, 16.0),
    )
    draw.text((58, 668), "error = VBD tension - analytic reference", fill=(91, 104, 120), font=_font(13))
    draw.text((VIDEO_PLOT_WIDTH - 112, 680), "time [s]", fill=(91, 104, 120), font=_font(13))
    return np.asarray(image, dtype=np.uint8)


def _pulley_panel_frame(
    time: list[float],
    tension_min: list[float],
    tension_max: list[float],
    ratio: list[float],
) -> np.ndarray:
    from PIL import Image, ImageDraw

    time_np = np.asarray(time, dtype=float)
    min_np = np.asarray(tension_min, dtype=float)
    max_np = np.asarray(tension_max, dtype=float)
    ratio_np = np.asarray(ratio, dtype=float)

    image = Image.new("RGB", (VIDEO_PLOT_WIDTH, VIDEO_HEIGHT), (250, 252, 255))
    draw = ImageDraw.Draw(image)
    draw.text((26, 22), "Single-Pulley Gradient", fill=(18, 27, 39), font=_font(24, bold=True))
    draw.text((26, 54), f"t = {time_np[-1]:.2f}s", fill=(91, 104, 120), font=_font(14, bold=True))
    _draw_metric(draw, (26, 82), "min", f"{min_np[-1]:.2f} N", (37, 99, 166))
    _draw_metric(draw, (176, 82), "max", f"{max_np[-1]:.2f} N", (179, 93, 50))
    _draw_metric(draw, (326, 82), "measured ratio", f"{ratio_np[-1]:.2f}", (117, 143, 49))

    _draw_axis(
        draw,
        (58, 185, VIDEO_PLOT_WIDTH - 24, 405),
        "Segment Tension Range [N]",
        time_np,
        [
            ("min", min_np, (37, 99, 166)),
            ("max", max_np, (179, 93, 50)),
        ],
        (18.0, 26.0),
    )
    _draw_axis(
        draw,
        (58, 480, VIDEO_PLOT_WIDTH - 24, 655),
        "Measured Max/Min Tension",
        time_np,
        [("max/min", ratio_np, (117, 143, 49))],
        (1.0, 1.35),
    )
    draw.text((VIDEO_PLOT_WIDTH - 112, 680), "time [s]", fill=(91, 104, 120), font=_font(13))
    return np.asarray(image, dtype=np.uint8)


def _capstan_panel_frame(
    time: list[float],
    middle_tension: list[float],
    reference: np.ndarray,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    time_np = np.asarray(time, dtype=float)
    middle_np = np.asarray(middle_tension, dtype=float)
    middle_ref = float(reference[1])
    error_np = middle_np - middle_ref

    image = Image.new("RGB", (VIDEO_PLOT_WIDTH, VIDEO_HEIGHT), (250, 252, 255))
    draw = ImageDraw.Draw(image)
    draw.text((26, 22), "Two-Pulley Cable", fill=(18, 27, 39), font=_font(24, bold=True))
    draw.text((26, 54), f"t = {time_np[-1]:.2f}s", fill=(91, 104, 120), font=_font(14, bold=True))
    _draw_metric(draw, (26, 82), "left load", f"{reference[0]:.1f} N", (37, 99, 166))
    _draw_metric(draw, (176, 82), "middle", f"{middle_np[-1]:.1f} N", (179, 93, 50))
    _draw_metric(draw, (326, 82), "right load", f"{reference[2]:.1f} N", (117, 143, 49))

    _draw_axis(
        draw,
        (58, 185, VIDEO_PLOT_WIDTH - 24, 405),
        "Middle-Span Tension [N]",
        time_np,
        [
            ("VBD", middle_np, (179, 93, 50)),
            ("reference", np.full_like(middle_np, middle_ref), (117, 143, 49)),
        ],
        (0.0, 40.0),
    )
    _draw_axis(
        draw,
        (58, 480, VIDEO_PLOT_WIDTH - 24, 655),
        "Middle Difference [N]",
        time_np,
        [("VBD - ref", error_np, (37, 99, 166))],
        (-12.0, 12.0),
    )
    draw.text((VIDEO_PLOT_WIDTH - 112, 680), "time [s]", fill=(91, 104, 120), font=_font(13))
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


def _write_viewer_frame(
    writer: subprocess.Popen,
    viewer: ViewerGL,
    frame_buffer: wp.array | None,
    panel: np.ndarray,
) -> wp.array:
    if writer.stdin is None:
        raise RuntimeError("ffmpeg stdin pipe was not opened")
    frame_buffer = viewer.get_frame(frame_buffer, render_ui=False)
    scene = frame_buffer.numpy()
    writer.stdin.write(np.ascontiguousarray(np.concatenate((scene, panel), axis=1)).tobytes())
    return frame_buffer


def _finish_video_writer(writer: subprocess.Popen) -> None:
    if writer.stdin is not None:
        writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")


def _render_hanging_video(output_path: Path, viewer: ViewerGL) -> None:
    device = "cuda:0"
    applied_downward_force = 5.0
    model, body, joint = build_hanging_cable(device)
    solver = newton.solvers.SolverVBD(model, iterations=LOAD_ITERATIONS)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    mass = float(model.body_mass.numpy()[body])
    expected_tension = mass * GRAVITY + applied_downward_force
    line_start = wp.zeros(1, dtype=wp.vec3, device=model.device)
    line_end = wp.zeros(1, dtype=wp.vec3, device=model.device)

    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(0.95, -1.65, -0.25), pitch=-4.0, yaw=-150.0)
    viewer.camera.look_at(wp.vec3(0.0, 0.0, -0.55))

    writer = _open_video_writer(output_path, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)
    frame_buffer = None
    frame_dt = 1.0 / VIDEO_FPS
    sim_dt = frame_dt / VIDEO_HANGING_SUBSTEPS
    video_time: list[float] = []
    video_tension: list[float] = []

    try:
        for frame in range(VIDEO_FPS * VIDEO_SECONDS):
            for _substep in range(VIDEO_HANGING_SUBSTEPS):
                state_0.clear_forces()
                wp.launch(
                    _set_body_linear_force,
                    dim=1,
                    inputs=[body, wp.vec3(0.0, 0.0, -applied_downward_force)],
                    outputs=[state_0.body_f],
                    device=model.device,
                )
                viewer.apply_forces(state_0)
                solver.step(state_0, state_1, control, None, sim_dt)
                state_0, state_1 = state_1, state_0

            tension = float(state_0.vbd.cable_tension.numpy()[joint])
            wp.launch(
                _update_cable_line,
                dim=1,
                inputs=[body, state_0.body_q],
                outputs=[line_start, line_end],
                device=model.device,
            )

            current_time = (frame + 1) * frame_dt
            video_time.append(current_time)
            video_tension.append(tension)
            panel = _hanging_panel_frame(video_time, video_tension, expected_tension)

            viewer.begin_frame((frame + 1) * frame_dt)
            viewer.log_state(state_0)
            viewer.log_lines("support_cable", line_start, line_end, (0.9, 0.78, 0.18), width=0.02)
            viewer.end_frame()
            frame_buffer = _write_viewer_frame(writer, viewer, frame_buffer, panel)
    finally:
        _finish_video_writer(writer)


def _render_pulley_video(output_path: Path, viewer: ViewerGL) -> None:
    device = "cuda:0"
    mu = 0.8
    model, bodies, joints, points = build_wrapped_cable(mu=mu, device=device)
    solver = newton.solvers.SolverVBD(model, iterations=WRAP_ITERATIONS, rigid_contact_hard=True)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    left_tangent = points[0] - points[1]
    left_tangent /= np.linalg.norm(left_tangent)
    right_tangent = points[-1] - points[-2]
    right_tangent /= np.linalg.norm(right_tangent)
    left_tangent_wp = wp.vec3(*left_tangent)
    right_tangent_wp = wp.vec3(*right_tangent)
    left_load = 20.0
    right_load = 20.0

    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(0.82, -1.55, 1.18), pitch=-4.0, yaw=-118.0)
    viewer.camera.look_at(wp.vec3(0.0, 0.0, 1.0))

    writer = _open_video_writer(output_path, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)
    frame_buffer = None
    frame_dt = 1.0 / VIDEO_FPS
    sim_dt = frame_dt / VIDEO_PULLEY_SUBSTEPS
    video_time: list[float] = []
    video_tension_min: list[float] = []
    video_tension_max: list[float] = []
    video_ratio: list[float] = []

    try:
        for frame in range(VIDEO_FPS * VIDEO_SECONDS):
            for _substep in range(VIDEO_PULLEY_SUBSTEPS):
                state_0.clear_forces()
                wp.launch(
                    _apply_endpoint_loads,
                    dim=1,
                    inputs=[
                        int(bodies[0]),
                        int(bodies[-1]),
                        left_load,
                        right_load,
                        left_tangent_wp,
                        right_tangent_wp,
                    ],
                    outputs=[state_0.body_f],
                    device=model.device,
                )
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0

            tensions = state_0.vbd.cable_tension.numpy()[joints]
            tension_min = float(np.min(tensions))
            tension_max = float(np.max(tensions))
            tension_ratio = tension_max / max(tension_min, 1.0e-12)
            current_time = (frame + 1) * frame_dt
            video_time.append(current_time)
            video_tension_min.append(tension_min)
            video_tension_max.append(tension_max)
            video_ratio.append(tension_ratio)
            panel = _pulley_panel_frame(video_time, video_tension_min, video_tension_max, video_ratio)

            viewer.begin_frame((frame + 1) * frame_dt)
            viewer.log_state(state_0)
            viewer.end_frame()
            frame_buffer = _write_viewer_frame(writer, viewer, frame_buffer, panel)
    finally:
        _finish_video_writer(writer)


def _render_capstan_video(output_path: Path, viewer: ViewerGL) -> None:
    device = "cuda:0"
    mu = 0.20
    model, bodies, joints, points, regions = build_capstan_pulley_system(mu=mu, device=device)
    solver = newton.solvers.SolverVBD(model, iterations=CAPSTAN_ITERATIONS, rigid_contact_hard=True)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    left_wrap_angle = 0.5 * np.pi
    right_wrap_angle = 0.5 * np.pi
    left_load = GRAVITY
    middle_reference = left_load * np.exp(mu * left_wrap_angle)
    right_load = middle_reference * np.exp(mu * right_wrap_angle)
    reference = np.asarray([left_load, middle_reference, right_load], dtype=float)

    left_tangent = points[0] - points[1]
    left_tangent /= np.linalg.norm(left_tangent)
    right_tangent = points[-1] - points[-2]
    right_tangent /= np.linalg.norm(right_tangent)
    left_tangent_wp = wp.vec3(*left_tangent)
    right_tangent_wp = wp.vec3(*right_tangent)

    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(0.0, -2.35, 1.05), pitch=-3.0, yaw=-90.0)
    viewer.camera.look_at(wp.vec3(0.0, 0.0, 0.88))

    writer = _open_video_writer(output_path, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)
    frame_buffer = None
    frame_dt = 1.0 / VIDEO_FPS
    sim_dt = frame_dt / VIDEO_CAPSTAN_SUBSTEPS
    video_time: list[float] = []
    video_middle_tension: list[float] = []

    try:
        for frame in range(VIDEO_FPS * VIDEO_SECONDS):
            for _substep in range(VIDEO_CAPSTAN_SUBSTEPS):
                state_0.clear_forces()
                wp.launch(
                    _apply_endpoint_loads,
                    dim=1,
                    inputs=[
                        int(bodies[0]),
                        int(bodies[-1]),
                        float(left_load),
                        float(right_load),
                        left_tangent_wp,
                        right_tangent_wp,
                    ],
                    outputs=[state_0.body_f],
                    device=model.device,
                )
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0

            tensions = state_0.vbd.cable_tension.numpy()[joints]
            middle_tension = float(np.mean(tensions[regions["middle"]]))
            current_time = (frame + 1) * frame_dt
            video_time.append(current_time)
            video_middle_tension.append(middle_tension)
            panel = _capstan_panel_frame(video_time, video_middle_tension, reference)

            viewer.begin_frame((frame + 1) * frame_dt)
            viewer.log_state(state_0)
            viewer.end_frame()
            frame_buffer = _write_viewer_frame(writer, viewer, frame_buffer, panel)
    finally:
        _finish_video_writer(writer)


def render_videos() -> None:
    if not wp.is_cuda_available():
        print("Skipping videos: ViewerGL video rendering requires CUDA.")
        return
    if shutil.which("ffmpeg") is None:
        print("Skipping videos: ffmpeg was not found.")
        return

    ctx = multiprocessing.get_context("spawn")
    for kind in ("hanging", "pulley", "capstan"):
        process = ctx.Process(target=_render_video_worker, args=(kind,))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{kind} video rendering failed with exit code {process.exitcode}.")


def _render_video_worker(kind: str) -> None:
    wp.init()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    viewer = ViewerGL(width=VIDEO_SCENE_WIDTH, height=VIDEO_HEIGHT, headless=True)
    try:
        viewer.show_ui = False
        if kind == "hanging":
            _render_hanging_video(ASSET_DIR / "cable_tension_hanging_newton.mp4", viewer)
        elif kind == "pulley":
            _render_pulley_video(ASSET_DIR / "cable_tension_pulley_newton.mp4", viewer)
        elif kind == "capstan":
            _render_capstan_video(ASSET_DIR / "cable_tension_capstan_newton.mp4", viewer)
        else:
            raise ValueError(f"Unknown video kind: {kind}")
    finally:
        viewer.close()


def main() -> None:
    wp.init()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    load_cases = {
        "T = mg": simulate_hanging_load(0.0),
        "T = mg + 5 N": simulate_hanging_load(5.0),
    }
    profiles = [simulate_wrapped_cable(mu) for mu in (0.0, 0.3, 0.8)]
    capstan = capstan_reference()
    capstan_sim = simulate_capstan_pulley_system()

    plot_load_cases(load_cases)
    plot_wrapped_profiles(profiles)
    plot_capstan_reference(capstan)
    plot_capstan_simulation(capstan_sim)
    render_videos()

    for name, data in load_cases.items():
        safe_name = name.lower().replace(" = ", "_").replace(" + ", "_plus_").replace(" ", "_")
        save_npz(safe_name, data)
    for profile in profiles:
        save_npz(f"single_pulley_mu_{profile['mu']:g}".replace(".", "p"), profile)
    save_npz("capstan_two_pulley", capstan_sim)

    summary = {
        "load_cases": {name: summarize_load_case(data) for name, data in load_cases.items()},
        "single_pulley_profiles": [
            {
                "mu": float(profile["mu"]),
                "min_tension_n": float(np.min(np.asarray(profile["tension"], dtype=float))),
                "max_tension_n": float(np.max(np.asarray(profile["tension"], dtype=float))),
                "measured_ratio": float(profile["measured_ratio"]),
                "capstan_ratio_bound": float(profile["capstan_ratio_bound"]),
            }
            for profile in profiles
        ],
        "capstan_reference": capstan,
        "capstan_two_pulley": {
            "mu": float(capstan_sim["mu"]),
            "left_wrap_angle": float(capstan_sim["left_wrap_angle"]),
            "right_wrap_angle": float(capstan_sim["right_wrap_angle"]),
            "labels": list(capstan_sim["labels"]),
            "reference_tension_n": np.asarray(capstan_sim["reference_tension"], dtype=float).tolist(),
            "diagnostic_tension_n": np.asarray(capstan_sim["diagnostic_tension"], dtype=float).tolist(),
            "middle_span_tension_n": float(capstan_sim["middle_span_tension"]),
            "relative_difference": (
                np.abs(
                    np.asarray(capstan_sim["diagnostic_tension"], dtype=float)
                    - np.asarray(capstan_sim["reference_tension"], dtype=float)
                )
                / np.asarray(capstan_sim["reference_tension"], dtype=float)
            ).tolist(),
        },
        "settings": {
            "load_iterations": LOAD_ITERATIONS,
            "load_dt": LOAD_DT,
            "load_steps": LOAD_STEPS,
            "wrap_iterations": WRAP_ITERATIONS,
            "wrap_dt": WRAP_DT,
            "wrap_steps": WRAP_STEPS,
            "capstan_iterations": CAPSTAN_ITERATIONS,
            "capstan_dt": CAPSTAN_DT,
            "capstan_steps": CAPSTAN_STEPS,
        },
    }
    with (DATA_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
