# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cable Cross-Slide Table
#
# Standalone cable-driven cross-slide table example. Two kinematic input
# pulleys drive a closed cable loop around five passive pulleys, pulling a
# two-axis slide/table carriage through cable contact and cable-end anchors.
#
# Run from the repository root:
#   uv run python newton/examples/cable/example_cable_cross_slide_table.py
#
# Headless test run:
#   uv run python newton/examples/cable/example_cable_cross_slide_table.py --viewer null --test
#
# Slower drive:
#   uv run python newton/examples/cable/example_cable_cross_slide_table.py --table-rect-period 24.0
#
# Slower headless test; use more frames so the longer period completes:
#   uv run python newton/examples/cable/example_cable_cross_slide_table.py --viewer null --test --table-rect-period 24.0 --num-frames 3000
#
# Finer cable discretization:
#   uv run python newton/examples/cable/example_cable_cross_slide_table.py --segment-length 0.010
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples


ATTACH_BASE = 0
ATTACH_SLIDE = 1
ATTACH_TABLE = 2

MOTOR_NONE = 0
MOTOR_INPUT_LEFT = 1
MOTOR_INPUT_RIGHT = 2

TABLE_RECT_HALF_X = 0.050
TABLE_RECT_HALF_Y = 0.060
TABLE_RECT_PERIOD = 16.0
CABLE_SEGMENT_LENGTH = 0.015
DEFAULT_SUBSTEPS = 5
DEFAULT_ITERATIONS = 5
TABLE_RECT_POINTS = (
    (-TABLE_RECT_HALF_X, -TABLE_RECT_HALF_Y),
    (TABLE_RECT_HALF_X, -TABLE_RECT_HALF_Y),
    (TABLE_RECT_HALF_X, TABLE_RECT_HALF_Y),
    (-TABLE_RECT_HALF_X, TABLE_RECT_HALF_Y),
)
TABLE_RECT_HIT_TOLERANCE = 0.025
TABLE_RECT_TEST_FRAMES = 2100
START_RAMP_DURATION = 1.2
TABLE_REPEAT_TOLERANCE = 5.0e-3
BODY_REPEAT_TOLERANCE = 5.0e-3
TABLE_TRACKING_TOLERANCE = 1.0e-2
JOINT_POSITION_GAP_TOLERANCE = 1.0e-3
JOINT_ANGULAR_GAP_TOLERANCE = 1.0e-3

CONTACT_KD = 0.0
CONTACT_MU = 1.0
SOFT_CONTACT_KE = 1.0e5
SOFT_JOINT_KE = 1.0e5
HARD_CONTACT_KE = 1.0e4
HARD_JOINT_KE = 1.0e6


def mode_default_contact_ke(contact_mode: str) -> float:
    return SOFT_CONTACT_KE if contact_mode == "soft" else HARD_CONTACT_KE


def mode_default_joint_ke(contact_mode: str) -> float:
    return SOFT_JOINT_KE if contact_mode == "soft" else HARD_JOINT_KE


@wp.kernel
def drive_input_pulleys(
    sim_time: wp.array[wp.float32],
    body_indices: wp.array[wp.int32],
    body_base_xforms: wp.array[wp.transform],
    body_motor: wp.array[wp.int32],
    pulley_radius: float,
    table_rect_period: float,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    tid = wp.tid()
    body = body_indices[tid]
    base_xform = body_base_xforms[tid]

    t = sim_time[0]
    ramp = wp.clamp(t / START_RAMP_DURATION, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    phase_time = t - wp.floor(t / table_rect_period) * table_rect_period
    side = 4.0 * phase_time / table_rect_period

    table_x = -TABLE_RECT_HALF_X
    table_y = -TABLE_RECT_HALF_Y
    if side < 1.0:
        table_x = -TABLE_RECT_HALF_X + 2.0 * TABLE_RECT_HALF_X * side
    elif side < 2.0:
        table_x = TABLE_RECT_HALF_X
        table_y = -TABLE_RECT_HALF_Y + 2.0 * TABLE_RECT_HALF_Y * (side - 1.0)
    elif side < 3.0:
        table_x = TABLE_RECT_HALF_X - 2.0 * TABLE_RECT_HALF_X * (side - 2.0)
        table_y = TABLE_RECT_HALF_Y
    else:
        table_y = TABLE_RECT_HALF_Y - 2.0 * TABLE_RECT_HALF_Y * (side - 3.0)

    target_x = ramp * table_x
    target_y = ramp * table_y

    # Convert desired world/table XY into the cable-drive command frame.
    # In this standalone model, a direct xy_table-style command traces
    # world (x, y) ~= (command_y, -command_x), so invert that mapping here.
    command_x = -target_y
    command_y = target_x

    q_left = (command_x + command_y) / pulley_radius
    q_right = (command_y - command_x) / pulley_radius

    p = wp.transform_get_translation(base_xform)
    q = wp.transform_get_rotation(base_xform)

    motor = body_motor[tid]
    if motor == MOTOR_INPUT_LEFT:
        q = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), q_left), q)
    elif motor == MOTOR_INPUT_RIGHT:
        q = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), q_right), q)

    xform = wp.transform(p, q)
    body_q0[body] = xform
    body_q1[body] = xform


@wp.kernel
def advance_time(sim_time: wp.array[wp.float32], dt: float):
    sim_time[0] = sim_time[0] + dt


@wp.kernel
def set_body_xforms(
    body_indices: wp.array[wp.int32],
    body_xforms: wp.array[wp.transform],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    tid = wp.tid()
    body = body_indices[tid]
    xform = body_xforms[tid]
    body_q0[body] = xform
    body_q1[body] = xform


def target_table_xy(t: float, table_rect_period: float = TABLE_RECT_PERIOD) -> tuple[float, float]:
    ramp = min(1.0, max(0.0, t / START_RAMP_DURATION))
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    phase_time = t - math.floor(t / table_rect_period) * table_rect_period
    side = 4.0 * phase_time / table_rect_period

    x = -TABLE_RECT_HALF_X
    y = -TABLE_RECT_HALF_Y
    if side < 1.0:
        x = -TABLE_RECT_HALF_X + 2.0 * TABLE_RECT_HALF_X * side
    elif side < 2.0:
        x = TABLE_RECT_HALF_X
        y = -TABLE_RECT_HALF_Y + 2.0 * TABLE_RECT_HALF_Y * (side - 1.0)
    elif side < 3.0:
        x = TABLE_RECT_HALF_X - 2.0 * TABLE_RECT_HALF_X * (side - 2.0)
        y = TABLE_RECT_HALF_Y
    else:
        y = TABLE_RECT_HALF_Y - 2.0 * TABLE_RECT_HALF_Y * (side - 3.0)

    return ramp * x, ramp * y


def drive_pulley_rotations(
    t: float,
    pulley_radius: float,
    table_rect_period: float = TABLE_RECT_PERIOD,
) -> tuple[float, float]:
    # Python mirror of drive_input_pulleys(), used only for viewer/debug logs.
    target_x, target_y = target_table_xy(t, table_rect_period)
    # See drive_input_pulleys(): command frame maps to world (y, -x).
    command_x = -target_y
    command_y = target_x
    return (command_x + command_y) / pulley_radius, (command_y - command_x) / pulley_radius


def _normalize_np(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1.0e-12:
        return v
    return v / n


def _nanmax_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else float("nan")


def _nanrms_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite * finite))) if len(finite) else float("nan")


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dtype=np.float64,
    )


def _quat_conj_np(q: np.ndarray) -> np.ndarray:
    return np.array((-q[0], -q[1], -q[2], q[3]), dtype=np.float64)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = q[:3]
    qw = q[3]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _transform_point_np(xform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return xform[:3] + _quat_rotate_np(_normalize_np(xform[3:]), point)


def _quat_angle_np(q: np.ndarray) -> float:
    q = _normalize_np(q)
    w = float(np.clip(abs(q[3]), -1.0, 1.0))
    return 2.0 * math.acos(w)


def _transform_multiply_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa = _normalize_np(a[3:])
    qb = _normalize_np(b[3:])
    p = a[:3] + _quat_rotate_np(qa, b[:3])
    q = _normalize_np(_quat_mul_np(qa, qb))
    return np.concatenate((p, q))


def _dim_color(color: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, c * scale)) for c in color)


def _make_body_kinematic(builder: newton.ModelBuilder, body: int):
    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)


def add_visual_bar(
    builder: newton.ModelBuilder,
    *,
    body: int,
    center: wp.vec3,
    half_extents: tuple[float, float, float],
    color: tuple[float, float, float],
    label: str,
    density: float = 350.0,
):
    # Visual-only guide/carriage bars; cable contact is handled by pulley shapes.
    cfg = newton.ModelBuilder.ShapeConfig(
        density=density,
        has_shape_collision=False,
        has_particle_collision=False,
    )
    return builder.add_shape_box(
        body=body,
        xform=wp.transform(center, wp.quat_identity()),
        hx=half_extents[0],
        hy=half_extents[1],
        hz=half_extents[2],
        cfg=cfg,
        color=color,
        label=label,
    )


def add_guided_pulley(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    axis: wp.vec3,
    sheave_diameter: float,
    cable_radius: float,
    *,
    body: int = -1,
    color: tuple[float, float, float] = (0.42, 0.45, 0.48),
    groove_width_scale: float = 3.0,
    flange_radius_scale: float = 3.0,
    flange_thickness_scale: float = 1.0,
    ke: float = 1.0e6,
    kd: float = 1.0e-1,
    mu: float = 0.0,
    density: float = 1000.0,
    label: str | None = None,
) -> tuple[int, int, int]:
    if sheave_diameter <= 0.0:
        raise ValueError("sheave_diameter must be positive")
    if cable_radius <= 0.0:
        raise ValueError("cable_radius must be positive")
    if float(wp.length(axis)) <= 1.0e-8:
        raise ValueError("axis must be non-zero")

    axis = wp.normalize(axis)
    q_axis = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis)

    sheave_radius = 0.5 * sheave_diameter
    groove_half_width = 0.5 * groove_width_scale * cable_radius
    flange_radius = sheave_radius + flange_radius_scale * cable_radius
    flange_half_thickness = 0.5 * flange_thickness_scale * cable_radius

    cfg = newton.ModelBuilder.ShapeConfig(density=density, ke=ke, kd=kd, mu=mu)
    flange_color = _dim_color(color, 0.68)

    sheave = builder.add_shape_cylinder(
        body=body,
        xform=wp.transform(center, q_axis),
        radius=sheave_radius,
        half_height=groove_half_width,
        cfg=cfg,
        color=color,
        label=f"{label}_sheave" if label else None,
    )
    flange_neg = builder.add_shape_cylinder(
        body=body,
        xform=wp.transform(center - axis * (groove_half_width + flange_half_thickness), q_axis),
        radius=flange_radius,
        half_height=flange_half_thickness,
        cfg=cfg,
        color=flange_color,
        label=f"{label}_flange_neg" if label else None,
    )
    flange_pos = builder.add_shape_cylinder(
        body=body,
        xform=wp.transform(center + axis * (groove_half_width + flange_half_thickness), q_axis),
        radius=flange_radius,
        half_height=flange_half_thickness,
        cfg=cfg,
        color=flange_color,
        label=f"{label}_flange_pos" if label else None,
    )
    return sheave, flange_neg, flange_pos


def add_pulley_rotation_dot(
    builder: newton.ModelBuilder,
    *,
    body: int,
    sheave_diameter: float,
    cable_radius: float,
    groove_width_scale: float,
    flange_thickness_scale: float,
    color: tuple[float, float, float] = (0.96, 0.92, 0.72),
    label: str | None = None,
) -> int:
    sheave_radius = 0.5 * sheave_diameter
    groove_half_width = 0.5 * groove_width_scale * cable_radius
    flange_half_thickness = 0.5 * flange_thickness_scale * cable_radius
    marker_radius = 0.75 * cable_radius
    marker_z = groove_half_width + 2.0 * flange_half_thickness + 0.35 * marker_radius
    marker_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
    )
    return builder.add_shape_sphere(
        body=body,
        xform=wp.transform(wp.vec3(0.78 * sheave_radius, 0.0, marker_z), wp.quat_identity()),
        radius=marker_radius,
        cfg=marker_cfg,
        color=color,
        label=f"{label}_rotation_dot" if label else None,
    )


def add_kinematic_guided_pulley(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    axis: wp.vec3,
    sheave_diameter: float,
    cable_radius: float,
    *,
    color: tuple[float, float, float],
    groove_width_scale: float,
    flange_radius_scale: float,
    flange_thickness_scale: float,
    ke: float,
    kd: float,
    mu: float,
    density: float,
    label: str,
) -> tuple[int, tuple[int, int, int, int]]:
    axis = wp.normalize(axis)
    body = builder.add_link(
        xform=wp.transform(center, wp.quat_identity()),
        is_kinematic=True,
        label=f"{label}_body",
    )
    shapes = add_guided_pulley(
        builder,
        center=wp.vec3(0.0, 0.0, 0.0),
        axis=axis,
        sheave_diameter=sheave_diameter,
        cable_radius=cable_radius,
        body=body,
        color=color,
        groove_width_scale=groove_width_scale,
        flange_radius_scale=flange_radius_scale,
        flange_thickness_scale=flange_thickness_scale,
        ke=ke,
        kd=kd,
        mu=mu,
        density=density,
        label=label,
    )
    marker = add_pulley_rotation_dot(
        builder,
        body=body,
        sheave_diameter=sheave_diameter,
        cable_radius=cable_radius,
        groove_width_scale=groove_width_scale,
        flange_thickness_scale=flange_thickness_scale,
        label=label,
    )
    _make_body_kinematic(builder, body)
    return body, (*shapes, marker)


def add_passive_guided_pulley(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    axis: wp.vec3,
    sheave_diameter: float,
    cable_radius: float,
    *,
    parent: int,
    color: tuple[float, float, float],
    groove_width_scale: float,
    flange_radius_scale: float,
    flange_thickness_scale: float,
    ke: float,
    kd: float,
    mu: float,
    density: float,
    axle_armature: float,
    axle_friction: float,
    label: str,
) -> tuple[int, int, tuple[int, int, int, int]]:
    axis = wp.normalize(axis)
    body = builder.add_link(xform=wp.transform(center, wp.quat_identity()), label=f"{label}_body")

    parent_pose = builder.body_q[parent] if parent >= 0 else wp.transform(center, wp.quat_identity())
    parent_position = wp.transform_get_translation(parent_pose)
    parent_rotation = wp.transform_get_rotation(parent_pose)
    parent_center = center if parent == -1 else wp.quat_rotate_inv(parent_rotation, center - parent_position)
    joint_axis = axis if parent == -1 else wp.quat_rotate_inv(parent_rotation, axis)

    joint = builder.add_joint_revolute(
        parent=parent,
        child=body,
        axis=joint_axis,
        parent_xform=wp.transform(parent_center, wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        armature=axle_armature,
        friction=axle_friction,
        label=f"{label}_free_axle",
    )
    shapes = add_guided_pulley(
        builder,
        center=wp.vec3(0.0, 0.0, 0.0),
        axis=axis,
        sheave_diameter=sheave_diameter,
        cable_radius=cable_radius,
        body=body,
        color=color,
        groove_width_scale=groove_width_scale,
        flange_radius_scale=flange_radius_scale,
        flange_thickness_scale=flange_thickness_scale,
        ke=ke,
        kd=kd,
        mu=mu,
        density=density,
        label=label,
    )
    marker = add_pulley_rotation_dot(
        builder,
        body=body,
        sheave_diameter=sheave_diameter,
        cable_radius=cable_radius,
        groove_width_scale=groove_width_scale,
        flange_thickness_scale=flange_thickness_scale,
        label=label,
    )
    return body, joint, (*shapes, marker)


def append_segment(points: list[wp.vec3], end: wp.vec3, segment_length: float):
    start = points[-1]
    length = float(wp.length(end - start))
    if length <= 1.0e-8:
        return
    count = max(1, int(math.ceil(length / segment_length)))
    for i in range(1, count + 1):
        u = float(i) / float(count)
        points.append(start * (1.0 - u) + end * u)


def append_arc_xy(
    points: list[wp.vec3],
    center: wp.vec3,
    radius: float,
    start_angle: float,
    end_angle: float,
    segment_length: float,
    *,
    direction: str,
):
    delta = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
    if direction == "cw" and delta > 0.0:
        delta -= 2.0 * math.pi
    elif direction == "ccw" and delta < 0.0:
        delta += 2.0 * math.pi

    arc_length = abs(delta) * radius
    count = max(3, int(math.ceil(arc_length / segment_length)))
    for i in range(count + 1):
        u = float(i) / float(count)
        angle = start_angle + delta * u
        point = wp.vec3(
            float(center[0]) + radius * math.cos(angle),
            float(center[1]) + radius * math.sin(angle),
            float(center[2]),
        )
        append_segment(points, point, segment_length)


def resample_equal_length_segments(points: list[wp.vec3], segment_length: float) -> tuple[list[wp.vec3], float]:
    if len(points) < 2:
        raise ValueError("points must contain at least two points")
    if segment_length <= 0.0:
        raise ValueError("segment_length must be positive")

    clean_points = [points[0]]
    distances = [0.0]
    total_length = 0.0
    for point in points[1:]:
        length = float(wp.length(point - clean_points[-1]))
        if length <= 1.0e-8:
            continue
        total_length += length
        clean_points.append(point)
        distances.append(total_length)

    if total_length <= 1.0e-8:
        raise ValueError("points must span a non-zero length")

    segment_count = max(2, int(math.ceil(total_length / segment_length)))
    equal_segment_length = total_length / float(segment_count)
    resampled = [clean_points[0]]

    point_index = 1
    for segment_index in range(1, segment_count):
        target_distance = equal_segment_length * float(segment_index)
        while point_index < len(clean_points) - 1 and distances[point_index] < target_distance:
            point_index += 1
        previous_distance = distances[point_index - 1]
        next_distance = distances[point_index]
        u = (target_distance - previous_distance) / (next_distance - previous_distance)
        resampled.append(clean_points[point_index - 1] * (1.0 - u) + clean_points[point_index] * u)

    resampled.append(clean_points[-1])
    return resampled, equal_segment_length


def create_cross_slide_cable_points(
    start: wp.vec3,
    pulley_centers: list[wp.vec3],
    pulley_radii: list[float],
    end: wp.vec3,
    cable_radius: float,
    segment_length: float,
    wrap_clearance_scale: float,
) -> list[wp.vec3]:
    """Create the closed-loop cable route through the seven pulley arcs."""
    pulley_arcs = (
        (0.0, 0.5 * math.pi, "ccw"),
        (-0.5 * math.pi, 0.5 * math.pi, "cw"),
        (-0.5 * math.pi, 0.0, "ccw"),
        (math.pi, 0.0, "cw"),
        (math.pi, -0.5 * math.pi, "ccw"),
        (0.5 * math.pi, -0.5 * math.pi, "cw"),
        (0.5 * math.pi, math.pi, "ccw"),
    )
    if len(pulley_centers) != len(pulley_arcs) or len(pulley_radii) != len(pulley_arcs):
        raise ValueError("cross-slide cable route expects seven pulleys")

    points = [start]
    wrap_clearance = wrap_clearance_scale * cable_radius
    # Green pulleys use their inner quadrants. Blue input pulleys and the
    # beige top pulley use the outside path.
    for center, radius, (start_angle, end_angle, direction) in zip(
        pulley_centers, pulley_radii, pulley_arcs, strict=True
    ):
        append_arc_xy(
            points,
            center,
            radius + wrap_clearance,
            start_angle,
            end_angle,
            segment_length,
            direction=direction,
        )
    append_segment(points, end, segment_length)
    return points


def filter_body_group_collisions(builder: newton.ModelBuilder, bodies: list[int]):
    for i, body_a in enumerate(bodies):
        for body_b in bodies[i + 1 :]:
            for shape_a in builder.body_shapes.get(body_a, []):
                for shape_b in builder.body_shapes.get(body_b, []):
                    builder.add_shape_collision_filter_pair(int(shape_a), int(shape_b))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = int(getattr(args, "substeps", DEFAULT_SUBSTEPS))
        sim_iterations = int(getattr(args, "iterations", DEFAULT_ITERATIONS))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.table_rect_period = float(getattr(args, "table_rect_period", TABLE_RECT_PERIOD))
        if self.table_rect_period <= 0.0:
            raise ValueError("--table-rect-period must be positive")

        contact_mode = getattr(args, "contact_mode", "soft")
        contact_ke_arg = getattr(args, "ke", None)
        contact_ke = float(mode_default_contact_ke(contact_mode) if contact_ke_arg is None else contact_ke_arg)
        contact_kd = float(getattr(args, "kd", CONTACT_KD))
        contact_mu_arg = getattr(args, "mu", None)
        contact_mu = float(CONTACT_MU if contact_mu_arg is None else contact_mu_arg)
        cable_stretch_ke = float(getattr(args, "cable_stretch_ke", 1.0e5))
        cable_stretch_kd = float(getattr(args, "cable_stretch_kd", 0.0))
        cable_bend_ke = float(getattr(args, "cable_bend_ke", 5.0e-5))
        cable_bend_kd = float(getattr(args, "cable_bend_kd", 1.0e-2))
        segment_length = float(getattr(args, "segment_length", CABLE_SEGMENT_LENGTH))
        if segment_length <= 0.0:
            raise ValueError("--segment-length must be positive")

        cable_radius = 0.003
        input_pulley_radius = 0.025
        green_sheave_radius = 0.015
        beige_sheave_radius = 0.025
        wrap_clearance_scale = float(getattr(args, "wrap_clearance_scale", 1.0))

        base_z = 0.006
        slide_z = 0.014
        table_z = 0.022
        pulley_z = 0.046

        blue = (0.12, 0.34, 0.76)
        green = (0.12, 0.58, 0.28)
        beige = (0.74, 0.63, 0.45)

        table_origin = wp.vec3(0.0, 0.0, table_z)
        self.left_anchor_local = wp.vec3(-0.028, -0.21, pulley_z - table_z)
        self.right_anchor_local = wp.vec3(0.028, -0.21, pulley_z - table_z)
        left_anchor_world = table_origin + self.left_anchor_local
        right_anchor_world = table_origin + self.right_anchor_local

        pulley_specs = [
            ("green_lower_left", ATTACH_SLIDE, MOTOR_NONE, green, wp.vec3(-0.045, -0.045, pulley_z), green_sheave_radius),
            ("blue_input_left", ATTACH_BASE, MOTOR_INPUT_LEFT, blue, wp.vec3(-0.19, 0.0, pulley_z), input_pulley_radius),
            ("green_upper_left", ATTACH_SLIDE, MOTOR_NONE, green, wp.vec3(-0.045, 0.045, pulley_z), green_sheave_radius),
            ("beige_top", ATTACH_TABLE, MOTOR_NONE, beige, wp.vec3(0.0, 0.19, pulley_z), beige_sheave_radius),
            ("green_upper_right", ATTACH_SLIDE, MOTOR_NONE, green, wp.vec3(0.045, 0.045, pulley_z), green_sheave_radius),
            ("blue_input_right", ATTACH_BASE, MOTOR_INPUT_RIGHT, blue, wp.vec3(0.19, 0.0, pulley_z), input_pulley_radius),
            ("green_lower_right", ATTACH_SLIDE, MOTOR_NONE, green, wp.vec3(0.045, -0.045, pulley_z), green_sheave_radius),
        ]
        pulley_centers = [spec[4] for spec in pulley_specs]
        pulley_radii = [spec[5] for spec in pulley_specs]

        route_points = create_cross_slide_cable_points(
            start=left_anchor_world,
            pulley_centers=pulley_centers,
            pulley_radii=pulley_radii,
            end=right_anchor_world,
            cable_radius=cable_radius,
            segment_length=segment_length,
            wrap_clearance_scale=wrap_clearance_scale,
        )
        cable_points, cable_segment_length = resample_equal_length_segments(route_points, segment_length)
        cable_quats = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
        cable_segment_count = len(cable_points) - 1
        # Build the rod straight so model.body_q is the cable's structural rest
        # shape. The wrapped route below is the initial state, so nonzero bend
        # stiffness starts the cable with physical bending strain.
        straight_points, straight_quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=left_anchor_world,
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=cable_segment_count * cable_segment_length,
            num_segments=cable_segment_count,
        )

        builder = newton.ModelBuilder()
        builder.rigid_gap = 5.0 * cable_radius
        builder.default_shape_cfg.ke = contact_ke
        builder.default_shape_cfg.kd = contact_kd
        builder.default_shape_cfg.mu = contact_mu

        add_visual_bar(
            builder,
            body=-1,
            center=wp.vec3(0.0, 0.0, base_z),
            half_extents=(0.205, 0.025, 0.006),
            color=blue,
            label="fixed_blue_base",
            density=0.0,
        )

        cable_cfg = builder.default_shape_cfg.copy()
        cable_cfg.density = 20.0
        cable_cfg.gap = 5.0 * cable_radius

        self.cable_bodies, cable_joints = builder.add_rod(
            positions=straight_points,
            quaternions=straight_quats,
            radius=cable_radius,
            cfg=cable_cfg,
            stretch_stiffness=cable_stretch_ke,
            stretch_damping=cable_stretch_kd,
            bend_stiffness=cable_bend_ke,
            bend_damping=cable_bend_kd,
            wrap_in_articulation=False,
            label="cable",
        )
        initial_cable_xforms = [
            wp.transform(cable_points[i], cable_quats[i]) for i in range(len(self.cable_bodies))
        ]
        filter_body_group_collisions(builder, self.cable_bodies)

        slide_origin = wp.vec3(0.0, 0.0, slide_z)
        table_origin_world = wp.vec3(0.0, 0.0, table_z)
        self.slide_body = builder.add_link(
            xform=wp.transform(slide_origin, wp.quat_identity()),
            label="green_slide_body",
        )
        self.table_body = builder.add_link(
            xform=wp.transform(table_origin_world, wp.quat_identity()),
            label="beige_table_body",
        )
        self.slide_origin_xy = (float(slide_origin[0]), float(slide_origin[1]))
        self.table_origin_xy = (float(table_origin_world[0]), float(table_origin_world[1]))
        self.table_rect_points = np.array(TABLE_RECT_POINTS, dtype=np.float32)
        self.table_rect_min_distances = np.full(len(TABLE_RECT_POINTS), np.inf, dtype=np.float32)
        add_visual_bar(
            builder,
            body=self.slide_body,
            center=wp.vec3(0.0, 0.0, 0.0),
            half_extents=(0.085, 0.052, 0.006),
            color=green,
            label="green_horizontal_carriage",
            density=45.0,
        )
        add_visual_bar(
            builder,
            body=self.table_body,
            center=wp.vec3(0.0, 0.0, 0.0),
            half_extents=(0.013, 0.215, 0.006),
            color=beige,
            label="beige_vertical_carriage",
            density=65.0,
        )

        joint_limit_ke = float(getattr(args, "joint_limit_ke", 2.0e3))
        joint_limit_kd = float(getattr(args, "joint_limit_kd", 1.0e-4))
        slide_joint = builder.add_joint_prismatic(
            parent=-1,
            child=self.slide_body,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(slide_origin, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            limit_lower=-0.07,
            limit_upper=0.07,
            limit_ke=joint_limit_ke,
            limit_kd=joint_limit_kd,
            friction=0.0,
            label="green_x_slide_axis",
        )
        table_joint = builder.add_joint_prismatic(
            parent=self.slide_body,
            child=self.table_body,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(table_origin_world - slide_origin, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            limit_lower=-0.08,
            limit_upper=0.08,
            limit_ke=joint_limit_ke,
            limit_kd=joint_limit_kd,
            friction=0.0,
            label="beige_y_table_axis",
        )
        articulation_joints = [slide_joint, table_joint]

        self.pulley_bodies: list[int] = []
        for i, (label, attach, _motor, color, center, sheave_radius) in enumerate(pulley_specs, start=1):
            pulley_kwargs = {
                "center": center,
                "axis": wp.vec3(0.0, 0.0, 1.0),
                "sheave_diameter": 2.0 * sheave_radius,
                "cable_radius": cable_radius,
                "color": color,
                "groove_width_scale": 3.1,
                "flange_radius_scale": 3.2,
                "flange_thickness_scale": 1.2,
                "ke": contact_ke,
                "kd": contact_kd,
                "mu": contact_mu,
                "density": 220.0,
                "label": f"cross_slide_{i}_{label}",
            }
            if attach == ATTACH_BASE:
                pulley_body, _ = add_kinematic_guided_pulley(builder, **pulley_kwargs)
            else:
                parent = self.slide_body if attach == ATTACH_SLIDE else self.table_body
                pulley_body, pulley_joint, _ = add_passive_guided_pulley(
                    builder,
                    parent=parent,
                    axle_armature=1.0e-4,
                    axle_friction=0.0,
                    **pulley_kwargs,
                )
                articulation_joints.append(pulley_joint)
            self.pulley_bodies.append(pulley_body)
        self.passive_pulley_bodies = [
            body for body, spec in zip(self.pulley_bodies, pulley_specs, strict=True) if spec[2] == MOTOR_NONE
        ]

        self.input_pulley_radius = input_pulley_radius
        kinematic_body_indices: list[int] = []
        kinematic_body_motor: list[int] = []
        kinematic_body_base_xforms: list[wp.transform] = []
        # Only the two blue input pulleys are kinematic motors; passive pulleys
        # rotate through their joints when the cable pushes on them.
        for body, spec in zip(self.pulley_bodies, pulley_specs, strict=True):
            motor = spec[2]
            if motor == MOTOR_NONE:
                continue
            kinematic_body_indices.append(body)
            kinematic_body_motor.append(motor)
            kinematic_body_base_xforms.append(builder.body_q[body])

        first_cable_body = self.cable_bodies[0]
        last_cable_body = self.cable_bodies[-1]
        anchor_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        builder.add_shape_sphere(
            body=first_cable_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            radius=1.6 * cable_radius,
            cfg=anchor_cfg,
            color=beige,
            label="visual_cable_end_left",
        )
        builder.add_shape_sphere(
            body=last_cable_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, cable_segment_length), wp.quat_identity()),
            radius=1.6 * cable_radius,
            cfg=anchor_cfg,
            color=beige,
            label="visual_cable_end_right",
        )

        left_anchor_joint = builder.add_joint_ball(
            parent=self.table_body,
            child=first_cable_body,
            parent_xform=wp.transform(self.left_anchor_local, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            armature=1.0e-5,
            friction=0.0,
            label="left_bottom_cable_fix",
        )
        right_anchor_joint = builder.add_joint_ball(
            parent=self.table_body,
            child=last_cable_body,
            parent_xform=wp.transform(self.right_anchor_local, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, cable_segment_length), wp.quat_identity()),
            armature=1.0e-5,
            friction=0.0,
            label="right_bottom_cable_fix_loop",
        )
        self.structural_joint_indices = [*articulation_joints, left_anchor_joint, right_anchor_joint]
        self.cable_joint_indices = cable_joints
        # The right anchor closes the cable loop. Keep it outside the
        # articulation because articulations must be tree-structured.
        builder.add_articulation(
            [*cable_joints, *articulation_joints, left_anchor_joint],
            label="cable_cross_slide_table",
        )

        builder.add_ground_plane()
        builder.color(balance_colors=False)

        sim_device = wp.get_device(args.device) if args.device else None
        self.model = builder.finalize(device=sim_device)
        self.model.set_gravity((0.0, 0.0, 0.0))
        self._joint_parent_np = self.model.joint_parent.numpy()
        self._joint_child_np = self.model.joint_child.numpy()
        self._joint_type_np = self.model.joint_type.numpy()
        self._joint_x_p_np = self.model.joint_X_p.numpy()
        self._joint_x_c_np = self.model.joint_X_c.numpy()
        self._joint_qd_start_np = self.model.joint_qd_start.numpy()
        self._joint_axis_np = self.model.joint_axis.numpy()
        self._shape_body_np = self.model.shape_body.numpy()
        self._body_com_np = self.model.body_com.numpy()

        rigid_contact_hard = contact_mode in ("hard", "hard-history")
        rigid_contact_history = contact_mode == "hard-history"
        joints_mode = getattr(args, "joints_mode", "hard")
        joint_linear_ke_arg = getattr(args, "joint_linear_ke", None)
        joint_linear_ke = float(
            mode_default_joint_ke(contact_mode) if joint_linear_ke_arg is None else joint_linear_ke_arg
        )
        joint_angular_ke = getattr(args, "joint_angular_ke", None)
        if joint_angular_ke is None:
            joint_angular_ke = joint_linear_ke
        else:
            joint_angular_ke = float(joint_angular_ke)

        # Create contacts before SolverVBD so the pipeline's contact-capacity
        # estimate is visible to VBD during its own preallocation.
        if rigid_contact_history:
            pipeline = newton.CollisionPipeline(self.model, broad_phase="explicit", contact_matching="latest")
            self.contacts = self.model.contacts(collision_pipeline=pipeline)
        else:
            self.contacts = self.model.contacts()

        print(
            f"[cable-cross-slide] segments={cable_segment_count} contact_mode={contact_mode} "
            f"joints_mode={joints_mode} period={self.table_rect_period:.3f}s "
            f"segment_length={segment_length:.4f}m ke={contact_ke:.3e} mu={contact_mu:.3f} "
            f"joint_ke=({joint_linear_ke:.3e},{joint_angular_ke:.3e})"
        )
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=sim_iterations,
            friction_epsilon=float(getattr(args, "friction_epsilon", 1.0e-2)),
            rigid_body_contact_buffer_size=256,
            rigid_contact_hard=rigid_contact_hard,
            rigid_contact_history=rigid_contact_history,
            rigid_joint_linear_ke=joint_linear_ke,
            rigid_joint_angular_ke=joint_angular_ke,
            rigid_joint_linear_kd=float(getattr(args, "joint_linear_kd", 0.0)),
            rigid_joint_angular_kd=float(getattr(args, "joint_angular_kd", 0.0)),
            rigid_avbd_beta=float(getattr(args, "rigid_avbd_beta", 0.0)),
            rigid_avbd_contact_alpha=getattr(args, "rigid_avbd_contact_alpha", None),
            rigid_avbd_gamma=float(getattr(args, "rigid_avbd_gamma", 0.999)),
        )
        # Keep contact history warm-starting available, but disable sticky
        # contact anchoring/deadzones so this transport test measures plain
        # cable-pulley contact/friction behavior.
        self.solver.rigid_contact_stick_motion_eps = 0.0
        self.solver.rigid_contact_stick_freeze_translation_eps = 0.0
        self.solver.rigid_contact_stick_freeze_angular_eps = 0.0
        if joints_mode == "soft":
            for j in articulation_joints:
                self.solver.set_joint_constraint_mode(j, hard=False)
            self.solver.set_joint_constraint_mode(left_anchor_joint, hard=False)
            self.solver.set_joint_constraint_mode(right_anchor_joint, hard=False)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        cable_body_indices = wp.array(self.cable_bodies, dtype=wp.int32, device=self.model.device)
        cable_body_xforms = wp.array(initial_cable_xforms, dtype=wp.transform, device=self.model.device)
        wp.launch(
            set_body_xforms,
            dim=cable_body_indices.shape[0],
            inputs=[cable_body_indices, cable_body_xforms, self.state_0.body_q, self.state_1.body_q],
            device=self.model.device,
        )
        # The wrapped cable pose is the initial condition, not a one-frame
        # teleport. VBD still uses model.body_q as the straight cable rest pose,
        # but its previous-pose buffer must match the active state to avoid a
        # fake first-step velocity.
        self.solver.body_q_prev = wp.clone(self.state_0.body_q, device=self.solver.device)

        self.kinematic_body_indices = wp.array(kinematic_body_indices, dtype=wp.int32, device=self.model.device)
        self.kinematic_body_base_xforms = wp.array(
            kinematic_body_base_xforms, dtype=wp.transform, device=self.model.device
        )
        self.kinematic_body_motor = wp.array(kinematic_body_motor, dtype=wp.int32, device=self.model.device)
        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.0, 0.0, 0.5), pitch=-90.0, yaw=90.0)

        self._frame_idx = 0
        self._table_x_min = float("inf")
        self._table_x_max = float("-inf")
        self._table_y_min = float("inf")
        self._table_y_max = float("-inf")
        self.contact_debug_metrics = bool(getattr(args, "contact_debug_metrics", False))
        self._init_metrics()

    def _init_metrics(self):
        self.metric_body_indices = [
            self.table_body,
            self.slide_body,
            *self.passive_pulley_bodies,
            self.cable_bodies[0],
            self.cable_bodies[-1],
        ]
        self.metric_body_labels = [self.model.body_label[int(i)] for i in self.metric_body_indices]
        self._metrics: dict[str, list] = {
            "time": [],
            "table_xy": [],
            "target_xy": [],
            "repeat_body_pos": [],
            "joint_pos_gap": [],
            "joint_ang_gap": [],
            "tracking_error": [],
            "slide_limit_violation": [],
            "table_limit_violation": [],
            "cable_speed_max": [],
            "cable_omega_max": [],
            "passive_omega_abs_max": [],
            "contact_count": [],
            "contact_penetration_max": [],
            "contact_penetration_rms": [],
            "contact_tangent_slip_speed_max": [],
            "contact_tangent_slip_speed_rms": [],
        }
        self._metric_worst_joint_pos_label = ""
        self._metric_worst_joint_ang_label = ""
        self._metric_worst_joint_pos_gap = 0.0
        self._metric_worst_joint_ang_gap = 0.0

    def _joint_world_frames(self, body_q: np.ndarray, joint_index: int) -> tuple[np.ndarray, np.ndarray]:
        world = np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float64)
        parent = int(self._joint_parent_np[joint_index])
        child = int(self._joint_child_np[joint_index])
        parent_pose = body_q[parent].astype(np.float64) if parent >= 0 else world
        child_pose = body_q[child].astype(np.float64)
        parent_frame = _transform_multiply_np(parent_pose, self._joint_x_p_np[joint_index].astype(np.float64))
        child_frame = _transform_multiply_np(child_pose, self._joint_x_c_np[joint_index].astype(np.float64))
        return parent_frame, child_frame

    def _compute_structural_joint_errors(self, body_q: np.ndarray) -> tuple[float, float, str, str]:
        max_pos_gap = 0.0
        max_ang_gap = 0.0
        max_pos_label = ""
        max_ang_label = ""

        for joint_index in self.structural_joint_indices:
            joint_type = int(self._joint_type_np[joint_index])
            parent_frame, child_frame = self._joint_world_frames(body_q, int(joint_index))
            delta = child_frame[:3] - parent_frame[:3]
            pos_gap = float(np.linalg.norm(delta))
            ang_gap = 0.0

            qd_start = int(self._joint_qd_start_np[joint_index])
            axis = np.array((1.0, 0.0, 0.0), dtype=np.float64)
            if 0 <= qd_start < len(self._joint_axis_np):
                axis = _normalize_np(self._joint_axis_np[qd_start].astype(np.float64))

            parent_q = _normalize_np(parent_frame[3:])
            child_q = _normalize_np(child_frame[3:])
            if joint_type == int(newton.JointType.PRISMATIC):
                axis_world = _normalize_np(_quat_rotate_np(parent_q, axis))
                off_axis = delta - np.dot(delta, axis_world) * axis_world
                pos_gap = float(np.linalg.norm(off_axis))
                ang_gap = _quat_angle_np(_quat_mul_np(_quat_conj_np(parent_q), child_q))
            elif joint_type == int(newton.JointType.REVOLUTE):
                axis_parent = _normalize_np(_quat_rotate_np(parent_q, axis))
                axis_child = _normalize_np(_quat_rotate_np(child_q, axis))
                axis_dot = float(np.clip(np.dot(axis_parent, axis_child), -1.0, 1.0))
                ang_gap = math.acos(axis_dot)
            elif joint_type == int(newton.JointType.BALL):
                ang_gap = 0.0
            elif joint_type == int(newton.JointType.FIXED):
                ang_gap = _quat_angle_np(_quat_mul_np(_quat_conj_np(parent_q), child_q))

            label = self.model.joint_label[int(joint_index)]
            if pos_gap > max_pos_gap:
                max_pos_gap = pos_gap
                max_pos_label = label
            if ang_gap > max_ang_gap:
                max_ang_gap = ang_gap
                max_ang_label = label

        return max_pos_gap, max_ang_gap, max_pos_label, max_ang_label

    def _body_point_velocity(self, body_q: np.ndarray, body_qd: np.ndarray, body: int, point_world: np.ndarray):
        if body < 0:
            return np.zeros(3, dtype=np.float64)

        pose = body_q[body].astype(np.float64)
        q = _normalize_np(pose[3:])
        com_world = _transform_point_np(pose, self._body_com_np[body].astype(np.float64))
        linear_velocity = body_qd[body, 0:3].astype(np.float64)
        angular_velocity = body_qd[body, 3:6].astype(np.float64)
        return linear_velocity + np.cross(angular_velocity, point_world - com_world)

    def _compute_contact_metrics(self, body_q: np.ndarray, body_qd: np.ndarray, contact_count: int):
        if contact_count <= 0:
            return 0.0, 0.0, 0.0, 0.0

        shape0 = self.contacts.rigid_contact_shape0.numpy()[:contact_count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:contact_count]
        point0 = self.contacts.rigid_contact_point0.numpy()[:contact_count]
        point1 = self.contacts.rigid_contact_point1.numpy()[:contact_count]
        normal = self.contacts.rigid_contact_normal.numpy()[:contact_count]
        margin0 = self.contacts.rigid_contact_margin0.numpy()[:contact_count]
        margin1 = self.contacts.rigid_contact_margin1.numpy()[:contact_count]

        penetrations = []
        tangent_slip_speeds = []
        for i in range(contact_count):
            s0 = int(shape0[i])
            s1 = int(shape1[i])
            b0 = int(self._shape_body_np[s0]) if s0 >= 0 else -1
            b1 = int(self._shape_body_np[s1]) if s1 >= 0 else -1

            p0_local = point0[i].astype(np.float64)
            p1_local = point1[i].astype(np.float64)
            p0_world = _transform_point_np(body_q[b0].astype(np.float64), p0_local) if b0 >= 0 else p0_local
            p1_world = _transform_point_np(body_q[b1].astype(np.float64), p1_local) if b1 >= 0 else p1_local
            n = _normalize_np(normal[i].astype(np.float64))

            # Same normal constraint value used by VBD: positive means overlap.
            thickness = float(margin0[i] + margin1[i])
            penetration = max(0.0, thickness - float(np.dot(n, p1_world - p0_world)))
            penetrations.append(penetration)

            v0 = self._body_point_velocity(body_q, body_qd, b0, p0_world)
            v1 = self._body_point_velocity(body_q, body_qd, b1, p1_world)
            relative_velocity = v1 - v0
            tangent_velocity = relative_velocity - np.dot(relative_velocity, n) * n
            tangent_slip_speeds.append(float(np.linalg.norm(tangent_velocity)))

        penetration_np = np.asarray(penetrations, dtype=np.float64)
        tangent_slip_np = np.asarray(tangent_slip_speeds, dtype=np.float64)
        return (
            float(np.max(penetration_np)),
            float(np.sqrt(np.mean(penetration_np * penetration_np))),
            float(np.max(tangent_slip_np)),
            float(np.sqrt(np.mean(tangent_slip_np * tangent_slip_np))),
        )

    def _record_metrics(self, body_q: np.ndarray, body_qd: np.ndarray, table_x: float, table_y: float):
        target_xy = np.array(target_table_xy(self.sim_time, self.table_rect_period), dtype=np.float64)
        table_xy = np.array((table_x, table_y), dtype=np.float64)
        joint_pos_gap, joint_ang_gap, joint_pos_label, joint_ang_label = self._compute_structural_joint_errors(body_q)

        if joint_pos_gap > self._metric_worst_joint_pos_gap:
            self._metric_worst_joint_pos_gap = joint_pos_gap
            self._metric_worst_joint_pos_label = joint_pos_label
        if joint_ang_gap > self._metric_worst_joint_ang_gap:
            self._metric_worst_joint_ang_gap = joint_ang_gap
            self._metric_worst_joint_ang_label = joint_ang_label

        cable_idx = np.asarray(self.cable_bodies, dtype=np.int32)
        cable_qd = body_qd[cable_idx]
        passive_qd = body_qd[np.asarray(self.passive_pulley_bodies, dtype=np.int32)]
        # SolverVBD.step() updates contact counts/state, but it does not fill
        # rigid_contact_force. Use SolverVBD.collect_rigid_contact_forces() with
        # a saved body_q_prev snapshot for force diagnostics.
        contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        contact_penetration_max = float("nan")
        contact_penetration_rms = float("nan")
        contact_tangent_slip_speed_max = float("nan")
        contact_tangent_slip_speed_rms = float("nan")
        if self.contact_debug_metrics:
            (
                contact_penetration_max,
                contact_penetration_rms,
                contact_tangent_slip_speed_max,
                contact_tangent_slip_speed_rms,
            ) = self._compute_contact_metrics(body_q, body_qd, contact_count)

        self._metrics["time"].append(float(self.sim_time))
        self._metrics["table_xy"].append(table_xy)
        self._metrics["target_xy"].append(target_xy)
        self._metrics["repeat_body_pos"].append(body_q[np.asarray(self.metric_body_indices, dtype=np.int32), 0:3].copy())
        self._metrics["joint_pos_gap"].append(float(joint_pos_gap))
        self._metrics["joint_ang_gap"].append(float(joint_ang_gap))
        self._metrics["tracking_error"].append(float(np.linalg.norm(table_xy - target_xy)))
        slide_x = float(body_q[self.slide_body, 0]) - self.slide_origin_xy[0]
        table_y_rel = float(body_q[self.table_body, 1]) - self.table_origin_xy[1]
        self._metrics["slide_limit_violation"].append(max(0.0, abs(slide_x) - 0.07))
        self._metrics["table_limit_violation"].append(max(0.0, abs(table_y_rel) - 0.08))
        self._metrics["cable_speed_max"].append(float(np.max(np.linalg.norm(cable_qd[:, 0:3], axis=1))))
        self._metrics["cable_omega_max"].append(float(np.max(np.linalg.norm(cable_qd[:, 3:6], axis=1))))
        self._metrics["passive_omega_abs_max"].append(float(np.max(np.abs(passive_qd[:, 5]))))
        self._metrics["contact_count"].append(contact_count)
        self._metrics["contact_penetration_max"].append(contact_penetration_max)
        self._metrics["contact_penetration_rms"].append(contact_penetration_rms)
        self._metrics["contact_tangent_slip_speed_max"].append(contact_tangent_slip_speed_max)
        self._metrics["contact_tangent_slip_speed_rms"].append(contact_tangent_slip_speed_rms)

    def get_metric_summary(self) -> dict[str, float | int | str | list[float]]:
        if len(self._metrics["time"]) == 0:
            return {}

        time = np.asarray(self._metrics["time"], dtype=np.float64)
        table_xy = np.asarray(self._metrics["table_xy"], dtype=np.float64)
        target_xy = np.asarray(self._metrics["target_xy"], dtype=np.float64)
        repeat_body_pos = np.asarray(self._metrics["repeat_body_pos"], dtype=np.float64)
        tracking = np.asarray(self._metrics["tracking_error"], dtype=np.float64)
        joint_pos_gap = np.asarray(self._metrics["joint_pos_gap"], dtype=np.float64)
        joint_ang_gap = np.asarray(self._metrics["joint_ang_gap"], dtype=np.float64)
        contact_penetration_max = np.asarray(self._metrics["contact_penetration_max"], dtype=np.float64)
        contact_penetration_rms = np.asarray(self._metrics["contact_penetration_rms"], dtype=np.float64)
        contact_tangent_slip_speed_max = np.asarray(
            self._metrics["contact_tangent_slip_speed_max"], dtype=np.float64
        )
        contact_tangent_slip_speed_rms = np.asarray(
            self._metrics["contact_tangent_slip_speed_rms"], dtype=np.float64
        )
        after_ramp = time >= START_RAMP_DURATION

        cycle_frames = int(round(self.table_rect_period / self.frame_dt))
        if len(time) > cycle_frames:
            idx = np.arange(0, len(time) - cycle_frames, dtype=np.int32)
            phase = np.mod(time[idx], self.table_rect_period)
            repeat_idx = idx[phase >= START_RAMP_DURATION]
        else:
            repeat_idx = np.zeros(0, dtype=np.int32)

        table_repeat_rms = float("nan")
        table_repeat_max = float("nan")
        body_repeat_rms = float("nan")
        body_repeat_max = float("nan")
        body_repeat_worst = ""
        body_repeat_samples = int(len(repeat_idx))
        if len(repeat_idx) > 0:
            table_repeat = table_xy[repeat_idx + cycle_frames] - table_xy[repeat_idx]
            table_repeat_norm = np.linalg.norm(table_repeat, axis=1)
            table_repeat_rms = float(np.sqrt(np.mean(table_repeat_norm * table_repeat_norm)))
            table_repeat_max = float(np.max(table_repeat_norm))

            body_repeat = repeat_body_pos[repeat_idx + cycle_frames] - repeat_body_pos[repeat_idx]
            body_repeat_norm = np.linalg.norm(body_repeat, axis=2)
            body_repeat_rms = float(np.sqrt(np.mean(body_repeat_norm * body_repeat_norm)))
            body_repeat_max = float(np.max(body_repeat_norm))
            worst_body = int(np.argmax(np.max(body_repeat_norm, axis=0)))
            body_repeat_worst = self.metric_body_labels[worst_body]

        tracking_eval = tracking[after_ramp] if np.any(after_ramp) else tracking
        tracking_rms = float(np.sqrt(np.mean(tracking_eval * tracking_eval)))
        tracking_max = float(np.max(tracking_eval))
        contact_penetration_max_eval = (
            contact_penetration_max[after_ramp] if np.any(after_ramp) else contact_penetration_max
        )
        contact_penetration_rms_eval = (
            contact_penetration_rms[after_ramp] if np.any(after_ramp) else contact_penetration_rms
        )
        contact_tangent_slip_speed_max_eval = (
            contact_tangent_slip_speed_max[after_ramp] if np.any(after_ramp) else contact_tangent_slip_speed_max
        )
        contact_tangent_slip_speed_rms_eval = (
            contact_tangent_slip_speed_rms[after_ramp] if np.any(after_ramp) else contact_tangent_slip_speed_rms
        )
        corner_errors = self.table_rect_min_distances.astype(np.float64)

        return {
            "frames": int(len(time)),
            "duration_s": float(time[-1]) if len(time) else 0.0,
            "repeat_samples": body_repeat_samples,
            "table_repeat_rms_m": table_repeat_rms,
            "table_repeat_max_m": table_repeat_max,
            "body_repeat_rms_m": body_repeat_rms,
            "body_repeat_max_m": body_repeat_max,
            "body_repeat_worst": body_repeat_worst,
            "table_tracking_rms_m": tracking_rms,
            "table_tracking_max_m": tracking_max,
            "table_x_min_m": float(np.min(table_xy[:, 0])),
            "table_x_max_m": float(np.max(table_xy[:, 0])),
            "table_y_min_m": float(np.min(table_xy[:, 1])),
            "table_y_max_m": float(np.max(table_xy[:, 1])),
            "corner_error_max_m": float(np.max(corner_errors)),
            "corner_errors_m": [float(x) for x in corner_errors],
            "joint_pos_gap_rms_m": float(np.sqrt(np.mean(joint_pos_gap * joint_pos_gap))),
            "joint_pos_gap_max_m": float(np.max(joint_pos_gap)),
            "joint_pos_gap_worst": self._metric_worst_joint_pos_label,
            "joint_ang_gap_rms_rad": float(np.sqrt(np.mean(joint_ang_gap * joint_ang_gap))),
            "joint_ang_gap_max_rad": float(np.max(joint_ang_gap)),
            "joint_ang_gap_worst": self._metric_worst_joint_ang_label,
            "slide_limit_violation_max_m": float(np.max(self._metrics["slide_limit_violation"])),
            "table_limit_violation_max_m": float(np.max(self._metrics["table_limit_violation"])),
            "cable_speed_max_mps": float(np.max(self._metrics["cable_speed_max"])),
            "cable_omega_max_radps": float(np.max(self._metrics["cable_omega_max"])),
            "passive_omega_abs_max_radps": float(np.max(self._metrics["passive_omega_abs_max"])),
            "contact_count_mean": float(np.mean(self._metrics["contact_count"])),
            "contact_count_max": int(np.max(self._metrics["contact_count"])),
            "contact_debug_metrics_enabled": bool(self.contact_debug_metrics),
            "contact_penetration_max_m": _nanmax_or_nan(contact_penetration_max_eval),
            "contact_penetration_rms_m": _nanrms_or_nan(contact_penetration_rms_eval),
            "contact_tangent_slip_speed_max_mps": _nanmax_or_nan(contact_tangent_slip_speed_max_eval),
            "contact_tangent_slip_speed_rms_mps": _nanrms_or_nan(contact_tangent_slip_speed_rms_eval),
            "finite": bool(np.all(np.isfinite(table_xy)) and np.all(np.isfinite(repeat_body_pos))),
        }

    def _update_table_bounds(self, table_x: float, table_y: float):
        if not np.isfinite(self._table_x_min):
            self._table_x_min = self._table_x_max = table_x
            self._table_y_min = self._table_y_max = table_y
        else:
            if table_x < self._table_x_min:
                self._table_x_min = table_x
            if table_x > self._table_x_max:
                self._table_x_max = table_x
            if table_y < self._table_y_min:
                self._table_y_min = table_y
            if table_y > self._table_y_max:
                self._table_y_max = table_y

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                drive_input_pulleys,
                dim=self.kinematic_body_indices.shape[0],
                inputs=[
                    self.sim_time_wp,
                    self.kinematic_body_indices,
                    self.kinematic_body_base_xforms,
                    self.kinematic_body_motor,
                    self.input_pulley_radius,
                    self.table_rect_period,
                    self.state_0.body_q,
                    self.state_1.body_q,
                ],
                device=self.model.device,
            )
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_time, dim=1, inputs=[self.sim_time_wp, self.sim_dt], device=self.model.device)
        self.sim_time += self.frame_dt

        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        table_x = float(body_q[self.table_body, 0]) - self.table_origin_xy[0]
        table_y = float(body_q[self.table_body, 1]) - self.table_origin_xy[1]
        table_xy = np.array((table_x, table_y), dtype=np.float32)
        table_rect_distances = np.linalg.norm(self.table_rect_points - table_xy, axis=1)
        self.table_rect_min_distances = np.minimum(self.table_rect_min_distances, table_rect_distances)
        self._update_table_bounds(table_x, table_y)
        self._record_metrics(body_q, body_qd, table_x, table_y)

        q_left, q_right = drive_pulley_rotations(
            self.sim_time,
            self.input_pulley_radius,
            self.table_rect_period,
        )
        self.viewer.log_scalar("Blue pulley 2 rotation [rad]", float(q_left))
        self.viewer.log_scalar("Blue pulley 6 rotation [rad]", float(q_right))
        self.viewer.log_scalar("Beige table X position [m]", table_x)
        self.viewer.log_scalar("Beige table Y position [m]", table_y)
        self._frame_idx += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.state_0.body_q is None:
            raise RuntimeError("Body state is not available.")

        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("NaN/Inf in body transforms.")
        cable_pos = body_q[[int(b) for b in self.cable_bodies], 0:3]
        if np.max(np.abs(cable_pos[:, 0])) > 0.5 or np.max(np.abs(cable_pos[:, 1])) > 0.5:
            raise ValueError("Cable moved outside the expected XY bounds.")
        if np.min(cable_pos[:, 2]) < -0.04 or np.max(cable_pos[:, 2]) > 0.12:
            raise ValueError("Cable left the expected Z range.")

        slide_pos = body_q[self.slide_body, 0:3]
        table_pos = body_q[self.table_body, 0:3]
        slide_x = float(slide_pos[0]) - self.slide_origin_xy[0]
        table_y = float(table_pos[1]) - self.table_origin_xy[1]
        joint_limit_tolerance = 0.005
        if not (-0.07 - joint_limit_tolerance <= slide_x <= 0.07 + joint_limit_tolerance):
            raise ValueError("Horizontal green carriage moved outside its travel range.")
        if not (-0.08 - joint_limit_tolerance <= table_y <= 0.08 + joint_limit_tolerance):
            raise ValueError("Vertical beige carriage moved outside its travel range.")
        if not (0.0 <= slide_pos[2] <= 0.04 and 0.0 <= table_pos[2] <= 0.05):
            raise ValueError("Table bodies left the ground-plane layout.")

        summary = self.get_metric_summary()
        if summary.get("repeat_samples", 0) > 0:
            if summary["body_repeat_max_m"] > BODY_REPEAT_TOLERANCE:
                raise ValueError(
                    f"Periodic body drift exceeded {BODY_REPEAT_TOLERANCE * 1000:.1f} mm: "
                    f"{summary['body_repeat_max_m'] * 1000:.2f} mm at body "
                    f"{summary['body_repeat_worst']!r}."
                )
            if summary["table_repeat_max_m"] > TABLE_REPEAT_TOLERANCE:
                raise ValueError(
                    f"Table cycle-to-cycle drift exceeded {TABLE_REPEAT_TOLERANCE * 1000:.1f} mm: "
                    f"{summary['table_repeat_max_m'] * 1000:.2f} mm."
                )
        if summary["table_tracking_max_m"] > TABLE_TRACKING_TOLERANCE:
            raise ValueError(
                f"Table tracking error exceeded {TABLE_TRACKING_TOLERANCE * 1000:.1f} mm: "
                f"{summary['table_tracking_max_m'] * 1000:.2f} mm."
            )
        if summary["joint_pos_gap_max_m"] > JOINT_POSITION_GAP_TOLERANCE:
            raise ValueError(
                f"Worst joint position gap exceeded {JOINT_POSITION_GAP_TOLERANCE * 1000:.1f} mm: "
                f"{summary['joint_pos_gap_max_m'] * 1000:.2f} mm at "
                f"{summary['joint_pos_gap_worst']!r}."
            )
        if summary["joint_ang_gap_max_rad"] > JOINT_ANGULAR_GAP_TOLERANCE:
            raise ValueError(
                f"Worst joint angular gap exceeded {JOINT_ANGULAR_GAP_TOLERANCE:.3e} rad: "
                f"{summary['joint_ang_gap_max_rad']:.3e} rad at "
                f"{summary['joint_ang_gap_worst']!r}."
            )

        missed_points = np.nonzero(self.table_rect_min_distances > TABLE_RECT_HIT_TOLERANCE)[0]
        if len(missed_points) > 0:
            details = []
            for point_index in missed_points:
                point = self.table_rect_points[point_index]
                distance = self.table_rect_min_distances[point_index]
                details.append(f"({point[0]:.3f}, {point[1]:.3f}) min error {distance:.4f} m")
            raise ValueError(
                "Cross-slide table did not hit every rectangle point within "
                f"{TABLE_RECT_HIT_TOLERANCE:.3f} m: {', '.join(details)}"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=TABLE_RECT_TEST_FRAMES)
        parser.add_argument(
            "--contact-mode",
            choices=["hard", "hard-history", "soft"],
            default="soft",
            help="Rigid contact mode. hard-history enables contact matching for contact warm-starting.",
        )
        parser.add_argument(
            "--joints-mode",
            choices=["hard", "soft"],
            default="hard",
            help="Constraint mode for non-cable slide, pulley, and anchor joints.",
        )
        parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="SolverVBD iterations per substep.")
        parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS, help="Simulation substeps per rendered frame.")
        parser.add_argument(
            "--table-rect-period",
            type=float,
            default=TABLE_RECT_PERIOD,
            help="Seconds per full commanded rectangle loop. Larger values drive the input pulleys more slowly.",
        )
        parser.add_argument(
            "--segment-length",
            type=float,
            default=CABLE_SEGMENT_LENGTH,
            help="Target cable segment length [m]. Smaller values create more cable segments and contacts.",
        )
        parser.add_argument(
            "--ke",
            type=float,
            default=None,
            help=(
                "Contact normal stiffness [N/m]. Defaults by --contact-mode: "
                f"soft={SOFT_CONTACT_KE:.1e}, hard/hard-history={HARD_CONTACT_KE:.1e}."
            ),
        )
        parser.add_argument("--kd", type=float, default=CONTACT_KD, help="Contact normal damping [N*s/m].")
        parser.add_argument(
            "--mu",
            type=float,
            default=None,
            help=f"Coulomb friction coefficient. Defaults to {CONTACT_MU:.1f}.",
        )
        parser.add_argument(
            "--joint-linear-ke",
            type=float,
            default=None,
            help=(
                "VBD non-cable joint linear stiffness. Defaults by --contact-mode: "
                f"soft={SOFT_JOINT_KE:.1e}, hard/hard-history={HARD_JOINT_KE:.1e}."
            ),
        )
        parser.add_argument(
            "--joint-angular-ke",
            type=float,
            default=None,
            help="VBD non-cable joint angular stiffness. Defaults to --joint-linear-ke.",
        )
        parser.add_argument("--joint-linear-kd", type=float, default=0.0, help="VBD non-cable joint linear damping.")
        parser.add_argument("--joint-angular-kd", type=float, default=0.0, help="VBD non-cable joint angular damping.")
        parser.add_argument("--joint-limit-ke", type=float, default=2.0e3, help="Prismatic joint limit stiffness.")
        parser.add_argument("--joint-limit-kd", type=float, default=1.0e-4, help="Prismatic joint limit damping.")
        parser.add_argument("--cable-stretch-ke", type=float, default=1.0e5, help="Cable stretch stiffness.")
        parser.add_argument("--cable-stretch-kd", type=float, default=0.0, help="Cable stretch damping.")
        parser.add_argument("--cable-bend-ke", type=float, default=5.0e-5, help="Cable bend/twist stiffness.")
        parser.add_argument("--cable-bend-kd", type=float, default=1.0e-2, help="Cable bend/twist damping.")
        parser.add_argument("--wrap-clearance-scale", type=float, default=1.0, help="Cable-to-pulley route clearance.")
        parser.add_argument("--friction-epsilon", type=float, default=1.0e-2, help="Friction smoothing velocity scale.")
        parser.add_argument("--rigid-avbd-beta", type=float, default=0.0, help="Rigid AVBD penalty ramp rate.")
        parser.add_argument(
            "--rigid-avbd-contact-alpha",
            type=float,
            default=None,
            help="Body-body hard contact C0/history alpha. Defaults to SolverVBD policy.",
        )
        parser.add_argument("--rigid-avbd-gamma", type=float, default=0.999, help="Rigid AVBD history decay.")
        parser.add_argument(
            "--contact-debug-metrics",
            action="store_true",
            default=False,
            help="Compute expensive per-contact penetration and tangential slip diagnostics.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
