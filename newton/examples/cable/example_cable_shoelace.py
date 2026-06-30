# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: Shoelace Bow
#
# Simulates a shoelace as a chain of rigid capsules (newton.ModelBuilder.add_rod)
# threaded through six static eyelet guides on a boot. Four guided segments pull
# the two bow loops forward and outward while the knot's standing parts tighten,
# then the guides release so the bow settles under contact friction. This
# exercises dense dynamic-to-static and self-contact handling in SolverVBD.
#
# All geometry comes from the compact shoelace.npz asset (a centerline, eyelet
# ring frames, and anchor segment indices); no source mesh or trimesh is needed
# at runtime.
#
# Command: python -m newton.examples cable_shoelace --viewer gl
###########################################################################

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

SHORT_ASSET_PATH = Path(newton.examples.get_asset("shoe/shoelace.npz"))
LONG_ASSET_PATH = Path(newton.examples.get_asset("shoe/shoelace_long.npz"))

# Simulation cadence.
FPS = 60
SIM_SUBSTEPS = 10
SIM_ITERATIONS = 20

# Rod geometry and material (SI units).
CABLE_RADIUS = 0.0007
CABLE_DENSITY = 150.0
STRETCH_STIFFNESS = 1.0e9
STRETCH_DAMPING = 1.0e2
BEND_STIFFNESS = 1.0
BEND_DAMPING = 1.0

# Contact model.
GRAVITY = -1.0
CONTACT_GAP = 1.0e-4
CONTACT_KE = 1.0e8
CONTACT_KD = 0.0
CONTACT_MU = 1.0
RELEASE_CONTACT_MU = 1000.0
RELEASE_FRICTION_LEAD_TIME = 0.25
DRAG_JOINT_KE = 1.0e7

# Guided bow-tying schedule (seconds and meters).
APPROACH_DURATION = 2.0
TIGHTEN_DURATION = 10.0
SETTLE_DURATION = 2.0
LOOP_STANDOFF = 0.02
LOOP_TIGHTEN_DISTANCE = 0.02
LOOP_TIGHTEN_UPWARD_RATIO = 0.75
STANDING_TIGHTEN_DISTANCE = 0.01
STANDING_TIGHTEN_UPWARD_RATIO = 0.5
SHOE_FRONT_DIRECTION = (0.0, -1.0, 0.0)

# Eyelet collider geometry.
EYELET_SEGMENTS = 16
EYELET_COLLIDER_RADIUS_SCALE = 0.65
COLLISION_GROUP = 1

EYELET_COLOR = (0.95, 0.08, 0.72)
CABLE_COLOR = (0.02, 0.42, 1.0)
ANCHOR_COLORS = (
    (0.15, 1.0, 0.25),  # left loop
    (1.0, 0.78, 0.05),  # right loop
    (1.0, 0.12, 0.18),  # left standing
    (0.68, 0.18, 1.0),  # right standing
)


def _load_asset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Shoelace asset not found: {path}")
    with np.load(path) as data:
        asset = {key: np.asarray(data[key]) for key in data.files}
    centerline = asset["centerline"]
    if centerline.ndim != 2 or centerline.shape[1] != 3 or len(centerline) < 12:
        raise ValueError(f"Invalid centerline in {path}: {centerline.shape}")
    if len(asset["eyelet_centers"]) < 6:
        raise ValueError(f"Asset must contain at least three eyelet rows: {path}")
    return asset


def _quat_z_to(direction: np.ndarray) -> wp.quat:
    """Shortest-arc rotation mapping +Z onto ``direction``."""
    normalized = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(normalized))
    if length <= 1.0e-12:
        return wp.quat_identity()
    normalized = normalized / length
    cosine = float(np.clip(normalized[2], -1.0, 1.0))
    if cosine > 0.99999:
        return wp.quat_identity()
    if cosine < -0.99999:
        return wp.quat(1.0, 0.0, 0.0, 0.0)
    axis = np.cross((0.0, 0.0, 1.0), normalized)
    axis /= np.linalg.norm(axis)
    half_angle = 0.5 * float(np.arccos(cosine))
    return wp.quat(*(axis * np.sin(half_angle)), float(np.cos(half_angle)))


def _add_eyelet_guide(
    builder: newton.ModelBuilder,
    body: int,
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    major_radius: float,
    tube_radius: float,
    cfg: newton.ModelBuilder.ShapeConfig,
    label: str,
) -> int:
    """Approximate one torus eyelet with a closed ring of static capsules."""
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = axis_v - float(axis_v @ axis_u) * axis_u
    axis_v /= np.linalg.norm(axis_v)

    collider_radius = min(EYELET_COLLIDER_RADIUS_SCALE * tube_radius, major_radius - CABLE_RADIUS - CONTACT_GAP)
    if collider_radius <= 0.0:
        raise ValueError(f"Cable radius leaves no open aperture in {label}")

    angles = np.linspace(0.0, 2.0 * np.pi, EYELET_SEGMENTS + 1)
    points = np.asarray([center + major_radius * (np.cos(a) * axis_u + np.sin(a) * axis_v) for a in angles])
    for index, (start, end) in enumerate(pairwise(points)):
        segment = end - start
        builder.add_shape_capsule(
            body=body,
            xform=wp.transform(wp.vec3(*(0.5 * (start + end))), _quat_z_to(segment)),
            radius=collider_radius,
            half_height=0.5 * float(np.linalg.norm(segment)),
            cfg=cfg,
            color=EYELET_COLOR,
            label=f"{label}_{index:02d}",
        )
    return EYELET_SEGMENTS


def _length_matched_straight_curve(points: np.ndarray) -> np.ndarray:
    """Straight rest curve with the same per-segment lengths as ``points``."""
    offsets = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    center = points.mean(axis=0)
    return center + np.column_stack((offsets - 0.5 * offsets[-1], np.zeros((len(points), 2))))


def _curve_body_xforms(points: np.ndarray, quaternions: list[wp.quat]) -> list[wp.transform]:
    return [
        wp.transform(wp.vec3(*(0.5 * (points[i] + points[i + 1]))), quaternions[i]) for i in range(len(points) - 1)
    ]


def _neighbor_filter_window(points: np.ndarray) -> int:
    """Number of adjacent capsules that overlap purely by construction."""
    minimum_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).min())
    contact_reach = 2.0 * CABLE_RADIUS + CONTACT_GAP
    return max(1, int(np.floor(contact_reach / minimum_length + 1.0e-9)) + 1)


def _filter_rod_neighbors(builder: newton.ModelBuilder, bodies: list[int], window: int) -> None:
    for first, first_body in enumerate(bodies):
        for second_body in bodies[first + 1 : min(first + 1 + window, len(bodies))]:
            for first_shape in builder.body_shapes.get(first_body, []):
                for second_shape in builder.body_shapes.get(second_body, []):
                    builder.add_shape_collision_filter_pair(int(first_shape), int(second_shape))


@wp.kernel
def _set_body_xforms(
    body_indices: wp.array(dtype=wp.int32),
    body_xforms: wp.array(dtype=wp.transform),
    body_q0: wp.array(dtype=wp.transform),
    body_q1: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    body = body_indices[tid]
    body_q0[body] = body_xforms[tid]
    body_q1[body] = body_xforms[tid]


@wp.kernel
def _set_joints_enabled(
    joint_indices: wp.array(dtype=wp.int32),
    enabled: wp.bool,
    joint_enabled: wp.array(dtype=wp.bool),
):
    joint_enabled[joint_indices[wp.tid()]] = enabled


@wp.kernel
def _set_contact_friction(
    material_mu: wp.array(dtype=wp.float32),
    elapsed: wp.array(dtype=wp.float32),
    switch_time: float,
    initial_mu: float,
    release_mu: float,
):
    mu = initial_mu
    if elapsed[0] >= switch_time:
        mu = release_mu
    material_mu[wp.tid()] = mu


@wp.kernel
def _set_driver_xforms(
    driver_bodies: wp.array(dtype=wp.int32),
    start_positions: wp.array(dtype=wp.vec3),
    start_velocities: wp.array(dtype=wp.vec3),
    front_positions: wp.array(dtype=wp.vec3),
    target_positions: wp.array(dtype=wp.vec3),
    elapsed: wp.array(dtype=wp.float32),
    motion_active: wp.array(dtype=wp.int32),
    approach_duration: float,
    tighten_duration: float,
    substep_offset: float,
    body_q0: wp.array(dtype=wp.transform),
    body_q1: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    position = start_positions[tid]
    if motion_active[0] != 0:
        time = wp.clamp(elapsed[0] + substep_offset, 0.0, approach_duration + tighten_duration)
        if time <= approach_duration:
            u = time / approach_duration
            u2 = u * u
            u3 = u2 * u
            h00 = 2.0 * u3 - 3.0 * u2 + 1.0
            h10 = u3 - 2.0 * u2 + u
            h01 = -2.0 * u3 + 3.0 * u2
            position = (
                h00 * start_positions[tid] + h10 * approach_duration * start_velocities[tid] + h01 * front_positions[tid]
            )
        else:
            u = (time - approach_duration) / tighten_duration
            smooth = 3.0 * u * u - 2.0 * u * u * u
            position = (1.0 - smooth) * front_positions[tid] + smooth * target_positions[tid]
    xform = wp.transform(position, wp.quat_identity())
    body = driver_bodies[tid]
    body_q0[body] = xform
    body_q1[body] = xform


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
        self.sim_time = 0.0

        self.guided_duration = APPROACH_DURATION + TIGHTEN_DURATION + SETTLE_DURATION
        self.friction_switch_time = self.guided_duration - min(RELEASE_FRICTION_LEAD_TIME, SETTLE_DURATION)

        asset_path = Path(args.asset) if args.asset is not None else (
            LONG_ASSET_PATH if args.lacing == "long" else SHORT_ASSET_PATH
        )
        asset = _load_asset(asset_path)
        self.centerline = np.asarray(asset["centerline"], dtype=np.float64)
        loop_segments = [int(i) for i in asset["loop_anchor_segments"]]
        standing_segments = [int(i) for i in asset["standing_anchor_segments"]]
        self.anchor_segments = [*loop_segments, *standing_segments]

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=GRAVITY)
        builder.rigid_gap = CONTACT_GAP
        builder.default_shape_cfg.ke = CONTACT_KE
        builder.default_shape_cfg.kd = CONTACT_KD
        builder.default_shape_cfg.mu = CONTACT_MU

        guide_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            collision_group=COLLISION_GROUP,
            has_shape_collision=True,
            has_particle_collision=False,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=CONTACT_MU,
            gap=CONTACT_GAP,
        )
        self.eyelet_body = builder.add_link(mass=0.0, is_kinematic=True, label="eyelet_guides")
        for index in range(len(asset["eyelet_centers"])):
            _add_eyelet_guide(
                builder,
                self.eyelet_body,
                np.asarray(asset["eyelet_centers"][index], dtype=np.float64),
                np.asarray(asset["eyelet_axis_u"][index], dtype=np.float64),
                np.asarray(asset["eyelet_axis_v"][index], dtype=np.float64),
                float(asset["eyelet_major_radii"][index]),
                float(asset["eyelet_tube_radii"][index]),
                guide_cfg,
                f"eyelet_{index:02d}",
            )

        cable_cfg = newton.ModelBuilder.ShapeConfig(
            density=CABLE_DENSITY,
            collision_group=COLLISION_GROUP,
            has_shape_collision=True,
            has_particle_collision=False,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=CONTACT_MU,
            gap=CONTACT_GAP,
        )
        initial_positions = [wp.vec3(*point) for point in self.centerline]
        initial_quaternions = newton.utils.create_parallel_transport_cable_quaternions(initial_positions)
        self.initial_cable_xforms = _curve_body_xforms(self.centerline, initial_quaternions)
        # Initial pose is always the extracted knot. The rest (zero-energy) shape is
        # either a length-matched straight curve (lace springs toward straight) or the
        # knot itself (knot is treated as the relaxed shape).
        if bool(args.flat_rest_shape):
            rest_positions = [wp.vec3(*point) for point in _length_matched_straight_curve(self.centerline)]
            rest_quaternions = newton.utils.create_parallel_transport_cable_quaternions(rest_positions)
        else:
            rest_positions = initial_positions
            rest_quaternions = initial_quaternions
        self.cable_bodies, _joints = builder.add_rod(
            positions=rest_positions,
            quaternions=rest_quaternions,
            radius=CABLE_RADIUS,
            cfg=cable_cfg,
            stretch_stiffness=STRETCH_STIFFNESS,
            stretch_damping=STRETCH_DAMPING,
            bend_stiffness=BEND_STIFFNESS,
            bend_damping=BEND_DAMPING,
            label="shoelace",
            color=CABLE_COLOR,
            body_frame_origin="com",
        )

        # One kinematic driver per anchor, joined to its rod segment by a ball
        # joint. Loop joints start enabled; the knot's standing joints engage only
        # once the bow loops have been pulled forward.
        self.anchor_bodies = [self.cable_bodies[index] for index in self.anchor_segments]
        initial_anchor_positions = np.asarray(
            [0.5 * (self.centerline[i] + self.centerline[i + 1]) for i in self.anchor_segments],
            dtype=np.float32,
        )
        self.driver_bodies = []
        self.drag_joints = []
        for index, (anchor_body, color) in enumerate(zip(self.anchor_bodies, ANCHOR_COLORS, strict=True)):
            driver = builder.add_link(mass=0.0, is_kinematic=True, label=f"shoelace_driver_{index}")
            self.driver_bodies.append(driver)
            self.drag_joints.append(
                builder.add_joint_ball(
                    parent=driver,
                    child=anchor_body,
                    parent_xform=wp.transform_identity(),
                    child_xform=wp.transform_identity(),
                    collision_filter_parent=False,
                    enabled=index < 2,
                    label=f"shoelace_drag_joint_{index}",
                )
            )
            for shape in builder.body_shapes.get(anchor_body, []):
                builder.shape_color[int(shape)] = color

        _filter_rod_neighbors(builder, self.cable_bodies, _neighbor_filter_window(self.centerline))
        builder.color()
        self.model = builder.finalize()
        self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
        self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=SIM_ITERATIONS,
            rigid_contact_hard=True,
            rigid_body_contact_buffer_size=int(args.contact_buffer),
            rigid_joint_linear_ke=DRAG_JOINT_KE,
            rigid_contact_history=True,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Pose the rod into the extracted knot and park every driver on its anchor.
        driver_xforms = [wp.transform(wp.vec3(*p), wp.quat_identity()) for p in initial_anchor_positions]
        pose_bodies = [*self.cable_bodies, *self.driver_bodies]
        pose_xforms = [*self.initial_cable_xforms, *driver_xforms]
        pose_indices_wp = wp.array(pose_bodies, dtype=wp.int32, device=self.model.device)
        pose_xforms_wp = wp.array(pose_xforms, dtype=wp.transform, device=self.model.device)
        wp.launch(
            _set_body_xforms,
            dim=len(pose_bodies),
            inputs=[pose_indices_wp, pose_xforms_wp, self.state_0.body_q, self.state_1.body_q],
            device=self.model.device,
        )
        self.solver.body_q_prev = wp.clone(self.state_0.body_q, device=self.solver.device)

        # Loop targets: forward off the third eyelet row, then outward and up.
        third_row_centers = np.asarray(asset["eyelet_centers"][4:6], dtype=np.float64)
        front_direction = np.asarray(SHOE_FRONT_DIRECTION, dtype=np.float64)
        loop_front = third_row_centers + LOOP_STANDOFF * front_direction
        loop_target = loop_front + LOOP_TIGHTEN_DISTANCE * np.asarray(
            ((-1.0, 0.0, LOOP_TIGHTEN_UPWARD_RATIO), (1.0, 0.0, LOOP_TIGHTEN_UPWARD_RATIO))
        )

        device = self.model.device
        self.driver_indices_wp = wp.array(self.driver_bodies, dtype=wp.int32, device=device)
        self.loop_driver_indices_wp = wp.array(self.driver_bodies[:2], dtype=wp.int32, device=device)
        self.standing_driver_indices_wp = wp.array(self.driver_bodies[2:], dtype=wp.int32, device=device)
        self.loop_joint_indices_wp = wp.array(self.drag_joints[:2], dtype=wp.int32, device=device)
        self.standing_joint_indices_wp = wp.array(self.drag_joints[2:], dtype=wp.int32, device=device)
        self.all_joint_indices_wp = wp.array(self.drag_joints, dtype=wp.int32, device=device)

        self.start_positions_wp = wp.array(initial_anchor_positions, dtype=wp.vec3, device=device)
        self.start_velocities_wp = wp.zeros(4, dtype=wp.vec3, device=device)
        self.front_positions_wp = wp.array(initial_anchor_positions, dtype=wp.vec3, device=device)
        self.target_positions_wp = wp.array(initial_anchor_positions, dtype=wp.vec3, device=device)
        self.front_positions_wp.assign(
            np.vstack((loop_front, initial_anchor_positions[2:])).astype(np.float32)
        )
        self.target_positions_wp.assign(
            np.vstack((loop_target, initial_anchor_positions[2:])).astype(np.float32)
        )
        self.elapsed_wp = wp.zeros(1, dtype=wp.float32, device=device)
        self.motion_active_wp = wp.zeros(1, dtype=wp.int32, device=device)

        self.loop_active = False
        self.standing_active = False
        self.released = False

        self.model.collide(self.state_0, self.contacts)
        self.viewer.set_model(self.model)
        self._configure_camera(asset)

        self.graph = None
        self.capture()

    def _configure_camera(self, asset: dict[str, np.ndarray]) -> None:
        points = np.vstack((self.centerline, np.asarray(asset["eyelet_centers"], dtype=np.float64)))
        margin = max(3.0 * CABLE_RADIUS, float(np.max(asset["eyelet_major_radii"])))
        bounds_min = points.min(axis=0) - margin
        bounds_max = points.max(axis=0) + margin
        target = 0.5 * (bounds_min + bounds_max)
        diagonal = float(np.linalg.norm(bounds_max - bounds_min))
        self.viewer.set_camera(
            pos=wp.vec3(target[0], target[1] - 1.55 * diagonal, target[2]),
            pitch=0.0,
            yaw=90.0,
        )
        camera = getattr(self.viewer, "camera", None)
        if camera is not None:
            camera.set_pivot(tuple(target))
        gui = getattr(self.viewer, "gui", None)
        if gui is not None and hasattr(gui, "_cam_speed"):
            gui._cam_speed = 0.5 * diagonal
            gui._cam_vel.fill(0.0)

    def _enable_joints(self, indices: wp.array, count: int, enabled: bool) -> None:
        wp.launch(
            _set_joints_enabled,
            dim=count,
            inputs=[indices, enabled, self.model.joint_enabled],
            device=self.model.device,
        )

    def _place_drivers(self, driver_indices: wp.array, count: int, positions: np.ndarray) -> None:
        xforms = wp.array(
            [wp.transform(wp.vec3(*p), wp.quat_identity()) for p in positions],
            dtype=wp.transform,
            device=self.model.device,
        )
        for body_q in (self.state_0.body_q, self.solver.body_q_prev):
            wp.launch(
                _set_body_xforms,
                dim=count,
                inputs=[driver_indices, xforms, body_q, body_q],
                device=self.model.device,
            )

    def _update_drive(self) -> None:
        elapsed = float(np.clip(self.sim_time, 0.0, self.guided_duration))
        self.elapsed_wp.fill_(elapsed)
        if self.released:
            return

        if not self.loop_active:
            body_q = self.state_0.body_q.numpy()
            body_qd = self.state_0.body_qd.numpy()
            loop_start = np.asarray(body_q[self.anchor_bodies[:2], :3], dtype=np.float32)
            loop_velocity = np.asarray(body_qd[self.anchor_bodies[:2], :3], dtype=np.float32)
            start = self.start_positions_wp.numpy()
            velocity = self.start_velocities_wp.numpy()
            start[:2] = loop_start
            velocity[:2] = loop_velocity
            self.start_positions_wp.assign(start)
            self.start_velocities_wp.assign(velocity)
            self._place_drivers(self.loop_driver_indices_wp, 2, loop_start)
            self._enable_joints(self.loop_joint_indices_wp, 2, True)
            self.motion_active_wp.fill_(1)
            self.loop_active = True

        if elapsed >= APPROACH_DURATION and not self.standing_active:
            body_q = self.state_0.body_q.numpy()
            standing_start = np.asarray(body_q[self.anchor_bodies[2:], :3], dtype=np.float32)
            directions = np.asarray(
                ((-1.0, 0.0, STANDING_TIGHTEN_UPWARD_RATIO), (1.0, 0.0, STANDING_TIGHTEN_UPWARD_RATIO)),
                dtype=np.float32,
            )
            for buffer, value in (
                (self.start_positions_wp, standing_start),
                (self.front_positions_wp, standing_start),
                (self.target_positions_wp, standing_start + STANDING_TIGHTEN_DISTANCE * directions),
            ):
                data = buffer.numpy()
                data[2:] = value
                buffer.assign(data)
            self._place_drivers(self.standing_driver_indices_wp, 2, standing_start)
            self._enable_joints(self.standing_joint_indices_wp, 2, True)
            self.standing_active = True

        if elapsed >= self.guided_duration:
            self._enable_joints(self.all_joint_indices_wp, len(self.drag_joints), False)
            self.released = True

    def capture(self) -> None:
        self.graph = None
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        for substep in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                _set_contact_friction,
                dim=self.model.shape_count,
                inputs=[
                    self.model.shape_material_mu,
                    self.elapsed_wp,
                    self.friction_switch_time,
                    CONTACT_MU,
                    RELEASE_CONTACT_MU,
                ],
                device=self.model.device,
            )
            wp.launch(
                _set_driver_xforms,
                dim=len(self.driver_bodies),
                inputs=[
                    self.driver_indices_wp,
                    self.start_positions_wp,
                    self.start_velocities_wp,
                    self.front_positions_wp,
                    self.target_positions_wp,
                    self.elapsed_wp,
                    self.motion_active_wp,
                    APPROACH_DURATION,
                    TIGHTEN_DURATION,
                    (substep + 1) * self.sim_dt,
                    self.state_0.body_q,
                    self.state_1.body_q,
                ],
                device=self.model.device,
            )
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self._update_drive()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Visibility is controlled by the viewer's built-in "Show Contacts" toggle.
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--lacing",
            choices=("short", "long"),
            default="short",
            help="Short lacing (top three eyelet rows) or long lacing (all rows).",
        )
        parser.add_argument(
            "--asset", type=Path, default=None, help="Override shoelace NPZ asset (defaults by --lacing)."
        )
        parser.add_argument("--contact-buffer", type=int, default=128, help="Per-body VBD rigid-contact capacity.")
        parser.add_argument(
            "--flat-rest-shape",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use a length-matched straight rest curve while keeping the extracted knot as the initial pose.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
