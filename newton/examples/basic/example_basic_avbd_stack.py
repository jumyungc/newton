# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic AVBD Stack
#
# Recreates the box stack and pyramid scenes from the AVBD 3D demo using
# Newton's VBD rigid-body solver. The scene keeps the AVBD demo cadence and
# solver tuning: 60 Hz, one simulation substep, and ten VBD iterations per frame.
#
# Command: python -m newton.examples basic_avbd_stack
#
###########################################################################

import argparse
import colorsys

import numpy as np
import warp as wp

import newton
import newton.examples


STACK_BOX_COUNT = 10
STACK_BOX_SIZE = 1.0
STACK_BOX_HALF_EXTENT = 0.5 * STACK_BOX_SIZE
STACK_BOX_SPACING = 1.5
PYRAMID_ROWS = 16
PYRAMID_HALF_EXTENTS = (0.5, 0.25, 0.25)
PYRAMID_X_SPACING = 1.01
PYRAMID_ROW_X_OFFSET = 0.5
PYRAMID_Z_SPACING = 0.85
PYRAMID_BASE_Z = 0.5
GROUND_HALF_EXTENTS = (50.0, 50.0, 0.5)

FRICTION = 0.5
CONTACT_KE = 1.0e8
CONTACT_KD = 0.0
CONTACT_K_START = 1.0
COLLISION_GAP = 0.0
COLLISION_MARGIN = 0.01
CONTACT_MATCHING_POS_THRESHOLD = 0.00075
CONTACT_MATCHING_NORMAL_DOT_THRESHOLD = 0.995

AVBD_ALPHA = 0.99
AVBD_BETA_LINEAR = 10000.0
AVBD_BETA_ANGULAR = 100.0
AVBD_GAMMA = 0.999
AVBD_PRIMAL_WARMSTART = 0.0
AVBD_ADAPTIVE_PRIMAL_WARMSTART = True

CONTACT_MODES = ("soft", "hard", "hard_history")
LAYOUTS = ("stack", "pyramid")
DEFAULT_LAYOUT = "pyramid"


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.layout = getattr(args, "layout", DEFAULT_LAYOUT) if args is not None else DEFAULT_LAYOUT
        self.contact_mode = getattr(args, "contact_mode", "hard_history") if args is not None else "hard_history"
        self.contact_ke = getattr(args, "contact_ke", CONTACT_KE) if args is not None else CONTACT_KE
        self.contact_kd = getattr(args, "contact_kd", CONTACT_KD) if args is not None else CONTACT_KD
        self.contact_k_start = getattr(args, "contact_k_start", CONTACT_K_START) if args is not None else CONTACT_K_START
        self.gap = getattr(args, "gap", COLLISION_GAP) if args is not None else COLLISION_GAP
        self.margin = getattr(args, "margin", COLLISION_MARGIN) if args is not None else COLLISION_MARGIN
        self.friction = getattr(args, "friction", FRICTION) if args is not None else FRICTION
        self.primal_warmstart = (
            getattr(args, "rigid_primal_warmstart", AVBD_PRIMAL_WARMSTART)
            if args is not None
            else AVBD_PRIMAL_WARMSTART
        )
        self.adaptive_primal_warmstart = (
            getattr(args, "rigid_adaptive_primal_warmstart", AVBD_ADAPTIVE_PRIMAL_WARMSTART)
            if args is not None
            else AVBD_ADAPTIVE_PRIMAL_WARMSTART
        )

        if self.contact_mode not in CONTACT_MODES:
            raise ValueError(f"contact_mode must be one of {CONTACT_MODES}, got {self.contact_mode!r}")
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {self.layout!r}")

        builder = newton.ModelBuilder(gravity=-10.0)
        builder.rigid_gap = self.gap

        box_cfg = newton.ModelBuilder.ShapeConfig(
            density=1.0,
            ke=self.contact_ke,
            kd=self.contact_kd,
            mu=self.friction,
            margin=self.margin,
            gap=self.gap,
        )
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=self.contact_ke,
            kd=self.contact_kd,
            mu=self.friction,
            margin=self.margin,
            gap=self.gap,
        )

        self.ground_top = 0.5 if self.layout == "stack" else 0.0
        ground_center_z = self.ground_top - GROUND_HALF_EXTENTS[2]
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, ground_center_z), wp.quat_identity()),
            hx=GROUND_HALF_EXTENTS[0],
            hy=GROUND_HALF_EXTENTS[1],
            hz=GROUND_HALF_EXTENTS[2],
            cfg=ground_cfg,
            color=(0.16, 0.16, 0.18),
            label="ground_box",
        )

        self.box_bodies = []
        self.initial_positions = []
        self.box_half_extents = []
        self.box_rows = []

        if self.layout == "stack":
            self._add_stack(builder, box_cfg)
        else:
            rows = getattr(args, "pyramid_rows", PYRAMID_ROWS) if args is not None else PYRAMID_ROWS
            self._add_pyramid(builder, box_cfg, rows)

        builder.color()
        # Force deterministic Gauss-Seidel scheduling: one dynamic rigid body per VBD color group.
        builder.body_color_groups = [np.array([body], dtype=int) for body in self.box_bodies]
        self.model = builder.finalize()

        use_contact_history = self.contact_mode == "hard_history"
        use_hard_contact = self.contact_mode != "soft"
        linear_beta = getattr(args, "linear_beta", None) if args is not None else None
        angular_beta = getattr(args, "angular_beta", None) if args is not None else None
        if linear_beta is None:
            linear_beta = 0.0 if self.contact_mode == "soft" else AVBD_BETA_LINEAR
        if angular_beta is None:
            angular_beta = 0.0 if self.contact_mode == "soft" else AVBD_BETA_ANGULAR

        rigid_contact_max = getattr(args, "rigid_contact_max", None) if args is not None else None
        if rigid_contact_max is None:
            rigid_contact_max = max(512, 16 * len(self.box_bodies))

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            contact_matching="latest" if use_contact_history else "disabled",
            contact_matching_pos_threshold=CONTACT_MATCHING_POS_THRESHOLD,
            contact_matching_normal_dot_threshold=CONTACT_MATCHING_NORMAL_DOT_THRESHOLD,
            deterministic=True,
            rigid_contact_max=rigid_contact_max,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_avbd_alpha=AVBD_ALPHA,
            rigid_avbd_linear_beta=linear_beta,
            rigid_avbd_angular_beta=angular_beta,
            rigid_avbd_gamma=AVBD_GAMMA,
            rigid_contact_k_start=self.contact_k_start,
            rigid_contact_history=use_contact_history,
            rigid_contact_hard=use_hard_contact,
            rigid_contact_stick_motion_eps=0.0,
            rigid_contact_stick_freeze_translation_eps=0.0,
            rigid_contact_stick_freeze_angular_eps=0.0,
            rigid_body_contact_buffer_size=64,
            rigid_primal_warmstart=self.primal_warmstart,
            rigid_adaptive_primal_warmstart=self.adaptive_primal_warmstart,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(7.0, -9.0, 6.0), pitch=-28.0, yaw=132.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 55.0

        self.capture()

    def _add_box(self, builder, cfg, pos, half_extents, label, row):
        body = builder.add_body(xform=wp.transform(pos, wp.quat_identity()), label=label)
        color = colorsys.hsv_to_rgb((len(self.box_bodies) * 0.61803398875) % 1.0, 0.6, 0.85)
        builder.add_shape_box(
            body=body,
            hx=half_extents[0],
            hy=half_extents[1],
            hz=half_extents[2],
            cfg=cfg,
            color=color,
            label=f"{label}_shape",
        )
        self.box_bodies.append(body)
        self.initial_positions.append((float(pos[0]), float(pos[1]), float(pos[2])))
        self.box_half_extents.append(half_extents)
        self.box_rows.append(row)
        return body

    def _add_stack(self, builder, box_cfg):
        half_extents = (STACK_BOX_HALF_EXTENT, STACK_BOX_HALF_EXTENT, STACK_BOX_HALF_EXTENT)
        for i in range(STACK_BOX_COUNT):
            self._add_box(
                builder,
                box_cfg,
                wp.vec3(0.0, 0.0, i * STACK_BOX_SPACING + STACK_BOX_SIZE),
                half_extents,
                f"box_{i}",
                i,
            )

    def _add_pyramid(self, builder, box_cfg, rows: int):
        half_extents = PYRAMID_HALF_EXTENTS
        for row in range(rows):
            row_count = rows - row
            for col in range(row_count):
                x = col * PYRAMID_X_SPACING + row * PYRAMID_ROW_X_OFFSET - rows / 2.0
                z = row * PYRAMID_Z_SPACING + PYRAMID_BASE_Z
                self._add_box(
                    builder,
                    box_cfg,
                    wp.vec3(x, 0.0, z),
                    half_extents,
                    f"box_{row}_{col}",
                    row,
                )

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        box_q = self.state_0.body_q.numpy()[self.box_bodies]
        box_qd = self.state_0.body_qd.numpy()[self.box_bodies]

        assert np.isfinite(box_q).all(), "Non-finite box poses"
        assert np.isfinite(box_qd).all(), "Non-finite box velocities"
        assert len(self.model.body_color_groups) == len(self.box_bodies), "Expected one VBD color per dynamic body"
        assert all(group.size == 1 for group in self.model.body_color_groups), "Expected fully sequential VBD body coloring"

        z_positions = box_q[:, 2]
        xy_positions = box_q[:, :2]
        y_positions = box_q[:, 1]
        initial_positions = np.array(self.initial_positions)
        half_z = np.array([extent[2] for extent in self.box_half_extents])

        min_allowed_z = np.min(self.ground_top + half_z - 0.25)
        max_allowed_z = np.max(initial_positions[:, 2]) + 2.0

        assert np.min(z_positions) > min_allowed_z, (
            f"Boxes penetrated the ground too much: min_z={np.min(z_positions):.3f} <= {min_allowed_z:.3f}"
        )
        assert np.max(z_positions) < max_allowed_z, (
            f"Stack rose above expected bounds: max_z={np.max(z_positions):.3f} >= {max_allowed_z:.3f}"
        )
        if self.layout == "stack":
            sorted_z = np.sort(z_positions)
            min_separation = np.min(np.diff(sorted_z))
            assert min_separation > 0.25, f"Stack collapsed vertically: min separation {min_separation:.3f}"
            assert np.max(np.linalg.norm(xy_positions, axis=1)) < 1.0, "Stack drifted too far horizontally"
        else:
            initial_xy = initial_positions[:, :2]
            max_initial_radius = np.max(np.linalg.norm(initial_xy, axis=1))
            row_means = np.array(
                [np.mean(z_positions[np.array(self.box_rows) == row]) for row in range(max(self.box_rows) + 1)]
            )
            expected_settled_top = self.ground_top + np.max(half_z) + max(self.box_rows) * 2.0 * np.max(half_z)
            assert np.max(np.linalg.norm(xy_positions, axis=1)) < max_initial_radius + 2.0, (
                "Pyramid drifted too far horizontally"
            )
            assert np.max(np.abs(y_positions)) < 2.0, "Pyramid toppled out of plane"
            assert np.max(z_positions) > 0.9 * expected_settled_top, "Pyramid collapsed vertically"
            assert np.min(np.diff(row_means)) > 0.2, "Pyramid rows did not remain stacked"
        assert (np.abs(box_qd) < 5.0e2).all(), "Box velocities too large"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--layout", choices=LAYOUTS, default=DEFAULT_LAYOUT, help="Box layout to simulate.")
        parser.add_argument(
            "--contact-mode",
            choices=CONTACT_MODES,
            default="hard_history",
            help="Rigid contact mode: soft penalty, hard AL, or hard AL with frame-to-frame history.",
        )
        parser.add_argument("--contact-ke", type=float, default=CONTACT_KE, help="Rigid contact stiffness.")
        parser.add_argument("--contact-kd", type=float, default=CONTACT_KD, help="Rigid contact damping.")
        parser.add_argument("--contact-k-start", type=float, default=CONTACT_K_START, help="AVBD contact k seed.")
        parser.add_argument("--friction", type=float, default=FRICTION, help="Rigid contact friction coefficient.")
        parser.add_argument("--gap", type=float, default=COLLISION_GAP, help="Contact detection gap.")
        parser.add_argument("--margin", type=float, default=COLLISION_MARGIN, help="Collision shape margin.")
        parser.add_argument("--linear-beta", type=float, default=None, help="Override AVBD linear beta.")
        parser.add_argument("--angular-beta", type=float, default=None, help="Override AVBD angular beta.")
        parser.add_argument(
            "--rigid-primal-warmstart",
            type=float,
            default=AVBD_PRIMAL_WARMSTART,
            help="Blend start pose toward inertial target before VBD iterations.",
        )
        parser.add_argument(
            "--rigid-adaptive-primal-warmstart",
            action=argparse.BooleanOptionalAction,
            default=AVBD_ADAPTIVE_PRIMAL_WARMSTART,
            help="Use AVBD reference-style acceleration-based primal warmstart.",
        )
        parser.add_argument("--rigid-contact-max", type=int, default=None, help="Rigid contact buffer capacity.")
        parser.add_argument("--pyramid-rows", type=int, default=PYRAMID_ROWS, help="Rows for the pyramid layout.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
