# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Box Pyramid
#
# Reproduces the pyramid stacking scene from avbd-demo3d. Builds a 2D
# pyramid of 136 boxes (16 rows) where each row has one fewer box than
# the row below. Defaults to the same computational budget as the original:
# dt=1/60, 1 substep, 10 iterations.
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

PYRAMID_VARIANTS = {
    "1": {
        "name": "new_alm_finite_large_gap",
        "shape_ke": 1.0e6,
        "shape_kd": 0.0,
        "gap": 0.5,
        "solver": {
            "rigid_contact_hard": True,
            "rigid_contact_compliant_alm": True,
            "rigid_joint_compliant_alm": True,
            "rigid_contact_history": True,
            "rigid_contact_stick_motion_eps": 1.0e-4,
            "rigid_contact_stick_freeze_translation_eps": 1.0e-4,
            "rigid_contact_stick_freeze_angular_eps": 1.0e-4,
        },
    },
    "2": {
        "name": "new_alm_inf_large_gap",
        "shape_ke": 1.0e10,
        "shape_kd": 0.0,
        "gap": 0.5,
        "solver": {
            "rigid_avbd_alpha": 0.95,
            "rigid_avbd_gamma": 0.999,
            "rigid_contact_hard": True,
            "rigid_contact_compliant_alm": True,
            "rigid_joint_compliant_alm": True,
            "rigid_contact_history": True,
        },
    },
    "3": {
        "name": "old_soft_large_gap",
        "shape_ke": 1.0e3,
        "shape_kd": 0.0,
        "gap": 0.5,
        "solver": {
            "rigid_avbd_alpha": 0.95,
            "rigid_avbd_gamma": 0.999,
            "rigid_contact_hard": False,
            "rigid_contact_compliant_alm": False,
            "rigid_joint_compliant_alm": False,
            "rigid_contact_history": True,
        },
    },
    "4": {
        "name": "new_alm_finite_small_gap",
        "shape_ke": 1.0e6,
        "shape_kd": 0.0,
        "gap": 0.05,
        "solver": {
            "rigid_contact_hard": True,
            "rigid_contact_compliant_alm": True,
            "rigid_joint_compliant_alm": True,
            "rigid_contact_history": True,
            "rigid_contact_stick_motion_eps": 1.0e-4,
            "rigid_contact_stick_freeze_translation_eps": 1.0e-4,
            "rigid_contact_stick_freeze_angular_eps": 1.0e-4,
        },
    },
    "5": {
        "name": "old_hard_matched_rho_large_gap",
        "shape_ke": 450.0,
        "ground_ke": 1350.0,
        "shape_kd": 0.0,
        "gap": 0.5,
        "solver": {
            "rigid_avbd_alpha": 0.95,
            "rigid_avbd_gamma": 0.999,
            "rigid_avbd_linear_beta": 0.0,
            "rigid_avbd_angular_beta": 0.0,
            "rigid_contact_hard": True,
            "rigid_contact_compliant_alm": False,
            "rigid_joint_compliant_alm": False,
            "rigid_contact_history": True,
            "rigid_contact_k_start": 1.0,
        },
    },
    "6": {
        "name": "old_hard_ramped_large_gap",
        "shape_ke": 1.0e10,
        "shape_kd": 0.0,
        "gap": 0.5,
        "solver": {
            "rigid_avbd_alpha": 0.96,
            "rigid_avbd_gamma": 0.999,
            "rigid_avbd_linear_beta": 3.0e3,
            "rigid_avbd_angular_beta": 1.0e2,
            "rigid_contact_hard": True,
            "rigid_contact_compliant_alm": False,
            "rigid_joint_compliant_alm": False,
            "rigid_contact_history": True,
            "rigid_contact_k_start": 1.0,
            "rigid_contact_stick_motion_eps": 0.0,
            "rigid_contact_stick_freeze_translation_eps": 0.0,
            "rigid_contact_stick_freeze_angular_eps": 0.0,
        },
    },
}

DEFAULT_PYRAMID_VARIANT = "6"


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer

        # Match avbd-demo3d by default: dt=1/60, 1 substep, 10 iterations.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = int(getattr(args, "iterations", 10) or 10)
        if self.iterations <= 0:
            raise ValueError(f"iterations must be positive, got {self.iterations}")
        self.pyramid_version = str(getattr(args, "pyramid_version", DEFAULT_PYRAMID_VARIANT) or DEFAULT_PYRAMID_VARIANT)
        if self.pyramid_version not in PYRAMID_VARIANTS:
            valid = ", ".join(sorted(PYRAMID_VARIANTS))
            raise ValueError(f"pyramid_version must be one of {valid}, got {self.pyramid_version!r}")
        self.pyramid_variant = PYRAMID_VARIANTS[self.pyramid_version]

        # Match avbd-demo3d: gravity=-10, density=1, friction=0.5, no damping.
        builder = newton.ModelBuilder(gravity=-10.0)

        builder.default_shape_cfg.density = 1.0
        builder.default_shape_cfg.mu = 0.5
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.ke = self.pyramid_variant.get("ground_ke", self.pyramid_variant["shape_ke"])
        builder.default_shape_cfg.kd = self.pyramid_variant["shape_kd"]
        builder.default_shape_cfg.gap = self.pyramid_variant["gap"]

        # avbd-demo3d uses a static box of full size {100, 100, 1} centered at z=-0.5.
        builder.add_shape_box(body=-1, xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5)), hx=50.0, hy=50.0, hz=0.5)
        builder.default_shape_cfg.ke = self.pyramid_variant["shape_ke"]

        # Box pyramid matching avbd-demo3d scenePyramid
        # Original: Rigid(solver, {1, 0.5, 0.5}, 1.0, 0.5, pos)
        # size={1, 0.5, 0.5} -> half extents (0.5, 0.25, 0.25)
        size = 16
        hx = 0.5
        hy = 0.25
        hz = 0.25
        pyramid_bodies = []

        for row in range(size):
            count = size - row
            # Match avbd-demo3d exactly: z = y * 0.85 + 0.5, x = x * 1.01 + y * 0.5 - SIZE/2
            z = row * 0.85 + 0.5
            for col in range(count):
                x = col * 1.01 + row * 0.5 - size / 2.0
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(x, 0.0, z), q=wp.quat_identity()),
                    label=f"box_{row}_{col}",
                )
                builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)
                pyramid_bodies.append(body)

        builder.color()
        # Match the AVBD stack report: serial one-body-per-color rigid VBD groups.
        builder.body_color_groups = [np.array([body], dtype=int) for body in pyramid_bodies]

        self.model = builder.finalize()

        pipeline = newton.CollisionPipeline(
            self.model,
            deterministic=True,
            contact_matching="latest",
            contact_matching_pos_threshold=0.00075,
            contact_matching_normal_dot_threshold=0.995,
        )
        self.contacts = self.model.contacts(collision_pipeline=pipeline)

        solver_kwargs = {
            "iterations": self.iterations,
            "rigid_body_contact_buffer_size": 512,
        }
        solver_kwargs.update(self.pyramid_variant["solver"])
        self.solver = newton.solvers.SolverVBD(self.model, **solver_kwargs)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(0.0, -24.0, 10.0)
            camera_target = wp.vec3(0.0, 0.0, 6.5)
            self.viewer.set_camera(pos=camera_pos, pitch=-8.0, yaw=90.0)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(camera_target)

        # Start paused so user can inspect the initial state
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = True

        self.capture()

    def capture(self):
        """Use uncaptured stepping for this contact-history stress case."""
        self.graph = None

    def simulate(self):
        """Execute all simulation substeps for one frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state to the viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify pyramid settled stably above the ground."""
        body_q = self.state_0.body_q.numpy()
        for i in range(self.model.body_count):
            z = body_q[i][2]
            assert z > -0.1, f"Body {i} fell below ground (z={z:.3f})"

        assert np.isfinite(body_q).all(), "Non-finite positions detected"

        rows = {}
        for i, raw_label in enumerate(self.model.body_label):
            label = str(raw_label)
            if label.startswith("box_"):
                _prefix, row, _col = label.split("_")
                rows.setdefault(int(row), []).append(i)

        row_z = np.asarray([np.mean(body_q[rows[row], 2]) for row in sorted(rows)])
        top_z = body_q[rows[max(rows)][0], 2]
        min_row_delta = np.min(np.diff(row_z))
        max_y = np.max(np.abs(body_q[:, 1]))

        assert top_z > 0.9 * 7.75, f"Pyramid top collapsed (z={top_z:.3f})"
        assert min_row_delta > 0.0, f"Pyramid row order inverted (min dz={min_row_delta:.3f})"
        assert max_y < 2.0, f"Pyramid drifted sideways (max |y|={max_y:.3f})"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--iterations", type=int, default=10, help="VBD solver iterations per frame.")
        parser.add_argument(
            "--pyramid-version",
            type=str,
            default=DEFAULT_PYRAMID_VARIANT,
            choices=sorted(PYRAMID_VARIANTS),
            help="ALM/AVBD pyramid comparison variant to run.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
