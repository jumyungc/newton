# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example VBD Franka Rigid Pickup

Validate monolithic ``SolverVBD`` parallel-jaw pickup of a light (20 g) damped
rigid cylinder or capsule under automatic body coloring.

``ModelBuilder.color()`` places the free object in the same color group as both
mimic-coupled Franka fingers, so the object and both fingers are updated in one
parallel sweep. Solving the closing contact damper inside the normal row keeps
the finger-object contacts stable through closing, lift, and hold without manual
color-group overrides. Contacts are warm-started (sticky contact matching with
``rigid_contact_history``), so a sustained grasp keeps its contact forces and
friction state between steps instead of rebuilding them every step.

Success criteria:
- The Franka hand descends over the resting object, closes its parallel jaws,
  lifts the object by about 50 cm, and holds it stationary for 2 s.
- Throughout the lift and hold, hand-frame slip stays below 5 mm and rotation
  after grasp stays below 5 degrees.
- During the hold, from 0.3 s after the lift ends, the object moves less than
  0.5 mm in the hand frame.

Runs:
- `python -m newton.examples vbd_franka_pickup`
- `python -m newton.examples vbd_franka_pickup --collider capsule`
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.solvers import SolverVBD

FPS = 60
SUBSTEPS = 12
ITERATIONS = 15
OBJECT_START = (0.375, -0.23, 0.014)

APPROACH = (-0.117156506, -0.389616609, -0.342368990, -1.966722608, -0.127911627, 1.597449064, 0.353197902)
PICK = (-0.271077037, 0.434059948, -0.195775643, -2.531195879, 0.422424048, 2.940695286, -0.078577317)
LIFT = (-0.152631506, -0.385116309, -0.314951152, -1.967067480, -0.116681263, 1.599001408, 0.342843592)

TIMES = np.array([0.0, 0.8, 1.6, 3.2, 5.2], dtype=np.float64)
PINCH_FRAME = round(FPS * 1.6)
HOLD_FRAME = round(FPS * 3.5)
TARGETS = np.array(
    [
        (*APPROACH, 0.04, 0.04),
        (*PICK, 0.04, 0.04),
        (*PICK, 0.01, 0.01),
        (*LIFT, 0.01, 0.01),
        (*LIFT, 0.01, 0.01),
    ],
    dtype=np.float32,
)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ],
        dtype=np.float64,
    )


def _quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return _quat_mul(_quat_mul(_quat_conjugate(q), qv), q)[:3]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.collider = getattr(args, "collider", "cylinder")
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0

        urdf = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        builder = newton.ModelBuilder()
        builder.add_urdf(str(urdf), floating=False, enable_self_collisions=False)
        builder.collapse_fixed_joints(joints_to_keep=["fr3/fr3_hand_joint"])
        builder.joint_q[:9] = TARGETS[0].tolist()
        builder.joint_target_ke[:9] = [32000.0] * 7 + [1500.0] * 2
        builder.joint_target_kd[:9] = [240.0] * 7 + [8.0] * 2

        def material(mu: float, gap: float) -> newton.ModelBuilder.ShapeConfig:
            return newton.ModelBuilder.ShapeConfig(mu=mu, ke=1.0e4, kd=30.0, gap=gap, density=0.0)

        for shape, body in enumerate(builder.shape_body):
            is_finger = body >= 0 and builder.body_label[body].endswith(("leftfinger", "rightfinger"))
            builder.shape_material_mu[shape] = 1.5 if is_finger else 0.5
            builder.shape_material_ke[shape] = 1.0e4
            builder.shape_material_kd[shape] = 30.0

        builder.add_ground_plane(cfg=material(mu=0.5, gap=0.001))

        self.hand = next(i for i, label in enumerate(builder.body_label) if label.endswith("fr3_hand"))
        self.left_finger = next(i for i, label in enumerate(builder.body_label) if label.endswith("leftfinger"))
        self.right_finger = next(i for i, label in enumerate(builder.body_label) if label.endswith("rightfinger"))

        self.candy = builder.add_body(
            xform=wp.transform(OBJECT_START, wp.quat_identity()),
            mass=0.020,
            inertia=wp.diag(wp.vec3(2.0e-6)),
            label=f"candy_{self.collider}",
        )
        rot_x = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.5)
        if self.collider == "cylinder":
            builder.add_shape_cylinder(
                self.candy,
                xform=wp.transform((0.0, 0.0, 0.0), rot_x),
                radius=0.014,
                half_height=0.014,
                cfg=material(mu=1.2, gap=0.0005),
                color=(0.85, 0.14, 0.16),
            )
        else:
            builder.add_shape_capsule(
                self.candy,
                xform=wp.transform((0.0, 0.0, 0.0), rot_x),
                radius=0.014,
                half_height=0.005,
                cfg=material(mu=1.2, gap=0.0005),
                color=(0.18, 0.62, 0.85),
            )

        builder.color(balance_colors=False)
        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._target_host = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        self._target_host[:9] = TARGETS[0]
        self.control.joint_target_q.assign(self._target_host)

        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="sap", contact_matching="sticky")
        self.contacts = self.collision_pipeline.contacts()
        self.solver = SolverVBD(
            self.model,
            iterations=ITERATIONS,
            rigid_joint_linear_ke=1.0e7,
            rigid_joint_angular_ke=1.0e6,
            rigid_body_contact_buffer_size=256,
            friction_epsilon=1.0e-3,
            rigid_contact_history=True,
        )

        self._pinch_rel_pos = None
        self._pinch_rel_quat = None
        self.max_lift_slip_mm = 0.0
        self.max_lift_turn_deg = 0.0
        self._hold_rel_pos = None
        self.max_hold_drift_mm = 0.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.85, -0.75, 0.42), -18.0, 132.0)

        self.graph = None
        device = self.model.device
        if device.is_cuda and wp.is_mempool_enabled(device):
            self.simulate()
            with wp.ScopedCapture(device=device) as capture:
                self.simulate()
            self.graph = capture.graph
            self.state_0.body_q.assign(self.model.body_q)
            self.state_0.body_qd.zero_()
            self.state_1.body_q.assign(self.model.body_q)
            self.state_1.body_qd.zero_()
            self.solver.reset(self.state_0)

    def _update_joint_targets(self, t: float) -> None:
        segment = min(int(np.searchsorted(TIMES, t, side="right") - 1), len(TIMES) - 2)
        span = TIMES[segment + 1] - TIMES[segment]
        alpha = min(max((t - TIMES[segment]) / span, 0.0), 1.0)
        blend = alpha * alpha * alpha * (10.0 - 15.0 * alpha + 6.0 * alpha * alpha)
        self._target_host[:9] = (1.0 - blend) * TARGETS[segment] + blend * TARGETS[segment + 1]
        self.control.joint_target_q.assign(self._target_host)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _record_grasp_metrics(self) -> None:
        body_q = self.state_0.body_q.numpy()
        hand_q = body_q[self.hand, 3:7].astype(np.float64)
        hand_q /= np.linalg.norm(hand_q)
        finger_mid = 0.5 * (body_q[self.left_finger, :3] + body_q[self.right_finger, :3]).astype(np.float64)
        candy_pos = body_q[self.candy, :3].astype(np.float64)
        candy_q = body_q[self.candy, 3:7].astype(np.float64)
        candy_q /= np.linalg.norm(candy_q)

        rel_pos = _quat_rotate_inv(hand_q, candy_pos - finger_mid)
        rel_quat = _quat_mul(_quat_conjugate(hand_q), candy_q)
        rel_quat /= np.linalg.norm(rel_quat)

        if self.frame == PINCH_FRAME:
            self._pinch_rel_pos = rel_pos
            self._pinch_rel_quat = rel_quat
        elif self.frame > PINCH_FRAME and self._pinch_rel_pos is not None:
            slip_mm = 1000.0 * float(np.linalg.norm(rel_pos - self._pinch_rel_pos))
            dot = min(abs(float(np.dot(rel_quat, self._pinch_rel_quat))), 1.0)
            turn_deg = math.degrees(2.0 * math.acos(dot))
            self.max_lift_slip_mm = max(self.max_lift_slip_mm, slip_mm)
            self.max_lift_turn_deg = max(self.max_lift_turn_deg, turn_deg)
        if self.frame == HOLD_FRAME:
            self._hold_rel_pos = rel_pos
        elif self.frame > HOLD_FRAME and self._hold_rel_pos is not None:
            drift_mm = 1000.0 * float(np.linalg.norm(rel_pos - self._hold_rel_pos))
            self.max_hold_drift_mm = max(self.max_hold_drift_mm, drift_mm)

    def step(self):
        self._update_joint_targets(self.frame * self.frame_dt)
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.frame += 1
        self.sim_time = self.frame * self.frame_dt
        self._record_grasp_metrics()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        body_q = self.state_0.body_q.numpy()
        candy_z = 1000.0 * float(body_q[self.candy, 2])
        ui.text(f"Collider: {self.collider} (20 g, r = 14 mm)")
        ui.text(f"Candy height: {candy_z:.1f} mm")
        ui.text(f"Max lift slip: {self.max_lift_slip_mm:.2f} mm (limit 5 mm)")
        ui.text(f"Max lift turn: {self.max_lift_turn_deg:.2f} deg (limit 5 deg)")
        ui.text(f"Max hold drift: {self.max_hold_drift_mm:.2f} mm (limit 0.5 mm)")

    def test_post_step(self):
        assert np.isfinite(self.state_0.body_q.numpy()).all(), "Poses must remain finite"
        assert np.isfinite(self.state_0.body_qd.numpy()).all(), "Velocities must remain finite"

    def test_final(self):
        assert self.frame >= 300, "Run at least 300 frames to complete the close, lift and hold"
        body_q = self.state_0.body_q.numpy()
        candy_z = float(body_q[self.candy, 2])
        assert candy_z > 0.14, f"Candy was dropped or thrown: final z = {candy_z:.3f} m (expected > 0.14 m)"
        assert self.max_lift_slip_mm < 5.0, (
            f"Candy slipped {self.max_lift_slip_mm:.2f} mm in the hand frame (limit 5 mm)"
        )
        assert self.max_lift_turn_deg < 5.0, (
            f"Candy rotated {self.max_lift_turn_deg:.2f} deg during lift and hold (limit 5 deg)"
        )
        assert self.max_hold_drift_mm < 0.5, (
            f"Candy moved {self.max_hold_drift_mm:.2f} mm in the hand frame during the hold (limit 0.5 mm)"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--collider", choices=("cylinder", "capsule"), default="cylinder")
        parser.set_defaults(num_frames=312)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
