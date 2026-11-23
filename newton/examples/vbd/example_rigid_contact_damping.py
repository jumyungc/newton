# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example VBD Rigid Contact Damping

Three rows check the normal contact damper of rigid bodies. The damper is
one-sided: it resists closing motion and acts on the compression gained
after touch. The landing and stacking rows check bounded physical behavior;
the sliding row compares against an independent ideal Coulomb reference.

  Landing row:
      Spheres fall from increasing heights onto a static slab with a stiff
      damper. Their first touch happens at a different point of the time
      step for each height. Every sphere lands softly: it sinks less than
      5 mm into the slab and bounces less than 5 mm.

  Stacking row:
      Four columns of four boxes fall onto each other with the same stiff
      damper and come to rest as straight columns: every box ends within
      5 mm of its rest height and of its column axis, tilted by less than
      one degree. The left two columns share a single solver color group;
      the right two split alternate levels across two color groups so that
      same-color, cross-color, and static-support contacts are all checked.

  Sliding row:
      A rider box on a pushed carrier slips at the Coulomb limit, so it
      accelerates at ``mu * g``. Three lanes use contact damping 0, 100 and
      1000 N*s/m. The damper acts along the normal only: the three riders
      stay nearly level with each other and with the reference marker (white
      line). For flat, settled, continuously sliding contact, backward Euler gives
      ``x = mu * g * dt^2 * n * (n + 1) / 2`` after ``n`` steps, which tends
      to ``mu * g * t^2 / 2``. After 1.2 s of pushing, the marker stops
      and a viscous brake brings the carriers to rest.

Success criteria:
- Landing row: no sphere visibly sinks into the slab or bounces after its first touch.
- Stacking row: both the single-color and mixed-color columns settle straight at rest.
- Sliding row: while the carriers are pushed, the three riders stay level with each other and with
  the white reference line; the carriers move ahead of them.

Runs:
- `python -m newton.examples rigid_contact_damping`
"""

import numpy as np
import warp as wp

import newton
import newton.examples

FPS = 60
SUBSTEPS = 2
ITERATIONS = 10
GRAVITY = 9.81
CONTACT_KE = 1.0e5
STIFF_KD = 1000.0
SLAB_HALF_HEIGHT = 0.05

SPHERE_RADIUS = 0.1
DROP_HEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
LANDING_Y = 1.6

BOX_HALF = (0.1, 0.1, 0.05)
COLUMN_X = (-0.9, -0.3, 0.3, 0.9)
COLUMN_LEVELS = 4
COLUMN_GAP = 0.02
STACKING_Y = 0.6

SLIDING_MU = 0.1
SLIDING_KD = (0.0, 100.0, 1000.0)
SLIDING_Y = (-0.6, -1.2, -1.8)
CARRIER_HALF = (0.9, 0.2, 0.05)
CARRIER_MASS = 2.0
RIDER_MASS = 1.0
RIDER_OFFSET = 0.3
PUSH_FORCE = CARRIER_MASS * 1.5 + SLIDING_MU * RIDER_MASS * GRAVITY
PUSH_START_STEP = 24  # substeps: 0.2 s
PUSH_END_STEP = 168  # substeps: 1.4 s
BRAKE_DAMPING = 20.0


@wp.kernel
def _apply_carrier_drive_kernel(
    step_counter: wp.array[wp.int32],
    carriers: wp.array[wp.int32],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
):
    lane = wp.tid()
    step = step_counter[0]
    carrier = carriers[lane]
    fx = 0.0
    if step >= PUSH_END_STEP:
        fx = -BRAKE_DAMPING * body_qd[carrier][0]
    elif step >= PUSH_START_STEP:
        fx = PUSH_FORCE
    body_f[carrier] = wp.spatial_vector(fx, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _record_damping_metrics_kernel(
    step_counter: wp.array[wp.int32],
    spheres: wp.array[wp.int32],
    riders: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    sphere_lowest: wp.array[float],
    sphere_rebound: wp.array[float],
    sphere_touched: wp.array[wp.int32],
    rider_travel: wp.array[float],
):
    step = step_counter[0] + 1
    step_counter[0] = step
    if step == PUSH_END_STEP:
        for k in range(riders.shape[0]):
            pos = wp.transform_get_translation(body_q[riders[k]])
            rider_travel[k] = pos[0] - RIDER_OFFSET
    for s in range(spheres.shape[0]):
        pos = wp.transform_get_translation(body_q[spheres[s]])
        lowest_point = pos[2] - SPHERE_RADIUS
        if lowest_point < sphere_lowest[s]:
            sphere_lowest[s] = lowest_point
        if sphere_touched[s] != 0 and lowest_point > sphere_rebound[s]:
            sphere_rebound[s] = lowest_point
        if lowest_point <= 0.0:
            sphere_touched[s] = 1


def _shape_cfg(mass, half_extents, kd, mu):
    volume = 8.0 * half_extents[0] * half_extents[1] * half_extents[2]
    return newton.ModelBuilder.ShapeConfig(
        density=mass / volume, ke=CONTACT_KE, kd=kd, mu=mu, mu_torsional=0.0, mu_rolling=0.0
    )


def _add_slab(builder, x, y, half_length, half_width, kd, mu):
    builder.add_shape_box(
        -1,
        xform=wp.transform((x, y, -SLAB_HALF_HEIGHT), wp.quat_identity()),
        hx=half_length,
        hy=half_width,
        hz=SLAB_HALF_HEIGHT,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, ke=CONTACT_KE, kd=kd, mu=mu, mu_torsional=0.0, mu_rolling=0.0),
        color=(0.32, 0.35, 0.38),
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_step = 0

        builder = newton.ModelBuilder()

        # Landing row.
        _add_slab(builder, 0.0, LANDING_Y, 1.8, 0.3, STIFF_KD, 0.5)
        sphere_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, ke=CONTACT_KE, kd=STIFF_KD, mu=0.5, mu_torsional=0.0, mu_rolling=0.0
        )
        self.spheres = []
        for index, height in enumerate(DROP_HEIGHTS):
            x = 0.4 * (index - 0.5 * (len(DROP_HEIGHTS) - 1))
            body = builder.add_body(
                xform=wp.transform((x, LANDING_Y, SPHERE_RADIUS + height), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3) * 0.4 * SPHERE_RADIUS**2),
                lock_inertia=True,
                label=f"Landing sphere: h={height:.1f} m",
            )
            builder.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=sphere_cfg, color=(0.2, 0.65, 0.9))
            self.spheres.append(body)

        # Stacking row: left two columns share one color group; right two alternate across two color groups.
        _add_slab(builder, 0.0, STACKING_Y, 1.3, 0.3, STIFF_KD, 0.5)
        box_cfg = _shape_cfg(1.0, BOX_HALF, STIFF_KD, 0.5)
        same_colors = ((0.65, 0.2, 0.15), (0.78, 0.34, 0.24), (0.88, 0.5, 0.36), (0.95, 0.68, 0.52))
        mixed_colors = ((0.68, 0.42, 0.12), (0.82, 0.56, 0.2), (0.91, 0.69, 0.32), (0.97, 0.81, 0.48))
        self.columns = []
        self.column_rest_positions = []
        mixed_odd_bodies = set()
        for col_idx, x in enumerate(COLUMN_X):
            palette = same_colors if col_idx < 2 else mixed_colors
            for level in range(COLUMN_LEVELS):
                rest_height = BOX_HALF[2] * (2 * level + 1)
                body = builder.add_body(
                    xform=wp.transform((x, STACKING_Y, rest_height + (level + 1) * COLUMN_GAP), wp.quat_identity()),
                    label=f"Column {col_idx + 1}: box {level + 1}",
                )
                builder.add_shape_box(
                    body, hx=BOX_HALF[0], hy=BOX_HALF[1], hz=BOX_HALF[2], cfg=box_cfg, color=palette[level]
                )
                self.columns.append(body)
                self.column_rest_positions.append((x, STACKING_Y, rest_height))
                if col_idx >= 2 and level % 2 == 1:
                    mixed_odd_bodies.add(body)

        # Sliding row: one lane per damping coefficient, on its own frictionless slab.
        rider_colors = ((0.65, 0.84, 0.72), (0.32, 0.75, 0.55), (0.14, 0.62, 0.42))
        self.riders = []
        self.carriers = []
        for lane_idx, (y, kd) in enumerate(zip(SLIDING_Y, SLIDING_KD, strict=True)):
            _add_slab(builder, 0.7, y, 1.8, 0.25, kd, 0.0)
            carrier = builder.add_body(
                xform=wp.transform((0.0, y, CARRIER_HALF[2]), wp.quat_identity()),
                label=f"Sliding carrier: kd={kd:g}",
            )
            builder.add_shape_box(
                carrier,
                hx=CARRIER_HALF[0],
                hy=CARRIER_HALF[1],
                hz=CARRIER_HALF[2],
                cfg=_shape_cfg(CARRIER_MASS, CARRIER_HALF, kd, SLIDING_MU),
                color=(0.48, 0.54, 0.62),
            )
            rider = builder.add_body(
                xform=wp.transform((RIDER_OFFSET, y, 2.0 * CARRIER_HALF[2] + BOX_HALF[2]), wp.quat_identity()),
                label=f"Sliding rider: kd={kd:g}",
            )
            builder.add_shape_box(
                rider,
                hx=BOX_HALF[0],
                hy=BOX_HALF[1],
                hz=BOX_HALF[2],
                cfg=_shape_cfg(RIDER_MASS, BOX_HALF, kd, SLIDING_MU),
                color=rider_colors[lane_idx],
            )
            self.carriers.append(carrier)
            self.riders.append(rider)

        builder.color()
        # Move the odd levels of the mixed-color columns out of every color group into one of their own.
        groups = [
            np.asarray([b for b in group if int(b) not in mixed_odd_bodies], dtype=np.int32)
            for group in builder.body_color_groups
        ]
        builder.body_color_groups = [
            groups[0],
            np.asarray(sorted(mixed_odd_bodies), dtype=np.int32),
            *groups[1:],
        ]

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=ITERATIONS)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        device = self.model.device
        self._spheres_wp = wp.array(self.spheres, dtype=wp.int32, device=device)
        self._carriers_wp = wp.array(self.carriers, dtype=wp.int32, device=device)
        self._riders_wp = wp.array(self.riders, dtype=wp.int32, device=device)
        self._step_counter_wp = wp.zeros(1, dtype=wp.int32, device=device)
        self._sphere_lowest_wp = wp.full(len(self.spheres), 1.0e6, dtype=float, device=device)
        self._sphere_rebound_wp = wp.zeros(len(self.spheres), dtype=float, device=device)
        self._sphere_touched_wp = wp.zeros(len(self.spheres), dtype=wp.int32, device=device)
        self._rider_travel_wp = wp.zeros(len(self.riders), dtype=float, device=device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.5, -4.9, 4.3), -44.0, 90.0)
        self._ref_line_start = wp.empty(1, dtype=wp.vec3, device=device)
        self._ref_line_end = wp.empty(1, dtype=wp.vec3, device=device)

        self.graph = None
        if device.is_cuda and wp.is_mempool_enabled(device):
            self.simulate()
            with wp.ScopedCapture(device=device) as capture:
                self.simulate()
            self.graph = capture.graph
            self._reset_initial_state()

    def _reset_initial_state(self):
        self.state_0.body_q.assign(self.model.body_q)
        self.state_0.body_qd.zero_()
        self.state_1.body_q.assign(self.model.body_q)
        self.state_1.body_qd.zero_()
        self.solver.reset(self.state_0)
        self._step_counter_wp.zero_()
        self._sphere_lowest_wp.fill_(1.0e6)
        self._sphere_rebound_wp.zero_()
        self._sphere_touched_wp.zero_()
        self._rider_travel_wp.zero_()
        self.sim_step = 0
        self.sim_time = 0.0

    def simulate(self):
        device = self.model.device
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                _apply_carrier_drive_kernel,
                dim=len(self.carriers),
                inputs=[self._step_counter_wp, self._carriers_wp, self.state_0.body_qd],
                outputs=[self.state_0.body_f],
                device=device,
            )
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(
                _record_damping_metrics_kernel,
                dim=1,
                inputs=[self._step_counter_wp, self._spheres_wp, self._riders_wp, self.state_0.body_q],
                outputs=[
                    self._sphere_lowest_wp,
                    self._sphere_rebound_wp,
                    self._sphere_touched_wp,
                    self._rider_travel_wp,
                ],
                device=device,
            )

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_step += self.sim_substeps
        self.sim_time = self.sim_step * self.sim_dt

    def _rider_reference(self):
        """Return ideal flat-sliding rider displacement [m] under backward Euler while pushed."""
        steps = min(max(self.sim_step, PUSH_START_STEP), PUSH_END_STEP) - PUSH_START_STEP
        return 0.5 * SLIDING_MU * GRAVITY * self.sim_dt**2 * steps * (steps + 1)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        x = RIDER_OFFSET + self._rider_reference()
        z = 2.0 * CARRIER_HALF[2] + 2.0 * BOX_HALF[2] + 0.01
        self._ref_line_start.assign(np.array([[x, SLIDING_Y[-1] - 0.3, z]], dtype=np.float32))
        self._ref_line_end.assign(np.array([[x, SLIDING_Y[0] + 0.3, z]], dtype=np.float32))
        self.viewer.log_lines("rider_reference", self._ref_line_start, self._ref_line_end, (1.0, 1.0, 1.0), width=0.01)
        self.viewer.end_frame()

    def gui(self, ui):
        """Describe the three damping checks and display live error bounds."""
        ui.text("Back to front:")
        ui.text("1. Blue spheres: soft landing across step phases (kd=1000)")
        ui.text("2. Red/amber columns: single- and mixed-color stacks")
        ui.text("3. Green riders on pushed carriers (kd = 0, 100, 1000):")
        ui.text("   Normal damping leaves Coulomb mu*g slip unchanged")
        ui.text("White line: ideal backward-Euler mu*g reference")
        body_q = self.state_0.body_q.numpy()
        touched = self._sphere_touched_wp.numpy().astype(bool)
        lowest = self._sphere_lowest_wp.numpy()
        rebound_arr = self._sphere_rebound_wp.numpy()
        sinking = 0.0 if not touched.any() else float(np.max(-1000.0 * lowest[touched]))
        rebound = float(1000.0 * rebound_arr.max())
        col_offset = float(1000.0 * np.max(np.abs(body_q[self.columns, :3] - np.asarray(self.column_rest_positions))))
        rider_x = body_q[self.riders, 0].astype(np.float64) - RIDER_OFFSET
        ui.separator()
        ui.text(f"Landing max sink / bounce: {max(sinking, 0.0):.2f} / {rebound:.2f} mm")
        ui.text(f"Stack max position offset: {col_offset:.2f} mm")
        ui.text(f"Rider lane spread: {1000.0 * (rider_x.max() - rider_x.min()):.2f} mm")

    def test_post_step(self):
        """Ensure poses and velocities remain finite on every frame."""
        assert np.isfinite(self.state_0.body_q.numpy()).all(), "Contact poses must remain finite"
        assert np.isfinite(self.state_0.body_qd.numpy()).all(), "Contact velocities must remain finite"

    def test_final(self):
        """Verify landing bounds, resting columns, and rider motion against the declared checks."""
        assert self.sim_step >= PUSH_END_STEP, "Run at least 84 frames to include every landing and the measured push"
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        # Landing row: soft landing, whatever part of the step the first touch falls in.
        touched = self._sphere_touched_wp.numpy().astype(bool)
        assert touched.all(), "Not every sphere reached the slab"
        sinking = -1000.0 * self._sphere_lowest_wp.numpy()
        rebound = 1000.0 * self._sphere_rebound_wp.numpy()
        assert sinking.max() < 5.0, f"A sphere sank {sinking.max():.1f} mm into the slab (limit 5 mm)"
        assert rebound.max() < 5.0, f"A sphere bounced {rebound.max():.1f} mm after its first touch (limit 5 mm)"

        # Stacking row: resting, straight columns (both single-color and mixed-color).
        column_q = body_q[self.columns]
        column_speed = np.linalg.norm(body_qd[self.columns, :3], axis=1)
        offset = 1000.0 * np.abs(column_q[:, :3] - np.asarray(self.column_rest_positions))
        tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (column_q[:, 3] ** 2 + column_q[:, 4] ** 2), -1.0, 1.0)))
        assert column_speed.max() < 0.05, f"A stacked box still moves at {column_speed.max():.3f} m/s (limit 0.05)"
        assert offset.max() < 5.0, f"A stacked box is {offset.max():.1f} mm from its rest position (limit 5 mm)"
        assert tilt.max() < 1.0, f"A stacked box is tilted by {tilt.max():.2f} degrees (limit 1 degree)"

        # Sliding row: every rider follows mu * g, and damping does not change the motion.
        travel = self._rider_travel_wp.numpy().astype(np.float64)
        reference = self._rider_reference()
        error = 100.0 * (travel - reference) / reference
        assert np.abs(error).max() < 5.0, (
            f"Rider travel differs from the closed form by {error[np.abs(error).argmax()]:+.1f} % (limit 5 %)"
        )
        spread = 100.0 * (travel.max() - travel.min()) / reference
        assert spread < 1.0, f"Rider travel differs by {spread:.2f} % between damping lanes (limit 1 %)"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=120)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
