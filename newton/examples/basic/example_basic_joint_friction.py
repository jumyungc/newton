# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Joint Friction (Pendulum + RK4 Validation)
#
# Five identical pendulums released from horizontal, each with a different
# Coulomb joint-friction torque mu (units: N*m). The pendulum + Coulomb
# friction problem has no closed-form solution (sin(theta) is nonlinear,
# sign(qd) is piecewise), so we validate against RK4 numerical integration
# of the same ODE:
#
#     I_pivot * theta_dd = m * g * (L/2) * cos(theta) - mu * sign(theta_d)
#
# (theta = 0 horizontal, theta = pi/2 hanging down; gravity along -Z.)
# The simulator and an in-line RK4 reference march the same equation in
# parallel; their angles and velocities are streamed via
# viewer.log_scalar so you see them overlap in the GL viewer.
#
# Command: uv run -m newton.examples basic_joint_friction
#
###########################################################################

import collections
import math

import numpy as np
import warp as wp

import newton
import newton.examples

try:
    from imgui_bundle import imgui, implot

    _HAS_IMPLOT = True
except ImportError:
    _HAS_IMPLOT = False


# Friction torques (N*m) per pendulum. mu=0 is included to visualize the
# integrator's implicit-Euler dissipation against the conservative RK4
# reference (mu=0 case will diverge from RK4 over many oscillations).
FRICTIONS = [0.0, 0.05, 0.2, 0.5, 1.5]
SPACING = 1.4
PIVOT_HEIGHT = 2.0

# Explicit physical parameters used by both the simulator (via add_link
# kwargs) and the RK4 reference, so the two integrate the same ODE.
ROD_MASS = 1.0
ROD_LENGTH = 1.0
ROD_RADIUS = 0.04
GRAVITY_MAG = 9.81

I_COM = (1.0 / 12.0) * ROD_MASS * ROD_LENGTH * ROD_LENGTH
I_PIVOT = I_COM + ROD_MASS * (ROD_LENGTH / 2.0) ** 2  # = (1/3) m L^2 = 1/3


def _read_joint_state(model, state):
    joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=state.body_q.device)
    joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=state.body_q.device)
    newton.eval_ik(model, state, joint_q, joint_qd)
    return joint_q.numpy(), joint_qd.numpy()


def _pendulum_deriv(theta, theta_d, mu):
    """RHS of the pendulum + Coulomb-friction ODE.

    Returns (theta_dot, theta_ddot).
    """
    sgn = 1.0 if theta_d > 0.0 else (-1.0 if theta_d < 0.0 else 0.0)
    grav_torque = ROD_MASS * GRAVITY_MAG * (ROD_LENGTH / 2.0) * math.cos(theta)
    fric_torque = mu * sgn
    theta_dd = (grav_torque - fric_torque) / I_PIVOT
    return theta_d, theta_dd


def _rk4_step(theta, theta_d, mu, dt):
    k1 = _pendulum_deriv(theta, theta_d, mu)
    k2 = _pendulum_deriv(theta + 0.5 * dt * k1[0], theta_d + 0.5 * dt * k1[1], mu)
    k3 = _pendulum_deriv(theta + 0.5 * dt * k2[0], theta_d + 0.5 * dt * k2[1], mu)
    k4 = _pendulum_deriv(theta + dt * k3[0], theta_d + dt * k3[1], mu)
    new_theta = theta + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    new_theta_d = theta_d + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    return new_theta, new_theta_d


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # RK4 substeps per simulator substep (kept small enough to be ground truth).
        self.rk4_substeps = 8
        self.rk4_dt = self.sim_dt / self.rk4_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(gravity=-GRAVITY_MAG, up_axis=2)

        # Build five pendulums sharing identical mass/inertia/length but with
        # different friction torques.
        self.dof_indices = []
        for i, mu in enumerate(FRICTIONS):
            x_offset = (i - (len(FRICTIONS) - 1) * 0.5) * SPACING

            # Body frame origin at the pivot; COM offset by (L/2, 0, 0); inertia
            # is specified about the COM.
            link = builder.add_link(
                xform=wp.transform(wp.vec3(x_offset, 0.0, PIVOT_HEIGHT), wp.quat_identity()),
                mass=ROD_MASS,
                com=wp.vec3(ROD_LENGTH / 2.0, 0.0, 0.0),
                inertia=wp.mat33(I_COM, 0.0, 0.0, 0.0, I_COM, 0.0, 0.0, 0.0, I_COM),
                lock_inertia=True,
            )

            # Visual capsule centered at the COM (so it spans pivot to end of rod).
            builder.add_shape_capsule(
                body=link,
                radius=ROD_RADIUS,
                half_height=ROD_LENGTH / 2.0,
                xform=wp.transform(wp.vec3(ROD_LENGTH / 2.0, 0.0, 0.0), wp.quat_identity()),
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            )

            j = builder.add_joint_revolute(
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent=-1,
                child=link,
                parent_xform=wp.transform(wp.vec3(x_offset, 0.0, PIVOT_HEIGHT), wp.quat_identity()),
                child_xform=wp.transform_identity(),
                target_ke=0.0,
                target_kd=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
                armature=0.0,
                friction=mu,
            )
            self.dof_indices.append(builder.joint_dof_count - 1)

            # Released from horizontal: joint_q = 0 means rod points along +X.
            builder.joint_q[self.dof_indices[-1]] = 0.0
            builder.joint_qd[self.dof_indices[-1]] = 0.0

            builder.add_articulation([j], label=f"pendulum_mu_{mu:.2f}")

        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=2)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)

        # Side-on view of the swinging plane: camera south of the rods,
        # looking +Y, slightly above the pivot row. Yaw=90 puts the +Y axis
        # into the screen so the pendulums swing in the visible X-Z plane.
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.0, -7.0, PIVOT_HEIGHT), pitch=-5.0, yaw=90.0)

        # RK4 reference state. We validate against RK4 in BOTH q and qd
        # internally (test_final asserts both), but plot only q_sim vs q_rk4
        # for visual clarity (one panel per pendulum, two overlaid traces).
        self._theta_rk4 = [0.0] * len(FRICTIONS)
        self._qd_rk4 = [0.0] * len(FRICTIONS)
        self._max_q_err = [0.0] * len(FRICTIONS)
        self._max_qd_err = [0.0] * len(FRICTIONS)

        # Rolling history buffers used by the ImPlot overlay callback.
        history = 600  # ~10 s at 60 fps
        self._q_sim_hist = [collections.deque(maxlen=history) for _ in FRICTIONS]
        self._q_rk4_hist = [collections.deque(maxlen=history) for _ in FRICTIONS]
        self._t_hist = collections.deque(maxlen=history)

        # Register a custom overlay window: per pendulum, one ImPlot chart
        # with q_sim and q_rk4 drawn together (sim default color, RK4 red).
        if _HAS_IMPLOT and hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_overlay, position="free")

        muvec = "  ".join(f"mu={mu:.2f}" for mu in FRICTIONS)
        print(f"\n[joint-friction pendulum] {len(FRICTIONS)} pendulums with RK4 reference")
        print(f"  rod: m={ROD_MASS}, L={ROD_LENGTH}, I_pivot={I_PIVOT:.4f}, g={GRAVITY_MAG}")
        print(f"  per-pendulum friction (N*m):  {muvec}")
        print(f"  ODE: I_pivot * theta_dd = m*g*(L/2)*cos(theta) - mu*sign(theta_d)")
        print(f"  RK4 inner step = sim_dt / {self.rk4_substeps} = {self.rk4_dt*1e6:.1f} us\n")

        self.capture()

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
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

        # March RK4 reference forward by frame_dt using sim_substeps * rk4_substeps inner steps.
        for _ in range(self.sim_substeps):
            for _ in range(self.rk4_substeps):
                for i, mu in enumerate(FRICTIONS):
                    self._theta_rk4[i], self._qd_rk4[i] = _rk4_step(
                        self._theta_rk4[i], self._qd_rk4[i], mu, self.rk4_dt
                    )

        # Track both q and qd errors against RK4 for the validation summary;
        # plot only q (sim and RK4 overlaid in one panel per pendulum) so
        # the viewer is uncluttered.
        q_sim, qd_sim = _read_joint_state(self.model, self.state_0)
        for i, mu in enumerate(FRICTIONS):
            dof = self.dof_indices[i]
            q_s = float(q_sim[dof])
            qd_s = float(qd_sim[dof])
            q_r = self._theta_rk4[i]
            qd_r = self._qd_rk4[i]
            q_err = q_s - q_r
            qd_err = qd_s - qd_r
            if abs(q_err) > self._max_q_err[i]:
                self._max_q_err[i] = abs(q_err)
            if abs(qd_err) > self._max_qd_err[i]:
                self._max_qd_err[i] = abs(qd_err)

            self._q_sim_hist[i].append(q_s)
            self._q_rk4_hist[i].append(q_r)
        self._t_hist.append(self.sim_time)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _render_overlay(self, _imgui_module):
        """Custom ImGui+ImPlot window: per pendulum, sim and RK4 angles overlaid.

        Sim trace uses ImPlot's default color; RK4 trace is forced red so the
        two are easy to distinguish.
        """
        if not _HAS_IMPLOT:
            return
        n = len(self._t_hist)
        if n < 2:
            return

        # ImPlot needs its own context, separate from the viewer's ImGui
        # context. Create it on first use.
        if not getattr(self, "_implot_ctx_created", False):
            implot.create_context()
            self._implot_ctx_created = True

        # Pin the overlay window to the right edge of the viewport on first
        # appearance so it doesn't land on top of the 3D scene.
        io = imgui.get_io()
        win_w = 460.0
        win_h = max(420.0, min(io.display_size.y - 40.0, 160.0 * len(FRICTIONS) + 60.0))
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size.x - win_w - 10.0, 10.0),
            imgui.Cond_.appearing,
        )
        imgui.set_next_window_size(imgui.ImVec2(win_w, win_h), imgui.Cond_.appearing)

        if imgui.begin("Pendulum: q_sim (blue) vs q_rk4 (red)")[0]:
            t = np.fromiter(self._t_hist, dtype=np.float32)
            red = imgui.ImVec4(0.95, 0.20, 0.20, 1.0)
            blue = imgui.ImVec4(0.20, 0.55, 1.0, 1.0)
            cond_always = imgui.Cond_.always.value

            # Sliding window: show last WINDOW_SEC of history, full pendulum
            # angle range (released from 0; oscillates roughly in [0, pi]).
            WINDOW_SEC = 5.0
            t_max = float(t[-1])
            t_min = max(0.0, t_max - WINDOW_SEC)
            Y_MIN, Y_MAX = -0.4, 3.5  # rad (~ -23 deg .. 200 deg)

            for i, mu in enumerate(FRICTIONS):
                q_sim = np.fromiter(self._q_sim_hist[i], dtype=np.float32)
                q_rk4 = np.fromiter(self._q_rk4_hist[i], dtype=np.float32)
                m = min(t.size, q_sim.size, q_rk4.size)
                if m < 2:
                    continue
                tt = t[-m:]
                implot.set_next_axes_limits(t_min, t_max, Y_MIN, Y_MAX, cond=cond_always)
                if implot.begin_plot(f"P{i}  mu={mu:.2f}", imgui.ImVec2(-1, 140)):
                    implot.setup_axes("t [s]", "theta [rad]")
                    # Draw RK4 first (red, thicker), then sim (blue, thinner)
                    # on top so blue shows centered when they agree.
                    implot.plot_line(
                        "q_rk4",
                        tt,
                        q_rk4[-m:],
                        spec=implot.Spec(line_color=red, line_weight=4.0),
                    )
                    implot.plot_line(
                        "q_sim",
                        tt,
                        q_sim[-m:],
                        spec=implot.Spec(line_color=blue, line_weight=2.0),
                    )
                    implot.end_plot()
        imgui.end()

    def test_final(self):
        # Tolerances reflect implicit-Euler discretization difference vs RK4 at
        # the example's chosen iteration count. Loosen if you reduce iterations
        # further; tighten if you raise them.
        TOL_Q = 0.50   # rad
        TOL_QD = 1.20  # rad/s
        print("\n[pendulum vs RK4 summary]")
        for i, mu in enumerate(FRICTIONS):
            ok_q = self._max_q_err[i] < TOL_Q
            ok_qd = self._max_qd_err[i] < TOL_QD
            tag = "OK  " if (ok_q and ok_qd) else "FAIL"
            print(f"  {tag} mu={mu:.2f}  max|q_sim - q_rk4|={self._max_q_err[i]:.4f}  "
                  f"max|qd_sim - qd_rk4|={self._max_qd_err[i]:.4f}")
            assert ok_q, f"P{i} q error {self._max_q_err[i]:.4f} exceeds {TOL_Q}"
            assert ok_qd, f"P{i} qd error {self._max_qd_err[i]:.4f} exceeds {TOL_QD}"
        print("  -> all pendulums match RK4 reference (q and qd) within tolerance")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
