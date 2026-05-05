# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for VBD per-DOF Coulomb joint friction.

Friction is modeled as a regularized Coulomb force ``f = -mu * tanh(qd / eps)``
applied along each free joint DOF. The tests below verify the model against
closed-form analytic expectations for revolute, prismatic, and multi-DOF cases.
"""

import unittest

import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices(mode="basic")

DT = 1.0 / 240.0


def _build_revolute_with_friction(device, mu, qd0=0.0, q0=0.0, mass=1.0, inertia=1.0, gravity=0.0):
    builder = newton.ModelBuilder(gravity=gravity)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=mass,
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
    builder.joint_q[0] = q0
    builder.joint_qd[0] = qd0
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device)


def _build_prismatic_with_friction(device, mu, qd0=0.0, q0=0.0, mass=1.0, gravity=0.0):
    builder = newton.ModelBuilder(gravity=gravity)
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
    builder.joint_q[0] = q0
    builder.joint_qd[0] = qd0
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device)


def _step(model, solver, n_steps, dt=DT, control=None):
    state_in = model.state()
    state_out = model.state()
    if control is None:
        control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    for _ in range(n_steps):
        model.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
    return state_in


def _read_joint_state(model, state):
    joint_q = wp.zeros(model.joint_coord_count, dtype=float, device=state.body_q.device)
    joint_qd = wp.zeros(model.joint_dof_count, dtype=float, device=state.body_q.device)
    newton.eval_ik(model, state, joint_q, joint_qd)
    return joint_q.numpy(), joint_qd.numpy()


# -------------------------------------------------------------
# T3: Coast-down - the cleanest analytic check.
# Free-spinning revolute with friction: qd should decay linearly
# at rate -mu/I until it stops (within regularization tolerance).
# -------------------------------------------------------------
def test_coast_down_revolute(test, device):
    mu = 1.0
    qd0 = 2.0
    inertia = 1.0
    model = _build_revolute_with_friction(device, mu=mu, qd0=qd0, inertia=inertia)
    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_avbd_beta=1.0e5)

    # After 1 second of coast-down, expect qd ~= qd0 - mu/I * 1.0 = 1.0
    n_steps_1s = int(round(1.0 / DT))
    state = _step(model, solver, n_steps=n_steps_1s)
    _, qd = _read_joint_state(model, state)
    expected_qd = qd0 - mu / inertia * (n_steps_1s * DT)
    test.assertAlmostEqual(float(qd[0]), expected_qd, delta=0.10,
                           msg=f"After 1 s coast-down, qd={qd[0]:.4f} vs expected {expected_qd:.4f}")

# -------------------------------------------------------------
# T1: Static stick - subthreshold drive does not move the joint.
# -------------------------------------------------------------
def test_static_stick_revolute(test, device):
    mu = 1.0
    drive_torque_target = 0.5  # below mu, friction should hold
    inertia = 1.0
    # Use a target_pos with a stiff drive that *would* exert ~0.5 N·m on the
    # zero-position joint, but friction holds it.
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=wp.mat33(inertia, 0.0, 0.0, 0.0, inertia, 0.0, 0.0, 0.0, inertia),
        lock_inertia=True,
    )
    drive_ke = 100.0
    target_pos = drive_torque_target / drive_ke  # so drive_ke * (target - 0) = drive_torque_target
    joint = builder.add_joint_revolute(
        axis=2,
        parent=-1,
        child=body,
        target_ke=drive_ke,
        target_kd=0.1,
        target_pos=target_pos,
        limit_ke=0.0,
        limit_kd=0.0,
        armature=0.0,
        friction=mu,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=15, rigid_avbd_beta=1.0e5)

    state = _step(model, solver, n_steps=int(round(0.5 / DT)))
    q, qd = _read_joint_state(model, state)

    # Joint should remain near zero - small creep from regularization is OK
    test.assertLess(abs(float(q[0])), 0.05,
                    msg=f"Static stick: q={q[0]:.4f} should be near 0 (subthreshold drive {drive_torque_target} N·m vs friction {mu} N·m)")
    test.assertLess(abs(float(qd[0])), 0.05,
                    msg=f"Static stick: qd={qd[0]:.4f} should be near 0")


# -------------------------------------------------------------
# T2: Sliding - super-threshold drive accelerates with net torque
# (drive - friction).
# -------------------------------------------------------------
def test_sliding_revolute(test, device):
    mu = 0.5
    drive_torque = 2.0
    inertia = 1.0

    # Apply constant external joint torque via control.joint_f (feedforward).
    # Net torque (in sliding regime) = drive_torque - mu.
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
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=15, rigid_avbd_beta=1.0e5)

    control = model.control()
    control.joint_f = wp.array([drive_torque], dtype=float, device=device)

    n_steps = int(round(0.5 / DT))
    state = _step(model, solver, n_steps=n_steps, control=control)
    _, qd = _read_joint_state(model, state)

    expected_qd = (drive_torque - mu) / inertia * (n_steps * DT)
    test.assertAlmostEqual(float(qd[0]), expected_qd, delta=0.15,
                           msg=f"Sliding: qd={qd[0]:.4f} vs expected {expected_qd:.4f} "
                               f"(drive={drive_torque}, mu={mu})")


# -------------------------------------------------------------
# T5: Prismatic coast-down (same as T3 but linear DOF).
# -------------------------------------------------------------
def test_coast_down_prismatic(test, device):
    mu = 1.0
    qd0 = 2.0
    mass = 1.0
    model = _build_prismatic_with_friction(device, mu=mu, qd0=qd0, mass=mass)
    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_avbd_beta=1.0e5)

    n_steps_1s = int(round(1.0 / DT))
    state = _step(model, solver, n_steps=n_steps_1s)
    _, qd = _read_joint_state(model, state)
    expected_qd = qd0 - mu / mass * (n_steps_1s * DT)
    test.assertAlmostEqual(float(qd[0]), expected_qd, delta=0.10,
                           msg=f"Prismatic coast-down: qd={qd[0]:.4f} vs expected {expected_qd:.4f}")


# -------------------------------------------------------------
# T6: Multi-DOF independence - two joints with different friction values.
# Each should obey its own mu.
# -------------------------------------------------------------
def test_multi_dof_independence(test, device):
    mu_a = 0.5
    mu_b = 2.0
    qd0 = 3.0

    builder = newton.ModelBuilder(gravity=0.0)
    body_a = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    body_b = builder.add_link(
        xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    joint_a = builder.add_joint_revolute(
        axis=2,
        parent=-1,
        child=body_a,
        target_ke=0.0,
        target_kd=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        armature=0.0,
        friction=mu_a,
    )
    joint_b = builder.add_joint_revolute(
        axis=2,
        parent=-1,
        child=body_b,
        parent_xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
        target_ke=0.0,
        target_kd=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        armature=0.0,
        friction=mu_b,
    )
    builder.joint_qd[0] = qd0
    builder.joint_qd[1] = qd0
    builder.add_articulation([joint_a])
    builder.add_articulation([joint_b])
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_avbd_beta=1.0e5)

    n_steps = int(round(0.5 / DT))
    state = _step(model, solver, n_steps=n_steps)
    _, qd = _read_joint_state(model, state)

    # Each joint decays at rate mu_i / I_i = mu_i (since I = 1)
    expected_qd_a = qd0 - mu_a * (n_steps * DT)
    expected_qd_b = qd0 - mu_b * (n_steps * DT)
    test.assertAlmostEqual(float(qd[0]), expected_qd_a, delta=0.10,
                           msg=f"Joint A (mu={mu_a}): qd={qd[0]:.4f} vs {expected_qd_a:.4f}")
    test.assertAlmostEqual(float(qd[1]), expected_qd_b, delta=0.10,
                           msg=f"Joint B (mu={mu_b}): qd={qd[1]:.4f} vs {expected_qd_b:.4f}")


# -------------------------------------------------------------
# T7: Energy dissipation - kinetic energy decreases monotonically
# under friction with no external work.
# -------------------------------------------------------------
def test_energy_dissipation(test, device):
    mu = 2.0
    qd0 = 2.0
    inertia = 1.0
    model = _build_revolute_with_friction(device, mu=mu, qd0=qd0, inertia=inertia)
    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_avbd_beta=1.0e5)

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    energies = []
    for _ in range(int(round(0.8 / DT))):
        _, qd = _read_joint_state(model, state_in)
        energies.append(0.5 * inertia * float(qd[0]) ** 2)
        model.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, DT)
        state_in, state_out = state_out, state_in

    # Monotonic non-increase (allow tiny numerical jitter)
    for i in range(1, len(energies)):
        test.assertLessEqual(energies[i], energies[i - 1] + 1e-3,
                             msg=f"Energy increased at step {i}: {energies[i]:.6f} > {energies[i-1]:.6f}")

    # Final energy should be much smaller than initial
    test.assertLess(energies[-1], 0.2 * energies[0],
                    msg=f"Final energy {energies[-1]:.4f} not << initial {energies[0]:.4f}")


# -------------------------------------------------------------
# T8: Zero friction baseline - regression check that mu=0 gives no friction.
# -------------------------------------------------------------
def test_zero_friction_no_effect(test, device):
    qd0 = 2.0
    inertia = 1.0
    model = _build_revolute_with_friction(device, mu=0.0, qd0=qd0, inertia=inertia)
    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_avbd_beta=1.0e5)

    n_steps = int(round(0.5 / DT))
    state = _step(model, solver, n_steps=n_steps)
    _, qd = _read_joint_state(model, state)
    # No friction, no drive, no gravity - velocity preserved
    test.assertAlmostEqual(float(qd[0]), qd0, delta=0.01,
                           msg=f"Zero-friction: qd={qd[0]:.4f} should equal qd0={qd0}")


class TestJointFriction(unittest.TestCase):
    pass


add_function_test(TestJointFriction, "test_coast_down_revolute", test_coast_down_revolute, devices=devices)
add_function_test(TestJointFriction, "test_static_stick_revolute", test_static_stick_revolute, devices=devices)
add_function_test(TestJointFriction, "test_sliding_revolute", test_sliding_revolute, devices=devices)
add_function_test(TestJointFriction, "test_coast_down_prismatic", test_coast_down_prismatic, devices=devices)
add_function_test(TestJointFriction, "test_multi_dof_independence", test_multi_dof_independence, devices=devices)
add_function_test(TestJointFriction, "test_energy_dissipation", test_energy_dissipation, devices=devices)
add_function_test(TestJointFriction, "test_zero_friction_no_effect", test_zero_friction_no_effect, devices=devices)


if __name__ == "__main__":
    unittest.main()
