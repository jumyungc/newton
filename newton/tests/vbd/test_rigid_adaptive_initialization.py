# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Adaptive initialization starts held bodies where they are held and free bodies in free flight."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_vbd_kernels import forward_step_rigid_bodies
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_loaded_joint_maintains_force_balance(test, device):
    """Maintain initialized force balance with few iterations at a large step.

    Two links slide on world prismatic joints and are tied by a fixed joint. One has a 1000 N/m drive
    spring, the other carries 20 N: balance is at 20 mm, where both bodies start. Starting every step
    from the free-flight pose leaves the pair at about 0.45 m with these settings.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    for index in range(2):
        body = builder.add_link(mass=0.1, inertia=wp.mat33(np.eye(3) * 0.001), lock_inertia=True)
        joints.append(
            builder.add_joint_prismatic(
                -1,
                body,
                axis=newton.Axis.X,
                parent_xform=wp.transform(wp.vec3(0.0, index * 0.1, 0.0), wp.quat_identity()),
                target_ke=1000.0 if index == 0 else 0.0,
                target_kd=20.0 if index == 0 else 0.0,
            )
        )
    builder.add_joint_fixed(0, 1, parent_xform=wp.transform(wp.vec3(0.0, 0.1, 0.0), wp.quat_identity()))
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    model.joint_q.assign([0.02, 0.02])
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    state_0, state_1 = model.state(), model.state()
    solver = newton.solvers.SolverVBD(model, iterations=10, rigid_adaptive_initialization=True)
    force = np.zeros((2, 6), dtype=np.float32)
    force[1, 0] = 20.0
    for _ in range(240):
        state_0.body_f.assign(force)
        solver.step(state_0, state_1, None, None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
    test.assertAlmostEqual(float(state_0.body_q.numpy()[0, 0]), 0.02, delta=1.0e-3)


def test_free_jointed_chain_falls_freely(test, device):
    """Jointed bodies in free fall keep the free-fall height with two iterations.

    A start without the applied acceleration makes a jointed chain lag gravity when the iterations are few.
    Coupling restarts must reject adaptive initialization until its acceleration history can be restored.
    """
    gravity, length, rate, steps = 9.81, 0.1, 60.0, 30
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))
    joints, previous = [], -1
    for index in range(4):
        body = builder.add_link(
            xform=wp.transform(wp.vec3((index + 0.5) * length, 0.0, 1.0), wp.quat_identity()),
            mass=0.2,
            inertia=wp.mat33(np.diag([1.0e-5, 1.7e-4, 1.7e-4])),
            lock_inertia=True,
        )
        if index == 0:
            joints.append(builder.add_joint_free(body))
        else:
            joints.append(
                builder.add_joint_revolute(
                    previous,
                    body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
                )
            )
        previous = body
    builder.add_articulation(joints)
    independent_body = builder.add_body(
        xform=wp.transform(wp.vec3(3.0, 0.0, 1.0), wp.quat_identity()),
        mass=0.2,
        inertia=wp.mat33(np.eye(3) * 0.001),
        lock_inertia=True,
    )
    builder.color()
    model = builder.finalize(device=device)
    expected = 1.0 - gravity * steps * (steps + 1) / (2.0 * rate * rate)

    def create_solver(adaptive, coupling_iterations):
        if coupling_iterations is None:
            return newton.solvers.SolverVBD(model, iterations=2, rigid_adaptive_initialization=adaptive)
        return SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda view: newton.solvers.SolverVBD(
                        view, iterations=2, rigid_adaptive_initialization=adaptive
                    ),
                    bodies=list(range(4)),
                    joints=joints,
                ),
                SolverCoupled.Entry(
                    name="semi",
                    solver=newton.solvers.SolverSemiImplicit,
                    bodies=[independent_body],
                    joints=[],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                # No feedback: another pass repeats the same interval and must preserve free fall.
                proxies=[SolverCoupledProxy.Proxy(source="vbd", destination="semi", bodies=[0], proxy_relaxation=0.0)],
                iterations=coupling_iterations,
            ),
        )

    for adaptive, coupling_iterations in ((True, None), (True, 1), (False, 2), (True, 2)):
        with test.subTest(adaptive=adaptive, coupling_iterations=coupling_iterations):
            if adaptive and coupling_iterations == 2:
                with test.assertRaisesRegex(ValueError, "rigid_adaptive_initialization.*coupling.*restart"):
                    create_solver(adaptive, coupling_iterations)
                continue
            solver = create_solver(adaptive, coupling_iterations)
            state_0, state_1 = model.state(), model.state()
            for _ in range(steps):
                state_0.clear_forces()
                solver.step(state_0, state_1, None, None, 1.0 / rate)
                state_0, state_1 = state_1, state_0
            np.testing.assert_allclose(state_0.body_q.numpy()[:, 2], expected, atol=1.0e-4)
            np.testing.assert_allclose(state_0.body_qd.numpy()[:, 2], -gravity * steps / rate, atol=1.0e-4)


def test_zero_iterations_keep_free_integration(test, device):
    """Without iterations the step is the free integration, also for a force that was absent one step earlier."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.01), lock_inertia=True)
    builder.color()
    model = builder.finalize(device=device)
    state_0, state_1 = model.state(), model.state()
    solver = newton.solvers.SolverVBD(model, iterations=0, rigid_adaptive_initialization=True)
    dt = 0.01
    force = np.zeros((1, 6), dtype=np.float32)
    for applied in (0.0, 4.0):
        force[0, 0] = applied
        state_0.body_f.assign(force)
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0
    test.assertAlmostEqual(float(state_0.body_q.numpy()[0, 0]), 4.0 / 2.0 * dt * dt, delta=1.0e-8)


def test_external_rigid_solver_is_rejected(test, device):
    """The option changes the internal rigid iterations and has no meaning without them."""
    builder = newton.ModelBuilder()
    builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01), lock_inertia=True)
    builder.color()
    model = builder.finalize(device=device)
    with test.assertRaisesRegex(ValueError, "rigid_adaptive_initialization"):
        newton.solvers.SolverVBD(model, integrate_with_external_rigid_solver=True, rigid_adaptive_initialization=True)
    default_solver = newton.solvers.SolverVBD(model)
    adaptive_solver = newton.solvers.SolverVBD(model, rigid_adaptive_initialization=True)
    test.assertFalse(default_solver.rigid_adaptive_initialization)
    test.assertTrue(adaptive_solver.rigid_adaptive_initialization)
    with test.assertRaises(AttributeError):
        default_solver.rigid_adaptive_initialization = True


def test_resumed_iterations_preserve_galilean_invariance(test, device):
    """Preserve relative motion after resuming iterations in uniformly translating connected bodies.

    Two force-free steps without iterations must still advance the acceleration history. Otherwise
    resuming with a load interprets the common velocity as acceleration and changes the relative motion.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for index in range(2):
        builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.1 * index, 0.0), wp.quat_identity()),
            mass=0.1,
            inertia=wp.mat33(np.eye(3) * 0.001),
            lock_inertia=True,
        )
    free = builder.add_joint_free(0)
    fixed = builder.add_joint_fixed(0, 1, parent_xform=wp.transform(wp.vec3(0.0, 0.1, 0.0), wp.quat_identity()))
    builder.add_articulation([free, fixed])
    builder.color()
    model = builder.finalize(device=device)
    dt = 0.01
    results = []
    for boost in (0.0, 0.1):
        state_0, state_1 = model.state(), model.state()
        velocity = np.zeros((2, 6), dtype=np.float32)
        velocity[:, 1] = boost
        state_0.body_qd.assign(velocity)
        solver = newton.solvers.SolverVBD(model, iterations=0, rigid_adaptive_initialization=True)
        for _ in range(2):
            state_0.clear_forces()
            solver.step(state_0, state_1, None, None, dt)
            state_0, state_1 = state_1, state_0
        solver.iterations = 2
        force = np.zeros((2, 6), dtype=np.float32)
        force[1, 1] = 20.0
        state_0.body_f.assign(force)
        solver.step(state_0, state_1, None, None, dt)
        position = state_1.body_q.numpy()[:, :3].astype(np.float64)
        velocity = state_1.body_qd.numpy()[:, :3].astype(np.float64)
        position[:, 1] -= boost * 3.0 * dt
        velocity[:, 1] -= boost
        results.append((position, velocity))
    # Allow float32 differencing error at the 0.1 m position scale and 0.01 s step.
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=0.0, atol=1.0e-4)


def test_start_keeps_only_acceleration_along_the_applied_direction(test, device):
    """The starting pose keeps the applied acceleration only as far as the body realized it along it.

    Three bodies under gravity whose previous step ended with a sideways, an upward and a half-gravity
    downward velocity change. The first two omit the gravity displacement; the third retains half of it.
    Each asymmetric body also spins steadily under a torque that balances its gyroscopic term. Keep
    that angular prediction unchanged: zero realized angular acceleration does not imply zero torque.
    """
    gravity, dt = 10.0, 0.01
    realized = np.array([[gravity, 0.0, 0.0], [0.0, 0.0, 0.5 * gravity], [0.0, 0.0, -0.5 * gravity]])
    count = len(realized)
    velocity = np.zeros((count, 6), dtype=np.float32)
    velocity[:, :3] = realized * dt
    inertia = np.diag([0.02, 0.03, 0.04])
    omega = np.array([4.0, 5.0, 6.0])
    velocity[:, 3:] = omega
    previous_velocity = np.zeros_like(velocity)
    previous_velocity[:, 3:] = omega
    force = np.zeros_like(velocity)
    force[:, 3:] = np.cross(omega, inertia @ omega)
    pose = np.tile(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (count, 1))
    body_q = wp.array(pose, dtype=wp.transform, device=device)
    body_inertia_q = wp.zeros(count, dtype=wp.transform, device=device)
    wp.launch(
        forward_step_rigid_bodies,
        dim=count,
        inputs=[
            dt,
            1,
            wp.array([dt], dtype=float, device=device),
            wp.array([wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=device),
            wp.zeros(count, dtype=wp.int32, device=device),
            wp.zeros(2, dtype=wp.bool, device=device),
            wp.array(force, dtype=wp.spatial_vector, device=device),
            wp.zeros(count, dtype=wp.vec3, device=device),
            wp.array([inertia] * count, dtype=wp.mat33, device=device),
            wp.ones(count, dtype=float, device=device),
            wp.array([np.linalg.inv(inertia)] * count, dtype=wp.mat33, device=device),
            body_q,
            wp.array(velocity, dtype=wp.spatial_vector, device=device),
            wp.array(previous_velocity, dtype=wp.spatial_vector, device=device),
        ],
        outputs=[wp.zeros(count, dtype=wp.transform, device=device), body_inertia_q, None, None],
        device=device,
    )
    start, target = body_q.numpy(), body_inertia_q.numpy()
    carried = pose[:, :3] + velocity[:, :3] * dt
    fall = np.array([0.0, 0.0, -gravity * dt * dt])
    np.testing.assert_allclose(target[:, :3], carried + fall, atol=1.0e-6)
    np.testing.assert_allclose(start[:, :3], carried + np.outer([0.0, 0.0, 0.5], fall), atol=1.0e-6)
    expected_rotation = np.concatenate((0.5 * dt * omega, [1.0]))
    expected_rotation /= np.linalg.norm(expected_rotation)
    expected_rotations = np.tile(expected_rotation, (count, 1))
    np.testing.assert_allclose(target[:, 3:], expected_rotations, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(start[:, 3:], expected_rotations, rtol=0.0, atol=1.0e-6)


def test_masked_reset_restores_free_fall_only_in_selected_world(test, device):
    """Resetting one world restores its first-step free-fall start while preserving another world's held start."""
    gravity, mass, dt = 10.0, 0.2, 0.01
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))
    for _ in range(2):
        builder.begin_world()
        b0 = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            mass=mass,
            inertia=wp.mat33(np.eye(3) * 1.0e-3),
            lock_inertia=True,
        )
        b1 = builder.add_link(
            xform=wp.transform(wp.vec3(0.1, 0.0, 1.0), wp.quat_identity()),
            mass=mass,
            inertia=wp.mat33(np.eye(3) * 1.0e-3),
            lock_inertia=True,
        )
        j0 = builder.add_joint_free(b0)
        j1 = builder.add_joint_fixed(
            b0,
            b1,
            parent_xform=wp.transform(wp.vec3(0.05, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.05, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j0, j1])
        builder.end_world()
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_adaptive_initialization=True)
    state_0, state_1 = model.state(), model.state()

    # Step 0 balances gravity with upward external force so both worlds record zero realized acceleration.
    hold_force = np.zeros((model.body_count, 6), dtype=np.float32)
    hold_force[:, 2] = mass * gravity
    state_0.body_f.assign(hold_force)
    solver.step(state_0, state_1, None, None, dt)
    state_0, state_1 = state_1, state_0

    # Release the upward force and reset only world 0: world 0 starts both links in free fall (-g * dt^2),
    # whereas unreset world 1 starts at the held pose (z = 1) so 1 Gauss-Seidel sweep across the fixed joint lags.
    state_0.clear_forces()
    z_before = state_0.body_q.numpy()[:, 2].copy()
    mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            solver.reset(state_0, world_mask=mask, flags=0)
            solver.step(state_0, state_1, None, None, dt)
        wp.capture_launch(capture.graph)
    else:
        solver.reset(state_0, world_mask=mask, flags=0)
        solver.step(state_0, state_1, None, None, dt)

    dz = state_1.body_q.numpy()[:, 2] - z_before
    free_fall = -gravity * dt * dt
    np.testing.assert_allclose(dz[:2], free_fall, atol=1.0e-6)
    test.assertGreater(float(dz[2]), 0.6 * free_fall)


class TestRigidAdaptiveInitialization(unittest.TestCase):
    pass


for _name, _test in (
    ("test_loaded_joint_maintains_force_balance", test_loaded_joint_maintains_force_balance),
    ("test_free_jointed_chain_falls_freely", test_free_jointed_chain_falls_freely),
    ("test_zero_iterations_keep_free_integration", test_zero_iterations_keep_free_integration),
    ("test_external_rigid_solver_is_rejected", test_external_rigid_solver_is_rejected),
    ("test_resumed_iterations_preserve_galilean_invariance", test_resumed_iterations_preserve_galilean_invariance),
    (
        "test_start_keeps_only_acceleration_along_the_applied_direction",
        test_start_keeps_only_acceleration_along_the_applied_direction,
    ),
    (
        "test_masked_reset_restores_free_fall_only_in_selected_world",
        test_masked_reset_restores_free_fall_only_in_selected_world,
    ),
):
    add_function_test(TestRigidAdaptiveInitialization, _name, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
