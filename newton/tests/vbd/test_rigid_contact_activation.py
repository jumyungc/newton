# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent controls for the opt-in rigid contact activation guard."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_contact_activation import create_accumulate_inactive_contact_metric
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_metric_oracle(test, device):
    """Warm history and loading damping choose the actual normal-response metric."""
    cases = (
        (0.002, -0.0015, 1.0, 0.0, False, 1000.0 * 100.0 / 1100.0),
        (0.0001, -0.0002, 0.0, 10.0, False, 1000.0 * 10100.0 / 11100.0),
        (0.0001, -0.0002, 0.0, 10.0, True, 1000.0 * 100.0 / 1100.0 + 10000.0),
        (0.002, -0.0005, 1.0, 0.0, False, 0.0),
        (0.002, -0.0005, 1.0, 10.0, True, 0.0),
    )
    with wp.ScopedDevice(device):
        for gap, dz, lam, kd, bidirectional, expected in cases:
            identity = wp.array([wp.transform_identity()], dtype=wp.transform)
            trial = wp.array([wp.transform((0.0, 0.0, dz), wp.quat_identity())], dtype=wp.transform)
            inputs = [
                0.001,
                wp.array([0], dtype=int),
                identity,
                identity,
                trial,
                wp.zeros(1, dtype=wp.vec3),
                wp.ones(1),
                wp.zeros(1, dtype=int),
                wp.full(1, 1000.0),
                wp.full(1, 100.0),
                wp.full(1, kd),
                wp.array([[0.0, 0.0, lam]], dtype=wp.vec3),
                wp.zeros(1, dtype=wp.vec3),
                0.0,
                bidirectional,
                wp.array([1], dtype=int),
                wp.zeros(1, dtype=int),
                wp.ones(1, dtype=int),
                wp.array([[0.1, 0.0, 0.0]], dtype=wp.vec3),
                wp.array([[0.1, 0.0, gap]], dtype=wp.vec3),
                wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3),
                wp.zeros(1),
                wp.zeros(1),
                wp.array([-1, 0], dtype=int),
                1,
                wp.ones(1, dtype=int),
                wp.zeros(1, dtype=int),
            ]
            skew = np.array([[0.0, -gap, 0.0], [gap, 0.0, -0.1], [0.0, 0.1, 0.0]])
            metric = np.diag([0.0, 0.0, expected])
            for width in (4, 32):
                outputs = [wp.zeros(1, dtype=wp.mat33) for _ in range(3)]
                activated = wp.zeros(1, dtype=wp.int32)
                kernel = create_accumulate_inactive_contact_metric(width)
                wp.launch(kernel, dim=width, inputs=inputs, outputs=[*outputs, activated])
                actual = [x.numpy()[0] for x in outputs]
                with test.subTest(gap=gap, dz=dz, kd=kd, bidirectional=bidirectional, width=width):
                    test.assertEqual(activated.numpy()[0], int(expected > 0.0))
                    np.testing.assert_allclose(actual[0], metric, rtol=2.0e-6, atol=1.0e-6)
                    np.testing.assert_allclose(actual[1], -skew.T @ metric, rtol=2.0e-6, atol=1.0e-6)
                    np.testing.assert_allclose(actual[2], skew.T @ metric @ skew, rtol=2.0e-6, atol=1.0e-6)


def test_departure_and_capture(test, device):
    """A penetrating gravity predictor must not resist an upward-driven Newton step."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e11, kd=0.0, mu=0.0, density=0.0, gap=0.01)
    body = builder.add_link(
        xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 0.0066666667),
        lock_inertia=True,
    )
    builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()),
        target_q=0.01,
        target_ke=1.0e5,
        target_kd=0.0,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    dt = 1.0 / 600.0
    expected = (1.0e5 * 0.01 - 9.81) / (1.0 / dt**2 + 1.0e5)
    poses = []
    residual_histories = []
    for guard in (False, True):
        solver = newton.solvers.SolverVBD(model, iterations=4, rigid_contact_activation_guard=guard)
        observables = solver.observables({solver.ObservableKind.SOLVER_RESIDUAL_HISTORY})
        if not guard:
            test.assertIsNone(solver._rigid_contact_trial_q)
            test.assertEqual(solver._contact_activation_kernels, ())
        state, out = model.state(), model.state()
        control = model.control()

        def reset(state=state, solver=solver):
            wp.copy(state.body_q, model.body_q)
            state.body_qd.zero_()
            solver.reset(state)

        def step(state=state, out=out, solver=solver, control=control, observables=observables):
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, dt, observables=observables)

        step()
        reset()
        step()
        eager = out.body_q.numpy().copy()
        poses.append(eager)
        history = observables.solver_residual_history.numpy().copy()
        test.assertTrue(np.all(np.isfinite(history)))
        test.assertTrue(np.all(history >= 0.0))
        test.assertGreater(float(history[0].max()), 0.0)
        residual_histories.append(history)
        test.assertAlmostEqual(float(eager[body, 2] - model.body_q.numpy()[body, 2]), expected, delta=2.0e-7)
        if device.is_cuda:
            reset()
            with wp.ScopedCapture(device=device) as capture:
                step()
            for _ in range(2):
                reset()
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(out.body_q.numpy(), eager, rtol=0.0, atol=2.0e-7)
                np.testing.assert_allclose(
                    observables.solver_residual_history.numpy(), history, rtol=1.0e-5, atol=1.0e-12
                )
        # Also without a residual request, a body whose trial crossed no inactive row keeps its trial pose.
        reset()
        pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, dt)
        np.testing.assert_allclose(out.body_q.numpy(), eager, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(poses[0], poses[1], rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(residual_histories[0], residual_histories[1], rtol=1.0e-5, atol=1.0e-12)


def test_crossing_trial_is_shortened(test, device):
    """Shorten a driven step whose trial crosses inactive contact rows, with and without a residual request.

    The residual describes the iterate the sweep starts from, so the guard must not change it.
    """
    clearance, dt = 1.0e-3, 1.0 / 600.0
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e7, kd=0.0, mu=0.0, density=0.0, gap=0.01)
    start = wp.transform((0.0, 0.0, 0.1 + clearance), wp.quat_identity())
    body = builder.add_link(xform=start, mass=1.0, inertia=wp.mat33(np.eye(3) * 0.0066666667), lock_inertia=True)
    builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    # The drive pulls the box 5 cm down, through the ground 1 mm below it.
    joint = builder.add_joint_prismatic(
        -1, body, axis=(0.0, 0.0, 1.0), parent_xform=start, target_q=-0.05, target_ke=1.0e5, target_kd=0.0
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    travel = {}
    residuals = {}
    for guard in (False, True):
        for residual in (False, True):
            solver = newton.solvers.SolverVBD(model, iterations=1, rigid_contact_activation_guard=guard)
            observables = solver.observables({solver.ObservableKind.SOLVER_RESIDUAL_HISTORY}) if residual else None
            state, out = model.state(), model.state()
            pipeline.collide(state, contacts)
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            solver.step(state, out, model.control(), contacts, dt, observables=observables)
            travel[guard, residual] = float(model.body_q.numpy()[body, 2] - out.body_q.numpy()[body, 2])
            if residual:
                residuals[guard] = observables.solver_residual_history.numpy().copy()
    # The unguarded sweep takes the whole drive step and ends deep in the ground.
    test.assertGreater(travel[False, False], 5.0 * clearance)
    # The guard adds the crossed rows' normal metric and solves again from the same iterate.
    test.assertLess(travel[True, False], 0.6 * travel[False, False])
    test.assertGreater(travel[True, False], 0.0)
    for guard in (False, True):
        test.assertAlmostEqual(travel[guard, True], travel[guard, False], delta=2.0e-7)
    # One iteration from the same start: the guard's added metric carries no force and must not
    # lower the recorded residual of that start.
    test.assertGreater(float(residuals[False].max()), 0.0)
    np.testing.assert_allclose(residuals[True], residuals[False], rtol=1.0e-5, atol=0.0)


def test_pair_rotations_and_offset_com(test, device):
    """Independent J-transpose K J oracle for a rotating pair and a kinematic partner."""

    def skew(v):
        x, y, z = v
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    theta = 0.6
    quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta)
    jt = np.cos(theta / 2.0) * np.eye(3) + skew(np.array([0.0, 0.0, np.sin(theta / 2.0)]))
    local = np.array([0.1, 0.2, 0.0])
    com = np.array([0.05, -0.01, 0.02])
    arm = local - com
    with wp.ScopedDevice(device):
        previous = wp.array(
            [wp.transform_identity(), wp.transform((0.0, 0.0, 0.001), wp.quat_identity())],
            dtype=wp.transform,
        )
        current = wp.array(
            [wp.transform((0.0, 0.0, 0.0), quat), wp.transform((0.0, 0.0, 0.001), quat)],
            dtype=wp.transform,
        )
        for same_color, dynamic_partner in ((True, True), (False, True), (True, False)):
            trial = wp.array(
                [wp.transform((0.0, 0.0, 0.002), quat), wp.transform((0.0, 0.0, 0.001), quat)],
                dtype=wp.transform,
            )
            inputs = [
                0.001,
                wp.array([0, 1], dtype=int),
                previous,
                current,
                trial,
                wp.array([com, com], dtype=wp.vec3),
                wp.array([1.0, 1.0 if dynamic_partner else 0.0], dtype=float),
                wp.array([0, 0 if same_color else 1], dtype=int),
                wp.full(1, 1000.0),
                wp.full(1, 100.0),
                wp.zeros(1),
                wp.zeros(1, dtype=wp.vec3),
                wp.zeros(1, dtype=wp.vec3),
                0.0,
                False,  # contact_damping_bidirectional
                wp.ones(1, dtype=int),
                wp.zeros(1, dtype=int),
                wp.ones(1, dtype=int),
                wp.array([local], dtype=wp.vec3),
                wp.array([local], dtype=wp.vec3),
                wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3),
                wp.zeros(1),
                wp.zeros(1),
                wp.array([0, 1], dtype=int),
                1,
                wp.ones(2, dtype=int),
                wp.zeros(2, dtype=int),
            ]
            factor = 2.0 if same_color and dynamic_partner else 1.0
            K = np.diag([0.0, 0.0, factor * 1000.0 * 100.0 / 1100.0])
            expected = (K, jt @ skew(arm) @ K, jt @ skew(arm).T @ K @ skew(arm) @ jt.T)
            for width in (4, 32):
                outputs = [wp.zeros(2, dtype=wp.mat33) for _ in range(3)]
                kernel = create_accumulate_inactive_contact_metric(width)
                wp.launch(kernel, dim=2 * width, inputs=inputs, outputs=[*outputs, wp.zeros(2, dtype=wp.int32)])
                actual = [x.numpy() for x in outputs]
                for body in range(2):
                    for x, reference in zip(actual, expected, strict=True):
                        np.testing.assert_allclose(
                            x[body],
                            reference if body == 0 or dynamic_partner else 0.0,
                            rtol=3.0e-6,
                            atol=2.0e-5,
                        )
                    H = np.block([[actual[0][body], actual[1][body].T], [actual[1][body], actual[2][body]]])
                    test.assertGreaterEqual(float(np.linalg.eigvalsh(H.astype(float)).min()), -2.0e-5)


class TestRigidContactActivation(unittest.TestCase):
    def test_requires_compliant_internal_integration(self):
        """Reject the trial guard when VBD does not own the compliant rigid solve."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.color()
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "requires internal rigid integration"):
            newton.solvers.SolverVBD(
                model, rigid_contact_activation_guard=True, integrate_with_external_rigid_solver=True
            )
        default_solver = newton.solvers.SolverVBD(model)
        guarded_solver = newton.solvers.SolverVBD(model, rigid_contact_activation_guard=True)
        self.assertFalse(default_solver.rigid_contact_activation_guard)
        self.assertTrue(guarded_solver.rigid_contact_activation_guard)
        with self.assertRaises(AttributeError):
            default_solver.rigid_contact_activation_guard = True


for test_name, test_function in (
    ("test_metric_oracle", test_metric_oracle),
    ("test_pair_rotations_and_offset_com", test_pair_rotations_and_offset_com),
    ("test_departure_and_capture", test_departure_and_capture),
    ("test_crossing_trial_is_shortened", test_crossing_trial_is_shortened),
):
    add_function_test(TestRigidContactActivation, test_name, test_function, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
