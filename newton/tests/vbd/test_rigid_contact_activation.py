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
        (0.002, -0.0015, 1.0, 0.0, 1000.0 * 100.0 / 1100.0),
        (0.0001, -0.0002, 0.0, 10.0, 1000.0 * 10100.0 / 11100.0),
        (0.002, -0.0005, 1.0, 0.0, 0.0),
    )
    with wp.ScopedDevice(device):
        for gap, dz, lam, kd, expected in cases:
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
                with test.subTest(gap=gap, dz=dz, kd=kd, width=width):
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
    for guard in (False, True):
        solver = newton.solvers.SolverVBD(model, iterations=4, rigid_contact_activation_guard=guard)
        if not guard:
            test.assertIsNone(solver._rigid_contact_trial_q)
            test.assertEqual(solver._contact_activation_kernels, ())
        state, out = model.state(), model.state()
        control = model.control()

        def reset(state=state, solver=solver):
            wp.copy(state.body_q, model.body_q)
            state.body_qd.zero_()
            solver.reset(state)

        def step(state=state, out=out, solver=solver, control=control):
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, dt)

        step()
        reset()
        step()
        eager = out.body_q.numpy().copy()
        poses.append(eager)
        test.assertAlmostEqual(float(eager[body, 2] - model.body_q.numpy()[body, 2]), expected, delta=2.0e-7)
        if device.is_cuda:
            reset()
            with wp.ScopedCapture(device=device) as capture:
                step()
            for _ in range(2):
                reset()
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(out.body_q.numpy(), eager, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(poses[0], poses[1], rtol=0.0, atol=2.0e-7)


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


def test_unloading_release_is_inactive(test, device):
    """Under unloading damping, a penetrating row with zero total force is inactive and guarded."""
    rho, ke, kd, dt = 1000.0, 100.0, 10.0, 0.001
    total_k = ke + kd / dt
    expected = rho * total_k / (rho + total_k)
    with wp.ScopedDevice(device):
        previous = wp.array([wp.transform((0.0, 0.0, -0.001), wp.quat_identity())], dtype=wp.transform)
        current = wp.array([wp.transform((0.0, 0.0, -0.0002), wp.quat_identity())], dtype=wp.transform)
        trial = wp.array([wp.transform((0.0, 0.0, -0.0012), wp.quat_identity())], dtype=wp.transform)
        inputs = [
            dt,
            wp.array([0], dtype=int),
            previous,
            current,
            trial,
            wp.zeros(1, dtype=wp.vec3),
            wp.ones(1),
            wp.zeros(1, dtype=int),
            wp.full(1, rho),
            wp.full(1, ke),
            wp.full(1, kd),
            wp.zeros(1, dtype=wp.vec3),
            wp.zeros(1, dtype=wp.vec3),
            0.0,
            wp.array([1], dtype=int),
            wp.zeros(1, dtype=int),
            wp.ones(1, dtype=int),
            wp.array([[0.1, 0.0, 0.0]], dtype=wp.vec3),
            wp.array([[0.1, 0.0, 0.0]], dtype=wp.vec3),
            wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3),
            wp.zeros(1),
            wp.zeros(1),
            wp.array([-1, 0], dtype=int),
            1,
            wp.ones(1, dtype=int),
            wp.zeros(1, dtype=int),
        ]
        for unloading, metric in ((True, expected), (False, 0.0)):
            for width in (4, 32):
                outputs = [wp.zeros(1, dtype=wp.mat33) for _ in range(3)]
                kernel = create_accumulate_inactive_contact_metric(width, unloading)
                wp.launch(kernel, dim=width, inputs=inputs, outputs=[*outputs, wp.zeros(1, dtype=wp.int32)])
                with test.subTest(unloading=unloading, width=width):
                    np.testing.assert_allclose(
                        outputs[0].numpy()[0], np.diag([0.0, 0.0, metric]), rtol=2.0e-6, atol=1.0e-6
                    )


def test_unloading_warm_shift_oracle(test, device):
    """Finite-rho release, shifted history, and all metric blocks from a scalar oracle."""
    ke, dt = 100.0, 0.001
    arm = np.array([0.1, 0.0, 0.0])
    skew = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.1], [0.0, 0.1, 0.0]])
    with wp.ScopedDevice(device):
        for rho in (100.0, 1000.0, 100000.0):
            for kd in (0.0, 10.0):
                for shift in (0.0, 0.0003):
                    lam, p0 = 0.02, 0.001
                    total_k = ke + kd / dt
                    # Solve rho*(C-e)+lambda = max(ke*e + d*(e-p0), 0).
                    release = (kd / dt) * p0 / total_k - lam / rho
                    for now_delta, trial_delta in (
                        (-0.00015, 0.00015),
                        (-0.00015, -0.00005),
                        (0.00015, 0.00025),
                        (-0.00015, -0.00025),
                    ):
                        active_trial = trial_delta > 0.0
                        inactive_now = now_delta < 0.0
                        metric = rho * total_k / (rho + total_k) if inactive_now and active_trial else 0.0
                        inputs = [
                            dt,
                            wp.array([0], dtype=int),
                            wp.array([wp.transform((0.0, 0.0, -shift - p0), wp.quat_identity())], dtype=wp.transform),
                            wp.array(
                                [wp.transform((0.0, 0.0, -shift - release - now_delta), wp.quat_identity())],
                                dtype=wp.transform,
                            ),
                            wp.array(
                                [wp.transform((0.0, 0.0, -shift - release - trial_delta), wp.quat_identity())],
                                dtype=wp.transform,
                            ),
                            wp.zeros(1, dtype=wp.vec3),
                            wp.ones(1),
                            wp.zeros(1, dtype=int),
                            wp.full(1, rho),
                            wp.full(1, ke),
                            wp.full(1, kd),
                            wp.array([[0.0, 0.0, lam]], dtype=wp.vec3),
                            wp.array([[0.0, 0.0, shift / 0.6]], dtype=wp.vec3),
                            0.6,
                            wp.ones(1, dtype=int),
                            wp.zeros(1, dtype=int),
                            wp.ones(1, dtype=int),
                            wp.array([arm], dtype=wp.vec3),
                            wp.array([arm], dtype=wp.vec3),
                            wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3),
                            wp.zeros(1),
                            wp.zeros(1),
                            wp.array([-1, 0], dtype=int),
                            1,
                            wp.ones(1, dtype=int),
                            wp.zeros(1, dtype=int),
                        ]
                        K = np.diag([0.0, 0.0, metric])
                        expected = (K, skew @ K, skew.T @ K @ skew)
                        for unloading in (False, True) if kd == 0.0 else (True,):
                            for width in (4, 32):
                                outputs = [wp.zeros(1, dtype=wp.mat33) for _ in range(3)]
                                activated = wp.zeros(1, dtype=wp.int32)
                                kernel = create_accumulate_inactive_contact_metric(width, unloading)
                                wp.launch(kernel, dim=width, inputs=inputs, outputs=[*outputs, activated])
                                with test.subTest(
                                    rho=rho,
                                    kd=kd,
                                    shift=shift,
                                    now=now_delta,
                                    trial=trial_delta,
                                    unloading=unloading,
                                    width=width,
                                ):
                                    eager = [x.numpy().copy() for x in outputs]
                                    for actual, reference in zip(eager, expected, strict=True):
                                        np.testing.assert_allclose(actual[0], reference, rtol=3.0e-6, atol=2.0e-5)
                                    if device.is_cuda:
                                        for x in outputs:
                                            x.zero_()
                                        with wp.ScopedCapture(device=device) as capture:
                                            wp.launch(kernel, dim=width, inputs=inputs, outputs=[*outputs, activated])
                                        wp.capture_launch(capture.graph)
                                        for x, reference in zip(outputs, eager, strict=True):
                                            np.testing.assert_array_equal(x.numpy(), reference)


class TestRigidContactActivation(unittest.TestCase):
    def test_requires_compliant_internal_integration(self):
        """Reject the trial guard when VBD does not own the compliant rigid solve."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.color()
        model = builder.finalize(device="cpu")
        for kwargs in ({"rigid_compliant_alm": False}, {"integrate_with_external_rigid_solver": True}):
            with self.assertRaisesRegex(ValueError, "requires compliant ALM and internal"):
                newton.solvers.SolverVBD(model, rigid_contact_activation_guard=True, **kwargs)


for test_name, test_function in (
    ("test_metric_oracle", test_metric_oracle),
    ("test_pair_rotations_and_offset_com", test_pair_rotations_and_offset_com),
    ("test_departure_and_capture", test_departure_and_capture),
    ("test_unloading_release_is_inactive", test_unloading_release_is_inactive),
    ("test_unloading_warm_shift_oracle", test_unloading_warm_shift_oracle),
):
    add_function_test(TestRigidContactActivation, test_name, test_function, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
