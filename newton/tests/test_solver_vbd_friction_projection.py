# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.vbd.friction import _project_friction_disk, _project_friction_interval
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _project_disks(
    displacement: wp.array[wp.vec2],
    old: wp.array[wp.vec2],
    mobility: wp.array[wp.mat22],
    bound: wp.array[float],
    force: wp.array[wp.vec2],
    metric: wp.array[wp.mat22],
):
    i = wp.tid()
    f, q = _project_friction_disk(displacement[i], old[i], mobility[i], bound[i])
    force[i] = f
    metric[i] = q


@wp.kernel
def _project_intervals(
    displacement: wp.array[float],
    old: wp.array[float],
    mobility: wp.array[float],
    bound: wp.array[float],
    force: wp.array[float],
    metric: wp.array[float],
):
    i = wp.tid()
    f, q = _project_friction_interval(displacement[i], old[i], mobility[i], bound[i])
    force[i] = f
    metric[i] = q


def _sample(displacement, old, mobility, bound, device):
    arrays = [
        wp.array(displacement, dtype=wp.vec2, device=device),
        wp.array(old, dtype=wp.vec2, device=device),
        wp.array(mobility, dtype=wp.mat22, device=device),
        wp.array(bound, dtype=float, device=device),
    ]
    force = wp.empty(len(bound), dtype=wp.vec2, device=device)
    metric = wp.empty(len(bound), dtype=wp.mat22, device=device)
    wp.launch(_project_disks, dim=len(bound), inputs=arrays, outputs=[force, metric], device=device)
    return force.numpy().astype(float), metric.numpy().astype(float)


def _reference(displacement, old, mobility, bound):
    """Solve the constrained quadratic with dense solves and independent bisection."""
    rhs = displacement + mobility @ old
    force = np.linalg.solve(mobility, rhs)
    if np.linalg.norm(force) <= bound:
        return force
    lo, hi = 0.0, np.linalg.norm(rhs) / bound
    for _ in range(512):
        eta = 0.5 * (lo + hi)
        force = np.linalg.solve(mobility + eta * np.eye(2), rhs)
        if np.linalg.norm(force) > bound:
            lo = eta
        else:
            hi = eta
    return np.linalg.solve(mobility + 0.5 * (lo + hi) * np.eye(2), rhs)


def test_vbd_friction_disk_dense_reference(test, device):
    """Match the constrained quadratic over anisotropy, scales, and reversals."""
    rows = []
    for ratio, angle in ((1.0, 0.0), (1e2, 0.31), (1e4, 0.9), (1e6, 0.31), (1e12, 0.0)):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for scale in (1e-6, 1.0, 1e6):
            w = scale * rotation @ np.diag([1.0 / ratio, 1.0]) @ rotation.T
            for bound in (1e-6, 1.0, 1e6):
                for amount in (0.2, 0.99, 1.001, 2.0, -2.0):
                    old = bound * np.array([0.7, -0.3])
                    target = amount * bound * np.array([0.6, 0.8])
                    rows.append((w @ (target - old), old, w, bound))
                rows.append((bound * scale * rotation @ np.array([1.0 / ratio, 1.0]), np.zeros(2), w, bound))
    displacement, old, mobility, bounds = [np.array(value, dtype=np.float32) for value in zip(*rows, strict=True)]
    force, metric = _sample(displacement, old, mobility, bounds, device)
    test.assertTrue(np.isfinite(force).all())
    test.assertTrue(np.isfinite(metric).all())
    for i in range(len(bounds)):
        with test.subTest(case=i):
            expected = _reference(
                displacement[i].astype(float), old[i].astype(float), mobility[i].astype(float), float(bounds[i])
            )
            np.testing.assert_allclose(force[i] / bounds[i], expected / bounds[i], atol=1e-6, rtol=1e-6)
            test.assertLessEqual(np.linalg.norm(force[i]) / bounds[i], 1.0 + 1e-6)


def test_vbd_friction_projection_covariance(test, device):
    """Preserve disk forces under basis rotation and coordinate-unit scaling."""
    w = np.array([[0.2, 0.07], [0.07, 1.1]])
    displacement, old = np.array([0.8, -0.9]), np.array([0.2, 0.3])
    angle = 0.73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rows = [(displacement, old, w, 1.0)]
    rows.append((rotation @ displacement, rotation @ old, rotation @ w @ rotation.T, 1.0))
    # q' = s*q implies lambda' = lambda/s and W' = s^2*W.
    for scale in (1e-6, 1e6):
        rows.append((scale * displacement, old / scale, scale * scale * w, 1.0 / scale))
    values = [np.array(value, dtype=np.float32) for value in zip(*rows, strict=True)]
    force, metric = _sample(*values, device)
    np.testing.assert_allclose(force[1], rotation @ force[0], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(metric[1], rotation @ metric[0] @ rotation.T, atol=2e-6, rtol=2e-6)
    for i, scale in enumerate((1e-6, 1e6), 2):
        np.testing.assert_allclose(force[i] * scale, force[0], atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(metric[i] * scale * scale, metric[0], atol=2e-6, rtol=2e-6)


def test_vbd_friction_projection_scalar_consistency(test, device):
    """Agree with the one-dimensional Coulomb interval along an isotropic disk axis."""
    displacement = np.array([-5.0, -0.2, 0.0, 0.2, 5.0], dtype=np.float32)
    old, mobility, bound = np.full(5, 0.3), np.full(5, 0.7), np.ones(5)
    force, metric = _sample(
        np.column_stack((displacement, np.zeros(5))),
        np.column_stack((old, np.zeros(5))),
        np.array([m * np.eye(2) for m in mobility]),
        bound,
        device,
    )
    inputs = [wp.array(value, dtype=float, device=device) for value in (displacement, old, mobility, bound)]
    scalar_force, scalar_metric = [wp.empty(5, dtype=float, device=device) for _ in range(2)]
    wp.launch(_project_intervals, dim=5, inputs=inputs, outputs=[scalar_force, scalar_metric], device=device)
    np.testing.assert_allclose(force[:, 0], scalar_force.numpy(), atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(metric[:, 0, 0], scalar_metric.numpy(), atol=2e-7, rtol=2e-7)
    np.testing.assert_array_equal(force[:, 1], 0.0)


def test_vbd_friction_projection_metric_bound(test, device):
    """Bound the finite-difference force Jacobian by the returned primal metric."""
    w = np.array([[0.2, 0.07], [0.07, 1.1]])
    old = np.array([0.2, 0.3])
    epsilon = 1e-3
    for displacement in (np.array([0.01, -0.03]), np.array([0.8, -0.9])):
        rows = [(displacement, old, w, 1.0)]
        for direction in np.eye(2):
            rows.extend((displacement + sign * epsilon * direction, old, w, 1.0) for sign in (-1, 1))
        values = [np.array(value, dtype=np.float32) for value in zip(*rows, strict=True)]
        force, metric = _sample(*values, device)
        derivative = np.column_stack(((force[2] - force[1]) / (2 * epsilon), (force[4] - force[3]) / (2 * epsilon)))
        np.testing.assert_allclose(derivative, derivative.T, atol=1e-4)
        test.assertGreaterEqual(np.linalg.eigvalsh(derivative)[0], -1e-4)
        test.assertGreaterEqual(np.linalg.eigvalsh(metric[0] - derivative)[0], -1e-4)
        test.assertGreater(np.linalg.eigvalsh(metric[0])[0], 0.0)


def test_vbd_friction_projection_disabled(test, device):
    """Clear forces and primal metrics for a disabled or zero-response channel."""
    force, metric = _sample(
        np.ones((3, 2)), np.ones((3, 2)), [np.zeros((2, 2)), np.eye(2), np.eye(2)], [1.0, 0.0, -1.0], device
    )
    np.testing.assert_array_equal(force, 0.0)
    np.testing.assert_array_equal(metric, 0.0)


def test_vbd_friction_interval_large_finite_inputs(test, device):
    """Preserve finite interval forces when the unscaled quadratic right-hand side overflows."""
    displacement = np.array([0.0, 1e20, -1e20, 1e20], dtype=np.float32)
    old = np.array([1e20, 1e20, 1e20, 0.0], dtype=np.float32)
    mobility = np.array([1e20, 1e20, 1e20, 1e-20], dtype=np.float32)
    bound = np.array([1e21, 1e19, 1e21, 1e20], dtype=np.float32)
    inputs = [wp.array(value, dtype=float, device=device) for value in (displacement, old, mobility, bound)]
    force, metric = [wp.empty(4, dtype=float, device=device) for _ in range(2)]
    wp.launch(_project_intervals, dim=4, inputs=inputs, outputs=[force, metric], device=device)
    trial = old.astype(float) + displacement.astype(float) / mobility.astype(float)
    expected_force = np.clip(trial, -bound.astype(float), bound.astype(float))
    expected_metric = 1.0 / (mobility.astype(float) * np.maximum(1.0, np.abs(trial) / bound.astype(float)))
    np.testing.assert_allclose(force.numpy(), expected_force, rtol=1e-6)
    np.testing.assert_allclose(metric.numpy(), expected_metric, rtol=1e-6)


def test_vbd_friction_disk_tiny_force_scale(test, device):
    """Preserve the disk bound when the squared force magnitude underflows float32."""
    bound = np.array([1e-30, 1e-30], dtype=np.float32)
    displacement = np.array([[2e-30, 0.0], [0.6e-30, 0.8e-30]], dtype=np.float32)
    force, metric = _sample(displacement, np.zeros((2, 2)), [np.eye(2), np.eye(2)], bound, device)
    np.testing.assert_allclose(force / bound[:, None], [[1.0, 0.0], [0.6, 0.8]], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(metric, [0.5 * np.eye(2), np.eye(2)], rtol=1e-6, atol=1e-6)


def test_vbd_friction_disk_extreme_mobility(test, device):
    """Keep the secular equation finite across the float32 mobility exponent range."""
    mobility = np.array(
        [np.diag([1e-24, 1.0]), np.diag([1e-36, 1.0]), np.diag([1e-36, 1e36]), np.diag([1e36, 1e-36])],
        dtype=np.float32,
    )
    displacement = np.array([[1e-24, 0.99], [1e-30, 0.99], [1e-36, 0.99e36], [0.99e36, 1e-36]], dtype=np.float32)
    force, metric = _sample(displacement, np.zeros((4, 2)), mobility, np.ones(4), device)
    test.assertTrue(np.isfinite(force).all())
    test.assertTrue(np.isfinite(metric).all())
    for i in range(4):
        expected = _reference(displacement[i].astype(float), np.zeros(2), mobility[i].astype(float), 1.0)
        np.testing.assert_allclose(force[i], expected, rtol=1e-6, atol=1e-6)
        test.assertLessEqual(np.linalg.norm(force[i]), 1.0 + 1e-6)


def test_vbd_friction_disk_single_active_axis(test, device):
    """Match the disk force and both metric axes when one coordinate is identically inactive."""
    rows = []
    for axis in (0, 1):
        for scale in (1e-12, 1.0, 1e12):
            w = scale * np.diag([0.01, 1.0])
            for amount in (2.0, -2.0):
                displacement, old = np.zeros(2), np.zeros(2)
                old[axis] = 0.3
                displacement[axis] = w[axis, axis] * amount
                rows.append((displacement, old, w, 1.0))
    # This transverse trial rounds to zero in float32, but its inputs are
    # nonzero and its double-precision residual must still affect projection.
    rows.append((np.array([0.02, -3e7]), np.array([0.0, 1e8]), np.diag([0.01, 0.3]), 1.0))
    values = [np.array(value, dtype=np.float32) for value in zip(*rows, strict=True)]
    force, metric = _sample(*values, device)
    displacement, old, mobility, bounds = values
    for i in range(len(bounds)):
        w = mobility[i].astype(float)
        r, previous = displacement[i].astype(float), old[i].astype(float)
        expected = _reference(r, previous, w, float(bounds[i]))
        eta = np.linalg.norm(r + w @ previous - w @ expected) / bounds[i]
        expected_metric = np.linalg.inv(w + eta * np.eye(2))
        np.testing.assert_allclose(force[i], expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(metric[i], expected_metric, rtol=1e-6)


def test_vbd_friction_disk_reduced_bound_cancellation(test, device):
    """Preserve force direction when a reduced normal load makes old friction exceed the new bound."""
    mobility = np.tile(0.7 * np.eye(2), (2, 1, 1)).astype(np.float32)
    old = np.array([[0.3, 0.0], [0.3, 0.2]], dtype=np.float32)
    displacement = -np.float32(0.7) * old
    bound = np.full(2, 1e-10, dtype=np.float32)
    force, metric = _sample(displacement, old, mobility, bound, device)
    for i in range(2):
        expected = _reference(
            displacement[i].astype(float), old[i].astype(float), mobility[i].astype(float), float(bound[i])
        )
        np.testing.assert_allclose(force[i] / bound[i], expected / bound[i], rtol=1e-6, atol=1e-6)
        test.assertTrue(np.isfinite(metric[i]).all())
        test.assertGreater(np.linalg.eigvalsh(metric[i])[0], 0.0)


class TestSolverVBDFrictionProjection(unittest.TestCase):
    pass


devices = get_test_devices()
for name, function in (
    ("test_vbd_friction_disk_dense_reference", test_vbd_friction_disk_dense_reference),
    ("test_vbd_friction_projection_covariance", test_vbd_friction_projection_covariance),
    ("test_vbd_friction_projection_scalar_consistency", test_vbd_friction_projection_scalar_consistency),
    ("test_vbd_friction_projection_metric_bound", test_vbd_friction_projection_metric_bound),
    ("test_vbd_friction_projection_disabled", test_vbd_friction_projection_disabled),
    ("test_vbd_friction_interval_large_finite_inputs", test_vbd_friction_interval_large_finite_inputs),
    ("test_vbd_friction_disk_tiny_force_scale", test_vbd_friction_disk_tiny_force_scale),
    ("test_vbd_friction_disk_extreme_mobility", test_vbd_friction_disk_extreme_mobility),
    ("test_vbd_friction_disk_single_active_axis", test_vbd_friction_disk_single_active_axis),
    ("test_vbd_friction_disk_reduced_bound_cancellation", test_vbd_friction_disk_reduced_bound_cancellation),
):
    add_function_test(TestSolverVBDFrictionProjection, name, function, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
