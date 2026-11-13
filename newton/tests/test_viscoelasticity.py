# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SLS material-point behavior, independent of rod geometry."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.viscoelasticity import sls_gain, sls_step_fraction, sls_update_history
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _hold_and_unload(
    parameters: wp.array[wp.vec4],  # equilibrium stiffness, ratio, tau, coordinate scale
    response: wp.array2d[wp.vec3],  # force, history, stored energy
):
    material = wp.tid()
    p = parameters[material]
    history = float(0.0)
    for step in range(40):
        deformation = 0.02 * p[3]
        dt = float(0.01)
        if step >= 20:
            deformation = 0.0
            dt = 0.025
        force = p[0] * deformation + p[0] * sls_gain(p[1], p[2], dt) * (deformation - history)
        history = sls_update_history(deformation, history, sls_step_fraction(p[2], dt))
        energy = 0.5 * p[0] * (deformation * deformation + p[1] * (deformation - history) ** 2.0)
        response[material, step] = wp.vec3(force, history, energy)


@wp.kernel
def _history_numerical_limits(response: wp.array[float]):
    history = float(0.0)
    fraction = sls_step_fraction(100.0, 1.0e-6)
    for _ in range(100000):
        history = sls_update_history(0.02, history, fraction)
    response[0] = history
    response[1] = sls_update_history(1.0e8, 1.0e-8, 0.0)
    response[2] = sls_update_history(1.0e-8, 1.0e8, 1.0)


def test_sls_material_point_hold_unload_and_coordinate_units(test, device):
    """Match analytic hold/unload, dissipation, elastic limits, and strain/extension coordinates.

    Rod solver tests cover geometry and lifecycle. This independent material-point test
    covers the constitutive law, including the signed force of a bilateral row on unloading.
    """
    parameters = wp.array(
        [(1000.0, 3.0, 0.4, 1.0), (100.0, 3.0, 0.4, 10.0), (1000.0, 0.0, 0.4, 1.0), (1000.0, 3.0, 0.0, 1.0)],
        dtype=wp.vec4,
        device=device,
    )
    response = wp.empty((4, 40), dtype=wp.vec3, device=device)
    wp.launch(_hold_and_unload, dim=4, inputs=[parameters], outputs=[response], device=device)
    values = response.numpy()
    n = np.arange(1, 21)
    a, b = 0.4 / 0.41, 0.4 / 0.425
    expected_hold = 20.0 * (1.0 + 3.0 * a**n)
    expected_unload = -60.0 * (1.0 - a**20) * b**n
    np.testing.assert_allclose(values[0, :, 0], np.concatenate((expected_hold, expected_unload)), atol=5.0e-5)
    test.assertLess(values[0, 20, 0], 0.0)  # The constitutive law itself is bilateral.
    for start in (0, 20):
        test.assertTrue(np.all(np.diff(values[0, start : start + 20, 2]) <= 0.0))

    # L0=0.1 m: k=EA/L0 with extension, or EA with strain. Physical forces agree.
    np.testing.assert_allclose(values[1, :, 0], values[0, :, 0], atol=5.0e-5)
    np.testing.assert_allclose(values[1, :, 1], 10.0 * values[0, :, 1], atol=2.0e-7)
    # The strain-coordinate energy is per reference length; multiply by L0 for physical energy.
    np.testing.assert_allclose(0.1 * values[1, :, 2], values[0, :, 2], atol=1.0e-6)
    for material in (2, 3):
        np.testing.assert_allclose(values[material, :, 0], np.repeat([20.0, 0.0], 20), atol=1.0e-6)


def test_sls_history_small_step_and_endpoint_precision(test, device):
    """Advance long-time relaxation and preserve exact zero/full-fraction history endpoints."""
    response = wp.empty(3, dtype=float, device=device)
    wp.launch(_history_numerical_limits, dim=1, outputs=[response], device=device)
    values = response.numpy()
    expected = 0.02 * -np.expm1(-100000 * np.log1p(1.0e-6 / 100.0))
    # Float32 incremental accumulation differs by about 0.08%; decay subtraction stays zero.
    np.testing.assert_allclose(values[0], expected, rtol=0.002, atol=0.0)
    np.testing.assert_array_equal(values[1:], np.array([1.0e-8, 1.0e-8], dtype=np.float32))


class TestViscoelasticity(unittest.TestCase):
    pass


add_function_test(
    TestViscoelasticity,
    "test_sls_material_point_hold_unload_and_coordinate_units",
    test_sls_material_point_hold_unload_and_coordinate_units,
    devices=get_test_devices(),
)
add_function_test(
    TestViscoelasticity,
    "test_sls_history_small_step_and_endpoint_precision",
    test_sls_history_small_step_and_endpoint_precision,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
