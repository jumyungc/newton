# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent elastic-force and vertex-tangent controls for local VBD tets."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_volumetric_neo_hookean_force_and_hessian
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _evaluate_tets(
    q: wp.array[wp.vec3],
    ids: wp.array2d[int],
    rest_inverse: wp.array[wp.mat33],
    materials: wp.array2d[float],
    force: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
):
    tet, corner = wp.tid()
    f, h = evaluate_volumetric_neo_hookean_force_and_hessian(
        tet, corner, q, q, ids, rest_inverse[tet], materials[tet, 0], materials[tet, 1], 0.0, 1.0
    )
    force[4 * tet + corner] = f
    hessian[4 * tet + corner] = h


def _oracle(q, rest_inverse, mu, lam):
    """Float64 stress at the represented positions, without an alpha offset."""
    f = (q[1:] - q[0]).T @ rest_inverse
    cof = np.column_stack((np.cross(f[:, 1], f[:, 2]), np.cross(f[:, 2], f[:, 0]), np.cross(f[:, 0], f[:, 1])))
    stress = mu * (f - cof) + (lam + mu) * (np.linalg.det(f) - 1.0) * cof
    volume = 1.0 / (6.0 * np.linalg.det(rest_inverse))
    forces = -(volume * stress @ rest_inverse.T).T
    return np.vstack((-forces.sum(axis=0), forces))


def _evaluate(device, positions, inverses, materials):
    count = len(positions)
    with wp.ScopedDevice(device):
        forces = wp.zeros(4 * count, dtype=wp.vec3)
        hessians = wp.zeros(4 * count, dtype=wp.mat33)
        wp.launch(
            _evaluate_tets,
            dim=(count, 4),
            inputs=[
                wp.array(np.asarray(positions, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3),
                wp.array(np.arange(4 * count, dtype=np.int32).reshape(-1, 4), dtype=int),
                wp.array(np.asarray(inverses, dtype=np.float32), dtype=wp.mat33),
                wp.array(np.asarray(materials, dtype=np.float32), dtype=float),
            ],
            outputs=[forces, hessians],
        )
        return forces.numpy().reshape(count, 4, 3), hessians.numpy().reshape(count, 4, 3, 3)


def test_undeformed_tet_has_zero_stress(test, device):
    positions, inverses, materials = [], [], []
    for mu in (0.0, 1e-8, 10000.0, 20833.333984375):
        for ratio in (0.0, 1.0, 10.0, 100.0, 1e4, 1e8):
            for shift in (0.0, 1.0, 100.0):
                positions.append(np.vstack((np.zeros((1, 3)), np.eye(3))) / 16.0 + shift)
                inverses.append(16.0 * np.eye(3))
                materials.append((mu, mu * ratio))
    forces, hessians = _evaluate(device, positions, inverses, materials)
    test.assertTrue(np.isfinite(hessians).all())
    np.testing.assert_array_equal(forces, np.zeros_like(forces))


def test_tet_affine_force_and_vertex_hessian(test, device):
    deformations = (
        np.eye(3),
        np.diag([0.875, 1.125, 1.0]),
        np.array([[1.0, 0.125, 0.0], [0.0, 1.0, 0.25], [0.0, 0.0, 1.0]]),
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.diag([-0.5, 1.0, 1.0]),
        np.diag([0.0, 1.0, 1.0]),
    )
    positions, inverses = [], []
    for rest_inverse in (16.0 * np.eye(3), np.array([[8.0, 4.0, 0.0], [0.0, 16.0, 0.0], [0.0, 0.0, 32.0]])):
        rest = np.vstack((np.zeros((1, 3)), np.linalg.inv(rest_inverse).T))
        for deformation in deformations:
            for shift in (0.0, 64.0):
                positions.append((rest @ deformation.T + shift).astype(np.float32))
                inverses.append(rest_inverse)
    for mu in (0.0, 1e-8, 10000.0):
        lam = 2.0 * mu
        forces, hessians = _evaluate(device, positions, inverses, [(mu, lam)] * len(positions))
        for case, (positions32, rest_inverse) in enumerate(zip(positions, inverses, strict=True)):
            q = positions32.astype(np.float64)
            expected = _oracle(q, rest_inverse, mu, lam)
            with test.subTest(mu=mu, case=case):
                np.testing.assert_allclose(forces[case], expected, rtol=3e-6, atol=mu * 1e-10)
                # Differentiate the independent force at fixed other vertices. The
                # determinant is affine in one vertex, so this diagonal block is exact.
                for corner in range(4):
                    derivative = np.zeros((3, 3))
                    epsilon = 1e-6
                    for axis in range(3):
                        plus, minus = q.copy(), q.copy()
                        plus[corner, axis] += epsilon
                        minus[corner, axis] -= epsilon
                        derivative[:, axis] = -(
                            _oracle(plus, rest_inverse, mu, lam)[corner] - _oracle(minus, rest_inverse, mu, lam)[corner]
                        ) / (2.0 * epsilon)
                    np.testing.assert_allclose(hessians[case, corner], derivative, rtol=3e-6, atol=mu * 1e-8)


class TestVBDTetStress(unittest.TestCase):
    pass


for _test in (test_undeformed_tet_has_zero_stress, test_tet_affine_force_and_vertex_hessian):
    add_function_test(TestVBDTetStress, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
