# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep the compliant damping reference fixed while the current iterate changes."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _body_step_kinematics,
    _compliant_contact_normal_response,
    _first_order_point,
    compute_rigid_contact_forces,
    contact_surface_separation,
    update_duals_body_body_contacts,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _reference(
    q: wp.array[wp.transform],
    prev: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    c0: wp.array[wp.vec3],
    lam: wp.array[wp.vec3],
    alpha: float,
    expected: wp.array[wp.vec3],
):
    i = wp.tid()
    a, b = 2 * i, 2 * i + 1
    xa, ca0, ca, wa, _ja = _body_step_kinematics(a, q, prev, com)
    xb, cb0, cb, wb, _jb = _body_step_kinematics(b, q, prev, com)
    pa = wp.transform_point(xa, wp.vec3(0.0))
    pb = wp.transform_point(xb, wp.vec3(0.0))
    an = _first_order_point(pa, ca0, ca, wa)
    bn = _first_order_point(pb, cb0, cb, wb)
    n = normal[i]
    shift = alpha * wp.dot(n, c0[i])
    current = -contact_surface_separation(an, bn, n, 0.1, 0.0) - shift
    # Evaluate the fixed geometry directly, not current + a cancelling displacement.
    start = -contact_surface_separation(pa, pb, n, 0.1, 0.0) - shift
    force, _elastic, _metric = _compliant_contact_normal_response(
        current, start, 5.0e6, 5.0e3, 1.0e5, wp.dot(n, lam[i]), 1.0 / 1920.0
    )
    expected[i] = n * force


def test_fixed_damping_reference(test, device):
    """Primal exporter and dual update share the direct step-start reference."""
    rng = np.random.default_rng(20261107)
    normals, starts, current, c0 = [], [], [], []
    for _ in range(8):
        n = rng.normal(size=3)
        n = (n / np.linalg.norm(n)).astype(np.float32)
        a = (1.0 + rng.uniform(-0.04, 0.04, 3)).astype(np.float32)
        b = (a + n * (0.1 - rng.uniform(-1.0e-4, 1.0e-4))).astype(np.float32)
        for _ in range(32):
            an = (a + rng.uniform(-0.001, 0.001, 3)).astype(np.float32)
            bn = (an + n * (0.1 - rng.uniform(0.0002, 0.001))).astype(np.float32)
            normals.append(n)
            starts.extend([np.r_[a, [0, 0, 0, 1]], np.r_[b, [0, 0, 0, 1]]])
            current.extend([np.r_[an, [0, 0, 0, 1]], np.r_[bn, [0, 0, 0, 1]]])
            c0.append(n * 1.0e-4)
    count = len(normals)
    with wp.ScopedDevice(device):
        q = wp.array(current, dtype=wp.transform)
        prev = wp.array(starts, dtype=wp.transform)
        com = wp.zeros(2 * count, dtype=wp.vec3)
        n = wp.array(normals, dtype=wp.vec3)
        shift = wp.array(c0, dtype=wp.vec3)
        initial = wp.array(np.array(normals) * 10.0, dtype=wp.vec3)
        lam = wp.empty_like(initial)
        zeros = wp.zeros(count, dtype=float)
        points = wp.zeros(count, dtype=wp.vec3)
        shape0 = wp.array(np.arange(count, dtype=np.int32) * 2, dtype=int)
        shape1 = wp.array(np.arange(count, dtype=np.int32) * 2 + 1, dtype=int)
        shape_body = wp.array(np.arange(2 * count, dtype=np.int32), dtype=int)
        nc = wp.array([count], dtype=int)
        margin = wp.full(count, 0.1, dtype=float)
        rho = wp.full(count, 1.0e5, dtype=float)
        ke = wp.full(count, 5.0e6, dtype=float)
        kd = wp.full(count, 5.0e3, dtype=float)
        expected = wp.empty(count, dtype=wp.vec3)
        output = [wp.empty(count, dtype=t) for t in (int, int, wp.vec3, wp.vec3, wp.vec3)]
        geometry = [
            nc,
            shape0,
            shape1,
            points,
            points,
            None,
            points,
            points,
            n,
            margin,
            zeros,
            shape_body,
            q,
            prev,
            com,
        ]
        for alpha in (0.0, 0.35):
            wp.copy(lam, initial)
            wp.launch(_reference, count, inputs=[q, prev, com, n, shift, initial, alpha, expected])
            wp.launch(
                compute_rigid_contact_forces,
                count,
                inputs=[1.0 / 1920.0, *geometry, rho, rho, ke, kd, zeros, rho, lam, shift, alpha, 0, 1, 0.01, False],
                outputs=output,
            )
            np.testing.assert_allclose(output[-1].numpy(), expected.numpy(), rtol=4.0e-7, atol=3.0e-5)
            inputs = [*geometry, 1.0 / 1920.0, zeros, shift, alpha, 0, 1, ke, kd, rho, rho, 1.0, False, rho, lam]
            update = update_duals_body_body_contacts
            wp.launch(update, count, inputs=inputs)
            np.testing.assert_allclose(lam.numpy(), expected.numpy(), rtol=4.0e-7, atol=3.0e-5)
            if device.is_cuda:
                with wp.ScopedCapture() as capture:
                    wp.copy(lam, initial)
                    wp.launch(update, count, inputs=inputs)
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(lam.numpy(), expected.numpy(), rtol=4.0e-7, atol=3.0e-5)


class TestRigidContactReference(unittest.TestCase):
    pass


add_function_test(
    TestRigidContactReference, "test_fixed_damping_reference", test_fixed_damping_reference, devices=get_test_devices()
)

if __name__ == "__main__":
    unittest.main()
