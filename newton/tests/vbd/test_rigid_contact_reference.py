# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep the compliant damping reference fixed while the current iterate changes."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _body_step_motion,
    _compliant_contact_normal_response,
    _contact_step_levers,
    _lever_displacement,
    _step_start_penetration,
    compute_rigid_contact_forces,
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
    xa, cda, wa, _ja = _body_step_motion(a, q, prev, com)
    xb, cdb, wb, _jb = _body_step_motion(b, q, prev, com)
    lsa, _lca = _contact_step_levers(a, xa, com, wp.vec3(0.0), wp.vec3(0.0))
    lsb, _lcb = _contact_step_levers(b, xb, com, wp.vec3(0.0), wp.vec3(0.0))
    da = _lever_displacement(cda, wa, lsa)
    db = _lever_displacement(cdb, wb, lsb)
    n = normal[i]
    shift = alpha * wp.dot(n, c0[i])
    # Evaluate the fixed geometry directly, not current + a cancelling displacement.
    start_raw = _step_start_penetration(a, xa, wp.vec3(0.0), b, xb, wp.vec3(0.0), n, 0.1, 0.0)
    current = start_raw + wp.dot(n, da - db) - shift
    start = start_raw - shift
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
        # The reference, the exporter and the dual update are separate kernels, and the compiler may fuse
        # the multiply-adds of the 0.1 m gap differently in each. Two float32 spacings of that gap
        # (1.5e-8 m) times the row stiffness (1e5 N/m) bound the difference; a wrong damping reference
        # is off by newtons.
        atol = 1.5e-3
        for alpha in (0.0, 0.35):
            wp.copy(lam, initial)
            wp.launch(_reference, count, inputs=[q, prev, com, n, shift, initial, alpha, expected])
            wp.launch(
                compute_rigid_contact_forces,
                count,
                inputs=[1.0 / 1920.0, *geometry, rho, ke, kd, zeros, rho, lam, shift, alpha, 0.01, False],
                outputs=output,
            )
            np.testing.assert_allclose(output[-1].numpy(), expected.numpy(), rtol=4.0e-7, atol=atol)
            inputs = [*geometry, 1.0 / 1920.0, zeros, shift, alpha, ke, kd, rho, rho, False, lam]
            update = update_duals_body_body_contacts
            wp.launch(update, count, inputs=inputs)
            np.testing.assert_allclose(lam.numpy(), expected.numpy(), rtol=4.0e-7, atol=atol)
            if device.is_cuda:
                with wp.ScopedCapture() as capture:
                    wp.copy(lam, initial)
                    wp.launch(update, count, inputs=inputs)
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(lam.numpy(), expected.numpy(), rtol=4.0e-7, atol=atol)


class TestRigidContactReference(unittest.TestCase):
    pass


add_function_test(
    TestRigidContactReference, "test_fixed_damping_reference", test_fixed_damping_reference, devices=get_test_devices()
)

if __name__ == "__main__":
    unittest.main()
