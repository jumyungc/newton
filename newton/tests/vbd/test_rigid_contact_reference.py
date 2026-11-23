# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep the compliant damping reference fixed while the current iterate changes."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.sdf_planar_recovery import MeshRecoveryData
from newton._src.sim.contacts import contact_surface_separation
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _compliant_contact_normal_response,
    build_body_body_contact_step_levers,
    compute_rigid_contact_forces,
    update_duals_body_body_contacts,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _reference(
    q: wp.array[wp.transform],
    prev: wp.array[wp.transform],
    normal: wp.array[wp.vec3],
    c0: wp.array[wp.vec3],
    lam: wp.array[wp.vec3],
    alpha: float,
    expected: wp.array[wp.vec3],
):
    i = wp.tid()
    a, b = 2 * i, 2 * i + 1
    # The bodies only translate and each contact point sits at its body origin.
    pa = wp.transform_get_translation(prev[a])
    pb = wp.transform_get_translation(prev[b])
    da = wp.transform_get_translation(q[a]) - pa
    db = wp.transform_get_translation(q[b]) - pb
    n = normal[i]
    shift = alpha * wp.dot(n, c0[i])
    # Evaluate the fixed geometry directly, not current + a cancelling displacement.
    start_raw = -contact_surface_separation(pa, pb, n, 0.1, 0.0)
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
        lam_angular = wp.zeros(count, dtype=wp.vec3)
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
        # Frictionless rows: the friction record and the packed tangent and rolling metrics stay zero.
        mu = wp.zeros(count, dtype=wp.vec3)
        rho_t = wp.zeros(count, dtype=wp.mat22)
        expected = wp.empty(count, dtype=wp.vec3)
        output = [wp.empty(count, dtype=t) for t in (int, int, wp.vec3, wp.vec3, wp.vec3, wp.vec3)]
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
        # The dual update reads the row geometry the solve stores once per step.
        levers = [wp.zeros(count, dtype=wp.vec3) for _ in range(4)] + [wp.zeros(count, dtype=float)]
        wp.launch(
            build_body_body_contact_step_levers,
            count,
            inputs=[nc, shape0, shape1, points, points, points, points, n, margin, zeros, shape_body, prev, com],
            outputs=levers,
        )
        # The reference, the exporter and the dual update are separate kernels, and the compiler may fuse
        # the multiply-adds of the 0.1 m gap differently in each. Two float32 spacings of that gap
        # (1.5e-8 m) times the row stiffness (1e5 N/m) bound the difference; a wrong damping reference
        # is off by newtons.
        atol = 1.5e-3
        recovery = MeshRecoveryData()
        generation = wp.zeros(1, dtype=int)
        for alpha in (0.0, 0.35):
            scaled_shift = wp.array(np.asarray(c0, dtype=np.float32) * np.float32(alpha), dtype=wp.vec3)
            wp.copy(lam, initial)
            wp.launch(_reference, count, inputs=[q, prev, n, shift, initial, alpha, expected])
            wp.launch(
                compute_rigid_contact_forces,
                count,
                inputs=[
                    1.0 / 1920.0,
                    *geometry,
                    rho,
                    ke,
                    kd,
                    mu,
                    rho_t,
                    lam,
                    lam_angular,
                    scaled_shift,
                    recovery,
                    generation,
                    False,
                ],
                outputs=output,
            )
            np.testing.assert_allclose(output[4].numpy(), expected.numpy(), rtol=4.0e-7, atol=atol)
            inputs = [
                nc,
                shape0,
                shape1,
                None,
                n,
                *levers,
                shape_body,
                q,
                prev,
                com,
                wp.zeros(2 * count, dtype=wp.mat33),
                0,
                1.0 / 1920.0,
                mu,
                scaled_shift,
                recovery,
                generation,
                ke,
                kd,
                rho_t,
                rho,
                zeros,
                rho_t,
                False,
                lam,
                lam_angular,
            ]
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
