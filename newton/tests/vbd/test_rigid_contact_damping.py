# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent force-reporting control for the default compression-only law."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_solver_force_reporting(test, device):
    """Both reporting APIs and captured mode preserve the default material law."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(ke=1.0e4, kd=2.0, mu=0.5, density=0.0, gap=0.02)
    builder.add_ground_plane(cfg=cfg)
    body = builder.add_body(
        xform=wp.transform((0.0, 0.0, 0.09), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 0.004),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    builder.color()
    with wp.ScopedDevice(device):
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model)
        dt = 1.0 / 600.0
        solver = newton.solvers.SolverVBD(model, iterations=4)
        state, out = model.state(), model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, out, None, contacts, dt)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        # Prescribed frozen-row unloading state with known material variables.
        state.body_q.assign([wp.transform((0.001, 0.0, 0.0901), wp.quat_identity())])
        solver.body_q_prev.assign([wp.transform((0.0, 0.0, 0.09), wp.quat_identity())])
        solver.body_body_contact_normal_rho.fill_(1.0e6)
        solver.body_body_contact_material_ke.fill_(1.0e4)
        solver.body_body_contact_material_kd.fill_(2.0)
        solver.body_body_contact_material_mu.fill_(0.5)
        solver.body_body_contact_tangent_rho.fill_(1.0e6)
        solver.body_body_contact_C0.zero_()
        n = contacts.rigid_contact_normal.numpy()[0]
        lam = np.zeros((solver.body_body_contact_lambda.shape[0], 3), dtype=np.float32)
        lam[0] = n * 100.0
        solver.body_body_contact_lambda.assign(lam)
        wrench = wp.zeros(contacts.rigid_contact_max, dtype=wp.spatial_vector)

        def report():
            values = solver.collect_rigid_contact_forces(state.body_q, solver.body_q_prev, contacts, dt)
            solver._export_body_body_contact_forces(state, contacts, dt, wrench)
            return values

        values = report()
        force = values[4].numpy()[0].copy()
        C = 0.1 - float(state.body_q.numpy()[0, 2])
        previous = 0.1 - float(solver.body_q_prev.numpy()[0, 2])
        k, d, rho, lam_n = 1.0e4, 2.0 / dt, 1.0e6, 100.0
        lo, hi = -1.0, 1.0
        for _ in range(100):
            e = (lo + hi) / 2.0
            physical = k * max(e, 0.0) + d * max(e - previous, 0.0)
            if rho * (C - e) + lam_n > physical:
                lo = e
            else:
                hi = e
        expected = physical
        # Collision stores radius/anchors in float32, unlike this float64 oracle.
        test.assertAlmostEqual(float(force @ n), expected, delta=0.002)
        elastic = k * max(e, 0.0)
        bound = 0.5 * elastic
        tangent = force - n * (force @ n)
        test.assertAlmostEqual(float(np.linalg.norm(tangent)), bound, delta=0.002)
        np.testing.assert_allclose(wrench.numpy()[0, :3], -force, rtol=2.0e-6, atol=1.0e-5)
        point0 = values[2].numpy()[0]
        np.testing.assert_allclose(wrench.numpy()[0, 3:], np.cross(point0, -force), rtol=2.0e-5, atol=1.0e-5)
        expected_wrench = wrench.numpy()[0].copy()
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                report()
            for _ in range(2):
                values[4].zero_()
                wrench.zero_()
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(values[4].numpy()[0], force)
                np.testing.assert_array_equal(wrench.numpy()[0], expected_wrench)


class TestRigidContactDamping(unittest.TestCase):
    pass


add_function_test(
    TestRigidContactDamping, "test_solver_force_reporting", test_solver_force_reporting, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
