# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent normal damping controls for compliant solving and force reporting."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    compute_rigid_contact_forces,
    evaluate_rigid_contact_from_collision,
    update_duals_body_body_contacts,
)
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


@wp.kernel
def _evaluate_damping(
    q: wp.array[wp.transform],
    previous: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    lam: wp.array[wp.vec3],
    bidirectional: bool,
    force: wp.array[wp.vec3],
    normal_metric: wp.array[float],
):
    row = wp.tid()
    _fa, _ta, _hla, _hala, _haaa, fb, _tb, hll, _halb, _haab = evaluate_rigid_contact_from_collision(
        -1,
        row,
        q,
        previous,
        com,
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0, 0.0, 1.0),
        -wp.transform_get_translation(q[row])[2],
        -wp.transform_get_translation(previous[row])[2],
        1000.0,
        1000.0,
        1000.0,
        100.0,
        lam[row],
        0.5,
        0.01,
        0,
        1,
        0.1,
        wp.vec3(0.0),
        bidirectional,
    )
    force[row] = fb
    normal_metric[row] = hll[2, 2]


def test_compliant_damping_modes(test, device):
    """Signed damping stays outside elastic ALM; default damping remains coupled."""
    # Four loaded rows have C=.01 and lambda_n=k*C=10 N. Their normal
    # velocities are -.1, 0, .1 and .2 m/s; a fifth row is fully unloaded.
    # The warm tangent exceeds the elastic Coulomb cap, so damping must not
    # enlarge that cap even when it doubles the total approach force.
    depths = (0.01, 0.01, 0.01, 0.01, -0.01)
    previous_depths = (0.0, 0.01, 0.02, 0.03, 0.02)
    count = len(depths)
    with wp.ScopedDevice(device):
        q = wp.array([wp.transform((0.0, 0.0, -c), wp.quat_identity()) for c in depths], dtype=wp.transform)
        previous = wp.array(
            [wp.transform((0.0, 0.0, -c), wp.quat_identity()) for c in previous_depths], dtype=wp.transform
        )
        com = wp.zeros(count, dtype=wp.vec3)
        initial = wp.array([[8.0, 0.0, 10.0]] * 4 + [[8.0, 0.0, 0.0]], dtype=wp.vec3)
        lam = wp.empty_like(initial)
        points = wp.zeros(count, dtype=wp.vec3)
        zeros = wp.zeros(count, dtype=float)
        rho = wp.full(count, 1000.0, dtype=float)
        kd = wp.full(count, 100.0, dtype=float)
        mu = wp.full(count, 0.5, dtype=float)
        geometry = [
            wp.array([count], dtype=int),
            wp.zeros(count, dtype=int),
            wp.array(np.arange(1, count + 1), dtype=int),
            points,
            points,
            None,
            points,
            points,
            wp.array([[0.0, 0.0, 1.0]] * count, dtype=wp.vec3),
            zeros,
            zeros,
            wp.array(np.arange(-1, count), dtype=int),
            q,
            previous,
            com,
        ]
        force = wp.empty(count, dtype=wp.vec3)
        metric = wp.empty(count, dtype=float)
        exported = [wp.empty(count, dtype=t) for t in (int, int, wp.vec3, wp.vec3, wp.vec3)]
        for bidirectional in (False, True):
            if bidirectional:
                # f_total=k*C-kd*v_n; the dual and cone retain elastic support.
                expected_force = [[5.0, 0.0, z] for z in (20.0, 10.0, 0.0, -10.0)] + [[0.0, 0.0, 0.0]]
                expected_dual = [[5.0, 0.0, 10.0]] * 4 + [[0.0, 0.0, 0.0]]
                # kd/h contributes even at v_n=0 and when total force is tensile.
                expected_metric = [1500.0] * 4 + [0.0]
            else:
                # First-touch material elimination: (k+kd/h)=2000, rho=1000,
                # giving total 40/3 N and elastic 20/3 N on the loading row.
                expected_force = [[10.0 / 3.0, 0.0, 40.0 / 3.0]] + [[5.0, 0.0, 10.0]] * 3 + [[0.0, 0.0, 0.0]]
                expected_dual = expected_force
                expected_metric = [2000.0 / 3.0, 500.0, 500.0, 500.0, 0.0]

            def evaluate(bidirectional=bidirectional):
                wp.copy(lam, initial)
                wp.launch(
                    _evaluate_damping, count, inputs=[q, previous, com, lam, bidirectional], outputs=[force, metric]
                )
                wp.launch(
                    compute_rigid_contact_forces,
                    count,
                    inputs=[0.1, *geometry, rho, rho, rho, kd, mu, rho, lam, points, 0.0, 0, 1, 0.01, bidirectional],
                    outputs=exported,
                )
                wp.launch(
                    update_duals_body_body_contacts,
                    count,
                    inputs=[*geometry, 0.1, mu, points, 0.0, 0, 1, rho, kd, rho, rho, 0.0, bidirectional, rho, lam],
                )

            def check(expected_force=expected_force, expected_dual=expected_dual, expected_metric=expected_metric):
                np.testing.assert_allclose(force.numpy(), expected_force, rtol=2.0e-6, atol=2.0e-5)
                np.testing.assert_allclose(exported[-1].numpy(), expected_force, rtol=2.0e-6, atol=2.0e-5)
                np.testing.assert_allclose(lam.numpy(), expected_dual, rtol=2.0e-6, atol=2.0e-5)
                np.testing.assert_allclose(metric.numpy(), expected_metric, rtol=2.0e-6, atol=2.0e-5)

            with test.subTest(bidirectional=bidirectional):
                evaluate()
                check()
                if device.is_cuda:
                    with wp.ScopedCapture() as capture:
                        evaluate()
                    wp.capture_launch(capture.graph)
                    check()


class TestRigidContactDamping(unittest.TestCase):
    pass


add_function_test(
    TestRigidContactDamping, "test_solver_force_reporting", test_solver_force_reporting, devices=get_test_devices()
)


add_function_test(
    TestRigidContactDamping, "test_compliant_damping_modes", test_compliant_damping_modes, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
