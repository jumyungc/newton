# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent scalar controls for the optional unilateral unloading law."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.vbd.rigid_vbd_kernels import _compliant_contact_normal_response
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _responses(
    rows: wp.array2d[float],
    default_result: wp.array[wp.vec3],
    compression: wp.array[wp.vec3],
    unloading: wp.array[wp.vec3],
):
    i = wp.tid()
    f, e, h = _compliant_contact_normal_response(
        rows[i, 0], rows[i, 1], rows[i, 2], rows[i, 3], rows[i, 4], rows[i, 5], rows[i, 6]
    )
    default_result[i] = wp.vec3(f, e, h)
    f, e, h = _compliant_contact_normal_response(
        rows[i, 0], rows[i, 1], rows[i, 2], rows[i, 3], rows[i, 4], rows[i, 5], rows[i, 6], False
    )
    compression[i] = wp.vec3(f, e, h)
    f, e, h = _compliant_contact_normal_response(
        rows[i, 0], rows[i, 1], rows[i, 2], rows[i, 3], rows[i, 4], rows[i, 5], rows[i, 6], True
    )
    unloading[i] = wp.vec3(f, e, h)


def test_unilateral_law(test, device):
    """Solve the material equation by independent float64 bisection."""
    rng = np.random.default_rng(71)
    rows = np.column_stack(
        (
            rng.uniform(-0.002, 0.004, 1000),
            rng.uniform(-0.002, 0.004, 1000),
            10.0 ** rng.uniform(2.0, 5.0, 1000),
            10.0 ** rng.uniform(-2.0, 2.0, 1000),
            10.0 ** rng.uniform(3.0, 6.0, 1000),
            rng.uniform(0.0, 20.0, 1000),
            np.full(1000, 1.0 / 600.0),
        )
    ).astype(np.float32)
    rows = np.vstack(
        (
            rows,
            np.array(
                [
                    [0.0099, 0.01, 1.0e4, 2.0, 1.0e6, 100.0, 1.0 / 600.0],
                    [0.005, 0.01, 1.0e4, 50.0, 1.0e6, 100.0, 1.0 / 600.0],
                    [0.0, -0.01, 100.0, 1.0e4, 9.216e9, 0.0, 1.0 / 9600.0],
                    [0.001, 0.001, 100.0, 0.0, 9.216e9, 0.1, 1.0 / 9600.0],
                ],
                dtype=np.float32,
            ),
        )
    )
    expected = []
    for C, raw_previous, k, kd, rho, lam, dt in rows.astype(float):
        previous = max(raw_previous, 0.0)
        d = kd / dt
        lo, hi = -1.0, 1.0
        for _ in range(100):
            e = (lo + hi) / 2.0
            residual = rho * (C - e) + lam - max(k * max(e, 0.0) + d * (e - previous), 0.0)
            if residual > 0.0:
                lo = e
            else:
                hi = e
        e = (lo + hi) / 2.0
        total = max(k * max(e, 0.0) + d * (e - previous), 0.0)
        # Positive unilateral force implies e > 0; differentiate its scalar elimination.
        metric = rho * (k + d) / (rho + k + d) if total > 0.0 else 0.0
        expected.append((total, min(total, k * max(e, 0.0)), metric))
    outputs = [wp.empty(len(rows), dtype=wp.vec3, device=device) for _ in range(3)]
    wp.launch(
        _responses, dim=len(rows), inputs=[wp.array(rows, dtype=float, device=device)], outputs=outputs, device=device
    )
    default, compression, unloading = [o.numpy() for o in outputs]
    np.testing.assert_array_equal(default, compression)
    np.testing.assert_allclose(unloading, expected, rtol=2.0e-5, atol=5.0e-5)
    test.assertTrue(np.all(unloading[:, 0] >= 0.0))
    test.assertTrue(np.all(unloading[:, 1] >= 0.0))
    test.assertTrue(np.all(unloading[:, 1] <= unloading[:, 0]))
    test.assertTrue(np.all(unloading[:, 2] >= 0.0))
    test.assertLess(unloading[-4, 0], compression[-4, 0])
    test.assertEqual(float(unloading[-3, 0]), 0.0)
    np.testing.assert_array_equal(unloading[-1], compression[-1])


def test_solver_force_reporting(test, device):
    """Both reporting APIs and captured mode use the law selected by this solver."""
    import newton  # noqa: PLC0415

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
        solvers = [
            newton.solvers.SolverVBD(model, iterations=4, rigid_contact_unloading_damping=law)
            for law in (False, True, False)
        ]
        forces = []
        for law, solver in zip((False, True, False), solvers, strict=True):
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

            def report(solver=solver, state=state, contacts=contacts, wrench=wrench):
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
                physical = (
                    max(k * max(e, 0.0) + d * (e - previous), 0.0)
                    if law
                    else (k * max(e, 0.0) + d * max(e - previous, 0.0))
                )
                if rho * (C - e) + lam_n > physical:
                    lo = e
                else:
                    hi = e
            expected = physical
            # Collision stores radius/anchors in float32, unlike this float64 oracle.
            test.assertAlmostEqual(float(force @ n), expected, delta=0.002)
            elastic = k * max(e, 0.0)
            bound = 0.5 * (min(expected, elastic) if law else elastic)
            tangent = force - n * (force @ n)
            test.assertAlmostEqual(float(np.linalg.norm(tangent)), bound, delta=0.002)
            np.testing.assert_allclose(wrench.numpy()[0, :3], -force, rtol=2.0e-6, atol=1.0e-5)
            point0 = values[2].numpy()[0]
            np.testing.assert_allclose(wrench.numpy()[0, 3:], np.cross(point0, -force), rtol=2.0e-5, atol=1.0e-5)
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    report()
                for _ in range(2):
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(values[4].numpy()[0], force)
            forces.append(force)
        np.testing.assert_array_equal(forces[0], forces[2])
        test.assertLess(float(forces[1] @ n), float(forces[0] @ n))


class TestRigidContactDamping(unittest.TestCase):
    def test_requires_compliant_mode(self):
        """Reject the optional contact law outside compliant ALM."""
        import newton  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.color()
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "requires compliant ALM"):
            newton.solvers.SolverVBD(model, rigid_compliant_alm=False, rigid_contact_unloading_damping=True)


add_function_test(
    TestRigidContactDamping, "test_solver_force_reporting", test_solver_force_reporting, devices=get_test_devices()
)


add_function_test(TestRigidContactDamping, "test_unilateral_law", test_unilateral_law, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
