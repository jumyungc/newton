# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent normal damping controls for compliant solving and force reporting."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_planar_recovery import MeshRecoveryData
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _NUM_CONTACT_THREADS_PER_BODY,
    _compliant_contact_normal_response,
    accumulate_body_body_contacts_per_body,
    build_body_body_contact_step_levers,
    compute_rigid_contact_forces,
    evaluate_rigid_contact_normal_sliding,
    update_duals_body_body_contacts,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_solver_force_reporting(test, device):
    """Both reporting APIs and captured mode preserve the default material law."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    # Sliding friction only: the angular channels would add their own couple to the reported wrench.
    cfg = builder.ShapeConfig(ke=1.0e4, kd=2.0, mu=0.5, mu_torsional=0.0, mu_rolling=0.0, density=0.0, gap=0.02)
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
        solver.body_body_contact_material_mu.fill_(wp.vec3(0.5, 0.0, 0.0))
        solver.body_body_contact_tangent_rho.fill_(wp.mat22(1.0e6, 1.0e6, 1.0, 0.0))
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


def test_reported_torque_about_current_com(test, device):
    """The reported torque on a body that turned during the step uses its current levers and pure couple."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(ke=1.0e4, kd=0.0, mu=0.5, mu_torsional=0.02, mu_rolling=0.01, density=0.0)
    for x in (0.0, 0.19):
        body = builder.add_body(
            xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
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
        solver = newton.solvers.SolverVBD(model, iterations=1)
        state, out = model.state(), model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, out, None, contacts, dt)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        b0 = int(model.shape_body.numpy()[int(contacts.rigid_contact_shape0.numpy()[0])])
        test.assertGreaterEqual(b0, 0)
        # Body 0 turned by 0.2 rad about z during the step; the row state is prescribed.
        poses = model.body_q.numpy()
        state.body_q.assign(poses)
        previous = poses.copy()
        previous[b0, 3:] = (0.0, 0.0, np.sin(-0.1), np.cos(-0.1))
        solver.body_q_prev.assign(previous)
        solver.body_body_contact_normal_rho.fill_(1.0e6)
        solver.body_body_contact_material_ke.fill_(1.0e4)
        solver.body_body_contact_material_kd.fill_(0.0)
        solver.body_body_contact_material_mu.fill_(wp.vec3(0.5, 0.02, 0.01))
        solver.body_body_contact_tangent_rho.fill_(wp.mat22(1.0e6, 1.0e6, 1.0, 0.0))
        solver.body_body_contact_C0.zero_()
        n = contacts.rigid_contact_normal.numpy()[0].astype(np.float64)
        lam = np.zeros((solver.body_body_contact_lambda.shape[0], 3), dtype=np.float32)
        lam[0] = n * 100.0
        solver.body_body_contact_lambda.assign(lam)
        pure_couple_1 = np.array([0.15, -0.25, 0.35], dtype=np.float32)
        lam_ang = np.zeros((solver.body_body_contact_lambda_angular.shape[0], 3), dtype=np.float32)
        lam_ang[0] = pure_couple_1
        solver.body_body_contact_lambda_angular.assign(lam_ang)
        wrench = wp.zeros(contacts.rigid_contact_max, dtype=wp.spatial_vector)
        solver._export_body_body_contact_forces(state, contacts, dt, wrench)
        force = wrench.numpy()[0, :3].astype(np.float64)
        normal_force = n * float(force @ n)
        friction = force - normal_force
        # The turn slides the anchor, so friction is present and off the lever.
        test.assertGreater(float(np.linalg.norm(friction)), 1.0)
        com = model.body_com.numpy()[b0].astype(np.float64)
        point = contacts.rigid_contact_point0.numpy()[0].astype(np.float64)
        anchor = point + contacts.rigid_contact_offset0.numpy()[0].astype(np.float64)
        # Body 0's current orientation is the identity.
        expected = np.cross(point - com, normal_force) + np.cross(anchor - com, friction) - pure_couple_1
        test.assertGreater(float(np.linalg.norm(expected)), 0.05)
        np.testing.assert_allclose(wrench.numpy()[0, 3:], expected, rtol=1.0e-5, atol=1.0e-6)


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
    _fa, _ta, _hla, _hala, _haaa, fb, _tb, hll, _halb, _haab = evaluate_rigid_contact_normal_sliding(
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
        wp.mat22(1000.0, 1000.0, 1.0, 0.0),
        100.0,
        lam[row],
        0.5,
        0.1,
        bidirectional,
    )
    force[row] = fb
    normal_metric[row] = hll[2, 2]


def test_compliant_damping_modes(test, device):
    """Bound every friction channel by elastic support in coupled and signed damping modes."""
    # Four loaded rows have C=.01 and lambda_n=k*C=10 N. Their normal
    # velocities are -.1, 0, .1 and .2 m/s; a fifth row is fully unloaded.
    # Warm sliding and angular multipliers exceed their elastic Coulomb caps,
    # so damping must not enlarge any cap even when it doubles the approach force.
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
        # Friction record per contact: (sliding, torsional, rolling); isotropic packed tangent metric.
        mu = wp.array([[0.5, 0.02, 0.03]] * count, dtype=wp.vec3)
        rho_t = wp.array([wp.mat22(1000.0, 1000.0, 1.0, 0.0)] * count, dtype=wp.mat22)
        rolling_rho = rho_t
        initial_angular = wp.array([[8.0, 0.0, 8.0]] * count, dtype=wp.vec3)
        lam_angular = wp.empty_like(initial_angular)
        angular_compliance = wp.zeros(count, dtype=wp.mat33)
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
        # Step-start row geometry for the gather and the dual update, from the production kernel.
        levers = [wp.zeros(count, dtype=wp.vec3) for _ in range(4)] + [wp.zeros(count, dtype=float)]
        wp.launch(
            build_body_body_contact_step_levers,
            count,
            inputs=[*geometry[:5], *geometry[6:12], previous, com],
            outputs=levers,
        )
        force = wp.empty(count, dtype=wp.vec3)
        metric = wp.empty(count, dtype=float)
        exported = [wp.empty(count, dtype=t) for t in (int, int, wp.vec3, wp.vec3, wp.vec3, wp.vec3)]
        post_dual_export = [wp.empty(count, dtype=wp.vec3) for _ in range(2)]
        color_group = wp.array(np.arange(count), dtype=wp.int32)
        body_counts = wp.ones(count, dtype=wp.int32)
        body_indices = wp.array(np.arange(count), dtype=wp.int32)
        body_records = wp.array([(-1, i, 0) for i in range(count)], dtype=wp.vec3i)
        gathered = [wp.zeros(count, dtype=wp.vec3) for _ in range(2)]
        gathered += [wp.zeros(count, dtype=wp.mat33) for _ in range(3)]
        for bidirectional in (False, True):
            if bidirectional:
                # f_total=k*C-kd*v_n; the dual and cone retain elastic support.
                expected_force = [[5.0, 0.0, z] for z in (20.0, 10.0, 0.0, -10.0)] + [[0.0, 0.0, 0.0]]
                expected_dual = [[5.0, 0.0, 10.0]] * 4 + [[0.0, 0.0, 0.0]]
                # kd/h contributes even at v_n=0 and when total force is tensile.
                expected_metric = [1500.0] * 4 + [0.0]
                elastic_load = np.array([10.0] * 4 + [0.0])
            else:
                # First-touch material elimination: (k+kd/h)=2000, rho=1000,
                # giving total 40/3 N and elastic 20/3 N on the loading row.
                expected_force = [[10.0 / 3.0, 0.0, 40.0 / 3.0]] + [[5.0, 0.0, 10.0]] * 3 + [[0.0, 0.0, 0.0]]
                expected_dual = expected_force
                expected_metric = [2000.0 / 3.0, 500.0, 500.0, 500.0, 0.0]
                elastic_load = np.array([20.0 / 3.0, 10.0, 10.0, 10.0, 0.0])
            expected_angular = elastic_load[:, None] * np.array([0.03, 0.0, 0.02])

            def export(outputs, bidirectional=bidirectional):
                wp.launch(
                    compute_rigid_contact_forces,
                    count,
                    inputs=[
                        0.1,
                        *geometry,
                        rho,
                        rho,
                        kd,
                        mu,
                        rho_t,
                        lam,
                        lam_angular,
                        points,
                        MeshRecoveryData(),
                        wp.zeros(1, dtype=int),
                        bidirectional,
                    ],
                    outputs=outputs,
                )

            def evaluate(bidirectional=bidirectional):
                wp.copy(lam, initial)
                wp.copy(lam_angular, initial_angular)
                wp.launch(
                    _evaluate_damping, count, inputs=[q, previous, com, lam, bidirectional], outputs=[force, metric]
                )
                for output in gathered:
                    output.zero_()
                wp.launch(
                    accumulate_body_body_contacts_per_body,
                    count * _NUM_CONTACT_THREADS_PER_BODY,
                    inputs=[
                        0.1,
                        color_group,
                        previous,
                        q,
                        com,
                        bidirectional,
                        rho,
                        rho,
                        kd,
                        mu,
                        rho_t,
                        rho,
                        rolling_rho,
                        lam,
                        lam_angular,
                        points,
                        MeshRecoveryData(),
                        wp.zeros(1, dtype=int),
                        geometry[0],
                        geometry[1],
                        geometry[2],
                        geometry[5],
                        geometry[8],
                        *levers,
                        1,
                        body_counts,
                        body_indices,
                        body_records,
                    ],
                    outputs=gathered,
                )
                export(exported)
                wp.launch(
                    update_duals_body_body_contacts,
                    count,
                    inputs=[
                        *geometry[:3],
                        geometry[5],
                        geometry[8],
                        *levers,
                        *geometry[11:],
                        angular_compliance,
                        0,
                        0.1,
                        mu,
                        points,
                        MeshRecoveryData(),
                        wp.zeros(1, dtype=int),
                        rho,
                        kd,
                        rho_t,
                        rho,
                        rho,
                        rolling_rho,
                        bidirectional,
                        lam,
                        lam_angular,
                    ],
                )
                # Reporting exports the solved angular multiplier, without another projection.
                export(exported[:4] + post_dual_export)

            def check(
                expected_force=expected_force,
                expected_dual=expected_dual,
                expected_metric=expected_metric,
                expected_angular=expected_angular,
            ):
                np.testing.assert_allclose(force.numpy(), expected_force, rtol=2.0e-6, atol=2.0e-5)
                np.testing.assert_allclose(exported[4].numpy(), expected_force, rtol=2.0e-6, atol=2.0e-5)
                np.testing.assert_allclose(lam.numpy(), expected_dual, rtol=2.0e-6, atol=2.0e-5)
                np.testing.assert_allclose(metric.numpy(), expected_metric, rtol=2.0e-6, atol=2.0e-5)
                np.testing.assert_allclose(gathered[0].numpy(), expected_force, rtol=2.0e-6, atol=2.0e-5)
                # Each geometric and surface lever is zero, so the gathered torque is the pure couple.
                np.testing.assert_allclose(gathered[1].numpy(), expected_angular, rtol=2.0e-6, atol=2.0e-6)
                np.testing.assert_allclose(lam_angular.numpy(), expected_angular, rtol=2.0e-6, atol=2.0e-6)
                np.testing.assert_allclose(post_dual_export[1].numpy(), expected_angular, rtol=2.0e-6, atol=2.0e-6)
                for output in gathered:
                    np.testing.assert_array_equal(output.numpy()[-1], 0.0)

            with test.subTest(bidirectional=bidirectional):
                evaluate()
                check()
                if device.is_cuda:
                    with wp.ScopedCapture() as capture:
                        evaluate()
                    wp.capture_launch(capture.graph)
                    check()


def test_compliant_contact_step_force_balance(test, device):
    """Touchdown and existing compression obey the same independent material force balance."""
    radius, speed, dt, mass = 0.1, 1.0, 1.0 / 60.0, 1.0
    stiffness, damping = 2500.0, 100.0
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(density=0.0, mu=0.0, ke=stiffness, kd=damping)
    builder.add_ground_plane(cfg=cfg)
    body = builder.add_body(
        xform=wp.transform((0.0, 0.0, radius), wp.quat_identity()),
        mass=mass,
        inertia=wp.mat33(np.eye(3) * 0.004),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body, radius=radius, cfg=cfg)
    builder.color()
    with wp.ScopedDevice(device):
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        # First touch covers 72% of the step before contact. Existing compression
        # requires nonzero history in both the primal response and dual update.
        for previous, iterations, tolerance in ((-0.012, 10, 1.0e-4), (-0.012, 11, 1.0e-4), (0.001, 20, 3.0e-6)):
            with test.subTest(previous_penetration=previous, iterations=iterations):
                solver = newton.solvers.SolverVBD(model, iterations=iterations)
                contacts = pipeline.contacts()
                state, out = model.state(), model.state()
                state.body_q.assign([wp.transform((0.0, 0.0, radius - previous), wp.quat_identity())])
                velocity = state.body_qd.numpy()
                velocity[0, 2] = -speed
                state.body_qd.assign(velocity)
                pipeline.collide(state, contacts)
                solver.step(state, out, None, contacts, dt)

                # Scalar backward-Euler balance on the loading branch:
                # (m/h^2)*(Cprev+h*speed-C) = k*C+(kd/h)*(C-max(Cprev,0)).
                inertia = mass / dt**2
                damper = damping / dt
                compression = (inertia * (previous + dt * speed) + damper * max(previous, 0.0)) / (
                    inertia + stiffness + damper
                )
                test.assertGreater(compression, max(previous, 0.0))
                expected_velocity = (previous - compression) / dt
                test.assertAlmostEqual(float(out.body_qd.numpy()[0, 2]), expected_velocity, delta=tolerance)


@wp.kernel
def _eval_damped_extreme_kernel(
    penetrations: wp.array[float],
    prev_penetrations: wp.array[float],
    kes: wp.array[float],
    kds: wp.array[float],
    rhos: wp.array[float],
    lams: wp.array[float],
    dt: float,
    out_force: wp.array[float],
    out_elastic: wp.array[float],
    out_metric: wp.array[float],
):
    i = wp.tid()
    f, f_el, m = _compliant_contact_normal_response(
        penetrations[i], prev_penetrations[i], kes[i], kds[i], rhos[i], lams[i], dt
    )
    out_force[i] = f
    out_elastic[i] = f_el
    out_metric[i] = m


def test_damped_normal_extreme_rho_and_stiff_ratio(test, device):
    """Verify damped normal elimination neither overflows at huge rho nor freezes warm duals at stiff ratios."""
    with wp.ScopedDevice(device):
        # Row 0: large rho (1e36) must stay finite: ke * rho (1e41) exceeds float32, although
        # rho * increment (1e33) does not.
        # Row 1: (ke + kd/dt) / rho = 5e9 / 200 = 2.5e7 > 2^24 (so s == 1.0 in float32)
        # with compression_increment = 0 and transition_force = 1.0, multiplier_excess = 1e6:
        # _alm_relaxed_ascent relaxes overshoot by a * excess = 4e-8 * 1e6 = 0.04 N (0.5 * ULP(1e6) = 0.03125 N).
        penetrations = wp.array([0.002, 1.0e-9], dtype=float)
        prev_penetrations = wp.array([0.001, 1.0e-9], dtype=float)
        kes = wp.array([1.0e5, 1.0e9], dtype=float)
        kds = wp.array([1.0e2, 4.0e6], dtype=float)
        rhos = wp.array([1.0e36, 200.0], dtype=float)
        lams = wp.array([100.0, 1000001.0], dtype=float)
        out_f = wp.empty(2, dtype=float)
        out_el = wp.empty(2, dtype=float)
        out_m = wp.empty(2, dtype=float)
        wp.launch(
            _eval_damped_extreme_kernel,
            dim=2,
            inputs=[penetrations, prev_penetrations, kes, kds, rhos, lams, 0.001],
            outputs=[out_f, out_el, out_m],
        )
        vals = out_f.numpy()
        el_vals = out_el.numpy()
        m_vals = out_m.numpy()
        test.assertTrue(np.all(np.isfinite(vals)))
        test.assertTrue(np.all(np.isfinite(el_vals)))
        test.assertTrue(np.all(np.isfinite(m_vals)))
        test.assertAlmostEqual(float(vals[0]), 300.0, delta=1.0e-3)
        test.assertAlmostEqual(float(el_vals[0]), 200.0, delta=1.0e-3)
        test.assertLess(float(vals[1]), 1000001.0)
        test.assertAlmostEqual(float(vals[1]), 1000001.0 - 0.04, delta=0.04)


def test_step_resolves_per_contact_materials(test, device):
    """Read per-contact stiffness, damping and friction in the solver step, with and without contact history."""
    shape_ke, shape_kd, shape_mu, shape_tor, shape_roll = 1.0e4, 20.0, 0.5, 0.02, 0.01
    # Per-contact stiffness, damping and friction scale (zero is unset), and the material each row must get.
    overrides = np.array([[0.0, 0.0, 0.0], [2.5e3, 0.0, 0.0], [4.0e4, 0.0, 2.0], [2.5e3, 7.0, 0.5]], dtype=np.float32)
    expected_ke_kd = np.array(
        [
            [shape_ke, shape_kd],
            [2.5e3, 0.25 * shape_kd],  # a softer row keeps the relaxation time kd / ke
            [4.0e4, shape_kd],  # a stiffer row keeps the authored damping
            [2.5e3, 7.0],  # an explicit damping replaces the scaled one
        ]
    )
    sliding_mu = np.array([shape_mu, shape_mu, 2.0 * shape_mu, 0.5 * shape_mu])
    mode_cases = (
        (newton.ShapeFrictionMode.NONE, np.zeros((4, 3))),
        (newton.ShapeFrictionMode.SLIDING, np.column_stack((sliding_mu, np.zeros(4), np.zeros(4)))),
        (
            newton.ShapeFrictionMode.SLIDING_TORSIONAL_ROLLING,
            np.column_stack((sliding_mu, np.full(4, shape_tor), np.full(4, shape_roll))),
        ),
    )
    count = len(overrides)
    with wp.ScopedDevice(device):
        for friction_mode, expected_mu in mode_cases:
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            cfg = builder.ShapeConfig(
                ke=shape_ke,
                kd=shape_kd,
                mu=shape_mu,
                mu_torsional=shape_tor,
                mu_rolling=shape_roll,
                friction_mode=friction_mode,
                density=0.0,
            )
            body = builder.add_body(
                xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3)),
                lock_inertia=True,
            )
            sphere = builder.add_shape_sphere(body, radius=0.01, cfg=cfg)
            ground = builder.add_ground_plane(cfg=cfg)
            builder.color()
            model = builder.finalize()
            for history in (False, True):
                with test.subTest(friction_mode=friction_mode.name, history=history):
                    contacts = newton.Contacts(
                        count, 0, device=device, per_contact_shape_properties=True, contact_matching=history
                    )
                    if history:
                        contacts._contact_matching_mode = "latest"
                    contacts.rigid_contact_count.fill_(count)
                    contacts.rigid_contact_shape0.fill_(sphere)
                    contacts.rigid_contact_shape1.fill_(ground)
                    contacts.rigid_contact_normal.assign([(0.0, 0.0, -1.0)] * count)
                    contacts.rigid_contact_stiffness.assign(overrides[:, 0])
                    contacts.rigid_contact_damping.assign(overrides[:, 1])
                    contacts.rigid_contact_friction.assign(overrides[:, 2])
                    contacts.contact_generation.fill_(1)
                    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_contact_history=history)
                    state, out = model.state(), model.state()
                    solver.step(state, out, None, contacts, 1.0 / 600.0)
                    np.testing.assert_allclose(
                        solver.body_body_contact_material_ke.numpy()[:count], expected_ke_kd[:, 0], rtol=1.0e-6
                    )
                    np.testing.assert_allclose(
                        solver.body_body_contact_material_kd.numpy()[:count], expected_ke_kd[:, 1], rtol=1.0e-6
                    )
                    np.testing.assert_allclose(
                        solver.body_body_contact_material_mu.numpy()[:count], expected_mu, rtol=1.0e-6
                    )


class TestRigidContactDamping(unittest.TestCase):
    pass


add_function_test(
    TestRigidContactDamping, "test_solver_force_reporting", test_solver_force_reporting, devices=get_test_devices()
)


add_function_test(
    TestRigidContactDamping,
    "test_compliant_contact_step_force_balance",
    test_compliant_contact_step_force_balance,
    devices=get_test_devices(),
)


add_function_test(
    TestRigidContactDamping, "test_compliant_damping_modes", test_compliant_damping_modes, devices=get_test_devices()
)
add_function_test(
    TestRigidContactDamping,
    "test_damped_normal_extreme_rho_and_stiff_ratio",
    test_damped_normal_extreme_rho_and_stiff_ratio,
    devices=get_test_devices(),
)
add_function_test(
    TestRigidContactDamping,
    "test_step_resolves_per_contact_materials",
    test_step_resolves_per_contact_materials,
    devices=get_test_devices(),
)
add_function_test(
    TestRigidContactDamping,
    "test_reported_torque_about_current_com",
    test_reported_torque_about_current_com,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
