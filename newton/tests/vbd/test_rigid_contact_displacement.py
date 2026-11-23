# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid contact point motion: small slips survive world translation, and rolling creates no energy."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_contact_activation import create_accumulate_inactive_contact_metric
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _body_rotation_increment,
    compute_body_body_contact_forces,
    compute_rigid_contact_forces,
    create_accumulate_body_body_contacts_per_body,
    step_body_body_contact_C0_lambda,
    update_duals_body_body_contacts,
)
from newton._src.solvers.vbd.vbd_coupling_kernels import _harvest_vbd_proxy_wrenches_kernel
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _contact_rotation_increments(q: wp.array[wp.quat], q_prev: wp.array[wp.quat], rotation: wp.array[wp.vec3]):
    """Evaluate the production contact rotation map for a small quaternion batch."""
    i = wp.tid()
    increment, _jacobian_t = _body_rotation_increment(
        wp.transform(wp.vec3(0.0), q[i]), wp.transform(wp.vec3(0.0), q_prev[i])
    )
    rotation[i] = increment


def test_small_rotation_contact_force_is_translation_invariant(test, device):
    """Preserve small rotational slip and exact zero increments at nonidentity orientations."""
    radius = 2.0**-10
    angle = 2.0**-18
    penetration = 2.0**-10
    material_ke = 2.0**10
    normal_force = material_ke * penetration
    rho_t = 1.0e6
    dt = 1.0 / 600.0
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        cfg = builder.ShapeConfig(ke=material_ke, kd=0.0, mu=1.0, density=0.0)
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        shape0 = builder.add_shape_sphere(body, radius=0.01, cfg=cfg)
        shape1 = builder.add_ground_plane(cfg=cfg)
        builder.color()
        model = builder.finalize()
        solver = newton.solvers.SolverVBD(model, iterations=4)
        contacts = newton.Contacts(1, 0, device=device)
        contacts.rigid_contact_count.fill_(1)
        contacts.rigid_contact_shape0.fill_(shape0)
        contacts.rigid_contact_shape1.fill_(shape1)
        contacts.rigid_contact_point0.assign([(radius, 0.0, 0.0)])
        contacts.rigid_contact_point1.assign([(radius, 0.0, penetration)])
        contacts.rigid_contact_normal.assign([(0.0, 0.0, -1.0)])
        contacts.contact_generation.fill_(1)
        state, out = model.state(), model.state()
        # Initialize the normal solver lifecycle before prescribing a frozen row.
        solver.step(state, out, None, contacts, dt)
        solver.body_body_contact_normal_rho.fill_(1.0e6)
        solver.body_body_contact_material_ke.fill_(material_ke)
        solver.body_body_contact_material_kd.zero_()
        solver.body_body_contact_material_mu.fill_(1.0)
        solver.body_body_contact_tangent_rho.fill_(rho_t)
        solver.body_body_contact_C0.zero_()
        solver.body_body_contact_lambda.assign([(0.0, 0.0, -normal_force)])
        wrench = wp.zeros(1, dtype=wp.spatial_vector)

        # Independent first-order point-motion oracle. A small z rotation moves
        # (r,0,0) by (0,r*omega,0). Friction on dynamic shape0 opposes that slip.
        # Use the float32 quaternion actually supplied to the solver.
        q = np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)], dtype=np.float32)
        omega_z = 2.0 * float(q[2])
        expected_force = np.array([0.0, -rho_t * radius * omega_z, normal_force])
        # The reported torque is about the body's current center of mass, from its current lever.
        qz, qw = float(q[2]), float(q[3])
        lever = radius * np.array([qw * qw - qz * qz, 2.0 * qw * qz, 0.0])
        expected_torque = np.cross(lever, expected_force)
        test.assertLess(abs(expected_force[1]), normal_force, "The row must remain inside its friction cone")

        reported = []
        for translation in (0.0, 1.0):
            with test.subTest(translation=translation):
                p = wp.vec3(translation, translation, translation)
                state.body_q.assign([wp.transform(p, wp.quat(*q))])
                solver.body_q_prev.assign([wp.transform(p, wp.quat_identity())])
                contacts.rigid_contact_point1.assign([(translation + radius, translation, translation + penetration)])
                values = solver.collect_rigid_contact_forces(state.body_q, solver.body_q_prev, contacts, dt)
                solver._export_body_body_contact_forces(state, contacts, dt, wrench)
                test.assertEqual(int(values[5].numpy()[0]), 1)
                force0 = -values[4].numpy()[0].astype(np.float64)
                exported = wrench.numpy()[0].astype(np.float64)
                np.testing.assert_allclose(force0, expected_force, rtol=2.0e-5, atol=1.0e-8)
                np.testing.assert_allclose(exported[:3], expected_force, rtol=2.0e-5, atol=1.0e-8)
                np.testing.assert_allclose(exported[3:], expected_torque, rtol=2.0e-5, atol=1.0e-10)
                reported.append(exported)
        if len(reported) == 2:
            np.testing.assert_allclose(reported[1], reported[0], rtol=2.0e-5, atol=1.0e-10)

        # An identity baseline hides cancellation in q * inverse(q_prev), especially under CUDA FMA.
        # Exercise exact zero and a tiny increment at a generic baseline, including q versus -q.
        baseline = np.array([1.0, 2.0, 3.0, 4.0])
        baseline /= np.linalg.norm(baseline)
        axis = np.array([2.0, -1.0, 3.0])
        axis /= np.linalg.norm(axis)
        dv = axis * np.sin(0.5e-6)
        dw = np.cos(0.5e-6)
        tiny = np.concatenate(
            (dw * baseline[:3] + baseline[3] * dv + np.cross(dv, baseline[:3]), [dw * baseline[3] - dv @ baseline[:3]])
        ).astype(np.float32)
        baseline = baseline.astype(np.float32)
        # Independent Hamilton product in float64, using the encoded inputs rather than the requested angle.
        a, b = tiny.astype(np.float64), baseline.astype(np.float64)
        expected_increment = 2.0 * (b[3] * a[:3] - a[3] * b[:3] - np.cross(a[:3], b[:3]))
        rotations = wp.zeros(4, dtype=wp.vec3, device=device)
        wp.launch(
            _contact_rotation_increments,
            dim=4,
            inputs=[
                wp.array([baseline, -baseline, tiny, -tiny], dtype=wp.quat, device=device),
                wp.array([baseline] * 4, dtype=wp.quat, device=device),
            ],
            outputs=[rotations],
            device=device,
        )
        increments = rotations.numpy()
        np.testing.assert_array_equal(increments[:2], np.zeros((2, 3)))
        np.testing.assert_allclose(increments[2:], np.tile(expected_increment, (2, 1)), rtol=5.0e-7, atol=1.0e-14)


def test_rolling_cylinder_keeps_its_speed(test, device):
    """Contact points on a curved surface must not feed energy into a rolling body."""
    radius, half_height, speed, dt, steps = 0.1, 0.1, 1.0, 1.0 / 120.0, 30
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        cfg = builder.ShapeConfig(density=0.0, mu=1.0, mu_torsional=0.0, mu_rolling=0.0)
        builder.add_ground_plane(cfg=cfg)
        transverse = (3.0 * radius**2 + (2.0 * half_height) ** 2) / 12.0
        body = builder.add_body(
            # The cylinder axis (body z) lies along world y, so the body rolls along x.
            xform=wp.transform(
                wp.vec3(0.0, 0.0, radius - 1.0e-4), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)
            ),
            mass=1.0,
            inertia=wp.mat33(np.diag([transverse, transverse, 0.5 * radius**2])),
            lock_inertia=True,
        )
        builder.add_shape_cylinder(body, radius=radius, half_height=half_height, cfg=cfg)
        builder.color()
        model = builder.finalize()
        solver = newton.solvers.SolverVBD(model, iterations=10)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        state, out = model.state(), model.state()
        velocity = state.body_qd.numpy()
        velocity[0, 0] = speed
        velocity[0, 4] = speed / radius
        state.body_qd.assign(velocity)
        for _ in range(steps):
            state.clear_forces()
            pipeline.collide(state, contacts)
            solver.step(state, out, None, contacts, dt)
            state, out = out, state
        # No rolling resistance is modelled and nothing drives the cylinder, so its speed is conserved.
        # Carrying the witness points around the axis with a fixed normal opens the gap ahead of the
        # contact and accelerates the roll by g * omega * dt / 1.5, which is +14 % over these steps.
        test.assertAlmostEqual(float(state.body_qd.numpy()[0, 0]), speed, delta=0.03 * speed)


def test_small_normal_rotation_is_translation_invariant(test, device):
    """Preserve tiny normal motion in the solve, history, reports and activation guard."""
    radius, dt, stiffness, penalty = 2.0**-10, 0.01, 1.0e6, 1.0e6
    cases = [
        (angle, height, seeded, body_a)
        for angle in (0.0, 2.0**-18)
        for height in (0.0, 1.0)
        for seeded in (False, True)
        for body_a in (False, True)
    ]
    count = len(cases)
    quaternions = np.array([[0.0, np.sin(a / 2.0), 0.0, np.cos(a / 2.0)] for a, _, _, _ in cases], dtype=np.float32)
    # Independent first-order map of the encoded quaternion: the point (r,0,0)
    # moves down by r*2*q_y. This is below the float32 ULP at world height 1 m.
    compression = radius * 2.0 * quaternions[:, 1].astype(np.float64)
    sign = np.array([-1.0 if body_a else 1.0 for _, _, _, body_a in cases])
    seeded = np.array([warm for _, _, warm, _ in cases])
    with wp.ScopedDevice(device):
        q = wp.array(
            [wp.transform((0.0, 0.0, h), wp.quat(*rot)) for (_, h, _, _), rot in zip(cases, quaternions, strict=True)],
            dtype=wp.transform,
        )
        previous = wp.array(
            [wp.transform((0.0, 0.0, h), wp.quat_identity()) for _, h, _, _ in cases], dtype=wp.transform
        )
        vectors = wp.zeros(count, dtype=wp.vec3)
        zeros = wp.zeros(count)
        body_ids = wp.array(np.arange(count), dtype=int)
        colors = wp.zeros(count, dtype=int)
        inverse_mass = wp.ones(count)
        rho = wp.full(count, penalty)
        ke = wp.full(count, stiffness)
        kd = wp.zeros(count)
        lam = wp.zeros(count, dtype=wp.vec3)
        normals = wp.array(np.column_stack((np.zeros((count, 2)), sign)), dtype=wp.vec3)
        shape_body = wp.array(
            [b for i, (_, _, _, a) in enumerate(cases) for b in ((i, -1) if a else (-1, i))], dtype=int
        )
        geometry = [
            wp.array([count], dtype=int),
            wp.array(np.arange(count) * 2, dtype=int),
            wp.array(np.arange(count) * 2 + 1, dtype=int),
            wp.array([(radius, 0.0, 0.0 if a else h) for _, h, _, a in cases], dtype=wp.vec3),
            wp.array([(radius, 0.0, h if a else 0.0) for _, h, _, a in cases], dtype=wp.vec3),
            vectors,
            vectors,
            vectors,
            normals,
            zeros,
            zeros,
        ]
        material = [rho, ke, kd, zeros, zeros, lam, vectors, 0.0]
        adjacency = [1, wp.ones(count, dtype=int), body_ids]
        for damping, bidirectional in ((0.0, False), (1.0e4, False), (1.0e4, True)):
            kd.fill_(damping)
            dashpot = damping / dt
            row_stiffness = stiffness if bidirectional else stiffness + dashpot
            initial_lambda = seeded * row_stiffness * compression
            lam.assign(np.column_stack((np.zeros((count, 2)), sign * initial_lambda)))
            expected_dual = row_stiffness / (row_stiffness + penalty) * (penalty * compression + initial_lambda)
            expected_force = expected_dual + (dashpot * compression if bidirectional else 0.0)
            report_inputs = [dt, *geometry, shape_body, q, previous, vectors, *material, 0.01, bidirectional]
            reported = [wp.empty(count, dtype=t) for t in (int, int, wp.vec3, wp.vec3, wp.vec3)]
            wrench = wp.empty(count, dtype=wp.spatial_vector)
            wp.launch(compute_rigid_contact_forces, count, inputs=report_inputs, outputs=reported)
            wp.launch(compute_body_body_contact_forces, count, inputs=report_inputs, outputs=[wrench])
            with test.subTest(damping=damping, bidirectional=bidirectional):
                np.testing.assert_allclose(reported[-1].numpy()[:, 2] * sign, expected_force, rtol=2.0e-6, atol=1.0e-10)
                np.testing.assert_allclose(-wrench.numpy()[:, 2] * sign, expected_force, rtol=2.0e-6, atol=1.0e-10)
            for width in (4, 32):
                gathered = [wp.zeros(count, dtype=t) for t in (wp.vec3, wp.vec3, wp.mat33, wp.mat33, wp.mat33)]
                wp.launch(
                    create_accumulate_body_body_contacts_per_body(width),
                    count * width,
                    inputs=[
                        dt,
                        body_ids,
                        previous,
                        q,
                        vectors,
                        inverse_mass,
                        colors,
                        0.01,
                        bidirectional,
                        *material,
                        *geometry,
                        shape_body,
                        *adjacency,
                    ],
                    outputs=gathered,
                )
                with test.subTest(damping=damping, bidirectional=bidirectional, width=width):
                    np.testing.assert_allclose(gathered[0].numpy()[:, 2], expected_force, rtol=2.0e-6, atol=1.0e-10)
            wp.launch(
                update_duals_body_body_contacts,
                count,
                inputs=[
                    *geometry,
                    shape_body,
                    q,
                    previous,
                    vectors,
                    dt,
                    zeros,
                    vectors,
                    0.0,
                    ke,
                    kd,
                    zeros,
                    rho,
                    bidirectional,
                    lam,
                ],
            )
            np.testing.assert_allclose(lam.numpy()[:, 2] * sign, expected_dual, rtol=2.0e-6, atol=1.0e-10)

            # The zero-increment controls must not activate; the tiny trial must
            # activate at either height without adding a physical force.
            lam.zero_()
            metric = penalty * row_stiffness / (penalty + row_stiffness)
            if bidirectional:
                metric += dashpot
            for width in (4, 32):
                guarded = [wp.zeros(count, dtype=wp.mat33) for _ in range(3)]
                activated = wp.zeros(count, dtype=int)
                guard_inputs = [
                    dt,
                    body_ids,
                    previous,
                    previous,
                    q,
                    vectors,
                    inverse_mass,
                    colors,
                    rho,
                    ke,
                    kd,
                    lam,
                    vectors,
                    0.0,
                    bidirectional,
                    *geometry[:5],
                    normals,
                    zeros,
                    zeros,
                    shape_body,
                    *adjacency,
                ]
                wp.launch(
                    create_accumulate_inactive_contact_metric(width),
                    count * width,
                    inputs=guard_inputs,
                    outputs=[*guarded, activated],
                )
                np.testing.assert_array_equal(activated.numpy(), compression > 0.0)
                np.testing.assert_allclose(
                    guarded[0].numpy()[:, 2, 2], (compression > 0.0) * metric, rtol=2.0e-6, atol=1.0e-6
                )

                # With the same tiny rotation already in the current iterate,
                # its ordinary contact block must not receive another guard metric.
                guard_inputs[3] = q
                for output in (*guarded, activated):
                    output.zero_()
                wp.launch(
                    create_accumulate_inactive_contact_metric(width),
                    count * width,
                    inputs=guard_inputs,
                    outputs=[*guarded, activated],
                )
                with test.subTest(damping=damping, bidirectional=bidirectional, width=width, current_active=True):
                    for output in (*guarded, activated):
                        np.testing.assert_array_equal(output.numpy(), 0)


def test_large_world_translation_invariance_and_c0_precision(test, device):
    """Verify 1000 m translation invariance for millimeter levers, offset COMs, C0, and proxy torques."""
    with wp.ScopedDevice(device):
        # 1. C0 normal precision on tilted normal with large tangential offset at 1000 m
        n_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        n_np /= np.linalg.norm(n_np)
        t_np = np.array([2.0, -1.0, 0.0], dtype=np.float64)
        t_np /= np.linalg.norm(t_np)
        expected_c0 = 1.0e-6
        p0_local = np.zeros(3, dtype=np.float32)
        p1_local = (-expected_c0 * n_np + 0.5 * t_np).astype(np.float32)
        n_f32 = n_np.astype(np.float32)
        n_f64 = n_f32.astype(np.float64)
        expected_exact = -float(np.dot(n_f64, p1_local.astype(np.float64))) * float(np.dot(n_f64, n_f64))
        # The two large products of dot(n, p1) are about 0.12 and cancel. Each may round by half a
        # float32 spacing there (3.7e-9), depending on evaluation order and fused multiply-adds.
        c0_tolerance = 1.0e-8

        c0_by_shift = []
        for world_shift in (0.0, 1000.0):
            X_world = wp.transform((world_shift, -world_shift, 0.5 * world_shift), wp.quat_identity())
            c0_out = wp.zeros(1, dtype=wp.vec3, device=device)
            wp.launch(
                step_body_body_contact_C0_lambda,
                dim=1,
                inputs=[
                    wp.array([1], dtype=int),
                    wp.array([0], dtype=int),
                    wp.array([1], dtype=int),
                    wp.array([p0_local], dtype=wp.vec3),
                    wp.array([p1_local], dtype=wp.vec3),
                    wp.zeros(1, dtype=wp.vec3),
                    wp.zeros(1, dtype=wp.vec3),
                    wp.array([n_np.astype(np.float32)], dtype=wp.vec3),
                    wp.zeros(1, dtype=float),
                    wp.zeros(1, dtype=float),
                    wp.array([0, 1], dtype=wp.int32),
                    wp.zeros(2, dtype=wp.int32),
                    wp.array([1.0, 1.0], dtype=float),
                    wp.zeros(2, dtype=wp.mat33),
                    wp.zeros(2, dtype=wp.vec3),
                    wp.zeros(2, dtype=float),
                    int(newton.BodyFlags.PROXY),
                    wp.array([X_world, X_world], dtype=wp.transform),
                    1.0,
                    100.0,
                    wp.array([1.0e6], dtype=float),
                    1.0,
                ],
                outputs=[
                    wp.zeros(1, dtype=float),
                    c0_out,
                    wp.zeros(1, dtype=wp.vec3),
                    wp.zeros(1, dtype=float),
                ],
            )
            c0_scalar = float(np.dot(c0_out.numpy()[0].astype(np.float64), n_f64))
            c0_by_shift.append(c0_scalar)
            test.assertAlmostEqual(c0_scalar, expected_exact, delta=c0_tolerance)
        test.assertEqual(c0_by_shift[1], c0_by_shift[0])

        # 2. Offset-COM slip & 1 mm lever force/torque + proxy wrench invariance between 0 m and 1000 m
        wrenches_by_shift = []
        proxy_by_shift = []
        q_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 1.0e-3)
        for world_shift in (0.0, 1000.0):
            base = np.array([world_shift, -0.75 * world_shift, 0.25 * world_shift], dtype=np.float32)
            X_prev = wp.transform(tuple(base), wp.quat_identity())
            X_cur = wp.transform(tuple(base), q_rot)
            body_q_prev = wp.array([X_prev, X_prev], dtype=wp.transform)
            body_q_cur = wp.array([X_cur, X_prev], dtype=wp.transform)
            body_com = wp.array([[0.05, -0.03, 0.02], [0.0, 0.0, 0.0]], dtype=wp.vec3)
            pt0 = wp.array([[0.051, -0.03, 0.02]], dtype=wp.vec3)
            pt1 = wp.array([[0.051, -0.03, 0.0199]], dtype=wp.vec3)
            off0 = wp.array([[0.0, 0.0, -0.002]], dtype=wp.vec3)
            off1 = wp.array([[0.0, 0.0, 0.002]], dtype=wp.vec3)
            normal_arr = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3)
            zeros_f = wp.zeros(1, dtype=float)
            wrench = wp.zeros(1, dtype=wp.spatial_vector)
            wp.launch(
                compute_body_body_contact_forces,
                dim=1,
                inputs=[
                    0.01,
                    wp.array([1], dtype=int),
                    wp.array([0], dtype=int),
                    wp.array([1], dtype=int),
                    pt0,
                    pt1,
                    None,
                    off0,
                    off1,
                    normal_arr,
                    zeros_f,
                    zeros_f,
                    wp.array([0, 1], dtype=wp.int32),
                    body_q_cur,
                    body_q_prev,
                    body_com,
                    wp.array([1.0e5], dtype=float),
                    wp.array([1.0e5], dtype=float),
                    zeros_f,
                    wp.array([0.8], dtype=float),
                    wp.array([1.0e5], dtype=float),
                    wp.array([[0.0, 0.0, 50.0]], dtype=wp.vec3),
                    wp.zeros(1, dtype=wp.vec3),
                    0.0,
                    1.0e-6,
                    False,
                ],
                outputs=[wrench],
            )
            w_np = wrench.numpy()[0].copy()
            wrenches_by_shift.append(w_np)

            proxy_out = wp.zeros(1, dtype=wp.spatial_vector)
            wp.launch(
                _harvest_vbd_proxy_wrenches_kernel,
                dim=1,
                inputs=[
                    wp.array([1], dtype=int),
                    wp.array([0], dtype=wp.int32),
                    wp.array([1], dtype=wp.int32),
                    pt0,
                    pt1,
                    wp.array([-w_np[:3]], dtype=wp.vec3),
                    normal_arr,
                    off0,
                    off1,
                    wp.array([0.0, 1.0], dtype=float),
                    wp.array([int(newton.BodyFlags.PROXY), 0], dtype=wp.int32),
                    wp.array([0, -1], dtype=int),
                    int(newton.BodyFlags.PROXY),
                    body_com,
                    body_q_cur,
                    proxy_out,
                ],
            )
            proxy_by_shift.append(proxy_out.numpy()[0].copy())

        # The row is active: the equalities below would also hold for three zero wrenches.
        test.assertGreater(float(np.linalg.norm(wrenches_by_shift[0][:3])), 1.0)
        np.testing.assert_allclose(wrenches_by_shift[1], wrenches_by_shift[0], rtol=2.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(proxy_by_shift[1], proxy_by_shift[0], rtol=2.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(proxy_by_shift[0], wrenches_by_shift[0], rtol=2.0e-6, atol=1.0e-7)


class TestRigidContactDisplacement(unittest.TestCase):
    pass


add_function_test(
    TestRigidContactDisplacement,
    "test_small_rotation_contact_force_is_translation_invariant",
    test_small_rotation_contact_force_is_translation_invariant,
    devices=get_test_devices(),
)


add_function_test(
    TestRigidContactDisplacement,
    "test_small_normal_rotation_is_translation_invariant",
    test_small_normal_rotation_is_translation_invariant,
    devices=get_test_devices(),
)


add_function_test(
    TestRigidContactDisplacement,
    "test_rolling_cylinder_keeps_its_speed",
    test_rolling_cylinder_keeps_its_speed,
    devices=get_test_devices(),
)


add_function_test(
    TestRigidContactDisplacement,
    "test_large_world_translation_invariance_and_c0_precision",
    test_large_world_translation_invariance_and_c0_precision,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
