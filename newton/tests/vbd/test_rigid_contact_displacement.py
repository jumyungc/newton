# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid contact point motion: small increments survive world translation, and rolling creates no energy."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_planar_recovery import MeshRecoveryData
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _NUM_CONTACT_THREADS_PER_BODY,
    _body_rotation_increment,
    _body_step_motion,
    accumulate_body_body_contacts_per_body,
    build_body_body_contact_step_levers,
    update_duals_body_body_contacts,
)
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
    """Preserve small normal motion, rotational slip and exact zero increments at nonidentity orientations."""
    radius = 2.0**-10
    angle = 2.0**-18
    penetration = 2.0**-10
    material_ke = 2.0**10
    normal_force = material_ke * penetration
    rho_t = 1.0e6
    dt = 1.0 / 600.0
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        # Sliding friction only: the small spin about the normal would otherwise load the torsional channel.
        cfg = builder.ShapeConfig(ke=material_ke, kd=0.0, mu=1.0, mu_torsional=0.0, mu_rolling=0.0, density=0.0)
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
        solver.body_body_contact_material_mu.fill_(wp.vec3(1.0, 0.0, 0.0))
        solver.body_body_contact_tangent_rho.fill_(wp.mat22(rho_t, rho_t, 1.0, 0.0))
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

        # A tiny normal approach must also survive the world-point rounding that
        # the solve avoids. Use a bottom-edge witness and rotate about y instead.
        stiffness = 1.0e6
        q_normal = np.array([0.0, np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0)], dtype=np.float32)
        compression = radius * 2.0 * float(q_normal[1])
        test.assertGreater(compression, 1.0e-9)
        test.assertLess(compression, np.spacing(np.float32(1.0)))
        solver.body_body_contact_material_ke.fill_(stiffness)
        solver.body_body_contact_material_mu.zero_()
        solver.body_body_contact_lambda_angular.zero_()
        contacts.rigid_contact_point0.assign([(radius, 0.0, -radius)])
        color_group = wp.array([body], dtype=wp.int32)
        counts, indices = wp.array([1], dtype=wp.int32), wp.array([0], dtype=wp.int32)
        records = wp.array([(body, -1, 0)], dtype=wp.vec3i)
        gathered = [wp.zeros(1, dtype=wp.vec3) for _ in range(2)]
        gathered += [wp.zeros(1, dtype=wp.mat33) for _ in range(3)]
        levers = [getattr(solver, "body_body_contact_lever_" + name) for name in ("s0", "c0", "s1", "c1")]
        levers.append(solver.body_body_contact_start_penetration)
        material = [
            solver.body_body_contact_normal_rho,
            solver.body_body_contact_material_ke,
            solver.body_body_contact_material_kd,
            solver.body_body_contact_material_mu,
            solver.body_body_contact_tangent_rho,
            solver.body_body_contact_torsional_rho,
            solver.body_body_contact_rolling_rho,
        ]
        gather_inputs = [
            dt,
            color_group,
            solver.body_q_prev,
            state.body_q,
            model.body_com,
            False,
            *material,
            solver.body_body_contact_lambda,
            solver.body_body_contact_lambda_angular,
            solver.body_body_contact_C0,
            MeshRecoveryData(),
            contacts.contact_generation,
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_surface_velocity,
            contacts.rigid_contact_normal,
            *levers,
            1,
            counts,
            indices,
            records,
        ]
        dual_inputs = [
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_surface_velocity,
            contacts.rigid_contact_normal,
            *levers,
            model.shape_body,
            state.body_q,
            solver.body_q_prev,
            model.body_com,
            solver.body_contact_free_angular_compliance,
            0,
            dt,
            material[3],
            solver.body_body_contact_C0,
            MeshRecoveryData(),
            contacts.contact_generation,
            material[1],
            material[2],
            material[4],
            material[0],
            material[5],
            material[6],
            False,
            solver.body_body_contact_lambda,
            solver.body_body_contact_lambda_angular,
        ]
        for translation in (0.0, 1.0):
            for warm in (0.0, stiffness * compression):
                with test.subTest(normal_translation=translation, normal_warm_start=warm):
                    position = wp.vec3(translation, translation, translation + radius)
                    state.body_q.assign([wp.transform(position, wp.quat(*q_normal))])
                    solver.body_q_prev.assign([wp.transform(position, wp.quat_identity())])
                    contacts.rigid_contact_point1.assign([(translation + radius, translation, translation)])
                    solver.body_body_contact_lambda.assign([(0.0, 0.0, -warm)])
                    solver._build_body_body_contact_step_levers(contacts)
                    lambda_normal = warm
                    for after_dual in (False, True):
                        expected_normal = 0.5 * (stiffness * compression + lambda_normal)
                        expected_force = np.array([0.0, 0.0, expected_normal])
                        expected_torque = np.array([0.0, -radius * expected_normal * float(q_normal[3]), 0.0])
                        qy, qw = float(q_normal[1]), float(q_normal[3])
                        lever_x = radius * (qw * qw - qy * qy - 2.0 * qw * qy)
                        expected_reported_torque = np.array([0.0, -lever_x * expected_normal, 0.0])
                        for output in gathered:
                            output.zero_()
                        wp.launch(
                            accumulate_body_body_contacts_per_body,
                            _NUM_CONTACT_THREADS_PER_BODY,
                            inputs=gather_inputs,
                            outputs=gathered,
                        )
                        values = solver.collect_rigid_contact_forces(state.body_q, solver.body_q_prev, contacts, dt)
                        solver._export_body_body_contact_forces(state, contacts, dt, wrench)
                        np.testing.assert_allclose(
                            gathered[0].numpy()[0],
                            expected_force,
                            rtol=2.0e-6,
                            atol=1.0e-9,
                        )
                        np.testing.assert_allclose(
                            gathered[1].numpy()[0],
                            expected_torque,
                            rtol=2.0e-6,
                            atol=1.0e-12,
                        )
                        np.testing.assert_allclose(
                            -values[4].numpy()[0],
                            expected_force,
                            rtol=2.0e-6,
                            atol=1.0e-9,
                        )
                        np.testing.assert_allclose(
                            wrench.numpy()[0, :3],
                            expected_force,
                            rtol=2.0e-6,
                            atol=1.0e-9,
                        )
                        np.testing.assert_allclose(
                            wrench.numpy()[0, 3:],
                            expected_reported_torque,
                            rtol=2.0e-6,
                            atol=1.0e-12,
                        )
                        if not after_dual:
                            wp.launch(update_duals_body_body_contacts, 1, inputs=dual_inputs)
                            np.testing.assert_allclose(
                                solver.body_body_contact_lambda.numpy()[0],
                                -expected_force,
                                rtol=2.0e-6,
                                atol=1.0e-9,
                            )
                            lambda_normal = expected_normal

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


def _expected_contact_step_rows(model, contacts, body_q_prev):
    """Recompute each active row's step-start levers and penetration in float64."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    com = model.body_com.numpy().astype(np.float64)
    names = ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal", "margin0", "margin1")
    shape0, shape1, point0, point1, offset0, offset1, normal, margin0, margin1 = (
        getattr(contacts, "rigid_contact_" + name).numpy()[:count].astype(np.float64) for name in names
    )

    def rotate(q, v):
        return v + 2.0 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)

    rows = np.zeros((count, 13))
    for i in range(count):
        ends = []
        for shape, point, offset in ((shape0[i], point0[i], offset0[i]), (shape1[i], point1[i], offset1[i])):
            body = shape_body[int(shape)] if shape >= 0 else -1
            if body < 0:
                # A world endpoint is a body at the identity with its COM at the origin.
                ends.append((point, point, point + offset))
                continue
            position, q = body_q_prev[body, :3].astype(np.float64), body_q_prev[body, 3:].astype(np.float64)
            center = position + rotate(q, com[body])
            world = position + rotate(q, point)
            ends.append((world, world - center, position + rotate(q, point + offset) - center))
        (world0, s0, c0), (world1, s1, c1) = ends
        penetration = margin0[i] + margin1[i] - normal[i] @ (world1 - world0)
        rows[i] = np.concatenate((s0, c0, s1, c1, [penetration]))
    return rows


def _stored_contact_step_rows(solver, count):
    """Return the solver's stored step-start levers and penetration for the first ``count`` rows."""
    levers = [getattr(solver, "body_body_contact_lever_" + name).numpy()[:count] for name in ("s0", "c0", "s1", "c1")]
    return np.concatenate([*levers, solver.body_body_contact_start_penetration.numpy()[:count, None]], axis=1)


def test_contact_step_levers_follow_rows_and_history(test, device):
    """Stored row geometry follows the rows and pose history of the iterations; reporting does not read it."""
    Slot = newton.solvers.SolverBase.CollisionSlot
    Frequency = newton.solvers.SolverBase.CollisionFrequencyType
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(ke=1.0e4, kd=10.0, mu=0.5, density=1000.0)
    # A second sphere on each body moves its COM off the contacting sphere's center.
    lower = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(lower, radius=0.2, cfg=cfg)
    builder.add_shape_sphere(lower, xform=wp.transform((0.0, 0.3, -0.3), wp.quat_identity()), radius=0.1, cfg=cfg)
    upper = builder.add_body(xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()))
    builder.add_shape_sphere(upper, radius=0.1, cfg=cfg)
    builder.add_shape_sphere(upper, xform=wp.transform((0.0, -0.2, 0.2), wp.quat_identity()), radius=0.05, cfg=cfg)
    builder.color()
    dt = 1.0 / 60.0

    with wp.ScopedDevice(device):
        model = builder.finalize()

        def touching_state(speed):
            """Move the upper body into contact after the solver was built, tilted and moving."""
            state = model.state()
            q = state.body_q.numpy()
            q[upper] = [0.05, 0.0, 0.29, *wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.4)]
            state.body_q.assign(q)
            qd = state.body_qd.numpy()
            qd[upper] = [speed, 0.0, 0.0, 0.0, 3.0, 0.0]
            state.body_qd.assign(qd)
            return state

        def check(solver, contacts, history):
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreater(count, 0)
            stored = _stored_contact_step_rows(solver, count)
            np.testing.assert_allclose(
                stored, _expected_contact_step_rows(model, contacts, history), rtol=1.0e-5, atol=2.0e-6
            )
            test.assertGreater(np.abs(stored[:, :12]).max(), 0.05)

        # First step: the pose history is rebaselined to the moved pose, not the pose at construction.
        solver = newton.solvers.SolverVBD(model, iterations=4)
        small = newton.CollisionPipeline(model, rigid_contact_max=4)
        contacts = small.contacts()
        state, out = touching_state(0.3), model.state()
        history = state.body_q.numpy().copy()
        small.collide(state, contacts)
        solver.step(state, out, None, contacts, dt)
        check(solver, contacts, history)

        # Reporting takes its own poses and history: overwriting the stored rows must not change it.
        history_array = wp.array(history, dtype=wp.transform)
        force = solver.collect_rigid_contact_forces(out.body_q, history_array, contacts, dt)[4].numpy().copy()
        test.assertGreater(np.abs(force).max(), 1.0)
        for name in ("lever_s0", "lever_c0", "lever_s1", "lever_c1"):
            getattr(solver, "body_body_contact_" + name).fill_(wp.vec3(1.0e3, 1.0e3, 1.0e3))
        solver.body_body_contact_start_penetration.fill_(1.0e3)
        np.testing.assert_array_equal(
            solver.collect_rigid_contact_forces(out.body_q, history_array, contacts, dt)[4].numpy(), force
        )

        # A larger contact buffer on the next step: storage grows with the other row state and is rebuilt.
        large = newton.CollisionPipeline(model, rigid_contact_max=16)
        large_contacts = large.contacts()
        test.assertGreater(large_contacts.rigid_contact_max, contacts.rigid_contact_max)
        history = out.body_q.numpy().copy()
        large.collide(out, large_contacts)
        solver.step(out, state, None, large_contacts, dt)
        test.assertEqual(solver.body_body_contact_start_penetration.shape[0], large_contacts.rigid_contact_max)
        check(solver, large_contacts, history)

        # Detection inside the iterations replaces the rows: the stored geometry belongs to the last refresh.
        owned = newton.solvers.SolverVBD(
            model,
            iterations=4,
            collision_pipeline=newton.CollisionPipeline(model, contact_matching="latest"),
            collision_frequency={Slot.RIGID: 1},
            collision_frequency_type={Slot.RIGID: Frequency.ITERATIONS},
        )
        state, out = touching_state(2.0), model.state()
        history = state.body_q.numpy().copy()
        large.collide(state, large_contacts)
        before_refresh = _expected_contact_step_rows(model, large_contacts, history)
        owned.step(state, out, None, None, dt)
        check(owned, owned.contacts, history)
        after_refresh = _expected_contact_step_rows(model, owned.contacts, history)
        # The comparison above is only meaningful if the refresh changed the rows.
        test.assertTrue(
            after_refresh.shape != before_refresh.shape or np.abs(after_refresh - before_refresh).max() > 1.0e-3
        )


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


@wp.kernel
def _body_step_translations(
    q: wp.array[wp.transform],
    q_prev: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    translation: wp.array[wp.vec3],
):
    """Evaluate the production center-of-mass step translation."""
    i = wp.tid()
    step, _rotation, _jacobian_t = _body_step_motion(i, q, q_prev, com)
    translation[i] = step


def test_step_start_contact_geometry_is_translation_invariant(test, device):
    """Keep a row's step translation, levers and start gap at any distance from the world origin."""
    distances = (0.0, 1.0, 100.0)
    count = len(distances)
    # 1. Each body turns by 1e-4 rad about its own origin, which stays in place, so its center of mass, 30 mm
    # from that origin, moves 3 um. At 100 m this is less than half the float32 spacing (7.6 um): a difference
    # of two world-space center-of-mass positions rounds the step to zero.
    offset, angle = 0.03, 1.0e-4
    turned = np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)], dtype=np.float32)
    # First-order motion of (0, offset, 0) under the encoded rotation about z, in float64.
    expected_step = np.array(
        [-2.0 * float(turned[2]) * float(turned[3]) * offset, -2.0 * float(turned[2]) ** 2 * offset, 0.0]
    )
    # 2. Two bodies 0.25 m apart along x, each with an offset center of mass, touch across a 0.1 mm gap. Their
    # levers and their gap depend on the relative pose only; world points at 100 m round both by micrometers.
    com = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    point0 = np.array([0.1249, 0.001, 0.0], dtype=np.float32)
    point1 = np.array([-0.125, 0.001, 0.0], dtype=np.float32)
    expected_gap = 0.25 + float(point1[0]) - float(point0[0])
    expected_levers = (point0.astype(np.float64) - com, point1.astype(np.float64) - com)
    with wp.ScopedDevice(device):
        q = wp.array([wp.transform((d, 0.0, 0.0), wp.quat(*turned)) for d in distances], dtype=wp.transform)
        q_prev = wp.array([wp.transform((d, 0.0, 0.0), wp.quat_identity()) for d in distances], dtype=wp.transform)
        translation = wp.zeros(count, dtype=wp.vec3)
        wp.launch(
            _body_step_translations,
            count,
            inputs=[q, q_prev, wp.array([(0.0, offset, 0.0)] * count, dtype=wp.vec3)],
            outputs=[translation],
        )
        np.testing.assert_allclose(translation.numpy(), np.tile(expected_step, (count, 1)), rtol=0.0, atol=5.0e-8)

        pair_q = wp.array(
            [wp.transform((d + shift, 0.0, 0.0), wp.quat_identity()) for d in distances for shift in (0.0, 0.25)],
            dtype=wp.transform,
        )
        levers = [wp.zeros(count, dtype=wp.vec3) for _ in range(4)] + [wp.zeros(count, dtype=float)]
        zeros = wp.zeros(count, dtype=wp.vec3)
        wp.launch(
            build_body_body_contact_step_levers,
            count,
            inputs=[
                wp.array([count], dtype=int),
                wp.array(np.arange(count) * 2, dtype=int),
                wp.array(np.arange(count) * 2 + 1, dtype=int),
                wp.array([point0] * count, dtype=wp.vec3),
                wp.array([point1] * count, dtype=wp.vec3),
                zeros,
                zeros,
                wp.array([(1.0, 0.0, 0.0)] * count, dtype=wp.vec3),
                wp.zeros(count, dtype=float),
                wp.zeros(count, dtype=float),
                wp.array(np.arange(2 * count), dtype=int),
                pair_q,
                wp.array([com] * (2 * count), dtype=wp.vec3),
            ],
            outputs=levers,
        )
        for lever, expected in zip((levers[0], levers[2]), expected_levers, strict=True):
            np.testing.assert_allclose(lever.numpy(), np.tile(expected, (count, 1)), rtol=0.0, atol=5.0e-8)
        np.testing.assert_allclose(levers[4].numpy(), -expected_gap, rtol=0.0, atol=5.0e-8)


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
    "test_contact_step_levers_follow_rows_and_history",
    test_contact_step_levers_follow_rows_and_history,
    devices=get_test_devices(),
)


add_function_test(
    TestRigidContactDisplacement,
    "test_step_start_contact_geometry_is_translation_invariant",
    test_step_start_contact_geometry_is_translation_invariant,
    devices=get_test_devices(),
)


add_function_test(
    TestRigidContactDisplacement,
    "test_rolling_cylinder_keeps_its_speed",
    test_rolling_cylinder_keeps_its_speed,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
