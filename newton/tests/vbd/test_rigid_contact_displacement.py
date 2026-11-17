# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid contact point motion: small slips survive world translation, and rolling creates no energy."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_vbd_kernels import _body_rotation_increment
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
        lever_torque = np.cross([radius, 0.0, 0.0], expected_force)
        expected_torque = float(q[3]) * lever_torque + np.cross(q[:3].astype(np.float64), lever_torque)
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
    "test_rolling_cylinder_keeps_its_speed",
    test_rolling_cylinder_keeps_its_speed,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
