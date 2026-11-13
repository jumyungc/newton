# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A frozen contact's small friction force must not depend on world translation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_small_rotation_contact_force_is_translation_invariant(test, device):
    """Preserve a 2^-28 m tangential slip through both force-reporting paths."""
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
        solver = newton.solvers.SolverVBD(model, iterations=4, rigid_compliant_alm=True)
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


class TestRigidContactDisplacement(unittest.TestCase):
    pass


add_function_test(
    TestRigidContactDisplacement,
    "test_small_rotation_contact_force_is_translation_invariant",
    test_small_rotation_contact_force_is_translation_invariant,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
