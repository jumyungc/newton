# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tangent conditioning changes the numerical metric, not the Coulomb law."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_vbd_kernels import update_duals_body_body_contacts
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _make_contact_fixture(device):
    """Static, kinematic, different-color, and same-color partners, in that order."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(ke=2.0**10, kd=0.0, mu=0.5, density=0.0)
    shapes = []
    for i in range(7):
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True, is_kinematic=i == 1)
        shapes.append(builder.add_shape_sphere(body, radius=0.01, cfg=cfg))
    ground = builder.add_ground_plane(cfg=cfg)
    builder.color()
    builder.body_color_groups = [np.array([0, 1, 2, 3, 5, 6], dtype=np.int32), np.array([4], dtype=np.int32)]
    model = builder.finalize(device=device)
    contacts = newton.Contacts(4, 0, device=device, rigid_contact_surface_velocity=True)
    contacts.rigid_contact_count.fill_(4)
    contacts.rigid_contact_shape0.assign([ground, shapes[1], shapes[3], shapes[5]])
    contacts.rigid_contact_shape1.assign([shapes[0], shapes[2], shapes[4], shapes[6]])
    contacts.rigid_contact_point1.assign([(0.0, 0.0, -(2.0**-10))] * 4)
    contacts.rigid_contact_normal.assign([(0.0, 0.0, 1.0)] * 4)
    contacts.contact_generation.fill_(1)
    return model, contacts


def _conditioned_solver(model, contacts, dt, **kwargs):
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_contact_history=False, **kwargs)
    # Use the owning prologue without moving the prescribed contact geometry.
    solver._refresh_rigid_contact_state(contacts, True)
    solver._step_body_body_contact_frame(contacts, model.body_q, dt, 1.0, 1.0)
    return solver


def test_tangent_scale_eligibility(test, device):
    """Scale actual automatic rho for every supported body-pair configuration."""
    with wp.ScopedDevice(device):
        model, contacts = _make_contact_fixture(device)
        dt = 2.0**-8
        default = _conditioned_solver(model, contacts, dt)
        unit = _conditioned_solver(model, contacts, dt, rigid_contact_tangent_scale=1.0)
        scaled = _conditioned_solver(model, contacts, dt, rigid_contact_tangent_scale=300.0)
        base_rho = default.body_body_contact_tangent_rho.numpy()[:4]
        test.assertTrue(np.all(base_rho > 0.0))
        test.assertEqual(float(default.body_inv_mass_effective.numpy()[1]), 0.0)
        colors = model.body_colors.numpy()
        test.assertNotEqual(colors[3], colors[4])
        test.assertEqual(colors[5], colors[6])
        np.testing.assert_array_equal(unit.body_body_contact_tangent_rho.numpy()[:4], base_rho)
        np.testing.assert_allclose(
            scaled.body_body_contact_tangent_rho.numpy()[:4] / base_rho,
            [300.0, 300.0, 300.0, 300.0],
            rtol=2.0e-6,
        )
        for name in ("normal_rho", "material_ke", "material_kd", "material_mu"):
            np.testing.assert_array_equal(
                getattr(scaled, "body_body_contact_" + name).numpy()[:4],
                getattr(default, "body_body_contact_" + name).numpy()[:4],
            )


def test_tangent_scale_preserves_loaded_coulomb_state(test, device):
    """Preserve loaded stick and the physical cap under belt-cancelled motion."""
    dt = 2.0**-8
    displacement = 2.0**-12
    normal_force = (2.0**10) * (2.0**-10)
    cap = 0.5 * normal_force
    with wp.ScopedDevice(device):
        model, contacts = _make_contact_fixture(device)
        previous = model.body_q.numpy()
        current = previous.copy()
        current[[0, 2, 4, 6], 0] += displacement
        state = model.state()
        state.body_q.assign(current)
        # Shape1 translates +x. The authored surface velocity cancels its
        # relative displacement exactly, using binary values in float32.
        contacts.rigid_contact_surface_velocity.assign([(-displacement / dt, 0.0, 0.0)] * 4)
        wrench = wp.zeros(4, dtype=wp.spatial_vector)
        rho_by_scale = []
        for scale in (1.0, 300.0):
            solver = _conditioned_solver(model, contacts, dt, rigid_contact_tangent_scale=scale)
            solver.body_q_prev.assign(previous)
            solver.body_body_contact_C0.zero_()
            rho_by_scale.append(solver.body_body_contact_tangent_rho.numpy()[:4])
            for tangent_seed in (0.125, 1.0):
                with test.subTest(scale=scale, tangent_seed=tangent_seed):
                    initial = np.zeros((*solver.body_body_contact_lambda.shape, 3), dtype=np.float32)
                    initial[:4] = [tangent_seed, 0.0, normal_force]
                    solver.body_body_contact_lambda.assign(initial)
                    expected = np.tile([min(tangent_seed, cap), 0.0, normal_force], (4, 1))
                    values = solver.collect_rigid_contact_forces(state.body_q, solver.body_q_prev, contacts, dt)
                    solver._export_body_body_contact_forces(state, contacts, dt, wrench)
                    test.assertEqual(int(values[5].numpy()[0]), 4)
                    np.testing.assert_allclose(values[4].numpy()[:4], expected, rtol=2.0e-6, atol=2.0e-6)
                    # CONTACT_F reports the opposite force, on shape0. Its
                    # point and COM are coincident for every row in this fixture.
                    np.testing.assert_allclose(wrench.numpy()[:, :3], -expected, rtol=2.0e-6, atol=2.0e-6)
                    np.testing.assert_allclose(wrench.numpy()[:, 3:], 0.0, atol=2.0e-6)
                    wp.launch(
                        update_duals_body_body_contacts,
                        dim=4,
                        inputs=[
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_surface_velocity,
                            contacts.rigid_contact_offset0,
                            contacts.rigid_contact_offset1,
                            contacts.rigid_contact_normal,
                            contacts.rigid_contact_margin0,
                            contacts.rigid_contact_margin1,
                            model.shape_body,
                            state.body_q,
                            solver.body_q_prev,
                            model.body_com,
                            dt,
                            solver.body_body_contact_material_mu,
                            solver.body_body_contact_C0,
                            solver.rigid_contact_alpha,
                            solver.rigid_contact_hard,
                            solver.rigid_compliant_alm,
                            solver.body_body_contact_material_ke,
                            solver.body_body_contact_material_kd,
                            solver.body_body_contact_tangent_rho,
                            solver.body_body_contact_normal_rho,
                            solver.rigid_linear_beta,
                            solver.rigid_contact_damping_bidirectional,
                            solver.body_body_contact_penalty_k,
                            solver.body_body_contact_lambda,
                        ],
                    )
                    np.testing.assert_allclose(
                        solver.body_body_contact_lambda.numpy()[:4], expected, rtol=2.0e-6, atol=2.0e-6
                    )
        # Every force comparison must exercise an actually scaled metric.
        np.testing.assert_allclose(rho_by_scale[1] / rho_by_scale[0], 300.0, rtol=2.0e-6)


def test_tangent_scale_validation(test, device):
    """Reject invalid conditioning scales and incompatible solver modes."""
    with wp.ScopedDevice(device):
        model, _ = _make_contact_fixture(device)
        for scale in (0.0, 0.5, -1.0, float("nan"), float("inf"), -float("inf")):
            with test.subTest(scale=scale), test.assertRaisesRegex(ValueError, "rigid_contact_tangent_scale"):
                newton.solvers.SolverVBD(model, rigid_contact_tangent_scale=scale)
        for mode in ({"rigid_compliant_alm": False}, {"integrate_with_external_rigid_solver": True}):
            with test.subTest(mode=mode), test.assertRaisesRegex(ValueError, "rigid_contact_tangent_scale"):
                newton.solvers.SolverVBD(model, rigid_contact_tangent_scale=300.0, **mode)
            # The neutral default remains valid for both existing solver modes.
            newton.solvers.SolverVBD(model, rigid_contact_tangent_scale=1.0, **mode)


class TestRigidContactTangentScale(unittest.TestCase):
    pass


for name, function in (
    ("test_tangent_scale_eligibility", test_tangent_scale_eligibility),
    ("test_tangent_scale_preserves_loaded_coulomb_state", test_tangent_scale_preserves_loaded_coulomb_state),
):
    add_function_test(TestRigidContactTangentScale, name, function, devices=get_test_devices())
add_function_test(
    TestRigidContactTangentScale,
    "test_tangent_scale_validation",
    test_tangent_scale_validation,
    devices=[device for device in get_test_devices() if device.is_cpu],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
