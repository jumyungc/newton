# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the experimental particle-global domain and first-step capture semantics."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import particle_vbd_global as pvg
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _model(device, *, mesh=False, radius=0.001, height=None):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.default_particle_radius = radius
    cfg = builder.default_shape_cfg.copy()
    cfg.mu = cfg.kd = 0.0
    if mesh:
        builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.2, 0.2, 0.01), cfg=cfg)
    else:
        builder.add_ground_plane(cfg=cfg)
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, (0.0105 if mesh else 0.0005) if height is None else height),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.02,
        cell_y=0.02,
        cell_z=0.02,
        density=1000.0,
        k_mu=5.0e4,
        k_lambda=1.0e6,
        k_damp=2.0,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
    )
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_mu = model.soft_contact_kd = 0.0
    return model


def test_particle_global_rejects_mesh_features_before_mutation(test, device):
    """Fail closed for unimplemented current-geometry mesh force selection; keep G0 available."""
    model = _model(device, mesh=True)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts()
    test.assertIsNotNone(contacts._soft_contact_mesh_features)
    state, output = model.state(), model.state()
    pipeline.collide(state, contacts)
    solver = newton.solvers.SolverVBD(model, iterations=3, particle_global_iterations=1)
    before_q, before_v = state.particle_q.numpy(), state.particle_qd.numpy()
    with test.assertRaisesRegex(NotImplementedError, "mesh.*contact"):
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
    np.testing.assert_array_equal(state.particle_q.numpy(), before_q)
    np.testing.assert_array_equal(state.particle_qd.numpy(), before_v)
    local = newton.solvers.SolverVBD(model, iterations=3)
    local.step(state, output, model.control(), contacts, 1.0 / 240.0)
    test.assertTrue(np.isfinite(output.particle_q.numpy()).all())


def test_particle_global_validates_ratio_threshold(test, device):
    """Reject nonfinite or out-of-range heuristic thresholds before stepping."""
    model = _model(device)
    solver = newton.solvers.SolverVBD(model, iterations=3, particle_global_iterations=1)
    for value in (float("nan"), float("inf"), -0.1, 1.1):
        with test.subTest(value=value), test.assertRaisesRegex(ValueError, "rho_min"):
            solver._rebuild_particle_global_vbd(rho_min=value)


def test_particle_global_first_step_capture(test, device):
    """Record a fresh solver's first step and compare with a distinct eager solver."""
    if not device.is_cuda or not wp.is_conditional_graph_supported():
        test.skipTest("Requires CUDA conditional graph support")
    model = _model(device)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
    results = []
    for capture_step in (False, True):
        solver = newton.solvers.SolverVBD(model, iterations=3, particle_global_iterations=1)
        state, output = model.state(), model.state()
        contacts = pipeline.contacts()
        control = model.control()
        pipeline.collide(state, contacts)
        if capture_step:
            with wp.ScopedCapture(device=device) as captured:
                with test.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
                solver.step(state, output, control, contacts, 1.0 / 240.0)
            wp.capture_launch(captured.graph)
        else:
            solver.step(state, output, control, contacts, 1.0 / 240.0)
        test.assertEqual(int(solver._particle_global_vbd.global_stats.numpy()[0]), 1)
        results.append(output.particle_q.numpy())
    np.testing.assert_allclose(results[0], results[1], atol=1.0e-7, rtol=1.0e-6)


def test_particle_global_schedule_and_refresh(test, device):
    """Preserve actual stage counts and live material refresh without replacing captured buffers."""
    model = _model(device, height=0.05)
    solver = newton.solvers.SolverVBD(model, iterations=4, particle_global_iterations=2)
    test.assertEqual(solver._particle_global_iteration_indices, frozenset((0, 2)))
    test.assertIsInstance(solver._particle_global_iteration_indices, frozenset)
    backend = solver._particle_global_vbd
    buffers = (backend._global_rhs.ptr, backend._global_coarse.phi.ptr)
    initial = model.particle_q.numpy()
    center = initial.mean(axis=0)
    initial = (initial - center) * np.array([1.05, 1.0, 0.95], dtype=np.float32) + center
    state, output = model.state(), model.state()
    state.particle_q.assign(initial)
    captured = None
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as captured:
            solver.step(state, output, model.control(), None, 1.0 / 240.0)
    material = model.tet_materials.numpy()
    material[:, :2] *= 1.25
    model.tet_materials.assign(material)
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES | newton.ModelFlags.SHAPE_PROPERTIES)
    test.assertIs(solver._particle_global_vbd, backend)
    test.assertEqual((backend._global_rhs.ptr, backend._global_coarse.phi.ptr), buffers)
    fresh = newton.solvers.SolverVBD(model, iterations=4, particle_global_iterations=2)
    results = []
    for candidate in (solver, fresh):
        if candidate is solver and captured is not None:
            wp.capture_launch(captured.graph)
        else:
            state, output = model.state(), model.state()
            state.particle_q.assign(initial)
            candidate.step(state, output, model.control(), None, 1.0 / 240.0)
        test.assertEqual(int(candidate._particle_global_vbd.global_stats.numpy()[0]), 2)
        results.append(output.particle_q.numpy())
    np.testing.assert_allclose(results[0], results[1], atol=1e-8, rtol=1e-6)
    model.soft_contact_mu = 0.2
    with test.assertRaisesRegex(ValueError, "frictionless"):
        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    model.soft_contact_mu = 0.0
    model.shape_material_kd.fill_(1.0)
    with test.assertRaisesRegex(ValueError, "undamped"):
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    model.shape_material_kd.zero_()
    material[:, 2] = -1.0
    model.tet_materials.assign(material)
    with test.assertRaisesRegex(ValueError, "constitutive"):
        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)


class TestParticleGlobalScope(unittest.TestCase):
    pass


@wp.kernel
def _merit_samples(
    deformation: wp.array[wp.mat33d],
    mu: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    energy: wp.array[wp.float64],
):
    i = wp.tid()
    f = deformation[i]
    energy[i] = pvg.tet_energy(
        wp.vec3d(0.0),
        f[:, 0],
        f[:, 1],
        f[:, 2],
        wp.identity(3, wp.float64),
        wp.identity(3, wp.float64),
        mu[i],
        lam[i],
        wp.float64(0.0),
        wp.float64(1.0),
    )


def test_particle_global_material_merit(test, device):
    """The expanded objective preserves nonzero-material differences and is finite at zero elasticity."""
    rng = np.random.default_rng(5721)
    fs = np.eye(3) + 0.15 * rng.normal(size=(12, 3, 3))
    mus = np.repeat([0.0, 5.0e4, 5.0e4], 4)
    lams = np.repeat([0.0, 1.05e6, 1.05e8], 4)
    energy = wp.zeros(len(fs), dtype=wp.float64, device=device)
    wp.launch(
        _merit_samples,
        len(fs),
        [
            wp.array(fs, dtype=wp.mat33d, device=device),
            wp.array(mus, dtype=wp.float64, device=device),
            wp.array(lams, dtype=wp.float64, device=device),
            energy,
        ],
        device=device,
    )
    actual = energy.numpy()
    np.testing.assert_array_equal(actual[:4], 0.0)
    shifted = (
        0.5 * mus[4:] * np.sum(fs[4:] ** 2, axis=(1, 2))
        + 0.5 * lams[4:] * (np.linalg.det(fs[4:]) - 1.0 - mus[4:] / lams[4:]) ** 2
    ) / 6.0
    np.testing.assert_allclose(actual[5:8] - actual[4], shifted[1:4] - shifted[0], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(actual[9:12] - actual[8], shifted[5:8] - shifted[4], rtol=1e-12, atol=1e-9)
    model = _model(device)
    material = model.tet_materials.numpy()
    material[:] = 0.0
    model.tet_materials.assign(material)
    solver = newton.solvers.SolverVBD(model, iterations=3, particle_global_iterations=1)
    state, output = model.state(), model.state()
    solver.step(state, output, model.control(), None, 1.0 / 240.0)
    test.assertTrue(np.isfinite(solver._particle_global_vbd._global_energy.numpy()).all())
    expected = model.particle_q.numpy().copy()
    expected[:, 2] -= 9.81 / 240.0**2
    np.testing.assert_allclose(output.particle_q.numpy(), expected, atol=1e-8, rtol=1e-6)
    for values in ((1e-8, 1e-8, 0.0), (-1.0, 1.0, 0.0), (1.0, 1.0, -1.0), (float("nan"), 1.0, 0.0)):
        material[:] = values
        model.tet_materials.assign(material)
        with test.subTest(material=values), test.assertRaisesRegex(ValueError, "constitutive"):
            newton.solvers.SolverVBD(model, iterations=3, particle_global_iterations=1)


def test_particle_global_coarse_component_isolation(test, device):
    """An indefinite component cannot disable another component's SPD coarse correction."""
    matrices = np.array([np.diag([1.0, 2.0, 3.0]), np.diag([-1.0, 2.0, 3.0])])
    for order in ([0, 1], [1, 0]):
        ac = wp.array(matrices[order], dtype=wp.float64, device=device)
        chol, inverse = wp.zeros_like(ac), wp.zeros_like(ac)
        valid = wp.zeros(2, dtype=int, device=device)
        stats = wp.zeros(2, dtype=int, device=device)
        ranks = wp.array([1, 1], dtype=int, device=device)
        wp.launch(pvg.factor_coarse, 2, [ac, chol, valid, stats, ranks], device=device)
        wp.launch(pvg.invert_coarse, (2, 3), [chol, inverse, valid, ranks], device=device)
        np.testing.assert_array_equal(valid.numpy(), np.array(order) == 0)
        np.testing.assert_array_equal(stats.numpy(), [2, 1])
        np.testing.assert_allclose(inverse.numpy()[order.index(0)], np.diag([1.0, 0.5, 1.0 / 3.0]), atol=1e-14)
        np.testing.assert_array_equal(inverse.numpy()[order.index(1)], 0.0)


def test_particle_global_preconditioner_scale(test, device):
    """Recover representable SPD inverses and a manufactured global correction at extreme scale."""
    rng = np.random.default_rng(20261022)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    scales = (1e-20, 1e-15, 1e-12, 1e-6, 1.0, 1e6, 1e12, 1e15, 1e20)
    matrices = np.asarray(
        [
            scale * (rotation @ np.diag(eigenvalues) @ rotation.T)
            for eigenvalues in ((1.0, 1.0, 1.0), (1.0, 2.0, 5.0), (1.0, 10.0, 100.0))
            for scale in scales
        ],
        dtype=np.float32,
    )
    blocks = wp.array(matrices, dtype=wp.mat33, device=device)
    inverse = wp.zeros_like(blocks)
    wp.launch(pvg.invert_blocks, len(matrices), [blocks, inverse], device=device)
    actual = inverse.numpy().astype(np.float64)
    test.assertTrue(np.isfinite(actual).all())
    np.testing.assert_allclose(
        matrices.astype(np.float64) @ actual,
        np.broadcast_to(np.eye(3), matrices.shape),
        rtol=0.0,
        atol=1e-5,
    )

    # Zero elasticity gives an independent exact inertial target. Non-affine
    # displacements prevent the affine coarse space from hiding a broken block term.
    model = _model(device, height=0.1)
    model.tet_materials.zero_()
    model.particle_mass.fill_(1e10)
    model.particle_inv_mass.fill_(1e-10)
    solver = newton.solvers.SolverVBD(model, iterations=3, particle_global_iterations=1)
    q = model.particle_q.numpy()
    target = q + np.asarray(rng.normal(size=q.shape) * 0.001, dtype=np.float32)
    solver.inertia.assign(target)
    solver.particle_q_prev.assign(q)
    solver.pos_prev_collision_detection.assign(q)
    state = model.state()
    backend = solver._particle_global_vbd
    captured = None
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as captured:
            backend.solve(state, None, 0.01)
        wp.capture_launch(captured.graph)
    else:
        backend.solve(state, None, 0.01)
    np.testing.assert_allclose(state.particle_q.numpy(), target, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(backend.global_stats.numpy(), [1, 1])


for function in (
    test_particle_global_rejects_mesh_features_before_mutation,
    test_particle_global_validates_ratio_threshold,
    test_particle_global_first_step_capture,
    test_particle_global_schedule_and_refresh,
    test_particle_global_material_merit,
    test_particle_global_coarse_component_isolation,
    test_particle_global_preconditioner_scale,
):
    add_function_test(TestParticleGlobalScope, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
