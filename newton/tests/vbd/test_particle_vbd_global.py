# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the particle-global tangent and local/global stepping on real devices."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import particle_vbd_global as pvg
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _tet_forces_host(
    q: np.ndarray, previous: np.ndarray, inverse: np.ndarray, material: np.ndarray, dt: float
) -> np.ndarray:
    f = (q[..., 1:, :] - q[..., :1, :]).swapaxes(-1, -2) @ inverse
    old = (previous[..., 1:, :] - previous[..., :1, :]).swapaxes(-1, -2) @ inverse
    ft = f.swapaxes(-1, -2)
    cof = np.stack(
        [
            np.cross(ft[..., 1, :], ft[..., 2, :]),
            np.cross(ft[..., 2, :], ft[..., 0, :]),
            np.cross(ft[..., 0, :], ft[..., 1, :]),
        ],
        axis=-1,
    )
    mu, lam, damping = material.T
    lam = lam + mu
    alpha = 1 + mu / lam
    volume = 1 / (6 * np.linalg.det(inverse))
    rate = (ft @ f - old.swapaxes(-1, -2) @ old) / dt
    stress = mu[:, None, None] * f + (lam * (np.linalg.det(f) - alpha))[..., None, None] * cof
    stress += 2 * damping[:, None, None] * (f @ rate)
    other = (-volume[:, None, None] * (stress @ inverse.swapaxes(-1, -2))).swapaxes(-1, -2)
    return np.concatenate([-other.sum(axis=-2, keepdims=True), other], axis=-2)


def _test_particle_global_exact_damping_tangent(test, device):
    """Verify that the full tangent matches central finite differences for both zero and nonzero strain rate."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for offset in (-0.03, 0.03):
            builder.add_soft_grid(
                pos=wp.vec3(offset, 0.0, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
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
        model.soft_contact_mu = 0.0
        model.soft_contact_kd = 0.0
        rest = model.particle_q.numpy().astype(float)
        flags = model.particle_flags.numpy()
        flags[rest[:, 2] > 0.04 - 1e-6] &= ~int(newton.ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)

        solver = newton.solvers.SolverVBD(
            model,
            iterations=3,
            particle_global_iterations=1,
        )
        solver._rebuild_particle_global_vbd(linear_iterations=16)
        backend = solver._particle_global_vbd
        test.assertIsNotNone(backend)
        test.assertEqual(backend._global_coarse.component_count, 2)

        active = ((flags & 1) != 0) & (model.particle_mass.numpy() > 0)
        ids = model.tet_indices.numpy()
        B = model.tet_poses.numpy().astype(float)
        mat = model.tet_materials.numpy().astype(float)
        mass = model.particle_mass.numpy().astype(float)
        centers = np.where(rest[:, 0, None] < 0, np.array([[-0.03, 0.0, 0.0]]), np.array([[0.03, 0.0, 0.0]]))
        deformation = np.array([[1.04, 0.15, 0.02], [0.0, 0.97, 0.04], [0.0, 0.0, 0.93]])
        q = ((rest - centers) @ deformation.T + centers).astype(np.float32).astype(float)
        inertia = (q + np.array([0.0001, -0.0002, 0.00015])).astype(np.float32).astype(float)
        h = 1.0 / 240.0
        solver.inertia.assign(inertia.astype(np.float32))
        backend._global_h = h
        backend._global_contacts = None
        state = model.state()
        state.particle_q.assign(q.astype(np.float32))

        for mode in ("zero_rate", "nonzero_rate"):
            old = q if mode == "zero_rate" else rest.astype(np.float32).astype(float)
            solver.particle_q_prev.assign(old.astype(np.float32))
            wp.launch(
                pvg.initialize,
                model.particle_count,
                [
                    state.particle_q,
                    solver.inertia,
                    model.particle_mass,
                    model.particle_flags,
                    1.0 / (h * h),
                    backend._global_rhs,
                    backend._global_diag,
                ],
                device=device,
            )
            wp.launch(
                pvg.assemble_tets,
                model.tet_count,
                [
                    state.particle_q,
                    solver.particle_q_prev,
                    model.tet_indices,
                    model.tet_poses,
                    model.tet_materials,
                    model.particle_flags,
                    model.particle_mass,
                    h,
                    backend._global_rhs,
                    backend._global_diag,
                    backend._global_F,
                    backend._global_C,
                    backend._global_E,
                ],
                device=device,
            )

            def residual(x, previous=old):
                force = mass[:, None] * (inertia - x) / (h * h)
                np.add.at(force, ids.ravel(), _tet_forces_host(x[ids], previous[ids], B, mat, h).reshape(-1, 3))
                force[~active] = 0
                return force

            rng = np.random.default_rng(421)
            for _ in range(3):
                u = rng.normal(size=q.shape)
                v = rng.normal(size=q.shape)
                u[~active] = 0
                v[~active] = 0
                uw = wp.array(u.astype(np.float32), dtype=wp.vec3, device=device)
                vw = wp.array(v.astype(np.float32), dtype=wp.vec3, device=device)
                au = wp.zeros_like(uw)
                av = wp.zeros_like(vw)
                backend._global_matvec(uw, au, au, 1.0, 0.0)
                backend._global_matvec(vw, av, av, 1.0, 0.0)
                an = au.numpy().astype(float)
                bn = av.numpy().astype(float)
                fd_best = min(
                    float(
                        np.linalg.norm(an - (-(residual(q + eps * u) - residual(q - eps * u)) / (2 * eps)))
                        / np.linalg.norm(-(residual(q + eps * u) - residual(q - eps * u)) / (2 * eps))
                    )
                    for eps in (1e-7, 3e-8, 1e-8)
                )
                sym = float(abs(np.sum(u * bn) - np.sum(v * an)) / max(np.linalg.norm(u) * np.linalg.norm(bn), 1e-30))
                test.assertLess(fd_best, 5.0e-6)
                test.assertLess(sym, 2.0e-6)


def _test_particle_global_runs_soft_cube_compression(test, device):
    """Check finite local/global steps, accepted-stage counts and warmed-up capture."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        cfg = builder.default_shape_cfg.copy()
        cfg.kd = 0.0
        cfg.mu = 0.0
        builder.add_ground_plane(cfg=cfg)
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.002),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=3,
            dim_y=3,
            dim_z=2,
            cell_x=0.015,
            cell_y=0.015,
            cell_z=0.015,
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
        model.soft_contact_mu = 0.0
        model.soft_contact_kd = 0.0

        def run_sim(g_iters):
            pipe = newton.CollisionPipeline(model, soft_contact_gap=0.002, enable_rigid_soft_full_surface_contact=True)
            contacts = pipe.contacts()
            solver = newton.solvers.SolverVBD(
                model,
                iterations=3,
                particle_global_iterations=g_iters,
            )
            if g_iters:
                solver._rebuild_particle_global_vbd(linear_iterations=24)
            s_in, s_out, ctl = model.state(), model.state(), model.control()
            dt = 1.0 / 240.0
            pipe.collide(s_in, contacts)
            solver.step(s_in, s_out, ctl, contacts, dt)
            s_in, s_out = s_out, s_in
            if device.is_cuda:
                pipe.collide(s_in, contacts)
                with wp.ScopedCapture(device=device) as cap:
                    solver.step(s_in, s_out, ctl, contacts, dt)
                wp.capture_launch(cap.graph)
                s_in, s_out = s_out, s_in
            else:
                pipe.collide(s_in, contacts)
                solver.step(s_in, s_out, ctl, contacts, dt)
                s_in, s_out = s_out, s_in
            q = s_in.particle_q.numpy().astype(float)
            test.assertTrue(np.isfinite(q).all())
            return solver

        run_sim(0)
        solver_g = run_sim(1)
        stats = solver_g._particle_global_vbd.global_stats.numpy()
        test.assertEqual(int(stats[0]), 2)
        test.assertEqual(int(stats[1]), 2)


class TestVBDParticleGlobal(unittest.TestCase):
    pass


add_function_test(
    TestVBDParticleGlobal,
    "test_particle_global_exact_damping_tangent",
    _test_particle_global_exact_damping_tangent,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDParticleGlobal,
    "test_particle_global_runs_soft_cube_compression",
    _test_particle_global_runs_soft_cube_compression,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
