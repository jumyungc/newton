from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import warp as wp

from bench.global_cable import codex_bgn_contact_kkt as contact_v1
from bench.global_cable import codex_bgn_global as bgn
from bench.global_cable import codex_ccdsn as ccdsn
from bench.global_cable import codex_rc_forest_newton as rc
from bench.global_cable import scenes


DT = 1.0 / 600.0


def contact_rows(J: np.ndarray, b0: np.ndarray, mu: np.ndarray) -> contact_v1.ContactRows:
    count = len(mu)
    empty = np.zeros((count, 3), dtype=np.float64)
    return contact_v1.ContactRows(
        J=sparse.csr_matrix(np.asarray(J, dtype=np.float64)),
        b0=np.asarray(b0, dtype=np.float64),
        gap=np.asarray(b0, dtype=np.float64)[0::3],
        mu=np.asarray(mu, dtype=np.float64),
        shape0=np.arange(count),
        shape1=-np.ones(count, dtype=np.int64),
        body0=np.arange(count),
        body1=-np.ones(count, dtype=np.int64),
        normal=empty,
        tangent1=empty,
        tangent2=empty,
        world0=empty,
        world1=empty,
        anchor0=empty,
        anchor1=empty,
        reported_count=count,
        capacity=count,
    )


class TestCCDSNCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ccdsn.owner_guard("tests")
        wp.init()

    def test_finite_stretch_kkt_equals_eliminated_penalty(self) -> None:
        H = sparse.diags([2.0, 3.0, 4.0, 5.0], format="csc")
        A = sparse.csr_matrix([
            [-1.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
        ])
        compliance = np.asarray([0.1, 0.2, 0.05])
        offset = np.asarray([0.03, -0.02, 0.01])
        rhs = np.asarray([0.2, -0.1, 0.05, 0.3])
        inverse = ccdsn.ChainKKTInverse(H, A, compliance)
        direction, force = inverse.free(rhs, offset)
        stiffness = sparse.diags(1.0 / compliance)
        eliminated_H = H + A.T @ stiffness @ A
        eliminated_rhs = rhs - A.T @ ((1.0 / compliance) * offset)
        expected = splu(eliminated_H.tocsc()).solve(eliminated_rhs)
        self.assertTrue(np.allclose(direction, expected, atol=2.0e-13))
        self.assertTrue(np.allclose(force, (offset + A @ direction) / compliance, atol=2.0e-13))

    def test_hard_stretch_and_frictionless_contact_match_monolithic_kkt(self) -> None:
        H = sparse.diags([2.0, 3.0, 4.0], format="csc")
        A = sparse.csr_matrix([[1.0, -1.0, 0.0]])
        J = sparse.csr_matrix([[0.0, 0.0, 1.0]])
        rhs = np.zeros(3)
        stretch_offset = np.asarray([0.1])
        contact_offset = np.asarray([-0.2])
        contact_compliance = 1.0e-3
        result = ccdsn.solve_coupled_quadratic(
            H,
            rhs,
            A,
            stretch_offset,
            np.zeros(1),
            J,
            contact_offset,
            contact_mode=ccdsn.FRICTIONLESS,
            contact_compliance=contact_compliance,
            dt=DT,
        )
        K = np.block([
            [H.toarray(), A.T.toarray(), -J.T.toarray()],
            [A.toarray(), np.zeros((1, 1)), np.zeros((1, 1))],
            [J.toarray(), np.zeros((1, 1)), np.asarray([[contact_compliance]])],
        ])
        expected = np.linalg.solve(
            K, np.concatenate((rhs, -stretch_offset, -contact_offset))
        )
        self.assertTrue(np.allclose(result.direction, expected[:3], atol=2.0e-12))
        self.assertTrue(np.allclose(result.stretch_force_N, expected[3:4], atol=2.0e-12))
        self.assertTrue(np.allclose(result.contact_force_N, expected[4:], atol=2.0e-12))
        self.assertGreater(result.contact_force_N[0], 0.0)
        self.assertLessEqual(result.info["stationarity_inf"], 1.0e-11)
        self.assertLessEqual(result.info["stretch_residual_inf_m"], 1.0e-11)

    def test_frictionless_inactive_contact_stays_inactive(self) -> None:
        result = ccdsn.solve_coupled_quadratic(
            sparse.eye(2, format="csc"),
            np.zeros(2),
            sparse.csr_matrix((0, 2)),
            np.empty(0),
            np.empty(0),
            sparse.csr_matrix([[1.0, 0.0]]),
            np.asarray([0.2]),
            contact_mode=ccdsn.FRICTIONLESS,
            dt=DT,
        )
        self.assertEqual(result.info["active_contact_indices"], [])
        self.assertEqual(result.contact_force_N[0], 0.0)
        self.assertAlmostEqual(result.info["physical_contact_value"][0], 0.2)

    def test_high_mu_sticking_is_exact_and_inside_cone(self) -> None:
        H = sparse.diags([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], format="csc")
        A = sparse.csr_matrix([[1.0, 0.0, 0.0, -1.0, 0.0, 0.0]])
        J = sparse.csr_matrix(np.hstack((np.eye(3), np.zeros((3, 3)))))
        result = ccdsn.solve_coupled_quadratic(
            H,
            np.zeros(6),
            A,
            np.asarray([0.05]),
            np.zeros(1),
            J,
            np.asarray([-0.1, 0.02, -0.01]),
            contact_mode=ccdsn.STICKING,
            contact_compliance=1.0e-3,
            friction_mu=np.asarray([1.0e4]),
            dt=DT,
        )
        force = result.contact_force_N.reshape(-1, 3)[0]
        self.assertGreaterEqual(force[0], 0.0)
        self.assertLessEqual(np.linalg.norm(force[1:]), 1.0e4 * force[0])
        self.assertLessEqual(result.info["active_contact_residual_inf_m"], 1.0e-10)
        self.assertLessEqual(result.info["cone_violation_N"], 1.0e-12)

    def test_analytic_desaxce_sliding_over_hard_chain_inverse(self) -> None:
        # The hard chain constraint d0=d3 makes the constrained normal
        # mobility 1/2 while both tangential mobilities remain one.  The
        # analytic sliding law is therefore lambda=(2,-1,0), u=(0,1,0).
        H = sparse.eye(4, format="csc")
        A = sparse.csr_matrix([[1.0, 0.0, 0.0, -1.0]])
        J = sparse.csr_matrix(np.hstack((np.eye(3), np.zeros((3, 1)))))
        result = ccdsn.solve_coupled_quadratic(
            H,
            np.zeros(4),
            A,
            np.zeros(1),
            np.zeros(1),
            J,
            np.asarray([-1.0, 2.0, 0.0]),
            contact_mode=ccdsn.DESAXCE,
            contact_compliance=0.0,
            friction_mu=np.asarray([0.5]),
            dt=DT,
        )
        np.testing.assert_allclose(
            result.contact_force_N, [2.0, -1.0, 0.0], atol=3.0e-10
        )
        np.testing.assert_allclose(
            result.info["compliant_contact_value"], [0.0, 1.0, 0.0], atol=3.0e-10
        )
        self.assertEqual(result.info["friction_law"]["sliding_contact_indices"], [0])
        self.assertLessEqual(result.info["natural_map_relative_inf"], 2.0e-9)
        self.assertLessEqual(result.info["primal_cone_violation_N"], 2.0e-9)
        self.assertLessEqual(result.info["dual_cone_violation_m"], 2.0e-9)
        self.assertLessEqual(result.info["complementarity_inf_Nm"], 2.0e-9)
        self.assertLess(result.info["friction_work_Nm"], 0.0)

    def test_mixed_stick_slide_uses_same_constrained_chain_delassus(self) -> None:
        H = sparse.eye(8, format="csc")
        A = sparse.csr_matrix([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        ])
        J = np.zeros((6, 8), dtype=np.float64)
        J[:3, :3] = np.eye(3)
        J[3:, 3:6] = np.eye(3)
        J[3, 0] = 0.2  # nonzero contact-contact coupling through constrained S
        kwargs = dict(
            H=H,
            rhs=np.zeros(8),
            A=A,
            stretch_offset=np.zeros(2),
            stretch_compliance=np.zeros(2),
            J=sparse.csr_matrix(J),
            contact_offset=np.asarray([-1.0, 0.2, 0.0, -1.0, 2.0, 0.0]),
            contact_mode=ccdsn.DESAXCE,
            contact_compliance=0.0,
            friction_mu=np.asarray([1.0e4, 0.5]),
            dt=DT,
        )
        result = ccdsn.solve_coupled_quadratic(**kwargs)
        law = result.info["friction_law"]
        self.assertEqual(law["sticking_contact_indices"], [0])
        self.assertEqual(law["moderate_friction_contact_indices"], [1])
        self.assertEqual(law["sliding_contact_indices"], [1])
        np.testing.assert_allclose(
            result.info["compliant_contact_value"][:3], 0.0, atol=2.0e-9
        )
        self.assertLessEqual(result.info["natural_map_relative_inf"], 2.0e-9)
        self.assertLessEqual(result.info["primal_cone_violation_N"], 2.0e-9)
        self.assertLessEqual(result.info["dual_cone_violation_m"], 2.0e-9)
        self.assertLessEqual(result.info["complementarity_inf_Nm"], 2.0e-9)
        self.assertLess(result.info["friction_work_Nm"], 0.0)
        with self.assertRaises(ccdsn.ContactModeNotApplicable):
            ccdsn.solve_coupled_quadratic(**kwargs, exact_contact_limit=1)

    def test_low_mu_sticking_fails_closed(self) -> None:
        with self.assertRaises(ccdsn.ContactModeNotApplicable):
            ccdsn.solve_coupled_quadratic(
                sparse.eye(3, format="csc"),
                np.zeros(3),
                sparse.csr_matrix((0, 3)),
                np.empty(0),
                np.empty(0),
                sparse.eye(3, format="csr"),
                np.asarray([-0.1, 0.2, 0.0]),
                contact_mode=ccdsn.STICKING,
                friction_mu=np.asarray([0.5]),
                dt=DT,
            )

    def test_matrix_free_delassus_matches_dense(self) -> None:
        H = sparse.diags([2.0, 3.0, 4.0, 5.0], format="csc")
        A = sparse.csr_matrix([[-1.0, 1.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0]])
        inverse = ccdsn.ChainKKTInverse(H, A, np.zeros(2))
        J = sparse.csr_matrix([[1.0, 0.0, 0.2, 0.0], [0.0, -0.3, 0.0, 1.0]])
        compliance = np.asarray([1.0e-3, 2.0e-3])
        response, _ = inverse.response(J.T.toarray())
        dense = np.asarray(J @ response) + np.diag(compliance)
        operator = ccdsn.delassus_linear_operator(inverse, J, compliance)
        x = np.asarray([0.7, -0.2])
        self.assertTrue(np.allclose(operator @ x, dense @ x, atol=2.0e-13))
        self.assertTrue(np.allclose(dense, dense.T, atol=2.0e-13))
        self.assertGreater(np.linalg.eigvalsh(dense)[0], 0.0)

    def test_poisoned_factorization_does_not_mutate_inputs(self) -> None:
        H = sparse.eye(2, format="csc")
        A = sparse.csr_matrix([[1.0, -1.0]])
        rhs = np.asarray([0.2, -0.1])
        J = sparse.csr_matrix([[1.0, 0.0]])
        snapshots = (H.copy(), A.copy(), rhs.copy(), J.copy())
        with mock.patch.object(ccdsn, "splu", side_effect=RuntimeError("poison")):
            with self.assertRaises(ccdsn.CoupledSolveError):
                ccdsn.solve_coupled_quadratic(
                    H,
                    rhs,
                    A,
                    np.asarray([0.0]),
                    np.zeros(1),
                    J,
                    np.asarray([-0.1]),
                    contact_mode=ccdsn.FRICTIONLESS,
                    dt=DT,
                )
        self.assertEqual((H != snapshots[0]).nnz, 0)
        self.assertEqual((A != snapshots[1]).nnz, 0)
        self.assertTrue(np.array_equal(rhs, snapshots[2]))
        self.assertEqual((J != snapshots[3]).nnz, 0)


class TestCCDSNPreparedCable(unittest.TestCase):
    def make_scene(self, n: int = 5):
        return scenes._make_cantilever(
            n, f"ccdsn_test_{n}", f"CCDSN test {n}", bend_ke=500.0, num_frames=1
        )

    def test_prepared_finite_kkt_matches_original_bgn_direction(self) -> None:
        with wp.ScopedDevice("cpu"):
            scene = self.make_scene()
            plan = bgn.prepare(scene, DT)
            original, predicted = bgn._predict_pose_fast(plan.data, scene.state_0, DT)
            previous = bgn.raw_by_joint(plan, original)
            prepared = ccdsn.assemble_prepared_chain_linearization(
                plan, predicted, predicted, previous,
                stretch_mode=ccdsn.FINITE_STRETCH,
            )
            inverse = ccdsn.ChainKKTInverse(
                prepared.H_backbone, prepared.A, prepared.stretch_compliance
            )
            lifted, _ = inverse.free(
                prepared.rhs_backbone, prepared.stretch_offset
            )
            H_full = bgn.sparse_matrix(plan, prepared.full_system).tocsc()
            direct = splu(H_full).solve(prepared.full_system.rhs.reshape(-1))
            relative = np.max(np.abs(lifted - direct)) / max(
                1.0, float(np.max(np.abs(direct), initial=0.0))
            )
            self.assertLessEqual(relative, 2.0e-9)

    def test_prepared_finite_and_hard_modes_both_couple_contact(self) -> None:
        with wp.ScopedDevice("cpu"):
            scene = self.make_scene(4)
            plan = bgn.prepare(scene, DT)
            original, predicted = bgn._predict_pose_fast(plan.data, scene.state_0, DT)
            previous = bgn.raw_by_joint(plan, original)
            J = np.zeros((3, 6 * plan.n), dtype=np.float64)
            last = plan.n - 1
            J[0, 6 * last + 2] = 1.0
            J[1, 6 * last + 0] = 1.0
            J[2, 6 * last + 1] = 1.0
            rows = contact_rows(J, np.asarray([-1.0e-3, 0.0, 0.0]), np.asarray([1.0e4]))
            for mode in (ccdsn.FINITE_STRETCH, ccdsn.HARD_STRETCH):
                with self.subTest(stretch_mode=mode):
                    result = ccdsn.solve_prepared_linearization(
                        plan,
                        predicted,
                        predicted,
                        previous,
                        rows,
                        stretch_mode=mode,
                        contact_mode=ccdsn.FRICTIONLESS,
                        contact_compliance=1.0e-8,
                    )
                    self.assertEqual(result.info["active_contact_indices"], [0])
                    self.assertGreater(result.contact_force_N[0], 0.0)
                    self.assertLessEqual(result.info["stationarity_inf"], 1.0e-7)
                    self.assertLessEqual(result.info["stretch_residual_inf_m"], 1.0e-7)
                    self.assertLessEqual(result.info["active_contact_residual_inf_m"], 1.0e-7)

    def test_one_full_dt_private_frame_with_hard_stretch_and_contact(self) -> None:
        with wp.ScopedDevice("cpu"):
            scene = self.make_scene(4)
            plan = bgn.prepare(scene, DT)
            J = np.zeros((3, 6 * plan.n), dtype=np.float64)
            last = plan.n - 1
            J[0, 6 * last + 2] = 1.0
            J[1, 6 * last + 0] = 1.0
            J[2, 6 * last + 1] = 1.0
            rows = contact_rows(J, np.asarray([-1.0e-4, 0.0, 0.0]), np.asarray([1.0e4]))
            before = tuple(value.copy() for value in ccdsn.al._state_arrays(scene.state_0))
            original_predict = ccdsn.bgn._predict_pose_fast
            with mock.patch.object(
                ccdsn.bgn, "_predict_pose_fast", wraps=original_predict
            ) as predictor:
                state, metadata = ccdsn.solve_one_linearized_frame(
                    scene,
                    rows,
                    dt=DT,
                    plan=plan,
                    stretch_mode=ccdsn.HARD_STRETCH,
                    contact_mode=ccdsn.FRICTIONLESS,
                    contact_compliance=1.0e-8,
                )
            self.assertEqual(predictor.call_count, 1)
            self.assertEqual(metadata["physical_advances"], 1)
            self.assertEqual(metadata["internal_time_substeps"], 0)
            self.assertEqual(metadata["nonlinear_outer_iterations"], 1)
            self.assertTrue(metadata["caller_state_unchanged"])
            self.assertTrue(all(
                np.array_equal(a, b)
                for a, b in zip(before, ccdsn.al._state_arrays(scene.state_0))
            ))
            self.assertIsNot(state, scene.state_0)
            self.assertTrue(rc.state_finite(state))
            self.assertLessEqual(
                metadata["coupled"]["stretch_residual_inf_m"], 1.0e-7
            )


if __name__ == "__main__":
    unittest.main()
