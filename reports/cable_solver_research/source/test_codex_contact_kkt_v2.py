from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from bench.global_cable import codex_bgn_contact_kkt as v1
from bench.global_cable import codex_contact_kkt_v2 as v2


def one_contact(*, mu: float, gap: float = -0.1,
                tangent: tuple[float, float] = (0.02, -0.01)) -> v1.ContactRows:
    J = sparse.csr_matrix(np.asarray([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]))
    empty = np.zeros((1, 3))
    return v1.ContactRows(
        J=J,
        b0=np.asarray([gap, tangent[0], tangent[1]], dtype=np.float64),
        gap=np.asarray([gap]),
        mu=np.asarray([mu]),
        shape0=np.asarray([0]), shape1=np.asarray([1]),
        body0=np.asarray([0]), body1=np.asarray([-1]),
        normal=empty, tangent1=empty, tangent2=empty,
        world0=empty, world1=empty, anchor0=empty, anchor1=empty,
        reported_count=1, capacity=1,
    )


def mixed_contacts() -> v1.ContactRows:
    J = sparse.csr_matrix(np.hstack((np.eye(6), np.zeros((6, 6)))))
    empty = np.zeros((2, 3))
    return v1.ContactRows(
        J=J,
        b0=np.asarray([-0.1, 0.02, -0.01, -0.08, 0.2, 0.0]),
        gap=np.asarray([-0.1, -0.08]),
        mu=np.asarray([1.0e4, 0.5]),
        shape0=np.asarray([0, 2]), shape1=np.asarray([1, 3]),
        body0=np.asarray([0, 1]), body1=np.asarray([-1, -1]),
        normal=empty, tangent1=empty, tangent2=empty,
        world0=empty, world1=empty, anchor0=empty, anchor1=empty,
        reported_count=2, capacity=2,
    )


class DirectStickingTests(unittest.TestCase):
    def test_exact_sticking_solution_is_inside_authored_cone(self) -> None:
        H = sparse.diags(np.asarray([2., 3., 4., 5., 6., 7.])).tocsc()
        rows = one_contact(mu=1.0e4)
        compliance = 1.0e-3
        result = v2.solve_sticking_active_set(
            H, np.zeros(6), rows, compliance=compliance,
            residual_tolerance=1.0e-11,
        )
        expected = np.asarray([
            0.1 / (0.5 + compliance),
            -0.02 / (1.0 / 3.0 + compliance),
            0.01 / (0.25 + compliance),
        ])
        self.assertTrue(np.allclose(result.contact_force_N, expected, atol=1.0e-10))
        self.assertEqual(result.info["solver_mode"], "direct_sticking_active_set")
        self.assertLessEqual(result.info["equality_residual_inf"], 1.0e-11)
        self.assertLessEqual(result.info["stationarity_inf"], 1.0e-12)
        self.assertLessEqual(result.info["cone_violation"], 1.0e-12)
        physical = rows.b0 + rows.J @ result.direction
        self.assertTrue(np.allclose(
            physical + compliance * result.contact_force_N, 0.0, atol=1.0e-11
        ))

    def test_low_friction_is_rejected_not_silently_stuck(self) -> None:
        H = sparse.eye(6, format="csc")
        rows = one_contact(mu=0.5, tangent=(0.2, 0.0))
        with self.assertRaises(v2.StickyNotApplicable):
            v2.solve_sticking_active_set(H, np.zeros(6), rows)
        result = v2.solve_dual_hybrid(
            H, np.zeros(6), rows, max_iters=300, stick_mu=1.0e3
        )
        self.assertEqual(result.info["solver_mode"], "associated_cone_fista_fallback")
        self.assertIn("high-friction", result.info["sticky_rejection"])
        self.assertLessEqual(result.info["cone_violation"], 1.0e-10)

    def test_no_active_contact_returns_free_solution(self) -> None:
        H = sparse.diags(np.asarray([2., 3., 4., 5., 6., 7.])).tocsc()
        rhs = np.arange(1.0, 7.0)
        rows = one_contact(mu=1.0e4, gap=0.2, tangent=(0.0, 0.0))
        result = v2.solve_sticking_active_set(H, rhs, rows)
        self.assertEqual(result.info["solver_mode"], "no_active_contact")
        self.assertTrue(np.allclose(result.direction, rhs / np.arange(2.0, 8.0)))
        self.assertEqual(len(result.contact_force_N), 0)

    def test_contact_limit_fails_closed_to_hybrid_fallback(self) -> None:
        H = sparse.eye(6, format="csc")
        rows = one_contact(mu=1.0e4)
        result = v2.solve_dual_hybrid(
            H, np.zeros(6), rows, direct_contact_limit=0, max_iters=300
        )
        self.assertEqual(result.info["solver_mode"], "associated_cone_fista_fallback")
        self.assertIn("exceed direct limit", result.info["sticky_rejection"])

    def test_mixed_stick_rows_are_condensed_before_small_cone_solve(self) -> None:
        H = sparse.eye(12, format="csc")
        rows = mixed_contacts()
        result = v2.solve_mixed_stick_slide_active_set(
            H, np.zeros(12), rows, max_iters=800, tolerance=1.0e-11
        )
        self.assertEqual(
            result.info["solver_mode"],
            "mixed_stick_condensation_plus_dense_cone",
        )
        self.assertEqual(result.info["sticking_contact_count"], 1)
        self.assertEqual(result.info["sliding_candidate_contact_count"], 1)
        self.assertTrue(result.info["reduced_cone"]["converged"])
        self.assertLessEqual(result.info["sticky_equality_residual_inf"], 1.0e-10)
        self.assertLessEqual(result.info["cone_violation"], 1.0e-10)
        self.assertLessEqual(result.info["projected_kkt_inf"], 1.0e-9)


class AdaptiveCollisionSafeSearchTests(unittest.TestCase):
    @staticmethod
    def evaluator(penetration) -> callable:
        def evaluate(alpha: float) -> dict[str, float | bool | int]:
            return {
                "finite": True,
                "maximum_penetration_m": float(penetration(alpha)),
                "active_rows": 1,
                "cable_objective": float((1.0 - alpha) ** 2),
                "payload": alpha,
            }
        return evaluate

    def test_full_safe_step_is_accepted_without_needless_backtracking(self) -> None:
        selected, trials, info = v2._adaptive_collision_safe_search(
            self.evaluator(lambda alpha: 1.0e-6 * alpha)
        )
        self.assertEqual(selected["alpha"], 1.0)
        self.assertEqual(len(trials), 1)
        self.assertEqual(info["status"], "full_step_safe")
        self.assertTrue(info["endpoint_safe"])

    def test_adaptive_search_goes_below_legacy_one_over_32_floor(self) -> None:
        # The gate is crossed at alpha=0.01.  Every historical trial is unsafe;
        # adaptive continuation must find and refine the narrow safe interval.
        selected, trials, info = v2._adaptive_collision_safe_search(
            self.evaluator(lambda alpha: 5.0e-4 * alpha)
        )
        self.assertTrue(info["endpoint_safe"])
        self.assertEqual(info["status"], "backtracked_endpoint_safe")
        self.assertLessEqual(selected["maximum_penetration_m"], 5.0e-6)
        self.assertGreater(selected["alpha"], 0.0099)
        self.assertLessEqual(selected["alpha"], 0.01)
        self.assertTrue(any(trial["alpha"] < 0.03125 for trial in trials))

    def test_bisection_returns_largest_certified_endpoint_in_bracket(self) -> None:
        selected, _, info = v2._adaptive_collision_safe_search(
            self.evaluator(lambda alpha: 20.0e-6 * alpha),
            bisection_iters=16,
        )
        self.assertTrue(info["endpoint_safe"])
        self.assertLessEqual(selected["maximum_penetration_m"], 5.0e-6)
        self.assertAlmostEqual(selected["alpha"], 0.25, delta=2.0e-6)
        self.assertGreater(info["bisection_iterations"], 0)

    def test_unsafe_current_pose_is_reported_and_cannot_pass_silently(self) -> None:
        selected, _, info = v2._adaptive_collision_safe_search(
            self.evaluator(lambda alpha: 6.0e-6 + alpha * 1.0e-6),
            bisection_iters=4,
        )
        self.assertEqual(selected["alpha"], 0.0)
        self.assertEqual(info["status"], "no_safe_endpoint")
        self.assertFalse(info["endpoint_safe"])
        self.assertGreater(selected["maximum_penetration_m"], 5.0e-6)

    def test_target_cannot_relax_frame_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "may not relax"):
            v2._adaptive_collision_safe_search(
                self.evaluator(lambda alpha: 0.0),
                penetration_target_m=np.nextafter(5.0e-6, np.inf),
            )


if __name__ == "__main__":
    unittest.main()
