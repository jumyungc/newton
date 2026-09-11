from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from bench.global_cable import codex_bgn_global as bgn
from bench.global_cable import codex_contact_safe_predictor as predictor
from bench.global_cable import codex_rc_forest_newton as rc


def collision_evaluator(function):
    def evaluate(alpha: float) -> dict[str, float | bool]:
        return {
            "maximum_penetration_m": float(function(alpha)),
            "overflow": False,
        }
    return evaluate


class SafeAlphaTests(unittest.TestCase):
    def test_safe_full_prediction_is_not_clamped(self) -> None:
        result = predictor.find_safe_predictor_alpha(
            collision_evaluator(lambda alpha: alpha * 1.0e-6)
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "full_prediction_safe")
        self.assertEqual(result.alpha, 1.0)
        self.assertEqual(len(result.evaluations), 2)

    def test_unsafe_prediction_is_clamped_to_largest_certified_endpoint(self) -> None:
        result = predictor.find_safe_predictor_alpha(
            collision_evaluator(lambda alpha: alpha * 20.0e-6),
            bisection_iters=20,
        )
        self.assertTrue(result.endpoint_safe)
        self.assertLessEqual(result.maximum_penetration_m, 5.0e-6)
        self.assertLessEqual(result.alpha, 0.25)
        self.assertAlmostEqual(result.alpha, 0.25, delta=1.0e-6)
        self.assertFalse(result.continuous_path_certified)

    def test_unsafe_original_is_rejected_without_claiming_a_clamp(self) -> None:
        result = predictor.find_safe_predictor_alpha(
            collision_evaluator(lambda alpha: 6.0e-6 + alpha * 1.0e-6)
        )
        self.assertFalse(result.accepted)
        self.assertFalse(result.endpoint_safe)
        self.assertEqual(result.status, "original_endpoint_unsafe")
        self.assertEqual(result.alpha, 0.0)
        self.assertEqual(len(result.evaluations), 1)

    def test_overflow_is_treated_as_unsafe(self) -> None:
        def evaluate(alpha: float) -> dict[str, float | bool]:
            return {
                "maximum_penetration_m": 0.0,
                "overflow": alpha > 0.5,
            }

        result = predictor.find_safe_predictor_alpha(evaluate, bisection_iters=16)
        self.assertTrue(result.endpoint_safe)
        self.assertLessEqual(result.alpha, 0.5)
        self.assertAlmostEqual(result.alpha, 0.5, delta=2.0e-5)

    def test_target_cannot_relax_existing_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "may not relax"):
            predictor.find_safe_predictor_alpha(
                collision_evaluator(lambda alpha: 0.0),
                penetration_target_m=np.nextafter(5.0e-6, np.inf),
            )


class PredictionInterpolationTests(unittest.TestCase):
    def test_full_alpha_reconstructs_prediction_without_mutating_inputs(self) -> None:
        data = SimpleNamespace(dynamic_bodies=np.asarray([0, 1], dtype=np.int64))
        original = rc.PoseState(
            p_com=np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
            q=np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        )
        half = 0.5 * np.pi / 2.0
        predicted = rc.PoseState(
            p_com=np.asarray([[0.1, -0.2, 0.3], [1.2, 2.1, 2.8]]),
            q=np.asarray([
                [0.0, 0.0, np.sin(half), np.cos(half)],
                [np.sin(half), 0.0, 0.0, np.cos(half)],
            ]),
        )
        original_before = original.copy()
        predicted_before = predicted.copy()
        direction = predictor.prediction_direction(data, original, predicted)
        reconstructed = bgn._retract_fast(data, original, direction, 1.0)
        self.assertTrue(np.allclose(reconstructed.p_com, predicted.p_com, atol=1.0e-14))
        # Quaternion signs are equivalent; this construction uses matching signs.
        self.assertTrue(np.allclose(reconstructed.q, predicted.q, atol=1.0e-14))
        self.assertTrue(np.array_equal(original.p_com, original_before.p_com))
        self.assertTrue(np.array_equal(original.q, original_before.q))
        self.assertTrue(np.array_equal(predicted.p_com, predicted_before.p_com))
        self.assertTrue(np.array_equal(predicted.q, predicted_before.q))


if __name__ == "__main__":
    unittest.main()
