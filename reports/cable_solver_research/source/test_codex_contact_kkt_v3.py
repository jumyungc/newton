from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from scipy import sparse

from bench.global_cable import codex_bgn_contact_kkt as v1
from bench.global_cable import codex_contact_kkt_v2 as v2
from bench.global_cable import codex_contact_kkt_v3 as v3
from bench.global_cable import codex_contact_kkt_v3_pile as v3_harness
from bench.global_cable import codex_contact_safe_predictor as predictor
from bench.global_cable import codex_rc_forest_newton as rc


def one_contact() -> v1.ContactRows:
    J = sparse.csr_matrix(np.asarray([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]))
    empty = np.zeros((1, 3))
    return v1.ContactRows(
        J=J,
        b0=np.asarray([-0.1, 0.02, -0.01]),
        gap=np.asarray([-0.1]),
        mu=np.asarray([1.0e4]),
        shape0=np.asarray([0]), shape1=np.asarray([1]),
        body0=np.asarray([0]), body1=np.asarray([-1]),
        normal=empty, tangent1=empty, tangent2=empty,
        world0=empty, world1=empty, anchor0=empty, anchor1=empty,
        reported_count=1, capacity=1,
    )


def safe_prediction() -> predictor.SafePredictorResult:
    target = rc.PoseState(
        p_com=np.asarray([[1.0, 2.0, 3.0]]),
        q=np.asarray([[0.0, 0.0, 0.0, 1.0]]),
    )
    initial = rc.PoseState(
        p_com=np.asarray([[0.5, 1.0, 1.5]]),
        q=np.asarray([[0.0, 0.0, 0.0, 1.0]]),
    )
    decision = predictor.SafeAlphaDecision(
        status="prediction_clamped_to_safe_endpoint",
        accepted=True,
        endpoint_safe=True,
        alpha=0.5,
        maximum_penetration_m=4.0e-6,
        penetration_target_m=5.0e-6,
        evaluations=(),
        monotonicity_assumed=True,
    )
    return predictor.SafePredictorResult(
        physical_target=target,
        initial_guess=initial,
        decision=decision,
        caller_unchanged=True,
        physical_target_unchanged=True,
        commit_aware_endpoint_certification=True,
    )


class FullDtTargetTests(unittest.TestCase):
    def test_float32_commit_projection_catches_measured_nanometre_drift(self) -> None:
        # Choose an origin near 4 cm whose float32 rounding moves the committed
        # surface inward by 1.025 nm, matching the failed canonical step's
        # 2.499635 -> 2.500660 um transition.
        base = float(np.float32(0.04))
        measured_drift_m = 1.025026e-9
        raw_z = base + measured_drift_m
        raw_penetration = 2.4996354264436582e-6
        plane_z = raw_z + raw_penetration
        data = SimpleNamespace(
            dynamic_bodies=np.asarray([0], dtype=np.int64),
            body_com=np.zeros((1, 3), dtype=np.float64),
        )
        raw = rc.PoseState(
            p_com=np.asarray([[0.0, 0.0, raw_z]]),
            q=np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        )
        committed, body_q = v3._float32_commit_roundtrip(data, raw)
        committed_penetration = plane_z - committed.p_com[0, 2]
        drift = committed_penetration - raw_penetration
        self.assertAlmostEqual(drift * 1.0e9, 1.025026, delta=1.0e-6)
        self.assertLess(raw_penetration, v3.PREDICTOR_HEADROOM_TARGET_M)
        self.assertGreater(
            committed_penetration, v3.PREDICTOR_HEADROOM_TARGET_M
        )
        self.assertTrue(v3._committed_within_internal_headroom({
            "overflow": False,
            "maximum_penetration_m": raw_penetration,
        }))
        self.assertFalse(v3._committed_within_internal_headroom({
            "overflow": False,
            "maximum_penetration_m": committed_penetration,
        }))
        # Reusing the certified body_q is idempotent: no second commit drift.
        committed_again, body_q_again = v3._float32_commit_roundtrip(
            data, committed
        )
        self.assertTrue(np.array_equal(body_q_again, body_q))
        self.assertTrue(np.array_equal(
            committed_again.p_com, committed.p_com
        ))

    def test_headroom_is_stricter_and_public_gate_is_not_relaxed(self) -> None:
        self.assertEqual(v3.PREDICTOR_HEADROOM_TARGET_M, 2.5e-6)
        self.assertEqual(v3.PUBLIC_SOLVER_PENETRATION_GATE_M, 5.0e-6)
        self.assertEqual(
            v3.PUBLIC_SOLVER_PENETRATION_GATE_M,
            v2.CONTACT_PENETRATION_GATE_M,
        )
        self.assertLess(
            v3.PREDICTOR_HEADROOM_TARGET_M,
            v3.PUBLIC_SOLVER_PENETRATION_GATE_M,
        )

    def test_committed_state_must_meet_internal_not_only_public_gate(self) -> None:
        at_headroom = {
            "overflow": False,
            "maximum_penetration_m": 2.5e-6,
        }
        between_gates = {
            "overflow": False,
            "maximum_penetration_m": 3.0e-6,
        }
        self.assertTrue(v3._committed_within_internal_headroom(at_headroom))
        self.assertFalse(
            v3._committed_within_internal_headroom(between_gates)
        )
        self.assertLess(
            between_gates["maximum_penetration_m"],
            v3.PUBLIC_SOLVER_PENETRATION_GATE_M,
        )

    def test_assembly_receives_physical_target_not_clamped_guess(self) -> None:
        prediction = safe_prediction()
        plan = SimpleNamespace()
        pose = object()
        previous = {}
        with mock.patch.object(v3.v1, "_assemble", return_value="system") as call:
            result = v3._assemble_against_physical_target(
                plan, pose, prediction, previous
            )
        self.assertEqual(result, "system")
        self.assertIs(call.call_args.args[2], prediction.physical_target)
        self.assertIsNot(call.call_args.args[2], prediction.initial_guess)

    def test_merit_receives_same_unchanged_physical_target(self) -> None:
        prediction = safe_prediction()
        plan = SimpleNamespace()
        pose = object()
        previous = {}
        expected = (3.0, {"stretch": 1.0})
        with mock.patch.object(v3.v1, "_objective", return_value=expected) as call:
            result = v3._objective_against_physical_target(
                plan, pose, prediction, previous
            )
        self.assertEqual(result, expected)
        self.assertIs(call.call_args.args[2], prediction.physical_target)
        self.assertIsNot(call.call_args.args[2], prediction.initial_guess)

    def test_predictor_metadata_preserves_one_dt_contract(self) -> None:
        info = safe_prediction().info()
        self.assertEqual(info["predictor_time_advance_count"], 0)
        self.assertEqual(info["physical_dt_fraction"], 1.0)
        self.assertTrue(info["physical_target_unchanged"])
        self.assertTrue(info["caller_unchanged"])
        self.assertTrue(info["commit_aware_endpoint_certification"])


class AlgebraParityTests(unittest.TestCase):
    def test_v3_uses_identical_v2_certified_contact_algebra(self) -> None:
        H = sparse.diags(np.asarray([2., 3., 4., 5., 6., 7.])).tocsc()
        rhs = np.zeros(6)
        rows = one_contact()
        expected = v2.solve_dual_hybrid(H, rhs, rows)
        actual = v3.solve_dual_hybrid(H, rhs, rows)
        self.assertEqual(actual.info["solver_mode"], expected.info["solver_mode"])
        self.assertTrue(np.array_equal(actual.direction, expected.direction))
        self.assertTrue(np.array_equal(
            actual.contact_force_N, expected.contact_force_N
        ))
        self.assertEqual(
            actual.info["projected_kkt_inf"],
            expected.info["projected_kkt_inf"],
        )
        self.assertEqual(
            actual.info["cone_violation"], expected.info["cone_violation"]
        )


class FailureCheckpointTests(unittest.TestCase):
    def test_exception_checkpoint_retains_proof_fields(self) -> None:
        last = {
            "physical_step": 31,
            "monitor_penetration_m": 2.0e-6,
            "time_advance_count": 1,
            "caller_unchanged": True,
        }
        checkpoint = v3_harness._exception_checkpoint(
            step=32,
            completed_steps=31,
            certified_steps=31,
            exc=RuntimeError("synthetic failure"),
            failed_call_ms=12.5,
            source_hash_before="same",
            source_hash_after="same",
            last_certified=last,
        )
        self.assertEqual(checkpoint["attempted_physical_step"], 32)
        self.assertEqual(
            checkpoint["completed_physical_steps_before_failure"], 31
        )
        self.assertEqual(
            checkpoint["last_certified"]["monitor_penetration_m"], 2.0e-6
        )
        self.assertEqual(checkpoint["failed_call_wall_ms"], 12.5)
        self.assertTrue(checkpoint["caller_unchanged_during_failed_call"])
        self.assertEqual(checkpoint["failed_step_time_advance_count"], 0)


if __name__ == "__main__":
    unittest.main()
