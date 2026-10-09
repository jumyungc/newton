from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from scipy import sparse

from bench.global_cable import codex_bgn_contact_kkt as v1
from bench.global_cable import codex_contact_kkt_v3 as v3
from bench.global_cable import codex_contact_kkt_v5 as v5
from bench.global_cable import codex_desaxce as desaxce
from bench.global_cable.test_codex_contact_kkt_v4 import (
    mixed_contacts,
    one_contact,
)


HIGH_W = np.asarray([
    [5.179218648349633, 1.2076823475965828, -0.2548715924647601,
     -0.9719484223873645, -1.4341459410827837, 0.15850684511299382],
    [1.2076823475965828, 9.278835605329139, 0.5180758855291243,
     -5.903092330382727, -0.0652752086133593, 3.846244935268347],
    [-0.2548715924647601, 0.5180758855291242, 4.414148656353993,
     -0.8113387012992807, 2.9279696282002496, -5.303001353198459],
    [-0.9719484223873645, -5.903092330382727, -0.8113387012992808,
     6.080187052161119, -0.5854293373852351, -4.287054997605523],
    [-1.4341459410827835, -0.06527520861335978, 2.9279696282002496,
     -0.5854293373852345, 6.599960987026361, -3.8518210011742107],
    [0.15850684511299404, 3.8462449352683468, -5.303001353198458,
     -4.287054997605523, -3.8518210011742107, 18.689423085498746],
])
HIGH_B = 100.0 * np.asarray([
    -8.97419645893467, 15.224378094266152, -6.635613664819046,
    -28.37161444637119, 14.153822410015302, 6.478363710812819,
])
HIGH_MU = np.asarray([1.3052458378137526, 0.3331980628748356])


class CertificateAwareAccuracyTests(unittest.TestCase):
    def test_high_impulse_relative_pass_is_refined_to_absolute_law_pass(self) -> None:
        baseline = desaxce.solve_desaxce_dense(
            HIGH_W,
            HIGH_B,
            HIGH_MU,
            tolerance=5.0e-9,
            max_iterations=400,
        )
        self.assertGreater(np.max(np.abs(baseline.impulse)), 800.0)
        self.assertLessEqual(
            baseline.info["natural_map_relative_inf"], 5.0e-9
        )
        self.assertTrue(any((
            baseline.info["cone_violation"] > 1.0e-8,
            baseline.info["dual_cone_violation"] > 1.0e-8,
            baseline.info["complementarity_inf"] > 1.0e-8,
            baseline.info["friction_work"] > 1.0e-8,
        )))

        refined = v5.solve_desaxce_certificate_aware(
            HIGH_W,
            HIGH_B,
            HIGH_MU,
            tolerance=5.0e-9,
            max_iterations=400,
        )
        self.assertTrue(refined.info["certificate_pass"])
        self.assertGreaterEqual(refined.info["certificate_refinements"], 1)
        self.assertLessEqual(
            refined.info["natural_map_relative_inf"], 5.0e-9
        )
        self.assertLessEqual(refined.info["cone_violation"], 1.0e-8)
        self.assertLessEqual(refined.info["dual_cone_violation"], 1.0e-8)
        self.assertLessEqual(refined.info["complementarity_inf"], 1.0e-8)
        self.assertLessEqual(refined.info["friction_work"], 1.0e-8)
        attempts = refined.info["certificate_attempts"]
        self.assertFalse(attempts[0]["pass"])
        self.assertTrue(attempts[-1]["pass"])
        self.assertTrue(all(
            right["solve_tolerance"] < left["solve_tolerance"]
            for left, right in zip(attempts, attempts[1:])
        ))

    def test_mixed_stick_desaxce_reports_absolute_certificate_pass(self) -> None:
        result = v5.solve_dual_hybrid(
            sparse.eye(12, format="csc"),
            np.zeros(12),
            mixed_contacts(),
            max_iters=80,
        )
        info = result.info
        self.assertTrue(info["desaxce"]["certificate_pass"])
        self.assertLessEqual(
            info["desaxce_primal_cone_violation"], 1.0e-8
        )
        self.assertLessEqual(
            info["desaxce_dual_cone_violation"], 1.0e-8
        )
        self.assertLessEqual(info["desaxce_complementarity_inf"], 1.0e-8)
        self.assertLessEqual(info["desaxce_friction_dissipation"], 1.0e-8)

    def test_refinement_failure_has_no_associated_fallback(self) -> None:
        H = sparse.eye(6, format="csc")
        rows = one_contact(mu=0.5)
        with mock.patch.object(
            v5.desaxce,
            "solve_desaxce_dense",
            side_effect=RuntimeError("synthetic refinement failure"),
        ), mock.patch.object(v1, "solve_dual_cone_qp") as associated:
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                v5.solve_dual_hybrid(H, np.zeros(6), rows)
        associated.assert_not_called()


class BindingRegressionTests(unittest.TestCase):
    def test_v5_frame_binding_does_not_mutate_v3(self) -> None:
        original = v3.solve_dual_hybrid
        bound = v5._bound_v3_frame()
        self.assertIs(bound.__globals__["solve_dual_hybrid"], v5.solve_dual_hybrid)
        self.assertIs(v3.solve_dual_hybrid, original)


if __name__ == "__main__":
    unittest.main()
