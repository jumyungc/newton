"""Focused semantic tests for the v6 results-review builder."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from bench.global_cable import codex_cable_report_v6 as report


ROOT = Path(__file__).resolve().parents[2]
HARD_VISUAL = (
    ROOT / "bench/_workspace/codex_hard_kkt_review/"
    "run_20260705T124208Z_1944273"
)


def paths() -> dict[str, Path]:
    value = report.default_paths(ROOT)
    value.update({
        "root": ROOT,
        "hard_visual_manifest": HARD_VISUAL / "render_manifest.json",
        "hard_video": HARD_VISUAL / "n128_hard_vs_finite_2x2.mp4",
        "hard_poster": HARD_VISUAL / "n128_hard_vs_finite_2x2_poster.png",
    })
    return value


class CableReportV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = report.validate_inputs(paths())

    def test_owner_and_every_pinned_input(self) -> None:
        self.assertEqual(report.guard_owner(ROOT, "v6 unit test"), report.EXPECTED_OWNER)
        self.assertEqual(self.values["report"]["schema"], "codex-cable-results-review/v5")
        self.assertEqual(self.values["hard_visual"]["media_qa"]["reader_frame_count"], 61)

    def test_hard_claim_uses_same_semantics_parity(self) -> None:
        hard = self.values["hard"]["n128"]
        parity = self.values["parity"]["n128"]
        self.assertTrue(hard["hard"]["pass"])
        self.assertEqual(hard["hard"]["completed_substeps"], 600)
        self.assertFalse(hard["gate"]["pass"])
        self.assertTrue(parity["pass"])
        self.assertTrue(parity["k2_vs_k12"]["pass"])
        self.assertLess(hard["hard"]["axial_abs_m"]["trajectory_max"], 5e-7)
        self.assertFalse(self.values["parity"]["n1024_executed"])
        scale = self.values["hard_scale"]["n512"]
        self.assertTrue(scale["phase1_600_steps"]["pass"])
        self.assertEqual(scale["phase1_600_steps"]["summary"]["completed_substeps"], 600)
        self.assertFalse(scale["full_gate"])
        self.assertEqual(scale["phase2_to_frame193"]["summary"]["failure"]["substep"], 605)
        k24 = self.values["hard_k24"]["n512"]
        self.assertTrue(k24["step1205_probe"]["pass"])
        self.assertFalse(k24["full_gate"])
        self.assertEqual(
            k24["continuation_to_frame193"]["summary"]["failure"]["global_step"],
            1209,
        )
        self.assertEqual(
            k24["continuation_to_frame193"]["summary"]["failure"]["accepted_steps"],
            18,
        )

    def test_contact_claims_keep_success_and_failures_separate(self) -> None:
        dense = self.values["dense"]
        self.assertEqual(dense["selected_robust_dense_default"], "soft_contact")
        self.assertEqual(dense["summaries"]["soft_contact"]["gate_pass_count"], 5)
        self.assertEqual(self.values["contact_v2"]["status"], "FAIL")
        self.assertEqual(self.values["contact_v2"]["result"]["completed_physical_steps"], 33)
        self.assertEqual(self.values["contact_v3"]["status"], "FAIL")
        self.assertEqual(self.values["contact_v3"]["result"]["certified_physical_steps"], 45)
        self.assertEqual(self.values["contact_v5"]["status"], "PASS")
        self.assertEqual(self.values["contact_v5"]["result"]["certified_physical_steps"], 60)
        self.assertTrue(self.values["contact_v5"]["result"]["candidate"]["pass"])

    def test_ccdsn_is_exact_law_small_oracle_only(self) -> None:
        value = self.values["ccdsn"]
        self.assertTrue(value["pass"])
        self.assertTrue(all(value["gates"].values()))
        self.assertIn("small dense", value["claim_boundary"])
        self.assertIn("no collision discovery", value["claim_boundary"])

    def test_machine_report_is_concise_and_honest(self) -> None:
        value = report.make_report(self.values, [])
        self.assertFalse(value["publication_ready"])
        self.assertFalse(value["universal_solver_established"])
        self.assertEqual(value["summary"]["visible_videos"], 3)
        self.assertEqual(value["summary"]["hard_n128_substeps"], 600)
        self.assertEqual(value["contact_negatives"][0]["certified_steps"], 32)
        self.assertEqual(value["contact_negatives"][0]["failure_attempted_step"], 33)
        self.assertEqual(value["contact_negatives"][1]["certified_steps"], 45)
        self.assertEqual(value["contact_negatives"][1]["failure_attempted_step"], 46)
        self.assertFalse(value["hard_n128"]["cross_semantics_gate_pass"])
        self.assertTrue(value["unified_contact_v5"]["pass"])
        self.assertFalse(value["unified_contact_v5"]["continuous_path_certified"])
        self.assertEqual(value["hard_n512_scale_boundary"]["total_completed_steps"], 1204)
        self.assertFalse(value["hard_n512_scale_boundary"]["full_gate"])
        self.assertTrue(value["hard_n512_scale_boundary"]["k24_step1205_pass"])
        self.assertEqual(value["hard_n512_scale_boundary"]["k24_failure_global_step"], 1209)
        self.assertEqual(value["hard_n512_scale_boundary"]["k24_failure_accepted_attempts"], 18)

    def test_html_has_three_videos_archive_and_no_internal_doc(self) -> None:
        html = report.render_html(report.make_report(self.values, []))
        parser = report.LinkParser()
        parser.feed(html)
        self.assertEqual(parser.video_tags, 3)
        self.assertIn(report.ARCHIVE_URL, html)
        self.assertIn("a router, not one universal solver", html.lower())
        self.assertIn("34× slower", html)
        self.assertIn("32 certified", html)
        self.assertIn("step 33 of 60", html)
        self.assertIn("45 certified", html)
        self.assertIn("step 46 of 60", html)
        self.assertIn("K24 path stalled after 18 accepted attempts; no K32", html)
        self.assertNotIn("cable_research.md", html)
        self.assertNotIn("research handoff", html.lower())

    def test_strict_json_rejects_duplicate_and_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
            with self.assertRaises(report.ReportError):
                report.strict_load(duplicate)
            nonfinite = Path(directory) / "nonfinite.json"
            nonfinite.write_text('{"x":NaN}', encoding="utf-8")
            with self.assertRaises(report.ReportError):
                report.strict_load(nonfinite)

    def test_links_and_responsive_css_contract(self) -> None:
        self.assertEqual(report.safe_relative("media/hard/a.mp4"), "media/hard/a.mp4")
        self.assertIsNone(report.safe_relative("#hard"))
        self.assertIsNone(report.safe_relative(report.ARCHIVE_URL))
        with self.assertRaises(report.ReportError):
            report.safe_relative("../private.md")
        self.assertIn("@media(max-width:900px)", report.CSS)
        self.assertIn("@media(max-width:520px)", report.CSS)
        self.assertNotIn("overflow-x:hidden", report.CSS)


if __name__ == "__main__":
    unittest.main()
