"""Reviewed, census-qualified addition scenarios on isolated synthetic CSV data."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.run_boundary_qualified_addition import (
    REVIEW_POLICY, execute_boundary_qualified_addition, verify_source_review,
)
from analysis.scenario_reporting_v2 import _read_table
from analysis.service_scenarios_v2 import scenario_row_sha256
from core.run_manifest import sha256_file
from tests.unit import test_service_addition_boundaries as addition_fixtures


class ReviewedAdditionTests(unittest.TestCase):
    def setUp(self):
        self.fx = addition_fixtures.B7CCandidateBoundaryTests("test_inside_candidate_passes_geography_not_review")
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.review = self.fx.root / "config/scenario_source_reviews/034027/approved.json"
        self.review.parent.mkdir(parents=True, exist_ok=True)
        self.approval = {
            "policy": REVIEW_POLICY,
            "municipality_code": "034027",
            "service_type": "pharmacy",
            "mode": "walk",
            "reviewed_candidates": [{
                "service_id": "s2", "decision": "approved_for_scenario",
                "reviewer": "Independent test reviewer", "reviewed_on": "2026-10-10",
                "evidence": [{"source_id": "synthetic-proof", "publisher": "Synthetic testing",
                              "locator": "fixture://candidate"}],
                "service_row_sha256": scenario_row_sha256(_read_table(self.fx.candidate_services).iloc[0]),
                "attachment_row_sha256": scenario_row_sha256(_read_table(self.fx.candidate_att).iloc[0]),
            }],
        }
        self.save()

    def save(self):
        self.review.write_text(json.dumps(self.approval, sort_keys=True), encoding="utf-8")

    def source_check(self, review=None):
        return verify_source_review(
            review_file=review or self.review, root=self.fx.root,
            snapshot=self.fx.snapshot,
            addition_services=_read_table(self.fx.candidate_services),
            addition_attachments=_read_table(self.fx.candidate_att),
        )

    def run_scenario(self, **overrides):
        args = dict(
            snapshot=self.fx.snapshot, canonical=self.fx.canonical,
            qualified_services=self.fx.qualified,
            addition_services=self.fx.candidate_services,
            addition_attachments=self.fx.candidate_att,
            review_file=self.review, city_name="Synthetic", analysis_date="2025-06-30",
            root=self.fx.root,
        )
        args.update(overrides)
        return execute_boundary_qualified_addition(**args)

    def test_reviewed_addition_runs_with_qualified_baseline(self):
        before = {str(p): sha256_file(p) for p in (
            *self.fx.canonical.items().values(), self.fx.qualified,
            self.fx.candidate_services, self.fx.candidate_att, self.review)}
        summary, output = self.run_scenario()
        self.assertIsNone(output)
        self.assertEqual(summary["approved_addition_ids"], ["s2"])
        self.assertEqual(summary["qualified_baseline_service_count"], 1)
        self.assertEqual(summary["scenario_service_count"], 2)
        self.assertFalse(summary["report_saved"])
        self.assertFalse((self.fx.root / "reports").exists())
        self.assertEqual(before, {str(Path(k)): sha256_file(Path(k)) for k in before})

    def test_immutable_report_is_cacheable_and_contains_proof(self):
        report_root = self.fx.root / "reports"
        summary, folder = self.run_scenario(write_report=True, output_root=report_root)
        self.assertTrue(summary["report_saved"])
        self.assertFalse(summary["report_cached"])
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual(manifest["input_provenance"]["source_review"]["review_sha256"], sha256_file(self.review))
        self.assertEqual(manifest["input_provenance"]["territorial_qualification"]["verified_input_sha256"]["footprint"], sha256_file(self.fx.footprint))
        again, directory = self.run_scenario(write_report=True, output_root=report_root)
        self.assertTrue(again["report_cached"])
        self.assertEqual(folder, directory)

    def test_preview_and_saved_execution_have_same_identity(self):
        preview, _ = self.run_scenario()
        saved, path = self.run_scenario(write_report=True, output_root=self.fx.root / "reports")
        self.assertEqual(preview["scenario_id"], saved["scenario_id"])
        self.assertEqual(preview["derived_addition_view_sha256"], saved["derived_addition_view_sha256"])
        self.assertEqual(preview["metrics"], saved["metrics"])
        self.assertTrue(path.exists())

    def test_edited_candidate_rejected_by_approved_row_hash(self):
        self.fx.edit_service(name="Modified after review")
        with patch("analysis.run_boundary_qualified_addition.run_verified_scenario") as route:
            with self.assertRaisesRegex(ValueError, "Approved service row hash mismatch"):
                self.run_scenario()
            route.assert_not_called()

    def test_changed_report_is_not_silently_overwritten(self):
        output_root = self.fx.root / "reports"
        _, dest = self.run_scenario(write_report=True, output_root=output_root)
        (dest / "metrics.json").write_text("tampered")
        with self.assertRaises(FileExistsError):
            self.run_scenario(write_report=True, output_root=output_root)

    def test_no_review_denied(self):
        self.review.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_scenario()

    def test_pending_and_hold_do_not_authorize(self):
        for status in ("hold", "pending", "approve_for_b6c_review", "approved_for_b6c_review"):
            with self.subTest(status=status):
                self.approval["reviewed_candidates"][0]["decision"] = status
                self.save()
                with patch("analysis.run_boundary_qualified_addition.run_verified_scenario") as route:
                    with self.assertRaisesRegex(ValueError, "no explicit scenario approval"):
                        self.run_scenario()
                    route.assert_not_called()

    def test_untrusted_review_directory_rejected(self):
        elsewhere = self.fx.root / "candidate_supplied_approval.json"
        elsewhere.write_text(self.review.read_text())
        with self.assertRaisesRegex(ValueError, "operator-controlled"):
            self.source_check(elsewhere)

    def test_b6b3_registry_cannot_be_used_as_approval(self):
        self.approval["policy"] = "b6b3_anncsu_health_evidence_register_v1"
        self.save()
        with self.assertRaisesRegex(ValueError, "policy"):
            self.source_check()

    def test_approval_hash_of_service_must_match(self):
        self.approval["reviewed_candidates"][0]["service_row_sha256"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "service row hash mismatch"):
            self.source_check()

    def test_approval_hash_of_attachment_must_match(self):
        self.approval["reviewed_candidates"][0]["attachment_row_sha256"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "attachment row hash mismatch"):
            self.source_check()

    def test_missing_reviewer_or_source_evidence_denied(self):
        row = self.approval["reviewed_candidates"][0]
        row["reviewer"] = " "
        self.save()
        with self.assertRaisesRegex(ValueError, "reviewer"):
            self.source_check()
        row["reviewer"] = "Independent test reviewer"
        row["evidence"] = []
        self.save()
        with self.assertRaisesRegex(ValueError, "Evidence required"):
            self.source_check()

    def test_wrong_municipality_rejected(self):
        self.approval["municipality_code"] = "077014"
        self.save()
        with self.assertRaisesRegex(ValueError, "municipality_code mismatch"):
            self.source_check()

    def test_duplicate_review_records_rejected(self):
        self.approval["reviewed_candidates"].append(dict(self.approval["reviewed_candidates"][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, "cover every candidate|Duplicate"):
            self.source_check()

    def test_missing_coordinate_preflight_rejects_even_with_approval(self):
        self.fx.edit_service(latitude=45.0)
        with patch("analysis.run_boundary_qualified_addition.run_verified_scenario") as route:
            with self.assertRaisesRegex(ValueError, "outside or without coordinates"):
                self.run_scenario()
            route.assert_not_called()

    def test_invalid_review_date_denied(self):
        self.approval["reviewed_candidates"][0]["reviewed_on"] = "yesterday"
        self.save()
        with self.assertRaises(ValueError):
            self.source_check()

    def test_wrong_mode_rejected(self):
        self.approval["mode"] = "drive"
        self.save()
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            self.source_check()


if __name__ == "__main__":
    unittest.main()
