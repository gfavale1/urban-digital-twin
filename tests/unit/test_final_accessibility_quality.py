"""Final B8 QA safety gates, synthetic numerical reconciliation, immutable export."""
from __future__ import annotations

import json
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quality.finalize_accessibility_quality import (
    build_assessment,
    evaluate_city,
    check_quality_denominators,
    ensure_baseline_equivalence,
    summarize_removal,
    validate_parameters,
    write_new_report,
    run,
)


def quality():
    cat = {
        "total": 3, "routable": 2, "ineligible": 1,
        "primary_routing_reason": {"eligible": 2, "location_not_validated": 1},
        "geography": {"inside_census_footprint": 2, "coordinates_missing": 1},
        "overlapping_issues": {"coordinates_missing": 1, "not_snapped": 1},
    }
    return {
        "services_total": 3, "services_routable": 2, "services_ineligible": 1,
        "primary_routing_reason": cat["primary_routing_reason"],
        "geography": cat["geography"], "overlapping_issues": cat["overlapping_issues"],
        "service_types": {"pharmacy": cat},
    }


def summary():
    return {
        "speed_m_s": 0.9, "origin_count": 3, "target_population": 170.0,
        "reachable_population": 150.0, "reachable_population_share": 150 / 170,
        "conditional_population_weighted_mean_min": 12.0,
        "coverage_by_minutes": {"10": 0.5, "15": 0.7, "20": 0.8},
        "service_count_total": 3, "service_count_routable": 2,
    }


def scenarios():
    row = summary()
    baseline = deepcopy(row)
    baseline.pop("speed_m_s")
    return (
        {"results": [row], "monotonicity_verified": True},
        {"baseline": baseline, "full_leave_one_out": True,
         "selection": "all_verified", "eligible_service_count": 2,
         "excluded_service_count": 1,
         "removals": [
             {"removed_service_id": "p1", "lost_reachable_population": 0.0,
              "coverage_loss_percentage_points": {"10": 2.0, "15": 3.0, "20": 1.0},
              "baseline_nearest_population": 100.0, "scenario": {}},
             {"removed_service_id": "p2", "lost_reachable_population": 20.0,
              "coverage_loss_percentage_points": {"10": 1.0, "15": 1.0, "20": 4.0},
              "baseline_nearest_population": 50.0, "scenario": {}},
         ]},
    )


class FinalQualityTests(unittest.TestCase):
    def setUp(self):
        self.q = check_quality_denominators(quality(), "pharmacy")
        self.s, self.r = scenarios()
        self.thresholds = (10, 15, 20)

    def compare(self):
        return ensure_baseline_equivalence(self.s, self.r, baseline_speed=0.9,
                                          thresholds=self.thresholds, quality=self.q)

    def test_valid_parameters(self):
        ids, speeds, thresholds = validate_parameters(["034027", "077014"],
                                                      (0.7, 0.9, 1.1), (10, 15, 20), 0.9)
        self.assertEqual(ids, ["034027", "077014"])
        self.assertEqual(speeds, (0.7, 0.9, 1.1))
        self.assertEqual(thresholds, (10, 15, 20))

    def test_no_duplicate_municipality(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_parameters(["034027", "034027"], (0.7, 0.9), (10,), 0.9)

    def test_invalid_speed_grid(self):
        for speeds in ((0.9, 0.7), (0.9, 0.9), (0.7, 1.1), (0.9, float("nan"))):
            with self.subTest(speeds=speeds), self.assertRaises(ValueError):
                validate_parameters(["034027"], speeds, (10, 15), 0.9)

    def test_invalid_thresholds(self):
        for minutes in ((10, 10), (20, 10), (0, 10), (), (10.5,)):
            with self.subTest(minutes=minutes), self.assertRaises(ValueError):
                validate_parameters(["034027"], (0.9,), minutes, 0.9)

    def test_quality_counts_reconcile(self):
        self.assertEqual(self.q["focus_services_excluded"], 1)
        self.assertEqual(self.q["focus_overlapping_issues"]["not_snapped"], 1)

    def test_quality_rejects_bad_exclusion_total(self):
        q = quality()
        q["service_types"]["pharmacy"]["primary_routing_reason"]["eligible"] = 1
        with self.assertRaisesRegex(ValueError, "Exclusive"):
            check_quality_denominators(q, "pharmacy")

    def test_missing_category_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "No audited"):
            check_quality_denominators(quality(), "hospital_establishment")

    def test_baselines_match(self):
        self.assertEqual(self.compare()["fully_recomputed_removals"], 2)

    def test_partial_removals_rejected(self):
        self.r["full_leave_one_out"] = False
        with self.assertRaisesRegex(ValueError, "partial"):
            self.compare()

    def test_duplicate_removal_ids_rejected(self):
        self.r["removals"][1]["removed_service_id"] = "p1"
        with self.assertRaisesRegex(ValueError, "eligibility"):
            self.compare()

    def test_inconsistent_eligibility_rejected(self):
        self.r["eligible_service_count"] = 3
        with self.assertRaises(ValueError):
            self.compare()

    def test_baseline_coverage_drift_rejected(self):
        self.r["baseline"]["coverage_by_minutes"]["15"] = 0.69
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            self.compare()

    def test_baseline_population_drift_rejected(self):
        self.r["baseline"]["target_population"] = 171.0
        with self.assertRaisesRegex(ValueError, "target_population"):
            self.compare()

    def test_negative_removal_loss_rejected(self):
        self.r["removals"][0]["coverage_loss_percentage_points"]["20"] = -1.0
        with self.assertRaisesRegex(ValueError, "improved coverage"):
            self.compare()

    def test_uncertainty_and_worst_removals(self):
        data = summarize_removal(self.r, self.thresholds)
        self.assertEqual(data["worst_coverage_loss_by_threshold"]["10"]["service_id"], "p1")
        self.assertEqual(data["worst_coverage_loss_by_threshold"]["20"]["service_id"], "p2")
        self.assertEqual(len(data["removals_losing_any_reachability"]), 1)

    def test_immutable_json_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "final.json"
            payload = {"status": "PASS", "policy": "test", "data": [1, 2]}
            write_new_report(output, payload)
            self.assertEqual(json.loads(output.read_text()), payload)
            with self.assertRaises(FileExistsError):
                write_new_report(output, payload)
            self.assertEqual(json.loads(output.read_text()), payload)

    def test_refuse_preflight_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.json"
            with self.assertRaisesRegex(ValueError, "Refusing"):
                write_new_report(target, {"status": "PREFLIGHT_ONLY"})
            self.assertFalse(target.exists())

    def test_write_mode_requires_explicit_output(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            run(["--write"])
        with self.assertRaisesRegex(ValueError, "requires"):
            run(["--output", "somewhere.json"])
        with self.assertRaisesRegex(ValueError, "Cannot write"):
            run(["--write", "--output", "x.json", "--preflight-only"])

    def test_integrated_real_engine_synthetic(self):
        from tests.unit.test_service_removal_robustness import ServiceRemovalRobustnessTests
        fixture = ServiceRemovalRobustnessTests(methodName="test_all_removals_never_promote_unverified")
        fixture.setUp()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inspection = {
                "quality": {**quality(), "input_sha256": {}, "graph_sha256": "synthetic"},
                "origins": fixture.origins,
                "qualified_services": fixture.services,
                "attachments": fixture.attachments,
                "nodes": None, "edges": None,
                "origins_path": root / "data" / "origins.parquet",
                "inputs_sha256": {"origins": "synthetic"},
            }
            with patch("quality.finalize_accessibility_quality.build_routing_graph",
                       return_value=(fixture.graph, {})), \
                 patch("quality.finalize_accessibility_quality.check_source_hashes"):
                result = evaluate_city(inspection, root=root,
                                       speeds=(0.7, 0.9, 1.1),
                                       thresholds=(5, 10, 20), baseline_speed=0.9)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["robustness"]["individual_removals"], 2)
            self.assertEqual(result["cross_experiment_baseline_check"]["status"], "PASS")
            self.assertEqual(result["quality"]["focus_services_excluded"], 1)

    def test_build_assessment_never_calls_experiments_when_preflight(self):
        from types import SimpleNamespace
        sample = {"snapshot": SimpleNamespace(municipality_code="123456"),
                  "quality": quality(), "inputs_sha256": {"origins": "hash"},
                  "origins_path": Path("/project/data/features/accessibility/123456/origins.parquet")}
        with patch("quality.finalize_accessibility_quality.inspect_city_inputs", return_value=sample), \
             patch("quality.finalize_accessibility_quality.check_source_hashes"), \
             patch("quality.finalize_accessibility_quality.sha256_file", return_value="hash"), \
             patch("quality.finalize_accessibility_quality.evaluate_city") as experiment:
            report = build_assessment(Path("/project"), ["123456"], preflight_only=True)
            experiment.assert_not_called()
            self.assertEqual(report["status"], "PREFLIGHT_ONLY")
            self.assertEqual(report["verification"]["speed_sensitivity"], "NOT_EXECUTED")


if __name__ == "__main__":
    unittest.main()
