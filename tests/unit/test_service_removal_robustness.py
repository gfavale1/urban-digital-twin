"""B8 one-at-a-time verified-destination removal: safety and numerical invariants."""
from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from analysis.analyze_service_removal_robustness import (
    _compare_removal,
    _thresholds,
    analyze_removal_in_memory,
)
from analysis.accessibility_engine import AccessibilityEngineResult


class ServiceRemovalRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.DiGraph()
        self.graph.add_edge("A", "B", length_m=600.0, walking_time_s=600 / 1.4)
        self.graph.add_edge("B", "A", length_m=600.0, walking_time_s=600 / 1.4)
        self.graph.add_edge("B", "C", length_m=600.0, walking_time_s=600 / 1.4)
        self.graph.add_edge("C", "B", length_m=600.0, walking_time_s=600 / 1.4)
        self.graph.add_node("isolated")
        self.origins = pd.DataFrame({
            "origin_id": ["O1", "O2", "O3"], "municipality_code": ["123456"] * 3,
            "population_total": [100.0, 50.0, 20.0], "geometry": [None] * 3,
            "population_reference_date": ["2023"] * 3, "census_year": ["2023"] * 3,
            "geography_version": ["BT2021"] * 3,
            "origin_method": ["equal_internal_nodes"] * 3,
        })
        self.services = pd.DataFrame({
            "service_id": ["P1", "P2", "P_BAD"], "service_type": ["pharmacy"] * 3,
            "name": ["North", "South", "Unverified"], "municipality_code": ["123456"] * 3,
            "source_name": ["health"] * 3, "source_record_id": ["1", "2", "3"],
            "source_reference_date": ["2025-06-30"] * 3,
            "retrieved_at": ["2025-07-01"] * 3,
            "operational_status": ["active"] * 3,
            "legacy_usable_for_accessibility": [True, True, False],
            "inside_operational_footprint": [True] * 3,
        })
        self.attachments = pd.DataFrame([
            dict(entity_id="O1", entity_kind="origin", mode="walk", node_id="A", snapped=True, snap_distance_m=0., attachment_quality="snapped"),
            dict(entity_id="O2", entity_kind="origin", mode="walk", node_id="C", snapped=True, snap_distance_m=0., attachment_quality="snapped"),
            dict(entity_id="O3", entity_kind="origin", mode="walk", node_id="isolated", snapped=True, snap_distance_m=0., attachment_quality="snapped"),
            dict(entity_id="P1", entity_kind="service", mode="walk", node_id="A", snapped=True, snap_distance_m=0., attachment_quality="snapped"),
            dict(entity_id="P2", entity_kind="service", mode="walk", node_id="C", snapped=True, snap_distance_m=0., attachment_quality="snapped"),
            dict(entity_id="P_BAD", entity_kind="service", mode="walk", node_id="B", snapped=True, snap_distance_m=0., attachment_quality="snapped"),
        ])

    def analyze(self, **kwargs):
        return analyze_removal_in_memory(
            self.graph, self.origins, self.services, self.attachments,
            walking_speed_m_s=1.0, thresholds=(5, 10, 20), **kwargs,
        )

    def test_all_removals_never_promote_unverified(self):
        report = self.analyze(top_k=None)
        self.assertTrue(report["full_leave_one_out"])
        self.assertEqual(report["eligible_service_count"], 2)
        self.assertEqual(report["excluded_service_count"], 1)
        self.assertEqual({r["removed_service_id"] for r in report["removals"]}, {"P1", "P2"})
        self.assertEqual(report["baseline"]["service_count_routable"], 2)
        for row in report["removals"]:
            self.assertEqual(row["scenario"]["service_count_total"], 3)
            self.assertEqual(row["scenario"]["service_count_routable"], 1)
            self.assertAlmostEqual(row["lost_reachable_population"], 0.0)
            self.assertGreater(row["coverage_loss_percentage_points"]["5"], 0)

    def test_removed_destinations_can_become_unreachable(self):
        self.graph.remove_edge("A", "B")
        self.graph.remove_edge("B", "A")
        report = self.analyze(top_k=None)
        self.assertEqual(report["baseline"]["reachable_population"], 150.0)
        self.assertEqual({r["lost_reachable_population"] for r in report["removals"]}, {50.0, 100.0})

    def test_partial_selection_uses_largest_baseline_nearest_population(self):
        report = self.analyze(top_k=1)
        self.assertEqual(len(report["removals"]), 1)
        self.assertFalse(report["full_leave_one_out"])
        self.assertEqual(report["removals"][0]["removed_service_id"], "P1")
        self.assertEqual(report["removals"][0]["baseline_nearest_population"], 100.0)

    def test_canonical_sources_not_mutated(self):
        original_services = self.services.copy(deep=True)
        original_attachments = self.attachments.copy(deep=True)
        original_edges = {k: dict(v) for k, v in self.graph.adj["A"].items()}
        self.analyze(top_k=None)
        pd.testing.assert_frame_equal(self.services, original_services)
        pd.testing.assert_frame_equal(self.attachments, original_attachments)
        self.assertEqual({k: dict(v) for k, v in self.graph.adj["A"].items()}, original_edges)

    def test_one_valid_destination_can_be_removed(self):
        self.services.loc[self.services["service_id"].eq("P2"), "legacy_usable_for_accessibility"] = False
        report = self.analyze(top_k=None)
        self.assertEqual(report["eligible_service_count"], 1)
        self.assertEqual(len(report["removals"]), 1)
        self.assertEqual(report["removals"][0]["scenario"]["reachable_population"], 0.0)
        self.assertIsNone(report["removals"][0]["scenario"]["conditional_population_weighted_mean_min"])

    def test_rejects_no_verified_destinations(self):
        self.services["legacy_usable_for_accessibility"] = False
        with self.assertRaisesRegex(ValueError, "verified routable"):
            self.analyze()

    def test_rejects_invalid_k(self):
        for value in (0, -1, True, 1.2, "2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.analyze(top_k=value)

    def test_rejects_invalid_thresholds(self):
        for vals in ((20, 10), (10, 10), (0,), ()):
            with self.subTest(vals=vals):
                with self.assertRaises(ValueError):
                    _thresholds(vals)

    def test_nonmonotonic_nearest_time_rejected(self):
        baseline = AccessibilityEngineResult(
            origins=pd.DataFrame({"origin_id": ["o"], "target_population": [1.],
                                  "nearest_service_time_min": [10.],
                                  "opportunity_count_within_10_min": [1]}),
            summary={"population_coverage_within_10_min": 1.0},
        )
        scenario = AccessibilityEngineResult(
            origins=pd.DataFrame({"origin_id": ["o"], "target_population": [1.],
                                  "nearest_service_time_min": [5.],
                                  "opportunity_count_within_10_min": [1]}),
            summary={"population_coverage_within_10_min": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "improved nearest"):
            _compare_removal(baseline, scenario, thresholds=(10,))

    def test_removal_does_not_require_mean_to_increase(self):
        # With unreachable origins excluded, conditional mean can fall after a removal.
        report = self.analyze(top_k=None)
        self.assertTrue(all(row["coverage_loss_percentage_points"]["10"] >= -1e-8
                            for row in report["removals"]))


if __name__ == "__main__":
    unittest.main()
