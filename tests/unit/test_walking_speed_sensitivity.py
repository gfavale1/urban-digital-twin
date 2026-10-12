"""Non-regression tests for read-only B8 walking-speed sensitivity."""
from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from analysis.analyze_walking_speed_sensitivity import (
    _speeds,
    analyze_in_memory,
    reweight_graph,
    validate_legacy_edge_speeds,
)
from core.analysis_spec import PopulationSelector, ServiceType


class WalkingSpeedSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.DiGraph()
        self.graph.add_edge("A", "B", length_m=900.0, walking_time_s=900 / 1.4)
        self.graph.add_edge("B", "A", length_m=900.0, walking_time_s=900 / 1.4)
        self.graph.add_node("isolated")
        self.origins = pd.DataFrame({
            "origin_id": ["O1", "O2"], "municipality_code": ["123456"] * 2,
            "population_total": [100.0, 50.0], "geometry": [None, None],
            "population_reference_date": ["2023"] * 2, "census_year": ["2023"] * 2,
            "geography_version": ["BT2021"] * 2,
            "origin_method": ["equal_internal_nodes"] * 2,
        })
        self.services = pd.DataFrame({
            "service_id": ["P1", "P2"], "service_type": ["pharmacy", "pharmacy"],
            "name": ["Inside", "Outside"], "municipality_code": ["123456"] * 2,
            "source_name": ["health"] * 2, "source_record_id": ["1", "2"],
            "source_reference_date": ["2025-06-30"] * 2,
            "retrieved_at": ["2025-07-01"] * 2,
            "operational_status": ["active"] * 2,
            "legacy_usable_for_accessibility": [True, True],
            "inside_operational_footprint": [True, False],
        })
        self.attachments = pd.DataFrame([
            dict(entity_id="O1", entity_kind="origin", mode="walk", node_id="A", snapped=True, snap_distance_m=100.0, attachment_quality="snapped"),
            dict(entity_id="O2", entity_kind="origin", mode="walk", node_id="isolated", snapped=True, snap_distance_m=0.0, attachment_quality="snapped"),
            dict(entity_id="P1", entity_kind="service", mode="walk", node_id="B", snapped=True, snap_distance_m=100.0, attachment_quality="snapped"),
            dict(entity_id="P2", entity_kind="service", mode="walk", node_id="A", snapped=True, snap_distance_m=0.0, attachment_quality="snapped"),
        ])

    def test_edges_are_checked_at_legacy_speed(self):
        edges = pd.DataFrame({"length_m": [100.0, 200.0], "walking_time_s": [100 / 1.4, 200 / 1.4]})
        self.assertIsNone(validate_legacy_edge_speeds(edges))

    def test_edges_reject_mixed_speeds(self):
        edges = pd.DataFrame({"length_m": [100.0, 200.0], "walking_time_s": [100 / 1.4, 200 / 0.9]})
        with self.assertRaisesRegex(ValueError, "not consistent"):
            validate_legacy_edge_speeds(edges)

    def test_edges_reject_nonfinite_and_zero(self):
        for bad in (0.0, -2.0, float('nan'), float('inf')):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_legacy_edge_speeds(pd.DataFrame({"length_m": [bad], "walking_time_s": [2.0]}))

    def test_graph_reweighting_does_not_mutate_baseline(self):
        fast = reweight_graph(self.graph, 1.1)
        slow = reweight_graph(self.graph, 0.7)
        self.assertAlmostEqual(self.graph["A"]["B"]["walking_time_s"], 900 / 1.4)
        self.assertAlmostEqual(fast["A"]["B"]["walking_time_s"], 900 / 1.1)
        self.assertAlmostEqual(slow["A"]["B"]["walking_time_s"], 900 / 0.7)

    def test_invalid_speed_profiles_are_rejected(self):
        for speeds in ((0.9, 0.7), (0.9, 0.9), (0.0,), (float('nan'),), ()):
            with self.subTest(speeds=speeds):
                with self.assertRaises(ValueError):
                    _speeds(speeds)

    def test_all_speed_scenarios_preserve_gate_and_connectivity(self):
        report = analyze_in_memory(self.graph, self.origins, self.services, self.attachments, thresholds=(10, 20, 30))
        self.assertTrue(report["monotonicity_verified"])
        slow, base, fast = report["results"]
        for row in (slow, base, fast):
            self.assertEqual(row["service_count_total"], 2)
            self.assertEqual(row["service_count_routable"], 1)
            self.assertEqual(row["target_population"], 150.0)
            self.assertEqual(row["reachable_population"], 100.0)
            self.assertEqual(row["reachable_population_share"], 2 / 3)
            # The geographically excluded P2 is located on the origin node.
            # Were it promoted, nearest travel would be 100/speed, not 1100/speed.
            self.assertAlmostEqual(row["conditional_population_weighted_mean_min"], 1100 / row["speed_m_s"] / 60)
        self.assertEqual(slow["coverage_by_minutes"]["20"], 0.0)
        self.assertEqual(base["coverage_by_minutes"]["20"], 0.0)
        self.assertAlmostEqual(fast["coverage_by_minutes"]["20"], 2 / 3)
        self.assertAlmostEqual(slow["coverage_by_minutes"]["30"], 2 / 3)
        self.assertAlmostEqual(base["coverage_by_minutes"]["30"], 2 / 3)
        self.assertAlmostEqual(fast["coverage_by_minutes"]["30"], 2 / 3)
        self.assertAlmostEqual(self.graph["A"]["B"]["walking_time_s"], 900 / 1.4)

    def test_snap_distance_is_rescaled_with_edges(self):
        report = analyze_in_memory(self.graph, self.origins, self.services, self.attachments,
                                   speeds=(0.7, 1.1), thresholds=(16, 18, 27))
        slow, fast = report["results"]
        self.assertEqual(slow["coverage_by_minutes"]["18"], 0.0)
        self.assertAlmostEqual(fast["coverage_by_minutes"]["18"], 2 / 3)
        self.assertAlmostEqual(slow["conditional_population_weighted_mean_min"] * 0.7,
                               fast["conditional_population_weighted_mean_min"] * 1.1)

    def test_unsnapped_destinations_are_not_promoted(self):
        attach = self.attachments.copy()
        mask = attach["entity_id"].eq("P1")
        attach.loc[mask, "snapped"] = False
        attach.loc[mask, "node_id"] = None
        attach.loc[mask, "snap_distance_m"] = float("nan")
        report = analyze_in_memory(self.graph, self.origins, self.services, attach, speeds=(0.9,))
        item = report["results"][0]
        self.assertEqual(item["service_count_routable"], 0)
        self.assertEqual(item["reachable_population"], 0.0)
        self.assertIsNone(item["conditional_population_weighted_mean_min"])

    def test_threshold_validation(self):
        for values in ((15, 10), (10, 10), (0, 10), ()):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    analyze_in_memory(self.graph, self.origins, self.services,
                                      self.attachments, thresholds=values)

    def test_population_proxy_missing_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            analyze_in_memory(self.graph, self.origins, self.services, self.attachments,
                              population_selector=PopulationSelector.AGE_5_9_PROXY,
                              speeds=(0.9,))


if __name__ == '__main__':
    unittest.main()
