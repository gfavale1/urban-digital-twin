import unittest
import warnings

import networkx as nx
import pandas as pd

from analysis.accessibility_engine import (
    AccessibilityEngineRequest,
    compute_accessibility,
)
from core.analysis_spec import PopulationSelector, ServiceType, TransportMode
from core.schema_v2 import ReachabilityStatus


class AccessibilityEngineV2Tests(unittest.TestCase):
    def setUp(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B", walking_time_s=60.0, length_m=60.0)
        graph.add_edge("B", "C", walking_time_s=60.0, length_m=60.0)
        graph.add_edge("C", "B", walking_time_s=60.0, length_m=60.0)
        graph.add_edge("B", "A", walking_time_s=60.0, length_m=60.0)
        graph.add_node("D")
        self.graph = graph

        self.origins = pd.DataFrame(
            {
                "origin_id": ["O1", "O2", "O3", "O4"],
                "network_node_id": ["A", "B", "D", None],
                "origin_snap_distance_m": [0.0, 0.0, 0.0, None],
                "population_total": [100.0, 50.0, 25.0, 10.0],
                "population_age_5_9_proxy": [10.0, 5.0, 2.0, 1.0],
            }
        )

        self.services = pd.DataFrame(
            {
                "service_id": ["S1", "S2", "H1"],
                "service_type": [
                    ServiceType.PRIMARY_SCHOOL.value,
                    ServiceType.PRIMARY_SCHOOL.value,
                    ServiceType.HOSPITAL_ESTABLISHMENT.value,
                ],
                "network_node_id": ["C", "B", "C"],
                "snap_distance_m": [0.0, 0.0, 0.0],
            }
        )

    def request(self, service_type=ServiceType.PRIMARY_SCHOOL):
        return AccessibilityEngineRequest(
            service_type=service_type,
            mode=TransportMode.WALK,
            thresholds_min=(1, 2),
            population_selector=PopulationSelector.TOTAL,
            travel_time_weight="walking_time_s",
            off_network_speed_m_s=1.0,
        )

    def test_nearest_time_counts_and_reachability(self):
        result = compute_accessibility(
            self.graph,
            self.origins,
            self.services,
            self.request(),
        )
        rows = result.origins.set_index("origin_id")

        self.assertEqual(float(rows.loc["O1", "nearest_service_time_min"]), 1.0)
        self.assertEqual(float(rows.loc["O2", "nearest_service_time_min"]), 0.0)
        self.assertEqual(int(rows.loc["O1", "opportunity_count_within_1_min"]), 1)
        self.assertEqual(int(rows.loc["O1", "opportunity_count_within_2_min"]), 2)
        self.assertTrue(bool(rows.loc["O1", "has_service_within_1_min"]))

        self.assertEqual(
            rows.loc["O3", "reachability_status"],
            ReachabilityStatus.NO_PATH.value,
        )
        self.assertTrue(pd.isna(rows.loc["O3", "nearest_service_time_min"]))
        self.assertEqual(
            rows.loc["O4", "reachability_status"],
            ReachabilityStatus.ORIGIN_NOT_SNAPPED.value,
        )

    def test_population_coverage_uses_target_population(self):
        result = compute_accessibility(
            self.graph,
            self.origins,
            self.services,
            self.request(),
        )
        # O1 + O2 are covered within 1 minute: 150 / total 185.
        self.assertAlmostEqual(
            result.summary["population_coverage_within_1_min"],
            150.0 / 185.0,
        )
        self.assertAlmostEqual(
            result.summary["population_coverage_within_2_min"],
            150.0 / 185.0,
        )

    def test_population_selector_changes_denominator_without_changing_routing(self):
        request = AccessibilityEngineRequest(
            service_type=ServiceType.PRIMARY_SCHOOL,
            mode=TransportMode.WALK,
            thresholds_min=(1, 2),
            population_selector=PopulationSelector.AGE_5_9_PROXY,
            travel_time_weight="walking_time_s",
            off_network_speed_m_s=1.0,
        )
        result = compute_accessibility(
            self.graph,
            self.origins,
            self.services,
            request,
        )
        self.assertAlmostEqual(
            result.summary["population_coverage_within_1_min"],
            15.0 / 18.0,
        )
        self.assertEqual(
            result.origins["population_selector"].unique().tolist(),
            [PopulationSelector.AGE_5_9_PROXY.value],
        )

    def test_service_unavailable_is_explicit(self):
        result = compute_accessibility(
            self.graph,
            self.origins,
            self.services,
            self.request(ServiceType.COMMUNITY_HOUSE),
        )
        rows = result.origins.set_index("origin_id")
        self.assertEqual(
            rows.loc["O1", "reachability_status"],
            ReachabilityStatus.SERVICE_UNAVAILABLE.value,
        )
        self.assertTrue(pd.isna(rows.loc["O1", "nearest_service_time_min"]))
        self.assertEqual(result.summary["service_count_total"], 0)

    def test_unsnapped_origin_does_not_emit_runtime_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = compute_accessibility(
                self.graph,
                self.origins,
                self.services,
                self.request(),
            )
        self.assertEqual(len(result.origins), len(self.origins))

    def test_invariants_hold(self):
        result = compute_accessibility(
            self.graph,
            self.origins,
            self.services,
            self.request(),
        )
        df = result.origins

        self.assertTrue(
            (
                df["opportunity_count_within_2_min"]
                >= df["opportunity_count_within_1_min"]
            ).all()
        )
        self.assertTrue(
            (
                df["has_service_within_1_min"]
                == (df["opportunity_count_within_1_min"] >= 1)
            ).all()
        )
        self.assertTrue(
            0.0 <= result.summary["population_coverage_within_1_min"] <= 1.0
        )


if __name__ == "__main__":
    unittest.main()
