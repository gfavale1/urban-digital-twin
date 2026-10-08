import unittest
import warnings

import networkx as nx
import pandas as pd

from analysis.accessibility_contracts import (
    compute_accessibility_from_canonical,
    prepare_canonical_accessibility_inputs,
    request_from_service_spec,
)
from core.analysis_spec import (
    PopulationSelector,
    ServiceAnalysisSpec,
    ServiceType,
    TransportMode,
)
from core.schema_v2 import ReachabilityStatus


class AccessibilityCanonicalContractsV2Tests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.DiGraph()
        self.graph.add_edge("A", "B", walking_time_s=60.0, length_m=60.0)
        self.graph.add_edge("B", "A", walking_time_s=60.0, length_m=60.0)
        self.graph.add_edge("B", "C", walking_time_s=60.0, length_m=60.0)
        self.graph.add_edge("C", "B", walking_time_s=60.0, length_m=60.0)
        self.graph.add_node("D")

        self.origins = pd.DataFrame(
            {
                "origin_id": ["O1", "O2", "O3"],
                "municipality_code": ["034027"] * 3,
                "geometry": [None, None, None],
                "population_total": [100.0, 50.0, 25.0],
                "population_age_lt5_proxy": [8.0, 4.0, 2.0],
                "population_age_5_9_proxy": [10.0, 5.0, 2.0],
                "population_age_10_14_proxy": [11.0, 6.0, 2.0],
                "population_age_15_19_proxy": [9.0, 5.0, 2.0],
                "population_reference_date": ["2023-12-31"] * 3,
                "census_year": ["2023"] * 3,
                "geography_version": ["BT2021"] * 3,
                "origin_method": ["representative_point"] * 3,
            }
        )

        self.services = pd.DataFrame(
            {
                "service_id": ["S1", "S2", "P1"],
                "service_type": [
                    ServiceType.PRIMARY_SCHOOL.value,
                    ServiceType.PRIMARY_SCHOOL.value,
                    ServiceType.PHARMACY.value,
                ],
                "name": ["Primary 1", "Primary 2", "Pharmacy 1"],
                "municipality_code": ["034027"] * 3,
                "source_name": ["MIM", "MIM", "MIN_SALUTE"],
                "source_record_id": ["m1", "m2", "p1"],
                "source_reference_date": ["2026-09-01", "2026-09-01", "2026-10-05"],
                "retrieved_at": ["2026-10-08T12:00:00+00:00"] * 3,
                "operational_status": ["active"] * 3,
            }
        )

        self.attachments = pd.DataFrame(
            [
                {
                    "entity_id": "O1",
                    "entity_kind": "origin",
                    "mode": "walk",
                    "node_id": "A",
                    "snapped": True,
                    "snap_distance_m": 0.0,
                    "attachment_quality": "exact",
                },
                {
                    "entity_id": "O2",
                    "entity_kind": "origin",
                    "mode": "walk",
                    "node_id": "B",
                    "snapped": True,
                    "snap_distance_m": 0.0,
                    "attachment_quality": "exact",
                },
                {
                    "entity_id": "O3",
                    "entity_kind": "origin",
                    "mode": "walk",
                    "node_id": "D",
                    "snapped": True,
                    "snap_distance_m": 0.0,
                    "attachment_quality": "exact",
                },
                {
                    "entity_id": "S1",
                    "entity_kind": "service",
                    "mode": "walk",
                    "node_id": "C",
                    "snapped": True,
                    "snap_distance_m": 0.0,
                    "attachment_quality": "exact",
                },
                {
                    "entity_id": "S2",
                    "entity_kind": "service",
                    "mode": "walk",
                    "node_id": "B",
                    "snapped": True,
                    "snap_distance_m": 0.0,
                    "attachment_quality": "exact",
                },
                {
                    "entity_id": "P1",
                    "entity_kind": "service",
                    "mode": "walk",
                    "node_id": "C",
                    "snapped": True,
                    "snap_distance_m": 0.0,
                    "attachment_quality": "exact",
                },
            ]
        )

    def primary_spec(self):
        return ServiceAnalysisSpec(
            service_type=ServiceType.PRIMARY_SCHOOL,
            modes=(TransportMode.WALK,),
            thresholds_min_by_mode={"walk": (1, 2)},
            population_selector=PopulationSelector.AGE_5_9_PROXY,
        )

    def request(self):
        return request_from_service_spec(
            self.primary_spec(),
            TransportMode.WALK,
            travel_time_weight="walking_time_s",
            off_network_speed_m_s=1.0,
        )

    def test_canonical_inputs_join_mode_specific_network_attachments(self):
        prepared = prepare_canonical_accessibility_inputs(
            self.origins,
            self.services,
            self.attachments,
            mode=TransportMode.WALK,
        )
        origin_rows = prepared.origins.set_index("origin_id")
        service_rows = prepared.services.set_index("service_id")
        self.assertEqual(origin_rows.loc["O1", "network_node_id"], "A")
        self.assertEqual(float(origin_rows.loc["O1", "origin_snap_distance_m"]), 0.0)
        self.assertEqual(service_rows.loc["S1", "network_node_id"], "C")
        self.assertEqual(float(service_rows.loc["S1", "snap_distance_m"]), 0.0)


    def test_attachment_table_overrides_legacy_routing_columns_on_entities(self):
        origins = self.origins.copy()
        origins["network_node_id"] = ["STALE", "STALE", "STALE"]
        origins["origin_snap_distance_m"] = [999.0, 999.0, 999.0]
        origins["snapped"] = [False, False, False]

        services = self.services.copy()
        services["network_node_id"] = ["STALE", "STALE", "STALE"]
        services["snap_distance_m"] = [999.0, 999.0, 999.0]
        services["snapped"] = [False, False, False]

        prepared = prepare_canonical_accessibility_inputs(
            origins,
            services,
            self.attachments,
            mode=TransportMode.WALK,
        )

        origin_rows = prepared.origins.set_index("origin_id")
        service_rows = prepared.services.set_index("service_id")
        self.assertEqual(origin_rows.loc["O1", "network_node_id"], "A")
        self.assertEqual(float(origin_rows.loc["O1", "origin_snap_distance_m"]), 0.0)
        self.assertTrue(bool(origin_rows.loc["O1", "snapped"]))
        self.assertEqual(service_rows.loc["S1", "network_node_id"], "C")
        self.assertEqual(float(service_rows.loc["S1", "snap_distance_m"]), 0.0)
        self.assertTrue(bool(service_rows.loc["S1", "snapped"]))

    def test_request_is_derived_from_service_analysis_spec(self):
        request = self.request()
        self.assertEqual(request.service_type, ServiceType.PRIMARY_SCHOOL)
        self.assertEqual(request.mode, TransportMode.WALK)
        self.assertEqual(request.thresholds_min, (1, 2))
        self.assertEqual(request.population_selector, PopulationSelector.AGE_5_9_PROXY)

    def test_canonical_compute_uses_school_target_population_proxy(self):
        result = compute_accessibility_from_canonical(
            self.graph,
            self.origins,
            self.services,
            self.attachments,
            self.request(),
        )
        rows = result.origins.set_index("origin_id")
        self.assertEqual(float(rows.loc["O1", "target_population"]), 10.0)
        self.assertEqual(float(rows.loc["O2", "target_population"]), 5.0)
        self.assertEqual(
            rows.loc["O3", "reachability_status"],
            ReachabilityStatus.NO_PATH.value,
        )
        self.assertAlmostEqual(
            result.summary["population_coverage_within_1_min"],
            15.0 / 17.0,
        )

    def test_missing_attachment_is_not_silently_treated_as_unsnapped(self):
        attachments = self.attachments.loc[
            ~(
                (self.attachments["entity_kind"] == "service")
                & (self.attachments["entity_id"] == "S2")
            )
        ].copy()
        with self.assertRaisesRegex(ValueError, "Missing walk NetworkAttachmentV2"):
            prepare_canonical_accessibility_inputs(
                self.origins,
                self.services,
                attachments,
                mode=TransportMode.WALK,
            )

    def test_explicit_unsnapped_attachment_propagates_to_engine(self):
        attachments = self.attachments.copy()
        mask = (
            (attachments["entity_kind"] == "origin")
            & (attachments["entity_id"] == "O3")
        )
        attachments.loc[mask, "snapped"] = False
        attachments.loc[mask, "node_id"] = None
        attachments.loc[mask, "snap_distance_m"] = None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            result = compute_accessibility_from_canonical(
                self.graph,
                self.origins,
                self.services,
                attachments,
                self.request(),
            )

        runtime_warnings = [
            item for item in caught
            if issubclass(item.category, RuntimeWarning)
        ]
        self.assertEqual(runtime_warnings, [])

        rows = result.origins.set_index("origin_id")
        self.assertEqual(
            rows.loc["O3", "reachability_status"],
            ReachabilityStatus.ORIGIN_NOT_SNAPPED.value,
        )

    def test_unrelated_service_attachment_is_not_required_for_one_request(self):
        attachments = self.attachments.loc[
            ~((self.attachments["entity_kind"] == "service") & (self.attachments["entity_id"] == "P1"))
        ].copy()

        result = compute_accessibility_from_canonical(
            self.graph,
            self.origins,
            self.services,
            attachments,
            self.request(),
        )

        self.assertEqual(result.summary["service_type"], ServiceType.PRIMARY_SCHOOL.value)
        self.assertEqual(result.summary["service_count_total"], 2)

    def test_duplicate_attachment_is_rejected(self):
        attachments = pd.concat(
            [self.attachments, self.attachments.iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "at most one attachment"):
            prepare_canonical_accessibility_inputs(
                self.origins,
                self.services,
                attachments,
                mode=TransportMode.WALK,
            )


if __name__ == "__main__":
    unittest.main()
