"""B6C.0 contract tests; no external datasets, I/O or database required."""
import dataclasses
import unittest

import networkx as nx
import pandas as pd

from analysis.accessibility_contracts import request_from_service_spec
from analysis.scenario_contracts_v2 import (
    ScenarioAction, ScenarioBaselineRef, ScenarioOperation, ScenarioSpec,
    preflight_scenario, request_sha256,
)
from core.analysis_spec import (
    AnalysisSpec, PopulationSelector, ServiceAnalysisSpec, ServiceType, TransportMode,
)


H = "a" * 64
G = "b" * 64


class B6C0ContractTests(unittest.TestCase):
    def setUp(self):
        self.analysis_spec = AnalysisSpec(
            city_name="Synthetic", analysis_date="2026-10-10", municipality_code="034027",
            walking_speed_m_s=1.4,
            services=(ServiceAnalysisSpec(
                service_type=ServiceType.PHARMACY,
                modes=(TransportMode.WALK,),
                thresholds_min_by_mode={"walk": (10, 15, 20)},
                population_selector=PopulationSelector.TOTAL,
            ),),
        )
        self.request = request_from_service_spec(
            self.analysis_spec.services[0], TransportMode.WALK,
            travel_time_weight="walking_time_s", off_network_speed_m_s=1.4,
        )
        self.ref = ScenarioBaselineRef(
            municipality_code="034027", analysis_spec_sha256=self.analysis_spec.spec_hash,
            request_sha256=request_sha256(self.request), origins_sha256=H,
            services_sha256=H, attachments_sha256=H, graph_sha256=G,
        )
        self.graph = nx.DiGraph()
        self.graph.add_edge("1", "2", walking_time_s=60.0, length_m=84.0)
        self.origins = pd.DataFrame({
            "origin_id": ["o1"], "municipality_code": ["034027"],
            "geometry": [None], "population_total": [100.0],
            "population_reference_date": ["2023-12-31"], "census_year": ["2023"],
            "geography_version": ["BT2021"], "origin_method": ["representative_point"],
        })
        self.services = pd.DataFrame({
            "service_id": ["s1"], "service_type": ["pharmacy"], "name": ["Test"],
            "municipality_code": ["034027"], "source_name": ["synthetic"],
            "source_record_id": ["1"], "source_reference_date": ["2025-06-30"],
            "retrieved_at": ["2026-10-10T10:00:00+00:00"],
            "operational_status": ["active"],
            "legacy_usable_for_accessibility": [True],
        })
        self.attachments = pd.DataFrame([
            {"entity_id": "o1", "entity_kind": "origin", "mode": "walk",
             "node_id": "1", "snapped": True, "snap_distance_m": 0.0,
             "attachment_quality": "exact", "graph_checksum": G},
            {"entity_id": "s1", "entity_kind": "service", "mode": "walk",
             "node_id": "2", "snapped": True, "snap_distance_m": 0.0,
             "attachment_quality": "exact", "graph_checksum": G},
        ])

    def spec(self, *ops):
        return ScenarioSpec(self.ref, ServiceType.PHARMACY, TransportMode.WALK, tuple(ops))

    def add(self, service_id="new"):
        return ScenarioOperation(ScenarioAction.ADD_SERVICE, service_id, H, H)

    def remove(self, service_id="s1"):
        return ScenarioOperation(ScenarioAction.REMOVE_SERVICE, service_id)

    def check(self, scenario, **overrides):
        params = dict(scenario=scenario, analysis_spec=self.analysis_spec,
                      request=self.request, graph=self.graph, origins=self.origins,
                      services=self.services, attachments=self.attachments)
        params.update(overrides)
        return preflight_scenario(**params)

    def test_valid_remove_is_read_only(self):
        scenario = self.spec(self.remove())
        originals = (self.origins.copy(deep=True), self.services.copy(deep=True),
                     self.attachments.copy(deep=True), self.graph.copy())
        result = self.check(scenario)
        self.assertEqual(result.scenario_id, scenario.scenario_id)
        pd.testing.assert_frame_equal(self.origins, originals[0])
        pd.testing.assert_frame_equal(self.services, originals[1])
        pd.testing.assert_frame_equal(self.attachments, originals[2])
        self.assertEqual(list(self.graph.edges(data=True)), list(originals[3].edges(data=True)))

    def test_valid_add_is_contract_only_not_authorization(self):
        result = self.check(self.spec(self.add()))
        self.assertEqual(result.checked_services, 1)
        self.assertNotIn("new", set(self.services["service_id"]))

    def test_order_does_not_change_identity(self):
        self.assertEqual(self.spec(self.add(), self.remove()).scenario_sha256,
                         self.spec(self.remove(), self.add()).scenario_sha256)

    def test_payload_changes_identity(self):
        a = self.spec(self.add())
        b = self.spec(ScenarioOperation("add_service", "new", "c" * 64, H))
        self.assertNotEqual(a.scenario_sha256, b.scenario_sha256)

    def test_duplicate_or_conflicting_operation_rejected(self):
        with self.assertRaisesRegex(ValueError, "only once"):
            self.spec(self.add("s1"), self.remove("s1"))

    def test_existing_add_rejected(self):
        with self.assertRaisesRegex(ValueError, "existing"):
            self.check(self.spec(self.add("s1")))

    def test_missing_remove_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            self.check(self.spec(self.remove("unknown")))

    def test_unvalidated_service_cannot_be_removed_as_eligible(self):
        services = self.services.copy()
        services.loc[0, "legacy_usable_for_accessibility"] = False
        with self.assertRaisesRegex(ValueError, "ineligible"):
            self.check(self.spec(self.remove()), services=services)

    def test_graph_checksum_mismatch_rejected(self):
        att = self.attachments.copy()
        att.loc[0, "graph_checksum"] = H
        with self.assertRaisesRegex(ValueError, "graph checksums"):
            self.check(self.spec(self.remove()), attachments=att)

    def test_node_not_in_graph_rejected(self):
        att = self.attachments.copy()
        att.loc[1, "node_id"] = "absent"
        with self.assertRaisesRegex(ValueError, "missing from supplied graph"):
            self.check(self.spec(self.remove()), attachments=att)

    def test_request_hash_change_rejected(self):
        request = dataclasses.replace(self.request, off_network_speed_m_s=1.3)
        with self.assertRaisesRegex(ValueError, "Routing request hash"):
            self.check(self.spec(self.remove()), request=request)

    def test_analysisspec_hash_change_rejected(self):
        changed = dataclasses.replace(self.analysis_spec, walking_speed_m_s=0.9)
        with self.assertRaisesRegex(ValueError, "AnalysisSpec hash"):
            self.check(self.spec(self.remove()), analysis_spec=changed)

    def test_wrong_municipality_origin_rejected(self):
        orig = self.origins.copy()
        orig.loc[0, "municipality_code"] = "016024"
        with self.assertRaisesRegex(ValueError, "origins must belong"):
            self.check(self.spec(self.remove()), origins=orig)

    def test_duplicate_baseline_service_rejected(self):
        services = pd.concat([self.services, self.services], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            self.check(self.spec(self.remove()), services=services)

    def test_missing_service_attachment_rejected(self):
        att = self.attachments.iloc[:1].copy()
        with self.assertRaisesRegex(ValueError, "Missing walk NetworkAttachmentV2"):
            self.check(self.spec(self.remove()), attachments=att)

    def test_invalid_hash_and_remove_payload_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            ScenarioBaselineRef("034027", "invalid", H, H, H, H, G)
        with self.assertRaisesRegex(ValueError, "must not carry"):
            ScenarioOperation("remove_service", "s1", H, H)

    def test_empty_operation_list_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.spec()


if __name__ == "__main__":
    unittest.main()
