"""B6C.1 pure synthetic overlays: no local data, database or file writes."""
import copy
import unittest
from dataclasses import replace

import networkx as nx
import pandas as pd

from analysis.accessibility_contracts import request_from_service_spec
from analysis.scenario_contracts_v2 import (
    ScenarioAction, ScenarioBaselineRef, ScenarioOperation, ScenarioSpec, request_sha256,
)
from analysis.service_scenarios_v2 import (
    execute_service_scenario, scenario_row_sha256,
)
from core.analysis_spec import (
    AnalysisSpec, PopulationSelector, ServiceAnalysisSpec, ServiceType, TransportMode,
)

H, G = "a" * 64, "b" * 64


class B6C1ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.spec = AnalysisSpec(
            city_name="Synthetic", analysis_date="2026-10-10",
            municipality_code="034027", walking_speed_m_s=1.4,
            services=(ServiceAnalysisSpec(
                service_type=ServiceType.PHARMACY, modes=(TransportMode.WALK,),
                thresholds_min_by_mode={"walk": (10, 15, 20)},
                population_selector=PopulationSelector.TOTAL,
            ),),
        )
        self.request = request_from_service_spec(
            self.spec.services[0], TransportMode.WALK,
            travel_time_weight="walking_time_s", off_network_speed_m_s=1.4,
        )
        self.ref = ScenarioBaselineRef(
            municipality_code="034027", analysis_spec_sha256=self.spec.spec_hash,
            request_sha256=request_sha256(self.request), origins_sha256=H,
            services_sha256=H, attachments_sha256=H, graph_sha256=G,
        )
        self.hashes = dict(origins=H, services=H, attachments=H, graph=G)
        self.graph = nx.DiGraph()
        self.graph.add_edge("1", "2", walking_time_s=600.0, length_m=840.0)
        self.graph.add_edge("2", "3", walking_time_s=600.0, length_m=840.0)
        self.origins = pd.DataFrame({
            "origin_id": ["o1", "o2"], "municipality_code": ["034027", "034027"],
            "geometry": [None, None], "population_total": [100.0, 50.0],
            "population_reference_date": ["2023-12-31", "2023-12-31"],
            "census_year": ["2023", "2023"], "geography_version": ["BT2021", "BT2021"],
            "origin_method": ["representative_point", "representative_point"],
        })
        self.services = pd.DataFrame({
            "service_id": ["s1"], "service_type": ["pharmacy"], "name": ["Existing"],
            "municipality_code": ["034027"], "source_name": ["trusted"],
            "source_record_id": ["1"], "source_reference_date": ["2025-06-30"],
            "retrieved_at": ["2026-10-10T10:00:00+00:00"],
            "operational_status": ["active"],
            "legacy_usable_for_accessibility": [True],
        })
        self.attachments = pd.DataFrame([
            dict(entity_id="o1", entity_kind="origin", mode="walk", node_id="1",
                 snapped=True, snap_distance_m=0.0, attachment_quality="exact", graph_checksum=G),
            dict(entity_id="o2", entity_kind="origin", mode="walk", node_id="2",
                 snapped=True, snap_distance_m=0.0, attachment_quality="exact", graph_checksum=G),
            dict(entity_id="s1", entity_kind="service", mode="walk", node_id="3",
                 snapped=True, snap_distance_m=0.0, attachment_quality="exact", graph_checksum=G),
        ])
        self.new_service = self.services.copy(deep=True)
        self.new_service["service_id"] = "s2"
        self.new_service["name"] = "Candidate"
        self.new_service["source_record_id"] = "2"
        self.new_attachment = self.attachments.iloc[2:].copy(deep=True)
        self.new_attachment["entity_id"] = "s2"
        self.new_attachment["node_id"] = "1"

    def add(self):
        return ScenarioOperation(
            ScenarioAction.ADD_SERVICE, "s2",
            scenario_row_sha256(self.new_service.iloc[0]),
            scenario_row_sha256(self.new_attachment.iloc[0]),
        )

    def remove(self):
        return ScenarioOperation(ScenarioAction.REMOVE_SERVICE, "s1")

    def execute(self, *ops, additions=True, **kwargs):
        params = dict(
            scenario=ScenarioSpec(self.ref, ServiceType.PHARMACY, TransportMode.WALK, ops),
            analysis_spec=self.spec, request=self.request, graph=self.graph,
            origins=self.origins, services=self.services, attachments=self.attachments,
            verified_input_sha256=self.hashes,
        )
        if additions:
            params["addition_services"] = self.new_service if any(o.action is ScenarioAction.ADD_SERVICE for o in ops) else None
            params["addition_attachments"] = self.new_attachment if any(o.action is ScenarioAction.ADD_SERVICE for o in ops) else None
        params.update(kwargs)
        return execute_service_scenario(**params)

    def test_add_improves_coverage_and_mean_time(self):
        result = self.execute(self.add())
        self.assertAlmostEqual(result.metrics["baseline"]["population_coverage_within_10_min"], 1/3)
        self.assertEqual(result.metrics["scenario"]["population_coverage_within_10_min"], 1.0)
        self.assertAlmostEqual(result.metrics["delta"]["population_coverage_within_10_min"], 2/3)
        self.assertAlmostEqual(result.metrics["baseline"]["mean_nearest_time_reachable_min"], 100/6)
        self.assertAlmostEqual(result.metrics["scenario"]["mean_nearest_time_reachable_min"], 10/3)
        changes = result.per_origin.set_index("origin_id")
        self.assertAlmostEqual(changes.loc["o1", "delta_nearest_time_min"], -20.0)
        self.assertEqual(changes.loc["o1", "delta_opportunity_count_within_10_min"], 1)
        self.assertEqual(result.metrics["scenario"]["service_count_routable"], 2)

    def test_remove_all_services_preserves_origins_and_zero_opportunities(self):
        result = self.execute(self.remove())
        self.assertEqual(result.metrics["scenario"]["service_count_routable"], 0)
        self.assertEqual(result.metrics["scenario"]["reachable_population"], 0)
        self.assertIsNone(result.metrics["scenario"]["mean_nearest_time_reachable_min"])
        self.assertIsNone(result.metrics["delta"]["mean_nearest_time_reachable_min"])
        self.assertTrue(result.per_origin["scenario_nearest_time_min"].isna().all())
        self.assertTrue((result.per_origin["delta_opportunity_count_within_20_min"] <= 0).all())

    def test_mixed_add_remove_still_computes_paired_results(self):
        result = self.execute(self.remove(), self.add())
        self.assertEqual(result.metrics["scenario"]["service_count_total"], 1)
        self.assertAlmostEqual(result.metrics["scenario"]["population_coverage_within_10_min"], 2/3)

    def test_inputs_are_immutable(self):
        saved = [df.copy(deep=True) for df in
                 (self.origins, self.services, self.attachments, self.new_service, self.new_attachment)]
        before_graph = copy.deepcopy(self.graph)
        self.execute(self.add())
        for a, b in zip((self.origins, self.services, self.attachments,
                         self.new_service, self.new_attachment), saved):
            pd.testing.assert_frame_equal(a, b)
        self.assertEqual(list(self.graph.edges(data=True)), list(before_graph.edges(data=True)))

    def test_row_hash_order_independent_sensitive_to_edit(self):
        row = self.new_service.iloc[0]
        self.assertEqual(scenario_row_sha256(row), scenario_row_sha256(row.iloc[::-1]))
        modified = row.copy()
        modified["name"] = "Tampered"
        self.assertNotEqual(scenario_row_sha256(row), scenario_row_sha256(modified))

    def test_reject_tampered_candidate_even_if_status_stays_valid(self):
        candidate = self.new_service.copy()
        candidate.loc[0, "name"] = "Altered"
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            self.execute(self.add(), addition_services=candidate)

    def test_reject_unvalidated_addition(self):
        self.new_service.loc[0, "legacy_usable_for_accessibility"] = False
        with self.assertRaisesRegex(ValueError, "not independently location-validated"):
            self.execute(self.add())

    def test_reject_inactive_addition(self):
        self.new_service.loc[0, "operational_status"] = "planned"
        with self.assertRaisesRegex(ValueError, "not independently location-validated"):
            self.execute(self.add())

    def test_reject_unmatched_mode_or_invalid_node(self):
        self.new_attachment.loc[self.new_attachment.index[0], "node_id"] = "999"
        with self.assertRaisesRegex(ValueError, "node missing from graph"):
            self.execute(self.add())
        self.new_attachment.loc[self.new_attachment.index[0], "mode"] = "drive"
        with self.assertRaisesRegex(ValueError, "scenario mode"):
            self.execute(self.add())

    def test_reject_wrong_municipality_or_type(self):
        self.new_service.loc[0, "municipality_code"] = "016024"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.execute(self.add())
        self.new_service.loc[0, "municipality_code"] = "034027"
        self.new_service.loc[0, "service_type"] = "hospital_establishment"
        with self.assertRaisesRegex(ValueError, "type differs"):
            self.execute(self.add())

    def test_refuse_missing_or_wrong_hash_attestation(self):
        with self.assertRaisesRegex(ValueError, "Verified baseline input hashes"):
            self.execute(self.remove(), verified_input_sha256=None)
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            self.execute(self.remove(), verified_input_sha256=dict(self.hashes, graph=H))

    def test_reject_payload_with_unclaimed_extra_rows(self):
        extra = pd.concat([self.new_service, self.new_service.assign(service_id="s3")], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "does not match declared"):
            self.execute(self.add(), addition_services=extra)

    def test_reject_same_source_identity_under_new_canonical_id(self):
        self.new_service.loc[0, "source_record_id"] = "1"
        with self.assertRaisesRegex(ValueError, "duplicates an existing canonical source identity"):
            self.execute(self.add())

    def test_output_hashes_reproducible_and_operation_order_independent(self):
        a = self.execute(self.add(), self.remove())
        b = self.execute(self.remove(), self.add())
        self.assertEqual(a.scenario_id, b.scenario_id)
        self.assertEqual(a.provenance["output_sha256"], b.provenance["output_sha256"])
        self.assertEqual(a.provenance["input_checksum_verification"], "caller_responsibility_not_rechecked")

    def test_reject_spec_or_routing_request_drift(self):
        wrong_request = replace(self.request, off_network_speed_m_s=0.9)
        with self.assertRaisesRegex(ValueError, "Routing request hash"):
            self.execute(self.remove(), request=wrong_request)

    def test_no_additions_allowed_without_operation(self):
        with self.assertRaisesRegex(ValueError, "Unexpected service addition payload"):
            self.execute(self.remove(), addition_services=self.new_service)


if __name__ == "__main__":
    unittest.main()
