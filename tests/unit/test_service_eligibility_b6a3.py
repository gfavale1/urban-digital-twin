"""B6A3 fail-closed canonical routing regression tests."""
from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from argparse import Namespace
import json

import networkx as nx
import pandas as pd

from analysis.accessibility_contracts import (
    prepare_canonical_accessibility_inputs,
    prepare_service_destinations_v2,
    compute_accessibility_from_canonical,
)
from analysis.accessibility_engine import AccessibilityEngineRequest
from core.analysis_spec import PopulationSelector, ServiceType, TransportMode
from quality.audit_service_eligibility_v2 import audit_destinations, run


def service(service_id, *, verified=True, status="active", service_type="pharmacy"):
    return {
        "service_id": service_id, "service_type": service_type, "name": service_id,
        "municipality_code": "034027", "source_name": "fixture", "source_record_id": service_id,
        "source_reference_date": "2025-06-30", "retrieved_at": None,
        "operational_status": status, "legacy_usable_for_accessibility": verified,
    }


def attach(entity_id, *, kind="service", node="B", snapped=True, mode="walk", component="component_000001"):
    return {
        "entity_id": entity_id, "entity_kind": kind, "mode": mode,
        "node_id": node if snapped else None, "snapped": bool(snapped),
        "snap_distance_m": 10.0 if snapped else None,
        "attachment_quality": "snapped" if snapped else "missing_geometry",
        "network_component_id": component, "is_largest_component": component == "component_000001",
        "graph_checksum": "sha-synthetic",
    }


def origin():
    return {
        "origin_id": "O1", "municipality_code": "034027", "geometry": None,
        "population_total": 100, "population_reference_date": "2023-01-01",
        "census_year": "2023", "geography_version": "test", "origin_method": "test",
    }


def request(mode="walk"):
    return AccessibilityEngineRequest(
        service_type=ServiceType("pharmacy"), mode=TransportMode(mode),
        thresholds_min=(5, 10), population_selector=PopulationSelector("population_total"),
        travel_time_weight="travel_time_s", off_network_speed_m_s=1.0,
        distance_weight="length_m",
    )


class B6A3RoutingGateTests(unittest.TestCase):
    def test_unsafe_snapped_service_still_has_attachment_but_no_routing_node(self):
        items = pd.DataFrame([service("ZANETTI", verified=False)])
        att = pd.DataFrame([attach("ZANETTI", node="A")])
        result = prepare_service_destinations_v2(items, att, mode=TransportMode.WALK)
        row = result.iloc[0]
        self.assertTrue(row["snapped"])
        self.assertEqual(row["attachment_node_id"], "A")
        self.assertFalse(row["routing_eligible"])
        self.assertEqual(row["routing_exclusion_reason"], "location_not_validated")
        self.assertIsNone(row["network_node_id"])
        self.assertTrue(pd.isna(row["snap_distance_m"]))
        self.assertEqual(att.iloc[0]["node_id"], "A")

    def test_missing_eligibility_evidence_is_not_accepted(self):
        item = service("S1")
        item.pop("legacy_usable_for_accessibility")
        result = prepare_service_destinations_v2(pd.DataFrame([item]), pd.DataFrame([attach("S1")]), mode=TransportMode.WALK)
        self.assertFalse(result.iloc[0]["routing_eligible"])
        self.assertEqual(result.iloc[0]["routing_exclusion_reason"], "missing_validation_evidence")

    def test_null_eligibility_evidence_is_not_accepted(self):
        result = prepare_service_destinations_v2(pd.DataFrame([service("S1", verified=None)]), pd.DataFrame([attach("S1")]), mode=TransportMode.WALK)
        self.assertEqual(result.iloc[0]["routing_exclusion_reason"], "missing_validation_evidence")

    def test_string_false_is_rejected_rather_than_coerced_to_true(self):
        with self.assertRaisesRegex(ValueError, "boolean or null"):
            prepare_service_destinations_v2(pd.DataFrame([service("S1", verified="False")]), pd.DataFrame([attach("S1")]), mode=TransportMode.WALK)

    def test_inactive_planned_service_is_not_eligible(self):
        result = prepare_service_destinations_v2(pd.DataFrame([service("S1", status="planned")]), pd.DataFrame([attach("S1")]), mode=TransportMode.WALK)
        self.assertEqual(result.iloc[0]["routing_exclusion_reason"], "service_not_active")

    def test_unknown_operational_status_fails_closed(self):
        result = prepare_service_destinations_v2(
            pd.DataFrame([service("S1", status=None)]),
            pd.DataFrame([attach("S1")]), mode=TransportMode.WALK,
        )
        self.assertEqual(result.iloc[0]["routing_exclusion_reason"], "service_not_active")

    def test_parma_count_shape_for_snap_without_validation(self):
        # The numbers are the externally audited Parma totals; synthetic
        # fixtures avoid tying a unit test to private raw data or its paths.
        rows = [service(f"VALID_{i}") for i in range(118)]
        rows += [service(f"UNVERIFIED_{i}", verified=False) for i in range(78)]
        att = [attach(f"VALID_{i}") for i in range(118)]
        att += [attach(f"UNVERIFIED_{i}", snapped=i < 13) for i in range(78)]
        evidence, report = audit_destinations(pd.DataFrame(rows), pd.DataFrame(att), mode=TransportMode.WALK)
        self.assertEqual(report["total_services"], 196)
        self.assertEqual(report["attachment_snapped"], 131)  # 118 validated + 13 unverified
        self.assertEqual(report["snapped_not_validated"], 13)
        self.assertEqual(report["eligible_destinations"], 118)

    def test_unsnapped_valid_service_is_not_eligible(self):
        result = prepare_service_destinations_v2(pd.DataFrame([service("S1")]), pd.DataFrame([attach("S1", snapped=False)]), mode=TransportMode.WALK)
        self.assertEqual(result.iloc[0]["routing_exclusion_reason"], "not_snapped")

    def test_valid_service_outside_largest_component_is_not_excluded(self):
        result = prepare_service_destinations_v2(pd.DataFrame([service("S1")]), pd.DataFrame([attach("S1", component="component_000134")]), mode=TransportMode.WALK)
        self.assertTrue(result.iloc[0]["routing_eligible"])
        self.assertEqual(result.iloc[0]["network_node_id"], "B")

    def test_missing_attachment_fails_closed_with_error(self):
        with self.assertRaisesRegex(ValueError, "Missing walk NetworkAttachmentV2"):
            prepare_service_destinations_v2(pd.DataFrame([service("S1")]), pd.DataFrame([attach("S2")]), mode=TransportMode.WALK)

    def test_duplicate_attachment_fails(self):
        with self.assertRaisesRegex(ValueError, "at most one attachment"):
            prepare_service_destinations_v2(pd.DataFrame([service("S1")]), pd.DataFrame([attach("S1"), attach("S1")]), mode=TransportMode.WALK)

    def test_mode_specific(self):
        attachments = pd.DataFrame([attach("S1", mode="walk", node="B"), attach("S1", mode="drive", node="D")])
        result = prepare_service_destinations_v2(pd.DataFrame([service("S1")]), attachments, mode=TransportMode.DRIVE)
        self.assertEqual(result.iloc[0]["network_node_id"], "D")

    def test_legacy_unsafe_cannot_influence_accessibility(self):
        g = nx.DiGraph()
        g.add_edge("A", "B", travel_time_s=60.0, length_m=100.0)
        services = pd.DataFrame([service("GOOD"), service("BAD", verified=False)])
        origins = pd.DataFrame([origin()])
        attachments = pd.DataFrame([
            attach("O1", kind="origin", node="A"),
            attach("GOOD", node="B"),
            attach("BAD", node="A"),
        ])
        result = compute_accessibility_from_canonical(g, origins, services, attachments, request())
        self.assertEqual(result.origins.iloc[0]["nearest_service_id"], "GOOD")
        self.assertEqual(result.summary["service_count_routable"], 1)
        self.assertEqual(result.origins.iloc[0]["opportunity_count_within_5_min"], 1)

    def test_directed_network_reachability_preserved(self):
        g = nx.DiGraph()
        g.add_edge("B", "A", travel_time_s=60.0, length_m=100.0)
        origins = pd.DataFrame([origin()])
        services = pd.DataFrame([service("GOOD")])
        attachments = pd.DataFrame([attach("O1", kind="origin", node="A"), attach("GOOD", node="B")])
        result = compute_accessibility_from_canonical(g, origins, services, attachments, request())
        self.assertNotEqual(result.origins.iloc[0]["reachability_status"], "reachable")

    def test_pure_audit_aggregates_excluded_attachments(self):
        services = pd.DataFrame([service("S1"), service("S2", verified=False), service("S3", verified=False)])
        attachments = pd.DataFrame([attach("S1"), attach("S2"), attach("S3", snapped=False)])
        evidence, report = audit_destinations(services, attachments, mode=TransportMode.WALK)
        self.assertEqual(report["eligible_destinations"], 1)
        self.assertEqual(report["attachment_snapped"], 2)
        self.assertEqual(report["snapped_not_validated"], 1)
        self.assertEqual(len(evidence), 3)
        self.assertEqual(report["eligible_by_service_type"], {"pharmacy": 1})

    def test_auditor_dry_run_creates_no_artifacts(self):
        items = pd.DataFrame([service("S1")])
        att = pd.DataFrame([attach("S1")])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "data" / "processed"
            sp = base / "services" / "034027" / "service_entities_v2_202425_20250630.parquet"
            ap = base / "network_attachments" / "034027" / "network_attachments_v2_walk_2023_202425_20250630.parquet"
            for path in (sp, ap):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic parquet placeholder")
            args = Namespace(municipality_code="034027", mode="walk", census_year="2023", school_year="202425", health_reference_date="20250630", dry_run=True)
            with patch("quality.audit_service_eligibility_v2.pd.read_parquet", side_effect=[items, att]):
                summary = run(args, root=root)
            self.assertEqual(summary["eligible_destinations"], 1)
            self.assertFalse((root / "data" / "features").exists())

    def test_auditor_writes_versioned_report_and_checksums(self):
        items = pd.DataFrame([service("GOOD"), service("BAD", verified=False)])
        att = pd.DataFrame([attach("GOOD"), attach("BAD")])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "data" / "processed"
            sp = base / "services" / "034027" / "service_entities_v2_202425_20250630.parquet"
            ap = base / "network_attachments" / "034027" / "network_attachments_v2_drive_2023_202425_20250630.parquet"
            for path in (sp, ap):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic parquet placeholder")
            att["mode"] = "drive"
            args = Namespace(municipality_code="034027", mode="drive", census_year="2023", school_year="202425", health_reference_date="20250630", dry_run=False)
            with patch("quality.audit_service_eligibility_v2.pd.read_parquet", side_effect=[items, att]):
                report = run(args, root=root)
            out = root / "data" / "features" / "quality" / "034027"
            csv_files = list(out.glob("*.csv"))
            json_files = list(out.glob("*.json"))
            self.assertEqual((len(csv_files), len(json_files)), (1, 1))
            self.assertEqual(report["snapped_not_validated"], 1)
            written = json.loads(json_files[0].read_text(encoding="utf-8"))
            self.assertEqual(written["eligibility_policy"], "B6A3_migrated_verified_location_v1")
            self.assertEqual(len(written["input_sha256"]["services"]), 64)
            self.assertEqual(len(written["output_csv_sha256"]), 64)

    def test_canonical_interface_rejects_missing_origin_attachment(self):
        with self.assertRaisesRegex(ValueError, "Missing walk NetworkAttachmentV2"):
            prepare_canonical_accessibility_inputs(pd.DataFrame([origin()]), pd.DataFrame([service("S1")]), pd.DataFrame([attach("S1")]), mode=TransportMode.WALK)


if __name__ == "__main__":
    unittest.main()
