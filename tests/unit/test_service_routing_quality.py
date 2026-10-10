"""Read-only synthetic checks for service quality attrition audit."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from core.analysis_spec import TransportMode
from core.run_manifest import sha256_file
from quality.derive_boundary_qualified_services import qualify_v2_services
from quality.audit_service_routing_quality import (
    summarize_service_quality, inspect_service_quality,
)
from quality.audit_v2_service_boundary import paths_for_snapshot
from quality.preflight_real_service_scenario import CitySnapshot
from transformation.build_network_attachments_v2 import graph_checksum


class ServiceQualityAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = CitySnapshot("034027")
        self.paths = paths_for_snapshot(self.root, self.snapshot)
        for p in self.paths.items().values():
            p.parent.mkdir(parents=True, exist_ok=True)
        self.footprint = gpd.GeoDataFrame(
            {"section": ["1"]},
            geometry=[Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])],
            crs="EPSG:4326",
        )
        self.services = pd.DataFrame({
            "service_id": ["a", "b", "c", "d", "e", "f"],
            "service_type": ["pharmacy"] * 5 + ["primary_school"],
            "name": list("abcdef"),
            "municipality_code": ["034027"] * 6,
            "source_name": ["Health"] * 5 + ["MIM"],
            "source_record_id": list("abcdef"),
            "source_reference_date": ["2025-06-30"] * 5 + ["2024-09-01"],
            "retrieved_at": ["2026-10-10"] * 6,
            "operational_status": ["active", "active", "active", "active", "inactive", "active"],
            "legacy_usable_for_accessibility": [True, False, True, False, True, True],
            "latitude": [0., 0., 0., None, 0., 0.],
            "longitude": [0., .2, 3., None, 0., .5],
            "crs": ["EPSG:4326"] * 6,
        })
        self.nodes = pd.DataFrame({"source_record_id": ["1", "2"]})
        self.edges = pd.DataFrame({"source_osm_node": ["1"], "target_osm_node": ["2"],
                                   "length_m": [50.], "walking_time_s": [40.]})
        self.nodes.to_csv(self.paths.nodes, index=False)
        self.edges.to_csv(self.paths.edges, index=False)
        sha = graph_checksum(self.paths.nodes, self.paths.edges)
        self.attachments = pd.DataFrame([
            dict(entity_id=svc, entity_kind="service", mode="walk",
                 node_id=None if svc in ("d", "f") else "1",
                 snapped=svc not in ("d", "f"), snap_distance_m=0. if svc not in ("d", "f") else None,
                 graph_checksum=sha, attachment_quality="synthetic")
            for svc in list("abcdef")
        ])
        self.save()

    def save(self):
        self.services.to_csv(self.paths.services, index=False)
        self.attachments.to_csv(self.paths.attachments, index=False)
        # Synthetic tests mock the footprint reader; file exists to test hashes.
        self.paths.footprint.write_bytes(b"synthetic-geography")

    def qualified(self):
        return qualify_v2_services(
            self.services, self.footprint, municipality_code="034027",
            canonical_sha256="a" * 64, footprint_sha256="b" * 64,
        )

    def inspect(self):
        def read_synthetic(path):
            for name, frame in (("services", self.services), ("attachments", self.attachments),
                                ("nodes", self.nodes), ("edges", self.edges)):
                if Path(path) == getattr(self.paths, name):
                    return frame.copy(deep=True)
            raise AssertionError(f"Unexpected synthetic input: {path}")

        with patch("quality.derive_boundary_qualified_services.gpd.read_parquet",
                   return_value=self.footprint), \
             patch("quality.derive_boundary_qualified_services._read_table", side_effect=read_synthetic), \
             patch("quality.audit_service_routing_quality._read_table", side_effect=read_synthetic):
            return inspect_service_quality(self.snapshot, root=self.root)

    def test_breakdowns_are_orthogonal_and_conservative(self):
        result = summarize_service_quality(self.qualified(), self.attachments, mode=TransportMode.WALK)
        self.assertEqual(result["services_total"], 6)
        self.assertEqual(result["services_routable"], 1)
        pharmacy = result["service_types"]["pharmacy"]
        self.assertEqual(pharmacy["total"], 5)
        self.assertEqual(pharmacy["routable"], 1)
        self.assertEqual(pharmacy["primary_routing_reason"], {
            "eligible": 1, "location_not_validated": 2,
            "outside_operational_footprint": 1, "service_not_active": 1,
        })
        self.assertEqual(pharmacy["geography"], {
            "coordinates_missing": 1, "inside_census_footprint": 3,
            "outside_census_footprint": 1,
        })
        self.assertEqual(pharmacy["overlapping_issues"]["not_snapped"], 1)
        self.assertEqual(pharmacy["overlapping_issues"]["location_validation_not_true"], 2)
        self.assertEqual(result["service_types"]["primary_school"]["primary_routing_reason"],
                         {"not_snapped": 1})

    def test_canonical_input_is_not_mutated(self):
        original = self.qualified()
        copy = original.copy(deep=True)
        original_att = self.attachments.copy(deep=True)
        summarize_service_quality(original, self.attachments, mode=TransportMode.WALK)
        pd.testing.assert_frame_equal(original, copy)
        pd.testing.assert_frame_equal(self.attachments, original_att)

    def test_missing_coords_are_not_outside(self):
        result = summarize_service_quality(self.qualified(), self.attachments, mode=TransportMode.WALK)
        self.assertEqual(result["geography"]["coordinates_missing"], 1)
        self.assertEqual(result["geography"]["outside_census_footprint"], 1)

    def test_invalid_partial_coords_fail_closed(self):
        self.services.loc[3, "latitude"] = 0.0
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.qualified()

    def test_incomplete_attachment_is_rejected(self):
        att = self.attachments.loc[self.attachments["entity_id"] != "a"].copy()
        with self.assertRaisesRegex(ValueError, "Missing walk"):
            summarize_service_quality(self.qualified(), att, mode=TransportMode.WALK)

    def test_no_geographic_self_attestation(self):
        raw = self.services.copy()
        raw["inside_operational_footprint"] = True
        with self.assertRaisesRegex(ValueError, "already boundary-qualified"):
            qualify_v2_services(raw, self.footprint, municipality_code="034027",
                                canonical_sha256="a" * 64, footprint_sha256="b" * 64)

    def test_real_path_integration_hashes_and_read_only(self):
        before = {k: sha256_file(v) for k, v in self.paths.items().items()}
        result = self.inspect()
        self.assertEqual(result["services_total"], 6)
        self.assertEqual(result["services_routable"], 1)
        self.assertEqual(result["input_sha256"], before)
        self.assertFalse(result["administrative_boundary_verified"])
        self.assertEqual({k: sha256_file(v) for k, v in self.paths.items().items()}, before)

    def test_graph_checksum_mismatch_rejected(self):
        self.edges["walking_time_s"] = [999.]
        self.edges.to_csv(self.paths.edges, index=False)
        with self.assertRaisesRegex(ValueError, "graph checksum"):
            self.inspect()

    def test_orphan_graph_node_rejected(self):
        self.attachments.loc[0, "node_id"] = "99999"
        self.save()
        with self.assertRaisesRegex(ValueError, "nonexistent network node"):
            self.inspect()

    def test_wrong_municipality_rejected(self):
        self.services.loc[0, "municipality_code"] = "016024"
        self.save()
        with self.assertRaisesRegex(ValueError, "municipality code mismatch"):
            self.inspect()

    def test_missing_file_fails_closed(self):
        self.paths.services.unlink()
        with self.assertRaises(FileNotFoundError):
            self.inspect()


if __name__ == "__main__":
    unittest.main()
