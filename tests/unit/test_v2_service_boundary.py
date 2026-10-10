"""B7B synthetic boundary audit: no network calls or municipal data required."""
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from analysis.scenario_reporting_v2 import ScenarioInputPaths
from core.analysis_spec import TransportMode
from core.run_manifest import sha256_file
from quality.audit_v2_service_boundary import (
    V2BoundaryPaths, inspect_v2_boundary, paths_for_snapshot,
)
from quality.preflight_real_service_scenario import CitySnapshot
from transformation.build_network_attachments_v2 import graph_checksum
import tests.unit.test_real_service_scenario_preflight as fixtures


class B7BV2BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.B6C3aTests(methodName="test_preflight_does_not_write_anything")
        self.fx.setUp()
        self.addCleanup(self.fx.temp.cleanup)
        self.snapshot = self.fx.snapshot
        self.fx.services["longitude"] = [0.0, 3.0]
        self.fx.services["latitude"] = [0.0, 0.0]
        self.fx.services["crs"] = ["EPSG:4326", "EPSG:4326"]
        self.fx.save()
        self.footprint = Path(self.fx.temp.name) / "footprint.parquet"
        self.areas = gpd.GeoDataFrame(
            {"section": ["s1"]},
            geometry=[Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])],
            crs="EPSG:4326",
        )
        self.save_footprint()
        self.paths = V2BoundaryPaths(self.footprint, self.fx.paths.services,
                                     self.fx.paths.attachments, self.fx.paths.nodes,
                                     self.fx.paths.edges)

    def save_footprint(self):
        # Tests isolate geometry logic without requiring PyArrow in this runtime.
        # Production uses the real GeoParquet census footprint.
        self.footprint.write_bytes(b"synthetic-census-footprint")

    def inspect(self):
        with patch("quality.audit_v2_service_boundary.gpd.read_parquet", return_value=self.areas.copy()):
            return inspect_v2_boundary(self.snapshot, self.paths)

    def test_valid_census_boundary_audit(self):
        data = self.inspect()
        self.assertEqual(data["routable_services"], 1)
        self.assertEqual(data["outside_not_routable"], 1)
        self.assertFalse(data["administrative_boundary_verified"])
        self.assertIn("additions_not_audited", data["scenario_coverage"])

    def test_read_only_inputs_preserved(self):
        before = {k: sha256_file(v) for k, v in self.paths.items().items()}
        self.inspect()
        self.assertEqual(before, {k: sha256_file(v) for k, v in self.paths.items().items()})

    def test_routable_outside_fails_closed(self):
        self.fx.services.loc[0, "longitude"] = 3.0
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "territorial routing leak"):
            self.inspect()

    def test_routable_missing_coordinate_fails_closed(self):
        self.fx.services.loc[0, "longitude"] = None
        self.fx.services.loc[0, "latitude"] = None
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "territorial routing leak"):
            self.inspect()

    def test_boundary_point_is_inside(self):
        self.fx.services.loc[0, "longitude"] = 1.0
        self.fx.save()
        self.assertEqual(self.inspect()["routable_services"], 1)

    def test_false_validation_flag_does_not_promote_outside(self):
        self.fx.services.loc[1, "legacy_usable_for_accessibility"] = False
        self.fx.save()
        self.assertEqual(self.inspect()["outside_not_routable"], 1)

    def test_rejects_foreign_service_municipality(self):
        self.fx.services.loc[1, "municipality_code"] = "016024"
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "municipality code"):
            self.inspect()

    def test_rejects_missing_or_extra_attachments(self):
        self.fx.attachments = self.fx.attachments.loc[self.fx.attachments["entity_id"] != "s1"]
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Missing walk NetworkAttachmentV2"):
            self.inspect()

        self.fx.attachments = pd.concat([self.fx.attachments, pd.DataFrame([{
            "entity_id": "s1", "entity_kind": "service", "mode": "walk", "node_id": "3",
            "snapped": True, "snap_distance_m": 0., "attachment_quality": "exact",
            "graph_checksum": graph_checksum(self.paths.nodes, self.paths.edges),
        }, {
            "entity_id": "extra", "entity_kind": "service", "mode": "walk", "node_id": "1",
            "snapped": True, "snap_distance_m": 0., "attachment_quality": "exact",
            "graph_checksum": graph_checksum(self.paths.nodes, self.paths.edges),
        }])], ignore_index=True)
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Unknown service entity_id"):
            self.inspect()

    def test_rejects_graph_change(self):
        self.fx.edges.loc[0, "length_m"] = 999.
        self.fx.edges.to_csv(self.paths.edges, index=False)
        with self.assertRaisesRegex(ValueError, "graph checksum"):
            self.inspect()

    def test_rejects_missing_boundary(self):
        self.footprint.unlink()
        with self.assertRaises(FileNotFoundError):
            self.inspect()

    def test_rejects_invalid_crs(self):
        self.fx.services.loc[0, "crs"] = "EPSG:3857"
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "EPSG:4326"):
            self.inspect()

    def test_rejects_invalid_coordinate(self):
        self.fx.services.loc[0, "latitude"] = 91.0
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.inspect()

    def test_rejects_bad_boundary_geometry(self):
        self.areas = gpd.GeoDataFrame({"section": ["s1"]},
                                      geometry=[Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])],
                                      crs="EPSG:4326")
        self.save_footprint()
        with self.assertRaisesRegex(ValueError, "invalid geometries"):
            self.inspect()

    def test_counts_multiple_categories_without_promoting(self):
        other = self.fx.services.iloc[[0]].copy()
        other.loc[:, "service_id"] = "school"
        other.loc[:, "service_type"] = "primary_school"
        other.loc[:, "source_record_id"] = "newschool"
        self.fx.services = pd.concat([self.fx.services, other], ignore_index=True)
        other_att = self.fx.attachments.loc[
            (self.fx.attachments["entity_kind"] == "service")
            & (self.fx.attachments["entity_id"] == "s1")
        ].copy()
        other_att.loc[:, "entity_id"] = "school"
        self.fx.attachments = pd.concat([self.fx.attachments, other_att], ignore_index=True)
        self.fx.save()
        data = self.inspect()
        self.assertEqual(data["routable_services"], 2)
        self.assertEqual(data["service_types"]["primary_school"]["routable"], 1)

    def test_path_conventions(self):
        paths = paths_for_snapshot(Path("/repo"), self.snapshot)
        self.assertEqual(paths.footprint.name, "034027_census_areas_2021.parquet")
        self.assertEqual(paths.services.name, "service_entities_v2_202425_20250630.parquet")
        self.assertEqual(paths.attachments.name, "network_attachments_v2_walk_2023_202425_20250630.parquet")


if __name__ == "__main__":
    unittest.main()
