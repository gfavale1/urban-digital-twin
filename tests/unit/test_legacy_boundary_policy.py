"""B7A boundary-policy reconciliation with synthetic GeoParquet fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import Point, box

from quality.audit_legacy_boundary_policy import inspect_boundary_policy, paths_for

# Unit-test geometry and checksum logic without requiring the optional pyarrow
# package in all test environments. Real GeoParquet I/O is exercised by CLI.


class B7ABoundaryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        import pandas as pd
        stub = patch("quality.audit_legacy_boundary_policy.gpd.read_parquet", side_effect=pd.read_pickle)
        stub.start()
        self.addCleanup(stub.stop)
        self.code = "034027"
        self.paths = paths_for(self.root, self.code)
        for path in self.paths.items().values():
            path.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame({"section": ["A"], "geometry": [box(0, 0, 2, 2)]},
                         geometry="geometry", crs="EPSG:4326").to_pickle(self.paths.footprint)
        # boundary point at x=0 must be included by covers(); null is outside.
        for key in ("enriched", "osm_only"):
            data = gpd.GeoDataFrame({
                "service_site_id": [f"{key}_{i}" for i in range(4)],
                "municipality_code": [self.code] * 4,
                "usable_before_boundary_filter": [True, True, True, False],
                "inside_municipality": [True, False, True, False],
                "excluded_by_municipality_boundary": [False, True, False, False],
                "usable_for_accessibility": [True, False, True, False],
                "network_node_id": ["12", None, "31", None],
                "geometry": [Point(1, 1), Point(3, 3), Point(0, 1), None],
            }, geometry="geometry", crs="EPSG:4326")
            data.to_pickle(getattr(self.paths, key))
            summary = {
                "boundary_source": str(self.paths.footprint),
                "policy": "municipality_only_supply; points exactly on the boundary are included via covers()",
                "service_sites_total": 4, "inside_municipality": 2,
                "outside_municipality": 2, "usable_before_boundary_filter": 3,
                "usable_excluded_outside_municipality": 1,
                "usable_after_boundary_filter": 2,
            }
            manifest = {"municipality_code": self.code, "service_layer": key,
                        "municipality_boundary_filter": summary,
                        "service_sites_total": 4, "service_sites_usable": 2}
            getattr(self.paths, f"{key}_manifest").write_text(json.dumps(manifest))

    def _edit_data(self, key, column, index, value):
        path = getattr(self.paths, key)
        import pandas as pd
        data = pd.read_pickle(path)
        if isinstance(value, str) and column in ("usable_for_accessibility",):
            data[column] = data[column].astype(object)
        data.loc[index, column] = value
        data.to_pickle(path)

    def _edit_manifest(self, key, field, value, *, nested=False):
        path = getattr(self.paths, f"{key}_manifest")
        doc = json.loads(path.read_text())
        if nested:
            doc["municipality_boundary_filter"][field] = value
        else:
            doc[field] = value
        path.write_text(json.dumps(doc))

    def test_passes_both_layers_and_boundary_point(self):
        report = inspect_boundary_policy(self.paths, self.code)
        self.assertEqual(report["layers"]["enriched"]["usable_after_boundary_filter"], 2)
        self.assertEqual(report["layers"]["osm_only"]["usable_excluded_outside_municipality"], 1)
        self.assertFalse(report["administrative_boundary_verified"])
        self.assertIn("not_administrative_boundary", report["boundary_basis"])

    def test_no_file_writes(self):
        before = {key: p.read_bytes() for key, p in self.paths.items().items()}
        inspect_boundary_policy(self.paths, self.code)
        self.assertEqual(before, {key: p.read_bytes() for key, p in self.paths.items().items()})

    def test_outside_routing_leak_rejected(self):
        self._edit_data("enriched", "usable_for_accessibility", 1, True)
        with self.assertRaisesRegex(ValueError, "leaked into routing supply"):
            inspect_boundary_policy(self.paths, self.code)

    def test_inside_flag_not_matching_geometry_rejected(self):
        self._edit_data("osm_only", "inside_municipality", 2, False)
        with self.assertRaisesRegex(ValueError, "differs from footprint geometry"):
            inspect_boundary_policy(self.paths, self.code)

    def test_exclusion_flag_mismatch_rejected(self):
        self._edit_data("enriched", "excluded_by_municipality_boundary", 1, False)
        with self.assertRaisesRegex(ValueError, "disagrees with policy"):
            inspect_boundary_policy(self.paths, self.code)

    def test_string_false_not_treated_as_true(self):
        self._edit_data("osm_only", "usable_for_accessibility", 1, "False")
        with self.assertRaisesRegex(ValueError, "explicit booleans"):
            inspect_boundary_policy(self.paths, self.code)

    def test_wrong_municipality_rejected(self):
        self._edit_data("osm_only", "municipality_code", 1, "016024")
        with self.assertRaisesRegex(ValueError, "municipality metadata"):
            inspect_boundary_policy(self.paths, self.code)

    def test_broken_boundary_source_rejected(self):
        self._edit_manifest("enriched", "boundary_source", "/unrelated/path", nested=True)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            inspect_boundary_policy(self.paths, self.code)

    def test_wrong_boundary_counts_rejected(self):
        self._edit_manifest("osm_only", "usable_after_boundary_filter", 3, nested=True)
        with self.assertRaisesRegex(ValueError, "manifest count mismatch"):
            inspect_boundary_policy(self.paths, self.code)

    def test_missing_manifests_fail_closed(self):
        self.paths.osm_only_manifest.unlink()
        with self.assertRaises(FileNotFoundError):
            inspect_boundary_policy(self.paths, self.code)

    def test_usable_without_network_node_rejected(self):
        self._edit_data("enriched", "network_node_id", 0, None)
        with self.assertRaisesRegex(ValueError, "no network node"):
            inspect_boundary_policy(self.paths, self.code)

    def test_path_conventions(self):
        self.assertEqual(self.paths.footprint.name, "034027_census_areas_2021.parquet")
        self.assertEqual(self.paths.enriched.name, "service_network_nodes_202425_20250630.parquet")
        self.assertEqual(self.paths.osm_only_manifest.name, "service_network_nodes_osm_only_manifest.json")
        with self.assertRaises(ValueError):
            paths_for(self.root, "34027")

    def test_boundary_geometry_invalid_rejected(self):
        gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, geometry="geometry",
                         crs="EPSG:4326").to_pickle(self.paths.footprint)
        with self.assertRaisesRegex(ValueError, "invalid geometries"):
            inspect_boundary_policy(self.paths, self.code)


if __name__ == "__main__":
    unittest.main()
