import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from core.schema_v2 import ORIGIN_V2
from transformation.build_network_attachments_v2 import input_paths, prepare_origin_points
from transformation.migrate_legacy_origins_v2 import default_paths, migrate_origins


class LegacyOriginV2MigrationTests(unittest.TestCase):
    def setUp(self):
        self.legacy = gpd.GeoDataFrame({
            "census_section_code": ["s001", "s002"],
            "network_node_id": ["100", "200"],
            "assigned_population": [14.5, 20.25],
            "reference_date": ["2023-12-31", "2023-12-31"],
            "assignment_method": ["equal_internal_nodes", "representative_point_nearest_node"],
            "assigned_age_5_9": [1.0, 2.0],
            "geometry": [Point(10.3, 44.8), Point(10.4, 44.9)],
        }, geometry="geometry", crs="EPSG:4326")

    def test_migrates_real_legacy_fields_without_changing_rows(self):
        migrated = migrate_origins(self.legacy, municipality_code="077014", census_year="2023")
        ORIGIN_V2.validate_columns(migrated.columns)
        self.assertEqual(len(migrated), 2)
        self.assertEqual(migrated["origin_id"].tolist(), [
            "origin:077014:2023:s001:100", "origin:077014:2023:s002:200",
        ])
        self.assertTrue(migrated.geometry.equals(self.legacy.geometry))
        self.assertEqual(migrated["population_total"].sum(), 34.75)
        self.assertEqual(migrated["population_age_5_9_proxy"].tolist(), [1.0, 2.0])
        self.assertNotIn("origin_id", self.legacy.columns)
        prepared = prepare_origin_points(migrated)
        self.assertEqual(len(prepared), 2)

    def test_missing_required_legacy_identity_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            migrate_origins(self.legacy.drop(columns=["network_node_id"]),
                            municipality_code="077014", census_year="2023")

    def test_duplicate_composite_ids_rejected(self):
        duplicated = self.legacy.copy()
        duplicated.loc[1, "census_section_code"] = "s001"
        duplicated.loc[1, "network_node_id"] = "100"
        with self.assertRaisesRegex(RuntimeError, "duplicati"):
            migrate_origins(duplicated, municipality_code="077014", census_year="2023")

    def test_no_invented_population_for_missing_values(self):
        missing = self.legacy.copy()
        missing.loc[0, "assigned_population"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            migrate_origins(missing, municipality_code="077014", census_year="2023")

    def test_bad_reference_or_method_rejected(self):
        bad = self.legacy.copy()
        bad.loc[0, "reference_date"] = "not-a-date"
        with self.assertRaisesRegex(ValueError, "invalid population"):
            migrate_origins(bad, municipality_code="077014", census_year="2023")
        bad = self.legacy.copy()
        bad.loc[0, "assignment_method"] = "new_unverified_method"
        with self.assertRaisesRegex(ValueError, "Unknown legacy"):
            migrate_origins(bad, municipality_code="077014", census_year="2023")

    def test_invalid_municipality_or_existing_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, "six ISTAT"):
            migrate_origins(self.legacy, municipality_code="77", census_year="2023")
        already = self.legacy.copy()
        already["origin_id"] = ["x", "y"]
        with self.assertRaisesRegex(ValueError, "only for legacy"):
            migrate_origins(already, municipality_code="077014", census_year="2023")

    def test_wrong_city_in_existing_legacy_field_rejected(self):
        bad = self.legacy.copy()
        bad["municipality_code"] = ["077014", "016024"]
        with self.assertRaisesRegex(ValueError, "does not match"):
            migrate_origins(bad, municipality_code="077014", census_year="2023")

    def test_nonpoint_legacy_geometry_rejected(self):
        bad = self.legacy.copy()
        bad.at[0, "geometry"] = None
        with self.assertRaisesRegex(ValueError, "Point"):
            migrate_origins(bad, municipality_code="077014", census_year="2023")

    def test_negative_population_rejected(self):
        bad = self.legacy.copy()
        bad.loc[0, "assigned_population"] = -1.0
        with self.assertRaisesRegex(ValueError, "negative"):
            migrate_origins(bad, municipality_code="077014", census_year="2023")

    def test_explicit_origin_input_does_not_change_default(self):
        args = Namespace(municipality_code="077014", census_year="2023",
                         school_year="202425", health_reference_date=pd.Timestamp("2025-06-30"),
                         mode="walk", origins_parquet=None)
        p = input_paths(args)
        self.assertEqual(p["origins"].name, "population_network_origins_2023.parquet")
        custom = Path("/tmp/versioned_origins.parquet")
        args.origins_parquet = custom
        self.assertEqual(input_paths(args)["origins"], custom)

    def test_migration_never_targets_legacy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, derived = default_paths(Path(tmp), "077014", "2023")
            self.assertNotEqual(source, derived)
            self.assertEqual(source.name, "population_network_origins_2023.parquet")
            self.assertEqual(derived.name, "population_network_origins_2023_v2_migrated.parquet")


if __name__ == "__main__":
    unittest.main()
