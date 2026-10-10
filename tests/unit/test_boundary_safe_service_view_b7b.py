"""B7B derived operational footprint gating: synthetic, no city files."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from analysis.accessibility_contracts import prepare_service_destinations_v2
from core.analysis_spec import TransportMode
from core.run_manifest import sha256_file
from quality.derive_v2_boundary_service_view_b7b import (
    POLICY, FLAG, qualify_v2_services, inspect_boundary_view, write_boundary_view,
)
from quality.audit_v2_boundary_policy_b7b import paths_for_snapshot
import tests.unit.test_v2_boundary_policy_b7b as fixtures


class B7BBoundarySafeViewTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.B7BV2BoundaryTests(methodName="test_valid_census_boundary_audit")
        self.fx.setUp()
        self.addCleanup(self.fx.fx.temp.cleanup)
        self.raw = self.fx.fx.services.copy(deep=True)
        self.raw.loc[1, "legacy_usable_for_accessibility"] = True
        self.raw.loc[1, "latitude"] = 0.0
        self.raw.loc[1, "longitude"] = 3.0
        self.areas = self.fx.areas.copy()
        self.sha1 = "a" * 64
        self.sha2 = "b" * 64
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def qualify(self):
        return qualify_v2_services(
            self.raw, self.areas, municipality_code="034027",
            canonical_sha256=self.sha1, footprint_sha256=self.sha2,
        )

    def test_outside_location_validated_but_no_longer_routable(self):
        original = self.raw.copy(deep=True)
        qualified = self.qualify()
        self.assertTrue(qualified["legacy_usable_for_accessibility"].all())
        self.assertEqual(qualified[FLAG].tolist(), [True, False])
        self.assertEqual(qualified["operational_footprint_sha256"].tolist(), [self.sha2] * 2)
        self.assertEqual(qualified["canonical_services_sha256"].tolist(), [self.sha1] * 2)
        self.assertNotIn(FLAG, self.raw.columns)
        pd.testing.assert_frame_equal(original, self.raw)

        gated = prepare_service_destinations_v2(
            qualified, self.fx.fx.attachments.loc[self.fx.fx.attachments["entity_kind"].eq("service")],
            mode=TransportMode.WALK,
        )
        self.assertEqual(gated["routing_eligible"].tolist(), [True, False])
        self.assertEqual(gated["routing_exclusion_reason"].tolist(), ["eligible", "outside_operational_footprint"])
        self.assertIsNone(gated.iloc[1]["network_node_id"])
        self.assertTrue(gated.iloc[1]["snapped"])
        self.assertTrue(pd.notna(gated.iloc[1]["attachment_node_id"]))

    def test_old_gate_is_unchanged_when_boundary_evidence_absent(self):
        # Legacy reports are immutable, but they are not B7B-certified.
        gated = prepare_service_destinations_v2(
            self.raw, self.fx.fx.attachments.loc[self.fx.fx.attachments["entity_kind"].eq("service")],
            mode=TransportMode.WALK,
        )
        self.assertTrue(gated["routing_eligible"].all())

    def test_missing_boundary_evidence_fails_closed(self):
        view = self.qualify()
        view[FLAG] = view[FLAG].astype(object)
        view.loc[1, FLAG] = None
        gated = prepare_service_destinations_v2(
            view, self.fx.fx.attachments.loc[self.fx.fx.attachments["entity_kind"].eq("service")],
            mode=TransportMode.WALK,
        )
        self.assertEqual(gated.loc[1, "routing_exclusion_reason"], "boundary_not_verified")
        self.assertFalse(gated.loc[1, "routing_eligible"])

    def test_string_boundary_false_is_rejected(self):
        view = self.qualify()
        view[FLAG] = view[FLAG].astype(object)
        view.loc[1, FLAG] = "False"
        with self.assertRaisesRegex(ValueError, "boolean or null"):
            prepare_service_destinations_v2(
                view, self.fx.fx.attachments.loc[self.fx.fx.attachments["entity_kind"].eq("service")],
                mode=TransportMode.WALK,
            )

    def test_invalid_latitude_rejected(self):
        self.raw.loc[1, "latitude"] = 91.0
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.qualify()

    def test_invalid_crs_rejected(self):
        self.raw.loc[0, "crs"] = "EPSG:3857"
        with self.assertRaisesRegex(ValueError, "EPSG:4326"):
            self.qualify()

    def test_no_double_qualification(self):
        qualified = self.qualify()
        with self.assertRaisesRegex(ValueError, "already boundary-qualified"):
            qualify_v2_services(qualified, self.areas, municipality_code="034027",
                                canonical_sha256=self.sha1, footprint_sha256=self.sha2)

    def test_wrong_municipality_rejected(self):
        with self.assertRaisesRegex(ValueError, "municipality code mismatch"):
            qualify_v2_services(self.raw, self.areas, municipality_code="016024",
                                canonical_sha256=self.sha1, footprint_sha256=self.sha2)

    def test_invalid_digest_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid footprint"):
            qualify_v2_services(self.raw, self.areas, municipality_code="034027",
                                canonical_sha256=self.sha1, footprint_sha256="n/a")

    def test_forged_boundary_geometry_rejected(self):
        self.areas = gpd.GeoDataFrame({"id": ["bad"]},
                                      geometry=[Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])],
                                      crs="EPSG:4326")
        with self.assertRaisesRegex(ValueError, "invalid geometries"):
            self.qualify()

    def test_read_only_inspection_on_real_path_convention(self):
        root = Path(self.fx.fx.temp.name)
        snap = self.fx.snapshot
        resolved = paths_for_snapshot(root, snap)
        resolved.services.parent.mkdir(parents=True, exist_ok=True)
        resolved.footprint.parent.mkdir(parents=True, exist_ok=True)
        resolved.services.write_bytes(b"synthetic service parquet")
        resolved.footprint.write_bytes(b"synthetic footprint parquet")
        before = [sha256_file(resolved.services), sha256_file(resolved.footprint)]
        with patch("quality.derive_v2_boundary_service_view_b7b._read_table", return_value=self.raw), \
             patch("quality.derive_v2_boundary_service_view_b7b.gpd.read_parquet", return_value=self.areas):
            view, report = inspect_boundary_view(snap, root=root)
        self.assertEqual(report["location_validated_outside"], 1)
        self.assertEqual(report["outside_footprint"], 1)
        self.assertFalse(report["scenario_routing_active"])
        self.assertTrue(view.loc[0, FLAG])
        self.assertFalse(view.loc[1, FLAG])
        self.assertEqual(before, [sha256_file(resolved.services), sha256_file(resolved.footprint)])

    def test_immutable_write_is_explicit_and_idempotent(self):
        root = Path(self.temp.name)
        view = self.qualify()
        info = {"policy": POLICY, "source_sha256": {"services": self.sha1, "footprint": self.sha2}}
        output = root / "derived.csv"
        self.assertFalse(output.exists())
        path, cached = write_boundary_view(view, info, output)
        self.assertEqual(path, output)
        self.assertFalse(cached)
        self.assertTrue(output.with_suffix(".csv.manifest.json").exists())
        _, cached = write_boundary_view(view, info, output)
        self.assertTrue(cached)
        manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
        self.assertEqual(manifest["view_sha256"], sha256_file(output))
        output.write_text("modified")
        with self.assertRaises(FileExistsError):
            write_boundary_view(view, info, output)

    def test_reject_invalid_artifact_suffix(self):
        with self.assertRaisesRegex(ValueError, "parquet or .csv"):
            write_boundary_view(self.qualify(), {}, Path(self.temp.name) / "file.txt")


if __name__ == "__main__":
    unittest.main()
