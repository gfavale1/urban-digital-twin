"""B7C.0: synthetic candidate preflight tests; never approve a real source."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from analysis.scenario_reporting_v2 import ScenarioInputPaths
from core.run_manifest import sha256_file
from quality.audit_service_addition_boundaries import inspect_boundary_additions
import tests.unit.test_scenario_reporting_v2 as fixtures


class B7CCandidateBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.B6C2ReportingTests(methodName="test_add_file_backed_execution_and_verified_manifest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = __import__("quality.preflight_real_service_scenario", fromlist=["CitySnapshot"]).CitySnapshot(
            municipality_code="034027", service_type=self.fixture.request.service_type,
            mode=self.fixture.request.mode,
        )
        self.canonical = ScenarioInputPaths(**{
            k: self.fixture.paths[k] for k in ("origins", "services", "attachments", "nodes", "edges")
        })
        self.qualified = self.root / "qualified.csv"
        b = pd.read_csv(self.canonical.services, dtype={"municipality_code": str})
        b["inside_operational_footprint"] = True
        b.to_csv(self.qualified, index=False)
        self.footprint = self.root / "footprint.parquet"
        self.footprint.write_bytes(b"fixture footprint bytes")
        self.footprint_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(9.9, 43.9, 10.1, 44.1)], crs="EPSG:4326")
        self.candidate_services = self.fixture.paths["addition_services"]
        self.candidate_att = self.fixture.paths["addition_attachments"]
        s = pd.read_csv(self.candidate_services, dtype={"municipality_code": str})
        s["latitude"] = 44.0
        s["longitude"] = 10.0
        s["crs"] = "EPSG:4326"
        s.to_csv(self.candidate_services, index=False)
        self.proof = {
            "qualified_view_sha256": sha256_file(self.qualified),
            "canonical_services_sha256": sha256_file(self.canonical.services),
            "footprint_sha256": sha256_file(self.footprint),
            "footprint_path": str(self.footprint.resolve()),
            "boundary_basis": "ISTAT_2021_census_area_footprint_not_administrative_boundary",
        }
        self.patched = patch("quality.audit_service_addition_boundaries.verify_boundary_view", return_value=self.proof)
        self.patched.start()
        self.addCleanup(self.patched.stop)
        footprint_reader = patch("quality.audit_service_addition_boundaries.gpd.read_parquet", return_value=self.footprint_gdf)
        footprint_reader.start()
        self.addCleanup(footprint_reader.stop)

    def check(self):
        return inspect_boundary_additions(
            snapshot=self.snapshot, canonical=self.canonical,
            qualified_services=self.qualified, addition_services=self.candidate_services,
            addition_attachments=self.candidate_att, root=self.root,
        )

    def edit_service(self, **updates):
        data = pd.read_csv(self.candidate_services, dtype={"municipality_code": str})
        for key, val in updates.items():
            data[key] = val
        data.to_csv(self.candidate_services, index=False)

    def edit_att(self, **updates):
        data = pd.read_csv(self.candidate_att)
        for key, val in updates.items():
            data[key] = val
        data.to_csv(self.candidate_att, index=False)

    def test_inside_candidate_passes_geography_not_review(self):
        result = self.check()
        self.assertEqual(result["candidate_service_ids"], ["s2"])
        self.assertEqual(result["qualified_baseline_routable"], 1)
        self.assertEqual(result["candidate_count"], 1)
        self.assertFalse(result["administrative_boundary_verified"])
        self.assertFalse(result["independent_source_review_approved"])
        self.assertFalse(result["scenario_authorized"])
        self.assertEqual(result["verified_input_sha256"]["footprint"], sha256_file(self.footprint))

    def test_all_inputs_are_read_only(self):
        paths = [*self.canonical.items().values(), self.qualified, self.footprint,
                 self.candidate_services, self.candidate_att]
        before = [sha256_file(path) for path in paths]
        self.check()
        self.assertEqual(before, [sha256_file(path) for path in paths])

    def test_candidate_outside_rejected(self):
        self.edit_service(latitude=45.0)
        with self.assertRaisesRegex(ValueError, "outside or without coordinates"):
            self.check()

    def test_missing_candidate_coordinate_rejected(self):
        self.edit_service(latitude=float("nan"))
        with self.assertRaisesRegex(ValueError, "Incomplete|outside or without coordinates"):
            self.check()

    def test_coordinate_crs_not_wgs84_rejected(self):
        self.edit_service(crs="EPSG:3857")
        with self.assertRaisesRegex(ValueError, "EPSG:4326"):
            self.check()

    def test_invalid_coordinate_rejected(self):
        self.edit_service(latitude=95.0)
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.check()

    def test_candidate_cannot_self_attest_boundary(self):
        self.edit_service(inside_operational_footprint=True)
        with self.assertRaisesRegex(ValueError, "self-certify"):
            self.check()

    def test_wrong_municipality_rejected(self):
        self.edit_service(municipality_code="016024")
        with self.assertRaisesRegex(ValueError, "another municipality"):
            self.check()

    def test_wrong_service_type_rejected(self):
        self.edit_service(service_type="hospital_establishment")
        with self.assertRaisesRegex(ValueError, "type differs"):
            self.check()

    def test_duplicate_baseline_service_identity_rejected(self):
        self.edit_service(source_record_id="1")
        with self.assertRaisesRegex(ValueError, "duplicates a canonical"):
            self.check()

    def test_duplicate_baseline_id_rejected(self):
        self.edit_service(service_id="s1")
        with self.assertRaisesRegex(ValueError, "reuses a canonical"):
            self.check()

    def test_wrong_mode_attachment_rejected(self):
        self.edit_att(mode="drive")
        with self.assertRaisesRegex(ValueError, "service IDs/mode"):
            self.check()

    def test_missing_graph_hash_rejected(self):
        self.edit_att(graph_checksum="f" * 64)
        with self.assertRaisesRegex(ValueError, "graph provenance"):
            self.check()

    def test_graph_node_not_present_rejected(self):
        self.edit_att(node_id="999999")
        with self.assertRaisesRegex(ValueError, "missing graph node"):
            self.check()

    def test_inactive_candidate_rejected(self):
        self.edit_service(operational_status="planned")
        with self.assertRaisesRegex(ValueError, "existing B6A3"):
            self.check()

    def test_location_unvalidated_candidate_rejected(self):
        self.edit_service(legacy_usable_for_accessibility=False)
        with self.assertRaisesRegex(ValueError, "existing B6A3"):
            self.check()

    def test_missing_candidate_attachment_file_rejected(self):
        self.candidate_att.unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_wrong_perimeter_proof_rejected(self):
        self.proof["footprint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "footprint"):
            self.check()


if __name__ == "__main__":
    unittest.main()
