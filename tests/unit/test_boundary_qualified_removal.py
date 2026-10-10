"""B7B opt-in boundary-qualified real-removal integration, synthetic CSV fixtures."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analysis.run_boundary_qualified_removal import (
    POLICY, execute_boundary_qualified_removal, verify_boundary_view,
)
from core.run_manifest import sha256_file
from quality.derive_boundary_qualified_services import POLICY as VIEW_POLICY, write_boundary_view
import tests.unit.test_real_service_removal as fixtures


class B7BBoundaryQualifiedRealScenarioTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.B6C3bRealRemovalTests("test_real_removal_runs_without_creating_files")
        self.fx.setUp()
        self.addCleanup(self.fx.temp.cleanup)
        self.addCleanup(self.fx.fx.temp.cleanup)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.footprint = self.root / "footprint.parquet"
        self.footprint.write_bytes(b"mocked census footprint input")
        # Both locations were verified upstream; only one lies in the footprint.
        self.fx.fx.services["legacy_usable_for_accessibility"] = True
        self.fx.fx.save()
        self.canonical = self.fx.paths
        self.raw = pd.read_csv(self.canonical.services, dtype={"municipality_code": str})
        self.view = self.raw.copy()
        self.view["inside_operational_footprint"] = [True, False]
        self.view["operational_footprint_sha256"] = sha256_file(self.footprint)
        self.view["canonical_services_sha256"] = sha256_file(self.canonical.services)
        self.view["territorial_policy"] = VIEW_POLICY
        self.meta = {
            "policy": VIEW_POLICY,
            "municipality_code": self.fx.snapshot.municipality_code,
            "boundary_basis": "ISTAT_2021_census_area_footprint_not_administrative_boundary",
            "administrative_boundary_verified": False,
            "source_paths": {
                "services": str(self.canonical.services.resolve()),
                "footprint": str(self.footprint.resolve()),
            },
            "source_sha256": {
                "services": sha256_file(self.canonical.services),
                "footprint": sha256_file(self.footprint),
            },
            "services_total": 2,
            "inside_footprint": 1,
            "outside_footprint": 1,
            "location_validated_outside": 1,
        }
        self.qualified = self.root / "derived" / "boundary_services.csv"
        write_boundary_view(self.view, self.meta, self.qualified)
        self.mock_view = patch(
            "analysis.run_boundary_qualified_removal.inspect_boundary_view",
            return_value=(self.view, self.meta),
        )
        self.mock_view.start()
        self.addCleanup(self.mock_view.stop)

    def verify(self):
        return verify_boundary_view(
            snapshot=self.fx.snapshot, canonical=self.canonical,
            qualified_services=self.qualified, root=self.root,
        )

    def execute(self, **kw):
        return execute_boundary_qualified_removal(
            snapshot=self.fx.snapshot, canonical=self.canonical,
            qualified_services=self.qualified, city_name="Synthetic city",
            analysis_date="2025-06-30", remove_service_id="s1", root=self.root,
            **kw,
        )

    def test_preview_qualified_baseline_and_historical_isolation(self):
        original_hashes = {k: sha256_file(v) for k, v in self.canonical.items().items()}
        report, dest = self.execute()
        self.assertIsNone(dest)
        self.assertEqual(report["policy"], POLICY)
        self.assertEqual(report["canonical_baseline_service_count"], 2)
        self.assertEqual(report["preflight_eligible_services"], 1)
        self.assertEqual(report["metrics"]["baseline"]["service_count_routable"], 1)
        self.assertEqual(report["metrics"]["scenario"]["service_count_routable"], 0)
        self.assertEqual(report["territorial_qualification"]["footprint_sha256"], sha256_file(self.footprint))
        self.assertEqual(original_hashes, {k: sha256_file(v) for k, v in self.canonical.items().items()})
        self.assertFalse((self.root / "reports").exists())
        # Legacy B6C.3b uses the original catalogue; B7B must have a new ID.
        legacy, _ = self.fx.execute()
        self.assertEqual(legacy["metrics"]["baseline"]["service_count_routable"], 2)
        self.assertNotEqual(legacy["scenario_id"], report["scenario_id"])

    def test_immutable_report_contains_territorial_provenance(self):
        root = self.root / "reports"
        a, path = self.execute(write_report=True, output_root=root)
        self.assertFalse(a["report_cached"])
        manifest = json.loads((path / "manifest.json").read_text())
        bound = manifest["input_provenance"]["territorial_qualification"]
        self.assertEqual(bound["qualified_view_sha256"], sha256_file(self.qualified))
        self.assertEqual(bound["canonical_services_sha256"], sha256_file(self.canonical.services))
        self.assertEqual(bound["footprint_sha256"], sha256_file(self.footprint))
        b, path2 = self.execute(write_report=True, output_root=root)
        self.assertEqual(path, path2)
        self.assertTrue(b["report_cached"])

    def test_modified_view_rejected(self):
        self.qualified.write_text("changed")
        with self.assertRaisesRegex(ValueError, "invalid manifest checksum"):
            self.verify()

    def test_view_and_forged_digest_rejected(self):
        fake = self.view.copy()
        fake["inside_operational_footprint"] = [True, True]
        fake.to_csv(self.qualified, index=False)
        path = self.qualified.with_suffix(".csv.manifest.json")
        meta = json.loads(path.read_text())
        meta["view_sha256"] = sha256_file(self.qualified)
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "independently reconstructed"):
            self.verify()

    def test_changed_canonical_rejected(self):
        original = self.canonical.services.read_bytes()
        self.canonical.services.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "canonical services changed|manifest evidence mismatch"):
            self.verify()

    def test_changed_footprint_rejected(self):
        self.footprint.write_bytes(b"different footprint")
        with self.assertRaisesRegex(ValueError, "footprint changed|manifest evidence mismatch"):
            self.verify()

    def test_ineligible_outside_service_rejected(self):
        with self.assertRaisesRegex(ValueError, "not an eligible baseline destination"):
            execute_boundary_qualified_removal(
                snapshot=self.fx.snapshot, canonical=self.canonical,
                qualified_services=self.qualified, city_name="Synthetic city",
                analysis_date="2025-06-30", remove_service_id="s2", root=self.root,
            )

    def test_missing_manifest_rejected(self):
        self.qualified.with_suffix(".csv.manifest.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "manifest"):
            self.verify()

    def test_manifest_wrong_path_rejected(self):
        path = self.qualified.with_suffix(".csv.manifest.json")
        manifest = json.loads(path.read_text())
        manifest["view_path"] = str(self.root / "elsewhere.csv")
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest path"):
            self.verify()

    def test_wrong_source_location_rejected(self):
        other = self.root / "fake_canonical.csv"
        other.write_bytes(self.canonical.services.read_bytes())
        from dataclasses import replace
        wrong = replace(self.canonical, services=other)
        with self.assertRaisesRegex(ValueError, "different canonical service catalogue"):
            verify_boundary_view(
                snapshot=self.fx.snapshot, canonical=wrong,
                qualified_services=self.qualified, root=self.root,
            )


if __name__ == "__main__":
    unittest.main()
