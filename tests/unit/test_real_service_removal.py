"""B6C.3b isolated synthetic testing; no municipality datasets required."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.run_real_service_removal import (
    _check_legacy_walk_network, execute_real_removal,
)
from core.analysis_spec import TransportMode
from core.run_manifest import sha256_file
from quality.preflight_real_service_scenario import CitySnapshot
from transformation.build_network_attachments_v2 import graph_checksum
import tests.unit.test_real_service_scenario_preflight as fixtures


class B6C3bRealRemovalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fx = fixtures.B6C3aTests(methodName="test_preflight_does_not_write_anything")
        self.fx.setUp()
        self.addCleanup(self.fx.temp.cleanup)
        self.paths = self.fx.paths
        self.snapshot = self.fx.snapshot
        self.output_root = Path(self.temp.name) / "reports"
        # Mirror the real walking network's frozen 1.4 m/s edge weighting.
        self.fx.edges["walking_time_s"] = self.fx.edges["length_m"] / 1.4
        self.fx.edges.to_csv(self.paths.edges, index=False)
        self.fx.attachments["graph_checksum"] = graph_checksum(self.paths.nodes, self.paths.edges)
        self.fx.save()

    def execute(self, service_id="s1", write=False, **kwargs):
        kwargs.setdefault("output_root", self.output_root if write else None)
        return execute_real_removal(
            snapshot=self.snapshot, paths=self.paths, service_id=service_id,
            city_name="Synthetic city", analysis_date="2025-06-30",
            write_report=write, **kwargs,
        )

    def test_real_removal_runs_without_creating_files(self):
        before = {n: sha256_file(p) for n, p in self.paths.items().items()}
        report, dest = self.execute()
        self.assertIsNone(dest)
        self.assertFalse(self.output_root.exists())
        self.assertEqual(before, {n: sha256_file(p) for n, p in self.paths.items().items()})
        self.assertEqual(report["preflight_eligible_services"], 1)
        self.assertEqual(report["execution_profile"], "legacy_v1_regression")
        self.assertEqual(report["thresholds_min"], [10, 15, 20])
        self.assertEqual(report["metrics"]["baseline"]["service_count_routable"], 1)
        self.assertEqual(report["metrics"]["scenario"]["service_count_routable"], 0)
        self.assertEqual(report["metrics"]["scenario"]["target_population_total"], 150.0)
        self.assertLessEqual(report["metrics"]["delta"]["population_coverage_within_10_min"], 0)

    def test_explicit_write_is_immutable_and_idempotent(self):
        before = {n: sha256_file(p) for n, p in self.paths.items().items()}
        report1, path1 = self.execute(write=True)
        self.assertTrue((path1 / "manifest.json").is_file())
        self.assertTrue((path1 / "comparison_origins.csv").is_file())
        self.assertFalse(report1["report_cached"])
        report2, path2 = self.execute(write=True)
        self.assertEqual(path1, path2)
        self.assertTrue(report2["report_cached"])
        self.assertEqual(before, {n: sha256_file(p) for n, p in self.paths.items().items()})
        with (path1 / "manifest.json").open() as file:
            self.assertEqual(json.load(file)["scenario_id"], report1["scenario_id"])
        (path1 / "metrics.json").write_text("tampered")
        with self.assertRaises(FileExistsError):
            self.execute(write=True)

    def test_unrelated_service_attachments_are_scoped_without_promotion(self):
        # Real municipal canonical files have schools and pharmacies together,
        # unlike the original single-category synthetic fixture.
        other = self.fx.services.iloc[[0]].copy()
        other.loc[:, "service_id"] = "education1"
        other.loc[:, "service_type"] = "primary_school"
        other.loc[:, "source_record_id"] = "school1"
        self.fx.services = pd.concat([self.fx.services, other], ignore_index=True)
        other_att = self.fx.attachments.loc[
            (self.fx.attachments["entity_kind"] == "service")
            & (self.fx.attachments["entity_id"] == "s1")
        ].copy()
        other_att.loc[:, "entity_id"] = "education1"
        self.fx.attachments = pd.concat([self.fx.attachments, other_att], ignore_index=True)
        self.fx.save()

        hashes_before = {key: sha256_file(path) for key, path in self.paths.items().items()}
        result, report_path = self.execute()
        self.assertIsNone(report_path)
        self.assertEqual(result["preflight_eligible_services"], 1)
        self.assertEqual(result["metrics"]["baseline"]["service_count_routable"], 1)
        self.assertEqual(result["metrics"]["scenario"]["service_count_routable"], 0)
        self.assertEqual(hashes_before, {key: sha256_file(path) for key, path in self.paths.items().items()})

    def test_selected_service_missing_attachment_remains_rejected(self):
        self.fx.attachments = self.fx.attachments.loc[
            self.fx.attachments["entity_id"] != "s1"
        ].copy()
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Service attachments missing"):
            self.execute()

    def test_ineligible_service_not_allowed(self):
        with self.assertRaisesRegex(ValueError, "not an eligible"):
            self.execute(service_id="s2")
        with self.assertRaisesRegex(ValueError, "not an eligible"):
            self.execute(service_id="not-here")

    def test_network_speed_mismatch_fails_closed(self):
        self.fx.edges.loc[0, "walking_time_s"] += 3
        self.fx.edges.to_csv(self.paths.edges, index=False)
        self.fx.attachments["graph_checksum"] = graph_checksum(self.paths.nodes, self.paths.edges)
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "inconsistent with the legacy walking speed"):
            self.execute()

    def test_changed_graph_refused_by_preflight(self):
        self.fx.edges.loc[0, "length_m"] = 200
        self.fx.edges.to_csv(self.paths.edges, index=False)
        with self.assertRaisesRegex(ValueError, "graph checksum"):
            self.execute()

    def test_all_ineligible_rejected(self):
        self.fx.services["legacy_usable_for_accessibility"] = False
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "No eligible services"):
            self.execute()

    def test_unsupported_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "walking only"):
            execute_real_removal(snapshot=CitySnapshot(municipality_code="034027", mode=TransportMode.DRIVE),
                paths=self.paths, service_id="s1", city_name="Synthetic", analysis_date="2025-06-30")

    def test_requires_write_destination(self):
        with self.assertRaisesRegex(ValueError, "Explicit output_root"):
            self.execute(write=True, output_root=None)

    def test_legacy_speed_validation(self):
        df = pd.DataFrame({"length_m": [140, 280], "walking_time_s": [100, 200]})
        _check_legacy_walk_network(df, 1.4)
        df.loc[0, "walking_time_s"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid walking network"):
            _check_legacy_walk_network(df, 1.4)


if __name__ == "__main__":
    unittest.main()
