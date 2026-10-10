"""B6C.2 isolated file-backed scenario run; never use real baseline artifacts."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from analysis.scenario_contracts_v2 import ScenarioAction, ScenarioOperation, ScenarioSpec
from analysis.service_scenarios_v2 import scenario_row_sha256
from analysis.scenario_reporting_v2 import (
    ScenarioInputPaths, run_verified_scenario, write_scenario_report,
)
from core.run_manifest import sha256_file
from transformation.build_network_attachments_v2 import graph_checksum


class B6C2ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        from tests.unit.test_service_scenarios_b6c1 import B6C1ScenarioTests
        fixture = B6C1ScenarioTests(methodName="test_inputs_are_immutable")
        fixture.setUp()
        self.fixture = fixture
        self.spec = fixture.spec
        self.request = fixture.request
        self.paths = {name: self.root / f"{name}.csv" for name in (
            "origins", "services", "attachments", "nodes", "edges",
            "addition_services", "addition_attachments")}
        nodes = pd.DataFrame({"source_record_id": ["1", "2", "3"]})
        edges = pd.DataFrame({
            "source_osm_node": ["1", "2"], "target_osm_node": ["2", "3"],
            "walking_time_s": [600.0, 600.0], "length_m": [840.0, 840.0],
        })
        nodes.to_csv(self.paths["nodes"], index=False)
        edges.to_csv(self.paths["edges"], index=False)
        self.graph_sha = graph_checksum(self.paths["nodes"], self.paths["edges"])
        self.attachments = fixture.attachments.copy(deep=True)
        self.attachments["graph_checksum"] = self.graph_sha
        self.additional_attachments = fixture.new_attachment.copy(deep=True)
        self.additional_attachments["graph_checksum"] = self.graph_sha
        for name, data in (
            ("origins", fixture.origins), ("services", fixture.services),
            ("attachments", self.attachments),
            ("addition_services", fixture.new_service),
            ("addition_attachments", self.additional_attachments),
        ):
            data.to_csv(self.paths[name], index=False)
        self.ref = replace(
            fixture.ref, origins_sha256=sha256_file(self.paths["origins"]),
            services_sha256=sha256_file(self.paths["services"]),
            attachments_sha256=sha256_file(self.paths["attachments"]),
            graph_sha256=self.graph_sha,
        )
        self.inputs = ScenarioInputPaths(**self.paths)
        # Hash rows exactly as the file-backed loader sees them.
        from analysis.scenario_reporting_v2 import _read_table
        sr = _read_table(self.paths["addition_services"]).iloc[0]
        ar = _read_table(self.paths["addition_attachments"]).iloc[0]
        self.add = ScenarioOperation(ScenarioAction.ADD_SERVICE, "s2",
                                     scenario_row_sha256(sr), scenario_row_sha256(ar))
        self.remove = ScenarioOperation(ScenarioAction.REMOVE_SERVICE, "s1")

    def run_it(self, *ops, inputs=None, ref=None):
        return run_verified_scenario(
            scenario=ScenarioSpec(ref or self.ref, self.request.service_type,
                                  self.request.mode, ops),
            analysis_spec=self.spec, request=self.request,
            inputs=inputs or self.inputs,
        )

    def test_add_file_backed_execution_and_verified_manifest(self):
        result, provenance = self.run_it(self.add)
        self.assertAlmostEqual(result.metrics["scenario"]["population_coverage_within_10_min"], 1.0)
        self.assertEqual(provenance["source_verification"], "file_bytes_verified_against_scenario_baseline")
        self.assertEqual(provenance["graph_sha256"], self.graph_sha)
        self.assertEqual(provenance["input_sha256"]["origins"], self.ref.origins_sha256)
        folder, cached = write_scenario_report(result=result, provenance=provenance,
                                               output_root=self.root / "reports")
        self.assertFalse(cached)
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual(manifest["scenario_id"], result.scenario_id)
        self.assertEqual(manifest["output_sha256"]["comparison_origins.csv"],
                         sha256_file(folder / "comparison_origins.csv"))
        self.assertEqual(len(pd.read_csv(folder / "comparison_origins.csv")), 2)
        metrics = json.loads((folder / "metrics.json").read_text())
        self.assertAlmostEqual(metrics["coverage_delta_percentage_points"][
            "population_coverage_within_10_min"], (2 / 3) * 100)
        self.assertIn("relative_change_percent", metrics)
        self.assertAlmostEqual(metrics["relative_change_percent"][
            "population_coverage_within_10_min"], 200.0)
        _, cached = write_scenario_report(result=result, provenance=provenance,
                                          output_root=self.root / "reports")
        self.assertTrue(cached)

    def test_remove_file_backed_zero_services(self):
        inputs = ScenarioInputPaths(**{k: self.paths[k] for k in
                                      ("origins", "services", "attachments", "nodes", "edges")})
        result, provenance = self.run_it(self.remove, inputs=inputs)
        self.assertEqual(result.metrics["scenario"]["service_count_routable"], 0)
        self.assertEqual(result.metrics["scenario"]["reachable_population"], 0)
        self.assertNotIn("addition_services", provenance["input_sha256"])

    def test_refuses_changed_baseline_bytes(self):
        self.paths["services"].write_bytes(self.paths["services"].read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "services SHA-256 mismatch"):
            self.run_it(self.add)

    def test_refuses_changed_graph(self):
        self.paths["edges"].write_bytes(self.paths["edges"].read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "graph node/edge SHA-256 mismatch"):
            self.run_it(self.add)

    def test_refuses_changed_addition_data(self):
        self.paths["addition_services"].write_text(
            self.paths["addition_services"].read_text().replace("Candidate", "Altered")
        )
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            self.run_it(self.add)

    def test_refuses_unvalidated_addition(self):
        text = self.paths["addition_services"].read_text()
        self.paths["addition_services"].write_text(text.replace("True", "False"))
        from analysis.scenario_reporting_v2 import _read_table
        row = _read_table(self.paths["addition_services"]).iloc[0]
        operation = replace(self.add, service_row_sha256=scenario_row_sha256(row))
        with self.assertRaisesRegex(ValueError, "not independently location-validated"):
            self.run_it(operation)

    def test_refuses_wrong_node_or_mode(self):
        from analysis.scenario_reporting_v2 import _read_table
        att = _read_table(self.paths["addition_attachments"])
        att["node_id"] = "no-such-node"
        att.to_csv(self.paths["addition_attachments"], index=False)
        operation = replace(self.add, attachment_row_sha256=scenario_row_sha256(
            _read_table(self.paths["addition_attachments"]).iloc[0]))
        with self.assertRaisesRegex(ValueError, "missing from graph"):
            self.run_it(operation)

    def test_report_refuses_changed_bytes(self):
        result, provenance = self.run_it(self.add)
        folder, _ = write_scenario_report(result=result, provenance=provenance,
                                          output_root=self.root / "reports")
        (folder / "metrics.json").write_bytes(b"tampered")
        with self.assertRaises(FileExistsError):
            write_scenario_report(result=result, provenance=provenance,
                                  output_root=self.root / "reports")

    def test_report_refuses_extra_or_partial_files(self):
        result, provenance = self.run_it(self.add)
        folder, _ = write_scenario_report(result=result, provenance=provenance,
                                          output_root=self.root / "reports")
        (folder / "unexpected.csv").write_text("unexpected")
        with self.assertRaises(FileExistsError):
            write_scenario_report(result=result, provenance=provenance,
                                  output_root=self.root / "reports")

    def test_report_refuses_forged_provenance(self):
        result, provenance = self.run_it(self.remove, inputs=ScenarioInputPaths(
            **{k: self.paths[k] for k in ("origins", "services", "attachments", "nodes", "edges")}))
        wrong = dict(provenance, scenario_execution={})
        with self.assertRaisesRegex(ValueError, "provenance"):
            write_scenario_report(result=result, provenance=wrong, output_root=self.root / "reports")

    def test_requires_pair_of_addition_paths(self):
        with self.assertRaisesRegex(ValueError, "both"):
            ScenarioInputPaths(
                **{k: self.paths[k] for k in ("origins", "services", "attachments", "nodes", "edges")},
                addition_services=self.paths["addition_services"])

    def test_reject_nonexistent_path_or_unknown_extension(self):
        from analysis.scenario_reporting_v2 import _read_table
        with self.assertRaises(FileNotFoundError):
            self.run_it(self.add, inputs=replace(self.inputs, origins=self.root / "missing.csv"))
        bad = self.root / "table.txt"
        bad.write_text("x,y\n1,2\n")
        with self.assertRaisesRegex(ValueError, "Unsupported input format"):
            _read_table(bad)

    def test_deterministic_reports_across_repeated_execution(self):
        result1, provenance1 = self.run_it(self.add)
        result2, provenance2 = self.run_it(self.add)
        from analysis.scenario_reporting_v2 import _artifacts
        self.assertEqual(_artifacts(result1, provenance1), _artifacts(result2, provenance2))
        self.assertEqual(provenance1, provenance2)


if __name__ == "__main__":
    unittest.main()
