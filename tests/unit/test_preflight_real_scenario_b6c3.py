"""B6C.3a: synthetic, read-only preflight; real Parquet is tested on WSL."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.scenario_reporting_v2 import ScenarioInputPaths
from core.analysis_spec import ServiceType, TransportMode
from quality.preflight_real_scenario_b6c3 import (
    CitySnapshot, canonical_paths, inspect_real_snapshot,
)
from transformation.build_network_attachments_v2 import graph_checksum


class B6C3aTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.paths = ScenarioInputPaths(**{
            name: root / f"{name}.csv" for name in
            ("origins", "services", "attachments", "nodes", "edges")
        })
        self.snapshot = CitySnapshot(municipality_code="034027")
        self.origin = pd.DataFrame({
            "origin_id": ["o1", "o2"], "municipality_code": ["034027", "034027"],
            "geometry": ["POINT(0 0)", "POINT(0 0)"], "population_total": [100., 50.],
            "population_reference_date": ["2023-12-31"] * 2,
            "census_year": ["2023"] * 2, "geography_version": ["BT2021"] * 2,
            "origin_method": ["representative_point"] * 2,
        })
        self.services = pd.DataFrame({
            "service_id": ["s1", "s2"], "service_type": ["pharmacy", "pharmacy"],
            "name": ["Validated", "Not validated"], "municipality_code": ["034027"] * 2,
            "source_name": ["Ministry"] * 2, "source_record_id": ["1", "2"],
            "source_reference_date": ["2025-06-30"] * 2,
            "retrieved_at": ["2026-10-10"] * 2,
            "operational_status": ["active"] * 2,
            "legacy_usable_for_accessibility": [True, False],
        })
        self.nodes = pd.DataFrame({"source_record_id": ["1", "2", "3"]})
        self.edges = pd.DataFrame({"source_osm_node": ["1", "2"],
                                   "target_osm_node": ["2", "3"],
                                   "walking_time_s": [300., 300.],
                                   "length_m": [400., 400.]})
        self.nodes.to_csv(self.paths.nodes, index=False)
        self.edges.to_csv(self.paths.edges, index=False)
        hashval = graph_checksum(self.paths.nodes, self.paths.edges)
        self.attachments = pd.DataFrame([
            dict(entity_id="o1", entity_kind="origin", mode="walk", node_id="1", snapped=True,
                 snap_distance_m=0., attachment_quality="exact", graph_checksum=hashval),
            dict(entity_id="o2", entity_kind="origin", mode="walk", node_id="2", snapped=True,
                 snap_distance_m=0., attachment_quality="exact", graph_checksum=hashval),
            dict(entity_id="s1", entity_kind="service", mode="walk", node_id="3", snapped=True,
                 snap_distance_m=0., attachment_quality="exact", graph_checksum=hashval),
            dict(entity_id="s2", entity_kind="service", mode="walk", node_id="2", snapped=True,
                 snap_distance_m=0., attachment_quality="exact", graph_checksum=hashval),
        ])
        self.save()

    def save(self):
        self.origin.to_csv(self.paths.origins, index=False)
        self.services.to_csv(self.paths.services, index=False)
        self.attachments.to_csv(self.paths.attachments, index=False)

    def inspect(self):
        return inspect_real_snapshot(self.snapshot, self.paths)

    def test_identifies_only_prevalidated_service(self):
        data = self.inspect()
        self.assertEqual(data["origin_count"], 2)
        self.assertEqual(data["selected_eligible_count"], 1)
        self.assertEqual(data["selected_ineligible_count"], 1)
        self.assertEqual(data["selected_eligibility_reasons"]["location_not_validated"], 1)
        self.assertEqual(data["example_removable_services"][0]["service_id"], "s1")
        self.assertFalse(data["scenario_eligible"])

    def test_preflight_does_not_write_anything(self):
        before = {p: p.read_bytes() for p in self.paths.items().values()}
        self.inspect()
        self.assertEqual(before, {p: p.read_bytes() for p in self.paths.items().values()})
        self.assertEqual(len(list(self.paths.nodes.parent.iterdir())), 5)

    def test_paths_derive_from_actual_attachment_output_conventions(self):
        p = canonical_paths(Path("/repo"), self.snapshot)
        self.assertEqual(p.origins.name, "population_network_origins_2023.parquet")
        self.assertEqual(p.services.name, "service_entities_v2_202425_20250630.parquet")
        self.assertEqual(p.attachments.name, "network_attachments_v2_walk_2023_202425_20250630.parquet")
        self.assertEqual(p.nodes.name, "walk_nodes.parquet")

    def test_missing_file_fails_closed(self):
        self.paths.origins.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing canonical"):
            self.inspect()

    def test_graph_changed_fails_closed(self):
        self.edges.loc[0, "length_m"] = 400.5
        self.edges.to_csv(self.paths.edges, index=False)
        with self.assertRaisesRegex(ValueError, "graph checksum"):
            self.inspect()

    def test_wrong_municipality_rejected(self):
        self.origin.loc[0, "municipality_code"] = "016024"
        self.save()
        with self.assertRaisesRegex(ValueError, "Origin municipality"):
            self.inspect()

    def test_missing_mode_attachment_rejected(self):
        self.attachments = self.attachments.loc[self.attachments.entity_id != "o2"]
        self.save()
        with self.assertRaisesRegex(ValueError, "Origin attachments"):
            self.inspect()

    def test_invalid_network_node_rejected(self):
        self.attachments.loc[self.attachments.entity_id == "s1", "node_id"] = "9999"
        self.save()
        with self.assertRaisesRegex(ValueError, "network node not present"):
            self.inspect()

    def test_all_ineligible_does_not_promote(self):
        self.services["legacy_usable_for_accessibility"] = False
        self.save()
        data = self.inspect()
        self.assertEqual(data["selected_eligible_count"], 0)
        self.assertEqual(data["example_removable_services"], [])

    def test_snapshot_validation(self):
        for kw in (dict(municipality_code="123"), dict(municipality_code="034027", health_reference_date="bad"),
                   dict(municipality_code="034027", census_year="20")):
            with self.assertRaises(ValueError):
                CitySnapshot(**kw)
        with self.assertRaises(ValueError):
            inspect_real_snapshot(self.snapshot, self.paths, examples=51)

    def test_no_service_type_rejected(self):
        self.services["service_type"] = "hospital_establishment"
        self.save()
        with self.assertRaisesRegex(ValueError, "No services"):
            self.inspect()


if __name__ == "__main__":
    unittest.main()
