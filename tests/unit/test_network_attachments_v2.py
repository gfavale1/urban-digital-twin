import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from core.analysis_spec import TransportMode
from transformation.build_network_attachments_v2 import (
    build_network_attachments,
    graph_checksum,
    prepare_service_points,
    snap_entities,
)


class NetworkAttachmentsV2Tests(unittest.TestCase):
    def setUp(self):
        self.nodes = gpd.GeoDataFrame(
            {
                "source_record_id": ["1", "2", "3"],
                "geometry": [
                    Point(12.0000, 45.0000),
                    Point(12.0010, 45.0000),
                    Point(12.0100, 45.0000),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        self.edges = gpd.GeoDataFrame(
            {
                "source_osm_node": ["1", "2"],
                "target_osm_node": ["2", "1"],
                "geometry": [
                    LineString([(12.0000, 45.0000), (12.0010, 45.0000)]),
                    LineString([(12.0010, 45.0000), (12.0000, 45.0000)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        self.origins = gpd.GeoDataFrame(
            {
                "origin_id": ["o1", "o2"],
                "municipality_code": ["000001", "000001"],
                "population_total": [10.0, 20.0],
                "population_reference_date": ["2023-12-31", "2023-12-31"],
                "census_year": ["2023", "2023"],
                "geography_version": ["test", "test"],
                "origin_method": ["test", "test"],
                "geometry": [Point(12.0000, 45.0000), Point(12.0100, 45.0000)],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        self.services = gpd.GeoDataFrame(
            {
                "service_id": ["s1", "s2"],
                "service_type": ["pharmacy", "pharmacy"],
                "name": ["A", "B"],
                "municipality_code": ["000001", "000001"],
                "source_name": ["test", "test"],
                "source_record_id": ["a", "b"],
                "source_reference_date": ["2025-01-01", "2025-01-01"],
                "retrieved_at": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
                "operational_status": ["active", "active"],
                "latitude": [45.0000, None],
                "longitude": [12.0010, None],
                "geometry": [Point(12.0010, 45.0000), None],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

    def test_builds_one_attachment_per_origin_and_service(self):
        attachments, summary = build_network_attachments(
            self.origins,
            self.services,
            self.nodes,
            self.edges,
            mode=TransportMode.DRIVE,
            max_snap_distance_m=1000.0,
            graph_checksum_value="abc",
        )
        self.assertEqual(len(attachments), 4)
        self.assertFalse(
            attachments.duplicated(["entity_kind", "entity_id", "mode"]).any()
        )
        self.assertEqual(summary["component_count"], 2)
        self.assertEqual(summary["largest_component_size"], 2)

    def test_mode_is_propagated(self):
        attachments, _ = build_network_attachments(
            self.origins,
            self.services,
            self.nodes,
            self.edges,
            mode=TransportMode.DRIVE,
            max_snap_distance_m=1000.0,
            graph_checksum_value="abc",
        )
        self.assertEqual(set(attachments["mode"]), {"drive"})

    def test_missing_service_geometry_is_explicit_unsnapped(self):
        attachments, _ = build_network_attachments(
            self.origins,
            self.services,
            self.nodes,
            self.edges,
            mode=TransportMode.WALK,
            max_snap_distance_m=1000.0,
            graph_checksum_value="abc",
        )
        row = attachments.loc[attachments["entity_id"] == "s2"].iloc[0]
        self.assertFalse(bool(row["snapped"]))
        self.assertTrue(pd.isna(row["node_id"]))
        self.assertEqual(row["attachment_quality"], "missing_geometry")

    def test_service_geometry_can_be_rebuilt_from_coordinates(self):
        services = self.services.copy()
        services.loc[services["service_id"] == "s1", "geometry"] = None
        prepared = prepare_service_points(services)
        row = prepared.loc[prepared["entity_id"] == "s1"].iloc[0]
        self.assertIsNotNone(row.geometry)
        self.assertEqual(row.geometry.geom_type, "Point")

    def test_snap_outlier_preserves_distance_but_clears_node(self):
        far = gpd.GeoDataFrame(
            {"entity_id": ["x"], "geometry": [Point(12.0200, 45.0000)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        from transformation.build_network_attachments_v2 import (
            build_network_components,
            prepare_network_nodes,
        )
        prepared_nodes = prepare_network_nodes(self.nodes)
        nodes_with_components, _ = build_network_components(prepared_nodes, self.edges)
        result = snap_entities(
            far,
            nodes_with_components,
            entity_kind="origin",
            mode=TransportMode.DRIVE,
            max_snap_distance_m=10.0,
            graph_checksum_value="abc",
        )
        row = result.iloc[0]
        self.assertFalse(bool(row["snapped"]))
        self.assertIsNone(row["node_id"])
        self.assertGreater(float(row["snap_distance_m"]), 10.0)
        self.assertEqual(row["attachment_quality"], "snap_outlier")

    def test_component_metadata_follows_snapped_node(self):
        attachments, _ = build_network_attachments(
            self.origins.iloc[[0]].copy(),
            self.services.iloc[[0]].copy(),
            self.nodes,
            self.edges,
            mode=TransportMode.WALK,
            max_snap_distance_m=1000.0,
            graph_checksum_value="abc",
        )
        row = attachments.loc[attachments["entity_id"] == "o1"].iloc[0]
        self.assertTrue(bool(row["snapped"]))
        self.assertEqual(row["network_component_id"], "component_000001")
        self.assertTrue(bool(row["is_largest_component"]))

    def test_graph_checksum_changes_when_source_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            nodes = tmp / "nodes.bin"
            edges = tmp / "edges.bin"
            nodes.write_bytes(b"nodes-a")
            edges.write_bytes(b"edges-a")
            first = graph_checksum(nodes, edges)
            edges.write_bytes(b"edges-b")
            second = graph_checksum(nodes, edges)
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
