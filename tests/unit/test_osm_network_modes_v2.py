import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import networkx as nx
from shapely.geometry import Polygon

from ingestion import osm_network


class OsmNetworkModesV2Tests(unittest.TestCase):
    def _graph(self, *, maxspeed="50", highway="residential"):
        graph = nx.MultiDiGraph()
        graph.graph["crs"] = "EPSG:4326"
        graph.add_node(1, x=16.0, y=40.0, street_count=1)
        graph.add_node(2, x=16.001, y=40.0, street_count=1)
        graph.add_edge(
            1,
            2,
            key=0,
            length=100.0,
            highway=highway,
            oneway=True,
            osmid=123,
            maxspeed=maxspeed,
        )
        return graph

    def test_cli_default_mode_remains_walk(self):
        with mock.patch.object(
            sys,
            "argv",
            ["osm_network.py", "--municipality-code", "034027"],
        ):
            args = osm_network.parse_args()

        self.assertEqual(args.mode, "walk")
        self.assertEqual(args.walking_speed, 1.4)
        self.assertIsNone(args.drive_fallback_speed_kph)

    def test_walk_canonical_contract_is_legacy_compatible(self):
        nodes, edges = osm_network.build_canonical_network(
            self._graph(),
            walking_speed=1.4,
        )

        self.assertEqual(
            list(edges.columns),
            [
                "source_record_id",
                "source_osm_node",
                "target_osm_node",
                "geometry",
                "length_m",
                "walking_time_s",
                "road_type",
                "oneway_normalized",
                "osm_way_ids",
                "attributes",
            ],
        )
        self.assertAlmostEqual(
            float(edges.loc[0, "walking_time_s"]),
            100.0 / 1.4,
            places=12,
        )
        self.assertNotIn("free_flow_travel_time_s", edges.columns)
        self.assertEqual(len(nodes), 2)

    def test_drive_canonical_contract_uses_free_flow_time(self):
        _, edges = osm_network.build_canonical_network(
            self._graph(maxspeed="50"),
            mode="drive",
        )

        self.assertNotIn("walking_time_s", edges.columns)
        self.assertIn("speed_kph", edges.columns)
        self.assertIn("speed_source", edges.columns)
        self.assertIn("maxspeed_raw", edges.columns)
        self.assertIn("speed_highway_class", edges.columns)
        self.assertIn("free_flow_travel_time_s", edges.columns)
        self.assertEqual(float(edges.loc[0, "speed_kph"]), 50.0)
        self.assertEqual(edges.loc[0, "speed_source"], "osm_maxspeed")
        self.assertAlmostEqual(
            float(edges.loc[0, "free_flow_travel_time_s"]),
            7.2,
            places=12,
        )

        osm_network.validate_canonical_network(
            *_canonical_pair(self._graph(maxspeed="50"), mode="drive"),
            mode="drive",
        )

    def test_graphml_and_silver_outputs_are_mode_specific(self):
        graph = self._graph()
        walk_nodes, walk_edges = osm_network.build_canonical_network(
            graph,
            walking_speed=1.4,
            mode="walk",
        )
        drive_nodes, drive_edges = osm_network.build_canonical_network(
            graph,
            mode="drive",
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with mock.patch.object(osm_network, "RAW_OSM_DIR", tmp_path / "raw"):
                self.assertEqual(
                    osm_network.graphml_path("034027", mode="walk").name,
                    "walk_network.graphml",
                )
                self.assertEqual(
                    osm_network.graphml_path("034027", mode="drive").name,
                    "drive_network.graphml",
                )

            with mock.patch.object(
                osm_network,
                "PROCESSED_OSM_DIR",
                tmp_path / "processed",
            ):
                walk_paths = osm_network.save_silver_datasets(
                    walk_nodes,
                    walk_edges,
                    "034027",
                    mode="walk",
                )
                drive_paths = osm_network.save_silver_datasets(
                    drive_nodes,
                    drive_edges,
                    "034027",
                    mode="drive",
                )

                self.assertEqual(walk_paths[0].name, "walk_nodes.parquet")
                self.assertEqual(walk_paths[1].name, "walk_edges.parquet")
                self.assertEqual(drive_paths[0].name, "drive_nodes.parquet")
                self.assertEqual(drive_paths[1].name, "drive_edges.parquet")
                for path in (*walk_paths, *drive_paths):
                    self.assertTrue(path.exists())

    def test_download_uses_requested_osmnx_network_type(self):
        dummy = self._graph()
        boundary = Polygon(
            [(16.0, 40.0), (16.01, 40.0), (16.01, 40.01), (16.0, 40.0)]
        )

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(osm_network, "RAW_OSM_DIR", Path(tmp)):
                with mock.patch.object(osm_network, "configure_osmnx"):
                    with mock.patch.object(
                        osm_network.ox.graph,
                        "graph_from_polygon",
                        return_value=dummy,
                    ) as download:
                        with mock.patch.object(osm_network.ox.io, "save_graphml"):
                            result = osm_network.load_or_download_graph(
                                boundary,
                                "034027",
                                refresh=True,
                                mode="drive",
                            )

        self.assertIs(result, dummy)
        self.assertEqual(
            download.call_args.kwargs["network_type"],
            "drive",
        )

    def test_drive_postgis_write_is_explicitly_blocked(self):
        with self.assertRaises(ValueError):
            osm_network.write_to_postgis(
                engine=None,
                nodes=None,
                edges=None,
                mode="drive",
            )


def _canonical_pair(graph, *, mode):
    return osm_network.build_canonical_network(
        graph,
        walking_speed=1.4,
        mode=mode,
    )


if __name__ == "__main__":
    unittest.main()
