import unittest

import pandas as pd

from analysis.network_graph import build_routing_graph


class NetworkGraphV2Tests(unittest.TestCase):
    def _nodes(self):
        return pd.DataFrame(
            {
                "source_record_id": ["1", "2", "3"],
            }
        )

    def test_builds_directed_graph_with_requested_weights(self):
        edges = pd.DataFrame(
            {
                "source_osm_node": ["1", "2"],
                "target_osm_node": ["2", "3"],
                "free_flow_travel_time_s": [10.0, 20.0],
                "length_m": [100.0, 200.0],
            }
        )

        graph, summary = build_routing_graph(
            self._nodes(),
            edges,
            travel_time_weight="free_flow_travel_time_s",
            distance_weight="length_m",
        )

        self.assertTrue(graph.is_directed())
        self.assertTrue(graph.has_edge("1", "2"))
        self.assertFalse(graph.has_edge("2", "1"))
        self.assertEqual(
            float(graph["1"]["2"]["free_flow_travel_time_s"]),
            10.0,
        )
        self.assertEqual(summary["nodes"], 3)
        self.assertEqual(summary["edges"], 2)

    def test_parallel_directed_edges_keep_fastest_then_shortest(self):
        edges = pd.DataFrame(
            {
                "source_osm_node": ["1", "1", "1"],
                "target_osm_node": ["2", "2", "2"],
                "free_flow_travel_time_s": [12.0, 10.0, 10.0],
                "length_m": [50.0, 150.0, 120.0],
            }
        )

        graph, summary = build_routing_graph(
            self._nodes(),
            edges,
            travel_time_weight="free_flow_travel_time_s",
            distance_weight="length_m",
        )

        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(
            float(graph["1"]["2"]["free_flow_travel_time_s"]),
            10.0,
        )
        self.assertEqual(float(graph["1"]["2"]["length_m"]), 120.0)
        self.assertEqual(summary["duplicate_directed_pairs_collapsed"], 2)

    def test_opposite_directions_remain_distinct(self):
        edges = pd.DataFrame(
            {
                "source_osm_node": ["1", "2"],
                "target_osm_node": ["2", "1"],
                "free_flow_travel_time_s": [10.0, 30.0],
                "length_m": [100.0, 100.0],
            }
        )

        graph, _ = build_routing_graph(
            self._nodes(),
            edges,
            travel_time_weight="free_flow_travel_time_s",
            distance_weight="length_m",
        )

        self.assertEqual(graph.number_of_edges(), 2)
        self.assertEqual(
            float(graph["1"]["2"]["free_flow_travel_time_s"]),
            10.0,
        )
        self.assertEqual(
            float(graph["2"]["1"]["free_flow_travel_time_s"]),
            30.0,
        )

    def test_rejects_missing_requested_weight(self):
        edges = pd.DataFrame(
            {
                "source_osm_node": ["1"],
                "target_osm_node": ["2"],
                "length_m": [100.0],
            }
        )

        with self.assertRaises(ValueError):
            build_routing_graph(
                self._nodes(),
                edges,
                travel_time_weight="free_flow_travel_time_s",
                distance_weight="length_m",
            )

    def test_rejects_unknown_endpoint(self):
        edges = pd.DataFrame(
            {
                "source_osm_node": ["1"],
                "target_osm_node": ["999"],
                "free_flow_travel_time_s": [10.0],
                "length_m": [100.0],
            }
        )

        with self.assertRaises(ValueError):
            build_routing_graph(
                self._nodes(),
                edges,
                travel_time_weight="free_flow_travel_time_s",
                distance_weight="length_m",
            )


if __name__ == "__main__":
    unittest.main()
