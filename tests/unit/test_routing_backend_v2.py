import unittest

import networkx as nx

from analysis.networkx_routing import NetworkXRoutingBackend


class NetworkXRoutingBackendV2Tests(unittest.TestCase):
    def test_directed_costs_are_origin_to_target(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B", travel_time_s=60.0, length_m=100.0)
        graph.add_edge("B", "C", travel_time_s=120.0, length_m=200.0)
        graph.add_edge("C", "B", travel_time_s=999.0, length_m=999.0)

        backend = NetworkXRoutingBackend(graph)
        costs = backend.costs_to_target(
            "C",
            travel_time_weight="travel_time_s",
            distance_weight="length_m",
        )

        self.assertEqual(float(costs.travel_time_s_by_node["C"]), 0.0)
        self.assertEqual(float(costs.travel_time_s_by_node["B"]), 120.0)
        self.assertEqual(float(costs.travel_time_s_by_node["A"]), 180.0)
        self.assertEqual(float(costs.distance_m_by_node["A"]), 300.0)

    def test_distance_follows_fastest_route_not_shortest_distance_route(self):
        graph = nx.DiGraph()

        # Fast route A -> B -> T: 20 s, 2000 m.
        graph.add_edge("A", "B", travel_time_s=10.0, length_m=1000.0)
        graph.add_edge("B", "T", travel_time_s=10.0, length_m=1000.0)

        # Short route A -> C -> T: 40 s, 200 m.
        graph.add_edge("A", "C", travel_time_s=20.0, length_m=100.0)
        graph.add_edge("C", "T", travel_time_s=20.0, length_m=100.0)

        backend = NetworkXRoutingBackend(graph)
        costs = backend.costs_to_target(
            "T",
            travel_time_weight="travel_time_s",
            distance_weight="length_m",
        )

        self.assertEqual(float(costs.travel_time_s_by_node["A"]), 20.0)
        self.assertEqual(float(costs.distance_m_by_node["A"]), 2000.0)

    def test_equal_time_prefers_shorter_route_deterministically(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B", travel_time_s=10.0, length_m=500.0)
        graph.add_edge("B", "T", travel_time_s=10.0, length_m=500.0)
        graph.add_edge("A", "C", travel_time_s=10.0, length_m=100.0)
        graph.add_edge("C", "T", travel_time_s=10.0, length_m=100.0)

        backend = NetworkXRoutingBackend(graph)
        costs = backend.costs_to_target(
            "T",
            travel_time_weight="travel_time_s",
            distance_weight="length_m",
        )

        self.assertEqual(float(costs.travel_time_s_by_node["A"]), 20.0)
        self.assertEqual(float(costs.distance_m_by_node["A"]), 200.0)

    def test_undirected_graph_does_not_need_reversal(self):
        graph = nx.Graph()
        graph.add_edge("A", "B", travel_time_s=30.0, length_m=50.0)
        backend = NetworkXRoutingBackend(graph)

        costs = backend.costs_to_target(
            "B",
            travel_time_weight="travel_time_s",
            distance_weight="length_m",
        )

        self.assertEqual(float(costs.travel_time_s_by_node["A"]), 30.0)
        self.assertEqual(float(costs.distance_m_by_node["A"]), 50.0)

    def test_backend_raises_for_unknown_target(self):
        graph = nx.DiGraph()
        graph.add_node("A")
        backend = NetworkXRoutingBackend(graph)

        with self.assertRaises(ValueError):
            backend.costs_to_target(
                "missing",
                travel_time_weight="travel_time_s",
                distance_weight="length_m",
            )


if __name__ == "__main__":
    unittest.main()
