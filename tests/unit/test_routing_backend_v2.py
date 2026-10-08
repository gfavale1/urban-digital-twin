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
