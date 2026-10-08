from __future__ import annotations

from typing import Any

import networkx as nx

from analysis.routing_engine import RoutingCosts


class NetworkXRoutingBackend:
    """NetworkX implementation of target-oriented shortest-path routing.

    For directed graphs, accessibility is evaluated from origin to service.
    Running single-source Dijkstra from the service therefore requires a
    reverse graph view. Undirected graphs are used directly.
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.routing_graph = graph.reverse(copy=False) if graph.is_directed() else graph

    def costs_to_target(
        self,
        target_node: Any,
        *,
        travel_time_weight: str,
        distance_weight: str | None,
    ) -> RoutingCosts:
        if target_node not in self.graph:
            raise ValueError(f"Target node {target_node!r} is absent from the routing graph.")

        travel_time = nx.single_source_dijkstra_path_length(
            self.routing_graph,
            source=target_node,
            weight=travel_time_weight,
        )

        distance = None
        if distance_weight is not None:
            distance = nx.single_source_dijkstra_path_length(
                self.routing_graph,
                source=target_node,
                weight=distance_weight,
            )

        return RoutingCosts(
            travel_time_s_by_node=travel_time,
            distance_m_by_node=distance,
        )
