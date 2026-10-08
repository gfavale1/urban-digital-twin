from __future__ import annotations

import heapq
import itertools
from typing import Any

import networkx as nx

from analysis.routing_engine import RoutingCosts


class NetworkXRoutingBackend:
    """NetworkX graph backend with origin -> target direction semantics.

    For directed graphs, target-oriented routing runs on a reverse graph view.
    When a distance weight is requested, distance is accumulated along the
    *same route that minimizes travel time*. This is essential for drive graphs:
    the shortest-distance route can differ from the free-flow fastest route.
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.routing_graph = graph.reverse(copy=False) if graph.is_directed() else graph

    def _time_and_distance_to_target(
        self,
        target_node: Any,
        *,
        travel_time_weight: str,
        distance_weight: str,
    ) -> tuple[dict[Any, float], dict[Any, float]]:
        # Lexicographic Dijkstra:
        #   primary objective   = minimum travel time
        #   deterministic tie   = minimum distance
        # The returned distance is therefore the distance of the selected
        # fastest route, not a separate shortest-distance route.
        best_time: dict[Any, float] = {target_node: 0.0}
        best_distance: dict[Any, float] = {target_node: 0.0}

        counter = itertools.count()
        queue: list[tuple[float, float, int, Any]] = [
            (0.0, 0.0, next(counter), target_node)
        ]

        while queue:
            current_time, current_distance, _, node = heapq.heappop(queue)

            stored_pair = (
                best_time.get(node, float("inf")),
                best_distance.get(node, float("inf")),
            )
            if (current_time, current_distance) != stored_pair:
                continue

            for neighbor, attributes in self.routing_graph[node].items():
                edge_time = float(attributes[travel_time_weight])
                edge_distance = float(attributes[distance_weight])

                candidate_time = current_time + edge_time
                candidate_distance = current_distance + edge_distance
                candidate_pair = (candidate_time, candidate_distance)

                known_pair = (
                    best_time.get(neighbor, float("inf")),
                    best_distance.get(neighbor, float("inf")),
                )

                if candidate_pair < known_pair:
                    best_time[neighbor] = candidate_time
                    best_distance[neighbor] = candidate_distance
                    heapq.heappush(
                        queue,
                        (
                            candidate_time,
                            candidate_distance,
                            next(counter),
                            neighbor,
                        ),
                    )

        return best_time, best_distance

    def costs_to_target(
        self,
        target_node: Any,
        *,
        travel_time_weight: str,
        distance_weight: str | None,
    ) -> RoutingCosts:
        if target_node not in self.graph:
            raise ValueError(
                f"Target node {target_node!r} is absent from the routing graph."
            )

        if distance_weight is None:
            travel_time = nx.single_source_dijkstra_path_length(
                self.routing_graph,
                source=target_node,
                weight=travel_time_weight,
            )
            distance = None
        else:
            travel_time, distance = self._time_and_distance_to_target(
                target_node,
                travel_time_weight=travel_time_weight,
                distance_weight=distance_weight,
            )

        return RoutingCosts(
            travel_time_s_by_node=travel_time,
            distance_m_by_node=distance,
        )
