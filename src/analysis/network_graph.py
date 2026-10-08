from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def normalize_node_id(value: Any) -> str | None:
    """Normalize OSM node identifiers exactly as the v2 attachment layer does."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    value = str(value).strip()
    if not value:
        return None
    if value.endswith(".0"):
        value = value[:-2]
    return value


def build_routing_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    travel_time_weight: str,
    distance_weight: str | None = "length_m",
) -> tuple[nx.DiGraph, dict[str, int | str | None]]:
    """Build a deterministic directed routing graph from canonical network tables.

    Parallel OSMnx arcs with the same directed ``u -> v`` pair are collapsed by
    choosing the lexicographically best edge:
      1. minimum travel-time cost;
      2. minimum distance when travel times tie.

    This preserves origin -> service directionality and makes the graph builder
    mode-agnostic: walking can use ``walking_time_s`` and driving can use
    ``free_flow_travel_time_s``.
    """
    if not str(travel_time_weight).strip():
        raise ValueError("travel_time_weight cannot be empty.")
    if distance_weight is not None and not str(distance_weight).strip():
        raise ValueError("distance_weight cannot be empty when provided.")

    required_nodes = {"source_record_id"}
    missing_nodes = sorted(required_nodes - set(nodes.columns))
    if missing_nodes:
        raise ValueError(f"Network nodes missing required columns: {missing_nodes}")

    required_edges = {
        "source_osm_node",
        "target_osm_node",
        travel_time_weight,
    }
    if distance_weight is not None:
        required_edges.add(distance_weight)

    missing_edges = sorted(required_edges - set(edges.columns))
    if missing_edges:
        raise ValueError(f"Network edges missing required columns: {missing_edges}")

    node_ids = [normalize_node_id(value) for value in nodes["source_record_id"]]
    if any(value is None for value in node_ids):
        raise ValueError("Network nodes contain invalid source_record_id values.")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Network source_record_id values must be unique.")

    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    node_id_set = set(node_ids)

    duplicate_pairs = 0
    invalid_endpoints: set[str] = set()

    for row in edges.itertuples(index=False):
        source = normalize_node_id(getattr(row, "source_osm_node"))
        target = normalize_node_id(getattr(row, "target_osm_node"))

        if source is None or target is None:
            raise ValueError("Network edges contain null/invalid endpoints.")

        if source not in node_id_set:
            invalid_endpoints.add(source)
        if target not in node_id_set:
            invalid_endpoints.add(target)
        if source not in node_id_set or target not in node_id_set:
            continue

        try:
            travel_time = float(getattr(row, travel_time_weight))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{travel_time_weight} must contain numeric values."
            ) from exc

        if not np.isfinite(travel_time) or travel_time < 0:
            raise ValueError(
                f"{travel_time_weight} must contain finite non-negative values."
            )

        distance = None
        if distance_weight is not None:
            try:
                distance = float(getattr(row, distance_weight))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{distance_weight} must contain numeric values."
                ) from exc

            if not np.isfinite(distance) or distance < 0:
                raise ValueError(
                    f"{distance_weight} must contain finite non-negative values."
                )

        candidate_pair = (
            travel_time,
            distance if distance is not None else 0.0,
        )

        if graph.has_edge(source, target):
            duplicate_pairs += 1
            current = graph[source][target]
            current_pair = (
                float(current[travel_time_weight]),
                (
                    float(current[distance_weight])
                    if distance_weight is not None
                    else 0.0
                ),
            )
            if candidate_pair >= current_pair:
                continue

        attributes: dict[str, float] = {
            travel_time_weight: travel_time,
        }
        if distance_weight is not None:
            attributes[distance_weight] = float(distance)
        graph.add_edge(source, target, **attributes)

    if invalid_endpoints:
        raise ValueError(
            "Network edges reference nodes absent from the node table. "
            f"Total={len(invalid_endpoints)}; "
            f"examples={sorted(invalid_endpoints)[:20]}"
        )

    if graph.number_of_edges() == 0:
        raise ValueError("Routing graph contains no valid directed edges.")

    summary: dict[str, int | str | None] = {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "input_edges": int(len(edges)),
        "duplicate_directed_pairs_collapsed": int(duplicate_pairs),
        "travel_time_weight": travel_time_weight,
        "distance_weight": distance_weight,
    }
    return graph, summary
