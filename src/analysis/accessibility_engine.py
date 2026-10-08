from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from analysis.networkx_routing import NetworkXRoutingBackend
from analysis.routing_engine import RoutingBackend
from core.analysis_spec import PopulationSelector, ServiceType, TransportMode
from core.schema_v2 import (
    ACCESSIBILITY_ORIGIN_V2,
    ReachabilityStatus,
    has_service_column,
    opportunity_count_column,
    population_column_for_selector,
    population_coverage_column,
)


@dataclass(frozen=True, slots=True)
class AccessibilityEngineRequest:
    """One service-type / mode accessibility computation.

    The engine is intentionally independent from CLI, file paths and a specific
    routing mode. A caller provides a graph whose travel-time edge attribute is
    named by ``travel_time_weight``.
    """

    service_type: ServiceType
    mode: TransportMode
    thresholds_min: tuple[int, ...]
    population_selector: PopulationSelector
    travel_time_weight: str
    off_network_speed_m_s: float
    distance_weight: str | None = "length_m"

    def __post_init__(self) -> None:
        thresholds = tuple(int(value) for value in self.thresholds_min)
        if not thresholds or tuple(sorted(set(thresholds))) != thresholds:
            raise ValueError(
                "thresholds_min must contain unique positive values in strictly increasing order."
            )
        if any(value <= 0 for value in thresholds):
            raise ValueError("thresholds_min values must be positive.")
        if not self.travel_time_weight.strip():
            raise ValueError("travel_time_weight cannot be empty.")
        if self.distance_weight is not None and not self.distance_weight.strip():
            raise ValueError("distance_weight cannot be empty when provided.")
        if not np.isfinite(float(self.off_network_speed_m_s)) or self.off_network_speed_m_s <= 0:
            raise ValueError("off_network_speed_m_s must be a finite positive value.")
        object.__setattr__(self, "thresholds_min", thresholds)


@dataclass(frozen=True, slots=True)
class AccessibilityEngineResult:
    origins: pd.DataFrame
    summary: dict[str, Any]


def _normalize_node_id(value: Any) -> str | None:
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


def _prepare_origins(
    origins: pd.DataFrame,
    population_selector: PopulationSelector,
) -> pd.DataFrame:
    population_column = population_column_for_selector(population_selector)
    required = {
        "origin_id",
        "network_node_id",
        "origin_snap_distance_m",
        population_column,
    }
    missing = sorted(required - set(origins.columns))
    if missing:
        raise ValueError(f"Accessibility origins missing required columns: {missing}")

    result = origins.copy().reset_index(drop=True)
    if result["origin_id"].isna().any() or result["origin_id"].astype(str).duplicated().any():
        raise ValueError("origin_id must be non-null and unique.")

    result["network_node_id"] = result["network_node_id"].map(_normalize_node_id)

    snap_distance = pd.to_numeric(result["origin_snap_distance_m"], errors="coerce")
    negative_snap = snap_distance.fillna(0.0) < 0
    invalid_snap = snap_distance.notna() & negative_snap
    if invalid_snap.any():
        raise ValueError("origin_snap_distance_m cannot be negative.")
    # Missing snapping distance is acceptable only for unsnapped origins.
    bad_missing_snap = result["network_node_id"].notna() & snap_distance.isna()
    if bad_missing_snap.any():
        raise ValueError("Snapped origins need a numeric origin_snap_distance_m.")
    result["origin_snap_distance_m"] = snap_distance.astype(float)

    target_population = pd.to_numeric(result[population_column], errors="coerce")
    if target_population.isna().any():
        raise ValueError(f"{population_column} contains missing/non-numeric values.")
    if (target_population < 0).any():
        raise ValueError(f"{population_column} cannot contain negative values.")
    result["target_population"] = target_population.astype(float)
    return result


def _prepare_services(
    services: pd.DataFrame,
    service_type: ServiceType,
) -> tuple[pd.DataFrame, int]:
    required = {
        "service_id",
        "service_type",
        "network_node_id",
        "snap_distance_m",
    }
    missing = sorted(required - set(services.columns))
    if missing:
        raise ValueError(f"Accessibility services missing required columns: {missing}")

    selected = services.loc[
        services["service_type"].astype(str) == service_type.value
    ].copy()
    total_service_count = int(len(selected))

    if selected.empty:
        return selected.reset_index(drop=True), total_service_count

    if selected["service_id"].isna().any() or selected["service_id"].astype(str).duplicated().any():
        raise ValueError(
            f"service_id must be non-null and unique within service_type={service_type.value}."
        )

    selected["network_node_id"] = selected["network_node_id"].map(_normalize_node_id)
    snap_distance = pd.to_numeric(selected["snap_distance_m"], errors="coerce")
    negative_snap = snap_distance.fillna(0.0) < 0
    bad_snap = selected["network_node_id"].notna() & (
        snap_distance.isna() | negative_snap
    )
    if bad_snap.any():
        raise ValueError("Snapped services need a non-negative numeric snap_distance_m.")
    selected["snap_distance_m"] = snap_distance.astype(float)
    return selected.reset_index(drop=True), total_service_count


def _validate_graph_membership(
    graph: nx.Graph,
    origins: pd.DataFrame,
    services: pd.DataFrame,
) -> None:
    graph_nodes = set(graph.nodes)

    origin_nodes = set(origins["network_node_id"].dropna())
    missing_origin_nodes = sorted(origin_nodes - graph_nodes)
    if missing_origin_nodes:
        raise ValueError(
            "Origin network_node_id values are absent from the graph: "
            f"{missing_origin_nodes[:20]}"
        )

    service_nodes = set(services["network_node_id"].dropna())
    missing_service_nodes = sorted(service_nodes - graph_nodes)
    if missing_service_nodes:
        raise ValueError(
            "Service network_node_id values are absent from the graph: "
            f"{missing_service_nodes[:20]}"
        )


def _validate_graph_weights(
    graph: nx.Graph,
    request: AccessibilityEngineRequest,
) -> None:
    for source, target, attributes in graph.edges(data=True):
        if request.travel_time_weight not in attributes:
            raise ValueError(
                f"Graph edge ({source}, {target}) is missing travel-time weight "
                f"{request.travel_time_weight!r}."
            )
        try:
            travel_time = float(attributes[request.travel_time_weight])
        except (TypeError, ValueError) as exc:
            raise ValueError("Graph travel-time weights must be numeric.") from exc
        if not np.isfinite(travel_time) or travel_time < 0:
            raise ValueError("Graph travel-time weights must be finite and non-negative.")

        if request.distance_weight is not None:
            if request.distance_weight not in attributes:
                raise ValueError(
                    f"Graph edge ({source}, {target}) is missing distance weight "
                    f"{request.distance_weight!r}."
                )
            try:
                distance = float(attributes[request.distance_weight])
            except (TypeError, ValueError) as exc:
                raise ValueError("Graph distance weights must be numeric.") from exc
            if not np.isfinite(distance) or distance < 0:
                raise ValueError("Graph distance weights must be finite and non-negative.")


def _origin_indices_by_node(origins: pd.DataFrame) -> dict[str, list[int]]:
    result: defaultdict[str, list[int]] = defaultdict(list)
    for index, node_id in enumerate(origins["network_node_id"]):
        if node_id is not None:
            result[node_id].append(index)
    return dict(result)


def compute_accessibility(
    graph: nx.Graph,
    origins: pd.DataFrame,
    services: pd.DataFrame,
    request: AccessibilityEngineRequest,
    *,
    routing_backend: RoutingBackend | None = None,
) -> AccessibilityEngineResult:
    """Compute location-based accessibility for one service type and one mode.

    Travel is evaluated from origin to service. For directed graphs, routing is
    therefore run from every service on a reverse view of the graph, matching
    the proven legacy walking implementation while remaining mode-agnostic.

    The engine deliberately performs no file I/O and no service geocoding or
    snapping. Those are upstream responsibilities.
    """

    prepared_origins = _prepare_origins(origins, request.population_selector)
    prepared_services, total_service_count = _prepare_services(
        services, request.service_type
    )
    _validate_graph_membership(graph, prepared_origins, prepared_services)
    _validate_graph_weights(graph, request)

    row_count = len(prepared_origins)
    nearest_time_s = np.full(row_count, np.inf, dtype=float)
    nearest_distance_m = np.full(row_count, np.inf, dtype=float)
    nearest_service_id = np.full(row_count, None, dtype=object)
    counts = {
        threshold: np.zeros(row_count, dtype=np.int32)
        for threshold in request.thresholds_min
    }

    routable_services = prepared_services.loc[
        prepared_services["network_node_id"].notna()
    ].copy()
    routable_service_count = int(len(routable_services))

    backend = routing_backend or NetworkXRoutingBackend(graph)

    origins_by_node = _origin_indices_by_node(prepared_origins)
    origin_snap_m = prepared_origins["origin_snap_distance_m"].to_numpy(dtype=float)

    for _, service in routable_services.iterrows():
        service_node = service["network_node_id"]
        service_snap_m = float(service["snap_distance_m"])
        service_snap_s = service_snap_m / request.off_network_speed_m_s
        service_id = str(service["service_id"])

        routing_costs = backend.costs_to_target(
            service_node,
            travel_time_weight=request.travel_time_weight,
            distance_weight=request.distance_weight,
        )
        time_lengths = routing_costs.travel_time_s_by_node
        distance_lengths = routing_costs.distance_m_by_node or {}

        for origin_node, row_indices in origins_by_node.items():
            if origin_node not in time_lengths:
                continue

            network_time_s = float(time_lengths[origin_node])
            network_distance_m = (
                float(distance_lengths[origin_node])
                if request.distance_weight is not None
                else np.nan
            )

            for row_index in row_indices:
                origin_snap_distance_m = float(origin_snap_m[row_index])
                origin_snap_s = origin_snap_distance_m / request.off_network_speed_m_s
                total_time_s = origin_snap_s + network_time_s + service_snap_s
                total_distance_m = (
                    origin_snap_distance_m + network_distance_m + service_snap_m
                    if request.distance_weight is not None
                    else np.nan
                )

                if total_time_s < nearest_time_s[row_index]:
                    nearest_time_s[row_index] = total_time_s
                    nearest_distance_m[row_index] = total_distance_m
                    nearest_service_id[row_index] = service_id

                for threshold in request.thresholds_min:
                    if total_time_s <= threshold * 60.0:
                        counts[threshold][row_index] += 1

    reachable = np.isfinite(nearest_time_s)
    output = pd.DataFrame(
        {
            "origin_id": prepared_origins["origin_id"].astype(str).to_numpy(),
            "service_type": request.service_type.value,
            "mode": request.mode.value,
            "population_selector": request.population_selector.value,
            "target_population": prepared_origins["target_population"].to_numpy(dtype=float),
        }
    )

    time_values = pd.array(
        [value / 60.0 if finite else pd.NA for value, finite in zip(nearest_time_s, reachable)],
        dtype="Float64",
    )
    output["nearest_service_time_min"] = time_values

    if request.distance_weight is not None:
        distance_values = pd.array(
            [value if finite else pd.NA for value, finite in zip(nearest_distance_m, reachable)],
            dtype="Float64",
        )
        output["nearest_service_distance_m"] = distance_values

    output["nearest_service_id"] = pd.array(
        [value if finite else pd.NA for value, finite in zip(nearest_service_id, reachable)],
        dtype="string",
    )

    statuses: list[str] = []
    for index, is_reachable in enumerate(reachable):
        if is_reachable:
            statuses.append(ReachabilityStatus.REACHABLE.value)
        elif prepared_origins.loc[index, "network_node_id"] is None:
            statuses.append(ReachabilityStatus.ORIGIN_NOT_SNAPPED.value)
        elif total_service_count == 0:
            statuses.append(ReachabilityStatus.SERVICE_UNAVAILABLE.value)
        elif routable_service_count == 0:
            statuses.append(ReachabilityStatus.SERVICE_NOT_SNAPPED.value)
        else:
            statuses.append(ReachabilityStatus.NO_PATH.value)
    output["reachability_status"] = pd.array(statuses, dtype="string")

    for threshold in request.thresholds_min:
        count_column = opportunity_count_column(threshold)
        has_column = has_service_column(threshold)
        output[count_column] = counts[threshold]
        output[has_column] = counts[threshold] > 0

    ACCESSIBILITY_ORIGIN_V2.validate_columns(output.columns)

    target_population_total = float(output["target_population"].sum())
    summary: dict[str, Any] = {
        "service_type": request.service_type.value,
        "mode": request.mode.value,
        "population_selector": request.population_selector.value,
        "target_population_total": target_population_total,
        "service_count_total": total_service_count,
        "service_count_routable": routable_service_count,
    }

    for threshold in request.thresholds_min:
        has_column = has_service_column(threshold)
        covered_population = float(
            output.loc[output[has_column], "target_population"].sum()
        )
        coverage = (
            covered_population / target_population_total
            if target_population_total > 0
            else None
        )
        summary[population_coverage_column(threshold)] = coverage

    return AccessibilityEngineResult(origins=output, summary=summary)
