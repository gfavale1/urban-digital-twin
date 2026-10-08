from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RoutingCosts:
    """Shortest-path costs from every reachable origin node to one target.

    The accessibility engine reasons in origin -> service direction. Backends
    are responsible for respecting directed-network semantics when computing
    these target-oriented costs.
    """

    travel_time_s_by_node: Mapping[Any, float]
    distance_m_by_node: Mapping[Any, float] | None = None


@runtime_checkable
class RoutingBackend(Protocol):
    """Minimal routing interface required by the accessibility engine."""

    def costs_to_target(
        self,
        target_node: Any,
        *,
        travel_time_weight: str,
        distance_weight: str | None,
    ) -> RoutingCosts:
        ...
