"""B6C.0: immutable scenario identity and fail-closed baseline preflight.

This phase does not apply an overlay, run routing, write files, or grant new
services routing eligibility. B6C.1 will implement overlay and comparisons.
The supplied file checksums must be independently verified by its caller.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import networkx as nx
import pandas as pd

from analysis.accessibility_contracts import prepare_canonical_accessibility_inputs
from analysis.accessibility_engine import AccessibilityEngineRequest
from core.analysis_spec import AnalysisSpec, ServiceType, TransportMode
from core.schema_v2 import NETWORK_ATTACHMENT_V2, ORIGIN_V2, SERVICE_V2


POLICY = "b6c0_scenario_contract_v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest.")


def request_sha256(request: AccessibilityEngineRequest) -> str:
    """Capture mode/thresholds/weights and off-network cost, not just AnalysisSpec."""
    return _sha256_json({
        "service_type": request.service_type.value,
        "mode": request.mode.value,
        "thresholds_min": list(request.thresholds_min),
        "population_selector": request.population_selector.value,
        "travel_time_weight": request.travel_time_weight,
        "distance_weight": request.distance_weight,
        "off_network_speed_m_s": float(request.off_network_speed_m_s),
    })


@dataclass(frozen=True, slots=True)
class ScenarioBaselineRef:
    municipality_code: str
    analysis_spec_sha256: str
    request_sha256: str
    origins_sha256: str
    services_sha256: str
    attachments_sha256: str
    graph_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.municipality_code, str) or not re.fullmatch(r"\d{6}", self.municipality_code):
            raise ValueError("Scenario municipality_code must be six digits, including leading zeros.")
        for field_name in (
            "analysis_spec_sha256", "request_sha256", "origins_sha256",
            "services_sha256", "attachments_sha256", "graph_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in (
            "municipality_code", "analysis_spec_sha256", "request_sha256",
            "origins_sha256", "services_sha256", "attachments_sha256", "graph_sha256",
        )}


class ScenarioAction(str, Enum):
    ADD_SERVICE = "add_service"
    REMOVE_SERVICE = "remove_service"


@dataclass(frozen=True, slots=True)
class ScenarioOperation:
    action: ScenarioAction
    service_id: str
    # For an addition these identify the *future* canonical row and attachment
    # payload; B6C.1 must verify the bytes and apply the existing B6A3 gate.
    service_row_sha256: str | None = None
    attachment_row_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ScenarioAction(self.action))
        if not isinstance(self.service_id, str) or not self.service_id.strip() or self.service_id != self.service_id.strip():
            raise ValueError("Scenario operation needs a nonblank canonical service_id.")
        if self.action is ScenarioAction.ADD_SERVICE:
            _required_sha256(self.service_row_sha256, "service_row_sha256")
            _required_sha256(self.attachment_row_sha256, "attachment_row_sha256")
        elif self.service_row_sha256 is not None or self.attachment_row_sha256 is not None:
            raise ValueError("remove_service must not carry add_service payload hashes.")

    def to_dict(self) -> dict[str, str]:
        result = {"action": self.action.value, "service_id": self.service_id}
        if self.action is ScenarioAction.ADD_SERVICE:
            result["service_row_sha256"] = self.service_row_sha256
            result["attachment_row_sha256"] = self.attachment_row_sha256
        return result


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    baseline: ScenarioBaselineRef
    service_type: ServiceType
    mode: TransportMode
    operations: tuple[ScenarioOperation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "service_type", ServiceType(self.service_type))
        object.__setattr__(self, "mode", TransportMode(self.mode))
        object.__setattr__(self, "operations", tuple(self.operations))
        if not self.operations or not all(isinstance(op, ScenarioOperation) for op in self.operations):
            raise ValueError("ScenarioSpec requires at least one valid operation.")
        ids = [op.service_id for op in self.operations]
        if len(ids) != len(set(ids)):
            raise ValueError("A service_id may occur only once in a scenario, even across actions.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": POLICY,
            "baseline": self.baseline.to_dict(),
            "service_type": self.service_type.value,
            "mode": self.mode.value,
            # Operations commute on distinct IDs. Sort so caller order does not change identity.
            "operations": sorted((op.to_dict() for op in self.operations),
                                 key=lambda row: (row["service_id"], row["action"])),
        }

    @property
    def scenario_sha256(self) -> str:
        return _sha256_json(self.to_dict())

    @property
    def scenario_id(self) -> str:
        return "b6c0_" + self.scenario_sha256[:16]


@dataclass(frozen=True, slots=True)
class ScenarioPreflightResult:
    scenario_id: str
    scenario_sha256: str
    checked_origins: int
    checked_services: int
    checked_attachments: int


def preflight_scenario(
    scenario: ScenarioSpec,
    analysis_spec: AnalysisSpec,
    request: AccessibilityEngineRequest,
    graph: nx.Graph,
    origins: pd.DataFrame,
    services: pd.DataFrame,
    attachments: pd.DataFrame,
) -> ScenarioPreflightResult:
    """Validate identities and existing supply; NEVER authorize a new service.

    Callers must first verify the baseline input file bytes against the hashes
    in ScenarioBaselineRef and the graph against its source manifest. This
    preflight additionally checks source attachment graph hashes for the mode.
    """
    baseline = scenario.baseline
    if analysis_spec.municipality_code != baseline.municipality_code:
        raise ValueError("AnalysisSpec municipality differs from scenario baseline.")
    if analysis_spec.spec_hash != baseline.analysis_spec_sha256:
        raise ValueError("AnalysisSpec hash differs from scenario baseline.")
    if request_sha256(request) != baseline.request_sha256:
        raise ValueError("Routing request hash differs from scenario baseline.")
    if scenario.service_type is not request.service_type or scenario.mode is not request.mode:
        raise ValueError("Scenario service/mode differs from routing request.")
    matching_specs = [s for s in analysis_spec.services if s.enabled and s.service_type == scenario.service_type]
    if len(matching_specs) != 1 or scenario.mode not in matching_specs[0].modes:
        raise ValueError("Service/mode is not enabled in AnalysisSpec.")
    service_spec = matching_specs[0]
    if (request.thresholds_min != service_spec.thresholds_min_by_mode[scenario.mode.value]
            or request.population_selector != service_spec.population_selector):
        raise ValueError("Request thresholds/population selector disagree with AnalysisSpec.")
    if scenario.mode is TransportMode.WALK and request.off_network_speed_m_s != analysis_spec.walking_speed_m_s:
        raise ValueError("Walking off-network speed disagrees with AnalysisSpec.")

    ORIGIN_V2.validate_columns(origins.columns)
    SERVICE_V2.validate_columns(services.columns)
    NETWORK_ATTACHMENT_V2.validate_columns(attachments.columns)
    if origins.empty or origins["origin_id"].isna().any() or origins["origin_id"].astype(str).duplicated().any():
        raise ValueError("Baseline needs unique nonempty origin IDs.")
    if services["service_id"].isna().any() or services["service_id"].astype(str).duplicated().any():
        raise ValueError("Baseline service IDs must be nonnull and unique.")
    if not origins["municipality_code"].astype(str).eq(baseline.municipality_code).all():
        raise ValueError("Baseline origins must belong to the specified municipality.")

    mode_rows = attachments.loc[attachments["mode"] == scenario.mode.value]
    if mode_rows.empty or "graph_checksum" not in mode_rows.columns:
        raise ValueError("Mode attachments must carry a graph_checksum provenance column.")
    if (mode_rows["graph_checksum"].isna().any()
            or not mode_rows["graph_checksum"].astype(str).eq(baseline.graph_sha256).all()):
        raise ValueError("Attachment graph checksums disagree with the baseline graph.")

    scoped_services = services.loc[
        services["service_type"].astype(str) == scenario.service_type.value
    ].copy()
    requested_ids = set(scoped_services["service_id"].astype(str))
    origin_ids = set(origins["origin_id"].astype(str))
    scoped_attachments = attachments.loc[
        (attachments["entity_kind"] == "origin")
        | ((attachments["entity_kind"] == "service")
           & attachments["entity_id"].astype(str).isin(requested_ids))
    ].copy()
    # Reuse canonical joining + the existing fail-closed B6A3 eligibility gate.
    prepared = prepare_canonical_accessibility_inputs(
        origins, scoped_services, scoped_attachments, mode=scenario.mode,
    )
    referenced_nodes = (set(prepared.origins["network_node_id"].dropna().astype(str))
                        | set(prepared.services["attachment_node_id"].dropna().astype(str)))
    if not referenced_nodes.issubset(set(graph.nodes)):
        raise ValueError("Mode attachment node_id missing from supplied graph.")

    eligible = set(prepared.services.loc[
        prepared.services["routing_eligible"], "service_id"
    ].astype(str))
    for operation in scenario.operations:
        if operation.action is ScenarioAction.ADD_SERVICE:
            if operation.service_id in set(services["service_id"].astype(str)):
                raise ValueError(f"Cannot add existing service_id: {operation.service_id}")
        elif operation.service_id not in eligible:
            raise ValueError(f"Cannot remove an absent or ineligible service: {operation.service_id}")

    return ScenarioPreflightResult(
        scenario_id=scenario.scenario_id,
        scenario_sha256=scenario.scenario_sha256,
        checked_origins=len(origins),
        checked_services=len(scoped_services),
        checked_attachments=len(scoped_attachments.loc[scoped_attachments["mode"] == scenario.mode.value]),
    )
