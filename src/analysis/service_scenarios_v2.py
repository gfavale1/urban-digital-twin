"""B6C.1: non-destructive service overlays and paired v2 access comparison.

Pure, in-memory execution. This module does not certify a candidate source,
verify files from paths, touch baseline files, or alter a transport graph.
Input byte checksums and independently reviewed eligibility remain caller duties.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from analysis.accessibility_contracts import compute_accessibility_from_canonical, prepare_service_destinations_v2
from analysis.accessibility_engine import AccessibilityEngineRequest, AccessibilityEngineResult
from analysis.scenario_contracts_v2 import (
    ScenarioAction, ScenarioSpec, preflight_scenario,
)
from core.analysis_spec import AnalysisSpec
from core.schema_v2 import (
    NETWORK_ATTACHMENT_V2, SERVICE_V2, has_service_column,
    opportunity_count_column, population_coverage_column,
)

POLICY = "b6c1_service_overlay_comparison_v1"


def _normalize(value: Any) -> Any:
    """Stable JSON encoding for canonical row values, including geometric points."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        if math.isnan(val):
            return None
        if not math.isfinite(val):
            raise ValueError("Non-finite value in scenario payload")
        return val
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(val) for val in value]
    if hasattr(value, "wkb_hex"):
        return {"geometry_wkb_hex": value.wkb_hex}
    if isinstance(value, np.ndarray):
        return [_normalize(val) for val in value.tolist()]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    raise ValueError(f"Unsupported scenario payload value: {type(value).__name__}")


def scenario_row_sha256(row: Mapping[str, Any] | pd.Series) -> str:
    """Hash complete row content, not only the service ID or geographic point."""
    document = {str(k): _normalize(v) for k, v in dict(row).items()}
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_verified_inputs(scenario: ScenarioSpec, hashes: Mapping[str, str] | None) -> None:
    """Require caller-reported, already verified file hashes; does not verify bytes."""
    if hashes is None:
        raise ValueError("Verified baseline input hashes are required")
    expected = {
        "origins": scenario.baseline.origins_sha256,
        "services": scenario.baseline.services_sha256,
        "attachments": scenario.baseline.attachments_sha256,
        "graph": scenario.baseline.graph_sha256,
    }
    if dict(hashes) != expected:
        raise ValueError("Verified baseline input hashes differ from ScenarioBaselineRef")


def _selected_additions(
    scenario: ScenarioSpec,
    baseline_services: pd.DataFrame,
    addition_services: pd.DataFrame | None,
    addition_attachments: pd.DataFrame | None,
    graph: nx.Graph,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    additions = {op.service_id: op for op in scenario.operations if op.action is ScenarioAction.ADD_SERVICE}
    if not additions:
        if addition_services is not None and not addition_services.empty:
            raise ValueError("Unexpected service addition payload")
        if addition_attachments is not None and not addition_attachments.empty:
            raise ValueError("Unexpected attachment addition payload")
        return baseline_services.iloc[:0].copy(), pd.DataFrame(columns=[])
    if addition_services is None or addition_attachments is None:
        raise ValueError("Addition requires a separate verified service and attachment catalogue")
    SERVICE_V2.validate_columns(addition_services.columns)
    NETWORK_ATTACHMENT_V2.validate_columns(addition_attachments.columns)
    if addition_services["service_id"].isna().any() or addition_services["service_id"].astype(str).duplicated().any():
        raise ValueError("Addition catalogue requires unique non-null service IDs")
    if set(addition_services["service_id"].astype(str)) != set(additions):
        raise ValueError("Addition catalogue does not match declared operation IDs")
    if not all(isinstance(x, str) for x in addition_services["service_id"]):
        raise ValueError("New service IDs must be strings")
    if set(addition_services["service_id"]) & set(baseline_services["service_id"].astype(str)):
        raise ValueError("Scenario addition cannot reuse a baseline service ID")
    # Detect source-identity duplication even when someone changes the canonical ID.
    identity_cols = ["service_type", "source_name", "source_record_id"]
    if addition_services[identity_cols].isna().any().any() or (
        addition_services[identity_cols].astype(str).apply(lambda x: x.str.strip().eq("")).any().any()
    ):
        raise ValueError("New services require a nonblank authoritative source identity")
    if addition_services.duplicated(subset=identity_cols).any():
        raise ValueError("Addition catalogue duplicates a source identity")
    old_identities = set(map(tuple, baseline_services[identity_cols].dropna().astype(str).to_numpy()))
    new_identities = set(map(tuple, addition_services[identity_cols].astype(str).to_numpy()))
    if old_identities & new_identities:
        raise ValueError("Added service duplicates an existing canonical source identity")
    if addition_attachments["entity_id"].isna().any() or addition_attachments.duplicated(subset=["entity_kind", "entity_id", "mode"]).any():
        raise ValueError("Addition attachments have null/duplicate IDs")
    if len(addition_attachments) != len(additions):
        raise ValueError("Exactly one attachment for each added service is required")
    if not addition_attachments["entity_kind"].eq("service").all() or not addition_attachments["mode"].eq(scenario.mode.value).all():
        raise ValueError("Added attachments must be service rows for the scenario mode")
    if set(addition_attachments["entity_id"].astype(str)) != set(additions):
        raise ValueError("Added attachment catalogue does not match operation IDs")

    services_by_id = addition_services.set_index("service_id", drop=False)
    att_by_id = addition_attachments.set_index("entity_id", drop=False)
    for sid, op in additions.items():
        sr = services_by_id.loc[sid]
        ar = att_by_id.loc[sid]
        if scenario_row_sha256(sr) != op.service_row_sha256 or scenario_row_sha256(ar) != op.attachment_row_sha256:
            raise ValueError(f"Declared addition payload hash mismatch: {sid}")
        if str(sr["municipality_code"]) != scenario.baseline.municipality_code:
            raise ValueError("Added service lies outside the chosen municipality")
        if str(sr["service_type"]) != scenario.service_type.value:
            raise ValueError("Added service type differs from the scenario")
        if "graph_checksum" not in ar.index or ar["graph_checksum"] != scenario.baseline.graph_sha256:
            raise ValueError("Added attachment graph checksum mismatch")

    # B6A3 fail-closed gate is authoritative for routing, not the scenario input.
    gated = prepare_service_destinations_v2(
        addition_services.copy(deep=True), addition_attachments.copy(deep=True), mode=scenario.mode,
    )
    if not gated["routing_eligible"].all():
        reasons = gated.loc[~gated["routing_eligible"], ["service_id", "routing_exclusion_reason"]].to_dict("records")
        raise ValueError(f"Added services are not independently location-validated/eligible: {reasons}")
    if not set(gated["network_node_id"].astype(str)).issubset(set(graph.nodes)):
        raise ValueError("Added service attachment node missing from graph")
    return addition_services.copy(deep=True), addition_attachments.copy(deep=True)


def _overlay(
    scenario: ScenarioSpec,
    services: pd.DataFrame,
    attachments: pd.DataFrame,
    additions: pd.DataFrame,
    added_attachments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    removed = {op.service_id for op in scenario.operations if op.action is ScenarioAction.REMOVE_SERVICE}
    remaining_services = services.loc[~services["service_id"].astype(str).isin(removed)].copy(deep=True)
    remaining_attachments = attachments.loc[
        ~((attachments["entity_kind"] == "service") & attachments["entity_id"].astype(str).isin(removed))
    ].copy(deep=True)
    new_services = pd.concat([remaining_services, additions], ignore_index=True)
    new_attachments = pd.concat([remaining_attachments, added_attachments], ignore_index=True)
    if new_services["service_id"].astype(str).duplicated().any():
        raise ValueError("Overlay produced duplicate canonical service IDs")
    return new_services, new_attachments


def _result_metrics(result: AccessibilityEngineResult, request: AccessibilityEngineRequest) -> dict[str, float | int | None]:
    df = result.origins
    pop = pd.to_numeric(df["target_population"], errors="raise").astype(float)
    time = pd.to_numeric(df["nearest_service_time_min"], errors="coerce").astype(float)
    reachable = time.notna()
    reachable_pop = float(pop.loc[reachable].sum())
    mean_time = float((time.loc[reachable] * pop.loc[reachable]).sum() / reachable_pop) if reachable_pop > 0 else None
    m: dict[str, float | int | None] = {
        "service_count_total": int(result.summary["service_count_total"]),
        "service_count_routable": int(result.summary["service_count_routable"]),
        "target_population_total": float(pop.sum()),
        "reachable_population": reachable_pop,
        "mean_nearest_time_reachable_min": mean_time,
    }
    for threshold in request.thresholds_min:
        m[population_coverage_column(threshold)] = result.summary[population_coverage_column(threshold)]
        m[f"population_weighted_cumulative_opportunities_within_{threshold}_min"] = (
            float((df[opportunity_count_column(threshold)].astype(float) * pop).sum() / pop.sum())
            if float(pop.sum()) > 0 else None
        )
    return m


def _output_sha256(frame: pd.DataFrame) -> str:
    rows = frame.sort_values("origin_id", kind="mergesort").to_dict("records")
    return scenario_row_sha256({"records": rows})


def _diff_metrics(a: Mapping[str, float | int | None], b: Mapping[str, float | int | None]) -> dict[str, float | None]:
    return {k: (float(b[k]) - float(v)) if v is not None and b[k] is not None else None for k, v in a.items()}


def _compare_origins(baseline: pd.DataFrame, scenario: pd.DataFrame,
                     request: AccessibilityEngineRequest) -> pd.DataFrame:
    if baseline["origin_id"].astype(str).duplicated().any() or scenario["origin_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate origin IDs in accessibility results")
    left = baseline.set_index("origin_id").sort_index()
    right = scenario.set_index("origin_id").sort_index()
    if not left.index.equals(right.index) or not left["target_population"].equals(right["target_population"]):
        raise ValueError("Scenario and baseline have different origins or target population")
    frame = pd.DataFrame(index=left.index)
    frame["target_population"] = left["target_population"]
    for label, data in (("baseline", left), ("scenario", right)):
        frame[f"{label}_nearest_time_min"] = data["nearest_service_time_min"]
        frame[f"{label}_reachability_status"] = data["reachability_status"]
        for threshold in request.thresholds_min:
            frame[f"{label}_{opportunity_count_column(threshold)}"] = data[opportunity_count_column(threshold)]
    a = pd.to_numeric(left["nearest_service_time_min"], errors="coerce")
    b = pd.to_numeric(right["nearest_service_time_min"], errors="coerce")
    frame["delta_nearest_time_min"] = (b - a).astype("Float64")
    for threshold in request.thresholds_min:
        col = opportunity_count_column(threshold)
        frame[f"delta_{col}"] = right[col].astype(int) - left[col].astype(int)
    return frame.reset_index()


def _assert_monotonicity(comp: pd.DataFrame, request: AccessibilityEngineRequest, *, only_adds: bool, only_removes: bool) -> None:
    if not (only_adds or only_removes):
        return
    a = pd.to_numeric(comp["baseline_nearest_time_min"], errors="coerce")
    b = pd.to_numeric(comp["scenario_nearest_time_min"], errors="coerce")
    if only_adds:
        if ((a.notna()) & (b.isna())).any() or ((a.notna()) & (b.notna()) & (b > a + 1e-9)).any():
            raise AssertionError("Adding services worsened nearest travel time/reachability")
    if only_removes:
        if ((a.isna()) & (b.notna())).any() or ((a.notna()) & (b.notna()) & (b < a - 1e-9)).any():
            raise AssertionError("Removing services improved nearest travel time/reachability")
    for threshold in request.thresholds_min:
        delta = comp[f"delta_{opportunity_count_column(threshold)}"]
        if (only_adds and (delta < 0).any()) or (only_removes and (delta > 0).any()):
            raise AssertionError("Cumulative opportunities violate scenario monotonicity")


@dataclass(frozen=True, slots=True)
class ScenarioExecutionResult:
    scenario_id: str
    baseline: AccessibilityEngineResult
    scenario: AccessibilityEngineResult
    per_origin: pd.DataFrame
    metrics: dict[str, dict[str, float | int | None]]
    provenance: dict[str, Any]


def execute_service_scenario(
    *, scenario: ScenarioSpec, analysis_spec: AnalysisSpec,
    request: AccessibilityEngineRequest, graph: nx.Graph,
    origins: pd.DataFrame, services: pd.DataFrame, attachments: pd.DataFrame,
    verified_input_sha256: Mapping[str, str] | None,
    addition_services: pd.DataFrame | None = None,
    addition_attachments: pd.DataFrame | None = None,
) -> ScenarioExecutionResult:
    """Execute a paired baseline/overlay with identical request, graph, origins.

    Verified input SHA-256 values must be computed from the actual source files
    by the caller; passing claimed values is not an independent verification.
    """
    _check_verified_inputs(scenario, verified_input_sha256)
    preflight_scenario(scenario, analysis_spec, request, graph, origins, services, attachments)
    additions, added_attachments = _selected_additions(
        scenario, services, addition_services, addition_attachments, graph,
    )
    overlay_services, overlay_attachments = _overlay(
        scenario, services, attachments, additions, added_attachments,
    )
    # Deterministic tie order (the engine uses first service on exact ties).
    baseline_services = services.sort_values("service_id", kind="mergesort").reset_index(drop=True)
    overlay_services = overlay_services.sort_values("service_id", kind="mergesort").reset_index(drop=True)
    base = compute_accessibility_from_canonical(
        graph, origins, baseline_services, attachments, request,
    )
    modified = compute_accessibility_from_canonical(
        graph, origins, overlay_services, overlay_attachments, request,
    )
    per_origin = _compare_origins(base.origins, modified.origins, request)
    only_adds = all(op.action is ScenarioAction.ADD_SERVICE for op in scenario.operations)
    only_removes = all(op.action is ScenarioAction.REMOVE_SERVICE for op in scenario.operations)
    _assert_monotonicity(per_origin, request, only_adds=only_adds, only_removes=only_removes)
    base_metrics = _result_metrics(base, request)
    new_metrics = _result_metrics(modified, request)
    if base_metrics["target_population_total"] != new_metrics["target_population_total"]:
        raise AssertionError("Scenario population differs from baseline")
    output_hashes = {
        "baseline_origins": _output_sha256(base.origins),
        "scenario_origins": _output_sha256(modified.origins),
        "comparison_origins": _output_sha256(per_origin),
        "metrics": scenario_row_sha256({"baseline": base_metrics, "scenario": new_metrics,
                                        "delta": _diff_metrics(base_metrics, new_metrics)}),
    }
    return ScenarioExecutionResult(
        scenario_id=scenario.scenario_id,
        baseline=base, scenario=modified, per_origin=per_origin,
        metrics={"baseline": base_metrics, "scenario": new_metrics,
                 "delta": _diff_metrics(base_metrics, new_metrics)},
        provenance={
            "policy": POLICY, "scenario_id": scenario.scenario_id,
            "scenario_sha256": scenario.scenario_sha256,
            "baseline": scenario.baseline.to_dict(),
            "operations": scenario.to_dict()["operations"],
            "eligible_addition_ids": sorted(additions["service_id"].astype(str).tolist()) if not additions.empty else [],
            "input_hashes": dict(verified_input_sha256),
            "input_checksum_verification": "caller_responsibility_not_rechecked",
            "output_sha256": output_hashes,
            "outputs_persisted": False,
        },
    )
