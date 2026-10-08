from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd

from analysis.accessibility_engine import (
    AccessibilityEngineRequest,
    AccessibilityEngineResult,
    compute_accessibility,
)
from core.analysis_spec import ServiceAnalysisSpec, TransportMode
from core.schema_v2 import (
    NETWORK_ATTACHMENT_V2,
    ORIGIN_V2,
    SERVICE_V2,
)


@dataclass(frozen=True, slots=True)
class CanonicalAccessibilityInputs:
    """Engine-ready tables assembled from canonical v2 entities + attachments."""

    origins: pd.DataFrame
    services: pd.DataFrame


def request_from_service_spec(
    service_spec: ServiceAnalysisSpec,
    mode: TransportMode,
    *,
    travel_time_weight: str,
    off_network_speed_m_s: float,
    distance_weight: str | None = "length_m",
) -> AccessibilityEngineRequest:
    """Translate one enabled ServiceAnalysisSpec/mode pair into an engine request."""

    mode = TransportMode(mode)
    if not service_spec.enabled:
        raise ValueError(
            f"service_type={service_spec.service_type.value} is disabled in AnalysisSpec."
        )
    if mode not in service_spec.modes:
        raise ValueError(
            f"mode={mode.value} is not enabled for service_type={service_spec.service_type.value}."
        )

    return AccessibilityEngineRequest(
        service_type=service_spec.service_type,
        mode=mode,
        thresholds_min=tuple(service_spec.thresholds_min_by_mode[mode.value]),
        population_selector=service_spec.population_selector,
        travel_time_weight=travel_time_weight,
        off_network_speed_m_s=float(off_network_speed_m_s),
        distance_weight=distance_weight,
    )


def _validate_attachment_table(attachments: pd.DataFrame) -> pd.DataFrame:
    NETWORK_ATTACHMENT_V2.validate_columns(attachments.columns)
    result = attachments.copy().reset_index(drop=True)

    valid_kinds = {"origin", "service"}
    kinds = set(result["entity_kind"].dropna().astype(str))
    invalid_kinds = sorted(kinds - valid_kinds)
    if invalid_kinds:
        raise ValueError(f"Invalid NetworkAttachmentV2 entity_kind values: {invalid_kinds}")

    valid_modes = {mode.value for mode in TransportMode}
    modes = set(result["mode"].dropna().astype(str))
    invalid_modes = sorted(modes - valid_modes)
    if invalid_modes:
        raise ValueError(f"Invalid NetworkAttachmentV2 mode values: {invalid_modes}")

    duplicate_mask = result.duplicated(
        subset=["entity_kind", "entity_id", "mode"],
        keep=False,
    )
    if duplicate_mask.any():
        sample = (
            result.loc[duplicate_mask, ["entity_kind", "entity_id", "mode"]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "NetworkAttachmentV2 must contain at most one attachment per "
            f"entity_kind/entity_id/mode. Examples: {sample}"
        )

    if result["entity_id"].isna().any():
        raise ValueError("NetworkAttachmentV2 entity_id cannot be null.")

    snapped_values = result["snapped"]
    if not snapped_values.map(lambda value: isinstance(value, bool)).all():
        raise ValueError("NetworkAttachmentV2 snapped must contain boolean values.")

    snap_distance = pd.to_numeric(result["snap_distance_m"], errors="coerce")
    snapped = snapped_values.astype(bool)

    # Avoid pandas/numexpr RuntimeWarning when nullable snap distances are
    # compared with zero. Null is valid for explicit unsnapped rows and is
    # handled separately by the snapped mask below.
    negative_snap = snap_distance.fillna(0.0) < 0
    bad_snapped = snapped & (
        result["node_id"].isna()
        | snap_distance.isna()
        | negative_snap
    )
    if bad_snapped.any():
        raise ValueError(
            "Snapped NetworkAttachmentV2 rows need node_id and non-negative snap_distance_m."
        )

    bad_unsnapped = (~snapped) & result["node_id"].notna()
    if bad_unsnapped.any():
        raise ValueError(
            "Unsnapped NetworkAttachmentV2 rows must store node_id=null."
        )

    result["snap_distance_m"] = snap_distance.astype(float)
    return result


def _mode_attachments(
    attachments: pd.DataFrame,
    *,
    entity_kind: str,
    mode: TransportMode,
    expected_entity_ids: pd.Series,
) -> pd.DataFrame:
    selected = attachments.loc[
        (attachments["entity_kind"].astype(str) == entity_kind)
        & (attachments["mode"].astype(str) == mode.value)
    ].copy()

    expected = set(expected_entity_ids.astype(str))
    actual = set(selected["entity_id"].astype(str))

    missing = sorted(expected - actual)
    if missing:
        raise ValueError(
            f"Missing {mode.value} NetworkAttachmentV2 rows for {entity_kind}s: {missing[:20]}"
        )

    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(
            f"Unknown {entity_kind} entity_id values in {mode.value} attachments: {unknown[:20]}"
        )

    return selected


def prepare_canonical_accessibility_inputs(
    origins: pd.DataFrame,
    services: pd.DataFrame,
    attachments: pd.DataFrame,
    *,
    mode: TransportMode,
) -> CanonicalAccessibilityInputs:
    """Assemble canonical v2 entities into the engine's narrow routing interface.

    Canonical entities intentionally keep network attachment separate from the
    domain entity. This function performs that explicit join and refuses to
    interpret a missing attachment row as an unsnapped entity: upstream code
    must record an explicit ``snapped=False`` attachment instead.
    """

    mode = TransportMode(mode)
    ORIGIN_V2.validate_columns(origins.columns)
    SERVICE_V2.validate_columns(services.columns)

    if origins["origin_id"].isna().any() or origins["origin_id"].astype(str).duplicated().any():
        raise ValueError("OriginV2 origin_id must be non-null and unique.")
    if services["service_id"].isna().any() or services["service_id"].astype(str).duplicated().any():
        raise ValueError("ServiceV2 service_id must be non-null and unique.")

    prepared_attachments = _validate_attachment_table(attachments)

    origin_attachments = _mode_attachments(
        prepared_attachments,
        entity_kind="origin",
        mode=mode,
        expected_entity_ids=origins["origin_id"],
    )
    service_attachments = _mode_attachments(
        prepared_attachments,
        entity_kind="service",
        mode=mode,
        expected_entity_ids=services["service_id"],
    )

    origin_attachment_view = origin_attachments[
        ["entity_id", "node_id", "snapped", "snap_distance_m"]
    ].rename(
        columns={
            "entity_id": "origin_id",
            "node_id": "network_node_id",
            "snap_distance_m": "origin_snap_distance_m",
        }
    )

    service_attachment_view = service_attachments[
        ["entity_id", "node_id", "snapped", "snap_distance_m"]
    ].rename(
        columns={
            "entity_id": "service_id",
            "node_id": "network_node_id",
        }
    )

    # NetworkAttachmentV2 is the authoritative source for routing attachment.
    # Canonical entities may temporarily carry legacy convenience columns during
    # migration; drop them before the join so pandas cannot create _x/_y columns
    # and accidentally leave the engine-facing network_node_id empty.
    origin_result = origins.drop(
        columns=[
            column
            for column in ("network_node_id", "snapped", "origin_snap_distance_m")
            if column in origins.columns
        ]
    ).copy()
    origin_result["origin_id"] = origin_result["origin_id"].astype(str)
    origin_attachment_view["origin_id"] = origin_attachment_view["origin_id"].astype(str)
    origin_result = origin_result.merge(
        origin_attachment_view,
        on="origin_id",
        how="left",
        validate="one_to_one",
    )
    origin_result.loc[~origin_result["snapped"], "network_node_id"] = None
    origin_result.loc[~origin_result["snapped"], "origin_snap_distance_m"] = None

    service_result = services.drop(
        columns=[
            column
            for column in ("network_node_id", "snapped", "snap_distance_m")
            if column in services.columns
        ]
    ).copy()
    service_result["service_id"] = service_result["service_id"].astype(str)
    service_attachment_view["service_id"] = service_attachment_view["service_id"].astype(str)
    service_result = service_result.merge(
        service_attachment_view,
        on="service_id",
        how="left",
        validate="one_to_one",
    )
    service_result.loc[~service_result["snapped"], "network_node_id"] = None
    service_result.loc[~service_result["snapped"], "snap_distance_m"] = None

    return CanonicalAccessibilityInputs(
        origins=origin_result,
        services=service_result,
    )


def compute_accessibility_from_canonical(
    graph: nx.Graph,
    origins: pd.DataFrame,
    services: pd.DataFrame,
    attachments: pd.DataFrame,
    request: AccessibilityEngineRequest,
) -> AccessibilityEngineResult:
    """Run one service-type/mode request from the frozen canonical v2 contracts.

    A canonical service table and attachment table may contain many service
    types. One accessibility request must depend only on the requested service
    type: missing or unsnapped attachments for unrelated services must not
    block the computation. Origin attachments remain global for the selected
    mode because every origin participates in the result table.
    """

    SERVICE_V2.validate_columns(services.columns)
    requested_services = services.loc[
        services["service_type"].astype(str) == request.service_type.value
    ].copy()

    requested_service_ids = set(
        requested_services["service_id"].dropna().astype(str)
    )

    attachment_kind = attachments["entity_kind"].astype(str)
    attachment_id = attachments["entity_id"].astype(str)
    scoped_attachments = attachments.loc[
        (attachment_kind == "origin")
        | (
            (attachment_kind == "service")
            & attachment_id.isin(requested_service_ids)
        )
    ].copy()

    canonical = prepare_canonical_accessibility_inputs(
        origins,
        requested_services,
        scoped_attachments,
        mode=request.mode,
    )
    return compute_accessibility(
        graph=graph,
        origins=canonical.origins,
        services=canonical.services,
        request=request,
    )
