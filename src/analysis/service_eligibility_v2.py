"""B6A3 routing eligibility for migrated canonical services (no file I/O).

This quality gate must run after the NetworkAttachmentV2 join and before
passing destinations into the mode-specific accessibility engine. It is
conservative by design: an OSM network snap never validates a POI address.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ELIGIBILITY_POLICY = "B6A3_migrated_verified_location_v1"


def _validation_flag(value: object) -> bool | None:
    """Accept boolean evidence only. Missing is not a positive assertion."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    raise ValueError(
        "legacy_usable_for_accessibility must be a boolean or null; "
        f"received {value!r}. Do not coerce strings such as 'False' to True."
    )


def apply_service_routing_gate(joined_services: pd.DataFrame) -> pd.DataFrame:
    """Annotate and mask *engine-facing* nodes of ineligible destinations.

    Input rows must already contain unique ServiceV2 + mode-matched attachment
    columns. ``snapped`` retains its geometric meaning, while
    ``routing_eligible`` is independent evidence for actual routing.
    """
    required = {"service_id", "operational_status", "snapped", "network_node_id", "snap_distance_m"}
    missing = sorted(required - set(joined_services.columns))
    if missing:
        raise ValueError(f"Joined services lack eligibility fields: {missing}")
    result = joined_services.copy()
    if result["service_id"].isna().any() or result["service_id"].astype(str).duplicated().any():
        raise ValueError("Eligibility gate requires unique, non-null service_id.")
    if "legacy_usable_for_accessibility" not in result:
        verified = pd.Series([None] * len(result), index=result.index, dtype=object)
    else:
        verified = result["legacy_usable_for_accessibility"].map(_validation_flag)

    result["attachment_node_id"] = result["network_node_id"].copy()
    result["attachment_snap_distance_m"] = result["snap_distance_m"].copy()
    reasons: list[str] = []
    for index, row in result.iterrows():
        if verified.loc[index] is None:
            reason = "missing_validation_evidence"
        elif not verified.loc[index]:
            reason = "location_not_validated"
        elif pd.isna(row["operational_status"]) or str(row["operational_status"]).strip().lower() != "active":
            reason = "service_not_active"
        elif not row["snapped"] or pd.isna(row["network_node_id"]):
            reason = "not_snapped"
        else:
            reason = "eligible"
        reasons.append(reason)

    result["routing_exclusion_reason"] = reasons
    result["routing_eligible"] = result["routing_exclusion_reason"].eq("eligible").astype(bool)
    ineligible = ~result["routing_eligible"]
    result.loc[ineligible, "network_node_id"] = None
    result.loc[ineligible, "snap_distance_m"] = float("nan")
    return result
