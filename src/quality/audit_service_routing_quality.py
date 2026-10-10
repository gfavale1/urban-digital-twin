"""Read-only, provenance-checked audit of canonical V2 service-routing attrition.

Report mutually exclusive routing exclusion reasons AND overlapping quality issues.
Missing coordinates are not evidence that a facility is geographically outside.
The perimeter is the operational ISTAT-2021 census footprint, not a certified
administrative boundary. This audit never promotes services or changes artifacts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.accessibility_contracts import prepare_service_destinations_v2
from analysis.scenario_reporting_v2 import _read_table
from core.analysis_spec import TransportMode
from core.run_manifest import sha256_file
from quality.audit_v2_service_boundary import paths_for_snapshot
from quality.derive_boundary_qualified_services import FLAG, inspect_boundary_view
from quality.preflight_real_service_scenario import ROOT, CitySnapshot
from transformation.build_network_attachments_v2 import graph_checksum, normalize_node_id

POLICY = "service_routing_quality_census_footprint_audit_v1"


def _count_by(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in
            sorted(Counter(values.astype(str)).items())}


def summarize_service_quality(
    qualified_services: pd.DataFrame,
    service_attachments: pd.DataFrame,
    *,
    mode: TransportMode,
) -> dict[str, Any]:
    """Aggregate all rows. A primary reason is not a complete diagnosis.

    ``qualified_services`` must come from requalification against the genuine
    footprint. This function is read-only and never validates source approvals.
    """
    if FLAG not in qualified_services.columns:
        raise ValueError("Missing independently derived operational footprint classification")
    if qualified_services.empty:
        raise ValueError("Quality audit requires at least one canonical service")
    if "latitude" not in qualified_services or "longitude" not in qualified_services:
        raise ValueError("Canonical services must expose both coordinate columns")
    if not qualified_services[FLAG].map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("Footprint classification must use explicit booleans")

    lat_raw = qualified_services["latitude"]
    lon_raw = qualified_services["longitude"]
    present_lat = lat_raw.notna()
    present_lon = lon_raw.notna()
    if (present_lat ^ present_lon).any():
        raise ValueError("Partially missing coordinates require separate source investigation")
    geocoded = present_lat & present_lon
    lat = pd.to_numeric(lat_raw, errors="coerce")
    lon = pd.to_numeric(lon_raw, errors="coerce")
    if (geocoded & (lat.isna() | lon.isna())).any():
        raise ValueError("Non-numeric coordinate values are not missing-coordinate evidence")
    inside = qualified_services[FLAG].astype(bool)
    if (inside & ~geocoded).any():
        raise ValueError("A service without coordinates cannot be classified inside")

    # Independent network validation already occurs before this aggregation in
    # inspect_service_quality. The canonical eligibility policy is reused here.
    gated = prepare_service_destinations_v2(
        qualified_services, service_attachments, mode=mode,
    )
    if len(gated) != len(qualified_services):
        raise AssertionError("Eligibility join unexpectedly changed canonical row count")
    if not gated["service_id"].astype(str).equals(qualified_services["service_id"].astype(str).reset_index(drop=True)):
        # protect the coordinate/routing juxtaposition against an accidental reorder
        by_id = qualified_services.set_index("service_id")
        lat_raw = gated["service_id"].map(by_id["latitude"])
        lon_raw = gated["service_id"].map(by_id["longitude"])
        geocoded = lat_raw.notna() & lon_raw.notna()
        inside = gated["service_id"].map(by_id[FLAG]).astype(bool)
    else:
        inside = inside.reset_index(drop=True)
        geocoded = geocoded.reset_index(drop=True)
    if gated["service_id"].duplicated().any():
        raise ValueError("Duplicate service IDs in audit output")

    status = gated["operational_status"].astype(str).str.strip().str.lower()
    snapped = gated["snapped"].astype(bool)
    eligible = gated["routing_eligible"].astype(bool)
    geostate = pd.Series(np.where(~geocoded, "coordinates_missing",
                                   np.where(inside, "inside_census_footprint",
                                            "outside_census_footprint")), index=gated.index)
    flags = {
        "coordinates_missing": ~geocoded,
        "outside_census_footprint": geocoded & ~inside,
        "location_validation_not_true": ~gated["legacy_usable_for_accessibility"].map(
            lambda x: isinstance(x, (bool, np.bool_)) and bool(x)
        ) if "legacy_usable_for_accessibility" in gated else pd.Series(True, index=gated.index),
        "not_snapped": ~snapped,
        "inactive_or_unknown_status": status.ne("active"),
    }
    # All denominator and primary-reason counts must reconcile exactly.
    by_type: dict[str, Any] = {}
    for service_type, group in gated.groupby("service_type", sort=True, dropna=False):
        idx = group.index
        total = int(len(group))
        reasons = _count_by(group["routing_exclusion_reason"])
        geography = _count_by(geostate.loc[idx])
        counts = {
            "total": total,
            "routable": int(eligible.loc[idx].sum()),
            "ineligible": int((~eligible.loc[idx]).sum()),
            "geography": geography,
            "primary_routing_reason": reasons,
            "overlapping_issues": {k: int(v.loc[idx].sum()) for k, v in flags.items()},
            "source_name_counts": _count_by(group["source_name"].fillna("missing")),
            "source_reference_date_counts": _count_by(group["source_reference_date"].fillna("missing")),
        }
        if counts["routable"] + counts["ineligible"] != total:
            raise AssertionError("Routing denominator does not reconcile")
        if sum(reasons.values()) != total or sum(geography.values()) != total:
            raise AssertionError("Exclusive breakdown does not reconcile")
        if int((eligible.loc[idx] & ~inside.loc[idx]).sum()):
            raise ValueError("Routable services outside operational census footprint")
        by_type[str(service_type)] = counts
    all_reasons = _count_by(gated["routing_exclusion_reason"])
    if sum(all_reasons.values()) != len(gated):
        raise AssertionError("Global reason breakdown does not reconcile")
    return {
        "services_total": int(len(gated)),
        "services_routable": int(eligible.sum()),
        "services_ineligible": int((~eligible).sum()),
        "geography": _count_by(geostate),
        "primary_routing_reason": all_reasons,
        "overlapping_issues": {k: int(v.sum()) for k, v in flags.items()},
        "service_types": by_type,
        "notes": {
            "geography": "Missing coordinates are distinct from confirmed outside-footprint locations.",
            "reason": "Primary routing reasons are exclusive and ordered; overlapping issues are not additive.",
            "denominator": "Canonical service entities (not necessarily distinct physical facilities).",
            "eligibility": "Baseline routing gate, not authorization of new or unreviewed candidates.",
        },
    }


def inspect_service_quality(snapshot: CitySnapshot, *, root: Path = ROOT) -> dict[str, Any]:
    """Validate canonical data and attachments, compute audited counts; never write."""
    paths = paths_for_snapshot(Path(root), snapshot)
    entries = paths.items()
    missing = [str(path) for path in entries.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing service-quality inputs:\n" + "\n".join(missing))
    hashes = {key: sha256_file(path) for key, path in entries.items()}
    qualified, territorial = inspect_boundary_view(snapshot, root=root)
    if territorial["source_sha256"]["services"] != hashes["services"] or \
       territorial["source_sha256"]["footprint"] != hashes["footprint"]:
        raise ValueError("Territorial source identity changed during audit")
    attachments = _read_table(paths.attachments)
    nodes = _read_table(paths.nodes)
    graph_sha = graph_checksum(paths.nodes, paths.edges)
    mode_att = attachments.loc[attachments["mode"].astype(str).eq(snapshot.mode.value)].copy()
    if mode_att.empty or "graph_checksum" not in mode_att:
        raise ValueError("Mode attachments missing graph checksum")
    if mode_att["graph_checksum"].isna().any() or not mode_att["graph_checksum"].astype(str).eq(graph_sha).all():
        raise ValueError("Mode attachments disagree with canonical graph checksum")
    service_att = mode_att.loc[mode_att["entity_kind"].astype(str).eq("service")].copy()
    if "source_record_id" not in nodes:
        raise ValueError("Network nodes lack source_record_id")
    node_ids = [normalize_node_id(v) for v in nodes["source_record_id"]]
    if any(v is None for v in node_ids) or len(set(node_ids)) != len(node_ids):
        raise ValueError("Invalid or duplicate network node IDs")
    snapped_att = service_att.loc[service_att["snapped"].eq(True)]
    snapped_nodes = [normalize_node_id(v) for v in snapped_att["node_id"]]
    graph_node_ids = set(node_ids)
    if any(v not in graph_node_ids for v in snapped_nodes):
        raise ValueError("Service attachment references nonexistent network node")
    result = summarize_service_quality(qualified, service_att, mode=snapshot.mode)
    if {key: sha256_file(path) for key, path in entries.items()} != hashes:
        raise ValueError("Inputs modified while running service-quality audit")
    return {
        "policy": POLICY,
        "municipality_code": snapshot.municipality_code,
        "mode": snapshot.mode.value,
        "boundary_basis": territorial["boundary_basis"],
        "administrative_boundary_verified": False,
        "input_sha256": hashes,
        "graph_sha256": graph_sha,
        **result,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--mode", choices=[m.value for m in TransportMode], default="walk")
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    snap = CitySnapshot(
        municipality_code=args.municipality_code, census_year=args.census_year,
        school_year=args.school_year, health_reference_date=args.health_reference_date,
        mode=TransportMode(args.mode),
    )
    report = inspect_service_quality(snap, root=args.root)
    print("=== SERVICE ROUTING QUALITY AUDIT — READ ONLY ===")
    print("Municipality %s / %s | all services %d, routable %d, excluded %d" % (
        report["municipality_code"], report["mode"], report["services_total"],
        report["services_routable"], report["services_ineligible"]))
    for service_type, item in report["service_types"].items():
        print(f"  {service_type}: {item['routable']}/{item['total']} routable")
        print("    exclusive reasons: " + json.dumps(item["primary_routing_reason"], sort_keys=True))
        print("    geography: " + json.dumps(item["geography"], sort_keys=True))
        print("    overlapping issues: " + json.dumps(item["overlapping_issues"], sort_keys=True))
    print("Census-area footprint only — not an administrative boundary. No data modified.")
    return report


if __name__ == "__main__":
    run()
