"""B7B: read-only geospatial audit of routable canonical ServiceV2 supply.

The operational perimeter is the ISTAT-2021 census-area footprint already
used by the legacy municipality-only network-ready policy (B7A). It is NOT
an authoritative administrative boundary. This audit never updates evidence,
location validation, baseline, scenario, or original files. It intentionally
checks all service types, not just the type selected in a scenario.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from analysis.accessibility_contracts import prepare_service_destinations_v2
from analysis.scenario_reporting_v2 import _read_table
from core.analysis_spec import TransportMode
from core.run_manifest import sha256_file
from core.schema_v2 import NETWORK_ATTACHMENT_V2, SERVICE_V2
from quality.audit_legacy_boundary_policy import _footprint_union, paths_for
from quality.preflight_real_service_scenario import ROOT, CitySnapshot, canonical_paths
from transformation.build_network_attachments_v2 import graph_checksum, normalize_node_id

POLICY = "b7b_v2_routable_supply_census_footprint_v1"


@dataclass(frozen=True, slots=True)
class V2BoundaryPaths:
    footprint: Path
    services: Path
    attachments: Path
    nodes: Path
    edges: Path

    def items(self) -> dict[str, Path]:
        return {name: Path(getattr(self, name)) for name in
                ("footprint", "services", "attachments", "nodes", "edges")}


def paths_for_snapshot(root: Path, snapshot: CitySnapshot) -> V2BoundaryPaths:
    canonical = canonical_paths(root, snapshot)
    legacy = paths_for(root, snapshot.municipality_code,
                       school_year=snapshot.school_year,
                       health_reference_date=snapshot.health_reference_date)
    return V2BoundaryPaths(legacy.footprint, canonical.services,
                           canonical.attachments, canonical.nodes, canonical.edges)


def _inside_census_footprint(services: pd.DataFrame, footprint: gpd.GeoDataFrame) -> pd.Series:
    """Treat missing coordinates as outside, never as geographical evidence."""
    if "latitude" not in services or "longitude" not in services:
        raise ValueError("ServiceV2 must provide latitude and longitude for a geospatial audit")
    if "crs" not in services:
        raise ValueError("ServiceV2 must declare the coordinate reference system")
    lat = pd.to_numeric(services["latitude"], errors="coerce")
    lon = pd.to_numeric(services["longitude"], errors="coerce")
    available = lat.notna() & lon.notna()
    if ((lat.notna() ^ lon.notna())
            | (available & (~np.isfinite(lat) | ~np.isfinite(lon)))
            | (available & ((lat < -90) | (lat > 90) | (lon < -180) | (lon > 180)))).any():
        raise ValueError("Incomplete, nonfinite or out-of-range ServiceV2 coordinates")
    if not services.loc[available, "crs"].astype(str).eq("EPSG:4326").all():
        raise ValueError("Geolocated ServiceV2 coordinates must declare EPSG:4326")
    result = pd.Series(False, index=services.index, dtype=bool)
    if available.any():
        points = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(lon.loc[available], lat.loc[available])],
            index=services.index[available], crs="EPSG:4326",
        ).to_crs(footprint.crs)
        union = _footprint_union(footprint)
        result.loc[available] = points.map(lambda point: bool(union.covers(point)))
    return result


def inspect_v2_boundary(snapshot: CitySnapshot, paths: V2BoundaryPaths) -> dict[str, Any]:
    """Check all baseline routing destinations against the frozen census footprint.

    This verifies routing eligibility, node IDs and location geometry independently;
    it does not attest administrative geography or hypothetical addition sources.
    """
    if not re.fullmatch(r"\d{6}", snapshot.municipality_code):
        raise ValueError("Invalid municipality code")
    items = paths.items()
    missing = [f"{name}: {path}" for name, path in items.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing B7B files:\n" + "\n".join(missing))
    before = {name: sha256_file(path) for name, path in items.items()}
    footprint = gpd.read_parquet(paths.footprint)
    _footprint_union(footprint)
    services = _read_table(paths.services)
    attachments = _read_table(paths.attachments)
    nodes = _read_table(paths.nodes)
    SERVICE_V2.validate_columns(services.columns)
    NETWORK_ATTACHMENT_V2.validate_columns(attachments.columns)
    if services.empty or services["service_id"].isna().any() or services["service_id"].astype(str).duplicated().any():
        raise ValueError("ServiceV2 must have unique nonnull service IDs")
    if not services["municipality_code"].astype(str).eq(snapshot.municipality_code).all():
        raise ValueError("Canonical service municipality code mismatch")
    graph_sha = graph_checksum(paths.nodes, paths.edges)
    mode = snapshot.mode.value
    mode_att = attachments.loc[attachments["mode"].astype(str).eq(mode)].copy()
    if mode_att.empty or "graph_checksum" not in mode_att.columns:
        raise ValueError("Mode attachments lack network provenance")
    if mode_att["graph_checksum"].isna().any() or not mode_att["graph_checksum"].astype(str).eq(graph_sha).all():
        raise ValueError("Attachment graph checksum differs from canonical network")
    if mode_att.duplicated(["entity_kind", "entity_id"]).any():
        raise ValueError("Duplicate mode attachments")
    service_att = mode_att.loc[mode_att["entity_kind"].eq("service")].copy()
    # Fail closed on extra/missing service attachments, not merely eligible ones.
    gated = prepare_service_destinations_v2(services, service_att, mode=snapshot.mode)
    if "source_record_id" not in nodes.columns:
        raise ValueError("Network nodes have no source_record_id")
    node_ids = [normalize_node_id(x) for x in nodes["source_record_id"]]
    if any(x is None for x in node_ids) or len(set(node_ids)) != len(node_ids):
        raise ValueError("Invalid network node identities")
    snapped = gated["snapped"].astype(bool)
    if not set(gated.loc[snapped, "attachment_node_id"].astype(str)).issubset(set(node_ids)):
        raise ValueError("Attached service node absent from network")
    inside = _inside_census_footprint(gated, footprint)
    routable = gated["routing_eligible"].astype(bool)
    outside_routable = routable & ~inside
    if outside_routable.any():
        examples = sorted(gated.loc[outside_routable, "service_id"].astype(str))[:10]
        raise ValueError(f"B7B territorial routing leak: {int(outside_routable.sum())} routable services outside census footprint; IDs: {examples}")
    if {name: sha256_file(path) for name, path in items.items()} != before:
        raise ValueError("B7B inputs changed during the audit")
    types = {}
    for service_type, frame in gated.groupby("service_type", dropna=False, sort=True):
        ix = frame.index
        types[str(service_type)] = {
            "total": int(len(frame)),
            "inside_footprint": int(inside.loc[ix].sum()),
            "routable": int(routable.loc[ix].sum()),
            "outside_not_routable": int((~inside.loc[ix] & ~routable.loc[ix]).sum()),
        }
    return {
        "policy": POLICY,
        "municipality_code": snapshot.municipality_code,
        "mode": mode,
        "boundary_basis": "ISTAT_2021_census_area_footprint_not_administrative_boundary",
        "administrative_boundary_verified": False,
        "total_services": int(len(gated)),
        "routable_services": int(routable.sum()),
        "outside_not_routable": int((~inside & ~routable).sum()),
        "service_types": types,
        "graph_sha256": graph_sha,
        "input_sha256": before,
        "scenario_coverage": "existing_baseline_and_remove_only_subsets; additions_not_audited",
        "result": "v2_routable_service_supply_within_census_footprint",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--mode", default="walk", choices=[m.value for m in TransportMode])
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    snapshot = CitySnapshot(
        municipality_code=args.municipality_code, census_year=args.census_year,
        school_year=args.school_year, health_reference_date=args.health_reference_date,
        mode=TransportMode(args.mode),
    )
    report = inspect_v2_boundary(snapshot, paths_for_snapshot(args.root, snapshot))
    print("=== B7B V2 SERVICE TERRITORIAL AUDIT — READ ONLY ===")
    print(f"Municipality {snapshot.municipality_code} / {snapshot.mode.value} | census footprint, NOT administrative boundary")
    print(f"total={report['total_services']} routable={report['routable_services']} "
          f"outside_not_routable={report['outside_not_routable']}")
    for service_type, counts in report["service_types"].items():
        print(f"  {service_type}: {counts['routable']}/{counts['total']} routable; "
              f"{counts['outside_not_routable']} outside and ineligible")
    print("B7B PASS: all routable V2 baseline services inside operational census footprint; no files changed")
    return report


if __name__ == "__main__":
    run()
