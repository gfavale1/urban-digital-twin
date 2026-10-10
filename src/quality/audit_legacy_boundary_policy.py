"""B7A read-only reconciliation of legacy municipality-only service supply.

Compares the enriched and OSM-only *network-ready* service layers with the
2021 ISTAT census-area footprint used by the legacy routing filter.
This is an audit of operational consistency, NOT authoritative municipal
boundary verification. Does not recalculate or promote canonical services.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from core.run_manifest import sha256_file

ROOT = Path(__file__).resolve().parents[2]
POLICY = "b7a_legacy_municipality_supply_reconciliation_v1"
LEGACY_POLICY_PREFIX = "municipality_only_supply;"
FLAG_COLUMNS = (
    "usable_before_boundary_filter", "inside_municipality",
    "excluded_by_municipality_boundary", "usable_for_accessibility",
)


@dataclass(frozen=True, slots=True)
class BoundaryAuditPaths:
    footprint: Path
    enriched: Path
    enriched_manifest: Path
    osm_only: Path
    osm_only_manifest: Path

    def items(self) -> dict[str, Path]:
        return {key: Path(getattr(self, key)) for key in (
            "footprint", "enriched", "enriched_manifest",
            "osm_only", "osm_only_manifest",
        )}


def paths_for(root: Path, municipality_code: str, *, school_year: str = "202425",
              health_reference_date: str = "2025-06-30") -> BoundaryAuditPaths:
    if not re.fullmatch(r"\d{6}", municipality_code):
        raise ValueError("municipality_code must have six digits")
    if not re.fullmatch(r"\d{6}", school_year):
        raise ValueError("school_year must have six digits")
    try:
        from datetime import date
        if date.fromisoformat(health_reference_date).isoformat() != health_reference_date:
            raise ValueError
    except ValueError as exc:
        raise ValueError("health_reference_date must be YYYY-MM-DD") from exc
    root = Path(root)
    acc = root / "data/features/accessibility" / municipality_code
    label = health_reference_date.replace("-", "")
    stem = f"service_network_nodes_{school_year}_{label}"
    return BoundaryAuditPaths(
        footprint=root / "data/processed/istat" / f"{municipality_code}_census_areas_2021.parquet",
        enriched=acc / f"{stem}.parquet",
        enriched_manifest=acc / f"{stem}_manifest.json",
        osm_only=acc / "service_network_nodes_osm_only.parquet",
        osm_only_manifest=acc / "service_network_nodes_osm_only_manifest.json",
    )


def _bool_column(frame: gpd.GeoDataFrame, column: str) -> pd.Series:
    """Do not coerce missing values or strings such as 'False' into truth."""
    values = frame[column]
    if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{column} must contain only explicit booleans")
    return values.astype(bool)


def _footprint_union(areas: gpd.GeoDataFrame):
    if areas.crs is None or areas.empty:
        raise ValueError("ISTAT census footprint must be nonempty and have CRS")
    geometry = areas.geometry
    if (geometry.isna().any() or geometry.is_empty.any()
            or not geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()
            or not geometry.is_valid.all()):
        raise ValueError("ISTAT census footprint has invalid geometries")
    try:
        union = geometry.union_all()
    except AttributeError:
        union = geometry.unary_union
    if union is None or union.is_empty or not union.is_valid:
        raise ValueError("ISTAT census footprint union is invalid")
    return union


def _audit_layer(layer: str, data: gpd.GeoDataFrame, manifest: dict[str, Any],
                 footprint: Path, boundary_geom: Any, boundary_crs: Any,
                 municipality_code: str) -> dict[str, int]:
    if not isinstance(manifest, dict):
        raise ValueError(f"{layer}: missing manifest object")
    if manifest.get("service_layer") != layer or manifest.get("municipality_code") != municipality_code:
        raise ValueError(f"{layer}: manifest layer/municipality mismatch")
    summary = manifest.get("municipality_boundary_filter")
    if not isinstance(summary, dict):
        raise ValueError(f"{layer}: missing boundary-filter provenance")
    if not str(summary.get("policy", "")).startswith(LEGACY_POLICY_PREFIX):
        raise ValueError(f"{layer}: legacy municipality-only policy missing")
    source = summary.get("boundary_source")
    if not isinstance(source, str) or not source.strip() or Path(source).resolve() != footprint.resolve():
        raise ValueError(f"{layer}: boundary footprint provenance mismatch")
    if not isinstance(data, gpd.GeoDataFrame) or data.crs is None:
        raise ValueError(f"{layer}: service layer must have a known CRS")
    required = {"service_site_id", "municipality_code", "geometry", "network_node_id", *FLAG_COLUMNS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{layer}: missing columns: {missing}")
    if data["service_site_id"].isna().any() or data["service_site_id"].astype(str).duplicated().any():
        raise ValueError(f"{layer}: duplicate/null service_site_id")
    if not data["municipality_code"].astype(str).eq(municipality_code).all():
        raise ValueError(f"{layer}: service municipality metadata disagrees with requested code")
    for name in FLAG_COLUMNS:
        _bool_column(data, name)
    coords = data.to_crs(boundary_crs).geometry
    if not coords.dropna().geom_type.eq("Point").all():
        raise ValueError(f"{layer}: non-point service geometry")
    inside = coords.map(lambda geom: bool(boundary_geom.covers(geom))
                        if geom is not None and not geom.is_empty else False).astype(bool)
    before = data["usable_before_boundary_filter"].astype(bool)
    recorded_inside = data["inside_municipality"].astype(bool)
    excluded = data["excluded_by_municipality_boundary"].astype(bool)
    usable = data["usable_for_accessibility"].astype(bool)
    if not recorded_inside.eq(inside).all():
        raise ValueError(f"{layer}: stored inside_municipality differs from footprint geometry")
    if not excluded.eq(before & ~inside).all():
        raise ValueError(f"{layer}: excluded_by_municipality_boundary disagrees with policy")
    if not usable.eq(before & inside).all():
        raise ValueError(f"{layer}: outside/ineligible service leaked into routing supply")
    if data.loc[usable, "network_node_id"].isna().any():
        raise ValueError(f"{layer}: a routing-usable service has no network node")
    expected = {
        "service_sites_total": int(len(data)),
        "inside_municipality": int(inside.sum()),
        "outside_municipality": int((~inside).sum()),
        "usable_before_boundary_filter": int(before.sum()),
        "usable_excluded_outside_municipality": int(excluded.sum()),
        "usable_after_boundary_filter": int(usable.sum()),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"{layer}: boundary manifest count mismatch: {key}")
    if manifest.get("service_sites_total") != expected["service_sites_total"]:
        raise ValueError(f"{layer}: manifest service_sites_total mismatch")
    if manifest.get("service_sites_usable") != expected["usable_after_boundary_filter"]:
        raise ValueError(f"{layer}: manifest service_sites_usable mismatch")
    return expected


def inspect_boundary_policy(paths: BoundaryAuditPaths, municipality_code: str) -> dict[str, Any]:
    """Re-check both persisted legacy supplies; NEVER write to disk."""
    if not re.fullmatch(r"\d{6}", municipality_code):
        raise ValueError("municipality_code must have six digits")
    inputs = paths.items()
    missing = [f"{name}: {p}" for name, p in inputs.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing boundary audit files:\n" + "\n".join(missing))
    before = {name: sha256_file(path) for name, path in inputs.items()}
    footprint = gpd.read_parquet(paths.footprint)
    union = _footprint_union(footprint)
    layers = {}
    for name in ("enriched", "osm_only"):
        services = gpd.read_parquet(getattr(paths, name))
        manifest = json.loads(getattr(paths, f"{name}_manifest").read_text(encoding="utf-8"))
        layers[name] = _audit_layer(name, services, manifest, paths.footprint, union,
                                    footprint.crs, municipality_code)
    if {name: sha256_file(path) for name, path in inputs.items()} != before:
        raise ValueError("Boundary input files changed during audit")
    return {
        "policy": POLICY,
        "municipality_code": municipality_code,
        "boundary_basis": "ISTAT_2021_census_area_footprint_not_administrative_boundary",
        "administrative_boundary_verified": False,
        "boundary_footprint_sha256": before["footprint"],
        "input_sha256": before,
        "layers": layers,
        "result": "legacy_boundary_filter_consistent",
        "notes": "Read-only; equal territorial rule for enriched and OSM-only. "
                 "Not a claim about authoritative municipal limits or v2 scenario geography.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    paths = paths_for(args.root, args.municipality_code,
                      school_year=args.school_year,
                      health_reference_date=args.health_reference_date)
    report = inspect_boundary_policy(paths, args.municipality_code)
    print("=== B7A MUNICIPALITY-ONLY SUPPLY AUDIT — READ ONLY ===")
    print(f"Municipality {args.municipality_code} | census footprint, NOT administrative boundary")
    for key, counts in report["layers"].items():
        print(f"{key}: total={counts['service_sites_total']}, "
              f"inside={counts['inside_municipality']}, "
              f"excluded_outside={counts['usable_excluded_outside_municipality']}, "
              f"routing_usable={counts['usable_after_boundary_filter']}")
    print("B7A PASS: legacy boundary flags and both manifests consistent; no files changed")
    return report


if __name__ == "__main__":
    run()
