"""B7C.0: independent, read-only territorial preflight for proposed ServiceV2 additions.

This is NOT approval of the source, a valid address, operating status or a
routing scenario. It verifies that supplied candidate coordinates are inside
the same *census footprint* as a verified B7B baseline and that graph/service
identities fit B6C.1's overlay contract. It writes nothing and never promotes
pending ANNCSU evidence. A later step must bind this proof to scenario routing.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from analysis.accessibility_contracts import prepare_service_destinations_v2
from analysis.run_boundary_qualified_removal import verify_boundary_view
from analysis.scenario_reporting_v2 import ScenarioInputPaths, _read_table
from core.analysis_spec import ServiceType, TransportMode
from core.run_manifest import sha256_file
from core.schema_v2 import NETWORK_ATTACHMENT_V2, SERVICE_V2
from quality.audit_v2_service_boundary import _inside_census_footprint
from quality.audit_legacy_boundary_policy import _footprint_union
from quality.preflight_real_service_scenario import (
    ROOT, CitySnapshot, canonical_paths, inspect_real_snapshot,
)
from transformation.build_network_attachments_v2 import normalize_node_id

POLICY = "b7c0_census_footprint_addition_preflight_v1"


def inspect_boundary_additions(
    *, snapshot: CitySnapshot, canonical: ScenarioInputPaths,
    qualified_services: Path, addition_services: Path,
    addition_attachments: Path, root: Path = ROOT,
) -> dict[str, Any]:
    """Check independent geography, candidate rows, and mode-specific attachment.

    Sources remain unapproved, even if the legacy location-validation flag is
    set. The official administrative boundary is NOT being certified here.
    """
    qualification = verify_boundary_view(
        snapshot=snapshot, canonical=canonical,
        qualified_services=Path(qualified_services), root=Path(root),
    )
    # Revalidate qualified baseline using the existing fail-closed network QA.
    derived_inputs = replace(canonical, services=Path(qualified_services))
    inspection = inspect_real_snapshot(snapshot, derived_inputs, examples=0)
    addition_services = Path(addition_services)
    addition_attachments = Path(addition_attachments)
    for label, path in (("addition_services", addition_services),
                        ("addition_attachments", addition_attachments)):
        if not path.is_file() or path.suffix.lower() not in (".parquet", ".csv"):
            raise FileNotFoundError(f"B7C valid {label} (.csv/.parquet) required: {path}")
    before = {
        "candidate_services": sha256_file(addition_services),
        "candidate_attachments": sha256_file(addition_attachments),
        "qualified_services": qualification["qualified_view_sha256"],
        "canonical_services": qualification["canonical_services_sha256"],
        "footprint": qualification["footprint_sha256"],
        "graph": inspection["graph_sha256"],
    }
    candidates = _read_table(addition_services)
    attachments = _read_table(addition_attachments)
    canonical_services = _read_table(canonical.services)
    footprint_path = Path(qualification["footprint_path"])
    footprint = gpd.read_parquet(footprint_path)
    _footprint_union(footprint)

    SERVICE_V2.validate_columns(candidates.columns)
    NETWORK_ATTACHMENT_V2.validate_columns(attachments.columns)
    if candidates.empty or candidates["service_id"].isna().any() or (
        candidates["service_id"].astype(str).duplicated().any()
    ):
        raise ValueError("B7C candidate service IDs must be distinct and nonempty")
    ids = set(candidates["service_id"].astype(str))
    if ids & set(canonical_services["service_id"].astype(str)):
        raise ValueError("B7C addition reuses a canonical baseline service ID")
    if not candidates["municipality_code"].astype(str).eq(snapshot.municipality_code).all():
        raise ValueError("B7C candidates belong to another municipality")
    if not candidates["service_type"].astype(str).eq(snapshot.service_type.value).all():
        raise ValueError("B7C candidate service type differs from analysis")
    forbidden = {
        "inside_operational_footprint", "operational_footprint_sha256",
        "canonical_services_sha256", "territorial_policy",
    }
    if forbidden & set(candidates.columns):
        raise ValueError("B7C candidate cannot self-certify its territorial status")

    identity = ["service_type", "source_name", "source_record_id"]
    if candidates[identity].isna().any().any() or (
        candidates[identity].astype(str).apply(lambda c: c.str.strip().eq("")).any().any()
    ):
        raise ValueError("B7C candidates require nonempty source identity")
    if candidates.duplicated(subset=identity).any():
        raise ValueError("B7C duplicate candidate source identities")
    existing = set(map(tuple, canonical_services[identity].dropna().astype(str).to_numpy()))
    proposed = set(map(tuple, candidates[identity].astype(str).to_numpy()))
    if existing & proposed:
        raise ValueError("B7C candidate duplicates a canonical source identity")

    # Boundary check is recomputed from untouched coordinates and verified
    # geometry; candidate-supplied flags are not accepted as evidence.
    inside = _inside_census_footprint(candidates, footprint)
    if not inside.all():
        outside = sorted(candidates.loc[~inside, "service_id"].astype(str))
        raise ValueError(f"B7C proposed services outside or without coordinates in census footprint: {outside[:10]}")

    if len(attachments) != len(candidates) or attachments["entity_id"].isna().any() or (
        attachments.duplicated(["entity_kind", "entity_id", "mode"]).any()
    ):
        raise ValueError("B7C requires exactly one distinct mode attachment per candidate")
    if (not attachments["entity_kind"].astype(str).eq("service").all()
            or not attachments["mode"].astype(str).eq(snapshot.mode.value).all()
            or set(attachments["entity_id"].astype(str)) != ids):
        raise ValueError("B7C candidate attachments do not match service IDs/mode")
    if ("graph_checksum" not in attachments or attachments["graph_checksum"].isna().any()
            or not attachments["graph_checksum"].astype(str).eq(inspection["graph_sha256"]).all()):
        raise ValueError("B7C attachment graph provenance mismatch")
    gated = prepare_service_destinations_v2(candidates, attachments, mode=snapshot.mode)
    if not gated["routing_eligible"].all():
        reasons = gated.loc[~gated["routing_eligible"], ["service_id", "routing_exclusion_reason"]].to_dict("records")
        raise ValueError(f"B7C proposed services fail the existing B6A3 location/network gate: {reasons}")

    nodes = _read_table(canonical.nodes)
    if "source_record_id" not in nodes or nodes["source_record_id"].isna().any():
        raise ValueError("B7C canonical graph nodes lack source identity")
    node_ids = [normalize_node_id(value) for value in nodes["source_record_id"]]
    if any(value is None for value in node_ids) or len(set(node_ids)) != len(node_ids):
        raise ValueError("B7C graph nodes have invalid or duplicate identity")
    if not set(gated["network_node_id"].astype(str)).issubset(set(node_ids)):
        raise ValueError("B7C candidate attachment points to missing graph node")
    after_files = {
        "candidate_services": addition_services,
        "candidate_attachments": addition_attachments,
        "qualified_services": Path(qualified_services),
        "canonical_services": canonical.services,
        "footprint": footprint_path,
    }
    for name, path in after_files.items():
        if sha256_file(path) != before[name]:
            raise ValueError(f"B7C input changed during preflight: {name}")
    # Graph checksum must be recalculated, not trusted as a string in a row.
    from transformation.build_network_attachments_v2 import graph_checksum
    if graph_checksum(canonical.nodes, canonical.edges) != before["graph"]:
        raise ValueError("B7C graph changed during preflight")
    return {
        "policy": POLICY,
        "municipality_code": snapshot.municipality_code,
        "mode": snapshot.mode.value,
        "service_type": snapshot.service_type.value,
        "candidate_count": len(candidates),
        "candidate_service_ids": sorted(ids),
        "verified_input_sha256": before,
        "footprint_basis": qualification["boundary_basis"],
        "administrative_boundary_verified": False,
        "qualified_baseline_routable": inspection["selected_eligible_count"],
        "result": "all_candidates_within_operational_census_footprint_and_technically_eligible",
        "independent_source_review_approved": False,
        "scenario_authorized": False,
        "notes": "Territorial/network preflight only; no approval of ANNCSU or independently supplied candidate sources",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--service-type", default="pharmacy", choices=[s.value for s in ServiceType])
    parser.add_argument("--mode", default="walk", choices=[m.value for m in TransportMode])
    parser.add_argument("--qualified-services", required=True, type=Path)
    parser.add_argument("--addition-services", required=True, type=Path)
    parser.add_argument("--addition-attachments", required=True, type=Path)
    parser.add_argument("--root", default=ROOT, type=Path)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    snapshot = CitySnapshot(
        municipality_code=args.municipality_code, census_year=args.census_year,
        school_year=args.school_year, health_reference_date=args.health_reference_date,
        service_type=ServiceType(args.service_type), mode=TransportMode(args.mode),
    )
    report = inspect_boundary_additions(
        snapshot=snapshot, canonical=canonical_paths(args.root, snapshot),
        qualified_services=args.qualified_services,
        addition_services=args.addition_services,
        addition_attachments=args.addition_attachments, root=args.root,
    )
    print("=== B7C.0 TERRITORIAL ADDITION PREFLIGHT — READ ONLY ===")
    print(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False))
    print("READ ONLY — no scenarios executed; no source, catalogue or report modified")
    return report


if __name__ == "__main__":
    run()
