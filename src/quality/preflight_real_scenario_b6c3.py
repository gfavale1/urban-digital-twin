"""B6C.3a — read-only canonical real-data preflight for service scenarios.

Discovers canonical municipal files using *existing* output conventions. It
checks identities, B6A3 routing eligibility and graph attachment fingerprint,
but NEVER authorizes sources, creates an overlay or writes files.

Usage:
  PYTHONPATH=src python src/quality/preflight_real_scenario_b6c3.py \\
      --municipality-code 034027 --service-type pharmacy --mode walk
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.accessibility_contracts import prepare_service_destinations_v2
from analysis.network_graph import normalize_node_id
from analysis.scenario_reporting_v2 import ScenarioInputPaths, _read_table
from core.analysis_spec import ServiceType, TransportMode
from core.run_manifest import sha256_file
from core.schema_v2 import ORIGIN_V2, SERVICE_V2, NETWORK_ATTACHMENT_V2
from transformation.build_network_attachments_v2 import graph_checksum

ROOT = Path(__file__).resolve().parents[2]
POLICY = "b6c3a_canonical_scenario_input_preflight_v1"


@dataclass(frozen=True, slots=True)
class CitySnapshot:
    municipality_code: str
    census_year: str = "2023"
    school_year: str = "202425"
    health_reference_date: str = "2025-06-30"
    mode: TransportMode = TransportMode.WALK
    service_type: ServiceType = ServiceType.PHARMACY

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{6}", self.municipality_code):
            raise ValueError("municipality_code must have six digits")
        if not re.fullmatch(r"\d{4}", self.census_year):
            raise ValueError("census_year must have four digits")
        if not re.fullmatch(r"\d{6}", self.school_year):
            raise ValueError("school_year must have six digits")
        if date.fromisoformat(self.health_reference_date).isoformat() != self.health_reference_date:
            raise ValueError("health_reference_date must be YYYY-MM-DD")
        object.__setattr__(self, "mode", TransportMode(self.mode))
        object.__setattr__(self, "service_type", ServiceType(self.service_type))


def canonical_paths(root: Path, snapshot: CitySnapshot) -> ScenarioInputPaths:
    """Derive existing B6 network-attachment and canonical file conventions."""
    root = Path(root)
    code = snapshot.municipality_code
    label = snapshot.health_reference_date.replace("-", "")
    return ScenarioInputPaths(
        origins=root / "data/features/accessibility" / code /
            f"population_network_origins_{snapshot.census_year}.parquet",
        services=root / "data/processed/services" / code /
            f"service_entities_v2_{snapshot.school_year}_{label}.parquet",
        attachments=root / "data/processed/network_attachments" / code /
            f"network_attachments_v2_{snapshot.mode.value}_{snapshot.census_year}_{snapshot.school_year}_{label}.parquet",
        nodes=root / "data/processed/osm" / code / f"{snapshot.mode.value}_nodes.parquet",
        edges=root / "data/processed/osm" / code / f"{snapshot.mode.value}_edges.parquet",
    )


def inspect_real_snapshot(snapshot: CitySnapshot, paths: ScenarioInputPaths,
                          *, examples: int = 5) -> dict[str, Any]:
    """Read-only diagnosis of a real canonical snapshot, not a source attestation.

    Same-file digest checks guard against ordinary concurrent edits. Both
    unsnapped and non-validated services remain in the canonical input.
    """
    if examples < 0 or examples > 50:
        raise ValueError("examples must be between 0 and 50")
    entries = paths.items()
    if set(entries) != {"origins", "services", "attachments", "nodes", "edges"}:
        raise ValueError("B6C.3a preflight takes baseline files only (no additions)")
    missing = [f"{name}: {path}" for name, path in entries.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing canonical files:\n" + "\n".join(missing))
    before = {name: sha256_file(path) for name, path in entries.items()}
    graph_sha = graph_checksum(paths.nodes, paths.edges)
    tables = {name: _read_table(path) for name, path in entries.items()}
    if {name: sha256_file(path) for name, path in entries.items()} != before:
        raise ValueError("Canonical input changed while being loaded")
    origins, services, attachments, nodes = (
        tables[name] for name in ("origins", "services", "attachments", "nodes")
    )
    ORIGIN_V2.validate_columns(origins.columns)
    SERVICE_V2.validate_columns(services.columns)
    NETWORK_ATTACHMENT_V2.validate_columns(attachments.columns)
    if origins.empty or origins["origin_id"].isna().any() or origins["origin_id"].astype(str).duplicated().any():
        raise ValueError("Origins must have nonempty unique origin_id")
    if services["service_id"].isna().any() or services["service_id"].astype(str).duplicated().any():
        raise ValueError("Services must have unique non-null service_id")
    code = snapshot.municipality_code
    if not origins["municipality_code"].astype(str).eq(code).all():
        raise ValueError("Origin municipality differs from requested municipality")
    if not services["municipality_code"].astype(str).eq(code).all():
        raise ValueError("Service municipality differs from requested municipality")
    selected = services.loc[services["service_type"].astype(str).eq(snapshot.service_type.value)].copy()
    if selected.empty:
        raise ValueError(f"No services of type {snapshot.service_type.value} in canonical input")

    mode = snapshot.mode.value
    mode_attachments = attachments.loc[attachments["mode"].astype(str).eq(mode)].copy()
    if mode_attachments.empty or "graph_checksum" not in mode_attachments:
        raise ValueError("Mode attachments missing or lack graph_checksum")
    if mode_attachments["graph_checksum"].isna().any() or not mode_attachments["graph_checksum"].astype(str).eq(graph_sha).all():
        raise ValueError("Mode attachment graph checksum differs from network files")
    if mode_attachments.duplicated(["entity_kind", "entity_id"]).any():
        raise ValueError("Duplicate mode attachment identities")
    origin_ids = set(origins["origin_id"].astype(str))
    origin_att = mode_attachments.loc[mode_attachments["entity_kind"].eq("origin")]
    if set(origin_att["entity_id"].astype(str)) != origin_ids:
        raise ValueError("Origin attachments do not exactly match origins")
    selected_ids = set(selected["service_id"].astype(str))
    service_att = mode_attachments.loc[
        mode_attachments["entity_kind"].eq("service")
        & mode_attachments["entity_id"].astype(str).isin(selected_ids)
    ].copy()
    if set(service_att["entity_id"].astype(str)) != selected_ids:
        raise ValueError("Service attachments missing for requested service type")
    if "source_record_id" not in nodes.columns:
        raise ValueError("Network nodes are missing source_record_id")
    node_ids = [normalize_node_id(value) for value in nodes["source_record_id"]]
    if any(value is None for value in node_ids) or len(node_ids) != len(set(node_ids)):
        raise ValueError("Invalid or duplicated network nodes")
    graph_nodes = set(node_ids)
    relevant = pd.concat([origin_att, service_att], ignore_index=True)
    snapped = relevant.loc[relevant["snapped"].eq(True)]
    if not set(snapped["node_id"].dropna().astype(str)).issubset(graph_nodes):
        raise ValueError("Snapped attachment refers to a network node not present in graph")

    # Reuse B6A3, rather than deciding eligibility based only on snap flags.
    gated = prepare_service_destinations_v2(selected, service_att, mode=snapshot.mode)
    eligible = gated.loc[gated["routing_eligible"]].sort_values("service_id", kind="mergesort")
    samples = [{"service_id": str(row.service_id), "name": str(row.name)}
               for row in eligible.head(examples).itertuples(index=False)]
    out = {
        "policy": POLICY,
        "municipality_code": code,
        "service_type": snapshot.service_type.value,
        "mode": mode,
        "input_paths": {name: str(path.resolve()) for name, path in entries.items()},
        "input_sha256": before,
        "graph_sha256": graph_sha,
        "origin_count": len(origins),
        "origin_attachment_count": len(origin_att),
        "all_service_count": len(services),
        "selected_service_count": len(selected),
        "selected_eligible_count": len(eligible),
        "selected_ineligible_count": len(selected) - len(eligible),
        "selected_eligibility_reasons": {
            str(key): int(value)
            for key, value in gated["routing_exclusion_reason"].value_counts().items()
        },
        "example_removable_services": samples,
        "scenario_eligible": False,
        "notes": (
            "Diagnostic only: B6A3 eligible means usable for baseline routing; "
            "not authorization of new ANNCSU candidates. This command does not "
            "write files, approve candidates, or execute an overlay."
        ),
    }
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--service-type", default="pharmacy", choices=[s.value for s in ServiceType])
    parser.add_argument("--mode", default="walk", choices=[m.value for m in TransportMode])
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    snapshot = CitySnapshot(
        municipality_code=args.municipality_code, census_year=args.census_year,
        school_year=args.school_year, health_reference_date=args.health_reference_date,
        service_type=ServiceType(args.service_type), mode=TransportMode(args.mode),
    )
    report = inspect_real_snapshot(snapshot, canonical_paths(args.root, snapshot),
                                   examples=args.examples)
    print("=== B6C.3a CANONICAL REAL-DATA PREFLIGHT — READ ONLY ===")
    print(f"City {snapshot.municipality_code} / {snapshot.mode.value} / {snapshot.service_type.value}")
    print(f"Origins: {report['origin_count']} | Services selected: {report['selected_service_count']}")
    print(f"Eligible: {report['selected_eligible_count']} | Ineligible: {report['selected_ineligible_count']}")
    print(f"Graph checksum: {report['graph_sha256']}")
    for item in report["example_removable_services"]:
        print(f"  removable: {item['service_id']}  {item['name']}")
    print("READ-ONLY — no baselines, scenarios, or reports changed")
    return report


if __name__ == "__main__":
    run()
