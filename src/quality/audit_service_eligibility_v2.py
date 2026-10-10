"""B6A3 auditable destination eligibility report; never alters canonical input.

Usage: PYTHONPATH=src python src/quality/audit_service_eligibility_v2.py
  --municipality-code 034027 --mode walk --dry-run
Remove --dry-run to write quality report CSV/JSON in data/features/quality/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from analysis.accessibility_contracts import prepare_service_destinations_v2
from analysis.service_eligibility_v2 import ELIGIBILITY_POLICY
from core.analysis_spec import TransportMode

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--mode", required=True, choices=[m.value for m in TransportMode])
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--dry-run", action="store_true", help="Read and report; no files written")
    args = parser.parse_args(argv)
    if len(args.municipality_code) != 6 or not args.municipality_code.isdigit():
        parser.error("--municipality-code must be exactly six digits")
    try:
        args.health_reference_date = pd.Timestamp(args.health_reference_date).strftime("%Y%m%d")
    except (ValueError, TypeError):
        parser.error("--health-reference-date must be a valid date")
    return args


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _as_count_dict(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(series.value_counts(dropna=False).items())}


def audit_destinations(
    services: pd.DataFrame,
    attachments: pd.DataFrame,
    *,
    mode: TransportMode,
) -> tuple[pd.DataFrame, dict]:
    """Pure, validated audit for service-type availability; no routing executed."""
    mode = TransportMode(mode)
    destinations = prepare_service_destinations_v2(services, attachments, mode=mode)
    display_fields = [
        "service_id", "service_type", "operational_status", "legacy_usable_for_accessibility",
        "snapped", "attachment_node_id", "attachment_snap_distance_m",
        "network_component_id", "is_largest_component", "routing_eligible", "routing_exclusion_reason",
    ]
    present = [field for field in display_fields if field in destinations.columns]
    evidence = destinations[present].sort_values("service_id", kind="stable").reset_index(drop=True)
    validated = (
        destinations.get("legacy_usable_for_accessibility", pd.Series(False, index=destinations.index))
        .map(lambda x: isinstance(x, (bool, np.bool_)) and bool(x))
    )
    snapped = destinations["snapped"].astype(bool)
    eligible = destinations["routing_eligible"].astype(bool)
    graph_checksums = []
    if "graph_checksum" in attachments.columns:
        selected = attachments.loc[
            (attachments["mode"].astype(str) == mode.value)
            & (attachments["entity_kind"].astype(str) == "service")
        ]
        graph_checksums = sorted(set(selected["graph_checksum"].dropna().astype(str)))
        if len(graph_checksums) > 1:
            raise ValueError("Mode service attachments have different graph checksums.")
    report = {
        "eligibility_policy": ELIGIBILITY_POLICY,
        "mode": mode.value,
        "total_services": int(len(destinations)),
        "attachment_snapped": int(snapped.sum()),
        "validated_legacy": int(validated.sum()),
        "validated_and_snapped": int((validated & snapped).sum()),
        "snapped_not_validated": int((~validated & snapped).sum()),
        "eligible_destinations": int(eligible.sum()),
        "exclusion_reasons": _as_count_dict(destinations["routing_exclusion_reason"]),
        "eligible_by_service_type": {
            str(k): int(v)
            for k, v in sorted(destinations.loc[eligible, "service_type"].value_counts().items())
        },
        "attachment_graph_checksums": graph_checksums,
        "notes": [
            "Attachment snapping is a geometric association, not independent validation.",
            "Only validated, active, snapped destinations are routable in canonical v2.",
            "Records on non-largest network components are not summarily excluded.",
            "This is a destination audit; no accessibility computation or service correction was performed.",
        ],
    }
    return evidence, report


def run(args: argparse.Namespace, *, root: Path = ROOT) -> dict:
    code = args.municipality_code
    label = args.health_reference_date
    services_path = root / "data" / "processed" / "services" / code / f"service_entities_v2_{args.school_year}_{label}.parquet"
    attachments_path = root / "data" / "processed" / "network_attachments" / code / (
        f"network_attachments_v2_{args.mode}_{args.census_year}_{args.school_year}_{label}.parquet"
    )
    for source in (services_path, attachments_path):
        if not source.is_file():
            raise FileNotFoundError(f"Missing canonical input: {source}")
    services = pd.read_parquet(services_path)
    attachments = pd.read_parquet(attachments_path)
    evidence, report = audit_destinations(services, attachments, mode=TransportMode(args.mode))
    report["municipality_code"] = code
    report["input_sha256"] = {"services": sha256_file(services_path), "attachments": sha256_file(attachments_path)}
    print("=== B6A3 V2 DESTINATION QUALITY GATE ===")
    print(f"Municipality: {code} | Mode: {args.mode}")
    for key in ("total_services", "attachment_snapped", "validated_legacy", "validated_and_snapped", "snapped_not_validated", "eligible_destinations"):
        print(f"{key}: {report[key]}")
    print(f"Exclusion reasons: {report['exclusion_reasons']}")
    if args.dry_run:
        print("DRY RUN — no files modified")
        return report
    directory = root / "data" / "features" / "quality" / code
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"service_destination_eligibility_v2_{args.mode}_{args.census_year}_{args.school_year}_{label}"
    csv_path = directory / f"{stem}.csv"
    json_path = directory / f"{stem}_manifest.json"
    evidence.to_csv(csv_path, index=False, encoding="utf-8-sig")
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["output_csv_sha256"] = sha256_file(csv_path)
    report["output_csv"] = str(csv_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(f"Audit CSV: {csv_path}")
    print(f"Manifest: {json_path}")
    return report


if __name__ == "__main__":
    run(parse_args())
