"""B7B: opt-in, census-footprint-qualified real walking removal scenario.

This does not replace B6C.3b. It verifies a separately materialized, immutable
boundary-qualified service view before routing. Source services and the ISTAT
2021 *census footprint* (not the certified municipality boundary) are bound
by SHA-256; legacy scenario IDs and results are preserved unchanged.

First explicitly publish the derived view:
  PYTHONPATH=src python src/quality/derive_boundary_qualified_services.py \\
    --municipality-code 034027 --mode walk --write \\
    --output data/features/quality/b7b/034027/walk_services.parquet

Then run this script with --qualified-services pointing to that file.
It is read-only unless --write-report is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from analysis.run_real_service_removal import execute_real_removal
from analysis.scenario_reporting_v2 import ScenarioInputPaths
from core.analysis_spec import ServiceType, TransportMode
from core.run_manifest import sha256_file
from quality.derive_boundary_qualified_services import (
    POLICY as VIEW_POLICY, _stage_file, inspect_boundary_view,
)
from quality.preflight_real_service_scenario import (
    ROOT, CitySnapshot, canonical_paths, inspect_real_snapshot,
)

POLICY = "b7b_qualified_real_service_removal_experiment_v1"


def verify_boundary_view(
    *, snapshot: CitySnapshot, canonical: ScenarioInputPaths,
    qualified_services: Path, root: Path = ROOT,
) -> dict[str, Any]:
    """Independently reconstruct the view from canonical sources and compare bytes.

    Reject fabricated manifests, stale perimeter/catalogue, and tampered views.
    Verifying the *whole* view is stronger than trusting a stored inside flag.
    """
    qualified_services = Path(qualified_services)
    root = Path(root)
    if qualified_services.suffix.lower() not in (".parquet", ".csv"):
        raise ValueError("B7B qualified service view must be .parquet or .csv")
    manifest_path = qualified_services.with_suffix(qualified_services.suffix + ".manifest.json")
    if not qualified_services.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("B7B qualified view and its manifest must both exist")
    if qualified_services.resolve() == canonical.services.resolve():
        raise ValueError("B7B view must not overwrite the canonical catalogue")

    original_sha = sha256_file(canonical.services)
    expected, evidence = inspect_boundary_view(snapshot, root=root)
    source_paths = evidence["source_paths"]
    if Path(source_paths["services"]).resolve() != canonical.services.resolve():
        raise ValueError("B7B view was built from a different canonical service catalogue")
    if evidence["source_sha256"]["services"] != original_sha:
        raise ValueError("B7B canonical services changed while deriving view")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("policy", "municipality_code", "source_paths", "source_sha256",
                "boundary_basis", "administrative_boundary_verified", "services_total",
                "inside_footprint", "outside_footprint", "location_validated_outside"):
        if manifest.get(key) != evidence[key]:
            raise ValueError(f"B7B derived-view manifest evidence mismatch: {key}")
    if manifest.get("view_path") != str(qualified_services.resolve()):
        raise ValueError("B7B derived-view manifest path mismatch")

    stored_hash = sha256_file(qualified_services)
    if manifest.get("view_sha256") != stored_hash:
        raise ValueError("B7B qualified service view has invalid manifest checksum")
    # Exact byte check makes the derived view reproducible from *these* inputs,
    # including all flag values, source columns, and provenance fingerprints.
    expected_hash = hashlib.sha256(_stage_file(expected, qualified_services)).hexdigest()
    if expected_hash != stored_hash:
        raise ValueError("B7B qualified view differs from independently reconstructed footprint view")
    if sha256_file(canonical.services) != original_sha or (
        sha256_file(Path(source_paths["footprint"])) != evidence["source_sha256"]["footprint"]
    ):
        raise ValueError("B7B canonical catalogue or footprint changed during verification")
    return {
        "policy": VIEW_POLICY,
        "boundary_basis": evidence["boundary_basis"],
        "administrative_boundary_verified": False,
        "canonical_services_path": source_paths["services"],
        "canonical_services_sha256": original_sha,
        "footprint_path": source_paths["footprint"],
        "footprint_sha256": evidence["source_sha256"]["footprint"],
        "qualified_view_path": str(qualified_services.resolve()),
        "qualified_view_sha256": stored_hash,
        "qualified_view_manifest_sha256": sha256_file(manifest_path),
        "services_total": evidence["services_total"],
        "inside_footprint": evidence["inside_footprint"],
        "outside_footprint": evidence["outside_footprint"],
        "validated_outside_footprint": evidence["location_validated_outside"],
        "scenario_routing_active": True,
    }


def execute_boundary_qualified_removal(
    *, snapshot: CitySnapshot, canonical: ScenarioInputPaths,
    qualified_services: Path, city_name: str, analysis_date: str,
    remove_service_id: str, root: Path = ROOT,
    write_report: bool = False, output_root: Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Verify original data and qualified view, then reuse B6C.3b unchanged."""
    inspection = inspect_real_snapshot(snapshot, canonical, examples=0)
    qualification = verify_boundary_view(
        snapshot=snapshot, canonical=canonical,
        qualified_services=qualified_services, root=root,
    )
    verified_paths = replace(canonical, services=Path(qualified_services))
    # This re-runs B6C.3a against the *qualified* view and B6C.2 hashes its
    # bytes before constructing an independent scenario baseline. Old B6C IDs
    # never change and no formerly excluded service becomes eligible.
    result, report_path = execute_real_removal(
        snapshot=snapshot, paths=verified_paths,
        service_id=remove_service_id, city_name=city_name,
        analysis_date=analysis_date, write_report=write_report,
        output_root=output_root, territorial_provenance=qualification,
    )
    if result["policy"] != POLICY:
        raise AssertionError("B7B scenario execution has the wrong policy")
    if result["preflight_eligible_services"] > inspection["selected_eligible_count"]:
        raise AssertionError("B7B territorial filter unexpectedly promoted a service")
    original_hashes = inspection["input_sha256"]
    for name, path in canonical.items().items():
        if sha256_file(path) != original_hashes[name]:
            raise ValueError(f"Canonical input changed during B7B scenario: {name}")
    if sha256_file(Path(qualification["footprint_path"])) != qualification["footprint_sha256"]:
        raise ValueError("Footprint changed during B7B scenario")
    if sha256_file(qualified_services) != qualification["qualified_view_sha256"]:
        raise ValueError("Qualified service view changed during B7B scenario")
    result["canonical_baseline_service_count"] = inspection["selected_eligible_count"]
    return result, report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--city-name", default=None)
    parser.add_argument("--analysis-date", required=True)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--service-type", default="pharmacy", choices=[s.value for s in ServiceType])
    parser.add_argument("--mode", default="walk", choices=[m.value for m in TransportMode])
    parser.add_argument("--remove-service-id", required=True)
    parser.add_argument("--qualified-services", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    snapshot = CitySnapshot(
        municipality_code=args.municipality_code,
        census_year=args.census_year, school_year=args.school_year,
        health_reference_date=args.health_reference_date,
        service_type=ServiceType(args.service_type), mode=TransportMode(args.mode),
    )
    summary, _ = execute_boundary_qualified_removal(
        snapshot=snapshot, canonical=canonical_paths(args.root, snapshot),
        qualified_services=args.qualified_services,
        remove_service_id=args.remove_service_id,
        city_name=args.city_name or f"Comune {snapshot.municipality_code}",
        analysis_date=args.analysis_date, root=args.root,
        write_report=args.write_report,
        output_root=(args.output_root or args.root / "data/features/scenarios/b7b")
        if args.write_report else None,
    )
    print("=== B7B QUALIFIED REAL REMOVAL — %s ===" %
          ("REPORT SAVED" if args.write_report else "READ ONLY"))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return summary


if __name__ == "__main__":
    run()
