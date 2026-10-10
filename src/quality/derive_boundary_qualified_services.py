"""B7B: derive a versioned, operational-footprint-qualified ServiceV2 view.

Keep the original canonical ServiceV2 catalogue and the legacy network-ready
layers unchanged. The ISTAT census footprint is an operational study perimeter,
NOT a certified municipal administrative boundary. This creates a *derived*
input for a future scenario run; it does not silently change existing reports.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from analysis.scenario_reporting_v2 import _read_table
from core.analysis_spec import TransportMode
from core.run_manifest import sha256_file
from core.schema_v2 import SERVICE_V2
from quality.audit_legacy_boundary_policy import _footprint_union
from quality.audit_v2_service_boundary import _inside_census_footprint, paths_for_snapshot
from quality.preflight_real_service_scenario import ROOT, CitySnapshot

POLICY = "b7b_operational_census_footprint_qualified_service_view_v1"
FLAG = "inside_operational_footprint"


def qualify_v2_services(
    services: pd.DataFrame, footprint: gpd.GeoDataFrame, *,
    municipality_code: str, canonical_sha256: str, footprint_sha256: str,
) -> pd.DataFrame:
    """Return an independent copy, with fail-closed inside/outside flags.

    Both SHA-256 values are stamped into the result so an unchanged spatial
    classification under changed inputs still receives a new file identity.
    """
    if not re.fullmatch(r"\d{6}", municipality_code):
        raise ValueError("municipality_code must have exactly six digits")
    for label, digest in (("canonical", canonical_sha256), ("footprint", footprint_sha256)):
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError(f"Invalid {label} SHA-256")
    SERVICE_V2.validate_columns(services.columns)
    if services.empty or services["service_id"].isna().any() or services["service_id"].astype(str).duplicated().any():
        raise ValueError("ServiceV2 must contain distinct, non-null service IDs")
    if not services["municipality_code"].astype(str).eq(municipality_code).all():
        raise ValueError("ServiceV2 municipality code mismatch")
    if any(c in services.columns for c in (FLAG, "operational_footprint_sha256", "canonical_services_sha256", "territorial_policy")):
        raise ValueError("Input is already boundary-qualified; require raw canonical ServiceV2")
    _footprint_union(footprint)
    inside = _inside_census_footprint(services, footprint)
    result = services.copy(deep=True)
    result[FLAG] = inside.astype(bool)
    result["operational_footprint_sha256"] = footprint_sha256
    result["canonical_services_sha256"] = canonical_sha256
    result["territorial_policy"] = POLICY
    return result


def inspect_boundary_view(snapshot: CitySnapshot, *, root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read, fingerprint, qualify, and recheck source bytes. No writes."""
    paths = paths_for_snapshot(Path(root), snapshot)
    for label, p in (("services", paths.services), ("footprint", paths.footprint)):
        if not p.is_file():
            raise FileNotFoundError(f"B7B missing {label}: {p}")
    source_hashes = {
        "services": sha256_file(paths.services),
        "footprint": sha256_file(paths.footprint),
    }
    raw = _read_table(paths.services)
    footprint = gpd.read_parquet(paths.footprint)
    view = qualify_v2_services(raw, footprint, municipality_code=snapshot.municipality_code,
                               canonical_sha256=source_hashes["services"],
                               footprint_sha256=source_hashes["footprint"])
    if source_hashes != {"services": sha256_file(paths.services), "footprint": sha256_file(paths.footprint)}:
        raise ValueError("B7B inputs changed during boundary qualification")
    inside = view[FLAG]
    before_routable = raw["legacy_usable_for_accessibility"].eq(True).sum()
    return view, {
        "policy": POLICY,
        "municipality_code": snapshot.municipality_code,
        "boundary_basis": "ISTAT_2021_census_area_footprint_not_administrative_boundary",
        "administrative_boundary_verified": False,
        "source_paths": {"services": str(paths.services.resolve()), "footprint": str(paths.footprint.resolve())},
        "source_sha256": source_hashes,
        "services_total": int(len(view)),
        "inside_footprint": int(inside.sum()),
        "outside_footprint": int((~inside).sum()),
        "legacy_location_validated": int(before_routable),
        "location_validated_outside": int((~inside & raw["legacy_usable_for_accessibility"].eq(True)).sum()),
        "scenario_routing_active": False,
    }


def _stage_file(view: pd.DataFrame, out: Path) -> bytes:
    """Build bytes off to the side, never write directly to the final artifact."""
    if out.suffix not in (".csv", ".parquet"):
        raise ValueError("Output must have .csv or .parquet extension")
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / out.name
        if out.suffix == ".csv":
            view.to_csv(staged, index=False, lineterminator="\n")
        else:
            view.to_parquet(staged, index=False)
        return staged.read_bytes()


def write_boundary_view(view: pd.DataFrame, report: dict[str, Any], output: Path) -> tuple[Path, bool]:
    """Only explicit caller action persists; refuse conflicting overwrite."""
    output = Path(output)
    if output.suffix not in (".parquet", ".csv"):
        raise ValueError("Boundary-qualified service artifact must be .parquet or .csv")
    payload = _stage_file(view, output)
    import hashlib
    digest = hashlib.sha256(payload).hexdigest()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    metadata = {**report, "view_path": str(output.resolve()), "view_sha256": digest}
    manifest_data = (json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if output.exists() or manifest_path.exists():
        if (not output.is_file() or not manifest_path.is_file()
                or output.read_bytes() != payload or manifest_path.read_bytes() != manifest_data):
            raise FileExistsError("B7B immutable derived view differs or has incomplete output")
        return output, True
    output.parent.mkdir(parents=True, exist_ok=True)
    # A future orchestrator must serialize concurrent publication of the same output.
    # Reject preexisting artifacts instead of overwriting them.
    output.write_bytes(payload)
    try:
        manifest_path.write_bytes(manifest_data)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output, False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--mode", default="walk", choices=[m.value for m in TransportMode])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.write != (args.output is not None):
        raise ValueError("Use --write and --output together; default is strictly read-only")
    snap = CitySnapshot(
        municipality_code=args.municipality_code, census_year=args.census_year,
        school_year=args.school_year, health_reference_date=args.health_reference_date,
        mode=TransportMode(args.mode),
    )
    view, report = inspect_boundary_view(snap, root=args.root)
    print("=== B7B CENSUS-FOOTPRINT QUALIFIED V2 VIEW ===")
    print("Municipality: %s | Service rows: %d | inside=%d outside=%d" % (
        snap.municipality_code, report["services_total"], report["inside_footprint"], report["outside_footprint"]))
    print("Location-validated services outside footprint: %d" % report["location_validated_outside"])
    print("Administrative boundary: NOT VERIFIED")
    if args.write:
        output, cached = write_boundary_view(view, report, args.output)
        print("DERIVED VIEW SAVED: %s | cached=%s" % (output, str(cached).lower()))
    else:
        print("READ-ONLY PREVIEW: canonical files and previous reports unchanged")
    return report


if __name__ == "__main__":
    run()
