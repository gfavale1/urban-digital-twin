"""B5C: standalone city-on-demand *source bootstrap* for methodology v2.

This runner resolves an ISTAT municipality, checks/acquires the pinned Bronze
snapshots and writes a v2 AnalysisSpec plus auditable RunManifest. It does NOT
execute network construction, routing, accessibility analysis, or Gold exports.
The original src/pipeline.py remains the legacy regression runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

from core.analysis_spec import AnalysisSpec
from core.config import DEFAULT_CONFIG
from core.municipality import MunicipalityContext
from core.paths import ROOT
from core.run_manifest import FreshnessStatus, RunManifest, SourceRecord as ManifestSourceRecord
from ingestion.source_acquisition import AcquisitionReport, inspect_sources, write_report


SOURCE_STAGE = "source_acquisition"
EXPECTED_SOURCE_SNAPSHOTS = {
    "istat_census_sections": "census_year",
    "istat_boundaries_2021": None,  # 2021 fixed territorial basis
    "mim_schools_state": "school_year",
    "mim_schools_paritary": "school_year",
    "mim_buildings": "building_year",
    "salute_pharmacies": "pharmacy_reference_date",
    "salute_hospital_establishments": "hospital_year",
}
EXPECTED_SOURCE_IDS = frozenset(EXPECTED_SOURCE_SNAPSHOTS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B5C city-on-demand methodology v2 bootstrap: resolve an ISTAT city, "
            "check official Bronze inputs, and record reproducible source provenance. "
            "Does not run accessibility computations."
        )
    )
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--city", help="Official ISTAT municipality name, e.g. Parma or Napoli.")
    location.add_argument("--municipality-code", help="Six-digit ISTAT municipality code.")
    parser.add_argument("--province-code", help="Disambiguate --city with three-digit province code.")
    parser.add_argument("--region-code", help="Disambiguate --city with two-digit region code.")
    parser.add_argument("--analysis-date", default=date.today().isoformat())
    parser.add_argument("--census-year", default=DEFAULT_CONFIG.census_year)
    parser.add_argument("--school-year", default=DEFAULT_CONFIG.school_year)
    parser.add_argument("--building-year", default=DEFAULT_CONFIG.building_year)
    parser.add_argument("--health-reference-date", default=DEFAULT_CONFIG.health_reference_date)
    parser.add_argument("--hospital-year", default=DEFAULT_CONFIG.hospital_year)
    parser.add_argument(
        "--fetch-supported", action="store_true",
        help="Explicitly acquire missing supported official sources (never overwrite cached Bronze).",
    )
    parser.add_argument(
        "--allow-contemporary-pharmacy-source", action="store_true",
        help="Allow a newer pharmacy registry for historical date filtering; completeness is not guaranteed.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Read-only preview: no network downloads, no artifacts, no analysis.",
    )
    args = parser.parse_args(argv)
    if args.city is None and (args.province_code is not None or args.region_code is not None):
        parser.error("--province-code and --region-code can only be used with --city.")
    if args.allow_contemporary_pharmacy_source and not args.fetch_supported:
        parser.error("--allow-contemporary-pharmacy-source requires --fetch-supported.")
    return args


def resolve_city(args: argparse.Namespace) -> MunicipalityContext:
    if args.city is not None:
        return MunicipalityContext.resolve_name(
            city_name=args.city,
            census_year=args.census_year,
            province_code=args.province_code,
            region_code=args.region_code,
        )
    return MunicipalityContext.resolve(
        municipality_code=args.municipality_code,
        census_year=args.census_year,
    )


def requested_snapshots(args: argparse.Namespace) -> dict[str, str]:
    return {
        "census_year": str(args.census_year),
        "school_year": str(args.school_year),
        "building_year": str(args.building_year),
        "pharmacy_reference_date": str(args.health_reference_date),
        "hospital_year": str(args.hospital_year),
    }


def _file_identity(path: str, root: Path) -> str:
    """Keep dataset fingerprints stable if the same repo is moved to another machine."""
    file_path = Path(path)
    try:
        return file_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Bronze source outside the repository root: {file_path}"
        ) from exc


def _source_fingerprint(source, root: Path) -> str | None:
    """Single file: SHA-256 of bytes. Multi-file: SHA-256 of sorted path/hash pairs.

    The full file-to-hash mapping is separately persisted in stage metrics.
    """
    if not source.paths:
        return None
    hashes = source.sha256
    if any(path not in hashes for path in source.paths):
        raise ValueError(f"Incomplete SHA-256 provenance for {source.source_id}.")
    if len(source.paths) == 1:
        return hashes[source.paths[0]]
    pairs = sorted((_file_identity(path, root), hashes[path]) for path in source.paths)
    canonical = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_report_context(
    report: AcquisitionReport,
    ctx: MunicipalityContext,
    snapshots: dict[str, str],
) -> None:
    if report.municipality_code != ctx.code:
        raise ValueError("Acquisition report municipality does not match requested ISTAT code.")
    for report_field, snapshot_key in (
        ("census_year", "census_year"),
        ("school_year", "school_year"),
        ("building_year", "building_year"),
        ("hospital_year", "hospital_year"),
        ("pharmacy_reference_date", "pharmacy_reference_date"),
    ):
        if str(getattr(report, report_field)) != snapshots[snapshot_key]:
            raise ValueError(f"Acquisition report vintage mismatch: {report_field}")
    for source in report.sources:
        snapshot_key = EXPECTED_SOURCE_SNAPSHOTS.get(source.source_id)
        expected = "2021" if source.source_id == "istat_boundaries_2021" else (
            snapshots[snapshot_key] if snapshot_key else None
        )
        if expected is not None and source.requested_snapshot != expected:
            raise ValueError(f"Acquisition source vintage mismatch: {source.source_id}")


def register_sources(
    manifest: RunManifest,
    report: AcquisitionReport,
    *,
    repo_root: Path,
) -> None:
    """Adapt Bronze reports to the existing RunManifest contracts.

    Acquisition status is *not* a freshness judgement. In particular, a cached
    historical pharmacy feed must not be marked 'fresh' simply for existing.
    """
    seen: set[str] = set()
    files: dict[str, dict[str, str]] = {}
    for source in report.sources:
        if source.source_id in seen:
            raise ValueError(f"Duplicate source ID in acquisition report: {source.source_id}")
        seen.add(source.source_id)
        if source.state in ("cached", "downloaded") and not source.paths:
            raise ValueError(f"Ready source without paths: {source.source_id}")
        source_checksum = _source_fingerprint(source, repo_root)
        for value in source.sha256.values():
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"Invalid SHA-256 value for {source.source_id}.")
        files[source.source_id] = {
            _file_identity(path, repo_root): source.sha256[path]
            for path in source.paths
            if path in source.sha256
        }
        note = source.note.strip()
        if source.state not in ("cached", "downloaded"):
            note = f"Acquisition state={source.state}. " + note
        manifest.register_source(
            source.source_id,
            ManifestSourceRecord(
                source_name=source.source_id,
                reference_date_or_period=source.requested_snapshot,
                # Neither the actual retrieval instant nor publication version
                # can be inferred from a pre-existing cached filename.
                retrieved_at=None,
                release_id_or_version=None,
                checksum_sha256=source_checksum,
                freshness_status=FreshnessStatus.UNKNOWN,
                source_url=source.source_url,
                license_name=None,
                warning=note or None,
            ),
        )
    if seen != EXPECTED_SOURCE_IDS:
        raise ValueError(
            f"Incomplete source inventory; missing={sorted(EXPECTED_SOURCE_IDS - seen)} "
            f"unexpected={sorted(seen - EXPECTED_SOURCE_IDS)}"
        )
    manifest.stages[SOURCE_STAGE].metrics["file_sha256_by_source"] = files
    manifest.stages[SOURCE_STAGE].metrics["source_states"] = {
        source.source_id: source.state for source in report.sources
    }


def acquire(ctx: MunicipalityContext, args: argparse.Namespace, *, allow_fetch: bool) -> AcquisitionReport:
    return inspect_sources(
        ctx,
        census_year=args.census_year,
        school_year=args.school_year,
        building_year=args.building_year,
        pharmacy_reference_date=args.health_reference_date,
        hospital_year=args.hospital_year,
        fetch_supported=allow_fetch,
        allow_contemporary_pharmacy_source=(
            args.allow_contemporary_pharmacy_source if allow_fetch else False
        ),
    )


def print_report(ctx: MunicipalityContext, report: AcquisitionReport, *, dry_run: bool) -> None:
    print("\n=== B5C CITY-ON-DEMAND SOURCE BOOTSTRAP ===")
    print(f"Municipality: {ctx.name} ({ctx.code})")
    print(f"Region: {ctx.region_code} | Province: {ctx.province_code}")
    if dry_run:
        print("Mode: READ-ONLY PREVIEW (no downloads or artifacts)")
    for source in report.sources:
        print(f"{source.source_id:34s} {source.state}")
    print(f"SOURCES READY: {report.ready}")
    print("Accessibility analysis: NOT EXECUTED (B5C bootstrap only)")


def run(args: argparse.Namespace, *, repo_root: Path = ROOT) -> int:
    ctx = resolve_city(args)
    spec = AnalysisSpec.default_for_city(
        ctx.name,
        analysis_date=str(args.analysis_date),
        municipality_code=ctx.code,
    )
    if spec.execution_profile != "methodology_v2":
        raise RuntimeError("B5C requires a methodology_v2 AnalysisSpec.")
    snapshots = requested_snapshots(args)

    if args.dry_run:
        # Preview must be safe even when --fetch-supported is supplied.
        report = acquire(ctx, args, allow_fetch=False)
        validate_report_context(report, ctx, snapshots)
        print_report(ctx, report, dry_run=True)
        return 0

    manifest = RunManifest.create(
        spec,
        repo_root=repo_root,
        routing_backend="not_executed_b5c_bootstrap",
        routing_parameters={
            "execution_scope": "official_source_bootstrap_only",
            "requested_source_snapshots": snapshots,
            "routing_executed": False,
        },
    )
    # AnalysisSpec v2 currently has no vintage fields. Preserve all vintages
    # in the manifest and avoid run_id collisions between distinct requests.
    snapshot_fingerprint = hashlib.sha256(
        json.dumps(snapshots, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    manifest.run_id = f"{manifest.run_id}_{snapshot_fingerprint}"
    run_dir = repo_root / "runs" / manifest.run_id
    # Never clobber provenance from a previous run in the same second.
    run_dir.mkdir(parents=True, exist_ok=False)
    spec_path = run_dir / "analysis_spec.json"
    manifest_path = run_dir / "run_manifest.json"
    sources_path = run_dir / "source_acquisition.json"

    spec.write_json(spec_path)
    stage = manifest.start_stage(SOURCE_STAGE)
    stage.metrics["requested_source_snapshots"] = snapshots
    stage.metrics["source_freshness_verified"] = False
    manifest.write_json(manifest_path)
    try:
        report = acquire(ctx, args, allow_fetch=bool(args.fetch_supported))
        validate_report_context(report, ctx, snapshots)
        write_report(report, sources_path)
        register_sources(manifest, report, repo_root=repo_root)
        stage.metrics["sources_ready"] = report.ready
        if report.ready:
            manifest.complete_stage(
                SOURCE_STAGE,
                outputs=[str(sources_path)],
                metrics={"source_count": len(report.sources)},
            )
        else:
            missing = [f"{s.source_id}={s.state}" for s in report.sources
                       if s.state not in ("cached", "downloaded")]
            manifest.fail_stage(SOURCE_STAGE, "Bronze source gate failed: " + ", ".join(missing))
    except Exception as exc:
        manifest.fail_stage(SOURCE_STAGE, f"{type(exc).__name__}: {exc}")
        manifest.write_json(manifest_path)
        raise
    manifest.write_json(manifest_path)
    print_report(ctx, report, dry_run=False)
    print(f"Run directory: {run_dir}")
    print(f"AnalysisSpec: {spec_path}")
    print(f"RunManifest: {manifest_path}")
    print(f"Source report: {sources_path}")
    if not report.ready:
        print("STOP: incomplete sources; downstream stages NOT executed.")
        return 2
    print("BOOTSTRAP READY: downstream v2 stages can be integrated in a later phase.")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
