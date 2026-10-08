"""B5B2: deterministic Bronze preflight + opt-in verified official acquisition.

Cached files are never refreshed or overwritten, and all newly acquired sources
are validated and recorded in the provenance manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.municipality import MunicipalityContext
from core.paths import ROOT
from ingestion import download_istat_boundaries, mim, official_downloads


RAW_ISTAT = ROOT / "data" / "raw" / "istat"
RAW_MIM = ROOT / "data" / "raw" / "mim"
RAW_SALUTE = ROOT / "data" / "raw" / "salute"
FEATURES_ACQUISITION = ROOT / "data" / "features" / "source_acquisition"

VALID_STATES = frozenset(
    {"cached", "downloaded", "missing", "ambiguous", "invalid", "unsupported_snapshot", "acquisition_failed"}
)


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    state: str
    requested_snapshot: str
    paths: tuple[str, ...] = ()
    sha256: dict[str, str] = field(default_factory=dict)
    source_url: str | None = None
    official_catalogue: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.state not in VALID_STATES:
            raise ValueError(f"Invalid source state: {self.state}")


@dataclass(frozen=True)
class AcquisitionReport:
    municipality_code: str
    municipality_name: str
    census_year: str
    school_year: str
    building_year: str
    hospital_year: str
    pharmacy_reference_date: str
    fetch_supported: bool
    generated_at_utc: str
    sources: tuple[SourceRecord, ...]
    allow_contemporary_pharmacy_source: bool = False

    @property
    def ready(self) -> bool:
        return all(source.state in {"cached", "downloaded"} for source in self.sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "sources"},
            "ready": self.ready,
            "sources": [asdict(source) for source in self.sources],
            "notes": [
                "B5B2 checks file presence, uniqueness, checksums, and newly downloaded file structure; it does not prove historical source completeness.",
                "Cached raw files are never refreshed and are not assumed up-to-date: their exact bytes are tracked using SHA-256.",
                "Official download URLs are pinned for 2023/202425 except the dated pharmacy feed discovered from its Ministry catalogue.",
                "A pharmacy analysis date is a temporal filter on the available registry, not the publication date of that registry.",
            ],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(
    source_id: str,
    state: str,
    snapshot: str,
    *,
    paths: tuple[Path, ...] = (),
    source_url: str | None = None,
    official_catalogue: str | None = None,
    note: str = "",
) -> SourceRecord:
    hashes = {_path.as_posix(): _sha256(_path) for _path in paths if _path.is_file()}
    return SourceRecord(
        source_id=source_id,
        state=state,
        requested_snapshot=snapshot,
        paths=tuple(path.as_posix() for path in paths),
        sha256=hashes,
        source_url=source_url,
        official_catalogue=official_catalogue,
        note=note,
    )


def _unique_cached(
    source_id: str,
    directory: Path,
    pattern: str,
    snapshot: str,
    *,
    official_catalogue: str | None = None,
) -> SourceRecord:
    paths = tuple(sorted(directory.glob(pattern))) if directory.exists() else ()
    if not paths:
        return _record(
            source_id, "missing", snapshot,
            official_catalogue=official_catalogue,
            note=f"No matching raw file in {directory} (pattern: {pattern}).",
        )
    if len(paths) != 1:
        return _record(
            source_id, "ambiguous", snapshot, paths=paths,
            official_catalogue=official_catalogue,
            note="Multiple raw files found. Explicit snapshot selection is required; no mtime fallback.",
        )
    path = paths[0]
    if not path.is_file() or path.stat().st_size == 0:
        return _record(
            source_id, "invalid", snapshot, paths=(path,),
            official_catalogue=official_catalogue,
            note="Raw input must be a nonempty file.",
        )
    return _record(source_id, "cached", snapshot, paths=(path,), official_catalogue=official_catalogue)


def inspect_istat_census(
    region_code: str,
    census_year: str,
    *,
    fetch_supported: bool = False,
) -> SourceRecord:
    year = str(census_year).strip()
    if year != "2023":
        return _record(
            "istat_census_sections", "unsupported_snapshot", year,
            note="Current istat.py and census source paths are fixed to the 2023 census; no silent substitution.",
        )
    directory = RAW_ISTAT / f"censimento_{year}" / f"Dati_regionali_{year}"
    catalogue = "https://www.istat.it/notizia/dati-per-sezioni-di-censimento/"
    record = _unique_cached(
        "istat_census_sections", directory,
        f"R{region_code}_*_{year}_sezioni.xlsx", year,
        official_catalogue=catalogue,
    )
    if record.state != "missing" or not fetch_supported:
        return record
    try:
        acquired = official_downloads.download_istat_region(
            region_code, year, directory,
            RAW_ISTAT / "downloads",
        )
        return _record(
            "istat_census_sections", "downloaded", year,
            paths=(acquired.path,), source_url=acquired.source_url,
            official_catalogue=catalogue, note=acquired.note,
        )
    except Exception as exc:
        return _record(
            "istat_census_sections", "acquisition_failed", year,
            official_catalogue=catalogue,
            note=f"{type(exc).__name__}: {exc}",
        )


def inspect_istat_boundaries(region_code: str, *, fetch_supported: bool) -> SourceRecord:
    region_code = download_istat_boundaries.normalize_region_code(region_code)
    sidecars = tuple(download_istat_boundaries.expected_sidecars(region_code))
    present, _ = download_istat_boundaries.validate_region_files(region_code)
    if present and all(path.is_file() and path.stat().st_size > 0 for path in sidecars):
        return _record("istat_boundaries_2021", "cached", "2021", paths=sidecars)
    if not fetch_supported:
        return _record(
            "istat_boundaries_2021", "missing", "2021",
            note="Region boundaries missing or incomplete; run again with --fetch-supported to download.",
        )
    try:
        download_istat_boundaries.ensure_region_boundaries(region_code)
        present, _ = download_istat_boundaries.validate_region_files(region_code)
        if not present or not all(path.is_file() and path.stat().st_size > 0 for path in sidecars):
            return _record(
                "istat_boundaries_2021", "invalid", "2021", paths=sidecars,
                note="Downloaded boundaries failed the sidecar/nonempty-file check.",
            )
        return _record("istat_boundaries_2021", "downloaded", "2021", paths=sidecars)
    except Exception as exc:
        return _record(
            "istat_boundaries_2021", "acquisition_failed", "2021",
            note=f"{type(exc).__name__}: {exc}",
        )


def inspect_mim_registry(
    dataset: str,
    school_year: str,
    *,
    fetch_supported: bool,
) -> SourceRecord:
    source_id = "mim_schools_state" if dataset == "SCUANAGRAFESTAT" else "mim_schools_paritary"
    snapshot = str(school_year)
    if snapshot != "202425":
        return _record(
            source_id, "unsupported_snapshot", snapshot,
            note="Existing MIM ingestion only supports explicit 202425 filename snapshots.",
        )
    directory = RAW_MIM / "schools" / snapshot
    filenames = mim.candidate_filenames(dataset, snapshot)
    paths = tuple(directory / name for name in filenames if (directory / name).exists())
    if len(paths) > 1:
        return _record(
            source_id, "ambiguous", snapshot, paths=paths,
            note="Multiple official snapshot candidates in cache; pin one before processing.",
        )
    if len(paths) == 1:
        path = paths[0]
        if not path.is_file() or path.stat().st_size == 0:
            return _record(source_id, "invalid", snapshot, paths=(path,), note="Empty/invalid MIM CSV.")
        return _record(source_id, "cached", snapshot, paths=(path,))
    if not fetch_supported:
        return _record(
            source_id, "missing", snapshot,
            note=f"No supported MIM raw CSV in {directory}; --fetch-supported permits official download.",
        )
    try:
        path, source_url = mim.download_official_file(
            dataset=dataset,
            school_year=snapshot,
            target_dir=directory,
            refresh=False,
        )
        if not path.is_file() or path.stat().st_size == 0:
            return _record(source_id, "invalid", snapshot, paths=(path,), note="Downloaded MIM file is empty.")
        return _record(
            source_id, "downloaded", snapshot, paths=(path,),
            source_url=(source_url if str(source_url).startswith("https://") else None),
        )
    except Exception as exc:
        return _record(
            source_id, "acquisition_failed", snapshot,
            note=f"{type(exc).__name__}: {exc}",
        )


def inspect_mim_buildings(
    building_year: str,
    *,
    fetch_supported: bool = False,
) -> SourceRecord:
    directory = RAW_MIM / "buildings" / str(building_year)
    record = _unique_cached(
        "mim_buildings", directory, "*.csv", str(building_year),
        official_catalogue="https://dati.istruzione.it/opendata/opendata/catalog/EDIANAGRAFESTA2021",
    )
    if record.state == "missing" and fetch_supported:
        if str(building_year) != "202425":
            return _record(
                "mim_buildings", "unsupported_snapshot", str(building_year),
                note="Only the official MIM 2024/25 building distribution is pinned.",
            )
        try:
            acquired = official_downloads.download_mim_buildings(building_year, directory)
            return _record(
                "mim_buildings", "downloaded", str(building_year),
                paths=(acquired.path,), source_url=acquired.source_url,
                official_catalogue=official_downloads.MIM_202425_BUILDINGS_URL,
                note=acquired.note,
            )
        except Exception as exc:
            return _record(
                "mim_buildings", "acquisition_failed", str(building_year),
                official_catalogue=official_downloads.MIM_202425_BUILDINGS_URL,
                note=f"{type(exc).__name__}: {exc}",
            )
    if record.state != "cached":
        return record
    filename = Path(record.paths[0]).name.upper()
    if "EDIANAGRAFESTA" not in filename:
        return _record(
            "mim_buildings", "invalid", str(building_year),
            paths=(Path(record.paths[0]),),
            note="Building CSV does not have the expected EDIANAGRAFESTA dataset prefix.",
        )
    return record


def inspect_health(
    pharmacy_reference_date: str,
    hospital_year: str,
    *,
    fetch_supported: bool = False,
    allow_contemporary_pharmacy_source: bool = False,
) -> tuple[SourceRecord, SourceRecord]:
    pharmacies = _unique_cached(
        "salute_pharmacies",
        RAW_SALUTE / "farmacie", "*.csv", str(pharmacy_reference_date),
        official_catalogue="https://www.dati.salute.gov.it/it/dataset/farmacie/",
    )
    hospitals = _unique_cached(
        "salute_hospital_establishments",
        RAW_SALUTE / f"strutture_ospedaliere_{hospital_year}",
        "*.csv", str(hospital_year),
        official_catalogue=(
            "https://www.dati.salute.gov.it/it/dataset/"
            "posti-letto-stabilimento-ospedaliero-e-disciplina-2023/"
            if str(hospital_year) == "2023" else None
        ),
    )
    if fetch_supported and pharmacies.state == "missing":
        try:
            acquired = official_downloads.download_salute_pharmacies(
                pharmacy_reference_date,
                RAW_SALUTE / "farmacie",
                allow_contemporary_source=allow_contemporary_pharmacy_source,
            )
            pharmacies = _record(
                "salute_pharmacies", "downloaded", str(pharmacy_reference_date),
                paths=(acquired.path,), source_url=acquired.source_url,
                official_catalogue=official_downloads.SALUTE_PHARMACIES_CATALOGUE,
                note=acquired.note,
            )
        except PermissionError as exc:
            pharmacies = _record(
                "salute_pharmacies", "unsupported_snapshot", str(pharmacy_reference_date),
                official_catalogue=official_downloads.SALUTE_PHARMACIES_CATALOGUE,
                note=str(exc),
            )
        except Exception as exc:
            pharmacies = _record(
                "salute_pharmacies", "acquisition_failed", str(pharmacy_reference_date),
                official_catalogue=official_downloads.SALUTE_PHARMACIES_CATALOGUE,
                note=f"{type(exc).__name__}: {exc}",
            )

    if fetch_supported and hospitals.state == "missing":
        if str(hospital_year) != "2023":
            hospitals = _record(
                "salute_hospital_establishments", "unsupported_snapshot", str(hospital_year),
                note="Only the official 2023 hospital file is pinned.",
            )
        else:
            try:
                acquired = official_downloads.download_salute_hospitals(
                    hospital_year,
                    RAW_SALUTE / f"strutture_ospedaliere_{hospital_year}",
                )
                hospitals = _record(
                    "salute_hospital_establishments", "downloaded", str(hospital_year),
                    paths=(acquired.path,), source_url=acquired.source_url,
                    official_catalogue=(
                        "https://www.dati.salute.gov.it/it/dataset/"
                        "posti-letto-stabilimento-ospedaliero-e-disciplina-2023/"
                    ),
                    note=acquired.note,
                )
            except Exception as exc:
                hospitals = _record(
                    "salute_hospital_establishments", "acquisition_failed", str(hospital_year),
                    note=f"{type(exc).__name__}: {exc}",
                )
    return pharmacies, hospitals


def inspect_sources(
    ctx: MunicipalityContext,
    *,
    census_year: str,
    school_year: str,
    building_year: str,
    pharmacy_reference_date: str,
    hospital_year: str,
    fetch_supported: bool = False,
    allow_contemporary_pharmacy_source: bool = False,
) -> AcquisitionReport:
    records = (
        inspect_istat_census(ctx.region_code, census_year, fetch_supported=fetch_supported),
        inspect_istat_boundaries(ctx.region_code, fetch_supported=fetch_supported),
        inspect_mim_registry("SCUANAGRAFESTAT", school_year, fetch_supported=fetch_supported),
        inspect_mim_registry("SCUANAGRAFEPAR", school_year, fetch_supported=fetch_supported),
        inspect_mim_buildings(building_year, fetch_supported=fetch_supported),
        *inspect_health(
            pharmacy_reference_date, hospital_year,
            fetch_supported=fetch_supported,
            allow_contemporary_pharmacy_source=allow_contemporary_pharmacy_source,
        ),
    )
    return AcquisitionReport(
        municipality_code=ctx.code,
        municipality_name=ctx.name,
        census_year=str(census_year),
        school_year=str(school_year),
        building_year=str(building_year),
        hospital_year=str(hospital_year),
        pharmacy_reference_date=str(pharmacy_reference_date),
        fetch_supported=fetch_supported,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        sources=records,
        allow_contemporary_pharmacy_source=allow_contemporary_pharmacy_source,
    )


def write_report(report: AcquisitionReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B5B2 Bronze source preflight + verified opt-in downloads (read-only by default)."
    )
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--city", help="Exact ISTAT municipality name.")
    location.add_argument("--municipality-code", help="Six-digit ISTAT code.")
    parser.add_argument("--province-code", default=None)
    parser.add_argument("--region-code", default=None)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--building-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--hospital-year", default="2023")
    parser.add_argument(
        "--fetch-supported", action="store_true",
        help="Acquire missing Bronze files from pinned verified official distributions; never refresh existing sources.",
    )
    parser.add_argument(
        "--allow-contemporary-pharmacy-source", action="store_true",
        help=("Explicitly permit downloading a present-day pharmacy catalogue file "
              "for an older analysis reference date; historical completeness is not guaranteed."),
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 2 if any source is missing/ambiguous/invalid/unsupported.",
    )
    parser.add_argument("--output-path", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.allow_contemporary_pharmacy_source and not args.fetch_supported:
        raise ValueError("--allow-contemporary-pharmacy-source requires --fetch-supported.")
    if args.city is not None:
        ctx = MunicipalityContext.resolve_name(
            args.city, census_year=args.census_year,
            province_code=args.province_code,
            region_code=args.region_code,
        )
    else:
        if args.province_code or args.region_code:
            raise ValueError("--province-code / --region-code require --city.")
        ctx = MunicipalityContext.resolve(args.municipality_code, census_year=args.census_year)

    report = inspect_sources(
        ctx,
        census_year=args.census_year,
        school_year=args.school_year,
        building_year=args.building_year,
        pharmacy_reference_date=args.health_reference_date,
        hospital_year=args.hospital_year,
        fetch_supported=args.fetch_supported,
        allow_contemporary_pharmacy_source=args.allow_contemporary_pharmacy_source,
    )

    label = args.health_reference_date.replace("-", "")
    path = (
        Path(args.output_path)
        if args.output_path else
        FEATURES_ACQUISITION / ctx.code /
        f"sources_{args.census_year}_{args.school_year}_{args.building_year}_{args.hospital_year}_{label}.json"
    )
    write_report(report, path)

    print("\n=== B5B2 SOURCE PREFLIGHT ===")
    print(f"Comune: {ctx.name} ({ctx.code})")
    print(f"Fetch supported: {args.fetch_supported}")
    for source in report.sources:
        print(f"{source.source_id:32s} {source.state}")
        if source.note:
            print(f"  {source.note}")
    print(f"READY: {report.ready}")
    print(f"Manifest: {path}")
    return 0 if report.ready or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
