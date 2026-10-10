"""B6A1: loss-aware, municipality-scoped import of an official ANNCSU ZIP.

This stage produces an auxiliary 2026 address evidence layer. It deliberately
DOES NOT modify MIM/Salute service positions, legacy outputs, or v2 routing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from core.municipality import MunicipalityContext
from core.paths import ROOT

REQUIRED_COLUMNS = (
    "CODICE_COMUNE", "CODICE_ISTAT", "PROGRESSIVO_NAZIONALE", "ODONIMO",
    "PROGRESSIVO_ACCESSO", "CIVICO", "ESPONENTE", "COORD_X_COMUNE",
    "COORD_Y_COMUNE", "METODO",
)
OUTPUT_COLUMNS = (
    "municipality_code", "belfiore_code", "street_id", "access_id",
    "street_raw", "street_normalized", "locality_raw", "house_number_raw",
    "house_number_normalized", "exponent_raw", "exponent_normalized",
    "longitude", "latitude", "coordinate_status", "method_code",
    "source_snapshot_date", "source_row_number",
)
SNAPSHOT_MEMBER = re.compile(r"^INDIR_[A-Z0-9]+_(\d{8})\.csv$", re.IGNORECASE)
HOUSE_NUMBER = re.compile(r"^0*([0-9]+)$")
MISSING = {"", "-", "SNC", "S.N.C.", "S N C", "NULL", "NAN"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def municipality_code(value: str) -> str:
    value = str(value).strip()
    if not value.isascii() or not value.isdigit() or len(value) != 6:
        raise ValueError("ANNCSU municipality code must contain exactly six ASCII digits")
    return value


def normalize_street(value: Any) -> str:
    """Conservative comparison key; do not guess toponym equivalences."""
    if value is None:
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()
    return re.sub(r"\s+", " ", value)


def normalize_house_number(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in MISSING:
        return ""
    match = HOUSE_NUMBER.fullmatch(text)
    return str(int(match.group(1))) if match else ""


def normalize_exponent(value: Any) -> str:
    text = normalize_street(value)
    if text in MISSING:
        return ""
    # Do not normalize unusual symbols into guessed house-number suffixes.
    return text if re.fullmatch(r"[A-Z0-9]{1,5}", text) else ""


def parse_longitude_latitude(longitude: str, latitude: str) -> tuple[str, str, str]:
    lon_raw, lat_raw = str(longitude or "").strip(), str(latitude or "").strip()
    if not lon_raw or not lat_raw:
        return "", "", "missing"
    try:
        lon = float(lon_raw.replace(",", "."))
        lat = float(lat_raw.replace(",", "."))
    except ValueError:
        return "", "", "invalid"
    # Italy-wide plausibility only; exact municipal membership is a later QA gate.
    if not (math.isfinite(lon) and math.isfinite(lat) and 6.0 <= lon <= 19.0 and 35.0 <= lat <= 48.0):
        return "", "", "invalid"
    return format(lon, ".10g"), format(lat, ".10g"), "valid"


@dataclass(frozen=True)
class AnncsuReport:
    municipality_code: str
    snapshot_date: str
    member_name: str
    archive_path: str
    archive_sha256: str
    regional_rows_scanned: int
    municipal_rows_selected: int
    municipal_unique_streets: int
    municipal_unique_street_civico_keys: int
    ambiguous_coordinate_keys: int
    coordinate_status_counts: dict[str, int]
    method_code_counts: dict[str, int]
    output_csv: str
    output_csv_sha256: str
    created_at_utc: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["notes"] = list(result["notes"])
        return result


def archive_metadata(path: Path) -> tuple[str, str]:
    """Require an unambiguous, root-level, dated CSV member and valid header."""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"ANNCSU archive missing/empty: {path}")
    try:
        with ZipFile(path) as archive:
            entries = [i for i in archive.infolist() if not i.is_dir()]
            if len(entries) != 1:
                raise ValueError("ANNCSU ZIP must contain exactly one file")
            item = entries[0]
            if Path(item.filename).name != item.filename:
                raise ValueError("Nested/path-traversing ANNCSU ZIP member rejected")
            match = SNAPSHOT_MEMBER.fullmatch(item.filename)
            if not match:
                raise ValueError(f"Unrecognized dated ANNCSU CSV member: {item.filename}")
            snapshot = match.group(1)
            datetime.strptime(snapshot, "%Y%m%d")
            filename_date = re.search(r"(\d{8})\.zip$", path.name, re.IGNORECASE)
            if filename_date and filename_date.group(1) != snapshot:
                raise ValueError("Archive filename date differs from member snapshot date")
            with archive.open(item) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", errors="strict", newline="") as text:
                    reader = csv.reader(text, delimiter=";")
                    header = next(reader, [])
            if not set(REQUIRED_COLUMNS).issubset(header):
                raise ValueError(f"ANNCSU columns missing: {sorted(set(REQUIRED_COLUMNS) - set(header))}")
            if len(set(header)) != len(header):
                raise ValueError("Duplicate ANNCSU header columns")
            return item.filename, snapshot
    except BadZipFile as exc:
        raise ValueError("Invalid ANNCSU ZIP") from exc


def iter_municipality_rows(archive_path: Path, member: str, code: str):
    with ZipFile(archive_path) as archive:
        with archive.open(member) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8-sig", errors="strict", newline="") as text:
                reader = csv.DictReader(text, delimiter=";", restkey="__extra__")
                if not set(REQUIRED_COLUMNS).issubset(reader.fieldnames or []):
                    raise ValueError("Invalid ANNCSU schema")
                for row_number, row in enumerate(reader, start=2):
                    if None in row or "__extra__" in row:
                        raise ValueError(f"Malformed ANNCSU CSV row {row_number}")
                    selected = (row.get("CODICE_ISTAT") or "").strip() == code
                    yield row_number, row if selected else None


def convert_row(row_number: int, row: dict[str, str], snapshot: str) -> dict[str, str | int]:
    lon, lat, coordinate_status = parse_longitude_latitude(
        row.get("COORD_X_COMUNE", ""), row.get("COORD_Y_COMUNE", "")
    )
    return {
        "municipality_code": row["CODICE_ISTAT"].strip(),
        "belfiore_code": (row.get("CODICE_COMUNE") or "").strip(),
        "street_id": (row.get("PROGRESSIVO_NAZIONALE") or "").strip(),
        "access_id": (row.get("PROGRESSIVO_ACCESSO") or "").strip(),
        "street_raw": (row.get("ODONIMO") or "").strip(),
        "street_normalized": normalize_street(row.get("ODONIMO")),
        "locality_raw": (row.get("LOCALITA'") or "").strip(),
        "house_number_raw": (row.get("CIVICO") or "").strip(),
        "house_number_normalized": normalize_house_number(row.get("CIVICO")),
        "exponent_raw": (row.get("ESPONENTE") or "").strip(),
        "exponent_normalized": normalize_exponent(row.get("ESPONENTE")),
        "longitude": lon,
        "latitude": lat,
        "coordinate_status": coordinate_status,
        "method_code": (row.get("METODO") or "").strip(),
        "source_snapshot_date": snapshot,
        "source_row_number": row_number,
    }


def _read_existing(report_path: Path, csv_path: Path, archive_hash: str) -> dict[str, Any] | None:
    if not report_path.exists() and not csv_path.exists():
        return None
    if not report_path.is_file() or not csv_path.is_file():
        raise FileExistsError("Partial ANNCSU output exists; inspect it before retrying")
    stored = json.loads(report_path.read_text(encoding="utf-8"))
    if (stored.get("archive_sha256") != archive_hash or
            stored.get("output_csv_sha256") != sha256_file(csv_path) or
            stored.get("output_csv") != csv_path.as_posix()):
        raise FileExistsError("ANNCSU outputs exist with different source/content; refusing overwrite")
    return stored


def import_municipality(
    archive_path: Path, code: str, *, output_root: Path | None = None,
    expected_sha256: str | None = None,
) -> tuple[Path, Path, dict[str, Any], bool]:
    code = municipality_code(code)
    archive_path = Path(archive_path).resolve()
    member, snapshot = archive_metadata(archive_path)
    source_hash = sha256_file(archive_path)
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Expected archive SHA-256 must be 64 hex characters")
        if source_hash != expected:
            raise ValueError("ANNCSU archive SHA-256 does not match the pinned expected checksum")
    base = Path(output_root) if output_root is not None else ROOT / "data"
    csv_path = base / "processed" / "anncsu" / code / f"addresses_{snapshot}.csv"
    report_path = base / "features" / "anncsu" / code / f"manifest_{snapshot}.json"
    existing = _read_existing(report_path, csv_path, source_hash)
    if existing is not None:
        return csv_path, report_path, existing, True

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = csv_path.with_suffix(".csv.tmp")
    temp_report = report_path.with_suffix(".json.tmp")
    if temp_csv.exists():
        raise FileExistsError(f"Temporary output exists, inspect before retrying: {temp_csv}")

    scanned, selected = 0, 0
    streets = set()
    keys = set()
    coordinate_status = Counter()
    methods = Counter()
    key_coordinates: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    try:
        with temp_csv.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row_number, row in iter_municipality_rows(archive_path, member, code):
                scanned += 1
                if row is None:
                    continue
                entry = convert_row(row_number, row, snapshot)
                selected += 1
                writer.writerow(entry)
                coordinate_status[entry["coordinate_status"]] += 1
                methods[entry["method_code"] or "<missing>"] += 1
                if entry["street_normalized"]:
                    streets.add(entry["street_normalized"])
                if entry["street_normalized"] and entry["house_number_normalized"]:
                    key = (entry["street_normalized"], entry["house_number_normalized"], entry["exponent_normalized"])
                    keys.add(key)
                    if entry["coordinate_status"] == "valid":
                        key_coordinates[key].add((entry["longitude"], entry["latitude"]))
        if selected == 0:
            raise ValueError(f"No ANNCSU records found for municipality {code}; check the regional archive")
        output_hash = sha256_file(temp_csv)
        manifest = AnncsuReport(
            municipality_code=code,
            snapshot_date=snapshot,
            member_name=member,
            archive_path=archive_path.as_posix(),
            archive_sha256=source_hash,
            regional_rows_scanned=scanned,
            municipal_rows_selected=selected,
            municipal_unique_streets=len(streets),
            municipal_unique_street_civico_keys=len(keys),
            ambiguous_coordinate_keys=sum(len(coords) > 1 for coords in key_coordinates.values()),
            coordinate_status_counts=dict(sorted(coordinate_status.items())),
            method_code_counts=dict(sorted(methods.items())),
            output_csv=csv_path.as_posix(),
            output_csv_sha256=output_hash,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            notes=(
                "Addresses and coordinates are snapshot evidence, not verified building entrances.",
                "METODO values are preserved as source codes; no unsupported quality ordering is assigned.",
                "No municipality geometry check or spatial snapping is performed in B6A1.",
                "2026 ANNCSU is temporally distinct from 2025 service records; no legacy or v2 baseline coordinates were changed.",
                "Duplicate street/civico keys with distinct coordinates are ambiguous and must not be resolved arbitrarily.",
            ),
        ).to_dict()
        if temp_report.exists():
            raise FileExistsError(f"Temporary report exists: {temp_report}")
        temp_report.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        temp_csv.replace(csv_path)
        temp_report.replace(report_path)
        return csv_path, report_path, manifest, False
    except BaseException:
        temp_csv.unlink(missing_ok=True)
        temp_report.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B6A1 ANNCSU ZIP importer, municipal filter, QA and provenance")
    parser.add_argument("--archive", type=Path, required=True, help="Official regional ANNCSU ZIP (no automatic fetch yet)")
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--municipality-code")
    location.add_argument("--city")
    parser.add_argument("--province-code", default=None)
    parser.add_argument("--region-code", default=None)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--expected-sha256", help="Pin the exact official archive bytes")
    args = parser.parse_args(argv)
    if not args.city and (args.province_code or args.region_code):
        parser.error("--province-code and --region-code require --city")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.city is not None:
        ctx = MunicipalityContext.resolve_name(
            city_name=args.city, census_year=args.census_year,
            province_code=args.province_code, region_code=args.region_code,
        )
        code = ctx.code
    else:
        code = municipality_code(args.municipality_code)
    csv_path, manifest_path, report, cached = import_municipality(
        args.archive, code, expected_sha256=args.expected_sha256
    )
    print("\n=== B6A1 ANNCSU MUNICIPAL IMPORT ===")
    print(f"Municipality: {code} | Snapshot: {report['snapshot_date']} | Cached: {cached}")
    print(f"Regional rows scanned: {report['regional_rows_scanned']}")
    print(f"Municipal addresses: {report['municipal_rows_selected']}")
    print(f"Coordinate status: {report['coordinate_status_counts']}")
    print(f"Ambiguous street+civico coordinate keys: {report['ambiguous_coordinate_keys']}")
    print(f"Index: {csv_path}")
    print(f"Manifest: {manifest_path}")
    print("No service positions or routing outputs modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
