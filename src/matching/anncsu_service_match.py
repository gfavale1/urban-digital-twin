"""B6A2 ANNCSU evidence-only exact-address matching for municipal services.

One row per source record, never modifies MIM/Salute files, v2 runs, or routing.
'Exact' means exact *normalized* street, civic and exponent, NOT verified entrance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from core.municipality import MunicipalityContext
from core.paths import ROOT
from ingestion.anncsu import municipality_code, normalize_exponent, normalize_house_number, normalize_street, sha256_file

ANNCSU_REQUIRED = {
    "municipality_code", "street_normalized", "house_number_normalized",
    "exponent_normalized", "longitude", "latitude", "coordinate_status",
    "method_code", "source_snapshot_date", "street_id", "access_id", "source_row_number",
}

# Keep administrative records distinct: schools != physical school buildings.
INPUTS = (
    ("school_building", "mim", "physical_school_buildings_{building_year}_geocoded.parquet", "building_code", "official_building_address"),
    ("school_registry_record", "mim", "schools_registry_{school_year}.parquet", "school_code", "school_address"),
    ("pharmacy", "salute", "pharmacy_sites_{pharmacy_label}.parquet", "service_site_id", "address"),
    ("hospital", "salute", "hospital_sites_{hospital_year}.parquet", "service_site_id", "address"),
)

OUTPUT_FIELDS = (
    "entity_type", "entity_id", "source_address", "source_file", "municipality_code",
    "reference_vintage", "existing_latitude", "existing_longitude", "existing_status",
    "existing_usable_for_accessibility", "parsed_street", "parsed_house_number", "parsed_exponent",
    "match_status", "candidate_count", "candidate_access_id", "candidate_street_id",
    "anncsu_latitude", "anncsu_longitude", "anncsu_method_code", "anncsu_snapshot_date",
    "distance_from_existing_m", "coordinate_accepted_for_accessibility",
)

# Intentional fail-closed parsing: no civic => no coordinate. Street aliases, spelling
# correction, fuzzy matching and non-standard civic ranges are NOT auto-resolved.
CIVIC = re.compile(
    r"^(?P<street>.+?)(?:\s*,\s*|\s+)"
    r"(?:N\.?\s*|NUMERO\s+)?"
    r"(?P<number>[0-9]{1,4})"
    r"(?:(?:\s*/\s*|\s*)(?P<exponent>[A-Z]{1,5}))?$",
    re.IGNORECASE,
)


def parse_service_address(value: Any) -> tuple[str, str, str, str]:
    """Return normalized address parts and parse status, without inventing civics."""
    if value is None or pd.isna(value):
        return "", "", "", "missing_address"
    raw = str(value).strip()
    if not raw or raw.upper() in {"NAN", "NULL", "NONE", "N.D.", "-"}:
        return "", "", "", "missing_address"
    # Forbid compound/range civics: cannot safely select one address position.
    if re.search(r"\b\d{1,4}\s*[-/]\s*\d{1,4}\b", raw):
        return "", "", "", "unparseable_address"
    match = CIVIC.fullmatch(raw.strip().upper().replace("’", "'"))
    if not match:
        return "", "", "", "missing_or_unparseable_civic"
    number = normalize_house_number(match.group("number"))
    exponent = normalize_exponent(match.group("exponent"))
    street = normalize_street(match.group("street"))
    # Guard against interpreting a year in a street name as a house number.
    if number and len(number) == 4 and 1800 <= int(number) <= 2100:
        return "", "", "", "missing_or_unparseable_civic"
    if not street or not number:
        return "", "", "", "missing_or_unparseable_civic"
    return street, number, exponent, "parsed"


def _valid_coordinate(lat: Any, lon: Any) -> tuple[float, float] | None:
    try:
        latitude, longitude = float(lat), float(lon)
    except (ValueError, TypeError):
        return None
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        return None
    if not (35 <= latitude <= 48 and 6 <= longitude <= 19):
        return None
    return latitude, longitude


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    k = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371008.8 * 2 * math.asin(min(1.0, math.sqrt(k)))


def read_anncsu_index(path: Path, municipality: str, snapshot: str) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    """Read independently verified B6A1 CSV, preserving every access record."""
    index: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not ANNCSU_REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f"ANNCSU index missing fields: {sorted(ANNCSU_REQUIRED - set(reader.fieldnames or []))}")
        seen_access = set()
        count = 0
        for item in reader:
            count += 1
            if item["municipality_code"] != municipality or item["source_snapshot_date"] != snapshot:
                raise ValueError("ANNCSU index contains a wrong municipality/snapshot")
            access_key = (item["municipality_code"], item["access_id"])
            if item["access_id"] and access_key in seen_access:
                raise ValueError("Duplicate ANNCSU access_id")
            seen_access.add(access_key)
            if item["street_normalized"] and item["house_number_normalized"]:
                key = (item["street_normalized"], item["house_number_normalized"], item["exponent_normalized"])
                index[key].append(item)
        if not count:
            raise ValueError("Empty ANNCSU index")
    return index


def _existing_status(record: dict, kind: str) -> tuple[str, str]:
    if kind == "school_building":
        return str(record.get("geocoding_status") or ""), ""
    if kind == "school_registry_record":
        return "not_geocoded_in_registry", ""
    return str(record.get("coordinate_resolution") or ""), str(record.get("usable_for_accessibility") or "")


def match_record(
    record: dict[str, Any], *, kind: str, entity_id: str, address_col: str,
    source_file: str, municipality: str, municipality_name: str, vintage: str,
    snapshot: str, index: dict[tuple[str, str, str], list[dict[str, str]]],
) -> dict[str, Any]:
    if not entity_id or entity_id.lower() in {"nan", "none", "null"}:
        raise ValueError(f"Blank identity in {kind}")
    addr = record.get(address_col)
    if addr is not None and pd.isna(addr):
        addr = None
    lat, lon = record.get("latitude"), record.get("longitude")
    old_coords = _valid_coordinate(lat, lon)
    existing_status, existing_usable = _existing_status(record, kind)
    out: dict[str, Any] = {
        "entity_type": kind, "entity_id": str(entity_id), "source_address": str(addr or ""),
        "source_file": source_file, "municipality_code": municipality,
        "reference_vintage": vintage, "existing_latitude": lat if old_coords else "",
        "existing_longitude": lon if old_coords else "", "existing_status": existing_status,
        "existing_usable_for_accessibility": existing_usable,
        "parsed_street": "", "parsed_house_number": "", "parsed_exponent": "",
        "match_status": "", "candidate_count": 0, "candidate_access_id": "",
        "candidate_street_id": "", "anncsu_latitude": "", "anncsu_longitude": "",
        "anncsu_method_code": "", "anncsu_snapshot_date": snapshot,
        "distance_from_existing_m": "", "coordinate_accepted_for_accessibility": False,
    }
    # Building registers can include linked facilities located outside target municipality.
    if kind == "school_building" and normalize_street(record.get("building_municipality_name")) != normalize_street(municipality_name):
        out["match_status"] = "outside_target_municipality"
        return out
    if kind != "school_building" and kind != "school_registry_record":
        if str(record.get("municipality_code") or "").zfill(6) != municipality:
            raise ValueError(f"Unexpected municipality code in {kind}: {record.get('municipality_code')}")
    if kind == "school_registry_record" and str(record.get("municipality_code") or "").zfill(6) != municipality:
        raise ValueError("School registry contains another municipality")
    street, house, exponent, parse_status = parse_service_address(addr)
    out.update(parsed_street=street, parsed_house_number=house, parsed_exponent=exponent)
    if parse_status != "parsed":
        out["match_status"] = parse_status
        return out
    candidates = index.get((street, house, exponent), [])
    out["candidate_count"] = len(candidates)
    if not candidates:
        out["match_status"] = "no_exact_address_match"
        return out
    points = {tuple(map(float, (p["latitude"], p["longitude"]))) for p in candidates
              if p["coordinate_status"] == "valid" and _valid_coordinate(p["latitude"], p["longitude"])}
    if len(candidates) != 1:
        # Distinct municipal accesses sharing a civic are not safe entrance-level matches,
        # even if B6A1 saw identical coordinates.
        out["match_status"] = "ambiguous_multiple_positions" if len(points) > 1 else "multiple_accesses_review"
        return out
    candidate = candidates[0]
    if candidate["coordinate_status"] != "valid" or not _valid_coordinate(candidate["latitude"], candidate["longitude"]):
        out["match_status"] = "matched_without_valid_coordinate"
        return out
    coords = _valid_coordinate(candidate["latitude"], candidate["longitude"])
    assert coords is not None
    out.update(
        match_status="exact_single_access_candidate", candidate_access_id=candidate["access_id"],
        candidate_street_id=candidate["street_id"], anncsu_latitude=candidate["latitude"],
        anncsu_longitude=candidate["longitude"], anncsu_method_code=candidate["method_code"],
    )
    if old_coords is not None:
        out["distance_from_existing_m"] = round(_haversine_m(old_coords, coords), 2)
    return out


def _read_inputs(data_root: Path, municipality: str, school_year: str, building_year: str,
                 pharmacy_date: str, hospital_year: str):
    values = dict(school_year=school_year, building_year=building_year,
                  pharmacy_label=pharmacy_date.replace("-", ""), hospital_year=hospital_year)
    paths = []
    for kind, parent, template, id_col, address_col in INPUTS:
        path = data_root / "processed" / parent / municipality / template.format(**values)
        if not path.is_file():
            raise FileNotFoundError(f"Required {kind} input not found: {path}")
        df = pd.read_parquet(path)
        required = {id_col, address_col}
        if kind == "school_building":
            required.add("building_municipality_name")
        else:
            required.add("municipality_code")
        if not required.issubset(df.columns):
            raise ValueError(f"Missing {kind} columns: {sorted(required - set(df.columns))}")
        if df[id_col].isna().any() or df[id_col].astype(str).duplicated().any():
            raise ValueError(f"Missing/duplicate {kind} identifiers: {id_col}")
        paths.append((kind, path, id_col, address_col, df))
    return paths


def _verified_anncsu(data_root: Path, municipality: str, snapshot: str) -> tuple[Path, dict[str, Any]]:
    address_csv = data_root / "processed" / "anncsu" / municipality / f"addresses_{snapshot}.csv"
    manifest_file = data_root / "features" / "anncsu" / municipality / f"manifest_{snapshot}.json"
    if not address_csv.is_file() or not manifest_file.is_file():
        raise FileNotFoundError("ANNCSU B6A1 index or manifest missing")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("municipality_code") != municipality or manifest.get("snapshot_date") != snapshot:
        raise ValueError("ANNCSU B6A1 manifest mismatch")
    if manifest.get("output_csv_sha256") != sha256_file(address_csv):
        raise ValueError("ANNCSU B6A1 CSV SHA-256 mismatch")
    return address_csv, manifest


def build_evidence(data_root: Path, municipality: str, *, snapshot: str,
                   school_year: str = "202425", building_year: str = "202425",
                   pharmacy_date: str = "2025-06-30", hospital_year: str = "2023") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    code = municipality_code(municipality)
    if not re.fullmatch(r"\d{8}", snapshot):
        raise ValueError("snapshot must be YYYYMMDD")
    datetime.strptime(snapshot, "%Y%m%d")
    csv_path, anncsu_manifest = _verified_anncsu(data_root, code, snapshot)
    index = read_anncsu_index(csv_path, code, snapshot)
    inputs = _read_inputs(data_root, code, school_year, building_year, pharmacy_date, hospital_year)
    school_df = next(df for kind, _, _, _, df in inputs if kind == "school_registry_record")
    municipality_names = {normalize_street(name) for name in school_df["municipality_name"].dropna().astype(str)} if "municipality_name" in school_df.columns else set()
    if len(municipality_names) != 1 or not next(iter(municipality_names)):
        raise ValueError("Cannot identify a unique target municipality name from MIM school registry")
    municipality_name = next(iter(municipality_names))
    vintages = dict(school_building=building_year, school_registry_record=school_year,
                    pharmacy=pharmacy_date, hospital=hospital_year)
    output: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for kind, path, id_col, address_col, df in inputs:
        input_hashes[kind] = sha256_file(path)
        for record in df.to_dict(orient="records"):
            output.append(match_record(
                record, kind=kind, entity_id=str(record[id_col]), address_col=address_col,
                source_file=path.relative_to(data_root).as_posix(), municipality=code,
                municipality_name=municipality_name, vintage=vintages[kind], snapshot=snapshot, index=index,
            ))
    statuses = Counter(row["match_status"] for row in output)
    by_kind = {kind: dict(sorted(Counter(r["match_status"] for r in output if r["entity_type"] == kind).items()))
               for kind, *_ in INPUTS}
    candidate_rows = [r for r in output if r["match_status"] == "exact_single_access_candidate"]
    missing_existing = sum(
        r["match_status"] == "exact_single_access_candidate" and r["existing_latitude"] == ""
        for r in output
    )
    report = {
        "municipality_code": code, "anncsu_snapshot_date": snapshot,
        "anncsu_csv_sha256": anncsu_manifest["output_csv_sha256"],
        "anncsu_archive_sha256": anncsu_manifest.get("archive_sha256"),
        "input_sha256": dict(sorted(input_hashes.items())), "source_row_count": len(output),
        "status_counts": dict(sorted(statuses.items())), "status_by_type": by_kind,
        "candidate_exact_unique_count": len(candidate_rows),
        "candidate_without_existing_coordinates": missing_existing,
        "reference_vintages": dict(sorted(vintages.items())),
        "notes": [
            "Exact matches are candidate evidence, not surveyed physical entrances.",
            "Neither service geometry nor v2/legacy accessibility was modified.",
            "No street alias/fuzzy matching; uncertain civic/duplicate access remains review/unresolved.",
            "No municipal polygon containment or street-access verification in B6A2.",
            "ANNCSU 2026 is temporally distinct from older school/health reference vintages.",
            "METODO is kept as a code without an inferred numeric quality ranking.",
            "School registry records are administrative and may share a physical building.",
            "A candidate is not automatically considered usable_for_accessibility.",
        ],
    }
    return output, report


def save_evidence(data_root: Path, municipality: str, snapshot: str,
                  rows: list[dict[str, Any]], report: dict[str, Any]) -> tuple[Path, Path, bool]:
    target = data_root / "features" / "anncsu" / municipality
    path_csv = target / f"service_address_matches_{snapshot}.csv"
    path_json = target / f"service_address_matches_{snapshot}.json"
    # Idempotent under identical inputs, fail closed for stale/preexisting results.
    if path_csv.exists() or path_json.exists():
        if not path_csv.is_file() or not path_json.is_file():
            raise FileExistsError("Incomplete B6A2 outputs exist")
        old = json.loads(path_json.read_text(encoding="utf-8"))
        if (old.get("input_sha256") != report.get("input_sha256") or
            old.get("anncsu_csv_sha256") != report.get("anncsu_csv_sha256") or
            old.get("reference_vintages") != report.get("reference_vintages") or
            old.get("status_counts") != report.get("status_counts") or
            old.get("status_by_type") != report.get("status_by_type") or
            old.get("source_row_count") != report.get("source_row_count") or
            old.get("output_csv_sha256") != sha256_file(path_csv)):
            raise FileExistsError("Existing B6A2 results disagree with inputs/content; refusing overwrite")
        return path_csv, path_json, True
    target.mkdir(parents=True, exist_ok=True)
    tmp_csv = path_csv.with_suffix(".csv.tmp")
    tmp_json = path_json.with_suffix(".json.tmp")
    if tmp_csv.exists() or tmp_json.exists():
        raise FileExistsError("Partial B6A2 temporary output exists")
    try:
        with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        complete = dict(report, output_csv=path_csv.as_posix(),
                        output_csv_sha256=sha256_file(tmp_csv),
                        generated_at_utc=datetime.now(timezone.utc).isoformat())
        tmp_json.write_text(json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_csv.replace(path_csv)
        tmp_json.replace(path_json)
    except BaseException:
        tmp_csv.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
        raise
    return path_csv, path_json, False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B6A2 exact ANNCSU -> MIM/Salute evidence, no coordinate mutation")
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--municipality-code")
    location.add_argument("--city")
    parser.add_argument("--province-code")
    parser.add_argument("--region-code")
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--snapshot-date", default="20260915")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--building-year", default="202425")
    parser.add_argument("--pharmacy-reference-date", default="2025-06-30")
    parser.add_argument("--hospital-year", default="2023")
    args = parser.parse_args(argv)
    if not args.city and (args.province_code or args.region_code):
        parser.error("--province-code and --region-code require --city")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.city:
        ctx = MunicipalityContext.resolve_name(city_name=args.city, census_year=args.census_year,
                                               province_code=args.province_code, region_code=args.region_code)
        code = ctx.code
    else:
        code = municipality_code(args.municipality_code)
    root = ROOT / "data"
    rows, report = build_evidence(root, code, snapshot=args.snapshot_date,
        school_year=args.school_year, building_year=args.building_year,
        pharmacy_date=args.pharmacy_reference_date, hospital_year=args.hospital_year)
    csv_path, report_path, cached = save_evidence(root, code, args.snapshot_date, rows, report)
    print("\n=== B6A2 ANNCSU SERVICE MATCH (EVIDENCE-ONLY) ===")
    print(f"Municipality: {code} | ANNCSU: {args.snapshot_date} | Cached: {cached}")
    print(f"Source records: {len(rows)}")
    for kind, statuses in report["status_by_type"].items():
        print(f"  {kind}: {sum(statuses.values())} {statuses}")
    print(f"Exact single-access candidates: {report['candidate_exact_unique_count']}")
    print(f"Candidates without existing coordinates: {report['candidate_without_existing_coordinates']}")
    print(f"Evidence: {csv_path}")
    print(f"Report: {report_path}")
    print("No legacy, v2, or service-position outputs modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
