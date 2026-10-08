from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from core.analysis_spec import ServiceType
from core.schema_v2 import (
    GeocodeQuality,
    OperationalStatus,
    SERVICE_V2,
)


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
PROCESSED_SERVICES_DIR = ROOT / "data" / "processed" / "services"
FEATURES_SERVICES_DIR = ROOT / "data" / "features" / "services"

SCHEMA_VERSION = "2.0.0"

EDUCATION_SERVICE_TYPES = {
    ServiceType.PRESCHOOL.value,
    ServiceType.PRIMARY_SCHOOL.value,
    ServiceType.LOWER_SECONDARY_SCHOOL.value,
    ServiceType.UPPER_SECONDARY_SCHOOL.value,
}


# ---------------------------------------------------------------------------
# CLI / generic helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Builds the parallel ServiceV2 layer from the frozen legacy "
            "canonical service layer. The legacy artifact is not modified."
        )
    )
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    args = parser.parse_args()

    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    if len(args.municipality_code) != 6 or not args.municipality_code.isdigit():
        raise ValueError("--municipality-code must contain exactly 6 digits.")

    args.health_reference_date = pd.Timestamp(
        args.health_reference_date
    ).normalize()
    return args


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return value


def clean_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> str | None:
    value = clean_text(value)
    if value is None:
        return None
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.upper().replace("°", " ").replace("º", " ")
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = clean_text(value)
    if text is None:
        return {}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                values = parsed
            else:
                values = [value]
        else:
            values = [value]

    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        item = clean_text(item)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def json_list(values: Iterable[Any]) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = clean_text(value)
        if value and value not in seen:
            seen.add(value)
            cleaned.append(value)
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))


def one_or_none(values: Iterable[Any]) -> str | None:
    cleaned = sorted({x for x in (clean_text(v) for v in values) if x})
    return cleaned[0] if len(cleaned) == 1 else None


def parse_iso_date(value: Any) -> str | None:
    value = clean_text(value)
    if value is None:
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def service_v2_id(legacy_site_id: str, service_type: str) -> str:
    return f"SVC2::{legacy_site_id}::{service_type}"


# ---------------------------------------------------------------------------
# Education classification
# ---------------------------------------------------------------------------

def classify_school_grade(value: Any) -> tuple[str | None, str]:
    """Classify an official MIM grade description conservatively.

    Returns (ServiceType.value | None, method). Unknown/organisational labels are
    deliberately left unresolved instead of being guessed from the school name.
    """

    text = normalize_text(value)
    if text is None:
        return None, "missing_grade_description"

    # Order matters: the secondary levels must be distinguished before generic
    # education tokens are considered.
    if any(token in text for token in ("INFANZIA", "MATERNA")):
        return ServiceType.PRESCHOOL.value, "mim_grade_description"

    if any(token in text for token in ("PRIMARIA", "ELEMENTARE")):
        return ServiceType.PRIMARY_SCHOOL.value, "mim_grade_description"

    lower_patterns = (
        # MIM 2024/25 commonly uses the compact official label
        # "SCUOLA PRIMO GRADO" in addition to secondary-school wording.
        r"\bSCUOLA PRIMO GRADO\b",
        r"\bSECONDARIA (DI )?(I|1|PRIMO) GRADO\b",
        r"\bSEC(ONDARIA)? (DI )?(I|1|PRIMO) GRADO\b",
        r"\bSCUOLA MEDIA\b",
        r"\bMEDIA INFERIORE\b",
    )
    if any(re.search(pattern, text) for pattern in lower_patterns):
        return ServiceType.LOWER_SECONDARY_SCHOOL.value, "mim_grade_description"

    upper_patterns = (
        # Symmetric compact MIM label, when present.
        r"\bSCUOLA SECONDO GRADO\b",
        r"\bSECONDARIA (DI )?(II|2|SECONDO) GRADO\b",
        r"\bSEC(ONDARIA)? (DI )?(II|2|SECONDO) GRADO\b",
        r"\bMEDIA SUPERIORE\b",
        r"\bISTRUZIONE SECONDARIA SUPERIORE\b",
    )
    if any(re.search(pattern, text) for pattern in upper_patterns):
        return ServiceType.UPPER_SECONDARY_SCHOOL.value, "mim_grade_description"

    # Common official upper-secondary grade/type descriptions. We only use
    # explicit pedagogical types, not generic organisational labels such as
    # "ISTITUTO SUPERIORE" or "ISTITUTO COMPRENSIVO".
    upper_tokens = (
        "LICEO",
        "ISTITUTO TECNICO",
        "IST TECNICO",
        # Official abbreviated MIM labels such as
        # "IST TEC COMMERCIALE E PER GEOMETRI".
        "IST TEC ",
        "TECNICO INDUSTRIALE",
        "TECNICO COMMERCIALE",
        "TECNICO ECONOMICO",
        "TECNICO TECNOLOGICO",
        "ISTITUTO PROFESSIONALE",
        "IST PROFESSIONALE",
        # Historical/abbreviated MIM upper-secondary labels used in the
        # 2024/25 registry (e.g. "IST PROF INDUSTRIA E ARTIGIANATO").
        "IST PROF ",
        "ISTITUTO D ARTE",
        "IST D ARTE",
        "ISTITUTO MAGISTRALE",
        "IST MAGISTRALE",
        "PROFESSIONALE",
    )
    if any(token in text for token in upper_tokens):
        return ServiceType.UPPER_SECONDARY_SCHOOL.value, "mim_grade_description"

    return None, "unclassified_grade_description"


def registry_grade_column(registry: pd.DataFrame) -> str:
    for column in ("grade_description", "school_type"):
        if column in registry.columns:
            return column
    raise RuntimeError(
        "MIM school registry has neither grade_description nor school_type."
    )


def load_school_registry(path: Path, municipality_code: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"MIM school registry not found: {path}")

    registry = pd.read_parquet(path).copy()
    if "school_code" not in registry.columns:
        raise RuntimeError("MIM school registry missing school_code.")

    registry["school_code"] = registry["school_code"].astype("string").str.strip()

    if "municipality_code" in registry.columns:
        registry["municipality_code"] = (
            registry["municipality_code"].astype("string").str.strip().str.zfill(6)
        )
        wrong = registry["municipality_code"].notna() & (
            registry["municipality_code"] != municipality_code
        )
        if wrong.any():
            raise RuntimeError(
                f"MIM registry contains {int(wrong.sum())} rows from another municipality."
            )

    if registry["school_code"].duplicated().any():
        duplicates = registry.loc[
            registry["school_code"].duplicated(keep=False), "school_code"
        ].dropna().astype(str).tolist()
        raise RuntimeError(f"Duplicate MIM school_code values: {duplicates[:20]}")

    return registry


def education_relations(
    legacy_education: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    grade_column = registry_grade_column(registry)
    registry_by_code = registry.set_index("school_code", drop=False)

    rows: list[dict[str, Any]] = []
    for _, site in legacy_education.iterrows():
        legacy_id = clean_text(site.get("service_site_id"))
        if legacy_id is None:
            raise RuntimeError("Legacy education row missing service_site_id.")

        provenance = parse_json_object(site.get("provenance_json"))
        codes = parse_list(provenance.get("linked_school_codes"))

        # Older migrated rows may carry the list directly as source_record_id.
        if not codes:
            codes = parse_list(site.get("source_record_id"))

        if not codes:
            rows.append(
                {
                    "legacy_service_site_id": legacy_id,
                    "school_code": None,
                    "grade_description": None,
                    "registry_type": None,
                    "service_type": None,
                    "classification_status": "unresolved_no_school_code",
                    "classification_method": "legacy_provenance",
                }
            )
            continue

        for code in codes:
            if code not in registry_by_code.index:
                rows.append(
                    {
                        "legacy_service_site_id": legacy_id,
                        "school_code": code,
                        "grade_description": None,
                        "registry_type": None,
                        "service_type": None,
                        "classification_status": "unresolved_registry_miss",
                        "classification_method": "school_code_lookup",
                    }
                )
                continue

            school = registry_by_code.loc[code]
            grade = clean_text(school.get(grade_column))
            service_type, method = classify_school_grade(grade)
            rows.append(
                {
                    "legacy_service_site_id": legacy_id,
                    "school_code": code,
                    "grade_description": grade,
                    "registry_type": clean_text(school.get("registry_type")),
                    "service_type": service_type,
                    "classification_status": (
                        "classified" if service_type else "unresolved_grade"
                    ),
                    "classification_method": method,
                    "source_reference_date": parse_iso_date(
                        school.get("source_catalog_data_as_of")
                    ),
                    "retrieved_at": clean_text(school.get("ingested_at_utc")),
                }
            )

    relation = pd.DataFrame(rows)
    if relation.empty:
        raise RuntimeError("Education site-to-service-type relation is empty.")
    return relation


def education_v2_rows(
    legacy_education: pd.DataFrame,
    relation: pd.DataFrame,
    school_year: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    by_site = {
        key: group.copy()
        for key, group in relation.groupby("legacy_service_site_id", sort=False)
    }

    for _, site in legacy_education.iterrows():
        legacy_id = clean_text(site.get("service_site_id"))
        if legacy_id is None:
            continue
        rel = by_site.get(legacy_id)
        if rel is None:
            continue

        classified = rel.loc[rel["service_type"].notna()].copy()
        for service_type, group in classified.groupby("service_type", sort=True):
            if service_type not in EDUCATION_SERVICE_TYPES:
                continue

            codes = group["school_code"].dropna().astype(str).tolist()
            grades = group["grade_description"].dropna().astype(str).tolist()
            ownerships = group["registry_type"].dropna().astype(str).tolist()
            ownership_unique = sorted(set(ownerships))
            if len(ownership_unique) == 1:
                service_subtype = ownership_unique[0]
            elif len(ownership_unique) > 1:
                service_subtype = "mixed_ownership"
            else:
                service_subtype = None

            source_reference_date = one_or_none(group.get("source_reference_date", []))
            retrieved_at = one_or_none(group.get("retrieved_at", []))

            rows.append(
                base_v2_row(
                    legacy=site,
                    service_type=service_type,
                    service_subtype=service_subtype,
                    source_name="MIM",
                    source_record_id=json_list(codes),
                    source_reference_date=source_reference_date,
                    source_reference_period=str(school_year),
                    retrieved_at=retrieved_at,
                    operational_status=OperationalStatus.ACTIVE.value,
                    extras={
                        "source_record_ids_json": json_list(codes),
                        "source_grade_descriptions_json": json_list(grades),
                        "school_ownerships_json": json_list(ownerships),
                        "classification_method": "MIM grade_description by linked school_code",
                    },
                )
            )

    return rows


# ---------------------------------------------------------------------------
# Health mapping
# ---------------------------------------------------------------------------

def health_service_type(legacy_subcategory: Any) -> str | None:
    value = clean_text(legacy_subcategory)
    if value == "pharmacy":
        return ServiceType.PHARMACY.value
    if value == "hospital":
        return ServiceType.HOSPITAL_ESTABLISHMENT.value
    if value in {"community_house", "casa_della_comunita", "cdc"}:
        return ServiceType.COMMUNITY_HOUSE.value
    return None


def source_reference_fields(legacy: pd.Series) -> tuple[str | None, str | None]:
    period = clean_text(legacy.get("reference_period"))
    if period is None:
        return None, None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
        return period, period
    return None, period


def legacy_retrieved_at(legacy: pd.Series) -> str | None:
    provenance = parse_json_object(legacy.get("provenance_json"))
    for key in (
        "retrieved_at",
        "ingested_at_utc",
        "source_retrieved_at",
        "retrieved_at_utc",
    ):
        value = clean_text(provenance.get(key))
        if value:
            return value
    return None


def health_v2_rows(legacy_health: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, legacy in legacy_health.iterrows():
        service_type = health_service_type(legacy.get("subcategory"))
        if service_type is None:
            continue

        reference_date, reference_period = source_reference_fields(legacy)
        rows.append(
            base_v2_row(
                legacy=legacy,
                service_type=service_type,
                service_subtype=None,
                source_name="Ministero della Salute",
                source_record_id=clean_text(legacy.get("source_record_id")),
                source_reference_date=reference_date,
                source_reference_period=reference_period,
                retrieved_at=legacy_retrieved_at(legacy),
                operational_status=OperationalStatus.ACTIVE.value,
                extras={},
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Common ServiceV2 mapping
# ---------------------------------------------------------------------------

def infer_geocode_quality(legacy: pd.Series) -> str:
    lon = clean_float(legacy.get("longitude"))
    lat = clean_float(legacy.get("latitude"))
    if lon is None or lat is None:
        return GeocodeQuality.UNRESOLVED.value

    coordinate_source = (clean_text(legacy.get("coordinate_source")) or "").lower()
    resolution = (clean_text(legacy.get("coordinate_resolution")) or "").lower()
    confidence = (clean_text(legacy.get("confidence")) or "").lower()

    if any(token in coordinate_source for token in ("ministero", "source_coordinate")):
        return GeocodeQuality.SOURCE_COORDINATE.value
    if resolution == "address" and confidence in {"high", "medium_high"}:
        return GeocodeQuality.ADDRESS_EXACT.value
    if resolution == "street_anchor":
        return GeocodeQuality.ADDRESS_APPROXIMATE.value
    return GeocodeQuality.FALLBACK.value


def base_v2_row(
    *,
    legacy: pd.Series,
    service_type: str,
    service_subtype: str | None,
    source_name: str,
    source_record_id: str | None,
    source_reference_date: str | None,
    source_reference_period: str | None,
    retrieved_at: str | None,
    operational_status: str,
    extras: dict[str, Any],
) -> dict[str, Any]:
    legacy_id = clean_text(legacy.get("service_site_id"))
    if legacy_id is None:
        raise RuntimeError("Legacy service row missing service_site_id.")

    provenance = parse_json_object(legacy.get("provenance_json"))
    geocode_method = clean_text(provenance.get("location_method"))
    if geocode_method is None:
        geocode_method = clean_text(legacy.get("coordinate_source"))

    row: dict[str, Any] = {
        "service_id": service_v2_id(legacy_id, service_type),
        "service_type": service_type,
        "service_subtype": service_subtype,
        "name": clean_text(legacy.get("name")) or legacy_id,
        "source_name": source_name,
        "source_record_id": source_record_id,
        "source_reference_date": source_reference_date,
        "source_reference_period": source_reference_period,
        "retrieved_at": retrieved_at,
        "raw_address": clean_text(legacy.get("address")),
        "normalized_address": None,
        "municipality_code": clean_text(legacy.get("municipality_code")),
        "latitude": clean_float(legacy.get("latitude")),
        "longitude": clean_float(legacy.get("longitude")),
        "crs": "EPSG:4326",
        "coordinate_source": clean_text(legacy.get("coordinate_source")),
        "geocode_method": geocode_method,
        "geocode_quality": infer_geocode_quality(legacy),
        "operational_status": operational_status,
        "status_date": None,
        "capacity_value": clean_float(legacy.get("capacity_value")),
        "capacity_unit": clean_text(legacy.get("capacity_unit")),
        "source_release_id_or_version": source_reference_period,
        "source_checksum": None,
        "legacy_service_site_id": legacy_id,
        "legacy_category": clean_text(legacy.get("category")),
        "legacy_subcategory": clean_text(legacy.get("subcategory")),
        "legacy_usable_for_accessibility": bool(
            legacy.get("usable_for_accessibility", False)
        ),
        "migration_status": (
            "complete"
            if retrieved_at is not None
            else "legacy_missing_retrieved_at"
        ),
    }
    row.update(extras)
    return row


def validate_v2(services: pd.DataFrame, municipality_code: str) -> None:
    SERVICE_V2.validate_columns(services.columns)
    if services.empty:
        raise RuntimeError("ServiceV2 layer is empty.")

    if services["service_id"].duplicated().any():
        duplicates = services.loc[
            services["service_id"].duplicated(keep=False), "service_id"
        ].tolist()
        raise RuntimeError(f"Duplicate service_id values: {duplicates[:20]}")

    wrong = services["municipality_code"].notna() & (
        services["municipality_code"] != municipality_code
    )
    if wrong.any():
        raise RuntimeError(
            f"{int(wrong.sum())} ServiceV2 rows belong to another municipality."
        )

    invalid_types = sorted(
        set(services["service_type"].dropna().astype(str))
        - {item.value for item in ServiceType}
    )
    if invalid_types:
        raise RuntimeError(f"Unknown ServiceV2 service_type values: {invalid_types}")

    usable = services["legacy_usable_for_accessibility"].fillna(False).astype(bool)
    bad_geometry = usable & (
        services["longitude"].isna() | services["latitude"].isna()
    )
    if bad_geometry.any():
        raise RuntimeError(
            f"{int(bad_geometry.sum())} usable ServiceV2 rows have no coordinates."
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def legacy_service_path(args: argparse.Namespace) -> Path:
    label = args.health_reference_date.strftime("%Y%m%d")
    return (
        PROCESSED_SERVICES_DIR
        / args.municipality_code
        / f"service_sites_{args.school_year}_{label}.parquet"
    )


def school_registry_path(args: argparse.Namespace) -> Path:
    return (
        PROCESSED_MIM_DIR
        / args.municipality_code
        / f"schools_registry_{args.school_year}.parquet"
    )


def to_geodataframe(services: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry = []
    for lon, lat in zip(services["longitude"], services["latitude"]):
        lon = clean_float(lon)
        lat = clean_float(lat)
        geometry.append(Point(lon, lat) if lon is not None and lat is not None else None)
    return gpd.GeoDataFrame(services.copy(), geometry=geometry, crs="EPSG:4326")


def main() -> None:
    args = parse_args()
    legacy_path = legacy_service_path(args)
    registry_path = school_registry_path(args)

    if not legacy_path.exists():
        raise FileNotFoundError(
            f"Legacy canonical service layer not found: {legacy_path}\n"
            "Run the existing services stage first."
        )

    legacy = pd.read_parquet(legacy_path).copy()
    required_legacy = {
        "service_site_id",
        "category",
        "subcategory",
        "name",
        "municipality_code",
        "longitude",
        "latitude",
        "usable_for_accessibility",
        "provenance_json",
    }
    missing = sorted(required_legacy - set(legacy.columns))
    if missing:
        raise RuntimeError(f"Legacy service layer missing columns: {missing}")

    education = legacy.loc[legacy["category"] == "education"].copy()
    health = legacy.loc[legacy["category"] == "health"].copy()

    registry = load_school_registry(registry_path, args.municipality_code)
    relation = education_relations(education, registry)

    rows = education_v2_rows(education, relation, str(args.school_year))
    rows.extend(health_v2_rows(health))
    services = pd.DataFrame(rows)

    validate_v2(services, args.municipality_code)
    services = services.sort_values(
        ["service_type", "legacy_service_site_id", "service_id"]
    ).reset_index(drop=True)
    gdf = to_geodataframe(services)

    label = args.health_reference_date.strftime("%Y%m%d")
    processed_dir = PROCESSED_SERVICES_DIR / args.municipality_code
    feature_dir = FEATURES_SERVICES_DIR / args.municipality_code
    processed_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    service_parquet = (
        processed_dir
        / f"service_entities_v2_{args.school_year}_{label}.parquet"
    )
    service_csv = (
        feature_dir
        / f"service_entities_v2_{args.school_year}_{label}.csv"
    )
    relation_parquet = (
        processed_dir
        / f"education_site_service_types_v2_{args.school_year}.parquet"
    )
    relation_csv = (
        feature_dir
        / f"education_site_service_types_v2_{args.school_year}.csv"
    )
    manifest_path = (
        feature_dir
        / f"service_entities_v2_{args.school_year}_{label}_manifest.json"
    )

    gdf.to_parquet(service_parquet, index=False)
    services.to_csv(service_csv, index=False, encoding="utf-8-sig")
    relation.to_parquet(relation_parquet, index=False)
    relation.to_csv(relation_csv, index=False, encoding="utf-8-sig")

    relation_status = relation["classification_status"].value_counts(dropna=False)
    v2_counts = services["service_type"].value_counts(dropna=False)

    legacy_education_sites = int(len(education))
    classified_site_ids = set(
        relation.loc[relation["service_type"].notna(), "legacy_service_site_id"]
    )
    unresolved_usable_education_sites = set(
        education.loc[
            education["usable_for_accessibility"].fillna(False).astype(bool),
            "service_site_id",
        ].astype(str)
    ) - classified_site_ids

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "schema_version": SCHEMA_VERSION,
        "municipality_code": args.municipality_code,
        "inputs": {
            "legacy_service_layer": str(legacy_path),
            "mim_school_registry": str(registry_path),
        },
        "legacy_service_rows": int(len(legacy)),
        "legacy_education_sites": legacy_education_sites,
        "legacy_health_sites": int(len(health)),
        "service_v2_rows": int(len(services)),
        "service_type_counts": {
            str(key): int(value) for key, value in v2_counts.items()
        },
        "education_relation_rows": int(len(relation)),
        "education_classification_status_counts": {
            str(key): int(value) for key, value in relation_status.items()
        },
        "education_sites_with_at_least_one_classified_type": int(
            len(classified_site_ids)
        ),
        "usable_education_sites_without_classified_type": int(
            len(unresolved_usable_education_sites)
        ),
        "usable_education_site_ids_without_classified_type": sorted(
            unresolved_usable_education_sites
        ),
        "migration_missing_retrieved_at_rows": int(
            (services["migration_status"] == "legacy_missing_retrieved_at").sum()
        ),
        "notes": [
            "This is a parallel migration artifact; the legacy canonical service layer is not modified.",
            "Education service_type is derived only from official MIM grade descriptions linked by school_code.",
            "A physical school site may produce multiple ServiceV2 rows when it hosts multiple education levels.",
            "Multiple school codes of the same education level at one physical site are deduplicated to one ServiceV2 opportunity.",
            "Unknown/organisational MIM grade labels are not guessed from school names.",
            "Hospital means hospital establishment accessibility, not emergency/DEA/pronto-soccorso accessibility.",
            "Legacy rows missing retrieval timestamps remain explicit as legacy_missing_retrieved_at rather than receiving invented timestamps.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n====================================")
    print(" CANONICAL SERVICE SCHEMA V2")
    print("====================================")
    print(f"Comune: {args.municipality_code}")
    print(f"Legacy rows (unchanged): {len(legacy)}")
    print(f"ServiceV2 rows: {len(services)}")
    print("\nService types:")
    print(services["service_type"].value_counts(dropna=False).to_string())
    print("\nEducation classification:")
    print(relation["classification_status"].value_counts(dropna=False).to_string())
    print(
        "Usable education sites without a classified v2 type: "
        f"{len(unresolved_usable_education_sites)}"
    )
    print("\n=== OUTPUT ===")
    for path in (
        service_parquet,
        service_csv,
        relation_parquet,
        relation_csv,
        manifest_path,
    ):
        print(f"✓ {path}")


if __name__ == "__main__":
    main()
