"""
Canonical Education matching/geolocation module.

This module consolidates the previously separate operational entry points for:
1. MIM school -> OSM school-site discovery/matching;
2. MIM physical-building geocoding;
3. MIM physical-building -> OSM site matching.

The legacy implementations are preserved internally during the regression-safe
refactor. The public CLI is the only supported operational entry point.
"""

from __future__ import annotations

import argparse as _cli_argparse
import sys as _sys
from pathlib import Path as _CliPath

_EDUCATION_ROOT = _CliPath(__file__).resolve().parents[2]
_EDUCATION_SRC = _EDUCATION_ROOT / "src"

if str(_EDUCATION_SRC) not in _sys.path:
    _sys.path.insert(0, str(_EDUCATION_SRC))

from core.config import DEFAULT_CONFIG as _DEFAULT_CONFIG




# ============================================================================
# SCHOOL / OSM SITE DISCOVERY AND MATCHING
# ============================================================================

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz
from shapely import wkt
from sqlalchemy import create_engine, text


# ============================================================
# PATHS / CONSTANTS
# ============================================================

school_sites_ROOT = Path(__file__).resolve().parents[2]

school_sites_PROCESSED_MIM_DIR = (
    school_sites_ROOT
    / "data"
    / "processed"
    / "mim"
)

school_sites_RAW_OSM_DIR = (
    school_sites_ROOT
    / "data"
    / "raw"
    / "osm"
)

school_sites_PROCESSED_OSM_DIR = (
    school_sites_ROOT
    / "data"
    / "processed"
    / "osm"
)

school_sites_DEFAULT_SCHOOL_YEAR = "202627"

school_sites_DEFAULT_AUTO_THRESHOLD = 85.0
school_sites_DEFAULT_REVIEW_THRESHOLD = 65.0
school_sites_DEFAULT_MIN_MARGIN = 8.0

# Distanza massima usata esclusivamente
# per identificare possibili duplicati OSM.
school_sites_DUPLICATE_MAX_DISTANCE_M = 35.0

school_sites_MISSING_TEXT_VALUES = {
    "",
    "NAN",
    "NONE",
    "NULL",
    "N/A",
    "NA",
    "N.D.",
    "ND",
    "NON DISPONIBILE",
    "NON DISP.",
    "-",
}


# ============================================================
# RECORD CHE RICHIEDONO REVISIONE
# ============================================================

school_sites_ROLE_REVIEW_PATTERNS = {
    "ISTITUTO COMPRENSIVO",
    "ISTITUTO SUPERIORE",
    "CENTRO TERRITORIALE",
    "CONVITTO ANNESSO",
}

school_sites_NAME_REVIEW_PATTERNS = {
    "CORSO SERALE",
    "CASA CIRCONDARIALE",
    "OSPEDALIERA",
    "CPIA",
    "CTP",
}


# ============================================================
# CLI
# ============================================================

def school_sites_parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Geolocalizzazione V2 delle scuole MIM "
            "tramite matching conservativo con OSM."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help=(
            "Codice ISTAT a 6 cifre. "
            "Esempio: 077014."
        ),
    )

    parser.add_argument(
        "--school-year",
        default=school_sites_DEFAULT_SCHOOL_YEAR,
        help=(
            "Anno scolastico MIM. "
            "Default: 202627."
        ),
    )

    parser.add_argument(
        "--refresh-osm",
        action="store_true",
        help=(
            "Interroga nuovamente OSM anche se "
            "esiste già uno snapshot locale."
        ),
    )

    parser.add_argument(
        "--auto-threshold",
        type=float,
        default=school_sites_DEFAULT_AUTO_THRESHOLD,
    )

    parser.add_argument(
        "--review-threshold",
        type=float,
        default=school_sites_DEFAULT_REVIEW_THRESHOLD,
    )

    parser.add_argument(
        "--min-margin",
        type=float,
        default=school_sites_DEFAULT_MIN_MARGIN,
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "municipality-code deve avere "
            "esattamente 6 cifre."
        )

    if not (
        0
        <= args.review_threshold
        <= args.auto_threshold
        <= 100
    ):
        raise ValueError(
            "Le soglie devono soddisfare: "
            "0 <= review <= auto <= 100."
        )

    if args.min_margin < 0:
        raise ValueError(
            "min-margin deve essere >= 0."
        )

    return args


# ============================================================
# DATABASE
# ============================================================

def school_sites_get_database_engine():
    load_dotenv(
        school_sites_ROOT / ".env"
    )

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL non definito "
            "nel file .env."
        )

    return create_engine(
        database_url
    )


def school_sites_load_municipality(
    engine,
    municipality_code,
):
    query = text("""
        SELECT
            id,
            istat_code,
            name,
            ST_AsText(geometry) AS geometry_wkt

        FROM municipality

        WHERE istat_code = :istat_code;
    """)

    with engine.connect() as connection:

        row = connection.execute(
            query,
            {
                "istat_code":
                    municipality_code
            },
        ).mappings().first()

    if row is None:
        raise RuntimeError(
            "Comune non presente in PostGIS. "
            "Eseguire prima l'ingestion ISTAT."
        )

    geometry = wkt.loads(
        row["geometry_wkt"]
    )

    if geometry.geom_type not in {
        "Polygon",
        "MultiPolygon",
    }:
        raise RuntimeError(
            "Boundary comunale non valido."
        )

    return {
        "id": row["id"],
        "istat_code":
            row["istat_code"],
        "name":
            row["name"],
        "geometry":
            geometry,
    }


# ============================================================
# GENERIC HELPERS
# ============================================================

def school_sites_is_missing(value):
    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

        if hasattr(result, "item"):
            return bool(
                result.item()
            )

    except Exception:
        pass

    return False


def school_sites_clean_text_value(value):
    if school_sites_is_missing(value):
        return None

    value = (
        str(value)
        .strip()
    )

    if (
        value.upper()
        in school_sites_MISSING_TEXT_VALUES
    ):
        return None

    return value


def school_sites_unique_nonempty(values):
    output = []

    for value in values:

        if value is None:
            continue

        if value not in output:
            output.append(value)

    return output


def school_sites_json_safe(value):
    if value is None:
        return None

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            school_sites_json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key):
                school_sites_json_safe(item)
            for key, item
            in value.items()
        }

    if school_sites_is_missing(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        return float(value)

    if hasattr(value, "item"):

        try:
            return school_sites_json_safe(
                value.item()
            )

        except Exception:
            pass

    return str(value)


def school_sites_strict_json_dumps(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def school_sites_normalize_text(value):
    value = school_sites_clean_text_value(
        value
    )

    if value is None:
        return None

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        char
        for char in value
        if not unicodedata.combining(
            char
        )
    )

    value = value.upper()

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value or None


def school_sites_normalize_address(value):
    value = school_sites_normalize_text(
        value
    )

    if value is None:
        return None

    # VIA FERMI10 -> VIA FERMI 10
    value = re.sub(
        r"(?<=[A-Z])(?=\d)",
        " ",
        value,
    )

    value = re.sub(
        r"(?<=\d)(?=[A-Z])",
        " ",
        value,
    )

    replacements = {
        "P ZA": "PIAZZA",
        "PZZA": "PIAZZA",
        "V LE": "VIALE",
        "VLE": "VIALE",
        "C SO": "CORSO",
        "SNC": "",
        "S N C": "",
    }

    for old, new in (
        replacements.items()
    ):
        value = value.replace(
            old,
            new,
        )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value or None


def school_sites_normalize_school_name(
    value,
    municipality_name=None,
):
    value = school_sites_normalize_text(
        value
    )

    if value is None:
        return None

    if municipality_name:

        municipality_normalized = (
            school_sites_normalize_text(
                municipality_name
            )
        )

        if municipality_normalized:

            pattern = (
                r"\b"
                + re.escape(
                    municipality_normalized
                )
                + r"\b"
            )

            value = re.sub(
                pattern,
                " ",
                value,
            )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value or None


def school_sites_school_name_core(
    value,
    municipality_name=None,
):
    value = school_sites_normalize_school_name(
        value,
        municipality_name,
    )

    if value is None:
        return None

    generic_tokens = {
        "SCUOLA",
        "ISTITUTO",
        "IST",
        "I",
        "C",
        "IC",
        "PLESSO",
        "SEDE",
        "MT",
        "MATERNA",
        "INFANZIA",
        "PRIMARIA",
        "SECONDARIA",
        "PRIMO",
        "SECONDO",
        "GRADO",
        "STATALE",
    }

    tokens = [
        token
        for token in value.split()
        if token not in generic_tokens
    ]

    if not tokens:
        return value

    return " ".join(tokens)


# ============================================================
# MIM REGISTRY
# ============================================================

def school_sites_registry_path(
    municipality_code,
    school_year,
):
    return (
        school_sites_PROCESSED_MIM_DIR
        / municipality_code
        / (
            f"schools_registry_"
            f"{school_year}.parquet"
        )
    )


def school_sites_contains_review_pattern(
    value,
    patterns,
):
    value = school_sites_normalize_text(
        value
    )

    if not value:
        return False

    return any(
        pattern in value
        for pattern in patterns
    )


def school_sites_build_review_reason(
    school_type,
    school_name,
):
    reasons = []

    if school_sites_contains_review_pattern(
        school_type,
        school_sites_ROLE_REVIEW_PATTERNS,
    ):
        reasons.append(
            "tipologia aggregata/"
            "organizzativa"
        )

    if school_sites_contains_review_pattern(
        school_name,
        school_sites_NAME_REVIEW_PATTERNS,
    ):
        reasons.append(
            "programma/sede speciale"
        )

    if not reasons:
        return None

    return "; ".join(
        reasons
    )


def school_sites_load_mim_registry(
    municipality_code,
    school_year,
    municipality_name,
):
    path = school_sites_registry_path(
        municipality_code,
        school_year,
    )

    if not path.exists():
        raise FileNotFoundError(
            "\nRegistro MIM Silver non trovato:\n"
            f"{path}\n\n"
            "Eseguire prima mim_schools.py."
        )

    schools = pd.read_parquet(
        path
    ).copy()

        # --------------------------------------------------------
    # SILVER SCHEMA COMPATIBILITY
    # --------------------------------------------------------
    # Canonical names introduced by the temporally aligned
    # 2024/25 MIM ingestion. Keep legacy aliases so that the
    # geolocation pipeline remains backward compatible.

    column_aliases = {
        # nuovo Silver 2024/25 -> nomi legacy usati dal matcher
        "school_address": "address",
        "grade_description": "school_type",
        "institute_reference_code": "reference_institute_code",
        "institute_reference_name": "reference_institute_name",
        "source_school_year": "school_year",
        "registry_type": "school_ownership",
        "municipality_code": "municipality_istat_code",
    }

    for canonical_column, legacy_column in column_aliases.items():

        if (
            legacy_column not in schools.columns
            and canonical_column in schools.columns
        ):
            schools[
                legacy_column
            ] = schools[
                canonical_column
            ]

    for canonical_column, legacy_column in column_aliases.items():

        if (
            legacy_column not in schools.columns
            and canonical_column in schools.columns
        ):
            schools[
                legacy_column
            ] = schools[
                canonical_column
            ]

    text_columns = [
        "school_name",
        "school_type",
        "address",
        "postal_code",
        "reference_institute_code",
        "reference_institute_name",
    ]

    for column in text_columns:

        if column in schools.columns:

            schools[column] = (
                schools[column]
                .apply(
                    school_sites_clean_text_value
                )
            )

    schools[
        "school_name_normalized"
    ] = (
        schools["school_name"]
        .apply(
            lambda value:
                school_sites_normalize_school_name(
                    value,
                    municipality_name,
                )
        )
    )

    schools[
        "school_name_core"
    ] = (
        schools["school_name"]
        .apply(
            lambda value:
                school_sites_school_name_core(
                    value,
                    municipality_name,
                )
        )
    )

    schools[
        "reference_name_normalized"
    ] = (
        schools[
            "reference_institute_name"
        ]
        .apply(
            lambda value:
                school_sites_normalize_school_name(
                    value,
                    municipality_name,
                )
        )
    )

    schools[
        "reference_name_core"
    ] = (
        schools[
            "reference_institute_name"
        ]
        .apply(
            lambda value:
                school_sites_school_name_core(
                    value,
                    municipality_name,
                )
        )
    )

    schools[
        "address_normalized"
    ] = (
        schools["address"]
        .apply(
            school_sites_normalize_address
        )
    )

    schools[
        "service_review_reason"
    ] = schools.apply(
        lambda row:
            school_sites_build_review_reason(
                row["school_type"],
                row["school_name"],
            ),
        axis=1,
    )

    schools[
        "requires_service_review"
    ] = (
        schools[
            "service_review_reason"
        ]
        .notna()
    )

    print(
        "\n=== MIM REGISTRY ==="
    )

    print(
        f"Record: {len(schools)}"
    )

    missing_addresses = (
        schools[
            "address_normalized"
        ]
        .isna()
        .sum()
    )

    print(
        "Indirizzi mancanti dopo "
        "normalizzazione: "
        f"{missing_addresses}"
    )

    service_review_count = (
        schools[
            "requires_service_review"
        ]
        .sum()
    )

    print(
        "Record che richiedono "
        "service review: "
        f"{service_review_count}"
    )

    return schools

# ============================================================
# OSM SNAPSHOT
# ============================================================

def school_sites_osm_school_snapshot_dir(
    municipality_code,
):
    directory = (
        school_sites_RAW_OSM_DIR
        / municipality_code
        / "school_features"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def school_sites_latest_osm_snapshot(
    municipality_code,
):
    directory = (
        school_sites_osm_school_snapshot_dir(
            municipality_code
        )
    )

    snapshots = sorted(
        directory.glob(
            "school_features_*.parquet"
        )
    )

    if not snapshots:
        return None

    return snapshots[-1]


def school_sites_configure_osmnx():
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 180


def school_sites_download_osm_school_features(
    boundary,
):
    school_sites_configure_osmnx()

    print(
        "\nDownload strutture scolastiche "
        "da OpenStreetMap..."
    )

    tags = {
        "amenity": [
            "school",
            "kindergarten",
        ],
        "building": [
            "school",
            "kindergarten",
        ],
    }

    features = (
        ox.features.features_from_polygon(
            boundary,
            tags,
        )
    )

    if features.empty:
        raise RuntimeError(
            "Nessuna struttura scolastica "
            "OSM trovata."
        )

    features = features.copy()

    features[
        "osm_element_type"
    ] = (
        features.index
        .get_level_values(0)
        .astype(str)
    )

    features[
        "osm_id"
    ] = (
        features.index
        .get_level_values(1)
        .astype(str)
    )

    features = features.reset_index(
        drop=True
    )

    source_tags = []

    for _, row in features.iterrows():

        tags_dict = {}

        for column in features.columns:

            if column in {
                "geometry",
                "osm_element_type",
                "osm_id",
            }:
                continue

            value = school_sites_json_safe(
                row[column]
            )

            if value is not None:
                tags_dict[
                    column
                ] = value

        source_tags.append(
            school_sites_strict_json_dumps(
                tags_dict
            )
        )

    features[
        "source_tags"
    ] = source_tags

    features[
        "retrieved_at"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    return gpd.GeoDataFrame(
        features[
            [
                "osm_element_type",
                "osm_id",
                "retrieved_at",
                "source_tags",
                "geometry",
            ]
        ],
        geometry="geometry",
        crs=features.crs,
    )


def school_sites_save_osm_snapshot(
    features,
    municipality_code,
):
    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    path = (
        school_sites_osm_school_snapshot_dir(
            municipality_code
        )
        / (
            "school_features_"
            f"{timestamp}.parquet"
        )
    )

    features.to_parquet(
        path,
        index=False,
    )

    print(
        "\n✓ Snapshot OSM salvato:"
    )

    print(
        f"  {path}"
    )

    return path


def school_sites_load_or_download_osm_features(
    boundary,
    municipality_code,
    refresh=False,
):
    snapshot = school_sites_latest_osm_snapshot(
        municipality_code
    )

    if (
        snapshot is not None
        and not refresh
    ):
        print(
            "\nCaricamento snapshot "
            "scolastico OSM locale:"
        )

        print(
            f"  {snapshot}"
        )

        return gpd.read_parquet(
            snapshot
        )

    features = (
        school_sites_download_osm_school_features(
            boundary
        )
    )

    school_sites_save_osm_snapshot(
        features,
        municipality_code,
    )

    return features


# ============================================================
# OSM CANDIDATES
# ============================================================

def school_sites_parse_source_tags(value):
    if value is None:
        return {}

    return json.loads(
        value
    )


def school_sites_first_nonempty(
    tags,
    keys,
):
    for key in keys:

        value = school_sites_clean_text_value(
            tags.get(key)
        )

        if value is not None:
            return value

    return None


def school_sites_feature_point(geometry):
    if geometry is None:
        return None

    if geometry.is_empty:
        return None

    if geometry.geom_type == "Point":
        return geometry

    return geometry.representative_point()


def school_sites_candidate_addresses(tags):
    addresses = []

    full_address = school_sites_first_nonempty(
        tags,
        [
            "addr:full",
        ],
    )

    if full_address:
        addresses.append(
            full_address
        )

    street = school_sites_first_nonempty(
        tags,
        [
            "addr:street",
            "addr:place",
        ],
    )

    number = school_sites_first_nonempty(
        tags,
        [
            "addr:housenumber",
        ],
    )

    if street and number:

        addresses.append(
            f"{street} {number}"
        )

    elif street:

        addresses.append(
            street
        )

    return school_sites_unique_nonempty(
        addresses
    )


def school_sites_build_osm_candidates(
    raw_features,
    municipality_name,
):
    rows = []

    for row in (
        raw_features.itertuples()
    ):

        tags = school_sites_parse_source_tags(
            row.source_tags
        )

        point = school_sites_feature_point(
            row.geometry
        )

        if point is None:
            continue

        name = school_sites_first_nonempty(
            tags,
            [
                "name",
            ],
        )

        official_name = school_sites_first_nonempty(
            tags,
            [
                "official_name",
            ],
        )

        alt_name = school_sites_first_nonempty(
            tags,
            [
                "alt_name",
                "short_name",
            ],
        )

        operator = school_sites_first_nonempty(
            tags,
            [
                "operator",
            ],
        )

        raw_primary_names = (
            school_sites_unique_nonempty(
                [
                    name,
                    official_name,
                    alt_name,
                ]
            )
        )

        normalized_primary_names = [
            school_sites_normalize_school_name(
                value,
                municipality_name,
            )
            for value
            in raw_primary_names
        ]

        normalized_primary_names = (
            school_sites_unique_nonempty(
                normalized_primary_names
            )
        )

        core_primary_names = [
            school_sites_school_name_core(
                value,
                municipality_name,
            )
            for value
            in raw_primary_names
        ]

        core_primary_names = (
            school_sites_unique_nonempty(
                core_primary_names
            )
        )

        operator_names = (
            school_sites_unique_nonempty(
                [
                    school_sites_normalize_school_name(
                        operator,
                        municipality_name,
                    )
                ]
            )
        )

        raw_addresses = (
            school_sites_candidate_addresses(
                tags
            )
        )

        normalized_addresses = (
            school_sites_unique_nonempty(
                [
                    school_sites_normalize_address(
                        value
                    )
                    for value
                    in raw_addresses
                ]
            )
        )

        rows.append(
            {
                "osm_element_type":
                    row.osm_element_type,

                "osm_id":
                    row.osm_id,

                "osm_key":
                    (
                        f"{row.osm_element_type}:"
                        f"{row.osm_id}"
                    ),

                "osm_name":
                    name,

                "osm_operator":
                    operator,

                "osm_amenity":
                    school_sites_clean_text_value(
                        tags.get("amenity")
                    ),

                "osm_building":
                    school_sites_clean_text_value(
                        tags.get("building")
                    ),

                "osm_postal_code":
                    school_sites_clean_text_value(
                        tags.get(
                            "addr:postcode"
                        )
                    ),

                "primary_names":
                    normalized_primary_names,

                "core_primary_names":
                    core_primary_names,

                "operator_names":
                    operator_names,

                "addresses":
                    normalized_addresses,

                "raw_addresses":
                    raw_addresses,

                "longitude":
                    float(point.x),

                "latitude":
                    float(point.y),

                "geometry":
                    point,
            }
        )

    candidates = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs=4326,
    )

    if candidates.empty:
        raise RuntimeError(
            "Nessun candidato OSM "
            "utilizzabile."
        )

    print(
        "\n=== OSM RAW CANDIDATES ==="
    )

    print(
        f"Candidati raw: "
        f"{len(candidates)}"
    )

    return candidates


# ============================================================
# SIMILARITY
# ============================================================

def school_sites_name_similarity(
    left,
    right,
):
    if not left or not right:
        return 0.0

    return float(
        max(
            fuzz.WRatio(
                left,
                right,
            ),
            fuzz.token_set_ratio(
                left,
                right,
            ),
        )
    )


def school_sites_address_similarity(
    left,
    right,
):
    """
    Per gli indirizzi NON usiamo
    token_set_ratio perché potrebbe dare
    100 a VIA FERMI vs VIA FERMI 10.
    """

    if not left or not right:
        return 0.0

    return float(
        max(
            fuzz.ratio(
                left,
                right,
            ),
            fuzz.token_sort_ratio(
                left,
                right,
            ),
        )
    )


def school_sites_best_list_similarity(
    left_values,
    right_values,
    similarity_function,
):
    best = 0.0

    for left in left_values:

        for right in right_values:

            best = max(
                best,
                similarity_function(
                    left,
                    right,
                ),
            )

    return best


# ============================================================
# OSM DUPLICATE CONSOLIDATION
# ============================================================

class school_sites_UnionFind:

    def __init__(self, size):
        self.parent = list(
            range(size)
        )

    def find(self, value):

        while (
            self.parent[value]
            != value
        ):

            self.parent[value] = (
                self.parent[
                    self.parent[value]
                ]
            )

            value = (
                self.parent[value]
            )

        return value

    def union(
        self,
        left,
        right,
    ):
        root_left = self.find(
            left
        )

        root_right = self.find(
            right
        )

        if root_left != root_right:

            self.parent[
                root_right
            ] = root_left


def school_sites_should_merge_candidates(
    left,
    right,
    distance_m,
):
    if (
        distance_m
        > school_sites_DUPLICATE_MAX_DISTANCE_M
    ):
        return False

    name_score = (
        school_sites_best_list_similarity(
            (
                left[
                    "primary_names"
                ]
                + left[
                    "core_primary_names"
                ]
            ),
            (
                right[
                    "primary_names"
                ]
                + right[
                    "core_primary_names"
                ]
            ),
            school_sites_name_similarity,
        )
    )

    left_addresses = set(
        left["addresses"]
    )

    right_addresses = set(
        right["addresses"]
    )

    exact_address = bool(
        left_addresses
        & right_addresses
    )

    left_has_name = bool(
        left["primary_names"]
    )

    right_has_name = bool(
        right["primary_names"]
    )

    # Stesso nome praticamente esatto.
    if (
        distance_m <= 35
        and name_score >= 97
    ):
        return True

    # Stesso indirizzo + nome fortemente compatibile.
    if (
        distance_m <= 30
        and exact_address
        and name_score >= 85
    ):
        return True

    # Un oggetto senza nome può essere il building
    # dell'amenity nominato nello stesso punto.
    if (
        distance_m <= 15
        and exact_address
        and (
            not left_has_name
            or not right_has_name
        )
    ):
        return True

    # Nodo amenity + building dello stesso sito.
    school_tags_left = {
        school_sites_normalize_text(
            left["osm_amenity"]
        ),
        school_sites_normalize_text(
            left["osm_building"]
        ),
    }

    school_tags_right = {
        school_sites_normalize_text(
            right["osm_amenity"]
        ),
        school_sites_normalize_text(
            right["osm_building"]
        ),
    }

    if (
        distance_m <= 20
        and name_score >= 92
        and (
            "SCHOOL"
            in school_tags_left
            or "KINDERGARTEN"
            in school_tags_left
        )
        and (
            "SCHOOL"
            in school_tags_right
            or "KINDERGARTEN"
            in school_tags_right
        )
    ):
        return True

    return False


def school_sites_candidate_completeness(
    candidate,
):
    score = 0

    if candidate["primary_names"]:
        score += 5

    if candidate["addresses"]:
        score += 4

    if candidate["osm_amenity"]:
        score += 2

    if candidate["osm_building"]:
        score += 1

    if (
        candidate[
            "osm_element_type"
        ]
        in {"way", "relation"}
    ):
        score += 1

    return score


def school_sites_make_site_id(
    osm_keys,
):
    joined = "|".join(
        sorted(osm_keys)
    )

    digest = hashlib.sha1(
        joined.encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    return (
        f"osm_school_site_{digest}"
    )


def school_sites_consolidate_osm_candidates(
    candidates,
):
    print(
        "\n=== OSM SITE CONSOLIDATION ==="
    )

    if len(candidates) == 1:
        projected = candidates.copy()

    else:
        metric_crs = (
            candidates.estimate_utm_crs()
        )

        projected = (
            candidates.to_crs(
                metric_crs
            )
        )

    records = (
        candidates.to_dict(
            orient="records"
        )
    )

    union_find = school_sites_UnionFind(
        len(records)
    )

    merge_pairs = 0

    for left_index in range(
        len(records)
    ):

        for right_index in range(
            left_index + 1,
            len(records),
        ):

            distance_m = (
                projected.geometry.iloc[
                    left_index
                ].distance(
                    projected.geometry.iloc[
                        right_index
                    ]
                )
            )

            if school_sites_should_merge_candidates(
                records[left_index],
                records[right_index],
                distance_m,
            ):

                union_find.union(
                    left_index,
                    right_index,
                )

                merge_pairs += 1

    groups = {}

    for index in range(
        len(records)
    ):

        root = union_find.find(
            index
        )

        groups.setdefault(
            root,
            []
        ).append(index)

    sites = []

    for indices in groups.values():

        members = [
            records[index]
            for index in indices
        ]

        representative = max(
            members,
            key=school_sites_candidate_completeness,
        )

        osm_keys = [
            member["osm_key"]
            for member in members
        ]

        primary_names = (
            school_sites_unique_nonempty(
                [
                    value
                    for member in members
                    for value in member[
                        "primary_names"
                    ]
                ]
            )
        )

        core_primary_names = (
            school_sites_unique_nonempty(
                [
                    value
                    for member in members
                    for value in member[
                        "core_primary_names"
                    ]
                ]
            )
        )

        operator_names = (
            school_sites_unique_nonempty(
                [
                    value
                    for member in members
                    for value in member[
                        "operator_names"
                    ]
                ]
            )
        )

        addresses = (
            school_sites_unique_nonempty(
                [
                    value
                    for member in members
                    for value in member[
                        "addresses"
                    ]
                ]
            )
        )

        raw_addresses = (
            school_sites_unique_nonempty(
                [
                    value
                    for member in members
                    for value in member[
                        "raw_addresses"
                    ]
                ]
            )
        )

        postcodes = (
            school_sites_unique_nonempty(
                [
                    member[
                        "osm_postal_code"
                    ]
                    for member
                    in members
                ]
            )
        )

        amenities = (
            school_sites_unique_nonempty(
                [
                    member[
                        "osm_amenity"
                    ]
                    for member
                    in members
                ]
            )
        )

        buildings = (
            school_sites_unique_nonempty(
                [
                    member[
                        "osm_building"
                    ]
                    for member
                    in members
                ]
            )
        )

        sites.append(
            {
                "site_id":
                    school_sites_make_site_id(
                        osm_keys
                    ),

                "representative_osm_key":
                    representative[
                        "osm_key"
                    ],

                "site_name":
                    representative[
                        "osm_name"
                    ],

                "site_address":
                    (
                        raw_addresses[0]
                        if raw_addresses
                        else None
                    ),

                "primary_names":
                    primary_names,

                "core_primary_names":
                    core_primary_names,

                "operator_names":
                    operator_names,

                "addresses":
                    addresses,

                "postcodes":
                    postcodes,

                "amenities":
                    amenities,

                "buildings":
                    buildings,

                "member_osm_keys":
                    osm_keys,

                "member_count":
                    len(members),

                "longitude":
                    representative[
                        "longitude"
                    ],

                "latitude":
                    representative[
                        "latitude"
                    ],

                "geometry":
                    representative[
                        "geometry"
                    ],
            }
        )

    sites = gpd.GeoDataFrame(
        sites,
        geometry="geometry",
        crs=4326,
    )

    duplicated_elements = (
        len(candidates)
        - len(sites)
    )

    print(
        f"Candidati raw: "
        f"{len(candidates)}"
    )

    print(
        f"Siti consolidati: "
        f"{len(sites)}"
    )

    print(
        "Elementi OSM assorbiti come "
        "duplicati: "
        f"{duplicated_elements}"
    )

    print(
        f"Coppie di merge: "
        f"{merge_pairs}"
    )

    print(
        "Siti composti da più "
        "elementi OSM: "
        f"{(sites['member_count'] > 1).sum()}"
    )

    return sites


# ============================================================
# MATCH SCORE
# ============================================================

def school_sites_school_name_score(
    school,
    site,
):
    school_names = (
        school_sites_unique_nonempty(
            [
                school.get(
                    "school_name_normalized"
                ),
                school.get(
                    "school_name_core"
                ),
            ]
        )
    )

    site_names = (
        site["primary_names"]
        + site["core_primary_names"]
    )

    return school_sites_best_list_similarity(
        school_names,
        site_names,
        school_sites_name_similarity,
    )


def school_sites_school_address_score(
    school,
    site,
):
    school_address = (
        school.get(
            "address_normalized"
        )
    )

    if not school_address:
        return 0.0

    return school_sites_best_list_similarity(
        [school_address],
        site["addresses"],
        school_sites_address_similarity,
    )


def school_sites_reference_support_score(
    school,
    site,
):
    reference_names = (
        school_sites_unique_nonempty(
            [
                school.get(
                    "reference_name_normalized"
                ),
                school.get(
                    "reference_name_core"
                ),
            ]
        )
    )

    if not reference_names:
        return 0.0

    site_names = (
        site["primary_names"]
        + site["core_primary_names"]
        + site["operator_names"]
    )

    return school_sites_best_list_similarity(
        reference_names,
        site_names,
        school_sites_name_similarity,
    )


def school_sites_operator_support_score(
    school,
    site,
):
    school_names = (
        school_sites_unique_nonempty(
            [
                school.get(
                    "school_name_normalized"
                ),
                school.get(
                    "school_name_core"
                ),
            ]
        )
    )

    return school_sites_best_list_similarity(
        school_names,
        site["operator_names"],
        school_sites_name_similarity,
    )


def school_sites_type_bonus(
    school_type,
    site,
):
    school_type = (
        school_sites_normalize_text(
            school_type
        )
        or ""
    )

    amenities = {
        school_sites_normalize_text(value)
        for value
        in site["amenities"]
        if value
    }

    buildings = {
        school_sites_normalize_text(value)
        for value
        in site["buildings"]
        if value
    }

    osm_types = (
        amenities
        | buildings
    )

    if (
        "INFANZIA"
        in school_type
    ):

        if (
            "KINDERGARTEN"
            in osm_types
        ):
            return 3.0

        if "SCHOOL" in osm_types:
            return 1.0

    else:

        if "SCHOOL" in osm_types:
            return 2.0

    return 0.0


def school_sites_postcode_adjustment(
    school,
    site,
):
    mim_postcode = school_sites_clean_text_value(
        school.get(
            "postal_code"
        )
    )

    site_postcodes = (
        site["postcodes"]
    )

    if (
        not mim_postcode
        or not site_postcodes
    ):
        return 0.0, None

    match = (
        mim_postcode
        in site_postcodes
    )

    if match:
        return 1.0, True

    return -5.0, False


def school_sites_score_match(
    school,
    site,
):
    name_score = school_sites_school_name_score(
        school,
        site,
    )

    address_score = (
        school_sites_school_address_score(
            school,
            site,
        )
    )

    reference_score = (
        school_sites_reference_support_score(
            school,
            site,
        )
    )

    operator_score = (
        school_sites_operator_support_score(
            school,
            site,
        )
    )

    has_name = (
        name_score > 0
    )

    has_address = (
        address_score > 0
    )

    if (
        has_name
        and has_address
    ):

        base_score = (
            0.72 * name_score
            + 0.28 * address_score
        )

    elif has_name:

        # Il nome da solo non deve
        # raggiungere automaticamente 100.
        base_score = (
            0.92 * name_score
        )

    elif has_address:

        # L'indirizzo da solo può produrre
        # un candidato, ma mai un auto-match.
        base_score = (
            0.75 * address_score
        )

    else:

        base_score = 0.0

    # --------------------------------------------------------
    # SUPPORT SIGNALS
    # --------------------------------------------------------

    reference_bonus = 0.0

    if reference_score >= 95:
        reference_bonus = 4.0

    elif reference_score >= 85:
        reference_bonus = 2.0

    operator_bonus = 0.0

    if operator_score >= 95:
        operator_bonus = 2.0

    elif operator_score >= 85:
        operator_bonus = 1.0

    school_type_bonus = (
        school_sites_type_bonus(
            school.get(
                "school_type"
            ),
            site,
        )
    )

    (
        postcode_bonus,
        postcode_match,
    ) = school_sites_postcode_adjustment(
        school,
        site,
    )

    # --------------------------------------------------------
    # HARD CONFLICT
    # --------------------------------------------------------

    conflict_penalty = 0.0

    # Indirizzo molto forte ma nomi
    # chiaramente differenti:
    # non vogliamo che lo stesso civico
    # produca falsi auto-match.
    if (
        address_score >= 90
        and 0 < name_score < 55
    ):
        conflict_penalty = 15.0

    final_score = (
        base_score
        + reference_bonus
        + operator_bonus
        + school_type_bonus
        + postcode_bonus
        - conflict_penalty
    )

    final_score = max(
        0.0,
        min(
            100.0,
            final_score,
        ),
    )

    return {
        "score":
            final_score,

        "name_score":
            name_score,

        "address_score":
            address_score,

        "reference_score":
            reference_score,

        "operator_score":
            operator_score,

        "type_bonus":
            school_type_bonus,

        "postcode_match":
            postcode_match,

        "conflict_penalty":
            conflict_penalty,
    }


# ============================================================
# MATCH CLASSIFICATION
# ============================================================

def school_sites_classify_match(
    school,
    best,
    second,
    auto_threshold,
    review_threshold,
    min_margin,
):
    if best is None:
        return (
            "unresolved",
            "nessun candidato OSM",
        )

    overall = float(
        best["score"]
    )

    name_score = float(
        best["name_score"]
    )

    address_score = float(
        best["address_score"]
    )

    second_score = (
        float(
            second["score"]
        )
        if second
        else 0.0
    )

    margin = (
        overall
        - second_score
    )

    # Record che potrebbero non rappresentare
    # una sede fisica indipendente.
    if school.get(
        "requires_service_review"
    ):

        if overall >= review_threshold:

            return (
                "review",
                "record MIM speciale/"
                "aggregato: verifica "
                "sede fisica necessaria",
            )

        return (
            "unresolved",
            "record MIM speciale e "
            "match debole",
        )

    # --------------------------------------------------------
    # AUTO RULE A
    # Nome + indirizzo entrambi forti.
    # --------------------------------------------------------

    strong_name_address = (
        overall
        >= auto_threshold
        and name_score
        >= 90
        and address_score
        >= 85
        and margin
        >= min_margin
    )

    # --------------------------------------------------------
    # AUTO RULE B
    # Nome quasi esatto e molto univoco.
    # Utile quando OSM non ha indirizzo.
    # --------------------------------------------------------

    near_exact_unique_name = (
        overall
        >= auto_threshold
        and name_score
        >= 97
        and margin
        >= 15
        and (
            address_score == 0
            or address_score >= 60
        )
    )

    # --------------------------------------------------------
    # AUTO RULE C
    # Indirizzo quasi esatto + nome forte.
    # --------------------------------------------------------

    strong_address = (
        overall
        >= auto_threshold
        and address_score
        >= 95
        and name_score
        >= 85
        and margin
        >= 10
    )

    if (
        strong_name_address
        or near_exact_unique_name
        or strong_address
    ):

        return (
            "matched_auto",
            "match OSM forte e univoco",
        )

    if overall >= review_threshold:

        if margin < min_margin:

            reason = (
                "candidato plausibile ma "
                "ambiguo rispetto al secondo"
            )

        elif name_score < 85:

            reason = (
                "nome non sufficientemente "
                "forte per auto-match"
            )

        elif address_score < 70:

            reason = (
                "indirizzo assente/debole; "
                "richiesta verifica"
            )

        else:

            reason = (
                "match plausibile ma non "
                "soddisfa le regole AUTO"
            )

        return (
            "review",
            reason,
        )

    return (
        "unresolved",
        "score sotto soglia",
    )


# ============================================================
# MATCHING
# ============================================================

def school_sites_match_schools(
    schools,
    sites,
    auto_threshold,
    review_threshold,
    min_margin,
):
    location_rows = []
    ranking_rows = []

    site_records = (
        sites.to_dict(
            orient="records"
        )
    )

    for _, school_series in (
        schools.iterrows()
    ):

        school = (
            school_series.to_dict()
        )

        scored = []

        for site in site_records:

            score_data = (
                school_sites_score_match(
                    school,
                    site,
                )
            )

            scored.append(
                {
                    **site,
                    **score_data,
                }
            )

        scored.sort(
            key=lambda item:
                item["score"],
            reverse=True,
        )

        top_matches = (
            scored[:3]
        )

        best = (
            top_matches[0]
            if top_matches
            else None
        )

        second = (
            top_matches[1]
            if len(top_matches) > 1
            else None
        )

        (
            status,
            status_reason,
        ) = school_sites_classify_match(
            school=school,
            best=best,
            second=second,
            auto_threshold=(
                auto_threshold
            ),
            review_threshold=(
                review_threshold
            ),
            min_margin=(
                min_margin
            ),
        )

        for rank, candidate in (
            enumerate(
                top_matches,
                start=1,
            )
        ):

            ranking_rows.append(
                {
                    "school_code":
                        school[
                            "school_code"
                        ],

                    "school_name":
                        school[
                            "school_name"
                        ],

                    "rank":
                        rank,

                    "site_id":
                        candidate[
                            "site_id"
                        ],

                    "site_name":
                        candidate[
                            "site_name"
                        ],

                    "site_address":
                        candidate[
                            "site_address"
                        ],

                    "member_count":
                        candidate[
                            "member_count"
                        ],

                    "match_score":
                        candidate[
                            "score"
                        ],

                    "name_score":
                        candidate[
                            "name_score"
                        ],

                    "address_score":
                        candidate[
                            "address_score"
                        ],

                    "reference_score":
                        candidate[
                            "reference_score"
                        ],

                    "operator_score":
                        candidate[
                            "operator_score"
                        ],

                    "postcode_match":
                        candidate[
                            "postcode_match"
                        ],

                    "longitude":
                        candidate[
                            "longitude"
                        ],

                    "latitude":
                        candidate[
                            "latitude"
                        ],
                }
            )

        if best is not None:

            best_score = float(
                best["score"]
            )

            second_score = (
                float(
                    second["score"]
                )
                if second
                else 0.0
            )

            margin = (
                best_score
                - second_score
            )

        else:

            best_score = 0.0
            second_score = 0.0
            margin = 0.0

        accepted = (
            status
            == "matched_auto"
        )

        location_rows.append(
            {
                # ---------------------------------------------
                # MIM
                # ---------------------------------------------

                "school_year":
                    school[
                        "school_year"
                    ],

                "school_code":
                    school[
                        "school_code"
                    ],

                "school_name":
                    school[
                        "school_name"
                    ],

                "school_type":
                    school[
                        "school_type"
                    ],

                "school_ownership":
                    school[
                        "school_ownership"
                    ],

                "address":
                    school[
                        "address"
                    ],

                "postal_code":
                    school[
                        "postal_code"
                    ],

                "municipality_istat_code":
                    school[
                        "municipality_istat_code"
                    ],

                "requires_service_review":
                    school[
                        "requires_service_review"
                    ],

                "service_review_reason":
                    school[
                        "service_review_reason"
                    ],

                # ---------------------------------------------
                # MATCH
                # ---------------------------------------------

                "geocoding_status":
                    status,

                "status_reason":
                    status_reason,

                "geocoding_method":
                    (
                        "osm_site_name_address_v2"
                        if best
                        else None
                    ),

                "geocoding_score":
                    best_score,

                "second_best_score":
                    second_score,

                "score_margin":
                    margin,

                "name_score":
                    (
                        best[
                            "name_score"
                        ]
                        if best
                        else None
                    ),

                "address_score":
                    (
                        best[
                            "address_score"
                        ]
                        if best
                        else None
                    ),

                "reference_score":
                    (
                        best[
                            "reference_score"
                        ]
                        if best
                        else None
                    ),

                "operator_score":
                    (
                        best[
                            "operator_score"
                        ]
                        if best
                        else None
                    ),

                "postcode_match":
                    (
                        best[
                            "postcode_match"
                        ]
                        if best
                        else None
                    ),

                # ---------------------------------------------
                # BEST SITE
                # ---------------------------------------------

                "candidate_site_id":
                    (
                        best[
                            "site_id"
                        ]
                        if best
                        else None
                    ),

                "candidate_site_name":
                    (
                        best[
                            "site_name"
                        ]
                        if best
                        else None
                    ),

                "candidate_site_address":
                    (
                        best[
                            "site_address"
                        ]
                        if best
                        else None
                    ),

                "candidate_site_member_count":
                    (
                        best[
                            "member_count"
                        ]
                        if best
                        else None
                    ),

                "candidate_longitude":
                    (
                        best[
                            "longitude"
                        ]
                        if best
                        else None
                    ),

                "candidate_latitude":
                    (
                        best[
                            "latitude"
                        ]
                        if best
                        else None
                    ),

                # ---------------------------------------------
                # ACCEPTED GEOMETRY
                # ---------------------------------------------

                "geometry_accepted":
                    accepted,

                "geometry_source":
                    (
                        "OSM"
                        if accepted
                        else None
                    ),

                "longitude":
                    (
                        best[
                            "longitude"
                        ]
                        if accepted
                        else None
                    ),

                "latitude":
                    (
                        best[
                            "latitude"
                        ]
                        if accepted
                        else None
                    ),
            }
        )

    locations = pd.DataFrame(
        location_rows
    )

    rankings = pd.DataFrame(
        ranking_rows
    )

    # --------------------------------------------------------
    # SHARED PHYSICAL SITE
    # --------------------------------------------------------

    site_counts = (
        locations[
            "candidate_site_id"
        ]
        .dropna()
        .value_counts()
    )

    locations[
        "candidate_shared_count"
    ] = (
        locations[
            "candidate_site_id"
        ]
        .map(site_counts)
        .fillna(0)
        .astype(int)
    )

    locations[
        "shared_site_candidate"
    ] = (
        locations[
            "candidate_shared_count"
        ]
        > 1
    )

    return (
        locations,
        rankings,
    )


# ============================================================
# SAVE
# ============================================================

def school_sites_serialize_site_lists(
    sites,
):
    output = sites.copy()

    list_columns = [
        "primary_names",
        "core_primary_names",
        "operator_names",
        "addresses",
        "raw_addresses",
        "postcodes",
        "amenities",
        "buildings",
        "member_osm_keys",
    ]

    for column in list_columns:

        if column not in output.columns:
            continue

        output[column] = (
            output[column]
            .apply(
                school_sites_strict_json_dumps
            )
        )

    return output


def school_sites_save_outputs(
    locations,
    rankings,
    raw_candidates,
    sites,
    municipality_code,
    school_year,
):
    mim_directory = (
        school_sites_PROCESSED_MIM_DIR
        / municipality_code
    )

    mim_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    osm_directory = (
        school_sites_PROCESSED_OSM_DIR
        / municipality_code
    )

    osm_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    locations_path = (
        mim_directory
        / (
            f"school_locations_"
            f"{school_year}_v2.parquet"
        )
    )

    rankings_path = (
        mim_directory
        / (
            f"school_match_candidates_"
            f"{school_year}_v2.parquet"
        )
    )

    raw_candidates_path = (
        osm_directory
        / "school_candidates_raw.parquet"
    )

    sites_path = (
        osm_directory
        / "school_sites.parquet"
    )

    locations.to_parquet(
        locations_path,
        index=False,
    )

    rankings.to_parquet(
        rankings_path,
        index=False,
    )

    school_sites_serialize_site_lists(
        raw_candidates
    ).to_parquet(
        raw_candidates_path,
        index=False,
    )

    school_sites_serialize_site_lists(
        sites
    ).to_parquet(
        sites_path,
        index=False,
    )

    print(
        "\n=== OUTPUT V2 ==="
    )

    print(
        f"✓ {locations_path}"
    )

    print(
        f"✓ {rankings_path}"
    )

    print(
        f"✓ {raw_candidates_path}"
    )

    print(
        f"✓ {sites_path}"
    )


# ============================================================
# SUMMARY
# ============================================================

def school_sites_print_match_table(
    dataframe,
):
    if dataframe.empty:

        print(
            "Nessun record."
        )

        return

    print(
        dataframe[
            [
                "school_code",
                "school_name",
                "address",
                "candidate_site_name",
                "candidate_site_address",
                "geocoding_score",
                "name_score",
                "address_score",
                "score_margin",
                "status_reason",
            ]
        ]
        .sort_values(
            "geocoding_score",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )


def school_sites_print_summary(
    locations,
    raw_candidates,
    sites,
):
    print(
        "\n===================================="
    )

    print(
        " SCHOOL GEOLOCATION V2 COMPLETATA"
    )

    print(
        "===================================="
    )

    print(
        f"Scuole MIM: {len(locations)}"
    )

    print(
        f"Candidati OSM raw: "
        f"{len(raw_candidates)}"
    )

    print(
        f"Siti OSM consolidati: "
        f"{len(sites)}"
    )

    print(
        "\nStato matching V2:"
    )

    print(
        locations[
            "geocoding_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    accepted_geometry_count = (
        locations[
            "geometry_accepted"
        ]
        .sum()
    )

    service_review_count = (
        locations[
            "requires_service_review"
        ]
        .sum()
    )

    shared_site_count = (
        locations[
            "shared_site_candidate"
        ]
        .sum()
    )

    print(
        "\nGeometry accettate "
        "automaticamente: "
        f"{accepted_geometry_count}"
    )

    print(
        "Record MIM che richiedono "
        "service review: "
        f"{service_review_count}"
    )

    print(
        "Record con sito candidato "
        "condiviso: "
        f"{shared_site_count}"
    )

    print(
        "\n=== MATCH AUTOMATICI V2 ==="
    )

    automatic_matches = locations[
        locations[
            "geocoding_status"
        ]
        == "matched_auto"
    ]

    school_sites_print_match_table(
        automatic_matches
    )

    print(
        "\n=== DA REVISIONARE V2 ==="
    )

    review_matches = locations[
        locations[
            "geocoding_status"
        ]
        == "review"
    ]

    school_sites_print_match_table(
        review_matches
    )

    print(
        "\n=== NON RISOLTI V2 ==="
    )

    unresolved_matches = locations[
        locations[
            "geocoding_status"
        ]
        == "unresolved"
    ]

    school_sites_print_match_table(
        unresolved_matches
    )


# ============================================================
# MAIN
# ============================================================

def school_sites_main():
    args = school_sites_parse_args()

    print(
        "\n===================================="
    )

    print(
        " MIM / OSM SCHOOL GEOLOCATION V2"
    )

    print(
        "===================================="
    )

    engine = (
        school_sites_get_database_engine()
    )

    # --------------------------------------------------------
    # MUNICIPALITY
    # --------------------------------------------------------

    municipality = (
        school_sites_load_municipality(
            engine,
            args.municipality_code,
        )
    )

    print(
        "\n=== COMUNE ==="
    )

    print(
        f"Nome: "
        f"{municipality['name']}"
    )

    print(
        "Codice ISTAT: "
        f"{municipality['istat_code']}"
    )

    # --------------------------------------------------------
    # MIM
    # --------------------------------------------------------

    schools = (
        school_sites_load_mim_registry(
            args.municipality_code,
            args.school_year,
            municipality["name"],
        )
    )

    # --------------------------------------------------------
    # OSM
    # --------------------------------------------------------

    raw_features = (
        school_sites_load_or_download_osm_features(
            boundary=(
                municipality[
                    "geometry"
                ]
            ),

            municipality_code=(
                args.municipality_code
            ),

            refresh=(
                args.refresh_osm
            ),
        )
    )

    raw_candidates = (
        school_sites_build_osm_candidates(
            raw_features,
            municipality["name"],
        )
    )

    # --------------------------------------------------------
    # CONSOLIDATE PHYSICAL SITES
    # --------------------------------------------------------

    sites = (
        school_sites_consolidate_osm_candidates(
            raw_candidates
        )
    )

    # --------------------------------------------------------
    # MATCH
    # --------------------------------------------------------

    print(
        "\n=== MATCHING MIM ↔ OSM SITES V2 ==="
    )

    locations, rankings = (
        school_sites_match_schools(
            schools=schools,
            sites=sites,

            auto_threshold=(
                args.auto_threshold
            ),

            review_threshold=(
                args.review_threshold
            ),

            min_margin=(
                args.min_margin
            ),
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    school_sites_save_outputs(
        locations=locations,
        rankings=rankings,
        raw_candidates=raw_candidates,
        sites=sites,
        municipality_code=(
            args.municipality_code
        ),
        school_year=(
            args.school_year
        ),
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    school_sites_print_summary(
        locations,
        raw_candidates,
        sites,
    )




# ============================================================================
# PHYSICAL BUILDING GEOCODING
# ============================================================================

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv
from shapely import wkt
from shapely.geometry import Point
from sqlalchemy import create_engine, text


# ============================================================
# PATHS / CONSTANTS
# ============================================================

building_geocode_ROOT = Path(__file__).resolve().parents[2]

building_geocode_PROCESSED_MIM_DIR = (
    building_geocode_ROOT / "data" / "processed" / "mim"
)

building_geocode_RAW_GEOCODING_DIR = (
    building_geocode_ROOT
    / "data"
    / "raw"
    / "mim"
    / "geocoding"
    / "nominatim_buildings"
)

building_geocode_DEFAULT_BUILDING_YEAR = "202425"

building_geocode_NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/search"
)

building_geocode_REQUEST_DELAY_SECONDS = 1.1


# ============================================================
# CLI
# ============================================================

def building_geocode_parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Geocoding degli edifici scolastici fisici MIM "
            "tramite indirizzo ufficiale e boundary ISTAT."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--building-year",
        default=building_geocode_DEFAULT_BUILDING_YEAR,
        help="Anno edilizia MIM, default 202425.",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignora i risultati presenti nella cache.",
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "municipality-code deve avere esattamente 6 cifre."
        )

    return args


# ============================================================
# DATABASE / MUNICIPALITY
# ============================================================

def building_geocode_get_database_engine():
    load_dotenv(
        building_geocode_ROOT / ".env"
    )

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL non definito."
        )

    return create_engine(
        database_url
    )


def building_geocode_load_municipality(
    engine,
    municipality_code,
):
    query = text("""
        SELECT
            istat_code,
            name,
            ST_AsText(geometry) AS geometry_wkt

        FROM municipality

        WHERE istat_code = :istat_code;
    """)

    with engine.connect() as connection:
        row = connection.execute(
            query,
            {
                "istat_code":
                    municipality_code
            },
        ).mappings().first()

    if row is None:
        raise RuntimeError(
            "Comune non presente in PostGIS."
        )

    geometry = wkt.loads(
        row["geometry_wkt"]
    )

    (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ) = geometry.bounds

    return {
        "istat_code":
            row["istat_code"],

        "name":
            row["name"],

        "geometry":
            geometry,

        "viewbox": (
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        ),
    }


def building_geocode_viewbox_string(
    municipality,
):
    (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ) = municipality[
        "viewbox"
    ]

    # Nominatim: left, top, right, bottom
    return (
        f"{min_lon},"
        f"{max_lat},"
        f"{max_lon},"
        f"{min_lat}"
    )


# ============================================================
# INPUT
# ============================================================

def building_geocode_load_buildings(
    municipality_code,
    building_year,
):
    municipality_dir = (
        building_geocode_PROCESSED_MIM_DIR
        / municipality_code
    )

    # Canonical filename used by the older pipeline.
    canonical_path = (
        municipality_dir
        / f"physical_school_buildings_{building_year}.parquet"
    )

    # Generalized pipeline output. The school year may differ from
    # the building year, therefore discover it rather than hard-code it.
    generalized_paths = sorted(
        municipality_dir.glob(
            f"physical_school_buildings_{building_year}_from_schools_*.parquet"
        )
    )

    if canonical_path.exists():
        path = canonical_path
    elif len(generalized_paths) == 1:
        path = generalized_paths[0]
    elif len(generalized_paths) > 1:
        raise RuntimeError(
            "Più layer edifici generalizzati compatibili trovati: "
            + ", ".join(str(p) for p in generalized_paths)
        )
    else:
        raise FileNotFoundError(
            "Layer edifici non trovato. Cercati: "
            f"{canonical_path} oppure "
            f"physical_school_buildings_{building_year}_from_schools_*.parquet"
        )

    buildings = (
        pd.read_parquet(
            path
        )
        .copy()
    )

    # Schema compatibility between the generalized building pipeline
    # and the geocoding pipeline.
    if (
        "postal_code" not in buildings.columns
        and "building_postal_code" in buildings.columns
    ):
        buildings["postal_code"] = buildings["building_postal_code"]

    required_columns = {
        "building_code",
        "building_municipality_name",
        "official_building_address",
        "postal_code",
    }

    missing = (
        required_columns
        - set(
            buildings.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Colonne mancanti: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if buildings[
        "building_code"
    ].duplicated().any():
        raise RuntimeError(
            "building_code duplicati nel layer fisico."
        )

    return buildings


# ============================================================
# CACHE
# ============================================================

def building_geocode_cache_path():
    building_geocode_RAW_GEOCODING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        building_geocode_RAW_GEOCODING_DIR
        / "cache.json"
    )


def building_geocode_load_cache():
    path = building_geocode_cache_path()

    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def building_geocode_save_cache(cache):
    path = building_geocode_cache_path()

    temporary = (
        path.with_suffix(
            ".json.tmp"
        )
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cache,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary.replace(
        path
    )


# ============================================================
# NOMINATIM CLIENT
# ============================================================

class building_geocode_NominatimClient:

    def __init__(
        self,
        cache,
        refresh=False,
    ):
        self.cache = cache
        self.refresh = refresh

        load_dotenv(
            building_geocode_ROOT / ".env"
        )

        self.user_agent = os.getenv(
            "NOMINATIM_USER_AGENT",
            (
                "urban-digital-twin/1.0 "
                "(academic research)"
            ),
        )

        self.email = os.getenv(
            "NOMINATIM_EMAIL"
        )

        self.last_request_time = 0.0

    def search(
        self,
        params,
    ):
        params = dict(
            params
        )

        params.update(
            {
                "format":
                    "jsonv2",

                "limit":
                    5,

                "countrycodes":
                    "it",

                "addressdetails":
                    1,
            }
        )

        if self.email:
            params[
                "email"
            ] = self.email

        cache_key = json.dumps(
            params,
            sort_keys=True,
            ensure_ascii=False,
        )

        if (
            cache_key in self.cache
            and not self.refresh
        ):
            return self.cache[
                cache_key
            ]["results"]

        elapsed = (
            time.time()
            - self.last_request_time
        )

        if (
            elapsed
            < building_geocode_REQUEST_DELAY_SECONDS
        ):
            time.sleep(
                building_geocode_REQUEST_DELAY_SECONDS
                - elapsed
            )

        response = requests.get(
            building_geocode_NOMINATIM_URL,
            params=params,
            headers={
                "User-Agent":
                    self.user_agent,
            },
            timeout=60,
        )

        self.last_request_time = (
            time.time()
        )

        response.raise_for_status()

        results = response.json()

        self.cache[
            cache_key
        ] = {
            "retrieved_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "params":
                params,

            "results":
                results,
        }

        building_geocode_save_cache(
            self.cache
        )

        return results


# ============================================================
# SPATIAL HELPERS
# ============================================================

def building_geocode_build_metric_boundary(
    geometry,
):
    boundary = gpd.GeoDataFrame(
        {
            "geometry": [
                geometry
            ]
        },
        crs=4326,
    )

    metric_crs = (
        boundary.estimate_utm_crs()
    )

    if metric_crs is None:
        raise RuntimeError(
            "Impossibile stimare CRS metrico."
        )

    boundary_metric = (
        boundary.to_crs(
            metric_crs
        )
        .geometry
        .iloc[0]
    )

    return (
        metric_crs,
        boundary_metric,
    )


def building_geocode_spatial_metrics(
    latitude,
    longitude,
    municipality_geometry,
    metric_crs,
    boundary_metric,
):
    point = Point(
        float(longitude),
        float(latitude),
    )

    inside = bool(
        municipality_geometry.covers(
            point
        )
    )

    point_metric = (
        gpd.GeoSeries(
            [point],
            crs=4326,
        )
        .to_crs(
            metric_crs
        )
        .iloc[0]
    )

    if inside:
        distance_to_municipality_m = 0.0
    else:
        distance_to_municipality_m = float(
            point_metric.distance(
                boundary_metric
            )
        )

    return {
        "inside_municipality":
            inside,

        "distance_to_municipality_m":
            distance_to_municipality_m,
    }


# ============================================================
# QUERY / RESULT
# ============================================================

def building_geocode_build_structured_params(
    building,
    municipality,
):
    return {
        "street":
            building[
                "official_building_address"
            ],

        "city":
            municipality["name"],

        "postalcode":
            (
                building[
                    "postal_code"
                ]
                if pd.notna(
                    building[
                        "postal_code"
                    ]
                )
                else None
            ),

        "viewbox":
            building_geocode_viewbox_string(
                municipality
            ),

        "bounded":
            1,
    }


def building_geocode_clean_params(params):
    return {
        key:
            value
        for (
            key,
            value,
        ) in params.items()
        if (
            value is not None
            and str(value).strip()
        )
    }


def building_geocode_result_road_address(
    result,
):
    address = (
        result.get(
            "address"
        )
        or {}
    )

    road = (
        address.get("road")
        or address.get("pedestrian")
        or address.get("residential")
        or address.get("place")
    )

    house_number = (
        address.get(
            "house_number"
        )
    )

    if (
        road
        and house_number
    ):
        return (
            f"{road} {house_number}"
        )

    return road


def building_geocode_score_candidate(
    building,
    result,
    municipality,
    metric_crs,
    boundary_metric,
):
    latitude = float(
        result["lat"]
    )

    longitude = float(
        result["lon"]
    )

    spatial = building_geocode_spatial_metrics(
        latitude=latitude,
        longitude=longitude,
        municipality_geometry=(
            municipality[
                "geometry"
            ]
        ),
        metric_crs=metric_crs,
        boundary_metric=(
            boundary_metric
        ),
    )

    address = (
        result.get(
            "address"
        )
        or {}
    )

    returned_postcode = (
        address.get(
            "postcode"
        )
    )

    expected_postcode = (
        building.get(
            "postal_code"
        )
    )

    postcode_match = None

    if (
        pd.notna(
            expected_postcode
        )
        and expected_postcode
        and returned_postcode
    ):
        postcode_match = (
            str(
                expected_postcode
            ).strip()
            == str(
                returned_postcode
            ).strip()
        )

    importance = float(
        result.get(
            "importance",
            0.0,
        )
        or 0.0
    )

    score = (
        70.0
        if spatial[
            "inside_municipality"
        ]
        else 0.0
    )

    if postcode_match is True:
        score += 15.0

    elif postcode_match is False:
        score -= 10.0

    result_type = (
        result.get(
            "type"
        )
        or ""
    )

    result_class = (
        result.get(
            "class"
        )
        or ""
    )

    if (
        result_class
        == "amenity"
        and result_type
        in {
            "school",
            "kindergarten",
            "college",
        }
    ):
        score += 10.0

    score += (
        5.0
        * min(
            importance,
            1.0,
        )
    )

    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )

    return {
        "latitude":
            latitude,

        "longitude":
            longitude,

        "display_name":
            result.get(
                "display_name"
            ),

        "result_class":
            result_class,

        "result_type":
            result_type,

        "result_address":
            building_geocode_result_road_address(
                result
            ),

        "result_postcode":
            returned_postcode,

        "postcode_match":
            postcode_match,

        "geocoder_score":
            score,

        **spatial,
    }


def building_geocode_geocode_building(
    building,
    municipality,
    client,
    metric_crs,
    boundary_metric,
):
    # Gli edifici ufficialmente appartenenti ad altro comune
    # restano nel dataset, ma non vengono forzati dentro Matera.
    building_municipality = (
        str(
            building[
                "building_municipality_name"
            ]
        )
        .strip()
        .upper()
    )

    target_municipality = (
        municipality[
            "name"
        ]
        .strip()
        .upper()
    )

    if (
        building_municipality
        != target_municipality
    ):
        return {
            "geocoding_status":
                "outside_target_municipality",

            "latitude":
                None,

            "longitude":
                None,

            "display_name":
                None,

            "result_class":
                None,

            "result_type":
                None,

            "result_address":
                None,

            "result_postcode":
                None,

            "postcode_match":
                None,

            "geocoder_score":
                None,

            "inside_municipality":
                False,

            "distance_to_municipality_m":
                None,
        }

    params = building_geocode_clean_params(
        building_geocode_build_structured_params(
            building,
            municipality,
        )
    )

    results = client.search(
        params
    )

    candidates = []

    for result in results:
        candidates.append(
            building_geocode_score_candidate(
                building=building,
                result=result,
                municipality=municipality,
                metric_crs=metric_crs,
                boundary_metric=(
                    boundary_metric
                ),
            )
        )

    if not candidates:
        # Secondo tentativo free-form sempre bounded.
        query_parts = [
            building[
                "official_building_address"
            ],
            municipality[
                "name"
            ],
            "Basilicata",
            "Italia",
        ]

        freeform_params = {
            "q":
                ", ".join(
                    str(value)
                    for value in query_parts
                    if (
                        value is not None
                        and str(value).strip()
                    )
                ),

            "viewbox":
                building_geocode_viewbox_string(
                    municipality
                ),

            "bounded":
                1,
        }

        results = client.search(
            freeform_params
        )

        for result in results:
            candidates.append(
                building_geocode_score_candidate(
                    building=building,
                    result=result,
                    municipality=municipality,
                    metric_crs=metric_crs,
                    boundary_metric=(
                        boundary_metric
                    ),
                )
            )

    if not candidates:
        return {
            "geocoding_status":
                "unresolved",

            "latitude":
                None,

            "longitude":
                None,

            "display_name":
                None,

            "result_class":
                None,

            "result_type":
                None,

            "result_address":
                None,

            "result_postcode":
                None,

            "postcode_match":
                None,

            "geocoder_score":
                None,

            "inside_municipality":
                None,

            "distance_to_municipality_m":
                None,
        }

    candidates.sort(
        key=lambda item:
            item[
                "geocoder_score"
            ],
        reverse=True,
    )

    best = candidates[0]

    if (
        best[
            "inside_municipality"
        ]
        and best[
            "geocoder_score"
        ]
        >= 80
    ):
        status = (
            "accepted_candidate"
        )

    elif best[
        "inside_municipality"
    ]:
        status = "review"

    else:
        status = (
            "outside_candidate"
        )

    return {
        "geocoding_status":
            status,

        **best,
    }


# ============================================================
# PROCESS
# ============================================================

def building_geocode_build_dataset(
    buildings,
    municipality,
    client,
):
    (
        metric_crs,
        boundary_metric,
    ) = building_geocode_build_metric_boundary(
        municipality[
            "geometry"
        ]
    )

    rows = []

    total = len(
        buildings
    )

    for position, (
        _,
        row,
    ) in enumerate(
        buildings.iterrows(),
        start=1,
    ):
        building = (
            row.to_dict()
        )

        print(
            f"[{position}/{total}] "
            f"{building['building_code']} "
            f"{building['official_building_address']}"
        )

        geocoded = building_geocode_geocode_building(
            building=building,
            municipality=municipality,
            client=client,
            metric_crs=metric_crs,
            boundary_metric=(
                boundary_metric
            ),
        )

        rows.append(
            {
                **building,
                **geocoded,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SAVE / SUMMARY
# ============================================================

def building_geocode_save_output(
    dataframe,
    municipality_code,
    building_year,
):
    path = (
        building_geocode_PROCESSED_MIM_DIR
        / municipality_code
        / (
            "physical_school_buildings_"
            f"{building_year}_geocoded.parquet"
        )
    )

    dataframe.to_parquet(
        path,
        index=False,
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ {path}"
    )


def building_geocode_print_summary(
    dataframe,
):
    print(
        "\n===================================="
    )

    print(
        " SCHOOL BUILDING GEOCODING COMPLETATO"
    )

    print(
        "===================================="
    )

    print(
        f"Edifici totali: {len(dataframe)}"
    )

    print(
        "\nGeocoding status:"
    )

    print(
        dataframe[
            "geocoding_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    accepted = (
        dataframe[
            "geocoding_status"
        ]
        == "accepted_candidate"
    )

    print(
        "\nEdifici con coordinate candidate accettate: "
        f"{int(accepted.sum())}"
    )

    print(
        "\n=== DA REVISIONARE / NON RISOLTI ==="
    )

    review = dataframe[
        ~accepted
    ]

    if review.empty:
        print(
            "Nessuno."
        )

    else:
        print(
            review[
                [
                    "building_code",
                    "building_municipality_name",
                    "official_building_address",
                    "geocoding_status",
                    "display_name",
                    "geocoder_score",
                ]
            ]
            .to_string(
                index=False
            )
        )


def building_geocode_main():
    args = building_geocode_parse_args()

    print(
        "\n===================================="
    )

    print(
        " MIM PHYSICAL SCHOOL BUILDING GEOCODING"
    )

    print(
        "===================================="
    )

    engine = (
        building_geocode_get_database_engine()
    )

    municipality = (
        building_geocode_load_municipality(
            engine,
            args.municipality_code,
        )
    )

    buildings = building_geocode_load_buildings(
        args.municipality_code,
        args.building_year,
    )

    cache = building_geocode_load_cache()

    print(
        "\n=== INPUT ==="
    )

    print(
        f"Comune target: {municipality['name']}"
    )

    print(
        f"Edifici fisici MIM: {len(buildings)}"
    )

    print(
        f"Cache esistente: {len(cache)}"
    )

    client = building_geocode_NominatimClient(
        cache=cache,
        refresh=args.refresh,
    )

    dataset = building_geocode_build_dataset(
        buildings=buildings,
        municipality=municipality,
        client=client,
    )

    building_geocode_save_output(
        dataset,
        args.municipality_code,
        args.building_year,
    )

    building_geocode_print_summary(
        dataset
    )




# ============================================================================
# PHYSICAL BUILDING / OSM MATCHING
# ============================================================================

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# PATHS / CONSTANTS
# ============================================================

building_match_ROOT = Path(__file__).resolve().parents[2]

building_match_PROCESSED_MIM_DIR = (
    building_match_ROOT / "data" / "processed" / "mim"
)

building_match_PROCESSED_OSM_DIR = (
    building_match_ROOT / "data" / "processed" / "osm"
)

building_match_DEFAULT_SCHOOL_YEAR = "202627"
building_match_DEFAULT_BUILDING_YEAR = "202425"

# Conservative thresholds.
building_match_AUTO_MIN_SCORE = 80.0
building_match_AUTO_MIN_MARGIN = 8.0
building_match_REVIEW_MIN_SCORE = 60.0

# Spatial evidence is secondary: it must support textual evidence,
# never replace it.
building_match_STRONG_DISTANCE_M = 120.0
building_match_PLAUSIBLE_DISTANCE_M = 250.0
building_match_CONFLICT_DISTANCE_M = 1000.0


# ============================================================
# CLI
# ============================================================

def building_match_parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Matching conservativo tra edifici scolastici MIM "
            "e siti scolastici OSM consolidati."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--school-year",
        default=building_match_DEFAULT_SCHOOL_YEAR,
        help="Anno scolastico anagrafica corrente, es. 202627.",
    )

    parser.add_argument(
        "--building-year",
        default=building_match_DEFAULT_BUILDING_YEAR,
        help="Anno di riferimento edilizia MIM, es. 202425.",
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "municipality-code deve avere esattamente 6 cifre."
        )

    return args


# ============================================================
# GENERIC HELPERS
# ============================================================

def building_match_is_missing(value):
    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

        if hasattr(result, "item"):
            return bool(result.item())

    except Exception:
        pass

    return False


def building_match_clean_text(value):
    if building_match_is_missing(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.upper() in {
        "NAN",
        "NONE",
        "NULL",
        "N/A",
        "NA",
        "N.D.",
        "ND",
        "NON DISPONIBILE",
        "-",
    }:
        return None

    return value


def building_match_normalize_text(value):
    value = building_match_clean_text(value)

    if value is None:
        return None

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = value.upper()

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value or None


def building_match_normalize_address(value):
    value = building_match_normalize_text(value)

    if value is None:
        return None

    replacements = {
        "PIAZZA GIOVANNI SEMERIA": "PIAZZA SEMERIA",
        "P ZZA": "PIAZZA",
        "P ZA": "PIAZZA",
        "V LE": "VIALE",
        "C DA": "CONTRADA",
        "LOCALITA CONTRADA": "CONTRADA",
        "LOCALITA": "",
        "S N C": "",
        "SNC": "",
        "S N": "",
    }

    for old, new in replacements.items():
        value = value.replace(
            old,
            new,
        )

    # "0" in some official addresses is a missing-house-number marker.
    value = re.sub(
        r"\s+0$",
        "",
        value,
    )

    value = re.sub(
        r"\s+S N$",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value or None


def building_match_similarity(
    left,
    right,
):
    if not left or not right:
        return 0.0

    return float(
        max(
            fuzz.ratio(
                left,
                right,
            ),
            fuzz.token_sort_ratio(
                left,
                right,
            ),
        )
    )


def building_match_haversine_m(
    lon1,
    lat1,
    lon2,
    lat2,
):
    values = [
        lon1,
        lat1,
        lon2,
        lat2,
    ]

    if any(
        value is None
        for value in values
    ):
        return None

    try:
        values = [
            float(value)
            for value in values
        ]
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not all(
        math.isfinite(value)
        for value in values
    ):
        return None

    (
        lon1,
        lat1,
        lon2,
        lat2,
    ) = values

    radius_m = 6371008.8

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(
        lat2 - lat1
    )

    delta_lambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            delta_phi / 2.0
        ) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(
            delta_lambda / 2.0
        ) ** 2
    )

    c = 2.0 * math.atan2(
        math.sqrt(a),
        math.sqrt(
            max(
                0.0,
                1.0 - a,
            )
        ),
    )

    return radius_m * c


def building_match_parse_possible_list(value):
    if building_match_is_missing(value):
        return []

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            building_match_clean_text(item)
            for item in value
            if building_match_clean_text(item)
        ]

    value = str(value).strip()

    if not value:
        return []

    # Several Silver datasets serialize list-valued fields as JSON.
    if (
        value.startswith("[")
        and value.endswith("]")
    ):
        try:
            decoded = json.loads(value)

            if isinstance(decoded, list):
                return [
                    building_match_clean_text(item)
                    for item in decoded
                    if building_match_clean_text(item)
                ]
        except Exception:
            pass

    return [
        value
    ]


def building_match_first_existing_column(
    dataframe,
    candidates,
    required=False,
):
    for column in candidates:
        if column in dataframe.columns:
            return column

    if required:
        raise RuntimeError(
            "Nessuna delle colonne attese è presente: "
            + ", ".join(candidates)
        )

    return None


# ============================================================
# LOAD INPUTS
# ============================================================

def building_match_load_inputs(
    municipality_code,
    school_year,
    building_year,
):
    mim_directory = (
        building_match_PROCESSED_MIM_DIR
        / municipality_code
    )

    osm_directory = (
        building_match_PROCESSED_OSM_DIR
        / municipality_code
    )

    buildings_path = (
        mim_directory
        / (
            "physical_school_buildings_"
            f"{building_year}_geocoded.parquet"
        )
    )

    links_path = (
        mim_directory
        / (
            "school_building_links_"
            f"{school_year}_from_"
            f"{building_year}.parquet"
        )
    )

    sites_path = (
        osm_directory
        / "school_sites.parquet"
    )

    for path in [
        buildings_path,
        links_path,
        sites_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"File richiesto non trovato: {path}"
            )

    buildings = pd.read_parquet(
        buildings_path
    )

    links = pd.read_parquet(
        links_path
    )

    # Compatibility with the canonical schema produced by
    # mim_school_buildings_generalized.py.
    # Keep the legacy alias internally because downstream matching
    # code still refers to CODICEEDIFICIO.
    if (
        "CODICEEDIFICIO" not in links.columns
        and "building_code" in links.columns
    ):
        links["CODICEEDIFICIO"] = links["building_code"]

    sites = pd.read_parquet(
        sites_path
    )

    if (
        "building_code"
        not in buildings.columns
    ):
        raise RuntimeError(
            "building_code mancante nel layer edifici."
        )

    if buildings[
        "building_code"
    ].duplicated().any():
        raise RuntimeError(
            "building_code duplicati nel layer edifici."
        )

    if (
        "school_code"
        not in links.columns
        or "CODICEEDIFICIO"
        not in links.columns
    ):
        raise RuntimeError(
            "Il dataset school_building_links non contiene "
            "school_code / CODICEEDIFICIO."
        )

    return (
        buildings,
        links,
        sites,
    )


# ============================================================
# LINKED SCHOOL NAMES
# ============================================================

def building_match_build_linked_school_evidence(
    links,
):
    name_columns = [
        column
        for column in [
            "school_name",
            "reference_institute_name",
            "institute_name",
        ]
        if column in links.columns
    ]

    grouped = {}

    for (
        building_code,
        group,
    ) in links.groupby(
        "CODICEEDIFICIO",
        dropna=True,
    ):
        school_codes = sorted(
            {
                str(value)
                for value
                in group[
                    "school_code"
                ]
                .dropna()
                .tolist()
            }
        )

        names = []

        for column in name_columns:
            for value in group[
                column
            ].tolist():
                cleaned = building_match_clean_text(
                    value
                )

                if (
                    cleaned
                    and cleaned
                    not in names
                ):
                    names.append(
                        cleaned
                    )

        grouped[
            str(building_code)
        ] = {
            "linked_school_codes":
                school_codes,

            "linked_school_names":
                names,
        }

    return grouped


# ============================================================
# OSM SITE ADAPTER
# ============================================================

def building_match_prepare_osm_sites(
    sites,
):
    site_id_column = (
        building_match_first_existing_column(
            sites,
            [
                "site_id",
                "osm_site_id",
                "candidate_site_id",
            ],
            required=True,
        )
    )

    longitude_column = (
        building_match_first_existing_column(
            sites,
            [
                "longitude",
                "site_longitude",
                "candidate_longitude",
                "lon",
            ],
            required=True,
        )
    )

    latitude_column = (
        building_match_first_existing_column(
            sites,
            [
                "latitude",
                "site_latitude",
                "candidate_latitude",
                "lat",
            ],
            required=True,
        )
    )

    address_columns = [
        column
        for column in [
            "site_address",
            "address",
            "osm_address",
            "candidate_site_address",
            "raw_address",
        ]
        if column in sites.columns
    ]

    scalar_name_columns = [
        column
        for column in [
            "site_name",
            "name",
            "candidate_site_name",
            "primary_name",
            "official_name",
        ]
        if column in sites.columns
    ]

    list_name_columns = [
        column
        for column in [
            "primary_names",
            "core_primary_names",
            "operator_names",
        ]
        if column in sites.columns
    ]

    rows = []

    for _, row in sites.iterrows():
        site_names = []

        for column in scalar_name_columns:
            value = building_match_clean_text(
                row.get(column)
            )

            if (
                value
                and value
                not in site_names
            ):
                site_names.append(
                    value
                )

        for column in list_name_columns:
            for value in (
                building_match_parse_possible_list(
                    row.get(column)
                )
            ):
                if (
                    value
                    and value
                    not in site_names
                ):
                    site_names.append(
                        value
                    )

        addresses = []

        for column in address_columns:
            values = (
                building_match_parse_possible_list(
                    row.get(column)
                )
            )

            for value in values:
                if (
                    value
                    and value
                    not in addresses
                ):
                    addresses.append(
                        value
                    )

        site_name = (
            site_names[0]
            if site_names
            else None
        )

        site_address = (
            addresses[0]
            if addresses
            else None
        )

        rows.append(
            {
                "site_id":
                    str(
                        row[
                            site_id_column
                        ]
                    ),

                "site_name":
                    site_name,

                "site_names":
                    site_names,

                "site_address":
                    site_address,

                "site_addresses":
                    addresses,

                "longitude":
                    row[
                        longitude_column
                    ],

                "latitude":
                    row[
                        latitude_column
                    ],
            }
        )

    output = pd.DataFrame(
        rows
    )

    if output[
        "site_id"
    ].duplicated().any():
        raise RuntimeError(
            "site_id OSM duplicati dopo adattamento."
        )

    return output


# ============================================================
# SCORING
# ============================================================

def building_match_best_name_similarity(
    linked_names,
    osm_names,
):
    best_score = 0.0
    best_left = None
    best_right = None

    normalized_linked = [
        (
            value,
            building_match_normalize_text(value),
        )
        for value in linked_names
        if building_match_normalize_text(value)
    ]

    normalized_osm = [
        (
            value,
            building_match_normalize_text(value),
        )
        for value in osm_names
        if building_match_normalize_text(value)
    ]

    for (
        original_left,
        normalized_left,
    ) in normalized_linked:
        for (
            original_right,
            normalized_right,
        ) in normalized_osm:
            score = building_match_similarity(
                normalized_left,
                normalized_right,
            )

            if score > best_score:
                best_score = score
                best_left = original_left
                best_right = original_right

    return (
        best_score,
        best_left,
        best_right,
    )


def building_match_best_address_similarity(
    official_address,
    osm_addresses,
):
    official_normalized = (
        building_match_normalize_address(
            official_address
        )
    )

    if not official_normalized:
        return (
            0.0,
            None,
        )

    best_score = 0.0
    best_address = None

    for address in osm_addresses:
        candidate = (
            building_match_normalize_address(
                address
            )
        )

        score = building_match_similarity(
            official_normalized,
            candidate,
        )

        if score > best_score:
            best_score = score
            best_address = address

    return (
        best_score,
        best_address,
    )


def building_match_score_site(
    building,
    linked_names,
    site,
):
    (
        name_score,
        matched_school_name,
        matched_osm_name,
    ) = building_match_best_name_similarity(
        linked_names,
        site[
            "site_names"
        ],
    )

    (
        address_score,
        matched_osm_address,
    ) = building_match_best_address_similarity(
        building.get(
            "official_building_address"
        ),
        site[
            "site_addresses"
        ],
    )

    geocoder_status = building_match_clean_text(
        building.get(
            "geocoding_status"
        )
    )

    geocoder_lon = (
        building.get(
            "longitude"
        )
    )

    geocoder_lat = (
        building.get(
            "latitude"
        )
    )

    distance_m = None

    if geocoder_status in {
        "accepted_candidate",
        "review",
    }:
        distance_m = building_match_haversine_m(
            geocoder_lon,
            geocoder_lat,
            site[
                "longitude"
            ],
            site[
                "latitude"
            ],
        )

    # Building-level matching: the official MIM building address is
    # the primary evidence. School names are secondary because the
    # administrative MIM name can differ substantially from the OSM POI.
    score = (
        0.60 * address_score
        + 0.28 * name_score
    )

    spatial_bonus = 0.0

    if distance_m is not None:
        if distance_m <= 60:
            spatial_bonus = 10.0

        elif (
            distance_m
            <= building_match_STRONG_DISTANCE_M
        ):
            spatial_bonus = 7.0

        elif (
            distance_m
            <= building_match_PLAUSIBLE_DISTANCE_M
        ):
            spatial_bonus = 3.0

        elif (
            distance_m
            >= building_match_CONFLICT_DISTANCE_M
        ):
            spatial_bonus = -10.0

    score += spatial_bonus

    # Near-exact official address is very strong physical evidence.
    if address_score >= 98:
        score += 5.0

    # Exact names remain useful as an additional confirmation.
    if name_score >= 97:
        score += 2.0

    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )

    return {
        "site_id":
            site[
                "site_id"
            ],

        "site_name":
            site[
                "site_name"
            ],

        "site_address":
            site[
                "site_address"
            ],

        "site_longitude":
            site[
                "longitude"
            ],

        "site_latitude":
            site[
                "latitude"
            ],

        "name_score":
            name_score,

        "address_score":
            address_score,

        "osm_geocoder_distance_m":
            distance_m,

        "matched_school_name":
            matched_school_name,

        "matched_osm_name":
            matched_osm_name,

        "matched_osm_address":
            matched_osm_address,

        "spatial_bonus":
            spatial_bonus,

        "match_score":
            score,
    }


# ============================================================
# STATUS RULES
# ============================================================

def building_match_classify_match(
    best,
    margin,
):
    if best is None:
        return (
            "unresolved",
            "nessun candidato OSM",
        )

    score = best[
        "match_score"
    ]

    name_score = best[
        "name_score"
    ]

    address_score = best[
        "address_score"
    ]

    distance_m = best[
        "osm_geocoder_distance_m"
    ]

    # Rule A: official address is nearly exact and either the independent
    # geocoder confirms the same place or the linked-school name is good.
    # This correctly handles cases such as Marconi, Dante Alighieri and
    # Loperfido-Olivetti without requiring identical administrative names.
    if (
        score >= 82.0
        and address_score >= 97.0
        and margin >= building_match_AUTO_MIN_MARGIN
        and (
            (
                distance_m is not None
                and distance_m <= building_match_STRONG_DISTANCE_M
            )
            or name_score >= 60.0
        )
    ):
        return (
            "matched_auto",
            "indirizzo ufficiale quasi esatto con conferma spaziale o nominale",
        )

    # Rule B: strong address, good linked-school name and unique candidate.
    if (
        score >= building_match_AUTO_MIN_SCORE
        and address_score >= 92.0
        and name_score >= 55.0
        and margin >= building_match_AUTO_MIN_MARGIN
        and (
            distance_m is None
            or distance_m <= building_match_PLAUSIBLE_DISTANCE_M
        )
    ):
        return (
            "matched_auto",
            "indirizzo forte, nome coerente e candidato univoco",
        )

    # Rule C: very strong school name plus independent spatial convergence.
    if (
        score >= building_match_AUTO_MIN_SCORE
        and name_score >= 92.0
        and distance_m is not None
        and distance_m <= building_match_STRONG_DISTANCE_M
        and margin >= building_match_AUTO_MIN_MARGIN
    ):
        return (
            "matched_auto",
            "nome molto forte e convergenza con geocoder edificio",
        )

    if score >= building_match_REVIEW_MIN_SCORE:
        if (
            margin
            < building_match_AUTO_MIN_MARGIN
        ):
            return (
                "review",
                "candidato plausibile ma ambiguo rispetto al secondo",
            )

        if (
            distance_m is not None
            and distance_m
            >= building_match_CONFLICT_DISTANCE_M
        ):
            return (
                "review",
                "testo plausibile ma conflitto spaziale con geocoder edificio",
            )

        return (
            "review",
            "evidenza plausibile ma insufficiente per auto-match",
        )

    return (
        "unresolved",
        "score sotto soglia",
    )


# ============================================================
# MATCHING
# ============================================================

def building_match_match_buildings(
    buildings,
    links,
    sites,
):
    linked_evidence = (
        building_match_build_linked_school_evidence(
            links
        )
    )

    prepared_sites = (
        building_match_prepare_osm_sites(
            sites
        )
    )

    # Determine the target municipality from the canonical
    # school-building links rather than using municipality-specific
    # hard-coded values.
    if "municipality_name" not in links.columns:
        raise RuntimeError(
            "municipality_name mancante nel dataset school_building_links."
        )

    target_municipalities = {
        (
            building_match_clean_text(value)
            or ""
        ).upper()
        for value in links["municipality_name"].dropna()
        if building_match_clean_text(value)
    }

    if len(target_municipalities) != 1:
        raise RuntimeError(
            "Impossibile determinare univocamente il comune target "
            "da school_building_links: "
            + ", ".join(sorted(target_municipalities))
        )

    target_municipality_name = next(
        iter(target_municipalities)
    )

    result_rows = []
    ranking_rows = []

    target_buildings = buildings.copy()

    total = len(
        target_buildings
    )

    for position, (
        _,
        building_row,
    ) in enumerate(
        target_buildings.iterrows(),
        start=1,
    ):
        building = (
            building_row.to_dict()
        )

        building_code = str(
            building[
                "building_code"
            ]
        )

        building_municipality = (
            building_match_clean_text(
                building.get(
                    "building_municipality_name"
                )
            )
            or ""
        ).upper()

        evidence = linked_evidence.get(
            building_code,
            {
                "linked_school_codes":
                    [],
                "linked_school_names":
                    [],
            },
        )

        linked_school_codes = (
            evidence[
                "linked_school_codes"
            ]
        )

        linked_school_names = (
            evidence[
                "linked_school_names"
            ]
        )

        print(
            f"[{position}/{total}] "
            f"{building_code} "
            f"{building.get('official_building_address')}"
        )

        if (
            building_municipality
            != target_municipality_name
        ):
            result_rows.append(
                {
                    "building_code":
                        building_code,

                    "building_municipality_name":
                        building.get(
                            "building_municipality_name"
                        ),

                    "official_building_address":
                        building.get(
                            "official_building_address"
                        ),

                    "linked_school_codes":
                        json.dumps(
                            linked_school_codes,
                            ensure_ascii=False,
                        ),

                    "linked_school_names":
                        json.dumps(
                            linked_school_names,
                            ensure_ascii=False,
                        ),

                    "osm_match_status":
                        "outside_target_municipality",

                    "status_reason":
                        (
                            "edificio ufficialmente appartenente a comune "
                            f"diverso dal target ({target_municipality_name})"
                        ),

                    "candidate_site_id":
                        None,

                    "candidate_site_name":
                        None,

                    "candidate_site_address":
                        None,

                    "candidate_longitude":
                        None,

                    "candidate_latitude":
                        None,

                    "match_score":
                        None,

                    "name_score":
                        None,

                    "address_score":
                        None,

                    "score_margin":
                        None,

                    "osm_geocoder_distance_m":
                        None,

                    "geometry_accepted":
                        False,

                    "longitude":
                        None,

                    "latitude":
                        None,
                }
            )

            continue

        candidates = []

        for _, site_row in (
            prepared_sites.iterrows()
        ):
            candidate = building_match_score_site(
                building=building,
                linked_names=(
                    linked_school_names
                ),
                site=(
                    site_row.to_dict()
                ),
            )

            candidates.append(
                candidate
            )

        candidates.sort(
            key=lambda item:
                item[
                    "match_score"
                ],
            reverse=True,
        )

        top_candidates = (
            candidates[:5]
        )

        for rank, candidate in enumerate(
            top_candidates,
            start=1,
        ):
            ranking_rows.append(
                {
                    "building_code":
                        building_code,

                    "rank":
                        rank,

                    **candidate,
                }
            )

        if not top_candidates:
            best = None
            margin = 0.0

        else:
            best = top_candidates[0]

            second_score = (
                top_candidates[1][
                    "match_score"
                ]
                if len(
                    top_candidates
                ) > 1
                else 0.0
            )

            margin = (
                best[
                    "match_score"
                ]
                - second_score
            )

        (
            status,
            reason,
        ) = building_match_classify_match(
            best,
            margin,
        )

        accepted = (
            status
            == "matched_auto"
        )

        result_rows.append(
            {
                "building_code":
                    building_code,

                "building_municipality_name":
                    building.get(
                        "building_municipality_name"
                    ),

                "official_building_address":
                    building.get(
                        "official_building_address"
                    ),

                "building_geocoding_status":
                    building.get(
                        "geocoding_status"
                    ),

                "building_geocoder_longitude":
                    building.get(
                        "longitude"
                    ),

                "building_geocoder_latitude":
                    building.get(
                        "latitude"
                    ),

                "linked_school_codes":
                    json.dumps(
                        linked_school_codes,
                        ensure_ascii=False,
                    ),

                "linked_school_names":
                    json.dumps(
                        linked_school_names,
                        ensure_ascii=False,
                    ),

                "osm_match_status":
                    status,

                "status_reason":
                    reason,

                "candidate_site_id":
                    (
                        best[
                            "site_id"
                        ]
                        if best
                        else None
                    ),

                "candidate_site_name":
                    (
                        best[
                            "site_name"
                        ]
                        if best
                        else None
                    ),

                "candidate_site_address":
                    (
                        best[
                            "site_address"
                        ]
                        if best
                        else None
                    ),

                "candidate_longitude":
                    (
                        best[
                            "site_longitude"
                        ]
                        if best
                        else None
                    ),

                "candidate_latitude":
                    (
                        best[
                            "site_latitude"
                        ]
                        if best
                        else None
                    ),

                "matched_school_name":
                    (
                        best[
                            "matched_school_name"
                        ]
                        if best
                        else None
                    ),

                "matched_osm_name":
                    (
                        best[
                            "matched_osm_name"
                        ]
                        if best
                        else None
                    ),

                "matched_osm_address":
                    (
                        best[
                            "matched_osm_address"
                        ]
                        if best
                        else None
                    ),

                "match_score":
                    (
                        best[
                            "match_score"
                        ]
                        if best
                        else None
                    ),

                "name_score":
                    (
                        best[
                            "name_score"
                        ]
                        if best
                        else None
                    ),

                "address_score":
                    (
                        best[
                            "address_score"
                        ]
                        if best
                        else None
                    ),

                "score_margin":
                    margin,

                "osm_geocoder_distance_m":
                    (
                        best[
                            "osm_geocoder_distance_m"
                        ]
                        if best
                        else None
                    ),

                "geometry_accepted":
                    accepted,

                "longitude":
                    (
                        best[
                            "site_longitude"
                        ]
                        if accepted
                        else None
                    ),

                "latitude":
                    (
                        best[
                            "site_latitude"
                        ]
                        if accepted
                        else None
                    ),
            }
        )

    return (
        pd.DataFrame(
            result_rows
        ),
        pd.DataFrame(
            ranking_rows
        ),
    )


# ============================================================
# SAVE / SUMMARY
# ============================================================

def building_match_save_outputs(
    matches,
    rankings,
    municipality_code,
    building_year,
):
    directory = (
        building_match_PROCESSED_MIM_DIR
        / municipality_code
    )

    matches_path = (
        directory
        / (
            "school_building_osm_matches_"
            f"{building_year}_v2.parquet"
        )
    )

    ranking_path = (
        directory
        / (
            "school_building_osm_candidates_"
            f"{building_year}_v2.parquet"
        )
    )

    matches.to_parquet(
        matches_path,
        index=False,
    )

    rankings.to_parquet(
        ranking_path,
        index=False,
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ {matches_path}"
    )

    print(
        f"✓ {ranking_path}"
    )


def building_match_print_summary(
    matches,
):
    print(
        "\n===================================="
    )

    print(
        " MIM BUILDING ↔ OSM MATCH V2 COMPLETATO"
    )

    print(
        "===================================="
    )

    print(
        f"Edifici totali: {len(matches)}"
    )

    print(
        "\nStato matching:"
    )

    print(
        matches[
            "osm_match_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    accepted = (
        matches[
            "osm_match_status"
        ]
        == "matched_auto"
    )

    print(
        "\nMatch automatici accettati: "
        f"{int(accepted.sum())}"
    )

    print(
        "\n=== MATCH AUTOMATICI ==="
    )

    automatic = (
        matches[
            accepted
        ]
    )

    if automatic.empty:
        print(
            "Nessuno."
        )

    else:
        print(
            automatic[
                [
                    "building_code",
                    "official_building_address",
                    "candidate_site_name",
                    "candidate_site_address",
                    "match_score",
                    "name_score",
                    "address_score",
                    "score_margin",
                    "osm_geocoder_distance_m",
                ]
            ]
            .sort_values(
                "match_score",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n=== DA REVISIONARE ==="
    )

    review = (
        matches[
            matches[
                "osm_match_status"
            ]
            == "review"
        ]
    )

    if review.empty:
        print(
            "Nessuno."
        )

    else:
        print(
            review[
                [
                    "building_code",
                    "official_building_address",
                    "candidate_site_name",
                    "candidate_site_address",
                    "match_score",
                    "name_score",
                    "address_score",
                    "score_margin",
                    "osm_geocoder_distance_m",
                    "status_reason",
                ]
            ]
            .sort_values(
                "match_score",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n=== NON RISOLTI ==="
    )

    unresolved = (
        matches[
            matches[
                "osm_match_status"
            ]
            == "unresolved"
        ]
    )

    if unresolved.empty:
        print(
            "Nessuno."
        )

    else:
        print(
            unresolved[
                [
                    "building_code",
                    "official_building_address",
                    "candidate_site_name",
                    "candidate_site_address",
                    "match_score",
                    "name_score",
                    "address_score",
                    "score_margin",
                    "status_reason",
                ]
            ]
            .sort_values(
                "match_score",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )

    shared = (
        matches[
            matches[
                "candidate_site_id"
            ]
            .notna()
        ]
        .groupby(
            "candidate_site_id"
        )[
            "building_code"
        ]
        .nunique()
    )

    shared = shared[
        shared > 1
    ]

    print(
        "\nSiti OSM candidati condivisi da più edifici: "
        f"{len(shared)}"
    )


# ============================================================
# MAIN
# ============================================================

def building_match_main():
    args = building_match_parse_args()

    print(
        "\n===================================="
    )

    print(
        " MIM PHYSICAL BUILDING ↔ OSM MATCH V2"
    )

    print(
        "===================================="
    )

    (
        buildings,
        links,
        sites,
    ) = building_match_load_inputs(
        municipality_code=(
            args.municipality_code
        ),
        school_year=(
            args.school_year
        ),
        building_year=(
            args.building_year
        ),
    )

    print(
        "\n=== INPUT ==="
    )

    print(
        f"Edifici MIM: {len(buildings)}"
    )

    print(
        f"Relazioni scuola-edificio: {len(links)}"
    )

    print(
        f"Siti OSM consolidati: {len(sites)}"
    )

    (
        matches,
        rankings,
    ) = building_match_match_buildings(
        buildings=buildings,
        links=links,
        sites=sites,
    )

    building_match_save_outputs(
        matches=matches,
        rankings=rankings,
        municipality_code=(
            args.municipality_code
        ),
        building_year=(
            args.building_year
        ),
    )

    building_match_print_summary(
        matches
    )




# ============================================================================
# PUBLIC CLI / ORCHESTRATION
# ============================================================================

_PUBLIC_STEPS = (
    "prepare",
    "school-sites",
    "building-geocode",
    "building-match",
)


def parse_args():
    parser = _cli_argparse.ArgumentParser(
        description=(
            "Canonical Education matching/geolocation pipeline. "
            "Consolida discovery OSM, geocoding edifici e matching "
            "edificio-OSM."
        )
    )

    parser.add_argument(
        "--step",
        choices=_PUBLIC_STEPS,
        default="prepare",
        help=(
            "prepare esegue school-sites + building-geocode + "
            "building-match; gli altri valori eseguono il singolo step."
        ),
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre.",
    )

    parser.add_argument(
        "--school-year",
        default=_DEFAULT_CONFIG.school_year,
        help="Anno scolastico MIM.",
    )

    parser.add_argument(
        "--building-year",
        default=_DEFAULT_CONFIG.building_year,
        help="Anno Anagrafe edilizia scolastica MIM.",
    )

    parser.add_argument(
        "--refresh-osm",
        action="store_true",
        help="Forza un nuovo snapshot OSM delle strutture scolastiche.",
    )

    parser.add_argument(
        "--refresh-geocoder",
        action="store_true",
        help="Ignora la cache del geocoder edifici.",
    )

    parser.add_argument(
        "--auto-threshold",
        type=float,
        default=85.0,
        help="Soglia AUTO del school-site matcher.",
    )

    parser.add_argument(
        "--review-threshold",
        type=float,
        default=65.0,
        help="Soglia REVIEW del school-site matcher.",
    )

    parser.add_argument(
        "--min-margin",
        type=float,
        default=8.0,
        help="Margine minimo del school-site matcher.",
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    args.school_year = str(
        args.school_year
    ).strip()

    args.building_year = str(
        args.building_year
    ).strip()

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "municipality-code deve avere esattamente 6 cifre."
        )

    return args


def _run_legacy_cli(main_function, argv):
    """
    Esegue una delle implementazioni consolidate usando la sua CLI originale.

    Questo adapter è intenzionale durante il refactor regression-safe: mantiene
    invariata la logica validata mentre rimuove gli entry point operativi
    separati. Verrà eliminato quando le funzioni interne saranno normalizzate.
    """
    previous_argv = list(_sys.argv)

    try:
        _sys.argv = [
            f"{__file__}:{main_function.__name__}",
            *argv,
        ]
        main_function()
    finally:
        _sys.argv = previous_argv


def run_school_sites(args):
    argv = [
        "--municipality-code",
        args.municipality_code,
        "--school-year",
        args.school_year,
        "--auto-threshold",
        str(args.auto_threshold),
        "--review-threshold",
        str(args.review_threshold),
        "--min-margin",
        str(args.min_margin),
    ]

    if args.refresh_osm:
        argv.append(
            "--refresh-osm"
        )

    _run_legacy_cli(
        school_sites_main,
        argv,
    )


def run_building_geocode(args):
    argv = [
        "--municipality-code",
        args.municipality_code,
        "--building-year",
        args.building_year,
    ]

    if args.refresh_geocoder:
        argv.append(
            "--refresh"
        )

    _run_legacy_cli(
        building_geocode_main,
        argv,
    )


def run_building_match(args):
    _run_legacy_cli(
        building_match_main,
        [
            "--municipality-code",
            args.municipality_code,
            "--school-year",
            args.school_year,
            "--building-year",
            args.building_year,
        ],
    )


def main():
    args = parse_args()

    if args.step in {
        "prepare",
        "school-sites",
    }:
        run_school_sites(
            args
        )

    if args.step in {
        "prepare",
        "building-geocode",
    }:
        run_building_geocode(
            args
        )

    if args.step in {
        "prepare",
        "building-match",
    }:
        run_building_match(
            args
        )


if __name__ == "__main__":
    main()
