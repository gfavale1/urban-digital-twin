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

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = (
    ROOT
    / "data"
    / "processed"
    / "mim"
)

RAW_OSM_DIR = (
    ROOT
    / "data"
    / "raw"
    / "osm"
)

PROCESSED_OSM_DIR = (
    ROOT
    / "data"
    / "processed"
    / "osm"
)

DEFAULT_SCHOOL_YEAR = "202627"

DEFAULT_AUTO_THRESHOLD = 85.0
DEFAULT_REVIEW_THRESHOLD = 65.0
DEFAULT_MIN_MARGIN = 8.0

# Distanza massima usata esclusivamente
# per identificare possibili duplicati OSM.
DUPLICATE_MAX_DISTANCE_M = 35.0

MISSING_TEXT_VALUES = {
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

ROLE_REVIEW_PATTERNS = {
    "ISTITUTO COMPRENSIVO",
    "ISTITUTO SUPERIORE",
    "CENTRO TERRITORIALE",
    "CONVITTO ANNESSO",
}

NAME_REVIEW_PATTERNS = {
    "CORSO SERALE",
    "CASA CIRCONDARIALE",
    "OSPEDALIERA",
    "CPIA",
    "CTP",
}


# ============================================================
# CLI
# ============================================================

def parse_args():
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
        default=DEFAULT_SCHOOL_YEAR,
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
        default=DEFAULT_AUTO_THRESHOLD,
    )

    parser.add_argument(
        "--review-threshold",
        type=float,
        default=DEFAULT_REVIEW_THRESHOLD,
    )

    parser.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN,
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

def get_database_engine():
    load_dotenv(
        ROOT / ".env"
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


def load_municipality(
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

def is_missing(value):
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


def clean_text_value(value):
    if is_missing(value):
        return None

    value = (
        str(value)
        .strip()
    )

    if (
        value.upper()
        in MISSING_TEXT_VALUES
    ):
        return None

    return value


def unique_nonempty(values):
    output = []

    for value in values:

        if value is None:
            continue

        if value not in output:
            output.append(value)

    return output


def json_safe(value):
    if value is None:
        return None

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key):
                json_safe(item)
            for key, item
            in value.items()
        }

    if is_missing(value):
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
            return json_safe(
                value.item()
            )

        except Exception:
            pass

    return str(value)


def strict_json_dumps(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    value = clean_text_value(
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


def normalize_address(value):
    value = normalize_text(
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


def normalize_school_name(
    value,
    municipality_name=None,
):
    value = normalize_text(
        value
    )

    if value is None:
        return None

    if municipality_name:

        municipality_normalized = (
            normalize_text(
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


def school_name_core(
    value,
    municipality_name=None,
):
    value = normalize_school_name(
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

def registry_path(
    municipality_code,
    school_year,
):
    return (
        PROCESSED_MIM_DIR
        / municipality_code
        / (
            f"schools_registry_"
            f"{school_year}.parquet"
        )
    )


def contains_review_pattern(
    value,
    patterns,
):
    value = normalize_text(
        value
    )

    if not value:
        return False

    return any(
        pattern in value
        for pattern in patterns
    )


def build_review_reason(
    school_type,
    school_name,
):
    reasons = []

    if contains_review_pattern(
        school_type,
        ROLE_REVIEW_PATTERNS,
    ):
        reasons.append(
            "tipologia aggregata/"
            "organizzativa"
        )

    if contains_review_pattern(
        school_name,
        NAME_REVIEW_PATTERNS,
    ):
        reasons.append(
            "programma/sede speciale"
        )

    if not reasons:
        return None

    return "; ".join(
        reasons
    )


def load_mim_registry(
    municipality_code,
    school_year,
    municipality_name,
):
    path = registry_path(
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
                    clean_text_value
                )
            )

    schools[
        "school_name_normalized"
    ] = (
        schools["school_name"]
        .apply(
            lambda value:
                normalize_school_name(
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
                school_name_core(
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
                normalize_school_name(
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
                school_name_core(
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
            normalize_address
        )
    )

    schools[
        "service_review_reason"
    ] = schools.apply(
        lambda row:
            build_review_reason(
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

def osm_school_snapshot_dir(
    municipality_code,
):
    directory = (
        RAW_OSM_DIR
        / municipality_code
        / "school_features"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def latest_osm_snapshot(
    municipality_code,
):
    directory = (
        osm_school_snapshot_dir(
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


def configure_osmnx():
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 180


def download_osm_school_features(
    boundary,
):
    configure_osmnx()

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

            value = json_safe(
                row[column]
            )

            if value is not None:
                tags_dict[
                    column
                ] = value

        source_tags.append(
            strict_json_dumps(
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


def save_osm_snapshot(
    features,
    municipality_code,
):
    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    path = (
        osm_school_snapshot_dir(
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


def load_or_download_osm_features(
    boundary,
    municipality_code,
    refresh=False,
):
    snapshot = latest_osm_snapshot(
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
        download_osm_school_features(
            boundary
        )
    )

    save_osm_snapshot(
        features,
        municipality_code,
    )

    return features


# ============================================================
# OSM CANDIDATES
# ============================================================

def parse_source_tags(value):
    if value is None:
        return {}

    return json.loads(
        value
    )


def first_nonempty(
    tags,
    keys,
):
    for key in keys:

        value = clean_text_value(
            tags.get(key)
        )

        if value is not None:
            return value

    return None


def feature_point(geometry):
    if geometry is None:
        return None

    if geometry.is_empty:
        return None

    if geometry.geom_type == "Point":
        return geometry

    return geometry.representative_point()


def candidate_addresses(tags):
    addresses = []

    full_address = first_nonempty(
        tags,
        [
            "addr:full",
        ],
    )

    if full_address:
        addresses.append(
            full_address
        )

    street = first_nonempty(
        tags,
        [
            "addr:street",
            "addr:place",
        ],
    )

    number = first_nonempty(
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

    return unique_nonempty(
        addresses
    )


def build_osm_candidates(
    raw_features,
    municipality_name,
):
    rows = []

    for row in (
        raw_features.itertuples()
    ):

        tags = parse_source_tags(
            row.source_tags
        )

        point = feature_point(
            row.geometry
        )

        if point is None:
            continue

        name = first_nonempty(
            tags,
            [
                "name",
            ],
        )

        official_name = first_nonempty(
            tags,
            [
                "official_name",
            ],
        )

        alt_name = first_nonempty(
            tags,
            [
                "alt_name",
                "short_name",
            ],
        )

        operator = first_nonempty(
            tags,
            [
                "operator",
            ],
        )

        raw_primary_names = (
            unique_nonempty(
                [
                    name,
                    official_name,
                    alt_name,
                ]
            )
        )

        normalized_primary_names = [
            normalize_school_name(
                value,
                municipality_name,
            )
            for value
            in raw_primary_names
        ]

        normalized_primary_names = (
            unique_nonempty(
                normalized_primary_names
            )
        )

        core_primary_names = [
            school_name_core(
                value,
                municipality_name,
            )
            for value
            in raw_primary_names
        ]

        core_primary_names = (
            unique_nonempty(
                core_primary_names
            )
        )

        operator_names = (
            unique_nonempty(
                [
                    normalize_school_name(
                        operator,
                        municipality_name,
                    )
                ]
            )
        )

        raw_addresses = (
            candidate_addresses(
                tags
            )
        )

        normalized_addresses = (
            unique_nonempty(
                [
                    normalize_address(
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
                    clean_text_value(
                        tags.get("amenity")
                    ),

                "osm_building":
                    clean_text_value(
                        tags.get("building")
                    ),

                "osm_postal_code":
                    clean_text_value(
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

def name_similarity(
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


def address_similarity(
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


def best_list_similarity(
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

class UnionFind:

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


def should_merge_candidates(
    left,
    right,
    distance_m,
):
    if (
        distance_m
        > DUPLICATE_MAX_DISTANCE_M
    ):
        return False

    name_score = (
        best_list_similarity(
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
            name_similarity,
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
        normalize_text(
            left["osm_amenity"]
        ),
        normalize_text(
            left["osm_building"]
        ),
    }

    school_tags_right = {
        normalize_text(
            right["osm_amenity"]
        ),
        normalize_text(
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


def candidate_completeness(
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


def make_site_id(
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


def consolidate_osm_candidates(
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

    union_find = UnionFind(
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

            if should_merge_candidates(
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
            key=candidate_completeness,
        )

        osm_keys = [
            member["osm_key"]
            for member in members
        ]

        primary_names = (
            unique_nonempty(
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
            unique_nonempty(
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
            unique_nonempty(
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
            unique_nonempty(
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
            unique_nonempty(
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
            unique_nonempty(
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
            unique_nonempty(
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
            unique_nonempty(
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
                    make_site_id(
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

def school_name_score(
    school,
    site,
):
    school_names = (
        unique_nonempty(
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

    return best_list_similarity(
        school_names,
        site_names,
        name_similarity,
    )


def school_address_score(
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

    return best_list_similarity(
        [school_address],
        site["addresses"],
        address_similarity,
    )


def reference_support_score(
    school,
    site,
):
    reference_names = (
        unique_nonempty(
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

    return best_list_similarity(
        reference_names,
        site_names,
        name_similarity,
    )


def operator_support_score(
    school,
    site,
):
    school_names = (
        unique_nonempty(
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

    return best_list_similarity(
        school_names,
        site["operator_names"],
        name_similarity,
    )


def type_bonus(
    school_type,
    site,
):
    school_type = (
        normalize_text(
            school_type
        )
        or ""
    )

    amenities = {
        normalize_text(value)
        for value
        in site["amenities"]
        if value
    }

    buildings = {
        normalize_text(value)
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


def postcode_adjustment(
    school,
    site,
):
    mim_postcode = clean_text_value(
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


def score_match(
    school,
    site,
):
    name_score = school_name_score(
        school,
        site,
    )

    address_score = (
        school_address_score(
            school,
            site,
        )
    )

    reference_score = (
        reference_support_score(
            school,
            site,
        )
    )

    operator_score = (
        operator_support_score(
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
        type_bonus(
            school.get(
                "school_type"
            ),
            site,
        )
    )

    (
        postcode_bonus,
        postcode_match,
    ) = postcode_adjustment(
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

def classify_match(
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

def match_schools(
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
                score_match(
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
        ) = classify_match(
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

def serialize_site_lists(
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
                strict_json_dumps
            )
        )

    return output


def save_outputs(
    locations,
    rankings,
    raw_candidates,
    sites,
    municipality_code,
    school_year,
):
    mim_directory = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    mim_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    osm_directory = (
        PROCESSED_OSM_DIR
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

    serialize_site_lists(
        raw_candidates
    ).to_parquet(
        raw_candidates_path,
        index=False,
    )

    serialize_site_lists(
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

def print_match_table(
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


def print_summary(
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

    print_match_table(
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

    print_match_table(
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

    print_match_table(
        unresolved_matches
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

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
        get_database_engine()
    )

    # --------------------------------------------------------
    # MUNICIPALITY
    # --------------------------------------------------------

    municipality = (
        load_municipality(
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
        load_mim_registry(
            args.municipality_code,
            args.school_year,
            municipality["name"],
        )
    )

    # --------------------------------------------------------
    # OSM
    # --------------------------------------------------------

    raw_features = (
        load_or_download_osm_features(
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
        build_osm_candidates(
            raw_features,
            municipality["name"],
        )
    )

    # --------------------------------------------------------
    # CONSOLIDATE PHYSICAL SITES
    # --------------------------------------------------------

    sites = (
        consolidate_osm_candidates(
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
        match_schools(
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

    save_outputs(
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

    print_summary(
        locations,
        raw_candidates,
        sites,
    )


if __name__ == "__main__":
    main()