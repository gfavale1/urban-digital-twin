import argparse
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz
from shapely import wkt
from shapely.geometry import Point
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

RAW_GEOCODING_DIR = (
    ROOT
    / "data"
    / "raw"
    / "mim"
    / "geocoding"
    / "nominatim"
)

DEFAULT_SCHOOL_YEAR = "202627"

NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/search"
)

# Public Nominatim: keep the batch single-threaded and rate-limited.
REQUEST_DELAY_SECONDS = 1.1

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
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Geocoding V2 degli indirizzi scolastici MIM "
            "tramite Nominatim, con ricerca locale bounded "
            "e cache persistente."
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
            "Anno scolastico nel formato YYYYyy. "
            "Default: 202627."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Ignora i risultati già presenti nella cache "
            "per le nuove query V2."
        ),
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

    args.school_year = (
        str(args.school_year)
        .strip()
    )

    if (
        not args.school_year.isdigit()
        or len(args.school_year) != 6
    ):
        raise ValueError(
            "school-year deve avere formato "
            "YYYYyy, es. 202627."
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
            "DATABASE_URL non definito nel file .env."
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
            province_code,
            region_code,
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

    (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    ) = geometry.bounds

    return {
        "id":
            row["id"],

        "istat_code":
            row["istat_code"],

        "name":
            row["name"],

        "province_code":
            row["province_code"],

        "region_code":
            row["region_code"],

        "geometry":
            geometry,

        "viewbox": (
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        ),
    }


# ============================================================
# TEXT HELPERS
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


def clean_text(value):
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


def normalize_text(value):
    value = clean_text(
        value
    )

    if value is None:
        return None

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(
            character
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
        "C DA": "CONTRADA",
        "SNC": "",
        "S N C": "",
    }

    for old, new in replacements.items():
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


def query_address(value):
    """
    Versione dell'indirizzo pensata per la query Nominatim.
    Conserva una forma human-readable e corregge abbreviazioni
    frequenti nel dataset MIM.
    """

    value = clean_text(
        value
    )

    if value is None:
        return None

    value = value.strip()

    # MORO28 -> MORO 28
    value = re.sub(
        r"(?<=[A-Za-z])(?=\d)",
        " ",
        value,
    )

    value = re.sub(
        r"(?<=\d)(?=[A-Za-z])",
        " ",
        value,
    )

    replacements = {
        r"\bC/DA\b": "Contrada",
        r"\bC\.DA\b": "Contrada",
        r"\bP\.ZA\b": "Piazza",
        r"\bPZA\b": "Piazza",
        r"\bV\.LE\b": "Viale",
        r"\bS\.N\.C\.\b": "",
        r"\bSNC\b": "",
    }

    for pattern, replacement in replacements.items():
        value = re.sub(
            pattern,
            replacement,
            value,
            flags=re.IGNORECASE,
        )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value or None


def valid_postal_code(value):
    value = clean_text(
        value
    )

    if value is None:
        return None

    match = re.search(
        r"\b\d{5}\b",
        value,
    )

    if match:
        return match.group(0)

    return None


def similarity(
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


# ============================================================
# INPUT DATA
# ============================================================

def load_registry(
    municipality_code,
    school_year,
):
    path = (
        PROCESSED_MIM_DIR
        / municipality_code
        / (
            "schools_registry_"
            f"{school_year}.parquet"
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Registro MIM non trovato: {path}"
        )

    registry = pd.read_parquet(
        path
    )

    if registry["school_code"].duplicated().any():
        raise RuntimeError(
            "Il registro MIM contiene school_code duplicati."
        )

    return registry


def load_v2_locations(
    municipality_code,
    school_year,
):
    path = (
        PROCESSED_MIM_DIR
        / municipality_code
        / (
            "school_locations_"
            f"{school_year}_v2.parquet"
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Matching V2 non trovato: {path}"
        )

    locations = pd.read_parquet(
        path
    )

    if locations["school_code"].duplicated().any():
        raise RuntimeError(
            "Il dataset V2 contiene school_code duplicati."
        )

    return locations


# ============================================================
# CACHE
# ============================================================

def cache_path():
    RAW_GEOCODING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        RAW_GEOCODING_DIR
        / "cache.json"
    )


def load_cache():
    path = cache_path()

    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            "Formato cache Nominatim inatteso."
        )

    return data


def save_cache(cache):
    path = cache_path()

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
# QUERY BUILDING V2
# ============================================================

def viewbox_string(
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

    # Nominatim viewbox:
    # left, top, right, bottom.
    return (
        f"{min_lon},"
        f"{max_lat},"
        f"{max_lon},"
        f"{min_lat}"
    )


def build_local_structured_params(
    school,
    municipality,
):
    address = query_address(
        school.get(
            "address"
        )
    )

    if address is None:
        return None

    params = {
        "street":
            address,

        "city":
            municipality["name"],

        "viewbox":
            viewbox_string(
                municipality
            ),

        "bounded":
            1,
    }

    region_name = clean_text(
        school.get(
            "region_name"
        )
    )

    if region_name:
        params[
            "state"
        ] = region_name

    postal_code = (
        valid_postal_code(
            school.get(
                "postal_code"
            )
        )
    )

    if postal_code:
        params[
            "postalcode"
        ] = postal_code

    return params


def build_local_address_params(
    school,
    municipality,
):
    """
    Secondo tentativo locale: indirizzo free-form,
    senza il nome della scuola, per evitare che una
    denominazione MIM molto diversa da OSM penalizzi la ricerca.
    """

    address = query_address(
        school.get(
            "address"
        )
    )

    if address is None:
        return None

    region_name = (
        clean_text(
            school.get(
                "region_name"
            )
        )
        or ""
    )

    parts = [
        address,
        municipality["name"],
        region_name,
        "Italia",
    ]

    return {
        "q":
            ", ".join(
                part
                for part in parts
                if part
            ),

        "viewbox":
            viewbox_string(
                municipality
            ),

        "bounded":
            1,
    }


def build_local_school_params(
    school,
    municipality,
):
    """
    Terzo tentativo locale: nome scuola + indirizzo.
    Utile quando Nominatim conosce direttamente il POI.
    """

    address = query_address(
        school.get(
            "address"
        )
    )

    school_name = clean_text(
        school.get(
            "school_name"
        )
    )

    if (
        address is None
        and school_name is None
    ):
        return None

    region_name = (
        clean_text(
            school.get(
                "region_name"
            )
        )
        or ""
    )

    parts = [
        school_name,
        address,
        municipality["name"],
        region_name,
        "Italia",
    ]

    return {
        "q":
            ", ".join(
                part
                for part in parts
                if part
            ),

        "viewbox":
            viewbox_string(
                municipality
            ),

        "bounded":
            1,
    }


def build_external_params(
    school,
):
    """
    Ultimo tentativo, non bounded.
    È esclusivamente diagnostico: serve per individuare
    record che potrebbero realmente riferirsi a un altro comune.
    """

    address = query_address(
        school.get(
            "address"
        )
    )

    school_name = clean_text(
        school.get(
            "school_name"
        )
    )

    region_name = (
        clean_text(
            school.get(
                "region_name"
            )
        )
        or "Basilicata"
    )

    parts = [
        school_name,
        address,
        region_name,
        "Italia",
    ]

    query = ", ".join(
        part
        for part in parts
        if part
    )

    if not query:
        return None

    return {
        "q":
            query,
    }


# ============================================================
# NOMINATIM CLIENT
# ============================================================

class NominatimClient:

    def __init__(
        self,
        cache,
        refresh=False,
    ):
        self.cache = cache
        self.refresh = refresh

        load_dotenv(
            ROOT / ".env"
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

                "namedetails":
                    1,
            }
        )

        if self.email:
            params[
                "email"
            ] = self.email

        # La chiave include l'intera query V2.
        # Le vecchie query della V1 possono quindi restare
        # nella stessa cache senza contaminare i nuovi risultati.
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
            < REQUEST_DELAY_SECONDS
        ):
            time.sleep(
                REQUEST_DELAY_SECONDS
                - elapsed
            )

        response = requests.get(
            NOMINATIM_URL,
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

        # Persistiamo subito ogni risposta:
        # se il batch si interrompe non perdiamo il lavoro.
        save_cache(
            self.cache
        )

        return results


# ============================================================
# GEOCODER RESULT HELPERS
# ============================================================

def result_name(result):
    namedetails = (
        result.get(
            "namedetails"
        )
        or {}
    )

    return (
        namedetails.get("name")
        or result.get("name")
        or result.get(
            "display_name"
        )
    )


def result_address_text(result):
    address = (
        result.get("address")
        or {}
    )

    road = (
        address.get("road")
        or address.get("pedestrian")
        or address.get("residential")
        or address.get("path")
        or address.get("footway")
        or address.get("place")
        or address.get("hamlet")
    )

    number = address.get(
        "house_number"
    )

    if road and number:
        return (
            f"{road} {number}"
        )

    return road


def result_locality(result):
    address = (
        result.get("address")
        or {}
    )

    return (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("hamlet")
    )


# ============================================================
# SPATIAL VALIDATION
# ============================================================

def build_metric_boundary(
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
            "Impossibile determinare un CRS metrico "
            "per il comune."
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


def spatial_metrics(
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

    distance_boundary_m = float(
        point_metric.distance(
            boundary_metric.boundary
        )
    )

    if inside:
        distance_municipality_m = 0.0

    else:
        distance_municipality_m = float(
            point_metric.distance(
                boundary_metric
            )
        )

    return {
        "inside_municipality":
            inside,

        "distance_to_boundary_m":
            distance_boundary_m,

        "distance_to_municipality_m":
            distance_municipality_m,
    }


# ============================================================
# RESULT SCORING
# ============================================================

def score_result(
    school,
    result,
    municipality_geometry,
    metric_crs,
    boundary_metric,
):
    latitude = float(
        result["lat"]
    )

    longitude = float(
        result["lon"]
    )

    mim_address = normalize_address(
        school.get(
            "address"
        )
    )

    candidate_address = (
        normalize_address(
            result_address_text(
                result
            )
        )
    )

    address_score = similarity(
        mim_address,
        candidate_address,
    )

    mim_name = normalize_text(
        school.get(
            "school_name"
        )
    )

    candidate_name = (
        normalize_text(
            result_name(
                result
            )
        )
    )

    name_score = similarity(
        mim_name,
        candidate_name,
    )

    spatial = spatial_metrics(
        latitude=latitude,
        longitude=longitude,
        municipality_geometry=(
            municipality_geometry
        ),
        metric_crs=metric_crs,
        boundary_metric=(
            boundary_metric
        ),
    )

    locality = result_locality(
        result
    )

    importance = float(
        result.get(
            "importance",
            0.0,
        )
        or 0.0
    )

    # Nel geocoder l'indirizzo è la prova principale.
    # Il nome è secondario perché molte query restituiscono
    # una strada/civico invece del POI scolastico.
    score = (
        0.70 * address_score
        + 0.20 * name_score
        + 10.0 * min(
            importance,
            1.0,
        )
    )

    if spatial[
        "inside_municipality"
    ]:
        score += 5.0

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

        "result_name":
            result_name(
                result
            ),

        "result_address":
            result_address_text(
                result
            ),

        "result_locality":
            locality,

        "result_type":
            result.get(
                "type"
            ),

        "result_class":
            result.get(
                "class"
            ),

        "osm_type":
            result.get(
                "osm_type"
            ),

        "osm_id":
            result.get(
                "osm_id"
            ),

        "address_score":
            address_score,

        "name_score":
            name_score,

        "importance":
            importance,

        "geocoder_score":
            score,

        **spatial,
    }


# ============================================================
# SCHOOL GEOCODING V2
# ============================================================

def geocode_school(
    school,
    client,
    municipality,
    metric_crs,
    boundary_metric,
):
    query_attempts = []

    structured = (
        build_local_structured_params(
            school,
            municipality,
        )
    )

    if structured:
        query_attempts.append(
            (
                "local_structured",
                structured,
                True,
            )
        )

    local_address = (
        build_local_address_params(
            school,
            municipality,
        )
    )

    if local_address:
        query_attempts.append(
            (
                "local_address",
                local_address,
                True,
            )
        )

    local_school = (
        build_local_school_params(
            school,
            municipality,
        )
    )

    if local_school:
        query_attempts.append(
            (
                "local_school",
                local_school,
                True,
            )
        )

    external = (
        build_external_params(
            school
        )
    )

    if external:
        query_attempts.append(
            (
                "external_diagnostic",
                external,
                False,
            )
        )

    all_candidates = []
    good_local_candidate_found = False

    for (
        query_type,
        params,
        is_local,
    ) in query_attempts:

        # La ricerca esterna è solo diagnostica e viene usata
        # soltanto se non abbiamo già una localizzazione locale
        # sufficientemente plausibile.
        if (
            query_type
            == "external_diagnostic"
            and good_local_candidate_found
        ):
            break

        results = client.search(
            params
        )

        current_candidates = []

        for result in results:
            scored = score_result(
                school=school,
                result=result,
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

            scored[
                "query_type"
            ] = query_type

            scored[
                "query_params"
            ] = json.dumps(
                params,
                ensure_ascii=False,
                sort_keys=True,
            )

            scored[
                "local_search"
            ] = is_local

            current_candidates.append(
                scored
            )

            all_candidates.append(
                scored
            )

        if (
            is_local
            and current_candidates
        ):
            inside_candidates = [
                candidate
                for candidate
                in current_candidates
                if candidate[
                    "inside_municipality"
                ]
            ]

            if inside_candidates:
                best_local = max(
                    candidate[
                        "geocoder_score"
                    ]
                    for candidate
                    in inside_candidates
                )

                # Con 75+ abbiamo già un candidato locale utile;
                # evitiamo la ricerca esterna, che potrebbe riportare
                # omonimie nella provincia.
                if best_local >= 75:
                    good_local_candidate_found = True
                    break

    if not all_candidates:
        return None, []

    # Deduplica lo stesso oggetto restituito da query differenti.
    unique = {}

    for candidate in all_candidates:
        key = (
            candidate[
                "osm_type"
            ],
            candidate[
                "osm_id"
            ],
            round(
                candidate[
                    "latitude"
                ],
                7,
            ),
            round(
                candidate[
                    "longitude"
                ],
                7,
            ),
        )

        current = unique.get(
            key
        )

        if (
            current is None
            or candidate[
                "geocoder_score"
            ]
            > current[
                "geocoder_score"
            ]
        ):
            unique[
                key
            ] = candidate

    candidates = list(
        unique.values()
    )

    # Priorità:
    # 1. dentro il boundary ISTAT;
    # 2. ottenuto da ricerca locale bounded;
    # 3. score.
    candidates.sort(
        key=lambda item: (
            bool(
                item[
                    "inside_municipality"
                ]
            ),
            bool(
                item[
                    "local_search"
                ]
            ),
            item[
                "geocoder_score"
            ],
        ),
        reverse=True,
    )

    return (
        candidates[0],
        candidates[:5],
    )


# ============================================================
# MAIN PROCESS
# ============================================================

def classify_geocoder_candidate(
    best,
):
    if best is None:
        return "unresolved"

    if not best[
        "inside_municipality"
    ]:
        return "outside_candidate"

    if (
        best[
            "geocoder_score"
        ]
        >= 85
        and best[
            "address_score"
        ]
        >= 80
    ):
        return "strong_candidate"

    if (
        best[
            "geocoder_score"
        ]
        >= 65
    ):
        return "review"

    return "weak_candidate"


def build_geocoding_dataset(
    registry,
    v2_locations,
    municipality,
    client,
):
    registry = registry.copy()

    required_v2_columns = [
        "school_code",
        "geocoding_status",
        "geometry_accepted",
        "candidate_site_id",
        "candidate_longitude",
        "candidate_latitude",
    ]

    missing_v2_columns = [
        column
        for column in required_v2_columns
        if column
        not in v2_locations.columns
    ]

    if missing_v2_columns:
        raise RuntimeError(
            "Colonne mancanti nel dataset V2: "
            + ", ".join(
                missing_v2_columns
            )
        )

    combined = registry.merge(
        v2_locations[
            required_v2_columns
        ],
        on="school_code",
        how="left",
        validate="one_to_one",
    )

    (
        metric_crs,
        boundary_metric,
    ) = build_metric_boundary(
        municipality[
            "geometry"
        ]
    )

    rows = []
    ranking_rows = []

    total = len(
        combined
    )

    for position, (
        _,
        row,
    ) in enumerate(
        combined.iterrows(),
        start=1,
    ):
        school = (
            row.to_dict()
        )

        print(
            f"[{position}/{total}] "
            f"{school['school_code']} "
            f"{school['school_name']}"
        )

        (
            best,
            candidates,
        ) = geocode_school(
            school=school,
            client=client,
            municipality=municipality,
            metric_crs=metric_crs,
            boundary_metric=(
                boundary_metric
            ),
        )

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            ranking_rows.append(
                {
                    "school_code":
                        school[
                            "school_code"
                        ],

                    "rank":
                        rank,

                    **candidate,
                }
            )

        status = (
            classify_geocoder_candidate(
                best
            )
        )

        if best is None:
            rows.append(
                {
                    "school_code":
                        school[
                            "school_code"
                        ],

                    "school_name":
                        school[
                            "school_name"
                        ],

                    "address":
                        school.get(
                            "address"
                        ),

                    "postal_code":
                        school.get(
                            "postal_code"
                        ),

                    "municipality_name":
                        school.get(
                            "municipality_name"
                        ),

                    "osm_v2_status":
                        school.get(
                            "geocoding_status"
                        ),

                    "osm_v2_geometry_accepted":
                        school.get(
                            "geometry_accepted"
                        ),

                    "osm_v2_site_id":
                        school.get(
                            "candidate_site_id"
                        ),

                    "geocoder_status":
                        status,

                    "provider":
                        "Nominatim",

                    "query_type":
                        None,

                    "query_params":
                        None,

                    "local_search":
                        None,

                    "latitude":
                        None,

                    "longitude":
                        None,

                    "display_name":
                        None,

                    "result_name":
                        None,

                    "result_address":
                        None,

                    "result_locality":
                        None,

                    "result_type":
                        None,

                    "result_class":
                        None,

                    "osm_type":
                        None,

                    "osm_id":
                        None,

                    "address_score":
                        None,

                    "name_score":
                        None,

                    "importance":
                        None,

                    "geocoder_score":
                        None,

                    "inside_municipality":
                        None,

                    "distance_to_boundary_m":
                        None,

                    "distance_to_municipality_m":
                        None,
                }
            )

            continue

        rows.append(
            {
                "school_code":
                    school[
                        "school_code"
                    ],

                "school_name":
                    school[
                        "school_name"
                    ],

                "address":
                    school.get(
                        "address"
                    ),

                "postal_code":
                    school.get(
                        "postal_code"
                    ),

                "municipality_name":
                    school.get(
                        "municipality_name"
                    ),

                "osm_v2_status":
                    school.get(
                        "geocoding_status"
                    ),

                "osm_v2_geometry_accepted":
                    school.get(
                        "geometry_accepted"
                    ),

                "osm_v2_site_id":
                    school.get(
                        "candidate_site_id"
                    ),

                "geocoder_status":
                    status,

                "provider":
                    "Nominatim",

                **best,
            }
        )

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            ranking_rows
        ),
    )


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    geocoding,
    candidates,
    municipality_code,
    school_year,
):
    directory = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    geocoding_path = (
        directory
        / (
            "school_geocoding_"
            f"{school_year}_v2.parquet"
        )
    )

    candidates_path = (
        directory
        / (
            "school_geocoding_candidates_"
            f"{school_year}_v2.parquet"
        )
    )

    geocoding.to_parquet(
        geocoding_path,
        index=False,
    )

    candidates.to_parquet(
        candidates_path,
        index=False,
    )

    print(
        "\n=== OUTPUT V2 ==="
    )

    print(
        f"✓ {geocoding_path}"
    )

    print(
        f"✓ {candidates_path}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    geocoding,
):
    print(
        "\n===================================="
    )

    print(
        " SCHOOL GEOCODING V2 COMPLETATO"
    )

    print(
        "===================================="
    )

    print(
        f"Record: {len(geocoding)}"
    )

    print(
        "\nGeocoder status:"
    )

    print(
        geocoding[
            "geocoder_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    inside_mask = (
        geocoding[
            "inside_municipality"
        ]
        .eq(True)
    )

    outside_mask = (
        geocoding[
            "inside_municipality"
        ]
        .eq(False)
    )

    inside_count = int(
        inside_mask.sum()
    )

    outside_count = int(
        outside_mask.sum()
    )

    unresolved_count = int(
        geocoding[
            "inside_municipality"
        ]
        .isna()
        .sum()
    )

    print(
        "\nDentro il comune: "
        f"{inside_count}"
    )

    print(
        "Fuori dal comune: "
        f"{outside_count}"
    )

    print(
        "Senza coordinate: "
        f"{unresolved_count}"
    )

    print(
        "\n=== RISULTATI FUORI COMUNE ==="
    )

    outside = (
        geocoding[
            outside_mask
        ]
    )

    if outside.empty:
        print(
            "Nessuno."
        )

    else:
        columns = [
            "school_code",
            "school_name",
            "address",
            "result_locality",
            "display_name",
            "query_type",
            "geocoder_score",
            "distance_to_municipality_m",
        ]

        print(
            outside[
                columns
            ]
            .sort_values(
                "distance_to_municipality_m"
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n=== CANDIDATI FORTI ==="
    )

    strong = geocoding[
        geocoding[
            "geocoder_status"
        ]
        == "strong_candidate"
    ]

    if strong.empty:
        print(
            "Nessuno."
        )

    else:
        print(
            strong[
                [
                    "school_code",
                    "school_name",
                    "address",
                    "result_name",
                    "result_address",
                    "query_type",
                    "address_score",
                    "name_score",
                    "geocoder_score",
                ]
            ]
            .sort_values(
                "geocoder_score",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n=== CANDIDATI DEBOLI / NON RISOLTI ==="
    )

    weak = geocoding[
        geocoding[
            "geocoder_status"
        ]
        .isin(
            [
                "weak_candidate",
                "unresolved",
            ]
        )
    ]

    if weak.empty:
        print(
            "Nessuno."
        )

    else:
        print(
            weak[
                [
                    "school_code",
                    "school_name",
                    "address",
                    "display_name",
                    "query_type",
                    "address_score",
                    "name_score",
                    "geocoder_score",
                ]
            ]
            .to_string(
                index=False
            )
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
        " MIM SCHOOL ADDRESS GEOCODING V2"
    )

    print(
        "===================================="
    )

    engine = (
        get_database_engine()
    )

    municipality = (
        load_municipality(
            engine,
            args.municipality_code,
        )
    )

    registry = load_registry(
        args.municipality_code,
        args.school_year,
    )

    v2_locations = (
        load_v2_locations(
            args.municipality_code,
            args.school_year,
        )
    )

    cache = load_cache()

    client = NominatimClient(
        cache=cache,
        refresh=args.refresh,
    )

    print(
        "\n=== INPUT ==="
    )

    print(
        f"Comune: "
        f"{municipality['name']}"
    )

    print(
        f"Scuole MIM: "
        f"{len(registry)}"
    )

    print(
        f"Cache query esistenti: "
        f"{len(cache)}"
    )

    (
        geocoding,
        candidates,
    ) = build_geocoding_dataset(
        registry=registry,
        v2_locations=v2_locations,
        municipality=municipality,
        client=client,
    )

    save_outputs(
        geocoding=geocoding,
        candidates=candidates,
        municipality_code=(
            args.municipality_code
        ),
        school_year=(
            args.school_year
        ),
    )

    print_summary(
        geocoding
    )


if __name__ == "__main__":
    main()
