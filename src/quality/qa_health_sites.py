import argparse
import json
import math
import os
import time
import unicodedata
from pathlib import Path

import pandas as pd
import geopandas as gpd
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz
from shapely import wkt
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_SALUTE_DIR = ROOT / "data" / "processed" / "salute"
RAW_GEOCODING_DIR = ROOT / "data" / "raw" / "salute" / "geocoding" / "nominatim"
FEATURES_SALUTE_DIR = ROOT / "data" / "features" / "salute"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
REQUEST_DELAY_SECONDS = 1.1

MISSING_TEXT_VALUES = {
    "",
    "-",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "QA spaziale dei siti Health. Individua coordinate sorgente "
            "sospette (es. duplicati su indirizzi diversi), geocodifica "
            "solo i record necessari o, opzionalmente, tutti i record, "
            "e confronta coordinate sorgente e candidati Nominatim."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--pharmacy-reference-date",
        default="2025-06-30",
        help="Snapshot farmacie YYYY-MM-DD. Default: 2025-06-30.",
    )

    parser.add_argument(
        "--hospital-year",
        default="2023",
        help="Anno dataset ospedaliero. Default: 2023.",
    )

    parser.add_argument(
        "--validate-all",
        action="store_true",
        help=(
            "Geocodifica anche i record con coordinate sorgente "
            "non sospette. Utile nel PoC; non necessario nella pipeline "
            "operativa nazionale."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignora la cache Nominatim e ripete le richieste.",
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code).strip().zfill(6)
    )

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "--municipality-code deve avere esattamente 6 cifre."
        )

    args.pharmacy_reference_date = (
        pd.Timestamp(args.pharmacy_reference_date).normalize()
    )

    return args


def normalize_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    value = str(value).strip()

    if value.lower() in MISSING_TEXT_VALUES:
        return ""

    return value


def ascii_normalize(value):
    value = normalize_text(value)

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    return " ".join(
        value.upper().split()
    )


def haversine_m(lat1, lon1, lat2, lon2):
    if any(
        pd.isna(value)
        for value in [
            lat1,
            lon1,
            lat2,
            lon2,
        ]
    ):
        return None

    radius = 6_371_008.8

    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))

    dphi = math.radians(
        float(lat2) - float(lat1)
    )

    dlambda = math.radians(
        float(lon2) - float(lon1)
    )

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return (
        2
        * radius
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a),
        )
    )


def get_database_engine():
    load_dotenv(ROOT / ".env")

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


def load_municipality(engine, municipality_code):
    query = text(
        """
        SELECT
            istat_code,
            name,
            ST_AsText(geometry) AS geometry_wkt
        FROM municipality
        WHERE istat_code = :istat_code;
        """
    )

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

    minx, miny, maxx, maxy = (
        geometry.bounds
    )

    return {
        "istat_code":
            row["istat_code"],

        "name":
            str(row["name"]).strip(),

        "geometry":
            geometry,

        "viewbox":
            (
                float(minx),
                float(maxy),
                float(maxx),
                float(miny),
            ),
    }


def load_health_sites(args):
    directory = (
        PROCESSED_SALUTE_DIR
        / args.municipality_code
    )

    reference_label = (
        args.pharmacy_reference_date
        .strftime("%Y%m%d")
    )

    pharmacy_path = (
        directory
        / (
            "pharmacy_sites_"
            f"{reference_label}.parquet"
        )
    )

    hospital_path = (
        directory
        / (
            "hospital_sites_"
            f"{args.hospital_year}.parquet"
        )
    )

    if not pharmacy_path.exists():
        raise FileNotFoundError(
            f"Silver farmacie non trovato: {pharmacy_path}"
        )

    if not hospital_path.exists():
        raise FileNotFoundError(
            f"Silver ospedali non trovato: {hospital_path}"
        )

    pharmacies = pd.read_parquet(
        pharmacy_path
    )

    hospitals = pd.read_parquet(
        hospital_path
    )

    return (
        pharmacies,
        hospitals,
    )


def add_municipality_coordinate_diagnostics(
    df,
    municipality,
    tolerance_m=500.0,
):
    """
    Valuta la coerenza spaziale delle coordinate sorgente rispetto
    al comune dichiarato.

    Una coordinata fuori dal comune non viene automaticamente
    scartata: diventa sospetta solo quando la distanza dal territorio
    comunale supera una tolleranza metrica generica.

    Questo evita falsi positivi per strutture immediatamente a ridosso
    del confine amministrativo.
    """

    result = df.copy()

    result[
        "source_coordinate_inside_municipality"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean",
    )

    result[
        "source_coordinate_distance_to_municipality_m"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Float64",
    )

    result[
        "source_coordinate_outside_municipality"
    ] = False

    present = (
        result[
            "source_coordinate_present"
        ]
        .fillna(False)
        .astype(bool)
    )

    present_index = result.index[
        present
    ]

    if len(present_index) == 0:
        return result

    municipality_geometry = (
        gpd.GeoSeries(
            [
                municipality[
                    "geometry"
                ]
            ],
            crs="EPSG:4326",
        )
    )

    projected_crs = (
        municipality_geometry
        .estimate_utm_crs()
    )

    if projected_crs is None:
        raise RuntimeError(
            "Impossibile determinare un CRS metrico "
            "per il controllo delle coordinate Health."
        )

    municipality_metric = (
        municipality_geometry
        .to_crs(
            projected_crs
        )
        .iloc[0]
    )

    points = gpd.GeoSeries(
        gpd.points_from_xy(
            result.loc[
                present_index,
                "source_longitude",
            ],
            result.loc[
                present_index,
                "source_latitude",
            ],
        ),
        index=present_index,
        crs="EPSG:4326",
    )

    points_metric = (
        points.to_crs(
            projected_crs
        )
    )

    inside = (
        points_metric.within(
            municipality_metric
        )
        | points_metric.touches(
            municipality_metric
        )
    )

    distances = (
        points_metric.distance(
            municipality_metric
        )
    )

    result.loc[
        present_index,
        "source_coordinate_inside_municipality",
    ] = inside.astype(bool)

    result.loc[
        present_index,
        "source_coordinate_distance_to_municipality_m",
    ] = distances.astype(float)

    result.loc[
        present_index,
        "source_coordinate_outside_municipality",
    ] = (
        distances
        > float(tolerance_m)
    ).astype(bool)

    return result


def detect_suspicious_pharmacy_coordinates(
    pharmacies,
    municipality,
    municipality_tolerance_m=500.0,
):
    df = pharmacies.copy()

    df["source_latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["source_longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df["source_coordinate_present"] = (
        df["source_latitude"].notna()
        & df["source_longitude"].notna()
    )

    df["normalized_address"] = (
        df["address"]
        .map(ascii_normalize)
    )

    df["coordinate_key"] = None

    present = (
        df["source_coordinate_present"]
    )

    df.loc[
        present,
        "coordinate_key",
    ] = (
        df.loc[
            present,
            "source_latitude",
        ]
        .round(6)
        .astype(str)
        + "|"
        + df.loc[
            present,
            "source_longitude",
        ]
        .round(6)
        .astype(str)
    )

    group_size = (
        df.loc[present]
        .groupby(
            "coordinate_key"
        )
        .size()
    )

    distinct_addresses = (
        df.loc[present]
        .groupby(
            "coordinate_key"
        )["normalized_address"]
        .nunique()
    )

    df[
        "duplicate_coordinate_group_size"
    ] = (
        df["coordinate_key"]
        .map(group_size)
        .fillna(0)
        .astype(int)
    )

    df[
        "duplicate_coordinate_distinct_addresses"
    ] = (
        df["coordinate_key"]
        .map(distinct_addresses)
        .fillna(0)
        .astype(int)
    )

    duplicate_suspicious = (
        df["source_coordinate_present"]
        & (
            df[
                "duplicate_coordinate_group_size"
            ]
            >= 2
        )
        & (
            df[
                "duplicate_coordinate_distinct_addresses"
            ]
            >= 2
        )
    )

    df = (
        add_municipality_coordinate_diagnostics(
            df,
            municipality,
            tolerance_m=(
                municipality_tolerance_m
            ),
        )
    )

    outside_suspicious = (
        df[
            "source_coordinate_outside_municipality"
        ]
        .fillna(False)
        .astype(bool)
    )

    df[
        "source_coordinate_suspicious"
    ] = (
        duplicate_suspicious
        | outside_suspicious
    )

    reasons = []

    for duplicate, outside in zip(
        duplicate_suspicious,
        outside_suspicious,
    ):
        if duplicate and outside:
            reason = (
                "duplicate_and_outside_municipality"
            )
        elif duplicate:
            reason = (
                "duplicate_coordinate"
            )
        elif outside:
            reason = (
                "outside_municipality"
            )
        else:
            reason = None

        reasons.append(
            reason
        )

    df[
        "source_coordinate_suspicion_reason"
    ] = reasons

    return df


def prepare_hospitals(
    hospitals,
    municipality,
    municipality_tolerance_m=500.0,
):
    df = hospitals.copy()

    df["source_latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["source_longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df["source_coordinate_present"] = (
        df["source_latitude"].notna()
        & df["source_longitude"].notna()
    )

    df[
        "duplicate_coordinate_group_size"
    ] = 0

    df[
        "duplicate_coordinate_distinct_addresses"
    ] = 0

    df = (
        add_municipality_coordinate_diagnostics(
            df,
            municipality,
            tolerance_m=(
                municipality_tolerance_m
            ),
        )
    )

    df[
        "source_coordinate_suspicious"
    ] = (
        df[
            "source_coordinate_outside_municipality"
        ]
        .fillna(False)
        .astype(bool)
    )

    df[
        "source_coordinate_suspicion_reason"
    ] = None

    df.loc[
        df[
            "source_coordinate_outside_municipality"
        ]
        .fillna(False)
        .astype(bool),
        "source_coordinate_suspicion_reason",
    ] = "outside_municipality"

    return df



def cache_path(municipality_code):
    directory = (
        RAW_GEOCODING_DIR
        / municipality_code
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        / "health_geocode_cache.json"
    )


def load_cache(path):
    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}


def save_cache(path, cache):
    path.write_text(
        json.dumps(
            cache,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def get_session():
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "urban-digital-twin-thesis/1.0 "
                "(academic research; health-site QA)"
            )
        }
    )

    return session


def build_queries(row, municipality_name):
    name = normalize_text(
        row.get("name")
    )

    address = normalize_text(
        row.get("address")
    )

    subcategory = normalize_text(
        row.get("subcategory")
    )

    queries = []

    if subcategory == "hospital":
        if name and address:
            queries.append(
                f"{name}, {address}, "
                f"{municipality_name}, Italia"
            )

        if address:
            queries.append(
                f"{address}, "
                f"{municipality_name}, Italia"
            )

        if name:
            queries.append(
                f"{name}, "
                f"{municipality_name}, Italia"
            )

    else:
        if address:
            queries.append(
                f"{address}, "
                f"{municipality_name}, Italia"
            )

        if name and address:
            queries.append(
                f"{name}, {address}, "
                f"{municipality_name}, Italia"
            )

        if name:
            queries.append(
                f"{name}, "
                f"{municipality_name}, Italia"
            )

    return list(
        dict.fromkeys(
            query
            for query in queries
            if query.strip()
        )
    )


def query_nominatim(
    session,
    query,
    municipality,
):
    left, top, right, bottom = (
        municipality["viewbox"]
    )

    params = {
        "q":
            query,

        "format":
            "jsonv2",

        "limit":
            5,

        "addressdetails":
            1,

        "namedetails":
            1,

        "countrycodes":
            "it",

        "bounded":
            1,

        "viewbox":
            (
                f"{left},{top},"
                f"{right},{bottom}"
            ),
    }

    response = session.get(
        NOMINATIM_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def municipality_match_score(
    candidate,
    municipality_name,
):
    address = candidate.get(
        "address",
        {}
    )

    candidate_values = [
        address.get(
            "city"
        ),
        address.get(
            "town"
        ),
        address.get(
            "village"
        ),
        address.get(
            "municipality"
        ),
    ]

    target = ascii_normalize(
        municipality_name
    )

    best = 0.0

    for value in candidate_values:
        if value:
            best = max(
                best,
                fuzz.ratio(
                    ascii_normalize(value),
                    target,
                ),
            )

    return best


def address_match_score(
    candidate,
    target_address,
):
    if not target_address:
        return 0.0

    display_name = normalize_text(
        candidate.get(
            "display_name"
        )
    )

    return float(
        fuzz.token_set_ratio(
            ascii_normalize(
                target_address
            ),
            ascii_normalize(
                display_name
            ),
        )
    )


def name_match_score(
    candidate,
    target_name,
):
    if not target_name:
        return 0.0

    namedetails = candidate.get(
        "namedetails",
        {}
    ) or {}

    possible_names = [
        namedetails.get(
            "name"
        ),
        candidate.get(
            "name"
        ),
        candidate.get(
            "display_name"
        ),
    ]

    scores = []

    for value in possible_names:
        if value:
            scores.append(
                fuzz.token_set_ratio(
                    ascii_normalize(
                        target_name
                    ),
                    ascii_normalize(
                        value
                    ),
                )
            )

    return (
        float(max(scores))
        if scores
        else 0.0
    )


def candidate_resolution(candidate):
    addresstype = normalize_text(
        candidate.get(
            "addresstype"
        )
    ).lower()

    candidate_type = normalize_text(
        candidate.get(
            "type"
        )
    ).lower()

    detailed = {
        "house",
        "building",
        "pharmacy",
        "hospital",
        "clinic",
        "doctors",
        "healthcare",
    }

    if (
        addresstype in detailed
        or candidate_type in detailed
    ):
        return "site_or_address_candidate"

    if (
        addresstype
        in {
            "road",
            "residential",
            "pedestrian",
        }
        or candidate_type
        in {
            "road",
            "residential",
            "pedestrian",
        }
    ):
        return "street_anchor_candidate"

    return "generic_candidate"


def score_candidate(
    candidate,
    row,
    municipality_name,
):
    municipality_score = (
        municipality_match_score(
            candidate,
            municipality_name,
        )
    )

    address_score = (
        address_match_score(
            candidate,
            normalize_text(
                row.get(
                    "address"
                )
            ),
        )
    )

    name_score = (
        name_match_score(
            candidate,
            normalize_text(
                row.get(
                    "name"
                )
            ),
        )
    )

    subcategory = normalize_text(
        row.get(
            "subcategory"
        )
    )

    if subcategory == "hospital":
        total = (
            0.45 * address_score
            + 0.35 * name_score
            + 0.20 * municipality_score
        )
    else:
        total = (
            0.65 * address_score
            + 0.15 * name_score
            + 0.20 * municipality_score
        )

    return {
        "score":
            float(total),

        "municipality_score":
            float(municipality_score),

        "address_score":
            float(address_score),

        "name_score":
            float(name_score),
    }


def best_geocoder_candidate(
    session,
    cache,
    cache_file,
    row,
    municipality,
    refresh,
):
    queries = build_queries(
        row,
        municipality["name"],
    )

    all_candidates = []

    for query in queries:
        cache_key = (
            municipality[
                "istat_code"
            ]
            + "||"
            + query
        )

        if (
            cache_key in cache
            and not refresh
        ):
            results = cache[
                cache_key
            ]

        else:
            results = query_nominatim(
                session,
                query,
                municipality,
            )

            cache[
                cache_key
            ] = results

            save_cache(
                cache_file,
                cache,
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

        for candidate in results:
            scored = score_candidate(
                candidate,
                row,
                municipality["name"],
            )

            all_candidates.append(
                {
                    "query":
                        query,

                    "candidate":
                        candidate,

                    **scored,
                }
            )

    if not all_candidates:
        return None

    all_candidates.sort(
        key=lambda item:
            item["score"],
        reverse=True,
    )

    return all_candidates[0]


def classify_result(
    row,
    best,
):
    source_present = bool(
        row[
            "source_coordinate_present"
        ]
    )

    suspicious = bool(
        row[
            "source_coordinate_suspicious"
        ]
    )

    outside_municipality = bool(
        row.get(
            "source_coordinate_outside_municipality",
            False,
        )
    )

    duplicate_suspicious = (
        int(
            row.get(
                "duplicate_coordinate_group_size",
                0,
            )
            or 0
        )
        >= 2
        and int(
            row.get(
                "duplicate_coordinate_distinct_addresses",
                0,
            )
            or 0
        )
        >= 2
    )

    if best is None:
        return (
            "unresolved",
            "no_geocoder_candidate",
        )

    candidate = best[
        "candidate"
    ]

    geocoder_latitude = float(
        candidate["lat"]
    )

    geocoder_longitude = float(
        candidate["lon"]
    )

    distance = None

    if source_present:
        distance = haversine_m(
            row[
                "source_latitude"
            ],
            row[
                "source_longitude"
            ],
            geocoder_latitude,
            geocoder_longitude,
        )

    resolution = candidate_resolution(
        candidate
    )

    strong_candidate = (
        best[
            "municipality_score"
        ]
        >= 80
        and best[
            "address_score"
        ]
        >= 70
        and best[
            "score"
        ]
        >= 72
    )

    if not source_present:
        if strong_candidate:
            return (
                resolution,
                "missing_source_coordinate_with_strong_candidate",
            )

        return (
            "review",
            "missing_source_coordinate_weak_candidate",
        )

    # --------------------------------------------------------
    # Source coordinate geographically incompatible with the
    # municipality declared by the institutional registry.
    # It is not silently replaced: a strong independent
    # candidate is required before declaring a conflict.
    # --------------------------------------------------------
    if outside_municipality:
        if strong_candidate:
            return (
                "source_coordinate_conflict",
                (
                    "source_coordinate_outside_municipality_"
                    "with_strong_candidate"
                ),
            )

        return (
            "review",
            (
                "source_coordinate_outside_municipality_"
                "requires_review"
            ),
        )

    # --------------------------------------------------------
    # Duplicate coordinate used by distinct addresses.
    # --------------------------------------------------------
    if suspicious and duplicate_suspicious:
        if (
            distance is not None
            and distance > 150
            and strong_candidate
        ):
            return (
                "source_coordinate_conflict",
                (
                    "duplicate_source_coordinate_"
                    "conflicts_with_address_geocoder"
                ),
            )

        if (
            distance is not None
            and distance <= 100
            and strong_candidate
        ):
            return (
                "source_coordinate_confirmed",
                (
                    "duplicate_source_coordinate_"
                    "but_geocoder_agrees"
                ),
            )

        return (
            "review",
            (
                "duplicate_source_coordinate_"
                "requires_review"
            ),
        )

    # --------------------------------------------------------
    # Normal source coordinate explicitly checked
    # (e.g. --validate-all).
    # --------------------------------------------------------
    if (
        distance is not None
        and distance <= 100
        and strong_candidate
    ):
        return (
            "source_coordinate_confirmed",
            "source_and_geocoder_agree",
        )

    if (
        distance is not None
        and distance > 250
        and strong_candidate
    ):
        return (
            "source_coordinate_conflict",
            "source_and_geocoder_disagree",
        )

    return (
        "review",
        "insufficient_agreement_for_auto_confirmation",
    )



def audit_records(
    records,
    municipality,
    validate_all,
    refresh,
):
    cache_file = cache_path(
        municipality[
            "istat_code"
        ]
    )

    cache = load_cache(
        cache_file
    )

    session = get_session()

    outputs = []

    for _, row in records.iterrows():
        should_geocode = (
            validate_all
            or not bool(
                row[
                    "source_coordinate_present"
                ]
            )
            or bool(
                row[
                    "source_coordinate_suspicious"
                ]
            )
        )

        base = {
            "service_site_id":
                row.get(
                    "service_site_id"
                ),

            "category":
                row.get(
                    "category"
                ),

            "subcategory":
                row.get(
                    "subcategory"
                ),

            "source_record_id":
                row.get(
                    "source_record_id"
                ),

            "name":
                row.get(
                    "name"
                ),

            "address":
                row.get(
                    "address"
                ),

            "source_latitude":
                row.get(
                    "source_latitude"
                ),

            "source_longitude":
                row.get(
                    "source_longitude"
                ),

            "source_coordinate_present":
                bool(
                    row[
                        "source_coordinate_present"
                    ]
                ),

            "duplicate_coordinate_group_size":
                int(
                    row[
                        "duplicate_coordinate_group_size"
                    ]
                ),

            "duplicate_coordinate_distinct_addresses":
                int(
                    row[
                        "duplicate_coordinate_distinct_addresses"
                    ]
                ),

            "source_coordinate_suspicious":
                bool(
                    row[
                        "source_coordinate_suspicious"
                    ]
                ),

            "source_coordinate_inside_municipality":
                row.get(
                    "source_coordinate_inside_municipality"
                ),

            "source_coordinate_distance_to_municipality_m":
                row.get(
                    "source_coordinate_distance_to_municipality_m"
                ),

            "source_coordinate_outside_municipality":
                bool(
                    row.get(
                        "source_coordinate_outside_municipality",
                        False,
                    )
                ),

            "source_coordinate_suspicion_reason":
                row.get(
                    "source_coordinate_suspicion_reason"
                ),

            "geocoded_for_qa":
                bool(
                    should_geocode
                ),
        }

        if not should_geocode:
            outputs.append(
                {
                    **base,

                    "geocoder_query":
                        None,

                    "geocoder_display_name":
                        None,

                    "geocoder_latitude":
                        None,

                    "geocoder_longitude":
                        None,

                    "geocoder_type":
                        None,

                    "geocoder_addresstype":
                        None,

                    "geocoder_score":
                        None,

                    "geocoder_address_score":
                        None,

                    "geocoder_name_score":
                        None,

                    "geocoder_municipality_score":
                        None,

                    "source_geocoder_distance_m":
                        None,

                    "qa_status":
                        "not_checked",

                    "qa_reason":
                        "source_coordinate_not_flagged",
                }
            )

            continue

        best = best_geocoder_candidate(
            session=session,
            cache=cache,
            cache_file=cache_file,
            row=row,
            municipality=municipality,
            refresh=refresh,
        )

        if best is None:
            status, reason = classify_result(
                row,
                best,
            )

            outputs.append(
                {
                    **base,

                    "geocoder_query":
                        None,

                    "geocoder_display_name":
                        None,

                    "geocoder_latitude":
                        None,

                    "geocoder_longitude":
                        None,

                    "geocoder_type":
                        None,

                    "geocoder_addresstype":
                        None,

                    "geocoder_score":
                        None,

                    "geocoder_address_score":
                        None,

                    "geocoder_name_score":
                        None,

                    "geocoder_municipality_score":
                        None,

                    "source_geocoder_distance_m":
                        None,

                    "qa_status":
                        status,

                    "qa_reason":
                        reason,
                }
            )

            continue

        candidate = best[
            "candidate"
        ]

        geocoder_latitude = float(
            candidate["lat"]
        )

        geocoder_longitude = float(
            candidate["lon"]
        )

        distance = None

        if bool(
            row[
                "source_coordinate_present"
            ]
        ):
            distance = haversine_m(
                row[
                    "source_latitude"
                ],
                row[
                    "source_longitude"
                ],
                geocoder_latitude,
                geocoder_longitude,
            )

        status, reason = classify_result(
            row,
            best,
        )

        outputs.append(
            {
                **base,

                "geocoder_query":
                    best[
                        "query"
                    ],

                "geocoder_display_name":
                    candidate.get(
                        "display_name"
                    ),

                "geocoder_latitude":
                    geocoder_latitude,

                "geocoder_longitude":
                    geocoder_longitude,

                "geocoder_type":
                    candidate.get(
                        "type"
                    ),

                "geocoder_addresstype":
                    candidate.get(
                        "addresstype"
                    ),

                "geocoder_score":
                    round(
                        best[
                            "score"
                        ],
                        2,
                    ),

                "geocoder_address_score":
                    round(
                        best[
                            "address_score"
                        ],
                        2,
                    ),

                "geocoder_name_score":
                    round(
                        best[
                            "name_score"
                        ],
                        2,
                    ),

                "geocoder_municipality_score":
                    round(
                        best[
                            "municipality_score"
                        ],
                        2,
                    ),

                "source_geocoder_distance_m":
                    (
                        round(
                            distance,
                            2,
                        )
                        if distance is not None
                        else None
                    ),

                "qa_status":
                    status,

                "qa_reason":
                    reason,
            }
        )

    return pd.DataFrame(
        outputs
    )


def main():
    args = parse_args()

    engine = get_database_engine()

    municipality = load_municipality(
        engine,
        args.municipality_code,
    )

    (
        pharmacies,
        hospitals,
    ) = load_health_sites(
        args
    )

    pharmacies = (
        detect_suspicious_pharmacy_coordinates(
            pharmacies,
            municipality,
        )
    )

    hospitals = prepare_hospitals(
            hospitals,
            municipality,
        )

    combined = pd.concat(
        [
            pharmacies,
            hospitals,
        ],
        ignore_index=True,
        sort=False,
    )

    print(
        "\n===================================="
    )
    print(
        " HEALTH SPATIAL QA"
    )
    print(
        "===================================="
    )

    print(
        "Comune: "
        f"{municipality['name']} "
        f"({municipality['istat_code']})"
    )

    print(
        "\nFarmacie: "
        f"{len(pharmacies)}"
    )

    print(
        "Farmacie senza coordinate: "
        f"{int((~pharmacies['source_coordinate_present']).sum())}"
    )

    print(
        "Farmacie con coordinate sorgente sospette: "
        f"{int(pharmacies['source_coordinate_suspicious'].sum())}"
    )

    suspicious_groups = (
        pharmacies.loc[
            pharmacies[
                "source_coordinate_suspicious"
            ],
            [
                "source_latitude",
                "source_longitude",
                "duplicate_coordinate_group_size",
                "duplicate_coordinate_distinct_addresses",
            ],
        ]
        .drop_duplicates()
    )

    if not suspicious_groups.empty:
        print(
            "\nGruppi coordinate sospette:"
        )

        print(
            suspicious_groups.to_string(
                index=False
            )
        )

    print(
        "\nOspedali/stabilimenti: "
        f"{len(hospitals)}"
    )

    print(
        "Ospedali senza coordinate: "
        f"{int((~hospitals['source_coordinate_present']).sum())}"
    )

    print(
        "\nEseguo QA Nominatim "
        + (
            "su tutti i record..."
            if args.validate_all
            else "solo sui record mancanti/sospetti..."
        )
    )

    audit = audit_records(
        combined,
        municipality,
        validate_all=args.validate_all,
        refresh=args.refresh,
    )

    output_dir = (
        FEATURES_SALUTE_DIR
        / args.municipality_code
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_label = (
        args.pharmacy_reference_date
        .strftime("%Y%m%d")
    )

    output_csv = (
        output_dir
        / (
            "health_spatial_qa_"
            f"{reference_label}.csv"
        )
    )

    output_parquet = (
        output_dir
        / (
            "health_spatial_qa_"
            f"{reference_label}.parquet"
        )
    )

    audit.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_parquet(
        output_parquet,
        index=False,
    )

    print(
        "\n=== QA STATUS ==="
    )

    print(
        audit[
            "qa_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n=== DETTAGLIO ==="
    )

    columns = [
        "subcategory",
        "source_record_id",
        "name",
        "address",
        "source_coordinate_suspicious",
        "duplicate_coordinate_group_size",
        "source_latitude",
        "source_longitude",
        "geocoder_latitude",
        "geocoder_longitude",
        "source_geocoder_distance_m",
        "geocoder_score",
        "qa_status",
        "qa_reason",
    ]

    print(
        audit[
            columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ {output_csv}"
    )

    print(
        f"✓ {output_parquet}"
    )


if __name__ == "__main__":
    main()
