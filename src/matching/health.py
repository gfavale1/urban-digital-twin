import argparse
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
from shapely.geometry import Point
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_SALUTE_DIR = ROOT / "data" / "processed" / "salute"
RAW_OSM_DIR = ROOT / "data" / "raw" / "osm"
FEATURES_SALUTE_DIR = ROOT / "data" / "features" / "salute"

DEFAULT_BUFFER_M = 1500.0
AUTO_THRESHOLD = 82.0
REVIEW_THRESHOLD = 65.0
MIN_MARGIN = 8.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Matching automatico tra siti Health del Ministero della Salute "
            "e POI OpenStreetMap. Le evidenze mancanti non sono trattate "
            "come evidenze negative: i pesi sono rinormalizzati."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre.",
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
        "--buffer-m",
        type=float,
        default=DEFAULT_BUFFER_M,
        help="Buffer OSM attorno al comune per discovery dei POI.",
    )

    parser.add_argument(
        "--refresh-osm",
        action="store_true",
        help="Scarica nuovamente i POI OSM anche se esiste uno snapshot locale.",
    )

    args = parser.parse_args()

    args.municipality_code = str(
        args.municipality_code
    ).strip().zfill(6)

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "--municipality-code deve avere esattamente 6 cifre."
        )

    args.pharmacy_reference_date = pd.Timestamp(
        args.pharmacy_reference_date
    ).normalize()

    if args.buffer_m < 0:
        raise ValueError("--buffer-m deve essere >= 0.")

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

    if value.lower() in {
        "",
        "-",
        "nan",
        "none",
        "null",
        "n/a",
        "na",
    }:
        return ""

    return " ".join(value.split())


def normalize_for_match(value):
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

    value = value.upper()

    replacements = {
        r"\bVIALE\b": "VIA",
        r"\bV\.LE\b": "VIA",
        r"\bPIAZZALE\b": "PIAZZA",
        r"\bP\.ZZA\b": "PIAZZA",
        r"\bCONTRADA\b": "C DA",
        r"\bC\.DA\b": "C DA",
        r"\bN\.\s*": "",
        r"\bNUMERO\b": "",
    }

    for pattern, replacement in replacements.items():
        value = re.sub(
            pattern,
            replacement,
            value,
        )

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value,
    )

    return " ".join(
        value.split()
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
            {"istat_code": municipality_code},
        ).mappings().first()

    if row is None:
        raise RuntimeError(
            "Comune non presente in PostGIS. "
            "Eseguire prima l'ingestion ISTAT."
        )

    return {
        "istat_code": row["istat_code"],
        "name": str(row["name"]).strip(),
        "geometry": wkt.loads(
            row["geometry_wkt"]
        ),
    }


def buffered_polygon(geometry, buffer_m):
    gdf = gpd.GeoDataFrame(
        {"geometry": [geometry]},
        crs="EPSG:4326",
    )

    if buffer_m <= 0:
        return geometry

    projected_crs = (
        gdf.estimate_utm_crs()
    )

    if projected_crs is None:
        raise RuntimeError(
            "Impossibile stimare un CRS metrico per il buffer."
        )

    buffered = (
        gdf.to_crs(
            projected_crs
        )
        .buffer(
            buffer_m
        )
    )

    result = gpd.GeoSeries(
        buffered,
        crs=projected_crs,
    ).to_crs(
        "EPSG:4326"
    )

    return result.iloc[0]


def health_osm_snapshot_path(
    municipality_code,
):
    directory = (
        RAW_OSM_DIR
        / municipality_code
        / "health_features"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = sorted(
        directory.glob(
            "health_features_*.parquet"
        ),
        reverse=True,
    )

    return directory, existing


def fetch_osm_health_features(
    polygon,
):
    tags = {
        "amenity": [
            "pharmacy",
            "hospital",
        ],
        "healthcare": [
            "pharmacy",
            "hospital",
        ],
    }

    if hasattr(
        ox,
        "features_from_polygon",
    ):
        gdf = ox.features_from_polygon(
            polygon,
            tags,
        )
    else:
        gdf = (
            ox.features
            .features_from_polygon(
                polygon,
                tags,
            )
        )

    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=["geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = gdf.reset_index()

    if gdf.crs is None:
        gdf = gdf.set_crs(
            "EPSG:4326"
        )
    else:
        gdf = gdf.to_crs(
            "EPSG:4326"
        )

    return gdf


def load_or_fetch_osm(
    municipality_code,
    polygon,
    refresh,
):
    directory, existing = (
        health_osm_snapshot_path(
            municipality_code
        )
    )

    if existing and not refresh:
        path = existing[0]
        gdf = gpd.read_parquet(path)

        return (
            gdf,
            path,
            False,
        )

    gdf = fetch_osm_health_features(
        polygon
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    path = (
        directory
        / (
            "health_features_"
            f"{timestamp}.parquet"
        )
    )

    gdf.to_parquet(
        path,
        index=False,
    )

    return (
        gdf,
        path,
        True,
    )


def classify_osm_subcategory(row):
    amenity = normalize_text(
        row.get("amenity")
    ).lower()

    healthcare = normalize_text(
        row.get("healthcare")
    ).lower()

    if (
        amenity == "pharmacy"
        or healthcare == "pharmacy"
    ):
        return "pharmacy"

    if (
        amenity == "hospital"
        or healthcare == "hospital"
    ):
        return "hospital"

    return None


def representative_point(geometry):
    if geometry is None:
        return None

    try:
        if geometry.is_empty:
            return None
    except Exception:
        return None

    if geometry.geom_type == "Point":
        return geometry

    return geometry.representative_point()


def first_present(row, columns):
    for column in columns:
        if column in row.index:
            value = normalize_text(
                row.get(column)
            )

            if value:
                return value

    return ""


def build_osm_address(row):
    street = first_present(
        row,
        [
            "addr:street",
            "addr:place",
        ],
    )

    number = first_present(
        row,
        [
            "addr:housenumber",
        ],
    )

    if street and number:
        return f"{street} {number}"

    return street


def prepare_osm_candidates(gdf):
    records = []

    for _, row in gdf.iterrows():
        subcategory = (
            classify_osm_subcategory(
                row
            )
        )

        if subcategory is None:
            continue

        point = representative_point(
            row.get(
                "geometry"
            )
        )

        if point is None:
            continue

        osm_type = normalize_text(
            row.get(
                "element_type"
            )
        )

        osm_id = normalize_text(
            row.get(
                "osmid"
            )
        )

        if not osm_id:
            for column in [
                "id",
                "osm_id",
            ]:
                osm_id = normalize_text(
                    row.get(column)
                )

                if osm_id:
                    break

        records.append(
            {
                "candidate_id":
                    (
                        f"{osm_type}:{osm_id}"
                        if osm_type
                        else osm_id
                    ),

                "subcategory":
                    subcategory,

                "name":
                    first_present(
                        row,
                        [
                            "name",
                            "brand",
                            "operator",
                        ],
                    ),

                "address":
                    build_osm_address(
                        row
                    ),

                "latitude":
                    float(point.y),

                "longitude":
                    float(point.x),

                "amenity":
                    normalize_text(
                        row.get(
                            "amenity"
                        )
                    ),

                "healthcare":
                    normalize_text(
                        row.get(
                            "healthcare"
                        )
                    ),

                "geometry":
                    point,
            }
        )

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326",
    )


def load_health_silver(args):
    directory = (
        PROCESSED_SALUTE_DIR
        / args.municipality_code
    )

    label = (
        args.pharmacy_reference_date
        .strftime("%Y%m%d")
    )

    pharmacy_path = (
        directory
        / f"pharmacy_sites_{label}.parquet"
    )

    hospital_path = (
        directory
        / f"hospital_sites_{args.hospital_year}.parquet"
    )

    if not pharmacy_path.exists():
        raise FileNotFoundError(
            str(pharmacy_path)
        )

    if not hospital_path.exists():
        raise FileNotFoundError(
            str(hospital_path)
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


def mark_suspicious_coordinates(
    pharmacies,
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
        .map(normalize_for_match)
    )

    present = df[
        "source_coordinate_present"
    ]

    df["coordinate_key"] = None

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

    address_count = (
        df.loc[present]
        .groupby(
            "coordinate_key"
        )["normalized_address"]
        .nunique()
    )

    df["source_coordinate_suspicious"] = (
        df["coordinate_key"]
        .map(group_size)
        .fillna(0)
        .ge(2)
        & df["coordinate_key"]
        .map(address_count)
        .fillna(0)
        .ge(2)
    )

    return df


def prepare_hospitals(hospitals):
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

    df["source_coordinate_suspicious"] = (
        False
    )

    return df


def haversine_m(lat1, lon1, lat2, lon2):
    values = [
        lat1,
        lon1,
        lat2,
        lon2,
    ]

    if any(
        pd.isna(value)
        for value in values
    ):
        return None

    radius = 6_371_008.8

    phi1 = math.radians(
        float(lat1)
    )

    phi2 = math.radians(
        float(lat2)
    )

    dphi = math.radians(
        float(lat2)
        - float(lat1)
    )

    dlambda = math.radians(
        float(lon2)
        - float(lon1)
    )

    a = (
        math.sin(
            dphi / 2
        ) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(
            dlambda / 2
        ) ** 2
    )

    return (
        2
        * radius
        * math.atan2(
            math.sqrt(a),
            math.sqrt(
                1 - a
            ),
        )
    )


def proximity_score(distance_m):
    if distance_m is None:
        return None

    if distance_m <= 50:
        return 100.0

    if distance_m <= 100:
        return 95.0

    if distance_m <= 250:
        return 80.0

    if distance_m <= 500:
        return 60.0

    if distance_m <= 1000:
        return 30.0

    return 0.0


def weighted_available_score(
    evidences,
):
    available = [
        (
            score,
            weight,
        )
        for score, weight
        in evidences
        if score is not None
    ]

    if not available:
        return None

    total_weight = sum(
        weight
        for _, weight
        in available
    )

    if total_weight <= 0:
        return None

    return sum(
        score * weight
        for score, weight
        in available
    ) / total_weight


def score_candidate(
    source,
    candidate,
):
    source_name = normalize_text(
        source.get("name")
    )

    source_address = normalize_text(
        source.get("address")
    )

    candidate_name = normalize_text(
        candidate.get("name")
    )

    candidate_address = normalize_text(
        candidate.get("address")
    )

    name_score = None

    if (
        source_name
        and candidate_name
    ):
        name_score = float(
            fuzz.token_set_ratio(
                normalize_for_match(
                    source_name
                ),
                normalize_for_match(
                    candidate_name
                ),
            )
        )

    address_score = None

    if (
        source_address
        and candidate_address
    ):
        address_score = float(
            fuzz.token_set_ratio(
                normalize_for_match(
                    source_address
                ),
                normalize_for_match(
                    candidate_address
                ),
            )
        )

    distance_m = None
    proximity = None

    source_present = bool(
        source.get(
            "source_coordinate_present",
            False,
        )
    )

    source_suspicious = bool(
        source.get(
            "source_coordinate_suspicious",
            False,
        )
    )

    if (
        source_present
        and not source_suspicious
    ):
        distance_m = haversine_m(
            source.get(
                "source_latitude"
            ),
            source.get(
                "source_longitude"
            ),
            candidate.get(
                "latitude"
            ),
            candidate.get(
                "longitude"
            ),
        )

        proximity = proximity_score(
            distance_m
        )

    if (
        normalize_text(
            source.get(
                "subcategory"
            )
        )
        == "hospital"
    ):
        score = (
            weighted_available_score(
                [
                    (
                        name_score,
                        0.55,
                    ),
                    (
                        address_score,
                        0.40,
                    ),
                    (
                        proximity,
                        0.05,
                    ),
                ]
            )
        )
    else:
        score = (
            weighted_available_score(
                [
                    (
                        name_score,
                        0.35,
                    ),
                    (
                        address_score,
                        0.50,
                    ),
                    (
                        proximity,
                        0.15,
                    ),
                ]
            )
        )

    return {
        "score":
            score,

        "name_score":
            name_score,

        "address_score":
            address_score,

        "source_distance_m":
            distance_m,

        "proximity_score":
            proximity,
    }


def classify_match(
    score,
    margin,
    name_score,
    address_score,
):
    strong_identity = (
        (
            address_score is not None
            and address_score >= 75
        )
        or (
            name_score is not None
            and name_score >= 80
        )
    )

    if (
        score is not None
        and score >= AUTO_THRESHOLD
        and margin >= MIN_MARGIN
        and strong_identity
    ):
        return "matched_auto"

    if (
        score is not None
        and score >= REVIEW_THRESHOLD
    ):
        return "review"

    return "unresolved"


def match_services(
    services,
    candidates,
):
    outputs = []
    candidate_rows = []

    for _, source in services.iterrows():
        subset = candidates.loc[
            candidates[
                "subcategory"
            ]
            == source[
                "subcategory"
            ]
        ]

        scored = []

        for _, candidate in subset.iterrows():
            metrics = score_candidate(
                source,
                candidate,
            )

            scored.append(
                {
                    "candidate":
                        candidate,

                    **metrics,
                }
            )

        scored = [
            item
            for item in scored
            if item[
                "score"
            ] is not None
        ]

        scored.sort(
            key=lambda item:
                item["score"],
            reverse=True,
        )

        if not scored:
            outputs.append(
                {
                    "service_site_id":
                        source[
                            "service_site_id"
                        ],

                    "subcategory":
                        source[
                            "subcategory"
                        ],

                    "source_record_id":
                        source[
                            "source_record_id"
                        ],

                    "name":
                        source[
                            "name"
                        ],

                    "address":
                        source[
                            "address"
                        ],

                    "source_coordinate_present":
                        bool(
                            source[
                                "source_coordinate_present"
                            ]
                        ),

                    "source_coordinate_suspicious":
                        bool(
                            source[
                                "source_coordinate_suspicious"
                            ]
                        ),

                    "match_status":
                        "unresolved",

                    "candidate_id":
                        None,

                    "candidate_name":
                        None,

                    "candidate_address":
                        None,

                    "candidate_latitude":
                        None,

                    "candidate_longitude":
                        None,

                    "match_score":
                        None,

                    "score_margin":
                        None,

                    "name_score":
                        None,

                    "address_score":
                        None,

                    "source_candidate_distance_m":
                        None,
                }
            )

            continue

        best = scored[0]

        second_score = (
            scored[1]["score"]
            if len(scored) > 1
            else 0.0
        )

        margin = (
            best["score"]
            - second_score
        )

        status = classify_match(
            best["score"],
            margin,
            best["name_score"],
            best["address_score"],
        )

        candidate = best[
            "candidate"
        ]

        outputs.append(
            {
                "service_site_id":
                    source[
                        "service_site_id"
                    ],

                "subcategory":
                    source[
                        "subcategory"
                    ],

                "source_record_id":
                    source[
                        "source_record_id"
                    ],

                "name":
                    source[
                        "name"
                    ],

                "address":
                    source[
                        "address"
                    ],

                "source_coordinate_present":
                    bool(
                        source[
                            "source_coordinate_present"
                        ]
                    ),

                "source_coordinate_suspicious":
                    bool(
                        source[
                            "source_coordinate_suspicious"
                        ]
                    ),

                "match_status":
                    status,

                "candidate_id":
                    candidate[
                        "candidate_id"
                    ],

                "candidate_name":
                    candidate[
                        "name"
                    ],

                "candidate_address":
                    candidate[
                        "address"
                    ],

                "candidate_latitude":
                    candidate[
                        "latitude"
                    ],

                "candidate_longitude":
                    candidate[
                        "longitude"
                    ],

                "match_score":
                    round(
                        best["score"],
                        2,
                    ),

                "score_margin":
                    round(
                        margin,
                        2,
                    ),

                "name_score":
                    (
                        round(
                            best[
                                "name_score"
                            ],
                            2,
                        )
                        if best[
                            "name_score"
                        ] is not None
                        else None
                    ),

                "address_score":
                    (
                        round(
                            best[
                                "address_score"
                            ],
                            2,
                        )
                        if best[
                            "address_score"
                        ] is not None
                        else None
                    ),

                "source_candidate_distance_m":
                    (
                        round(
                            best[
                                "source_distance_m"
                            ],
                            2,
                        )
                        if best[
                            "source_distance_m"
                        ] is not None
                        else None
                    ),
            }
        )

        for rank, item in enumerate(
            scored[:5],
            start=1,
        ):
            candidate = item[
                "candidate"
            ]

            candidate_rows.append(
                {
                    "service_site_id":
                        source[
                            "service_site_id"
                        ],

                    "rank":
                        rank,

                    "candidate_id":
                        candidate[
                            "candidate_id"
                        ],

                    "candidate_name":
                        candidate[
                            "name"
                        ],

                    "candidate_address":
                        candidate[
                            "address"
                        ],

                    "candidate_latitude":
                        candidate[
                            "latitude"
                        ],

                    "candidate_longitude":
                        candidate[
                            "longitude"
                        ],

                    "score":
                        round(
                            item[
                                "score"
                            ],
                            2,
                        ),

                    "name_score":
                        (
                            round(
                                item[
                                    "name_score"
                                ],
                                2,
                            )
                            if item[
                                "name_score"
                            ] is not None
                            else None
                        ),

                    "address_score":
                        (
                            round(
                                item[
                                    "address_score"
                                ],
                                2,
                            )
                            if item[
                                "address_score"
                            ] is not None
                            else None
                        ),

                    "source_distance_m":
                        (
                            round(
                                item[
                                    "source_distance_m"
                                ],
                                2,
                            )
                            if item[
                                "source_distance_m"
                            ] is not None
                            else None
                        ),
                }
            )

    return (
        pd.DataFrame(
            outputs
        ),
        pd.DataFrame(
            candidate_rows
        ),
    )


def main():
    args = parse_args()

    engine = get_database_engine()

    municipality = load_municipality(
        engine,
        args.municipality_code,
    )

    query_polygon = buffered_polygon(
        municipality[
            "geometry"
        ],
        args.buffer_m,
    )

    print(
        "\n===================================="
    )
    print(
        " HEALTH MIM/OSM MATCH"
    )
    print(
        "===================================="
    )

    print(
        f"Comune: {municipality['name']} "
        f"({municipality['istat_code']})"
    )

    print(
        f"OSM discovery buffer: "
        f"{args.buffer_m:.0f} m"
    )

    (
        osm_raw,
        snapshot_path,
        downloaded,
    ) = load_or_fetch_osm(
        args.municipality_code,
        query_polygon,
        args.refresh_osm,
    )

    print(
        "\nOSM snapshot: "
        f"{snapshot_path}"
    )

    print(
        "Scaricato ora: "
        f"{downloaded}"
    )

    candidates = prepare_osm_candidates(
        osm_raw
    )

    print(
        "\nOSM health candidates:"
    )

    if candidates.empty:
        print(
            "  Nessun candidato."
        )
    else:
        print(
            candidates[
                "subcategory"
            ]
            .value_counts()
            .to_string()
        )

    (
        pharmacies,
        hospitals,
    ) = load_health_silver(
        args
    )

    pharmacies = (
        mark_suspicious_coordinates(
            pharmacies
        )
    )

    hospitals = prepare_hospitals(
        hospitals
    )

    services = pd.concat(
        [
            pharmacies,
            hospitals,
        ],
        ignore_index=True,
        sort=False,
    )

    (
        matches,
        candidate_table,
    ) = match_services(
        services,
        candidates,
    )

    output_dir = (
        PROCESSED_SALUTE_DIR
        / args.municipality_code
    )

    feature_dir = (
        FEATURES_SALUTE_DIR
        / args.municipality_code
    )

    feature_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label = (
        args.pharmacy_reference_date
        .strftime("%Y%m%d")
    )

    match_path = (
        output_dir
        / (
            "health_osm_matches_"
            f"{label}.parquet"
        )
    )

    match_csv = (
        feature_dir
        / (
            "health_osm_matches_"
            f"{label}.csv"
        )
    )

    candidates_csv = (
        feature_dir
        / (
            "health_osm_match_candidates_"
            f"{label}.csv"
        )
    )

    osm_candidates_path = (
        output_dir
        / "health_osm_candidates.parquet"
    )

    matches.to_parquet(
        match_path,
        index=False,
    )

    matches.to_csv(
        match_csv,
        index=False,
        encoding="utf-8-sig",
    )

    candidate_table.to_csv(
        candidates_csv,
        index=False,
        encoding="utf-8-sig",
    )

    candidates.to_parquet(
        osm_candidates_path,
        index=False,
    )

    print(
        "\n=== MATCH STATUS ==="
    )

    print(
        matches[
            "match_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n=== DETTAGLIO ==="
    )

    print(
        matches[
            [
                "subcategory",
                "source_record_id",
                "name",
                "address",
                "source_coordinate_suspicious",
                "match_status",
                "candidate_name",
                "candidate_address",
                "match_score",
                "score_margin",
                "name_score",
                "address_score",
                "source_candidate_distance_m",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== OUTPUT ==="
    )
    print(
        f"✓ {osm_candidates_path}"
    )
    print(
        f"✓ {match_path}"
    )
    print(
        f"✓ {match_csv}"
    )
    print(
        f"✓ {candidates_csv}"
    )


if __name__ == "__main__":
    main()
