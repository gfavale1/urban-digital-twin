import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from rapidfuzz import fuzz
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
PROCESSED_ISTAT_DIR = ROOT / "data" / "processed" / "istat"
FEATURES_MIM_DIR = ROOT / "data" / "features" / "mim"
RAW_MIM_DIR = ROOT / "data" / "raw" / "mim"

DEFAULT_SCHOOL_YEAR = "202425"
DEFAULT_BUILDING_YEAR = "202425"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Geocodifica esclusivamente i nuovi building_code MIM "
            "non presenti nel precedente dataset spaziale."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre.",
    )

    parser.add_argument(
        "--school-year",
        default=DEFAULT_SCHOOL_YEAR,
    )

    parser.add_argument(
        "--building-year",
        default=DEFAULT_BUILDING_YEAR,
    )

    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.1,
        help="Pausa minima tra richieste Nominatim.",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignora la cache locale per le query del run.",
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


def clean_text(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip()

    return value or None


def normalize_text(value):
    value = clean_text(value)

    if value is None:
        return ""

    value = value.upper()

    value = (
        value
        .replace("`", "'")
        .replace("’", "'")
    )

    value = re.sub(
        r"[^A-Z0-9À-ÖØ-Ý' ]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def clean_address_for_query(value):
    value = clean_text(value)

    if value is None:
        return None

    value = (
        value
        .replace("`", "'")
        .replace("’", "'")
    )

    value = re.sub(
        r"\bS\.?\s*N\.?\s*C\.?\b",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\bS\.?\s*N\.?\b",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip(" ,")

    return value


def load_inputs(args):
    processed_dir = (
        PROCESSED_MIM_DIR
        / args.municipality_code
    )

    features_dir = (
        FEATURES_MIM_DIR
        / args.municipality_code
    )

    buildings_path = (
        processed_dir
        / (
            "physical_school_buildings_"
            f"{args.building_year}_"
            f"from_schools_{args.school_year}_"
            "with_reused_locations.parquet"
        )
    )

    comparison_path = (
        features_dir
        / (
            "school_building_code_comparison_"
            f"{args.building_year}.csv"
        )
    )

    census_path = (
        PROCESSED_ISTAT_DIR
        / (
            f"{args.municipality_code}_"
            "census_areas_2021.parquet"
        )
    )

    for path in [
        buildings_path,
        comparison_path,
        census_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"File richiesto non trovato: {path}"
            )

    buildings = pd.read_parquet(
        buildings_path
    )

    comparison = pd.read_csv(
        comparison_path,
        dtype={
            "building_code":
                str,
        },
    )

    census = gpd.read_parquet(
        census_path
    )

    buildings[
        "building_code"
    ] = (
        buildings[
            "building_code"
        ]
        .astype("string")
        .str.strip()
    )

    comparison[
        "building_code"
    ] = (
        comparison[
            "building_code"
        ]
        .astype("string")
        .str.strip()
    )

    new_codes = set(
        comparison.loc[
            comparison[
                "comparison_status"
            ]
            == "new_only",
            "building_code",
        ]
        .dropna()
    )

    new_buildings = (
        buildings[
            buildings[
                "building_code"
            ].isin(
                new_codes
            )
        ]
        .copy()
    )

    # Generalizable safeguard:
    # MIM CodiceEdificio begins with the six-digit municipality code.
    new_buildings = (
        new_buildings[
            new_buildings[
                "building_code"
            ]
            .astype(str)
            .str.startswith(
                args.municipality_code
            )
        ]
        .copy()
    )

    if new_buildings.empty:
        raise RuntimeError(
            "Nessun building_code new_only trovato nel comune target."
        )

    return (
        new_buildings,
        census,
        buildings_path,
        comparison_path,
    )


def municipality_context(
    new_buildings,
    census,
):
    municipality_names = (
        new_buildings[
            "building_municipality_name"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if len(
        municipality_names
    ) != 1:
        raise RuntimeError(
            "Impossibile determinare un unico nome comune dai nuovi edifici: "
            + repr(
                municipality_names
            )
        )

    municipality_name = (
        municipality_names[0]
    )

    postal_codes = (
        new_buildings[
            "building_postal_code"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    postal_code = (
        postal_codes[0]
        if len(
            postal_codes
        ) == 1
        else None
    )

    census_wgs84 = census.to_crs(
        4326
    )

    municipality_geometry = (
        census_wgs84.geometry.union_all()
    )

    minx, miny, maxx, maxy = (
        municipality_geometry.bounds
    )

    viewbox = (
        f"{minx},{maxy},"
        f"{maxx},{miny}"
    )

    return {
        "municipality_name":
            municipality_name,

        "postal_code":
            postal_code,

        "geometry":
            municipality_geometry,

        "viewbox":
            viewbox,
    }


def cache_paths(
    args,
):
    cache_dir = (
        RAW_MIM_DIR
        / "geocoding"
        / args.municipality_code
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        cache_dir
        / (
            "nominatim_new_school_buildings_"
            f"{args.building_year}.json"
        )
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


def save_cache(
    path,
    cache,
):
    path.write_text(
        json.dumps(
            cache,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def query_nominatim(
    session,
    address,
    context,
    pause_seconds,
):
    url = (
        "https://nominatim.openstreetmap.org/search"
    )

    common = {
        "format":
            "jsonv2",

        "limit":
            5,

        "countrycodes":
            "it",

        "addressdetails":
            1,

        "bounded":
            1,

        "viewbox":
            context[
                "viewbox"
            ],
    }

    structured = {
        **common,
        "street":
            address,

        "city":
            context[
                "municipality_name"
            ],
    }

    if context[
        "postal_code"
    ]:
        structured[
            "postalcode"
        ] = context[
            "postal_code"
        ]

    response = session.get(
        url,
        params=structured,
        timeout=60,
    )

    response.raise_for_status()

    results = response.json()

    time.sleep(
        pause_seconds
    )

    # One conservative free-form fallback only if structured search fails.
    if not results:
        freeform = {
            **common,
            "q":
                (
                    f"{address}, "
                    f"{context['municipality_name']}, Italia"
                ),
        }

        response = session.get(
            url,
            params=freeform,
            timeout=60,
        )

        response.raise_for_status()

        results = response.json()

        time.sleep(
            pause_seconds
        )

    return results


def score_candidate(
    official_address,
    candidate,
    municipality_geometry,
):
    display_name = clean_text(
        candidate.get(
            "display_name"
        )
    ) or ""

    official_norm = normalize_text(
        official_address
    )

    display_norm = normalize_text(
        display_name
    )

    address_score = float(
        fuzz.token_set_ratio(
            official_norm,
            display_norm,
        )
    )

    try:
        longitude = float(
            candidate[
                "lon"
            ]
        )

        latitude = float(
            candidate[
                "lat"
            ]
        )
    except Exception:
        return None

    point = Point(
        longitude,
        latitude,
    )

    inside = bool(
        municipality_geometry.covers(
            point
        )
    )

    candidate_type = (
        clean_text(
            candidate.get(
                "type"
            )
        )
        or ""
    )

    candidate_class = (
        clean_text(
            candidate.get(
                "class"
            )
        )
        or clean_text(
            candidate.get(
                "category"
            )
        )
        or ""
    )

    return {
        "display_name":
            display_name,

        "longitude":
            longitude,

        "latitude":
            latitude,

        "inside_target_municipality":
            inside,

        "address_score":
            address_score,

        "osm_type":
            candidate_type,

        "osm_class":
            candidate_class,

        "osm_id":
            candidate.get(
                "osm_id"
            ),

        "osm_type_prefix":
            candidate.get(
                "osm_type"
            ),

        "place_id":
            candidate.get(
                "place_id"
            ),
    }


def classify_top_candidate(
    candidates,
):
    if not candidates:
        return (
            "unresolved",
            "none",
        )

    top = candidates[0]

    if not top[
        "inside_target_municipality"
    ]:
        return (
            "outside_candidate",
            "low",
        )

    score = top[
        "address_score"
    ]

    if score >= 85:
        return (
            "accepted_candidate",
            "medium_high",
        )

    if score >= 70:
        return (
            "review",
            "medium",
        )

    return (
        "unresolved",
        "low",
    )


def geocode_unique_addresses(
    new_buildings,
    context,
    args,
):
    cache_path = cache_paths(
        args
    )

    cache = load_cache(
        cache_path
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent":
                (
                    "urban-digital-twin-thesis/1.0 "
                    f"(academic research; municipality "
                    f"{args.municipality_code})"
                )
        }
    )

    unique_rows = (
        new_buildings[
            [
                "official_building_address",
                "building_postal_code",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    results_by_address = {}
    candidate_rows = []

    for _, row in unique_rows.iterrows():
        original_address = clean_text(
            row[
                "official_building_address"
            ]
        )

        query_address = (
            clean_address_for_query(
                original_address
            )
        )

        if not query_address:
            results_by_address[
                original_address
            ] = {
                "status":
                    "unresolved",

                "confidence":
                    "none",

                "top":
                    None,

                "query_address":
                    None,
            }

            continue

        cache_key = (
            f"{args.municipality_code}|"
            f"{normalize_text(query_address)}"
        )

        if (
            not args.refresh
            and cache_key in cache
        ):
            raw_candidates = (
                cache[
                    cache_key
                ][
                    "results"
                ]
            )

            source = (
                "cache"
            )

        else:
            raw_candidates = (
                query_nominatim(
                    session=session,
                    address=query_address,
                    context=context,
                    pause_seconds=(
                        args.pause_seconds
                    ),
                )
            )

            cache[
                cache_key
            ] = {
                "query_address":
                    query_address,

                "municipality_code":
                    args.municipality_code,

                "queried_at_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "results":
                    raw_candidates,
            }

            save_cache(
                cache_path,
                cache,
            )

            source = (
                "nominatim"
            )

        scored = []

        for rank, candidate in enumerate(
            raw_candidates,
            start=1,
        ):
            score = score_candidate(
                official_address=(
                    original_address
                ),
                candidate=candidate,
                municipality_geometry=(
                    context[
                        "geometry"
                    ]
                ),
            )

            if score is None:
                continue

            score[
                "rank_raw"
            ] = rank

            scored.append(
                score
            )

        scored.sort(
            key=lambda item: (
                item[
                    "inside_target_municipality"
                ],
                item[
                    "address_score"
                ],
            ),
            reverse=True,
        )

        status, confidence = (
            classify_top_candidate(
                scored
            )
        )

        top = (
            scored[0]
            if scored
            else None
        )

        results_by_address[
            original_address
        ] = {
            "status":
                status,

            "confidence":
                confidence,

            "top":
                top,

            "query_address":
                query_address,

            "source":
                source,
        }

        for candidate_rank, candidate in enumerate(
            scored,
            start=1,
        ):
            candidate_rows.append(
                {
                    "official_building_address":
                        original_address,

                    "query_address":
                        query_address,

                    "candidate_rank":
                        candidate_rank,

                    **candidate,
                }
            )

    return (
        results_by_address,
        pd.DataFrame(
            candidate_rows
        ),
        cache_path,
    )


def build_output(
    new_buildings,
    results_by_address,
):
    rows = []

    for _, row in new_buildings.iterrows():
        item = row.to_dict()

        official_address = clean_text(
            row[
                "official_building_address"
            ]
        )

        result = results_by_address.get(
            official_address,
            {
                "status":
                    "unresolved",

                "confidence":
                    "none",

                "top":
                    None,

                "query_address":
                    None,

                "source":
                    None,
            },
        )

        top = result.get(
            "top"
        )

        item[
            "new_geocoder_status"
        ] = result[
            "status"
        ]

        item[
            "new_geocoder_confidence"
        ] = result[
            "confidence"
        ]

        item[
            "new_geocoder_query_address"
        ] = result.get(
            "query_address"
        )

        item[
            "new_geocoder_query_source"
        ] = result.get(
            "source"
        )

        if top:
            item[
                "new_geocoder_longitude"
            ] = top[
                "longitude"
            ]

            item[
                "new_geocoder_latitude"
            ] = top[
                "latitude"
            ]

            item[
                "new_geocoder_result_address"
            ] = top[
                "display_name"
            ]

            item[
                "new_geocoder_address_score"
            ] = top[
                "address_score"
            ]

            item[
                "new_geocoder_inside_target"
            ] = top[
                "inside_target_municipality"
            ]

            item[
                "new_geocoder_osm_type"
            ] = top[
                "osm_type"
            ]

            item[
                "new_geocoder_osm_class"
            ] = top[
                "osm_class"
            ]

        else:
            item[
                "new_geocoder_longitude"
            ] = None

            item[
                "new_geocoder_latitude"
            ] = None

            item[
                "new_geocoder_result_address"
            ] = None

            item[
                "new_geocoder_address_score"
            ] = None

            item[
                "new_geocoder_inside_target"
            ] = None

            item[
                "new_geocoder_osm_type"
            ] = None

            item[
                "new_geocoder_osm_class"
            ] = None

        rows.append(
            item
        )

    return pd.DataFrame(
        rows
    )


def save_outputs(
    result,
    candidates,
    args,
    cache_path,
    buildings_path,
    comparison_path,
):
    features_dir = (
        FEATURES_MIM_DIR
        / args.municipality_code
    )

    processed_dir = (
        PROCESSED_MIM_DIR
        / args.municipality_code
    )

    csv_path = (
        features_dir
        / (
            "new_school_buildings_geocoded_"
            f"{args.building_year}.csv"
        )
    )

    parquet_path = (
        processed_dir
        / (
            "new_school_buildings_geocoded_"
            f"{args.building_year}.parquet"
        )
    )

    candidates_path = (
        features_dir
        / (
            "new_school_buildings_geocode_candidates_"
            f"{args.building_year}.csv"
        )
    )

    manifest_path = (
        features_dir
        / (
            "new_school_buildings_geocode_"
            f"{args.building_year}_manifest.json"
        )
    )

    result.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    result.to_parquet(
        parquet_path,
        index=False,
    )

    candidates.to_csv(
        candidates_path,
        index=False,
        encoding="utf-8-sig",
    )

    unique_address_count = int(
        result[
            "official_building_address"
        ].nunique()
    )

    status_counts = {
        str(key):
            int(value)
        for key, value in (
            result[
                "new_geocoder_status"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        )
    }

    manifest = {
        "municipality_code":
            args.municipality_code,

        "school_year":
            args.school_year,

        "building_year":
            args.building_year,

        "new_building_count":
            int(
                len(
                    result
                )
            ),

        "unique_address_queries":
            unique_address_count,

        "status_counts":
            status_counts,

        "input_buildings":
            str(
                buildings_path
            ),

        "input_comparison":
            str(
                comparison_path
            ),

        "cache_path":
            str(
                cache_path
            ),

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "csv":
            csv_path,

        "parquet":
            parquet_path,

        "candidates":
            candidates_path,

        "manifest":
            manifest_path,
    }


def main():
    args = parse_args()

    print(
        "\n===================================="
    )
    print(
        " GEOCODE NEW SCHOOL BUILDINGS"
    )
    print(
        "===================================="
    )

    (
        new_buildings,
        census,
        buildings_path,
        comparison_path,
    ) = load_inputs(
        args
    )

    context = municipality_context(
        new_buildings,
        census,
    )

    print(
        f"Municipality: "
        f"{context['municipality_name']} "
        f"({args.municipality_code})"
    )

    print(
        f"New building_code: "
        f"{len(new_buildings)}"
    )

    print(
        "Unique official addresses: "
        f"{new_buildings['official_building_address'].nunique()}"
    )

    print(
        "\nBuildings:"
    )

    for _, row in (
        new_buildings[
            [
                "building_code",
                "official_building_address",
            ]
        ]
        .sort_values(
            "building_code"
        )
        .iterrows()
    ):
        print(
            f"  {row['building_code']} "
            f"→ {row['official_building_address']}"
        )

    (
        results_by_address,
        candidates,
        cache_path,
    ) = geocode_unique_addresses(
        new_buildings,
        context,
        args,
    )

    result = build_output(
        new_buildings,
        results_by_address,
    )

    outputs = save_outputs(
        result=result,
        candidates=candidates,
        args=args,
        cache_path=cache_path,
        buildings_path=buildings_path,
        comparison_path=comparison_path,
    )

    print(
        "\n=== RESULTS ==="
    )

    print(
        result[
            [
                "building_code",
                "official_building_address",
                "new_geocoder_status",
                "new_geocoder_confidence",
                "new_geocoder_address_score",
                "new_geocoder_result_address",
            ]
        ]
        .sort_values(
            "building_code"
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nStatus counts:"
    )

    print(
        result[
            "new_geocoder_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n=== OUTPUT ==="
    )

    for path in outputs.values():
        print(
            f"✓ {path}"
        )

    print(
        "\nNOTA METODOLOGICA:"
    )

    print(
        "Questo step genera candidati Nominatim solamente per "
        "building_code new_only. Non modifica il dataset finale."
    )

    print(
        "Gli indirizzi identici condividono la stessa query di geocoding, "
        "ma non vengono automaticamente fusi nello stesso school_site."
    )


if __name__ == "__main__":
    main()
