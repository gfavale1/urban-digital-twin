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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Geocodifica gli edifici scolastici MIM ancora privi di "
            "localizzazione dopo il riuso e le validazioni già applicate."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
    )

    parser.add_argument(
        "--school-year",
        default="202425",
    )

    parser.add_argument(
        "--building-year",
        default="202425",
    )

    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.1,
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
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

    value = (
        value.upper()
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


def load_pending(args):
    directory = (
        PROCESSED_MIM_DIR
        / args.municipality_code
    )

    source_path = (
        directory
        / (
            "physical_school_buildings_"
            f"{args.building_year}_"
            f"from_schools_{args.school_year}_"
            "with_reused_locations.parquet"
        )
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Dataset edifici non trovato: {source_path}"
        )

    df = pd.read_parquet(
        source_path
    )

    df[
        "building_code"
    ] = (
        df[
            "building_code"
        ]
        .astype("string")
        .str.strip()
    )

    pending = (
        df[
            df[
                "location_reuse_status"
            ]
            == "needs_localization"
        ]
        .copy()
    )

    # Exclude buildings belonging to another municipality.
    pending = (
        pending[
            pending[
                "building_code"
            ]
            .astype(str)
            .str.startswith(
                args.municipality_code
            )
        ]
        .copy()
    )

    # Exclude already validated new buildings.
    validated_new_path = (
        directory
        / (
            "new_school_buildings_validated_"
            f"{args.building_year}.parquet"
        )
    )

    if validated_new_path.exists():
        validated = pd.read_parquet(
            validated_new_path
        )

        validated[
            "building_code"
        ] = (
            validated[
                "building_code"
            ]
            .astype("string")
            .str.strip()
        )

        if (
            "validated_location_usable"
            in validated.columns
        ):
            validated_codes = set(
                validated.loc[
                    validated[
                        "validated_location_usable"
                    ]
                    == True,
                    "building_code",
                ]
            )

            pending = (
                pending[
                    ~pending[
                        "building_code"
                    ].isin(
                        validated_codes
                    )
                ]
                .copy()
            )

    return (
        pending,
        source_path,
    )


def build_context(
    pending,
    args,
):
    census_path = (
        PROCESSED_ISTAT_DIR
        / (
            f"{args.municipality_code}_"
            "census_areas_2021.parquet"
        )
    )

    if not census_path.exists():
        raise FileNotFoundError(
            f"Dataset ISTAT non trovato: {census_path}"
        )

    census = gpd.read_parquet(
        census_path
    ).to_crs(
        4326
    )

    geometry = (
        census.geometry.union_all()
    )

    minx, miny, maxx, maxy = (
        geometry.bounds
    )

    municipality_names = (
        pending[
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
            "Impossibile determinare un unico comune target."
        )

    return {
        "name":
            municipality_names[0],

        "geometry":
            geometry,

        "viewbox":
            f"{minx},{maxy},{maxx},{miny}",
    }


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

    params = {
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

        "street":
            address,

        "city":
            context[
                "name"
            ],
    }

    response = session.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    results = response.json()

    time.sleep(
        pause_seconds
    )

    if not results:
        params = {
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

            "q":
                (
                    f"{address}, "
                    f"{context['name']}, Italia"
                ),
        }

        response = session.get(
            url,
            params=params,
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
    context,
):
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

    display_name = (
        clean_text(
            candidate.get(
                "display_name"
            )
        )
        or ""
    )

    score = float(
        fuzz.token_set_ratio(
            normalize_text(
                official_address
            ),
            normalize_text(
                display_name
            ),
        )
    )

    point = Point(
        longitude,
        latitude,
    )

    inside = bool(
        context[
            "geometry"
        ].covers(
            point
        )
    )

    return {
        "display_name":
            display_name,

        "longitude":
            longitude,

        "latitude":
            latitude,

        "inside_target":
            inside,

        "address_score":
            score,

        "osm_type":
            candidate.get(
                "type"
            ),

        "osm_class":
            (
                candidate.get(
                    "class"
                )
                or candidate.get(
                    "category"
                )
            ),

        "osm_id":
            candidate.get(
                "osm_id"
            ),

        "place_id":
            candidate.get(
                "place_id"
            ),
    }


def main():
    args = parse_args()

    pending, source_path = (
        load_pending(
            args
        )
    )

    if pending.empty:
        print(
            "Nessun edificio pending da geocodificare."
        )
        return

    context = build_context(
        pending,
        args,
    )

    cache_dir = (
        RAW_MIM_DIR
        / "geocoding"
        / args.municipality_code
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        cache_dir
        / (
            "nominatim_remaining_school_buildings_"
            f"{args.building_year}.json"
        )
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
                    "(academic research)"
                )
        }
    )

    unique_addresses = (
        pending[
            "official_building_address"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    results_by_address = {}
    candidate_rows = []

    for official_address in unique_addresses:
        query_address = (
            clean_address_for_query(
                official_address
            )
        )

        key = (
            f"{args.municipality_code}|"
            f"{normalize_text(query_address)}"
        )

        if (
            not args.refresh
            and key in cache
        ):
            raw_results = (
                cache[
                    key
                ][
                    "results"
                ]
            )

            query_source = (
                "cache"
            )

        else:
            raw_results = (
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
                key
            ] = {
                "query_address":
                    query_address,

                "queried_at_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "results":
                    raw_results,
            }

            save_cache(
                cache_path,
                cache,
            )

            query_source = (
                "nominatim"
            )

        scored = []

        for candidate in raw_results:
            item = score_candidate(
                official_address,
                candidate,
                context,
            )

            if item is not None:
                scored.append(
                    item
                )

        scored.sort(
            key=lambda row: (
                row[
                    "inside_target"
                ],
                row[
                    "address_score"
                ],
            ),
            reverse=True,
        )

        results_by_address[
            official_address
        ] = {
            "query_address":
                query_address,

            "query_source":
                query_source,

            "candidates":
                scored,
        }

        for rank, item in enumerate(
            scored,
            start=1,
        ):
            candidate_rows.append(
                {
                    "official_building_address":
                        official_address,

                    "candidate_rank":
                        rank,

                    **item,
                }
            )

    summary_rows = []

    for _, row in pending.iterrows():
        address = row[
            "official_building_address"
        ]

        result = (
            results_by_address.get(
                address,
                {
                    "query_address":
                        None,

                    "query_source":
                        None,

                    "candidates":
                        [],
                },
            )
        )

        candidates = result[
            "candidates"
        ]

        top = (
            candidates[0]
            if candidates
            else None
        )

        summary_rows.append(
            {
                "building_code":
                    row[
                        "building_code"
                    ],

                "official_building_address":
                    address,

                "linked_school_names":
                    row.get(
                        "linked_school_names"
                    ),

                "query_address":
                    result[
                        "query_address"
                    ],

                "query_source":
                    result[
                        "query_source"
                    ],

                "candidate_longitude":
                    (
                        top[
                            "longitude"
                        ]
                        if top
                        else None
                    ),

                "candidate_latitude":
                    (
                        top[
                            "latitude"
                        ]
                        if top
                        else None
                    ),

                "candidate_display_name":
                    (
                        top[
                            "display_name"
                        ]
                        if top
                        else None
                    ),

                "candidate_address_score":
                    (
                        top[
                            "address_score"
                        ]
                        if top
                        else None
                    ),

                "candidate_inside_target":
                    (
                        top[
                            "inside_target"
                        ]
                        if top
                        else None
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    candidates = pd.DataFrame(
        candidate_rows
    )

    features_dir = (
        FEATURES_MIM_DIR
        / args.municipality_code
    )

    summary_path = (
        features_dir
        / (
            "remaining_school_buildings_geocoded_"
            f"{args.building_year}.csv"
        )
    )

    candidates_path = (
        features_dir
        / (
            "remaining_school_buildings_geocode_candidates_"
            f"{args.building_year}.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    candidates.to_csv(
        candidates_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n===================================="
    )

    print(
        " GEOCODE REMAINING SCHOOL BUILDINGS"
    )

    print(
        "===================================="
    )

    print(
        f"Pending buildings: {len(pending)}"
    )

    print(
        f"Unique address queries: {len(unique_addresses)}"
    )

    print(
        "\n=== RESULTS ==="
    )

    print(
        summary[
            [
                "building_code",
                "official_building_address",
                "candidate_display_name",
                "candidate_address_score",
                "candidate_longitude",
                "candidate_latitude",
                "candidate_inside_target",
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
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ {summary_path}"
    )

    print(
        f"✓ {candidates_path}"
    )

    print(
        "\nNOTA:"
    )

    print(
        "Le coordinate sono candidati di geocoding e non vengono "
        "automaticamente accettate come geometria finale."
    )


if __name__ == "__main__":
    main()
