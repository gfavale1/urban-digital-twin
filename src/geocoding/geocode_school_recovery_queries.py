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
PROCESSED_ISTAT_DIR = ROOT / "data" / "processed" / "istat"
FEATURES_MIM_DIR = ROOT / "data" / "features" / "mim"
RAW_MIM_DIR = ROOT / "data" / "raw" / "mim"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Esegue query di recupero geocoding definite in un CSV esterno. "
            "Le eccezioni/alias specifici del comune restano nei dati di "
            "validazione e non vengono hard-coded nella pipeline."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
    )

    parser.add_argument(
        "--queries-file",
        required=True,
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


def load_context(args):
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

    census = (
        gpd.read_parquet(
            census_path
        )
        .to_crs(4326)
    )

    geometry = (
        census.geometry.union_all()
    )

    minx, miny, maxx, maxy = (
        geometry.bounds
    )

    return {
        "geometry":
            geometry,

        "viewbox":
            f"{minx},{maxy},{maxx},{miny}",
    }


def load_queries(args):
    path = Path(
        args.queries_file
    )

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Queries file non trovato: {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "building_code": str,
        },
    )

    required = {
        "building_code",
        "query_address",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Colonne mancanti nel queries file: "
            + ", ".join(sorted(missing))
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

    if df[
        "building_code"
    ].duplicated().any():
        raise RuntimeError(
            "building_code duplicati nel queries file."
        )

    return df, path


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


def request_nominatim(
    session,
    params,
    pause_seconds,
):
    url = (
        "https://nominatim.openstreetmap.org/search"
    )

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


def query_variants(
    session,
    municipality_name,
    query_address,
    query_name,
    context,
    pause_seconds,
):
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

    variants = []

    if query_name:
        variants.append(
            {
                **common,
                "q":
                    (
                        f"{query_name}, "
                        f"{query_address}, "
                        f"{municipality_name}, Italia"
                    ),
            }
        )

    variants.append(
        {
            **common,
            "street":
                query_address,

            "city":
                municipality_name,
        }
    )

    variants.append(
        {
            **common,
            "q":
                (
                    f"{query_address}, "
                    f"{municipality_name}, Italia"
                ),
        }
    )

    all_results = []

    seen = set()

    for variant_index, params in enumerate(
        variants,
        start=1,
    ):
        results = request_nominatim(
            session,
            params,
            pause_seconds,
        )

        for result in results:
            key = (
                result.get("osm_type"),
                result.get("osm_id"),
                result.get("place_id"),
            )

            if key in seen:
                continue

            seen.add(key)

            result = dict(result)
            result[
                "_query_variant"
            ] = variant_index

            all_results.append(
                result
            )

    return all_results


def score_candidate(
    query_address,
    query_name,
    candidate,
    geometry,
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

    address_score = float(
        fuzz.token_set_ratio(
            normalize_text(
                query_address
            ),
            normalize_text(
                display_name
            ),
        )
    )

    name_score = None

    if clean_text(query_name):
        name_score = float(
            fuzz.token_set_ratio(
                normalize_text(
                    query_name
                ),
                normalize_text(
                    display_name
                ),
            )
        )

    inside = bool(
        geometry.covers(
            Point(
                longitude,
                latitude,
            )
        )
    )

    if name_score is None:
        combined = address_score
    else:
        combined = (
            0.65
            * address_score
            + 0.35
            * name_score
        )

    # A candidate outside the municipality is never preferred.
    if not inside:
        combined -= 100.0

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
            address_score,

        "name_score":
            name_score,

        "combined_score":
            combined,

        "query_variant":
            candidate.get(
                "_query_variant"
            ),

        "osm_type":
            candidate.get(
                "osm_type"
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

    queries, queries_path = (
        load_queries(
            args
        )
    )

    context = load_context(
        args
    )

    # Derive target municipality name from the query context file only if supplied;
    # otherwise use Nominatim free-form target based on municipality code workflows.
    # For this recovery dataset, municipality_name is an explicit optional column.
    if (
        "municipality_name"
        in queries.columns
        and queries[
            "municipality_name"
        ].notna().any()
    ):
        municipality_names = (
            queries[
                "municipality_name"
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
                "municipality_name non univoco nel queries file."
            )

        municipality_name = (
            municipality_names[0]
        )

    else:
        # Read the municipality name from the already processed building dataset
        # would unnecessarily couple this generic recovery tool to MIM.
        # Therefore require it in the queries CSV when not inferable.
        raise RuntimeError(
            "Aggiungere la colonna municipality_name al queries file."
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
            "nominatim_school_recovery_"
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

    summary_rows = []
    candidate_rows = []

    for _, row in queries.iterrows():
        building_code = (
            row[
                "building_code"
            ]
        )

        query_address = clean_text(
            row[
                "query_address"
            ]
        )

        query_name = clean_text(
            row.get(
                "query_name"
            )
        )

        cache_key = (
            f"{args.municipality_code}|"
            f"{normalize_text(query_address)}|"
            f"{normalize_text(query_name)}"
        )

        if (
            not args.refresh
            and cache_key in cache
        ):
            raw_results = (
                cache[
                    cache_key
                ][
                    "results"
                ]
            )

            source = "cache"

        else:
            raw_results = query_variants(
                session=session,
                municipality_name=(
                    municipality_name
                ),
                query_address=(
                    query_address
                ),
                query_name=(
                    query_name
                ),
                context=context,
                pause_seconds=(
                    args.pause_seconds
                ),
            )

            cache[
                cache_key
            ] = {
                "building_code":
                    building_code,

                "query_address":
                    query_address,

                "query_name":
                    query_name,

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

            source = "nominatim"

        scored = []

        for candidate in raw_results:
            item = score_candidate(
                query_address=query_address,
                query_name=query_name,
                candidate=candidate,
                geometry=(
                    context[
                        "geometry"
                    ]
                ),
            )

            if item is not None:
                scored.append(
                    item
                )

        scored.sort(
            key=lambda item: (
                item[
                    "inside_target"
                ],
                item[
                    "combined_score"
                ],
            ),
            reverse=True,
        )

        top = (
            scored[0]
            if scored
            else None
        )

        summary_rows.append(
            {
                "building_code":
                    building_code,

                "query_address":
                    query_address,

                "query_name":
                    query_name,

                "query_source":
                    source,

                "candidate_display_name":
                    (
                        top[
                            "display_name"
                        ]
                        if top
                        else None
                    ),

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

                "candidate_inside_target":
                    (
                        top[
                            "inside_target"
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

                "candidate_name_score":
                    (
                        top[
                            "name_score"
                        ]
                        if top
                        else None
                    ),

                "candidate_combined_score":
                    (
                        top[
                            "combined_score"
                        ]
                        if top
                        else None
                    ),
            }
        )

        for rank, item in enumerate(
            scored,
            start=1,
        ):
            candidate_rows.append(
                {
                    "building_code":
                        building_code,

                    "query_address":
                        query_address,

                    "query_name":
                        query_name,

                    "candidate_rank":
                        rank,

                    **item,
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

    features_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        features_dir
        / (
            "school_geocoding_recovery_"
            f"{args.building_year}.csv"
        )
    )

    candidates_path = (
        features_dir
        / (
            "school_geocoding_recovery_candidates_"
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
        " SCHOOL GEOCODING RECOVERY"
    )
    print(
        "===================================="
    )

    print(
        f"Queries: {len(queries)}"
    )

    print(
        "\n=== RESULTS ==="
    )

    print(
        summary.to_string(
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
        "Gli alias e gli arricchimenti degli indirizzi restano nel CSV "
        "di validazione; questo script non modifica il raw MIM."
    )


if __name__ == "__main__":
    main()
