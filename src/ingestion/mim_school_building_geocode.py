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

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = (
    ROOT / "data" / "processed" / "mim"
)

RAW_GEOCODING_DIR = (
    ROOT
    / "data"
    / "raw"
    / "mim"
    / "geocoding"
    / "nominatim_buildings"
)

DEFAULT_BUILDING_YEAR = "202425"

NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/search"
)

REQUEST_DELAY_SECONDS = 1.1


# ============================================================
# CLI
# ============================================================

def parse_args():
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
        default=DEFAULT_BUILDING_YEAR,
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

def get_database_engine():
    load_dotenv(
        ROOT / ".env"
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


def load_municipality(
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

def load_buildings(
    municipality_code,
    building_year,
):
    municipality_dir = (
        PROCESSED_MIM_DIR
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
        return json.load(
            file
        )


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

        save_cache(
            self.cache
        )

        return results


# ============================================================
# SPATIAL HELPERS
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

def build_structured_params(
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
            viewbox_string(
                municipality
            ),

        "bounded":
            1,
    }


def clean_params(params):
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


def result_road_address(
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


def score_candidate(
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

    spatial = spatial_metrics(
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
            result_road_address(
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


def geocode_building(
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

    params = clean_params(
        build_structured_params(
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
            score_candidate(
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
                viewbox_string(
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
                score_candidate(
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

def build_dataset(
    buildings,
    municipality,
    client,
):
    (
        metric_crs,
        boundary_metric,
    ) = build_metric_boundary(
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

        geocoded = geocode_building(
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

def save_output(
    dataframe,
    municipality_code,
    building_year,
):
    path = (
        PROCESSED_MIM_DIR
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


def print_summary(
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


def main():
    args = parse_args()

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
        get_database_engine()
    )

    municipality = (
        load_municipality(
            engine,
            args.municipality_code,
        )
    )

    buildings = load_buildings(
        args.municipality_code,
        args.building_year,
    )

    cache = load_cache()

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

    client = NominatimClient(
        cache=cache,
        refresh=args.refresh,
    )

    dataset = build_dataset(
        buildings=buildings,
        municipality=municipality,
        client=client,
    )

    save_output(
        dataset,
        args.municipality_code,
        args.building_year,
    )

    print_summary(
        dataset
    )


if __name__ == "__main__":
    main()
