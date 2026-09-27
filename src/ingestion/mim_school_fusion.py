import argparse
import math
from pathlib import Path

import pandas as pd


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

DEFAULT_SCHOOL_YEAR = "202627"

CONSENSUS_DISTANCE_M = 120.0
AGREEMENT_DISTANCE_M = 150.0
CONFLICT_DISTANCE_M = 500.0


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fusion conservativa tra matching scolastico "
            "MIM-OSM e geocoding Nominatim V2."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--school-year",
        default=DEFAULT_SCHOOL_YEAR,
        help="Anno scolastico, es. 202627.",
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
            "municipality-code deve avere 6 cifre."
        )

    return args


# ============================================================
# HELPERS
# ============================================================

def is_missing(value):
    if value is None:
        return True

    try:
        return bool(
            pd.isna(value)
        )
    except Exception:
        return False


def as_float(value):
    if is_missing(value):
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def as_bool(value):
    if is_missing(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    text_value = (
        str(value)
        .strip()
        .lower()
    )

    if text_value in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if text_value in {
        "false",
        "0",
        "no",
        "n",
    }:
        return False

    return None


def haversine_m(
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

    earth_radius_m = 6371008.8

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

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
        math.sqrt(1.0 - a),
    )

    return (
        earth_radius_m
        * c
    )


def normalized_lower(value):
    if is_missing(value):
        return None

    return (
        str(value)
        .strip()
        .lower()
    )


# ============================================================
# INPUT
# ============================================================

def load_inputs(
    municipality_code,
    school_year,
):
    directory = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    registry_path = (
        directory
        / (
            "schools_registry_"
            f"{school_year}.parquet"
        )
    )

    osm_path = (
        directory
        / (
            "school_locations_"
            f"{school_year}_v2.parquet"
        )
    )

    geocoder_path = (
        directory
        / (
            "school_geocoding_"
            f"{school_year}_v2.parquet"
        )
    )

    for path in [
        registry_path,
        osm_path,
        geocoder_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"File richiesto non trovato: {path}"
            )

    registry = pd.read_parquet(
        registry_path
    )

    osm = pd.read_parquet(
        osm_path
    )

    geocoder = pd.read_parquet(
        geocoder_path
    )

    for (
        name,
        dataframe,
    ) in [
        ("registry", registry),
        ("osm", osm),
        ("geocoder", geocoder),
    ]:
        if (
            "school_code"
            not in dataframe.columns
        ):
            raise RuntimeError(
                f"{name}: school_code mancante."
            )

        if dataframe[
            "school_code"
        ].duplicated().any():
            raise RuntimeError(
                f"{name}: school_code duplicati."
            )

    return (
        registry,
        osm,
        geocoder,
    )


# ============================================================
# GEOCODER CLASSIFICATION
# ============================================================

def geocoder_is_school_poi(row):
    result_class = (
        normalized_lower(
            row.get(
                "geocoder_result_class"
            )
        )
    )

    result_type = (
        normalized_lower(
            row.get(
                "geocoder_result_type"
            )
        )
    )

    return (
        result_class
        in {
            "amenity",
            "education",
        }
        and result_type
        in {
            "school",
            "kindergarten",
            "college",
        }
    )


def geocoder_is_road_like(row):
    result_class = (
        normalized_lower(
            row.get(
                "geocoder_result_class"
            )
        )
    )

    result_type = (
        normalized_lower(
            row.get(
                "geocoder_result_type"
            )
        )
    )

    if result_class == "highway":
        return True

    return result_type in {
        "road",
        "residential",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "service",
        "pedestrian",
        "footway",
    }


# ============================================================
# FUSION LOGIC
# ============================================================

def fuse_row(row):
    osm_status = row.get(
        "osm_status"
    )

    osm_geometry_accepted = (
        as_bool(
            row.get(
                "osm_geometry_accepted"
            )
        )
        is True
    )

    osm_lon = as_float(
        row.get(
            "osm_longitude"
        )
    )

    osm_lat = as_float(
        row.get(
            "osm_latitude"
        )
    )

    osm_candidate_lon = as_float(
        row.get(
            "osm_candidate_longitude"
        )
    )

    osm_candidate_lat = as_float(
        row.get(
            "osm_candidate_latitude"
        )
    )

    geocoder_status = row.get(
        "geocoder_status"
    )

    geocoder_inside = (
        as_bool(
            row.get(
                "geocoder_inside_municipality"
            )
        )
        is True
    )

    geocoder_lon = as_float(
        row.get(
            "geocoder_longitude"
        )
    )

    geocoder_lat = as_float(
        row.get(
            "geocoder_latitude"
        )
    )

    geocoder_has_geometry = (
        geocoder_lon is not None
        and geocoder_lat is not None
    )

    distance_m = haversine_m(
        osm_candidate_lon,
        osm_candidate_lat,
        geocoder_lon,
        geocoder_lat,
    )

    school_poi = (
        geocoder_is_school_poi(
            row
        )
    )

    road_like = (
        geocoder_is_road_like(
            row
        )
    )

    osm_name_score = (
        as_float(
            row.get(
                "osm_name_score"
            )
        )
        or 0.0
    )

    geocoder_address_score = (
        as_float(
            row.get(
                "geocoder_address_score"
            )
        )
        or 0.0
    )

    geocoder_score = (
        as_float(
            row.get(
                "geocoder_score"
            )
        )
        or 0.0
    )

    requires_service_review = (
        as_bool(
            row.get(
                "requires_service_review"
            )
        )
        is True
    )

    # --------------------------------------------------------
    # CROSS-CHECK FLAGS
    # --------------------------------------------------------

    if (
        distance_m is not None
        and geocoder_inside
    ):
        if (
            distance_m
            <= AGREEMENT_DISTANCE_M
        ):
            crosscheck = (
                "agreement"
            )

        elif (
            distance_m
            >= CONFLICT_DISTANCE_M
        ):
            crosscheck = (
                "conflict"
            )

        else:
            crosscheck = (
                "partial_agreement"
            )

    elif (
        geocoder_has_geometry
        and not geocoder_inside
    ):
        crosscheck = (
            "geocoder_outside"
        )

    else:
        crosscheck = (
            "not_available"
        )

    # --------------------------------------------------------
    # RULE 1
    # OSM AUTO WINS
    # --------------------------------------------------------

    if (
        osm_status
        == "matched_auto"
        and osm_geometry_accepted
        and osm_lon is not None
        and osm_lat is not None
    ):
        return {
            "final_status":
                "accepted_osm",

            "final_geometry_source":
                "OSM",

            "final_longitude":
                osm_lon,

            "final_latitude":
                osm_lat,

            "location_confidence":
                "high",

            "fusion_reason":
                (
                    "OSM V2 auto-match forte; "
                    "geocoder usato solo come cross-check."
                ),

            "osm_geocoder_distance_m":
                distance_m,

            "crosscheck_status":
                crosscheck,

            "geocoder_school_poi":
                school_poi,

            "geocoder_road_like":
                road_like,

            "service_role_status":
                (
                    "review_required"
                    if requires_service_review
                    else "standard_candidate"
                ),
        }

    # --------------------------------------------------------
    # RULE 2
    # STRONG SCHOOL POI FROM GEOCODER
    # --------------------------------------------------------

    if (
        geocoder_status
        == "strong_candidate"
        and geocoder_inside
        and geocoder_has_geometry
        and school_poi
    ):
        return {
            "final_status":
                "accepted_geocoder_poi",

            "final_geometry_source":
                "Nominatim",

            "final_longitude":
                geocoder_lon,

            "final_latitude":
                geocoder_lat,

            "location_confidence":
                "high",

            "fusion_reason":
                (
                    "Geocoder forte dentro il comune "
                    "e risultato classificato come POI scolastico."
                ),

            "osm_geocoder_distance_m":
                distance_m,

            "crosscheck_status":
                crosscheck,

            "geocoder_school_poi":
                school_poi,

            "geocoder_road_like":
                road_like,

            "service_role_status":
                (
                    "review_required"
                    if requires_service_review
                    else "standard_candidate"
                ),
        }

    # --------------------------------------------------------
    # RULE 3
    # OSM REVIEW + INDEPENDENT GEOCODER CONSENSUS
    # --------------------------------------------------------

    if (
        osm_status == "review"
        and geocoder_inside
        and geocoder_has_geometry
        and distance_m is not None
        and distance_m
        <= CONSENSUS_DISTANCE_M
        and osm_name_score >= 85.0
        and geocoder_address_score >= 80.0
        and geocoder_score >= 65.0
    ):
        return {
            "final_status":
                "accepted_consensus",

            # OSM site position is preferred because it represents
            # an educational feature rather than a road centroid.
            "final_geometry_source":
                "OSM_consensus",

            "final_longitude":
                osm_candidate_lon,

            "final_latitude":
                osm_candidate_lat,

            "location_confidence":
                "high",

            "fusion_reason":
                (
                    "OSM review e geocoder indipendente "
                    "convergono spazialmente e semanticamente."
                ),

            "osm_geocoder_distance_m":
                distance_m,

            "crosscheck_status":
                crosscheck,

            "geocoder_school_poi":
                school_poi,

            "geocoder_road_like":
                road_like,

            "service_role_status":
                (
                    "review_required"
                    if requires_service_review
                    else "standard_candidate"
                ),
        }

    # --------------------------------------------------------
    # RULE 4
    # STRONG ADDRESS ONLY
    # --------------------------------------------------------

    if (
        geocoder_status
        == "strong_candidate"
        and geocoder_inside
        and geocoder_has_geometry
    ):
        return {
            "final_status":
                "review_geocoder_address",

            "final_geometry_source":
                None,

            "final_longitude":
                None,

            "final_latitude":
                None,

            "location_confidence":
                "review",

            "fusion_reason":
                (
                    "Geocoder forte ma il risultato non è "
                    "un POI scolastico; possibile strada/civico."
                ),

            "osm_geocoder_distance_m":
                distance_m,

            "crosscheck_status":
                crosscheck,

            "geocoder_school_poi":
                school_poi,

            "geocoder_road_like":
                road_like,

            "service_role_status":
                (
                    "review_required"
                    if requires_service_review
                    else "standard_candidate"
                ),
        }

    # --------------------------------------------------------
    # RULE 5
    # EXTERNAL RESULT
    # --------------------------------------------------------

    if (
        geocoder_status
        == "outside_candidate"
    ):
        return {
            "final_status":
                "review_outside",

            "final_geometry_source":
                None,

            "final_longitude":
                None,

            "final_latitude":
                None,

            "location_confidence":
                "review",

            "fusion_reason":
                (
                    "Geocoder restituisce una posizione "
                    "esterna al boundary ISTAT."
                ),

            "osm_geocoder_distance_m":
                distance_m,

            "crosscheck_status":
                crosscheck,

            "geocoder_school_poi":
                school_poi,

            "geocoder_road_like":
                road_like,

            "service_role_status":
                (
                    "review_required"
                    if requires_service_review
                    else "standard_candidate"
                ),
        }

    # --------------------------------------------------------
    # RULE 6
    # REVIEW
    # --------------------------------------------------------

    if (
        osm_status == "review"
        or geocoder_status
        in {
            "review",
            "weak_candidate",
        }
    ):
        return {
            "final_status":
                "review",

            "final_geometry_source":
                None,

            "final_longitude":
                None,

            "final_latitude":
                None,

            "location_confidence":
                "review",

            "fusion_reason":
                (
                    "Evidenza presente ma insufficiente "
                    "per accettare automaticamente la geometria."
                ),

            "osm_geocoder_distance_m":
                distance_m,

            "crosscheck_status":
                crosscheck,

            "geocoder_school_poi":
                school_poi,

            "geocoder_road_like":
                road_like,

            "service_role_status":
                (
                    "review_required"
                    if requires_service_review
                    else "standard_candidate"
                ),
        }

    # --------------------------------------------------------
    # RULE 7
    # UNRESOLVED
    # --------------------------------------------------------

    return {
        "final_status":
            "unresolved",

        "final_geometry_source":
            None,

        "final_longitude":
            None,

        "final_latitude":
            None,

        "location_confidence":
            "unresolved",

        "fusion_reason":
            (
                "Nessuna evidenza sufficientemente "
                "affidabile per una localizzazione."
            ),

        "osm_geocoder_distance_m":
            distance_m,

        "crosscheck_status":
            crosscheck,

        "geocoder_school_poi":
            school_poi,

        "geocoder_road_like":
            road_like,

        "service_role_status":
            (
                "review_required"
                if requires_service_review
                else "standard_candidate"
            ),
    }


# ============================================================
# BUILD DATASET
# ============================================================

def build_fusion_dataset(
    registry,
    osm,
    geocoder,
):
    osm_columns = {
        "school_code":
            "school_code",

        "geocoding_status":
            "osm_status",

        "geometry_accepted":
            "osm_geometry_accepted",

        "status_reason":
            "osm_status_reason",

        "geocoding_score":
            "osm_score",

        "name_score":
            "osm_name_score",

        "address_score":
            "osm_address_score",

        "score_margin":
            "osm_score_margin",

        "candidate_site_id":
            "osm_candidate_site_id",

        "candidate_site_name":
            "osm_candidate_site_name",

        "candidate_site_address":
            "osm_candidate_site_address",

        "candidate_longitude":
            "osm_candidate_longitude",

        "candidate_latitude":
            "osm_candidate_latitude",

        "longitude":
            "osm_longitude",

        "latitude":
            "osm_latitude",

        "requires_service_review":
            "requires_service_review",

        "service_review_reason":
            "service_review_reason",
    }

    available_osm_columns = [
        column
        for column in osm_columns
        if column in osm.columns
    ]

    osm_subset = (
        osm[
            available_osm_columns
        ]
        .rename(
            columns={
                column:
                    osm_columns[
                        column
                    ]
                for column
                in available_osm_columns
            }
        )
    )

    geocoder_columns = {
        "school_code":
            "school_code",

        "geocoder_status":
            "geocoder_status",

        "latitude":
            "geocoder_latitude",

        "longitude":
            "geocoder_longitude",

        "display_name":
            "geocoder_display_name",

        "result_name":
            "geocoder_result_name",

        "result_address":
            "geocoder_result_address",

        "result_locality":
            "geocoder_result_locality",

        "result_type":
            "geocoder_result_type",

        "result_class":
            "geocoder_result_class",

        "address_score":
            "geocoder_address_score",

        "name_score":
            "geocoder_name_score",

        "geocoder_score":
            "geocoder_score",

        "inside_municipality":
            "geocoder_inside_municipality",

        "distance_to_municipality_m":
            "geocoder_distance_to_municipality_m",

        "query_type":
            "geocoder_query_type",
    }

    available_geocoder_columns = [
        column
        for column in geocoder_columns
        if column in geocoder.columns
    ]

    geocoder_subset = (
        geocoder[
            available_geocoder_columns
        ]
        .rename(
            columns={
                column:
                    geocoder_columns[
                        column
                    ]
                for column
                in available_geocoder_columns
            }
        )
    )

    base_columns = [
        "school_code",
        "school_name",
        "school_type",
        "school_ownership",
        "address",
        "postal_code",
        "municipality_name",
        "province_name",
        "region_name",
        "municipality_istat_code",
    ]

    available_base_columns = [
        column
        for column in base_columns
        if column in registry.columns
    ]

    output = (
        registry[
            available_base_columns
        ]
        .merge(
            osm_subset,
            on="school_code",
            how="left",
            validate="one_to_one",
        )
        .merge(
            geocoder_subset,
            on="school_code",
            how="left",
            validate="one_to_one",
        )
    )

    fusion_rows = []

    for _, row in output.iterrows():
        fusion_rows.append(
            fuse_row(
                row.to_dict()
            )
        )

    fusion_dataframe = (
        pd.DataFrame(
            fusion_rows
        )
    )

    output = pd.concat(
        [
            output.reset_index(
                drop=True
            ),
            fusion_dataframe.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    if (
        output["school_code"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "school_code duplicati dopo la fusion."
        )

    return output


# ============================================================
# SAVE / SUMMARY
# ============================================================

def save_output(
    dataframe,
    municipality_code,
    school_year,
):
    directory = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    path = (
        directory
        / (
            "school_locations_final_"
            f"{school_year}.parquet"
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

    return path


def print_summary(
    dataframe,
):
    print(
        "\n===================================="
    )

    print(
        " SCHOOL LOCATION FUSION COMPLETATA"
    )

    print(
        "===================================="
    )

    print(
        f"Record MIM: {len(dataframe)}"
    )

    print(
        "\nFinal status:"
    )

    print(
        dataframe[
            "final_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    accepted_mask = (
        dataframe[
            "final_status"
        ]
        .isin(
            [
                "accepted_osm",
                "accepted_geocoder_poi",
                "accepted_consensus",
            ]
        )
    )

    accepted_count = int(
        accepted_mask.sum()
    )

    print(
        "\nLocalizzazioni accettate: "
        f"{accepted_count}"
    )

    print(
        "Da revisionare/non risolte: "
        f"{len(dataframe) - accepted_count}"
    )

    print(
        "\nCross-check OSM ↔ geocoder:"
    )

    print(
        dataframe[
            "crosscheck_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nService role:"
    )

    print(
        dataframe[
            "service_role_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    conflicts = dataframe[
        dataframe[
            "crosscheck_status"
        ]
        .isin(
            [
                "conflict",
                "geocoder_outside",
            ]
        )
    ]

    print(
        "\n=== CONFLITTI ==="
    )

    if conflicts.empty:
        print(
            "Nessuno."
        )

    else:
        columns = [
            "school_code",
            "school_name",
            "address",
            "osm_status",
            "geocoder_status",
            "osm_candidate_site_name",
            "geocoder_display_name",
            "osm_geocoder_distance_m",
            "final_status",
        ]

        columns = [
            column
            for column in columns
            if column in conflicts.columns
        ]

        print(
            conflicts[
                columns
            ]
            .sort_values(
                "osm_geocoder_distance_m",
                ascending=False,
                na_position="last",
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n=== LOCALIZZAZIONI ACCETTATE ==="
    )

    accepted = dataframe[
        accepted_mask
    ]

    if accepted.empty:
        print(
            "Nessuna."
        )

    else:
        columns = [
            "school_code",
            "school_name",
            "address",
            "final_status",
            "final_geometry_source",
            "location_confidence",
            "final_longitude",
            "final_latitude",
            "crosscheck_status",
            "service_role_status",
        ]

        print(
            accepted[
                columns
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
        " MIM SCHOOL LOCATION FUSION"
    )

    print(
        "===================================="
    )

    (
        registry,
        osm,
        geocoder,
    ) = load_inputs(
        args.municipality_code,
        args.school_year,
    )

    print(
        "\n=== INPUT ==="
    )

    print(
        f"Registry MIM: {len(registry)}"
    )

    print(
        f"OSM matching V2: {len(osm)}"
    )

    print(
        f"Geocoding V2: {len(geocoder)}"
    )

    final_dataset = (
        build_fusion_dataset(
            registry=registry,
            osm=osm,
            geocoder=geocoder,
        )
    )

    save_output(
        final_dataset,
        args.municipality_code,
        args.school_year,
    )

    print_summary(
        final_dataset
    )


if __name__ == "__main__":
    main()
