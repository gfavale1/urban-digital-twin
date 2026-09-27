import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
PROCESSED_OSM_DIR = ROOT / "data" / "processed" / "osm"
FEATURES_MIM_DIR = ROOT / "data" / "features" / "mim"

DEFAULT_SCHOOL_YEAR = "202425"
DEFAULT_BUILDING_YEAR = "202425"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Confronta i nuovi edifici scolastici MIM con i siti "
            "scolastici OSM già consolidati."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
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
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--osm-sites-file",
        default=None,
        help=(
            "Path opzionale al file school_sites.parquet. "
            "Se omesso viene cercato automaticamente sotto "
            "data/processed/osm/<municipality-code>/."
        ),
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

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    value = (
        value.upper()
        .replace("`", "'")
        .replace("’", "'")
    )

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

    return value


def parse_json_list(value):
    value = clean_text(value)

    if value is None:
        return []

    try:
        decoded = json.loads(
            value
        )

        if isinstance(
            decoded,
            list,
        ):
            return [
                str(item)
                for item in decoded
                if clean_text(item)
            ]
    except Exception:
        pass

    return [
        value
    ]


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

    try:
        values = [
            float(value)
            for value in values
        ]
    except Exception:
        return None

    lon1, lat1, lon2, lat2 = values

    radius = 6371008.8

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lon2 - lon1
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
                max(
                    0.0,
                    1.0 - a,
                )
            ),
        )
    )


def detect_column(
    dataframe,
    candidates,
    required=False,
):
    for column in candidates:
        if column in dataframe.columns:
            return column

    if required:
        raise RuntimeError(
            "Nessuna delle colonne trovata: "
            + ", ".join(
                candidates
            )
        )

    return None


def load_inputs(args):
    directory = (
        PROCESSED_MIM_DIR
        / args.municipality_code
    )

    new_path = (
        directory
        / (
            "new_school_buildings_geocoded_"
            f"{args.building_year}.parquet"
        )
    )

    if args.osm_sites_file:
        osm_path = Path(
            args.osm_sites_file
        )

        if not osm_path.is_absolute():
            osm_path = (
                ROOT
                / osm_path
            )

    else:
        osm_path = (
            PROCESSED_OSM_DIR
            / args.municipality_code
            / "school_sites.parquet"
        )

    if not new_path.exists():
        raise FileNotFoundError(
            f"Nuovi edifici geocodificati non trovati: {new_path}"
        )

    if not osm_path.exists():
        raise FileNotFoundError(
            "Siti scolastici OSM non trovati. "
            f"Path cercato: {osm_path}. "
            "Usare --osm-sites-file per specificare un file diverso."
        )

    new_buildings = pd.read_parquet(
        new_path
    )

    osm_sites = pd.read_parquet(
        osm_path
    )

    return (
        new_buildings,
        osm_sites,
        new_path,
        osm_path,
    )


def prepare_osm_sites(
    osm_sites,
):
    site_id_col = detect_column(
        osm_sites,
        [
            "site_id",
            "osm_site_id",
            "id",
        ],
        required=True,
    )

    name_col = detect_column(
        osm_sites,
        [
            "site_name",
            "name",
            "osm_name",
        ],
    )

    address_col = detect_column(
        osm_sites,
        [
            "site_address",
            "address",
            "osm_address",
            "addr_full",
        ],
    )

    lon_col = detect_column(
        osm_sites,
        [
            "longitude",
            "lon",
            "x",
        ],
        required=True,
    )

    lat_col = detect_column(
        osm_sites,
        [
            "latitude",
            "lat",
            "y",
        ],
        required=True,
    )

    out = pd.DataFrame(
        {
            "osm_site_id":
                osm_sites[
                    site_id_col
                ],

            "osm_site_name":
                (
                    osm_sites[
                        name_col
                    ]
                    if name_col
                    else None
                ),

            "osm_site_address":
                (
                    osm_sites[
                        address_col
                    ]
                    if address_col
                    else None
                ),

            "osm_longitude":
                osm_sites[
                    lon_col
                ],

            "osm_latitude":
                osm_sites[
                    lat_col
                ],
        }
    )

    return out


def best_school_name_score(
    linked_names,
    osm_name,
):
    osm_norm = normalize_text(
        osm_name
    )

    if not osm_norm:
        return 0.0

    scores = []

    for school_name in linked_names:
        school_norm = normalize_text(
            school_name
        )

        if not school_norm:
            continue

        scores.append(
            float(
                fuzz.token_set_ratio(
                    school_norm,
                    osm_norm,
                )
            )
        )

    return (
        max(
            scores
        )
        if scores
        else 0.0
    )


def address_score(
    official_address,
    osm_address,
):
    official_norm = normalize_text(
        official_address
    )

    osm_norm = normalize_text(
        osm_address
    )

    # Missing evidence must not be interpreted as negative evidence.
    if (
        not official_norm
        or not osm_norm
    ):
        return None

    return float(
        fuzz.token_set_ratio(
            official_norm,
            osm_norm,
        )
    )


def distance_component(
    distance_m,
):
    if distance_m is None:
        return 0.0

    if distance_m <= 50:
        return 100.0

    if distance_m <= 100:
        return 90.0

    if distance_m <= 250:
        return 75.0

    if distance_m <= 500:
        return 55.0

    if distance_m <= 1000:
        return 30.0

    return 0.0


def score_pair(
    building,
    osm_site,
):
    linked_names = parse_json_list(
        building.get(
            "linked_school_names"
        )
    )

    name_score = (
        best_school_name_score(
            linked_names,
            osm_site[
                "osm_site_name"
            ],
        )
    )

    addr_score = address_score(
        building.get(
            "official_building_address"
        ),
        osm_site[
            "osm_site_address"
        ],
    )

    distance_m = haversine_m(
        building.get(
            "new_geocoder_longitude"
        ),
        building.get(
            "new_geocoder_latitude"
        ),
        osm_site[
            "osm_longitude"
        ],
        osm_site[
            "osm_latitude"
        ],
    )

    dist_score = distance_component(
        distance_m
    )

    # Missing evidence is not scored as zero. The base weights are
    # redistributed across the evidence that is actually available.
    #
    # Name is the strongest evidence. Address is important when OSM
    # provides it. Distance is only supporting evidence because the
    # Nominatim point can be street-level rather than building-level.
    evidence = [
        ("name", name_score, 0.50),
        ("address", addr_score, 0.35),
        ("distance", dist_score, 0.15),
    ]

    available = [
        (label, value, weight)
        for label, value, weight in evidence
        if value is not None
    ]

    weight_sum = sum(
        weight
        for _, _, weight in available
    )

    total_score = (
        sum(
            value * weight
            for _, value, weight in available
        )
        / weight_sum
        if weight_sum > 0
        else 0.0
    )

    evidence_count = len(
        available
    )

    return {
        "name_score":
            name_score,

        "address_score":
            addr_score,

        "distance_from_geocoder_m":
            distance_m,

        "distance_score":
            dist_score,

        "evidence_count":
            evidence_count,

        "total_score":
            total_score,
    }


def classify_match(
    candidates,
):
    if not candidates:
        return (
            "unresolved",
            "low",
            None,
        )

    top = candidates[0]

    second_score = (
        candidates[1][
            "total_score"
        ]
        if len(
            candidates
        ) > 1
        else 0.0
    )

    margin = (
        top[
            "total_score"
        ]
        - second_score
    )

    name_score = top[
        "name_score"
    ]

    addr_score = top[
        "address_score"
    ]

    total = top[
        "total_score"
    ]

    evidence_count = top.get(
        "evidence_count",
        0,
    )

    # Conservative thresholds: street-level Nominatim is not enough.
    # Automatic matching requires strong name evidence and at least
    # two independent evidence components.
    if (
        total >= 82
        and name_score >= 75
        and margin >= 10
        and evidence_count >= 2
    ):
        return (
            "matched_auto",
            "high",
            margin,
        )

    if (
        total >= 65
        and (
            name_score >= 55
            or (
                addr_score is not None
                and addr_score >= 85
            )
        )
        and evidence_count >= 2
    ):
        return (
            "review",
            "medium",
            margin,
        )

    return (
        "unresolved",
        "low",
        margin,
    )


def run_matching(
    new_buildings,
    osm_sites,
    top_k,
):
    summary_rows = []
    candidate_rows = []

    for _, building in new_buildings.iterrows():
        candidates = []

        for _, osm_site in osm_sites.iterrows():
            scores = score_pair(
                building,
                osm_site,
            )

            candidate = {
                "building_code":
                    building[
                        "building_code"
                    ],

                "official_building_address":
                    building.get(
                        "official_building_address"
                    ),

                "linked_school_names":
                    building.get(
                        "linked_school_names"
                    ),

                "osm_site_id":
                    osm_site[
                        "osm_site_id"
                    ],

                "osm_site_name":
                    osm_site[
                        "osm_site_name"
                    ],

                "osm_site_address":
                    osm_site[
                        "osm_site_address"
                    ],

                "osm_longitude":
                    osm_site[
                        "osm_longitude"
                    ],

                "osm_latitude":
                    osm_site[
                        "osm_latitude"
                    ],

                **scores,
            }

            candidates.append(
                candidate
            )

        candidates.sort(
            key=lambda row: (
                row[
                    "total_score"
                ],
                row[
                    "name_score"
                ],
                row[
                    "address_score"
                ],
            ),
            reverse=True,
        )

        top_candidates = (
            candidates[
                :top_k
            ]
        )

        for rank, candidate in enumerate(
            top_candidates,
            start=1,
        ):
            candidate[
                "candidate_rank"
            ] = rank

            candidate_rows.append(
                candidate
            )

        status, confidence, margin = (
            classify_match(
                candidates
            )
        )

        top = (
            candidates[0]
            if candidates
            else {}
        )

        summary_rows.append(
            {
                "building_code":
                    building[
                        "building_code"
                    ],

                "official_building_address":
                    building.get(
                        "official_building_address"
                    ),

                "linked_school_names":
                    building.get(
                        "linked_school_names"
                    ),

                "new_geocoder_status":
                    building.get(
                        "new_geocoder_status"
                    ),

                "new_geocoder_result_address":
                    building.get(
                        "new_geocoder_result_address"
                    ),

                "new_geocoder_longitude":
                    building.get(
                        "new_geocoder_longitude"
                    ),

                "new_geocoder_latitude":
                    building.get(
                        "new_geocoder_latitude"
                    ),

                "osm_match_status":
                    status,

                "osm_match_confidence":
                    confidence,

                "osm_score_margin":
                    margin,

                "osm_candidate_site_id":
                    top.get(
                        "osm_site_id"
                    ),

                "osm_candidate_site_name":
                    top.get(
                        "osm_site_name"
                    ),

                "osm_candidate_site_address":
                    top.get(
                        "osm_site_address"
                    ),

                "osm_candidate_longitude":
                    top.get(
                        "osm_longitude"
                    ),

                "osm_candidate_latitude":
                    top.get(
                        "osm_latitude"
                    ),

                "osm_match_score":
                    top.get(
                        "total_score"
                    ),

                "osm_name_score":
                    top.get(
                        "name_score"
                    ),

                "osm_address_score":
                    top.get(
                        "address_score"
                    ),

                "osm_distance_from_geocoder_m":
                    top.get(
                        "distance_from_geocoder_m"
                    ),

                "osm_evidence_count":
                    top.get(
                        "evidence_count"
                    ),
            }
        )

    return (
        pd.DataFrame(
            summary_rows
        ),
        pd.DataFrame(
            candidate_rows
        ),
    )


def save_outputs(
    summary,
    candidates,
    args,
):
    features_dir = (
        FEATURES_MIM_DIR
        / args.municipality_code
    )

    processed_dir = (
        PROCESSED_MIM_DIR
        / args.municipality_code
    )

    summary_csv = (
        features_dir
        / (
            "new_school_buildings_osm_matches_"
            f"{args.building_year}.csv"
        )
    )

    summary_parquet = (
        processed_dir
        / (
            "new_school_buildings_osm_matches_"
            f"{args.building_year}.parquet"
        )
    )

    candidates_csv = (
        features_dir
        / (
            "new_school_buildings_osm_candidates_"
            f"{args.building_year}.csv"
        )
    )

    summary.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_parquet(
        summary_parquet,
        index=False,
    )

    candidates.to_csv(
        candidates_csv,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        summary_csv,
        summary_parquet,
        candidates_csv,
    )


def main():
    args = parse_args()

    print(
        "\n===================================="
    )
    print(
        " MATCH NEW SCHOOL BUILDINGS ↔ OSM"
    )
    print(
        "===================================="
    )

    (
        new_buildings,
        osm_raw,
        new_path,
        osm_path,
    ) = load_inputs(
        args
    )

    osm_sites = prepare_osm_sites(
        osm_raw
    )

    print(
        f"New buildings: {len(new_buildings)}"
    )

    print(
        f"OSM school sites: {len(osm_sites)}"
    )

    summary, candidates = run_matching(
        new_buildings,
        osm_sites,
        args.top_k,
    )

    (
        summary_csv,
        summary_parquet,
        candidates_csv,
    ) = save_outputs(
        summary,
        candidates,
        args,
    )

    print(
        "\n=== RESULTS ==="
    )

    columns = [
        "building_code",
        "official_building_address",
        "osm_match_status",
        "osm_match_confidence",
        "osm_candidate_site_name",
        "osm_match_score",
        "osm_name_score",
        "osm_address_score",
        "osm_distance_from_geocoder_m",
        "osm_evidence_count",
        "osm_score_margin",
    ]

    print(
        summary[
            columns
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
        summary[
            "osm_match_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ {summary_csv}"
    )

    print(
        f"✓ {summary_parquet}"
    )

    print(
        f"✓ {candidates_csv}"
    )

    print(
        "\nNOTA:"
    )

    print(
        "Nessun candidato OSM viene applicato al dataset finale "
        "in questo step."
    )

    print(
        "Le soglie sono conservative perché i punti Nominatim "
        "possono rappresentare solamente la strada."
    )


if __name__ == "__main__":
    main()
