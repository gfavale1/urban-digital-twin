import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
PROCESSED_OSM_DIR = ROOT / "data" / "processed" / "osm"
PROCESSED_SALUTE_DIR = ROOT / "data" / "processed" / "salute"

FEATURES_COMPARISON_DIR = ROOT / "data" / "features" / "comparison"

DEFAULT_THRESHOLDS_M = [25, 50, 100, 250]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark entity/spatial quality tra registri istituzionali "
            "e OSM per Education e Health."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--school-year",
        default="202425",
    )

    parser.add_argument(
        "--health-reference-date",
        default="2025-06-30",
    )

    parser.add_argument(
        "--distance-thresholds-m",
        nargs="+",
        type=int,
        default=DEFAULT_THRESHOLDS_M,
        help="Soglie di prossimità Education. Default: 25 50 100 250.",
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
            "--municipality-code deve avere esattamente 6 cifre."
        )

    args.health_reference_date = (
        pd.Timestamp(args.health_reference_date)
        .normalize()
    )

    args.distance_thresholds_m = sorted(
        set(args.distance_thresholds_m)
    )

    return args


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def clean_text(value):
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
        "nan",
        "none",
        "null",
    }:
        return ""

    return value


def normalize_text(value):
    value = clean_text(value)

    if not value:
        return ""

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        ch
        for ch in value
        if not unicodedata.combining(ch)
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

    return value


def similarity(left, right):
    left = normalize_text(left)
    right = normalize_text(right)

    if not left or not right:
        return np.nan

    return float(
        fuzz.token_set_ratio(
            left,
            right,
        )
    )


def load_inputs(args):
    health_label = (
        args.health_reference_date
        .strftime("%Y%m%d")
    )

    paths = {
        "education_final":
            PROCESSED_MIM_DIR
            / args.municipality_code
            / f"school_sites_{args.school_year}.parquet",

        "education_osm":
            PROCESSED_OSM_DIR
            / args.municipality_code
            / "school_sites.parquet",

        "health_final":
            PROCESSED_SALUTE_DIR
            / args.municipality_code
            / f"health_sites_final_{health_label}.parquet",

        "health_osm":
            PROCESSED_SALUTE_DIR
            / args.municipality_code
            / "health_osm_candidates.parquet",
    }

    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{label}: {path}"
            )

    education_final = gpd.read_parquet(
        paths["education_final"]
    )

    education_osm = gpd.read_parquet(
        paths["education_osm"]
    )

    health_final = gpd.read_parquet(
        paths["health_final"]
    )

    health_osm = gpd.read_parquet(
        paths["health_osm"]
    )

    return (
        education_final,
        education_osm,
        health_final,
        health_osm,
        paths,
    )


def ensure_crs(gdf, label):
    if gdf.crs is None:
        # All project service-site coordinates are WGS84 by construction.
        gdf = gdf.set_crs(
            "EPSG:4326",
            allow_override=True,
        )

    return gdf


def nearest_pairs(
    left,
    right,
    left_id,
    right_id,
    left_name,
    right_name,
    left_address,
    right_address,
):
    left = ensure_crs(
        left.copy(),
        "left",
    )

    right = ensure_crs(
        right.copy(),
        "right",
    )

    metric_crs = (
        left.estimate_utm_crs()
    )

    if metric_crs is None:
        raise RuntimeError(
            "Impossibile determinare CRS metrico per benchmark Education."
        )

    left_metric = (
        left.to_crs(metric_crs)
    )

    right_metric = (
        right.to_crs(metric_crs)
    )

    right_columns = [
        right_id,
        right_name,
        right_address,
        "geometry",
    ]

    joined = gpd.sjoin_nearest(
        left_metric,
        right_metric[
            right_columns
        ],
        how="left",
        distance_col="nearest_distance_m",
        lsuffix="institutional",
        rsuffix="osm",
    )

    # Handle exact-distance ties deterministically.
    joined = (
        joined.sort_values(
            [
                left_id,
                "nearest_distance_m",
                right_id,
            ],
            na_position="last",
        )
        .drop_duplicates(
            subset=[left_id],
            keep="first",
        )
        .copy()
    )

    joined[
        "name_similarity"
    ] = [
        similarity(
            left_value,
            right_value,
        )
        for left_value, right_value
        in zip(
            joined[left_name],
            joined[right_name],
        )
    ]

    joined[
        "address_similarity"
    ] = [
        similarity(
            left_value,
            right_value,
        )
        for left_value, right_value
        in zip(
            joined[left_address],
            joined[right_address],
        )
    ]

    return joined


def education_benchmark(
    education_final,
    education_osm,
    thresholds,
):
    official = education_final.copy()
    osm = education_osm.copy()

    official_pairs = nearest_pairs(
        left=official,
        right=osm,
        left_id="school_site_id",
        right_id="site_id",
        left_name="name",
        right_name="site_name",
        left_address="address",
        right_address="site_address",
    )

    # Reverse direction for OSM-unpaired/proximity analysis.
    osm_pairs = nearest_pairs(
        left=osm,
        right=official,
        left_id="site_id",
        right_id="school_site_id",
        left_name="site_name",
        right_name="name",
        left_address="site_address",
        right_address="address",
    )

    summary = {
        "institutional_sites":
            int(len(official)),

        "osm_sites":
            int(len(osm)),

        "count_delta_osm_minus_institutional":
            int(
                len(osm)
                - len(official)
            ),

        "automatic_coordinates":
            int(
                (
                    official[
                        "coordinate_origin"
                    ]
                    == "automatic"
                )
                .sum()
            ),

        "ground_truth_coordinates":
            int(
                (
                    official[
                        "coordinate_origin"
                    ]
                    == "ground_truth"
                )
                .sum()
            ),

        "automatic_coordinate_share":
            float(
                (
                    official[
                        "coordinate_origin"
                    ]
                    == "automatic"
                )
                .mean()
            ),

        "ground_truth_coordinate_share":
            float(
                (
                    official[
                        "coordinate_origin"
                    ]
                    == "ground_truth"
                )
                .mean()
            ),

        "nearest_osm_distance_m": {
            "min":
                float(
                    official_pairs[
                        "nearest_distance_m"
                    ].min()
                ),

            "median":
                float(
                    official_pairs[
                        "nearest_distance_m"
                    ].median()
                ),

            "mean":
                float(
                    official_pairs[
                        "nearest_distance_m"
                    ].mean()
                ),

            "p90":
                float(
                    official_pairs[
                        "nearest_distance_m"
                    ].quantile(0.90)
                ),

            "max":
                float(
                    official_pairs[
                        "nearest_distance_m"
                    ].max()
                ),
        },

        "institutional_proximity_coverage":
            {},

        "osm_proximity_to_institutional":
            {},
    }

    for threshold in thresholds:
        institutional_count = int(
            (
                official_pairs[
                    "nearest_distance_m"
                ]
                <= threshold
            )
            .sum()
        )

        osm_count = int(
            (
                osm_pairs[
                    "nearest_distance_m"
                ]
                <= threshold
            )
            .sum()
        )

        summary[
            "institutional_proximity_coverage"
        ][
            str(threshold)
        ] = {
            "count":
                institutional_count,

            "share":
                (
                    institutional_count
                    / len(official)
                    if len(official)
                    else None
                ),

            "without_nearby_osm":
                int(
                    len(official)
                    - institutional_count
                ),
        }

        summary[
            "osm_proximity_to_institutional"
        ][
            str(threshold)
        ] = {
            "count":
                osm_count,

            "share":
                (
                    osm_count
                    / len(osm)
                    if len(osm)
                    else None
                ),

            "without_nearby_institutional":
                int(
                    len(osm)
                    - osm_count
                ),
        }

    return (
        official_pairs,
        osm_pairs,
        summary,
    )


def health_benchmark(
    health_final,
    health_osm,
):
    official = health_final.copy()
    osm = health_osm.copy()

    official[
        "benchmark_counterpart_status"
    ] = np.select(
        [
            (
                official[
                    "osm_match_status"
                ]
                == "matched_auto"
            ),

            (
                (
                    official[
                        "osm_match_status"
                    ]
                    != "matched_auto"
                )
                & (
                    official[
                        "coordinate_source"
                    ]
                    == "geocoder_osm_consensus"
                )
                & official[
                    "osm_candidate_id"
                ].notna()
            ),
        ],
        [
            "matched_auto",
            "spatial_consensus",
        ],
        default="no_confirmed_osm_counterpart",
    )

    counterpart_mask = (
        official[
            "benchmark_counterpart_status"
        ]
        .isin(
            [
                "matched_auto",
                "spatial_consensus",
            ]
        )
    )

    official[
        "has_confirmed_osm_counterpart"
    ] = counterpart_mask

    category_rows = []

    for subcategory in [
        "pharmacy",
        "hospital",
    ]:
        official_sub = official.loc[
            official[
                "subcategory"
            ]
            == subcategory
        ].copy()

        osm_sub = osm.loc[
            osm[
                "subcategory"
            ]
            == subcategory
        ].copy()

        matched_auto = int(
            (
                official_sub[
                    "benchmark_counterpart_status"
                ]
                == "matched_auto"
            )
            .sum()
        )

        spatial_consensus = int(
            (
                official_sub[
                    "benchmark_counterpart_status"
                ]
                == "spatial_consensus"
            )
            .sum()
        )

        confirmed = int(
            official_sub[
                "has_confirmed_osm_counterpart"
            ]
            .sum()
        )

        confirmed_ids = set(
            official_sub.loc[
                official_sub[
                    "has_confirmed_osm_counterpart"
                ],
                "osm_candidate_id",
            ]
            .dropna()
            .astype(str)
        )

        osm_ids = set(
            osm_sub[
                "candidate_id"
            ]
            .dropna()
            .astype(str)
        )

        unpaired_osm_ids = (
            osm_ids
            - confirmed_ids
        )

        category_rows.append(
            {
                "subcategory":
                    subcategory,

                "institutional_sites":
                    int(
                        len(
                            official_sub
                        )
                    ),

                "osm_pois":
                    int(
                        len(
                            osm_sub
                        )
                    ),

                "matched_auto":
                    matched_auto,

                "spatial_consensus":
                    spatial_consensus,

                "confirmed_osm_counterparts":
                    confirmed,

                "confirmed_counterpart_share":
                    (
                        confirmed
                        / len(
                            official_sub
                        )
                        if len(
                            official_sub
                        )
                        else None
                    ),

                "institutional_without_confirmed_osm_counterpart":
                    int(
                        len(
                            official_sub
                        )
                        - confirmed
                    ),

                "unique_confirmed_osm_ids":
                    int(
                        len(
                            confirmed_ids
                        )
                    ),

                "unpaired_osm_pois":
                    int(
                        len(
                            unpaired_osm_ids
                        )
                    ),
            }
        )

    source_present = int(
        official[
            "source_coordinate_present"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    suspicious = int(
        official[
            "source_coordinate_suspicious"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    summary = {
        "institutional_sites":
            int(
                len(
                    official
                )
            ),

        "osm_pois":
            int(
                len(
                    osm
                )
            ),

        "source_coordinate_present":
            source_present,

        "source_coordinate_missing":
            int(
                len(
                    official
                )
                - source_present
            ),

        "source_coordinate_suspicious":
            suspicious,

        "source_coordinate_suspicious_share_of_all":
            (
                suspicious
                / len(
                    official
                )
                if len(
                    official
                )
                else None
            ),

        "coordinate_source_counts":
            {
                str(key):
                    int(value)
                for key, value
                in official[
                    "coordinate_source"
                ]
                .value_counts(
                    dropna=False
                )
                .items()
            },

        "counterpart_status_counts":
            {
                str(key):
                    int(value)
                for key, value
                in official[
                    "benchmark_counterpart_status"
                ]
                .value_counts(
                    dropna=False
                )
                .items()
            },

        "by_subcategory":
            category_rows,
    }

    return (
        official,
        pd.DataFrame(
            category_rows
        ),
        summary,
    )


def make_summary_table(
    education_summary,
    health_category_df,
):
    rows = []

    education_100 = (
        education_summary[
            "institutional_proximity_coverage"
        ][
            "100"
        ]
        if "100"
        in education_summary[
            "institutional_proximity_coverage"
        ]
        else None
    )

    education_250 = (
        education_summary[
            "institutional_proximity_coverage"
        ][
            "250"
        ]
        if "250"
        in education_summary[
            "institutional_proximity_coverage"
        ]
        else None
    )

    rows.append(
        {
            "domain":
                "education",

            "subcategory":
                "school",

            "institutional_sites":
                education_summary[
                    "institutional_sites"
                ],

            "osm_sites_or_pois":
                education_summary[
                    "osm_sites"
                ],

            "institutional_covered_metric":
                (
                    education_100[
                        "count"
                    ]
                    if education_100
                    else None
                ),

            "institutional_covered_share":
                (
                    education_100[
                        "share"
                    ]
                    if education_100
                    else None
                ),

            "coverage_definition":
                (
                    "nearest OSM school site <=100 m"
                    if education_100
                    else None
                ),

            "institutional_without_counterpart_metric":
                (
                    education_250[
                        "without_nearby_osm"
                    ]
                    if education_250
                    else None
                ),

            "unpaired_osm_metric":
                (
                    education_summary[
                        "osm_proximity_to_institutional"
                    ][
                        "250"
                    ][
                        "without_nearby_institutional"
                    ]
                    if "250"
                    in education_summary[
                        "osm_proximity_to_institutional"
                    ]
                    else None
                ),

            "notes":
                (
                    "Education uses transparent spatial-proximity coverage; "
                    "it is not labelled exact entity recall."
                ),
        }
    )

    for _, row in health_category_df.iterrows():
        rows.append(
            {
                "domain":
                    "health",

                "subcategory":
                    row[
                        "subcategory"
                    ],

                "institutional_sites":
                    int(
                        row[
                            "institutional_sites"
                        ]
                    ),

                "osm_sites_or_pois":
                    int(
                        row[
                            "osm_pois"
                        ]
                    ),

                "institutional_covered_metric":
                    int(
                        row[
                            "confirmed_osm_counterparts"
                        ]
                    ),

                "institutional_covered_share":
                    float(
                        row[
                            "confirmed_counterpart_share"
                        ]
                    ),

                "coverage_definition":
                    (
                        "accepted entity match or geocoder-OSM spatial consensus"
                    ),

                "institutional_without_counterpart_metric":
                    int(
                        row[
                            "institutional_without_confirmed_osm_counterpart"
                        ]
                    ),

                "unpaired_osm_metric":
                    int(
                        row[
                            "unpaired_osm_pois"
                        ]
                    ),

                "notes":
                    (
                        "Unpaired OSM POIs are not automatically false positives."
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    args = parse_args()

    (
        education_final,
        education_osm,
        health_final,
        health_osm,
        paths,
    ) = load_inputs(
        args
    )

    (
        education_pairs,
        education_reverse_pairs,
        education_summary,
    ) = education_benchmark(
        education_final,
        education_osm,
        args.distance_thresholds_m,
    )

    (
        health_detail,
        health_category_df,
        health_summary,
    ) = health_benchmark(
        health_final,
        health_osm,
    )

    summary_table = make_summary_table(
        education_summary,
        health_category_df,
    )

    output_dir = (
        FEATURES_COMPARISON_DIR
        / args.municipality_code
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    education_pairs_csv = (
        output_dir
        / "education_institutional_nearest_osm.csv"
    )

    education_reverse_csv = (
        output_dir
        / "education_osm_nearest_institutional.csv"
    )

    health_detail_csv = (
        output_dir
        / "health_entity_benchmark.csv"
    )

    summary_csv = (
        output_dir
        / "entity_spatial_quality_summary.csv"
    )

    summary_json = (
        output_dir
        / "entity_spatial_quality_summary.json"
    )

    education_pairs.drop(
        columns=[
            "geometry",
            "index_osm",
            "index_institutional",
        ],
        errors="ignore",
    ).to_csv(
        education_pairs_csv,
        index=False,
        encoding="utf-8-sig",
    )

    education_reverse_pairs.drop(
        columns=[
            "geometry",
            "index_osm",
            "index_institutional",
        ],
        errors="ignore",
    ).to_csv(
        education_reverse_csv,
        index=False,
        encoding="utf-8-sig",
    )

    health_detail.drop(
        columns=["geometry"],
        errors="ignore",
    ).to_csv(
        health_detail_csv,
        index=False,
        encoding="utf-8-sig",
    )

    summary_table.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "generated_at_utc":
            utc_now_iso(),

        "municipality_code":
            args.municipality_code,

        "inputs": {
            key:
                str(value)
            for key, value
            in paths.items()
        },

        "education":
            education_summary,

        "health":
            health_summary,

        "methodological_notes": [
            (
                "Education completeness is reported as proximity coverage at "
                "multiple metric thresholds because the historical building "
                "matcher was not a complete one-to-one physical-site matcher."
            ),
            (
                "A school site farther than a proximity threshold is reported "
                "as lacking a nearby OSM counterpart at that threshold, not "
                "automatically as an OSM false negative."
            ),
            (
                "Health counterpart coverage uses accepted automatic entity "
                "matches plus geocoder-OSM spatial consensus. Unpaired OSM POIs "
                "are not automatically labelled false positives."
            ),
            (
                "Ground-truth coordinates in the Education PoC are retained as "
                "validation evidence and must not be interpreted as national "
                "zero-touch automation."
            ),
        ],
    }

    summary_json.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=============================================="
    )
    print(
        " ENTITY / SPATIAL QUALITY BENCHMARK"
    )
    print(
        "=============================================="
    )

    print(
        f"Comune: {args.municipality_code}"
    )

    print(
        "\nEDUCATION"
    )

    print(
        "  institutional physical sites: "
        f"{education_summary['institutional_sites']}"
    )

    print(
        "  OSM school sites: "
        f"{education_summary['osm_sites']}"
    )

    print(
        "  automatic coordinates: "
        f"{education_summary['automatic_coordinates']} "
        f"({education_summary['automatic_coordinate_share'] * 100:.2f}%)"
    )

    print(
        "  ground-truth coordinates: "
        f"{education_summary['ground_truth_coordinates']} "
        f"({education_summary['ground_truth_coordinate_share'] * 100:.2f}%)"
    )

    print(
        "  nearest OSM distance median: "
        f"{education_summary['nearest_osm_distance_m']['median']:.2f} m"
    )

    for threshold in args.distance_thresholds_m:
        item = education_summary[
            "institutional_proximity_coverage"
        ][
            str(threshold)
        ]

        reverse = education_summary[
            "osm_proximity_to_institutional"
        ][
            str(threshold)
        ]

        print(
            f"  institutional with OSM <= {threshold:>3} m: "
            f"{item['count']}/{education_summary['institutional_sites']} "
            f"({item['share'] * 100:.2f}%)"
        )

        print(
            f"  OSM with institutional <= {threshold:>3} m: "
            f"{reverse['count']}/{education_summary['osm_sites']} "
            f"({reverse['share'] * 100:.2f}%)"
        )

    print(
        "\nHEALTH"
    )

    print(
        "  institutional sites: "
        f"{health_summary['institutional_sites']}"
    )

    print(
        "  OSM POIs: "
        f"{health_summary['osm_pois']}"
    )

    print(
        "  source coordinate present: "
        f"{health_summary['source_coordinate_present']}"
    )

    print(
        "  suspicious source coordinates: "
        f"{health_summary['source_coordinate_suspicious']}"
    )

    for row in health_summary[
        "by_subcategory"
    ]:
        print(
            f"\n  {row['subcategory'].upper()}"
        )

        print(
            "    institutional: "
            f"{row['institutional_sites']}"
        )

        print(
            "    OSM: "
            f"{row['osm_pois']}"
        )

        print(
            "    matched_auto: "
            f"{row['matched_auto']}"
        )

        print(
            "    spatial_consensus: "
            f"{row['spatial_consensus']}"
        )

        print(
            "    confirmed counterpart: "
            f"{row['confirmed_osm_counterparts']}/"
            f"{row['institutional_sites']} "
            f"({row['confirmed_counterpart_share'] * 100:.2f}%)"
        )

        print(
            "    institutional without confirmed counterpart: "
            f"{row['institutional_without_confirmed_osm_counterpart']}"
        )

        print(
            "    unpaired OSM POIs: "
            f"{row['unpaired_osm_pois']}"
        )

    print(
        "\n=== OUTPUT ==="
    )

    for path in [
        education_pairs_csv,
        education_reverse_csv,
        health_detail_csv,
        summary_csv,
        summary_json,
    ]:
        print(
            f"✓ {path}"
        )


if __name__ == "__main__":
    main()
