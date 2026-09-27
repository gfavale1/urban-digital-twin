import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

FEATURES_ACCESSIBILITY_DIR = (
    ROOT / "data" / "features" / "accessibility"
)

FEATURES_COMPARISON_DIR = (
    ROOT / "data" / "features" / "comparison"
)

GROUPS = [
    "school",
    "pharmacy",
    "hospital",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Confronta in modo controllato lo scenario OSM-only con "
            "lo scenario institutional+OSM (enriched), verificando prima "
            "che popolazione e origin nodes siano identici."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--census-year",
        default="2023",
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
        "--thresholds-min",
        nargs="+",
        type=int,
        default=[10, 15, 20],
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    args.health_reference_date = (
        pd.Timestamp(
            args.health_reference_date
        )
        .normalize()
    )

    args.thresholds_min = sorted(
        set(
            args.thresholds_min
        )
    )

    return args


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def assert_close(
    left,
    right,
    label,
    atol=1e-6,
):
    if not np.isclose(
        float(left),
        float(right),
        atol=atol,
        rtol=1e-10,
    ):
        raise AssertionError(
            f"{label}: {left} != {right}"
        )


def get_paths(args):
    acc_dir = (
        FEATURES_ACCESSIBILITY_DIR
        / args.municipality_code
    )

    health_label = (
        args.health_reference_date
        .strftime("%Y%m%d")
    )

    enriched_suffix = (
        f"{args.census_year}_"
        f"{args.school_year}_"
        f"{health_label}"
    )

    return {
        "origins_enriched":
            acc_dir
            / (
                "walking_accessibility_origins_"
                f"{enriched_suffix}.parquet"
            ),

        "origins_osm":
            acc_dir
            / (
                "walking_accessibility_origins_"
                f"osm_only_{args.census_year}.parquet"
            ),

        "summary_enriched":
            acc_dir
            / (
                "walking_accessibility_summary_"
                f"{enriched_suffix}.json"
            ),

        "summary_osm":
            acc_dir
            / (
                "walking_accessibility_summary_"
                f"osm_only_{args.census_year}.json"
            ),

        # Count the exact service supply consumed by the accessibility
        # algorithm, after municipality-boundary eligibility and network
        # snapping. Using the pre-filter canonical layer here would make the
        # reported service counts inconsistent with the routing results.
        "services_enriched":
            acc_dir
            / (
                "service_network_nodes_"
                f"{args.school_year}_"
                f"{health_label}.parquet"
            ),

        "services_osm":
            acc_dir
            / "service_network_nodes_osm_only.parquet",
    }


def validate_common_origins(
    enriched,
    osm,
):
    required = [
        "census_section_code",
        "network_node_id",
        "assigned_population",
    ]

    for column in required:
        if column not in enriched.columns:
            raise RuntimeError(
                f"Enriched origins: colonna mancante {column}"
            )

        if column not in osm.columns:
            raise RuntimeError(
                f"OSM-only origins: colonna mancante {column}"
            )

    if len(
        enriched
    ) != len(
        osm
    ):
        raise RuntimeError(
            "Numero origin rows diverso tra scenari: "
            f"{len(enriched)} vs {len(osm)}"
        )

    sort_columns = [
        "census_section_code",
        "network_node_id",
        "assigned_population",
    ]

    left = (
        enriched[
            sort_columns
        ]
        .copy()
        .sort_values(
            sort_columns
        )
        .reset_index(
            drop=True
        )
    )

    right = (
        osm[
            sort_columns
        ]
        .copy()
        .sort_values(
            sort_columns
        )
        .reset_index(
            drop=True
        )
    )

    if not (
        left[
            "census_section_code"
        ]
        .astype(str)
        .equals(
            right[
                "census_section_code"
            ]
            .astype(str)
        )
    ):
        raise RuntimeError(
            "Le sezioni censuarie non coincidono tra gli scenari."
        )

    if not (
        left[
            "network_node_id"
        ]
        .astype(str)
        .equals(
            right[
                "network_node_id"
            ]
            .astype(str)
        )
    ):
        raise RuntimeError(
            "Gli origin network nodes non coincidono tra gli scenari."
        )

    if not np.allclose(
        left[
            "assigned_population"
        ]
        .astype(float)
        .to_numpy(),
        right[
            "assigned_population"
        ]
        .astype(float)
        .to_numpy(),
        atol=1e-9,
        rtol=1e-12,
    ):
        raise RuntimeError(
            "I pesi di popolazione non coincidono tra gli scenari."
        )

    population_enriched = float(
        enriched[
            "assigned_population"
        ]
        .sum()
    )

    population_osm = float(
        osm[
            "assigned_population"
        ]
        .sum()
    )

    assert_close(
        population_enriched,
        population_osm,
        "population equality",
    )

    return {
        "origin_rows":
            int(
                len(
                    enriched
                )
            ),

        "population":
            population_enriched,

        "same_sections":
            True,

        "same_network_nodes":
            True,

        "same_population_weights":
            True,
    }


def validate_threshold_monotonicity(
    origins,
    label,
    thresholds,
):
    failures = []

    for group in GROUPS:
        previous = None

        for threshold in thresholds:
            column = (
                f"{group}_within_"
                f"{threshold}_min"
            )

            if column not in origins.columns:
                failures.append(
                    f"{label}: missing {column}"
                )
                continue

            current = (
                origins[
                    column
                ]
                .astype(bool)
                .to_numpy()
            )

            if previous is not None:
                violations = int(
                    np.sum(
                        previous
                        & ~current
                    )
                )

                if violations:
                    failures.append(
                        f"{label} {group}: "
                        f"{violations} monotonicity violations"
                    )

            previous = current

    if failures:
        raise RuntimeError(
            "\n".join(
                failures
            )
        )


def service_counts(
    services,
):
    """
    Conta esclusivamente i servizi effettivamente eleggibili
    per il calcolo di accessibility.

    Il numero totale di record canonici è una metrica di data
    completeness distinta e non deve essere confuso con la
    supply utilizzata nel routing.
    """

    if "usable_for_accessibility" not in services.columns:
        raise RuntimeError(
            "Colonna usable_for_accessibility assente "
            "dal canonical service layer."
        )

    usable = services.loc[
        services[
            "usable_for_accessibility"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()

    return {
        "school":
            int(
                (
                    usable[
                        "category"
                    ]
                    == "education"
                )
                .sum()
            ),

        "pharmacy":
            int(
                (
                    usable[
                        "subcategory"
                    ]
                    == "pharmacy"
                )
                .sum()
            ),

        "hospital":
            int(
                (
                    usable[
                        "subcategory"
                    ]
                    == "hospital"
                )
                .sum()
            ),
    }


def municipality_summary(
    payload,
):
    return payload[
        "municipality_summary"
    ]


def comparison_rows(
    osm_summary,
    enriched_summary,
    osm_counts,
    enriched_counts,
    thresholds,
):
    rows = []

    for group in GROUPS:
        osm_count = int(
            osm_counts[
                group
            ]
        )

        enriched_count = int(
            enriched_counts[
                group
            ]
        )

        rows.append(
            {
                "service_group":
                    group,

                "metric":
                    "usable_service_count",

                "unit":
                    "count",

                "osm_only":
                    osm_count,

                "enriched":
                    enriched_count,

                "delta_enriched_minus_osm":
                    enriched_count
                    - osm_count,

                "relative_change_percent":
                    (
                        (
                            enriched_count
                            - osm_count
                        )
                        / osm_count
                        * 100.0
                    )
                    if osm_count
                    else None,
            }
        )

        osm_reachable = float(
            osm_summary[
                f"{group}_reachable_share"
            ]
        ) * 100.0

        enriched_reachable = float(
            enriched_summary[
                f"{group}_reachable_share"
            ]
        ) * 100.0

        rows.append(
            {
                "service_group":
                    group,

                "metric":
                    "reachable_population_share",

                "unit":
                    "percentage_points",

                "osm_only":
                    osm_reachable,

                "enriched":
                    enriched_reachable,

                "delta_enriched_minus_osm":
                    enriched_reachable
                    - osm_reachable,

                "relative_change_percent":
                    (
                        (
                            enriched_reachable
                            - osm_reachable
                        )
                        / osm_reachable
                        * 100.0
                    )
                    if osm_reachable
                    else None,
            }
        )

        osm_time = float(
            osm_summary[
                f"{group}_weighted_mean_nearest_time_min"
            ]
        )

        enriched_time = float(
            enriched_summary[
                f"{group}_weighted_mean_nearest_time_min"
            ]
        )

        rows.append(
            {
                "service_group":
                    group,

                "metric":
                    "weighted_mean_nearest_time_min",

                "unit":
                    "minutes",

                "osm_only":
                    osm_time,

                "enriched":
                    enriched_time,

                "delta_enriched_minus_osm":
                    enriched_time
                    - osm_time,

                "relative_change_percent":
                    (
                        (
                            enriched_time
                            - osm_time
                        )
                        / osm_time
                        * 100.0
                    )
                    if osm_time
                    else None,
            }
        )

        for threshold in thresholds:
            osm_share = float(
                osm_summary[
                    (
                        f"{group}_population_share_"
                        f"within_{threshold}_min"
                    )
                ]
            ) * 100.0

            enriched_share = float(
                enriched_summary[
                    (
                        f"{group}_population_share_"
                        f"within_{threshold}_min"
                    )
                ]
            ) * 100.0

            rows.append(
                {
                    "service_group":
                        group,

                    "metric":
                        (
                            "population_share_"
                            f"within_{threshold}_min"
                        ),

                    "unit":
                        "percentage_points",

                    "osm_only":
                        osm_share,

                    "enriched":
                        enriched_share,

                    "delta_enriched_minus_osm":
                        enriched_share
                        - osm_share,

                    "relative_change_percent":
                        (
                            (
                                enriched_share
                                - osm_share
                            )
                            / osm_share
                            * 100.0
                        )
                        if osm_share
                        else None,
                }
            )

            osm_population = float(
                osm_summary[
                    (
                        f"{group}_population_"
                        f"within_{threshold}_min"
                    )
                ]
            )

            enriched_population = float(
                enriched_summary[
                    (
                        f"{group}_population_"
                        f"within_{threshold}_min"
                    )
                ]
            )

            rows.append(
                {
                    "service_group":
                        group,

                    "metric":
                        (
                            "population_"
                            f"within_{threshold}_min"
                        ),

                    "unit":
                        "people",

                    "osm_only":
                        osm_population,

                    "enriched":
                        enriched_population,

                    "delta_enriched_minus_osm":
                        enriched_population
                        - osm_population,

                    "relative_change_percent":
                        (
                            (
                                enriched_population
                                - osm_population
                            )
                            / osm_population
                            * 100.0
                        )
                        if osm_population
                        else None,
                }
            )

    return pd.DataFrame(
        rows
    )


def print_compact_table(
    comparison,
    thresholds,
):
    print(
        "\n=============================================="
    )
    print(
        " OSM-ONLY vs INSTITUTIONAL + OSM"
    )
    print(
        "=============================================="
    )

    for group in GROUPS:
        subset = comparison.loc[
            comparison[
                "service_group"
            ]
            == group
        ]

        def row(metric):
            return subset.loc[
                subset[
                    "metric"
                ]
                == metric
            ].iloc[0]

        print(
            f"\n{group.upper()}"
        )

        count = row(
            "usable_service_count"
        )

        print(
            "  usable service count: "
            f"{count['osm_only']:.0f} -> "
            f"{count['enriched']:.0f} "
            f"(Δ {count['delta_enriched_minus_osm']:+.0f})"
        )

        reachable = row(
            "reachable_population_share"
        )

        print(
            "  reachable population: "
            f"{reachable['osm_only']:.2f}% -> "
            f"{reachable['enriched']:.2f}% "
            f"(Δ {reachable['delta_enriched_minus_osm']:+.2f} pp)"
        )

        time = row(
            "weighted_mean_nearest_time_min"
        )

        print(
            "  mean nearest time: "
            f"{time['osm_only']:.2f} -> "
            f"{time['enriched']:.2f} min "
            f"(Δ {time['delta_enriched_minus_osm']:+.2f} min)"
        )

        for threshold in thresholds:
            share = row(
                (
                    "population_share_"
                    f"within_{threshold}_min"
                )
            )

            people = row(
                (
                    "population_"
                    f"within_{threshold}_min"
                )
            )

            print(
                f"  within {threshold:>2} min: "
                f"{share['osm_only']:.2f}% -> "
                f"{share['enriched']:.2f}% "
                f"(Δ {share['delta_enriched_minus_osm']:+.2f} pp; "
                f"{people['delta_enriched_minus_osm']:+.2f} people)"
            )


def main():
    args = parse_args()

    paths = get_paths(
        args
    )

    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{label}: {path}"
            )

    origins_enriched = pd.read_parquet(
        paths[
            "origins_enriched"
        ]
    )

    origins_osm = pd.read_parquet(
        paths[
            "origins_osm"
        ]
    )

    validate_threshold_monotonicity(
        origins_enriched,
        "enriched",
        args.thresholds_min,
    )

    validate_threshold_monotonicity(
        origins_osm,
        "osm_only",
        args.thresholds_min,
    )

    common_origin_check = (
        validate_common_origins(
            origins_enriched,
            origins_osm,
        )
    )

    enriched_payload = load_json(
        paths[
            "summary_enriched"
        ]
    )

    osm_payload = load_json(
        paths[
            "summary_osm"
        ]
    )

    enriched_summary = municipality_summary(
        enriched_payload
    )

    osm_summary = municipality_summary(
        osm_payload
    )

    assert_close(
        enriched_summary[
            "population"
        ],
        osm_summary[
            "population"
        ],
        "summary population equality",
    )

    services_enriched = pd.read_parquet(
        paths[
            "services_enriched"
        ]
    )

    services_osm = pd.read_parquet(
        paths[
            "services_osm"
        ]
    )

    enriched_counts = service_counts(
        services_enriched
    )

    osm_counts = service_counts(
        services_osm
    )

    comparison = comparison_rows(
        osm_summary=osm_summary,
        enriched_summary=enriched_summary,
        osm_counts=osm_counts,
        enriched_counts=enriched_counts,
        thresholds=args.thresholds_min,
    )

    output_dir = (
        FEATURES_COMPARISON_DIR
        / args.municipality_code
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "osm_only_vs_enriched_accessibility.csv"
    )

    json_path = (
        output_dir
        / "osm_only_vs_enriched_accessibility.json"
    )

    comparison.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "municipality_code":
            args.municipality_code,

        "experimental_control":
            common_origin_check,

        "thresholds_min":
            args.thresholds_min,

        "service_counts": {
            "osm_only":
                osm_counts,

            "enriched":
                enriched_counts,
        },

        "comparison_records":
            comparison.to_dict(
                orient="records"
            ),

        "interpretation_rule":
            (
                "For coverage shares, positive delta means higher "
                "coverage in the enriched scenario. For nearest-time "
                "metrics, negative delta means lower walking time in "
                "the enriched scenario."
            ),
    }

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Experimental control:"
    )

    print(
        "  same origin rows: "
        f"{common_origin_check['origin_rows']}"
    )

    print(
        "  same population: "
        f"{common_origin_check['population']:.2f}"
    )

    print(
        "  same sections: YES"
    )

    print(
        "  same network nodes: YES"
    )

    print(
        "  same population weights: YES"
    )

    print_compact_table(
        comparison,
        args.thresholds_min,
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ {csv_path}"
    )

    print(
        f"✓ {json_path}"
    )


if __name__ == "__main__":
    main()
