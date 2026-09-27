import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
FEATURES_MIM_DIR = ROOT / "data" / "features" / "mim"

DEFAULT_SCHOOL_YEAR = "202425"
DEFAULT_BUILDING_YEAR = "202425"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Confronta il nuovo insieme di edifici MIM temporalmente coerente "
            "con un dataset spaziale curato precedente e riusa le localizzazioni "
            "solo tramite exact building_code."
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
        "--old-curated-file",
        default=None,
        help=(
            "File Parquet spaziale curato precedente. "
            "Se omesso cerca physical_school_buildings_<building-year>_curated_v2.parquet."
        ),
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
            "municipality-code deve avere 6 cifre."
        )

    return args


def clean_building_code(series):
    return (
        series
        .astype("string")
        .str.strip()
        .replace(
            {
                "":
                    pd.NA,
                "nan":
                    pd.NA,
                "None":
                    pd.NA,
            }
        )
    )


def load_inputs(args):
    directory = (
        PROCESSED_MIM_DIR
        / args.municipality_code
    )

    new_path = (
        directory
        / (
            "physical_school_buildings_"
            f"{args.building_year}_from_schools_{args.school_year}.parquet"
        )
    )

    if args.old_curated_file:
        old_path = Path(
            args.old_curated_file
        )

        if not old_path.is_absolute():
            old_path = (
                ROOT
                / old_path
            )
    else:
        old_path = (
            directory
            / (
                "physical_school_buildings_"
                f"{args.building_year}_curated_v2.parquet"
            )
        )

    if not new_path.exists():
        raise FileNotFoundError(
            f"Nuovo dataset edifici non trovato: {new_path}"
        )

    if not old_path.exists():
        raise FileNotFoundError(
            f"Dataset curato precedente non trovato: {old_path}"
        )

    new = pd.read_parquet(
        new_path
    )

    old = pd.read_parquet(
        old_path
    )

    for name, dataframe in [
        ("new", new),
        ("old", old),
    ]:
        if "building_code" not in dataframe.columns:
            raise RuntimeError(
                f"building_code mancante nel dataset {name}."
            )

        dataframe[
            "building_code"
        ] = clean_building_code(
            dataframe[
                "building_code"
            ]
        )

        if dataframe[
            "building_code"
        ].duplicated().any():
            duplicates = (
                dataframe.loc[
                    dataframe[
                        "building_code"
                    ].duplicated(
                        keep=False
                    ),
                    "building_code",
                ]
                .dropna()
                .unique()
                .tolist()
            )

            raise RuntimeError(
                f"building_code duplicati nel dataset {name}: "
                + ", ".join(
                    map(
                        str,
                        duplicates,
                    )
                )
            )

    return (
        new,
        old,
        new_path,
        old_path,
    )


def build_comparison(
    new,
    old,
):
    new_codes = set(
        new[
            "building_code"
        ].dropna()
    )

    old_codes = set(
        old[
            "building_code"
        ].dropna()
    )

    shared = sorted(
        new_codes
        & old_codes
    )

    new_only = sorted(
        new_codes
        - old_codes
    )

    old_only = sorted(
        old_codes
        - new_codes
    )

    rows = []

    for code in shared:
        rows.append(
            {
                "building_code":
                    code,

                "comparison_status":
                    "shared",
            }
        )

    for code in new_only:
        rows.append(
            {
                "building_code":
                    code,

                "comparison_status":
                    "new_only",
            }
        )

    for code in old_only:
        rows.append(
            {
                "building_code":
                    code,

                "comparison_status":
                    "old_only",
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    return (
        comparison,
        shared,
        new_only,
        old_only,
    )


def select_reusable_columns(
    old,
):
    preferred_columns = [
        "building_code",

        "automated_location_status",
        "automated_longitude",
        "automated_latitude",

        "final_location_status",
        "final_longitude",
        "final_latitude",
        "final_geometry_source",

        "curated_location_status",
        "curated_geometry_source",
        "curated_longitude",
        "curated_latitude",
        "curated_confidence",
        "curated_geometry_resolution",
        "curation_reason",

        "ground_truth_case_id",
        "ground_truth_site_label",
        "ground_truth_verification_status",
        "ground_truth_temporal_alignment",
        "ground_truth_merge_recommendation",
        "ground_truth_latitude",
        "ground_truth_longitude",
        "ground_truth_has_coordinates",
        "ground_truth_coordinate_status",
        "ground_truth_source_url",
        "ground_truth_secondary_source_url",
        "ground_truth_notes",

        "recovery2_status",
        "recovery2_longitude",
        "recovery2_latitude",
        "recovery2_source_type",
        "recovery2_confidence",
        "recovery2_geometry_resolution",
        "recovery2_primary_source_url",
        "recovery2_secondary_source_url",
        "recovery2_notes",

        "geocoder_status",
        "geocoder_longitude",
        "geocoder_latitude",
        "geocoder_result_address",
        "geocoder_address_score",

        "osm_match_status",
        "osm_candidate_site_id",
        "osm_candidate_site_name",
        "osm_candidate_site_address",
        "osm_candidate_longitude",
        "osm_candidate_latitude",
        "osm_match_score",
        "osm_name_score",
        "osm_address_score",
    ]

    existing = [
        column
        for column in preferred_columns
        if column in old.columns
    ]

    return old[
        existing
    ].copy()


def build_reused_dataset(
    new,
    old,
):
    reusable = select_reusable_columns(
        old
    )

    rename_map = {}

    for column in reusable.columns:
        if column == "building_code":
            continue

        rename_map[
            column
        ] = (
            "previous_"
            + column
        )

    reusable = reusable.rename(
        columns=rename_map
    )

    merged = new.merge(
        reusable,
        on="building_code",
        how="left",
        validate="one_to_one",
    )

    merged[
        "location_reuse_available"
    ] = merged[
        "previous_curated_longitude"
    ].notna() & merged[
        "previous_curated_latitude"
    ].notna() if (
        "previous_curated_longitude"
        in merged.columns
        and "previous_curated_latitude"
        in merged.columns
    ) else False

    merged[
        "location_reuse_status"
    ] = "needs_localization"

    merged.loc[
        merged[
            "location_reuse_available"
        ],
        "location_reuse_status",
    ] = "reused_exact_building_code"

    if (
        "previous_curated_longitude"
        in merged.columns
        and "previous_curated_latitude"
        in merged.columns
    ):
        merged[
            "reused_longitude"
        ] = merged[
            "previous_curated_longitude"
        ]

        merged[
            "reused_latitude"
        ] = merged[
            "previous_curated_latitude"
        ]

    else:
        merged[
            "reused_longitude"
        ] = pd.NA

        merged[
            "reused_latitude"
        ] = pd.NA

    merged[
        "reused_location_source"
    ] = (
        merged[
            "previous_curated_geometry_source"
        ]
        if "previous_curated_geometry_source"
        in merged.columns
        else pd.NA
    )

    merged[
        "reused_location_confidence"
    ] = (
        merged[
            "previous_curated_confidence"
        ]
        if "previous_curated_confidence"
        in merged.columns
        else pd.NA
    )

    merged[
        "reused_location_status"
    ] = (
        merged[
            "previous_curated_location_status"
        ]
        if "previous_curated_location_status"
        in merged.columns
        else pd.NA
    )

    return merged


def build_new_only_table(
    merged,
):
    columns = [
        column
        for column in [
            "building_code",
            "official_building_address",
            "building_municipality_name",
            "building_postal_code",
            "linked_school_count",
            "linked_school_codes",
            "linked_school_names",
            "location_reuse_status",
        ]
        if column in merged.columns
    ]

    return (
        merged.loc[
            merged[
                "location_reuse_status"
            ]
            == "needs_localization",
            columns,
        ]
        .copy()
    )


def save_outputs(
    args,
    merged,
    comparison,
    new_only_table,
    shared,
    new_only,
    old_only,
    new_path,
    old_path,
):
    processed_dir = (
        PROCESSED_MIM_DIR
        / args.municipality_code
    )

    features_dir = (
        FEATURES_MIM_DIR
        / args.municipality_code
    )

    features_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reused_parquet = (
        processed_dir
        / (
            "physical_school_buildings_"
            f"{args.building_year}_"
            f"from_schools_{args.school_year}_"
            "with_reused_locations.parquet"
        )
    )

    reused_csv = (
        features_dir
        / (
            "physical_school_buildings_"
            f"{args.building_year}_"
            f"from_schools_{args.school_year}_"
            "with_reused_locations.csv"
        )
    )

    comparison_path = (
        features_dir
        / (
            "school_building_code_comparison_"
            f"{args.building_year}.csv"
        )
    )

    new_only_path = (
        features_dir
        / (
            "school_buildings_needing_localization_"
            f"{args.building_year}.csv"
        )
    )

    manifest_path = (
        features_dir
        / (
            "school_building_location_reuse_"
            f"{args.building_year}_manifest.json"
        )
    )

    merged.to_parquet(
        reused_parquet,
        index=False,
    )

    merged.to_csv(
        reused_csv,
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    new_only_table.to_csv(
        new_only_path,
        index=False,
        encoding="utf-8-sig",
    )

    reusable_count = int(
        merged[
            "location_reuse_available"
        ].sum()
    )

    total_new = int(
        len(
            merged
        )
    )

    reuse_pct = (
        100.0
        * reusable_count
        / total_new
        if total_new
        else 0.0
    )

    manifest = {
        "municipality_code":
            args.municipality_code,

        "school_year":
            args.school_year,

        "building_year":
            args.building_year,

        "new_temporally_aligned_dataset":
            str(
                new_path
            ),

        "previous_curated_spatial_dataset":
            str(
                old_path
            ),

        "new_building_count":
            total_new,

        "old_building_count":
            int(
                len(
                    set(
                        pd.read_parquet(
                            old_path,
                            columns=[
                                "building_code",
                            ],
                        )[
                            "building_code"
                        ].dropna()
                    )
                )
            ),

        "shared_building_codes":
            int(
                len(
                    shared
                )
            ),

        "new_only_building_codes":
            int(
                len(
                    new_only
                )
            ),

        "old_only_building_codes":
            int(
                len(
                    old_only
                )
            ),

        "new_buildings_with_reusable_coordinates":
            reusable_count,

        "new_buildings_needing_localization":
            int(
                (
                    merged[
                        "location_reuse_status"
                    ]
                    == "needs_localization"
                ).sum()
            ),

        "location_reuse_pct_of_new_buildings":
            reuse_pct,

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
        "reused_parquet":
            reused_parquet,

        "reused_csv":
            reused_csv,

        "comparison":
            comparison_path,

        "new_only":
            new_only_path,

        "manifest":
            manifest_path,

        "manifest_data":
            manifest,
    }


def main():
    args = parse_args()

    print(
        "\n===================================="
    )
    print(
        " SCHOOL BUILDING LOCATION REUSE"
    )
    print(
        "===================================="
    )

    (
        new,
        old,
        new_path,
        old_path,
    ) = load_inputs(
        args
    )

    (
        comparison,
        shared,
        new_only,
        old_only,
    ) = build_comparison(
        new,
        old,
    )

    merged = build_reused_dataset(
        new,
        old,
    )

    new_only_table = (
        build_new_only_table(
            merged
        )
    )

    outputs = save_outputs(
        args=args,
        merged=merged,
        comparison=comparison,
        new_only_table=new_only_table,
        shared=shared,
        new_only=new_only,
        old_only=old_only,
        new_path=new_path,
        old_path=old_path,
    )

    m = outputs[
        "manifest_data"
    ]

    print(
        "\n=== BUILDING CODE COMPARISON ==="
    )
    print(
        f"Old buildings: {m['old_building_count']}"
    )
    print(
        f"New buildings: {m['new_building_count']}"
    )
    print(
        f"Shared building_code: {m['shared_building_codes']}"
    )
    print(
        f"New-only building_code: {m['new_only_building_codes']}"
    )
    print(
        f"Old-only building_code: {m['old_only_building_codes']}"
    )

    print(
        "\n=== LOCATION REUSE ==="
    )
    print(
        "New buildings with reusable coordinates: "
        f"{m['new_buildings_with_reusable_coordinates']}"
    )
    print(
        "New buildings still needing localization: "
        f"{m['new_buildings_needing_localization']}"
    )
    print(
        "Reuse coverage over new building set: "
        f"{m['location_reuse_pct_of_new_buildings']:.2f}%"
    )

    print(
        "\n=== OUTPUT ==="
    )

    for key in [
        "reused_parquet",
        "reused_csv",
        "comparison",
        "new_only",
        "manifest",
    ]:
        print(
            f"✓ {outputs[key]}"
        )

    print(
        "\nNOTA METODOLOGICA:"
    )
    print(
        "Le localizzazioni vengono riutilizzate esclusivamente "
        "quando il building_code MIM coincide esattamente."
    )
    print(
        "Le relazioni scuola-edificio restano quelle del nuovo "
        "dataset temporalmente coerente 2024/25."
    )


if __name__ == "__main__":
    main()
