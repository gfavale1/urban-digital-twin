import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
FEATURES_MIM_DIR = ROOT / "data" / "features" / "mim"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Applica un dataset di validazione esterno/manuale ai candidati "
            "OSM di nuovi edifici scolastici. Il codice è generalizzabile; "
            "le decisioni specifiche del comune restano nel CSV di patch."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
    )

    parser.add_argument(
        "--building-year",
        default="202425",
    )

    parser.add_argument(
        "--validation-file",
        required=True,
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    return args


def as_bool(value):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "si",
        "sì",
    }


def main():
    args = parse_args()

    processed_dir = (
        PROCESSED_MIM_DIR
        / args.municipality_code
    )

    features_dir = (
        FEATURES_MIM_DIR
        / args.municipality_code
    )

    matches_path = (
        processed_dir
        / (
            "new_school_buildings_osm_matches_"
            f"{args.building_year}.parquet"
        )
    )

    validation_path = Path(
        args.validation_file
    )

    if not validation_path.is_absolute():
        validation_path = (
            ROOT
            / validation_path
        )

    if not matches_path.exists():
        raise FileNotFoundError(
            f"Match OSM non trovato: {matches_path}"
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation patch non trovato: {validation_path}"
        )

    matches = pd.read_parquet(
        matches_path
    )

    validation = pd.read_csv(
        validation_path,
        dtype={
            "building_code": str,
        },
    )

    matches[
        "building_code"
    ] = (
        matches[
            "building_code"
        ]
        .astype("string")
        .str.strip()
    )

    validation[
        "building_code"
    ] = (
        validation[
            "building_code"
        ]
        .astype("string")
        .str.strip()
    )

    if validation[
        "building_code"
    ].duplicated().any():
        raise RuntimeError(
            "building_code duplicati nel validation patch."
        )

    merged = matches.merge(
        validation,
        on="building_code",
        how="left",
        validate="one_to_one",
    )

    merged[
        "validation_use_osm_candidate"
    ] = merged[
        "use_osm_candidate"
    ].map(
        as_bool
    )

    merged[
        "validated_longitude"
    ] = pd.NA

    merged[
        "validated_latitude"
    ] = pd.NA

    use_mask = (
        merged[
            "validation_use_osm_candidate"
        ]
        & merged[
            "osm_candidate_longitude"
        ].notna()
        & merged[
            "osm_candidate_latitude"
        ].notna()
    )

    merged.loc[
        use_mask,
        "validated_longitude",
    ] = merged.loc[
        use_mask,
        "osm_candidate_longitude",
    ]

    merged.loc[
        use_mask,
        "validated_latitude",
    ] = merged.loc[
        use_mask,
        "osm_candidate_latitude",
    ]

    merged[
        "validated_coordinate_source"
    ] = pd.NA

    merged.loc[
        use_mask,
        "validated_coordinate_source",
    ] = "osm_candidate_validated_by_official_source"

    merged[
        "validated_location_usable"
    ] = (
        merged[
            "validated_longitude"
        ].notna()
        & merged[
            "validated_latitude"
        ].notna()
    )

    output_csv = (
        features_dir
        / (
            "new_school_buildings_validated_"
            f"{args.building_year}.csv"
        )
    )

    output_parquet = (
        processed_dir
        / (
            "new_school_buildings_validated_"
            f"{args.building_year}.parquet"
        )
    )

    merged.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    merged.to_parquet(
        output_parquet,
        index=False,
    )

    print(
        "\n===================================="
    )
    print(
        " APPLY SCHOOL BUILDING VALIDATION"
    )
    print(
        "===================================="
    )

    print(
        f"Rows: {len(merged)}"
    )

    print(
        "Validated usable locations: "
        f"{int(merged['validated_location_usable'].sum())}"
    )

    print(
        "\nValidation statuses:"
    )

    print(
        merged[
            "validation_status"
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
        f"✓ {output_csv}"
    )

    print(
        f"✓ {output_parquet}"
    )

    print(
        "\nNOTA:"
    )

    print(
        "Una coordinata con coordinate_resolution=site o site_anchor "
        "non deve essere interpretata come geometria esatta del singolo edificio."
    )


if __name__ == "__main__":
    main()
