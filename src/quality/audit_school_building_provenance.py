import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
FEATURES_MIM_DIR = ROOT / "data" / "features" / "mim"


AUTOMATIC_LEGACY_STATUSES = {
    "validated_automated",
    "accepted_automated_address",
}

GROUND_TRUTH_LEGACY_STATUSES = {
    "validated_ground_truth",
}

PENDING_LEGACY_STATUSES = {
    "official_site_confirmed_location_pending",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit di provenance delle localizzazioni degli edifici scolastici. "
            "Separa risultati automatici da ground truth/manual validation."
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

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    return args


def clean_code(series):
    return (
        series
        .astype("string")
        .str.strip()
    )


def first_existing(
    dataframe,
    columns,
):
    for column in columns:
        if column in dataframe.columns:
            return column

    return None


def numeric_value(
    row,
    candidates,
):
    for column in candidates:
        if column not in row.index:
            continue

        value = row.get(column)

        if pd.isna(value):
            continue

        try:
            return float(value)
        except Exception:
            continue

    return None


def text_value(
    row,
    candidates,
):
    for column in candidates:
        if column not in row.index:
            continue

        value = row.get(column)

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            return value

    return None


def bool_like(value):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return (
        str(value)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "y",
            "si",
            "sì",
            "recovered",
            "validated",
            "accepted",
        }
    )


def load_optional_parquet(path):
    if path.exists():
        return pd.read_parquet(
            path
        )

    return pd.DataFrame()


def load_optional_csv(path):
    if path.exists():
        return pd.read_csv(
            path,
            dtype={
                "building_code":
                    str,
            },
        )

    return pd.DataFrame()


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

    canonical_path = (
        processed_dir
        / (
            "physical_school_buildings_"
            f"{args.building_year}_"
            f"from_schools_{args.school_year}.parquet"
        )
    )

    legacy_path = (
        processed_dir
        / (
            "physical_school_buildings_"
            f"{args.building_year}_curated_v2.parquet"
        )
    )

    new_validated_path = (
        processed_dir
        / (
            "new_school_buildings_validated_"
            f"{args.building_year}.parquet"
        )
    )

    fallback_path = (
        features_dir
        / (
            "school_osm_street_fallback_"
            f"{args.building_year}.csv"
        )
    )

    if not canonical_path.exists():
        raise FileNotFoundError(
            f"Dataset canonico non trovato: {canonical_path}"
        )

    canonical = pd.read_parquet(
        canonical_path
    )

    legacy = load_optional_parquet(
        legacy_path
    )

    new_validated = (
        load_optional_parquet(
            new_validated_path
        )
    )

    fallback = load_optional_csv(
        fallback_path
    )

    canonical[
        "building_code"
    ] = clean_code(
        canonical[
            "building_code"
        ]
    )

    for dataframe in [
        legacy,
        new_validated,
        fallback,
    ]:
        if (
            not dataframe.empty
            and "building_code"
            in dataframe.columns
        ):
            dataframe[
                "building_code"
            ] = clean_code(
                dataframe[
                    "building_code"
                ]
            )

    legacy_by_code = (
        legacy.set_index(
            "building_code"
        )
        if (
            not legacy.empty
            and "building_code"
            in legacy.columns
        )
        else pd.DataFrame()
    )

    validated_by_code = (
        new_validated.set_index(
            "building_code"
        )
        if (
            not new_validated.empty
            and "building_code"
            in new_validated.columns
        )
        else pd.DataFrame()
    )

    fallback_by_code = (
        fallback.set_index(
            "building_code"
        )
        if (
            not fallback.empty
            and "building_code"
            in fallback.columns
        )
        else pd.DataFrame()
    )

    rows = []

    for _, canonical_row in canonical.iterrows():
        code = (
            canonical_row[
                "building_code"
            ]
        )

        municipality_name = (
            text_value(
                canonical_row,
                [
                    "building_municipality_name",
                ],
            )
        )

        in_target = (
            str(code).startswith(
                args.municipality_code
            )
        )

        automatic_available = False
        automatic_longitude = None
        automatic_latitude = None
        automatic_method = None
        automatic_resolution = None
        automatic_confidence = None

        ground_truth_available = False
        ground_truth_longitude = None
        ground_truth_latitude = None
        ground_truth_method = None
        ground_truth_resolution = None

        legacy_status = None
        legacy_recovery_status = None

        # ----------------------------------------------------
        # 1. PREVIOUS CURATED DATASET
        # ----------------------------------------------------
        if (
            not legacy_by_code.empty
            and code in legacy_by_code.index
        ):
            legacy_row = (
                legacy_by_code.loc[
                    code
                ]
            )

            if isinstance(
                legacy_row,
                pd.DataFrame,
            ):
                raise RuntimeError(
                    f"building_code duplicato nel legacy: {code}"
                )

            legacy_status = (
                text_value(
                    legacy_row,
                    [
                        "curated_location_status",
                        "final_location_status",
                        "location_status",
                    ],
                )
            )

            legacy_recovery_status = (
                text_value(
                    legacy_row,
                    [
                        "recovery2_status",
                        "recovery_status",
                    ],
                )
            )

            curated_lon = (
                numeric_value(
                    legacy_row,
                    [
                        "curated_longitude",
                        "final_longitude",
                        "longitude",
                    ],
                )
            )

            curated_lat = (
                numeric_value(
                    legacy_row,
                    [
                        "curated_latitude",
                        "final_latitude",
                        "latitude",
                    ],
                )
            )

            if (
                legacy_status
                in AUTOMATIC_LEGACY_STATUSES
                and curated_lon is not None
                and curated_lat is not None
            ):
                automatic_available = True
                automatic_longitude = (
                    curated_lon
                )
                automatic_latitude = (
                    curated_lat
                )

                if (
                    legacy_status
                    == "validated_automated"
                ):
                    automatic_method = (
                        "legacy_osm_geocoder_automatic"
                    )
                    automatic_resolution = (
                        "site_or_address"
                    )
                    automatic_confidence = (
                        "high"
                    )

                else:
                    automatic_method = (
                        "legacy_nominatim_automatic"
                    )
                    automatic_resolution = (
                        "address"
                    )
                    automatic_confidence = (
                        "medium"
                    )

            if (
                legacy_status
                in GROUND_TRUTH_LEGACY_STATUSES
                and curated_lon is not None
                and curated_lat is not None
            ):
                ground_truth_available = True
                ground_truth_longitude = (
                    curated_lon
                )
                ground_truth_latitude = (
                    curated_lat
                )
                ground_truth_method = (
                    "legacy_manual_ground_truth"
                )
                ground_truth_resolution = (
                    text_value(
                        legacy_row,
                        [
                            "curated_geometry_resolution",
                            "ground_truth_coordinate_status",
                        ],
                    )
                    or "site"
                )

            recovery_lon = (
                numeric_value(
                    legacy_row,
                    [
                        "recovery2_longitude",
                    ],
                )
            )

            recovery_lat = (
                numeric_value(
                    legacy_row,
                    [
                        "recovery2_latitude",
                    ],
                )
            )

            recovery_usable = (
                recovery_lon is not None
                and recovery_lat is not None
            )

            if recovery_usable:
                ground_truth_available = True
                ground_truth_longitude = (
                    recovery_lon
                )
                ground_truth_latitude = (
                    recovery_lat
                )
                ground_truth_method = (
                    "legacy_manual_recovery"
                )
                ground_truth_resolution = (
                    text_value(
                        legacy_row,
                        [
                            "recovery2_geometry_resolution",
                        ],
                    )
                    or "site"
                )

        # ----------------------------------------------------
        # 2. NEW BUILDINGS VALIDATED MANUALLY
        # ----------------------------------------------------
        if (
            not validated_by_code.empty
            and code in validated_by_code.index
        ):
            validated_row = (
                validated_by_code.loc[
                    code
                ]
            )

            if isinstance(
                validated_row,
                pd.DataFrame,
            ):
                raise RuntimeError(
                    f"building_code duplicato nei nuovi validati: {code}"
                )

            usable = (
                bool_like(
                    validated_row.get(
                        "validated_location_usable"
                    )
                )
            )

            lon = numeric_value(
                validated_row,
                [
                    "validated_longitude",
                ],
            )

            lat = numeric_value(
                validated_row,
                [
                    "validated_latitude",
                ],
            )

            if (
                usable
                and lon is not None
                and lat is not None
            ):
                ground_truth_available = True
                ground_truth_longitude = lon
                ground_truth_latitude = lat
                ground_truth_method = (
                    "official_source_validation"
                )
                ground_truth_resolution = (
                    text_value(
                        validated_row,
                        [
                            "coordinate_resolution",
                        ],
                    )
                    or "site"
                )

        # ----------------------------------------------------
        # 3. NATIONAL AUTOMATIC OSM STREET FALLBACK
        # ----------------------------------------------------
        if (
            not fallback_by_code.empty
            and code in fallback_by_code.index
        ):
            fallback_row = (
                fallback_by_code.loc[
                    code
                ]
            )

            if isinstance(
                fallback_row,
                pd.DataFrame,
            ):
                raise RuntimeError(
                    f"building_code duplicato nel fallback: {code}"
                )

            fallback_status = (
                text_value(
                    fallback_row,
                    [
                        "fallback_status",
                    ],
                )
            )

            lon = numeric_value(
                fallback_row,
                [
                    "candidate_longitude",
                ],
            )

            lat = numeric_value(
                fallback_row,
                [
                    "candidate_latitude",
                ],
            )

            if (
                fallback_status
                in {
                    "address_candidate",
                    "street_anchor_candidate",
                }
                and lon is not None
                and lat is not None
            ):
                automatic_available = True
                automatic_longitude = lon
                automatic_latitude = lat
                automatic_method = (
                    "osm_street_fallback"
                )

                if (
                    fallback_status
                    == "address_candidate"
                ):
                    automatic_resolution = (
                        "address"
                    )
                    automatic_confidence = (
                        "high"
                    )
                else:
                    automatic_resolution = (
                        "street_anchor"
                    )
                    automatic_confidence = (
                        "medium"
                    )

        # ----------------------------------------------------
        # EVALUATION STATUS
        # ----------------------------------------------------
        if not in_target:
            evaluation_status = (
                "outside_target_municipality"
            )

        elif (
            automatic_available
            and ground_truth_available
        ):
            evaluation_status = (
                "automatic_and_ground_truth"
            )

        elif automatic_available:
            evaluation_status = (
                "automatic_only"
            )

        elif ground_truth_available:
            evaluation_status = (
                "ground_truth_only"
            )

        else:
            evaluation_status = (
                "unresolved"
            )

        rows.append(
            {
                "building_code":
                    code,

                "official_building_address":
                    canonical_row.get(
                        "official_building_address"
                    ),

                "building_municipality_name":
                    municipality_name,

                "in_target_municipality":
                    in_target,

                "legacy_status":
                    legacy_status,

                "legacy_recovery_status":
                    legacy_recovery_status,

                "automatic_location_available":
                    automatic_available,

                "automatic_longitude":
                    automatic_longitude,

                "automatic_latitude":
                    automatic_latitude,

                "automatic_method":
                    automatic_method,

                "automatic_resolution":
                    automatic_resolution,

                "automatic_confidence":
                    automatic_confidence,

                "ground_truth_available":
                    ground_truth_available,

                "ground_truth_longitude":
                    ground_truth_longitude,

                "ground_truth_latitude":
                    ground_truth_latitude,

                "ground_truth_method":
                    ground_truth_method,

                "ground_truth_resolution":
                    ground_truth_resolution,

                "evaluation_status":
                    evaluation_status,
            }
        )

    audit = pd.DataFrame(
        rows
    )

    output_csv = (
        features_dir
        / (
            "school_building_provenance_audit_"
            f"{args.building_year}.csv"
        )
    )

    output_parquet = (
        processed_dir
        / (
            "school_building_provenance_audit_"
            f"{args.building_year}.parquet"
        )
    )

    audit.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_parquet(
        output_parquet,
        index=False,
    )

    target = (
        audit[
            audit[
                "in_target_municipality"
            ]
            == True
        ]
    )

    print(
        "\n===================================="
    )
    print(
        " SCHOOL BUILDING PROVENANCE AUDIT"
    )
    print(
        "===================================="
    )

    print(
        f"Canonical buildings: {len(audit)}"
    )

    print(
        f"Target municipality buildings: {len(target)}"
    )

    print(
        "Outside target: "
        f"{int((~audit['in_target_municipality']).sum())}"
    )

    print(
        "\n=== AUTOMATIC PIPELINE ==="
    )

    automatic_count = int(
        target[
            "automatic_location_available"
        ].sum()
    )

    print(
        f"Automatic usable locations: {automatic_count}/{len(target)} "
        f"({100 * automatic_count / len(target):.2f}%)"
    )

    print(
        "\nAutomatic methods:"
    )

    print(
        target.loc[
            target[
                "automatic_location_available"
            ],
            "automatic_method",
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nAutomatic resolutions:"
    )

    print(
        target.loc[
            target[
                "automatic_location_available"
            ],
            "automatic_resolution",
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n=== GROUND TRUTH / MANUAL VALIDATION ==="
    )

    gt_count = int(
        target[
            "ground_truth_available"
        ].sum()
    )

    print(
        f"Ground-truth locations available: {gt_count}/{len(target)}"
    )

    print(
        "\nGround-truth methods:"
    )

    print(
        target.loc[
            target[
                "ground_truth_available"
            ],
            "ground_truth_method",
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n=== EVALUATION STATUS ==="
    )

    print(
        audit[
            "evaluation_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    unresolved = (
        target[
            target[
                "evaluation_status"
            ]
            == "unresolved"
        ]
    )

    print(
        "\nUnresolved target buildings: "
        f"{len(unresolved)}"
    )

    if not unresolved.empty:
        print(
            unresolved[
                [
                    "building_code",
                    "official_building_address",
                ]
            ]
            .to_string(
                index=False
            )
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
        "Il dataset automatico e il ground truth sono mantenuti separati. "
        "Una localizzazione manuale non viene conteggiata come risultato "
        "della pipeline nazionale."
    )


if __name__ == "__main__":
    main()
