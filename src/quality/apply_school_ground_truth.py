import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
FEATURES_MIM_DIR = ROOT / "data" / "features" / "mim"

DEFAULT_BUILDING_YEAR = "202425"


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Applica il ground truth documentato alle localizzazioni "
            "degli edifici scolastici MIM, preservando integralmente "
            "la provenance automatica."
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
# HELPERS
# ============================================================

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


def as_float(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def split_building_codes(value):
    value = clean_text(value)

    if value is None:
        return []

    return [
        part.strip()
        for part in value.split(";")
        if part.strip()
    ]


# ============================================================
# INPUT
# ============================================================

def load_inputs(
    municipality_code,
    building_year,
):
    processed_dir = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    feature_dir = (
        FEATURES_MIM_DIR
        / municipality_code
    )

    automated_path = (
        processed_dir
        / (
            "physical_school_buildings_"
            f"{building_year}_final_v2.parquet"
        )
    )

    ground_truth_path = (
        feature_dir
        / (
            "school_site_ground_truth_seed_"
            f"{building_year}.csv"
        )
    )

    for path in [
        automated_path,
        ground_truth_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"File richiesto non trovato: {path}"
            )

    automated = pd.read_parquet(
        automated_path
    )

    ground_truth = pd.read_csv(
        ground_truth_path,
        dtype=str,
    )

    if (
        "building_code"
        not in automated.columns
    ):
        raise RuntimeError(
            "building_code mancante nel dataset automatico."
        )

    if automated[
        "building_code"
    ].duplicated().any():
        raise RuntimeError(
            "building_code duplicati nel dataset automatico."
        )

    required_gt = {
        "review_case_id",
        "building_codes",
        "verification_status",
        "latitude",
        "longitude",
        "source_url",
    }

    missing = (
        required_gt
        - set(
            ground_truth.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Colonne mancanti nel ground truth: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    return (
        automated,
        ground_truth,
        automated_path,
        ground_truth_path,
    )


# ============================================================
# GROUND TRUTH NORMALIZATION
# ============================================================

def explode_ground_truth(
    ground_truth,
):
    rows = []

    for _, row in ground_truth.iterrows():
        codes = split_building_codes(
            row.get(
                "building_codes"
            )
        )

        for building_code in codes:
            item = row.to_dict()

            item[
                "building_code"
            ] = building_code

            item[
                "ground_truth_latitude"
            ] = as_float(
                row.get(
                    "latitude"
                )
            )

            item[
                "ground_truth_longitude"
            ] = as_float(
                row.get(
                    "longitude"
                )
            )

            item[
                "ground_truth_has_coordinates"
            ] = (
                item[
                    "ground_truth_latitude"
                ]
                is not None
                and item[
                    "ground_truth_longitude"
                ]
                is not None
            )

            rows.append(
                item
            )

    exploded = pd.DataFrame(
        rows
    )

    if exploded.empty:
        return exploded

    duplicate_codes = (
        exploded[
            "building_code"
        ][
            exploded[
                "building_code"
            ].duplicated(
                keep=False
            )
        ]
        .unique()
        .tolist()
    )

    if duplicate_codes:
        raise RuntimeError(
            "Uno stesso building_code compare in più righe "
            "del ground truth: "
            + ", ".join(
                duplicate_codes
            )
        )

    return exploded


# ============================================================
# MERGE / CURATION
# ============================================================

def build_curated_dataset(
    automated,
    ground_truth_exploded,
):
    gt_columns = [
        "building_code",
        "review_case_id",
        "verified_site_label",
        "verification_status",
        "temporal_alignment",
        "merge_recommendation",
        "ground_truth_latitude",
        "ground_truth_longitude",
        "ground_truth_has_coordinates",
        "coordinate_status",
        "source_url",
        "secondary_source_url",
        "notes",
    ]

    gt_columns = [
        column
        for column in gt_columns
        if column
        in ground_truth_exploded.columns
    ]

    gt = (
        ground_truth_exploded[
            gt_columns
        ]
        .copy()
    )

    gt = gt.rename(
        columns={
            "review_case_id":
                "ground_truth_case_id",

            "verified_site_label":
                "ground_truth_site_label",

            "verification_status":
                "ground_truth_verification_status",

            "temporal_alignment":
                "ground_truth_temporal_alignment",

            "merge_recommendation":
                "ground_truth_merge_recommendation",

            "coordinate_status":
                "ground_truth_coordinate_status",

            "source_url":
                "ground_truth_source_url",

            "secondary_source_url":
                "ground_truth_secondary_source_url",

            "notes":
                "ground_truth_notes",
        }
    )

    merged = automated.merge(
        gt,
        on="building_code",
        how="left",
        validate="one_to_one",
    )

    final_rows = []

    for _, row in merged.iterrows():
        result = row.to_dict()

        gt_has_coordinates = bool(
            row.get(
                "ground_truth_has_coordinates"
            )
        ) if pd.notna(
            row.get(
                "ground_truth_has_coordinates"
            )
        ) else False

        gt_status = clean_text(
            row.get(
                "ground_truth_verification_status"
            )
        )

        auto_status = clean_text(
            row.get(
                "final_location_status"
            )
        )

        auto_lon = as_float(
            row.get(
                "final_longitude"
            )
        )

        auto_lat = as_float(
            row.get(
                "final_latitude"
            )
        )

        gt_lon = as_float(
            row.get(
                "ground_truth_longitude"
            )
        )

        gt_lat = as_float(
            row.get(
                "ground_truth_latitude"
            )
        )

        result[
            "automated_location_status"
        ] = auto_status

        result[
            "automated_longitude"
        ] = auto_lon

        result[
            "automated_latitude"
        ] = auto_lat

        result[
            "ground_truth_address_confirmed"
        ] = (
            gt_status is not None
        )

        # ----------------------------------------------------
        # Officially verified coordinate wins.
        # It is interpreted as a verified school-site point,
        # not necessarily the footprint centroid of each building.
        # ----------------------------------------------------

        if (
            gt_has_coordinates
            and gt_lon is not None
            and gt_lat is not None
        ):
            result[
                "curated_location_status"
            ] = (
                "validated_ground_truth"
            )

            result[
                "curated_geometry_source"
            ] = (
                "official_source_map_link"
            )

            result[
                "curated_longitude"
            ] = gt_lon

            result[
                "curated_latitude"
            ] = gt_lat

            result[
                "curated_confidence"
            ] = "high"

            result[
                "curated_geometry_resolution"
            ] = (
                "verified_school_site_point"
            )

            result[
                "curation_reason"
            ] = (
                "Coordinate recuperate da una fonte ufficiale "
                "documentata nel ground truth."
            )

        # ----------------------------------------------------
        # No GT coordinate: preserve an already usable
        # automated coordinate. Official address confirmation
        # is stored separately and does not silently upgrade
        # spatial precision.
        # ----------------------------------------------------

        elif (
            auto_status
            in {
                "validated",
                "accepted_address",
            }
            and auto_lon is not None
            and auto_lat is not None
        ):
            result[
                "curated_location_status"
            ] = (
                "validated_automated"
                if auto_status
                == "validated"
                else "accepted_automated_address"
            )

            result[
                "curated_geometry_source"
            ] = row.get(
                "final_geometry_source"
            )

            result[
                "curated_longitude"
            ] = auto_lon

            result[
                "curated_latitude"
            ] = auto_lat

            result[
                "curated_confidence"
            ] = row.get(
                "location_confidence"
            )

            result[
                "curated_geometry_resolution"
            ] = (
                "automated_point"
            )

            if gt_status:
                result[
                    "curation_reason"
                ] = (
                    "Coordinate automatiche preservate; "
                    "la fonte ufficiale conferma il sito/indirizzo "
                    "ma non fornisce ancora una coordinata verificata."
                )
            else:
                result[
                    "curation_reason"
                ] = (
                    "Coordinate provenienti esclusivamente "
                    "dalla pipeline automatica."
                )

        # ----------------------------------------------------
        # Official source confirms the site/address but no
        # trustworthy point is available yet.
        # ----------------------------------------------------

        elif gt_status:
            result[
                "curated_location_status"
            ] = (
                "official_site_confirmed_location_pending"
            )

            result[
                "curated_geometry_source"
            ] = None

            result[
                "curated_longitude"
            ] = None

            result[
                "curated_latitude"
            ] = None

            result[
                "curated_confidence"
            ] = "pending"

            result[
                "curated_geometry_resolution"
            ] = None

            result[
                "curation_reason"
            ] = (
                "Esistenza/indirizzo del sito confermati "
                "da fonte ufficiale, ma manca una coordinata "
                "sufficientemente verificata."
            )

        # ----------------------------------------------------
        # Outside municipality.
        # ----------------------------------------------------

        elif (
            auto_status
            == "outside_target_municipality"
        ):
            result[
                "curated_location_status"
            ] = (
                "outside_target_municipality"
            )

            result[
                "curated_geometry_source"
            ] = None

            result[
                "curated_longitude"
            ] = None

            result[
                "curated_latitude"
            ] = None

            result[
                "curated_confidence"
            ] = "excluded"

            result[
                "curated_geometry_resolution"
            ] = None

            result[
                "curation_reason"
            ] = (
                "Edificio ufficialmente esterno "
                "al comune target."
            )

        else:
            result[
                "curated_location_status"
            ] = "unresolved"

            result[
                "curated_geometry_source"
            ] = None

            result[
                "curated_longitude"
            ] = None

            result[
                "curated_latitude"
            ] = None

            result[
                "curated_confidence"
            ] = "unresolved"

            result[
                "curated_geometry_resolution"
            ] = None

            result[
                "curation_reason"
            ] = (
                "Nessuna coordinata sufficientemente "
                "affidabile disponibile."
            )

        final_rows.append(
            result
        )

    return pd.DataFrame(
        final_rows
    )


# ============================================================
# OUTPUTS
# ============================================================

def save_outputs(
    curated,
    municipality_code,
    building_year,
):
    processed_dir = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    feature_dir = (
        FEATURES_MIM_DIR
        / municipality_code
    )

    parquet_path = (
        processed_dir
        / (
            "physical_school_buildings_"
            f"{building_year}_curated.parquet"
        )
    )

    csv_path = (
        feature_dir
        / (
            "physical_school_buildings_"
            f"{building_year}_curated.csv"
        )
    )

    pending_path = (
        feature_dir
        / (
            "school_buildings_location_pending_after_gt_"
            f"{building_year}.csv"
        )
    )

    curated.to_parquet(
        parquet_path,
        index=False,
    )

    curated.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    target_mask = (
        curated[
            "curated_location_status"
        ]
        != "outside_target_municipality"
    )

    usable_mask = (
        curated[
            "curated_longitude"
        ].notna()
        & curated[
            "curated_latitude"
        ].notna()
    )

    pending = (
        curated[
            target_mask
            & ~usable_mask
        ]
        .copy()
    )

    pending_columns = [
        column
        for column in [
            "building_code",
            "official_building_address",
            "ground_truth_case_id",
            "ground_truth_site_label",
            "ground_truth_verification_status",
            "ground_truth_temporal_alignment",
            "ground_truth_coordinate_status",
            "ground_truth_source_url",
            "curated_location_status",
            "curation_reason",
        ]
        if column in pending.columns
    ]

    pending[
        pending_columns
    ].to_csv(
        pending_path,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        parquet_path,
        csv_path,
        pending_path,
    )


def print_summary(
    curated,
    automated_path,
    ground_truth_path,
    parquet_path,
    csv_path,
    pending_path,
):
    print(
        "\n===================================="
    )
    print(
        " SCHOOL BUILDING GROUND TRUTH APPLIED"
    )
    print(
        "===================================="
    )

    print(
        f"Automated input: {automated_path}"
    )

    print(
        f"Ground truth: {ground_truth_path}"
    )

    print(
        f"\nEdifici totali: {len(curated)}"
    )

    print(
        "\nCurated status:"
    )

    print(
        curated[
            "curated_location_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    target_mask = (
        curated[
            "curated_location_status"
        ]
        != "outside_target_municipality"
    )

    usable_mask = (
        curated[
            "curated_longitude"
        ].notna()
        & curated[
            "curated_latitude"
        ].notna()
    )

    target_count = int(
        target_mask.sum()
    )

    usable_count = int(
        (
            target_mask
            & usable_mask
        ).sum()
    )

    gt_coordinate_count = int(
        (
            curated[
                "curated_location_status"
            ]
            == "validated_ground_truth"
        ).sum()
    )

    address_confirmed_count = int(
        (
            target_mask
            & curated[
                "ground_truth_address_confirmed"
            ]
        ).sum()
    )

    pending_count = (
        target_count
        - usable_count
    )

    coverage = (
        100.0
        * usable_count
        / target_count
        if target_count
        else 0.0
    )

    print(
        "\nMatera target:"
    )

    print(
        f"  edifici target: {target_count}"
    )

    print(
        "  coordinate da ground truth: "
        f"{gt_coordinate_count}"
    )

    print(
        "  edifici con sito/indirizzo ufficialmente "
        f"confermato: {address_confirmed_count}"
    )

    print(
        "  edifici con coordinate utilizzabili: "
        f"{usable_count}/{target_count}"
    )

    print(
        f"  coverage spaziale: {coverage:.2f}%"
    )

    print(
        f"  ancora senza coordinate: {pending_count}"
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ Curated Parquet: {parquet_path}"
    )

    print(
        f"✓ Curated CSV: {csv_path}"
    )

    print(
        f"✓ Remaining pending: {pending_path}"
    )

    print(
        "\nNOTA METODOLOGICA:"
    )

    print(
        "Le coordinate ground-truth rappresentano punti "
        "di sito scolastico verificati; non vengono interpretate "
        "automaticamente come centroidi dei singoli fabbricati."
    )

    print(
        "I risultati automatici originali restano nel dataset "
        "come colonne di provenance."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    (
        automated,
        ground_truth,
        automated_path,
        ground_truth_path,
    ) = load_inputs(
        args.municipality_code,
        args.building_year,
    )

    gt_exploded = explode_ground_truth(
        ground_truth
    )

    print(
        "\n===================================="
    )
    print(
        " APPLY SCHOOL GROUND TRUTH"
    )
    print(
        "===================================="
    )

    print(
        f"Edifici automatici: {len(automated)}"
    )

    print(
        "Righe ground truth: "
        f"{len(ground_truth)}"
    )

    print(
        "Building code coperti dal ground truth: "
        f"{len(gt_exploded)}"
    )

    curated = build_curated_dataset(
        automated,
        gt_exploded,
    )

    (
        parquet_path,
        csv_path,
        pending_path,
    ) = save_outputs(
        curated,
        args.municipality_code,
        args.building_year,
    )

    print_summary(
        curated,
        automated_path,
        ground_truth_path,
        parquet_path,
        csv_path,
        pending_path,
    )


if __name__ == "__main__":
    main()
