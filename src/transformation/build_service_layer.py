import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
PROCESSED_SALUTE_DIR = ROOT / "data" / "processed" / "salute"
FEATURES_SALUTE_DIR = ROOT / "data" / "features" / "salute"

PROCESSED_SERVICES_DIR = ROOT / "data" / "processed" / "services"
FEATURES_SERVICES_DIR = ROOT / "data" / "features" / "services"


CANONICAL_COLUMNS = [
    "service_site_id",
    "domain_service_site_id",
    "category",
    "subcategory",
    "name",
    "municipality_code",
    "municipality_name",
    "address",
    "postal_code",
    "longitude",
    "latitude",
    "source_system",
    "source_dataset",
    "source_record_id",
    "reference_period",
    "coordinate_source",
    "coordinate_resolution",
    "confidence",
    "resolution_status",
    "usable_for_accessibility",
    "capacity_value",
    "capacity_unit",
    "provenance_json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Costruisce il canonical Service layer a partire dai "
            "dataset finali Education e Health."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre.",
    )

    parser.add_argument(
        "--school-year",
        default="202425",
        help="Anno scolastico MIM. Default: 202425.",
    )

    parser.add_argument(
        "--health-reference-date",
        default="2025-06-30",
        help="Data snapshot Health YYYY-MM-DD. Default: 2025-06-30.",
    )

    parser.add_argument(
        "--hospital-year",
        default="2023",
        help="Anno del dataset ospedaliero. Default: 2023.",
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
        pd.Timestamp(args.health_reference_date).normalize()
    )

    return args


def utc_now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip()

    if value in {
        "",
        "-",
        "nan",
        "NaN",
        "None",
        "NULL",
    }:
        return None

    return value


def clean_float(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return None


def canonical_id(domain, raw_id):
    raw_id = clean_text(raw_id)

    if raw_id is None:
        raise ValueError(
            f"Identificativo mancante per dominio {domain}."
        )

    return (
        f"{domain.upper()}::{raw_id}"
    )


def compact_json(data):
    clean = {}

    for key, value in data.items():
        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except Exception:
            pass

        if isinstance(
            value,
            (
                pd.Timestamp,
                datetime,
            ),
        ):
            value = value.isoformat()

        elif hasattr(
            value,
            "item",
        ):
            try:
                value = value.item()
            except Exception:
                pass

        clean[str(key)] = value

    return json.dumps(
        clean,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def school_input_path(args):
    return (
        PROCESSED_MIM_DIR
        / args.municipality_code
        / (
            f"school_sites_{args.school_year}.parquet"
        )
    )


def health_input_path(args):
    label = (
        args.health_reference_date
        .strftime("%Y%m%d")
    )

    return (
        PROCESSED_SALUTE_DIR
        / args.municipality_code
        / (
            f"health_sites_final_{label}.parquet"
        )
    )


def health_manifest_path(args):
    label = (
        args.health_reference_date
        .strftime("%Y%m%d")
    )

    return (
        FEATURES_SALUTE_DIR
        / args.municipality_code
        / f"health_sites_final_{label}_manifest.json"
    )


def validate_health_snapshot(args):
    """
    Verify that the Health layer consumed by the canonical Service layer
    was built with the temporal parameters requested by the pipeline.
    """
    manifest_path = health_manifest_path(args)

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Manifest Health non trovato: "
            f"{manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    expected_date = (
        args.health_reference_date
        .date()
        .isoformat()
    )

    actual_code = str(
        manifest.get(
            "municipality_code",
            "",
        )
    ).strip().zfill(6)

    actual_date = str(
        manifest.get(
            "pharmacy_reference_date",
            "",
        )
    ).strip()

    actual_hospital_year = str(
        manifest.get(
            "hospital_reference_year",
            "",
        )
    ).strip()

    expected_hospital_year = str(
        args.hospital_year
    ).strip()

    errors = []

    if actual_code != args.municipality_code:
        errors.append(
            "municipality_code: "
            f"atteso {args.municipality_code}, "
            f"trovato {actual_code or '<missing>'}"
        )

    if actual_date != expected_date:
        errors.append(
            "pharmacy_reference_date: "
            f"atteso {expected_date}, "
            f"trovato {actual_date or '<missing>'}"
        )

    if actual_hospital_year != expected_hospital_year:
        errors.append(
            "hospital_reference_year: "
            f"atteso {expected_hospital_year}, "
            f"trovato {actual_hospital_year or '<missing>'}"
        )

    if errors:
        raise RuntimeError(
            "Health snapshot incompatibile con i parametri "
            "del Service layer:\n  - "
            + "\n  - ".join(errors)
        )

    return manifest_path


def load_inputs(args):
    health_manifest = validate_health_snapshot(
        args
    )

    school_path = school_input_path(
        args
    )

    health_path = health_input_path(
        args
    )

    if not school_path.exists():
        raise FileNotFoundError(
            f"Dataset Education non trovato: {school_path}"
        )

    if not health_path.exists():
        raise FileNotFoundError(
            f"Dataset Health non trovato: {health_path}"
        )

    schools = pd.read_parquet(
        school_path
    )

    health = pd.read_parquet(
        health_path
    )

    return (
        schools,
        health,
        school_path,
        health_path,
    )


def map_education(
    schools,
    school_year,
):
    rows = []

    canonical_source_columns = {
        "school_site_id",
        "category",
        "subcategory",
        "name",
        "municipality_code",
        "municipality_name",
        "address",
        "postal_code",
        "longitude",
        "latitude",
        "coordinate_origin",
        "coordinate_resolution",
        "confidence",
        "source_system",
        "source_record_id",
        "reference_period",
        "usable_for_accessibility",
    }

    for _, row in schools.iterrows():
        usable = bool(
            row.get(
                "usable_for_accessibility",
                False,
            )
        )

        raw_id = row.get(
            "school_site_id"
        )

        extra = {
            column:
                row.get(column)
            for column in schools.columns
            if column not in canonical_source_columns
            and column != "geometry"
        }

        # Preserve key spatial/provenance semantics from Education.
        extra[
            "original_coordinate_origin"
        ] = row.get(
            "coordinate_origin"
        )

        extra[
            "original_reference_period"
        ] = row.get(
            "reference_period"
        )

        rows.append(
            {
                "service_site_id":
                    canonical_id(
                        "education",
                        raw_id,
                    ),

                "domain_service_site_id":
                    clean_text(
                        raw_id
                    ),

                "category":
                    clean_text(
                        row.get(
                            "category"
                        )
                    )
                    or "education",

                "subcategory":
                    clean_text(
                        row.get(
                            "subcategory"
                        )
                    ),

                "name":
                    clean_text(
                        row.get(
                            "name"
                        )
                    ),

                "municipality_code":
                    clean_text(
                        row.get(
                            "municipality_code"
                        )
                    ),

                "municipality_name":
                    clean_text(
                        row.get(
                            "municipality_name"
                        )
                    ),

                "address":
                    clean_text(
                        row.get(
                            "address"
                        )
                    ),

                "postal_code":
                    clean_text(
                        row.get(
                            "postal_code"
                        )
                    ),

                "longitude":
                    clean_float(
                        row.get(
                            "longitude"
                        )
                    ),

                "latitude":
                    clean_float(
                        row.get(
                            "latitude"
                        )
                    ),

                "source_system":
                    clean_text(
                        row.get(
                            "source_system"
                        )
                    )
                    or "MIM",

                "source_dataset":
                    (
                        "MIM school sites "
                        f"{school_year}"
                    ),

                "source_record_id":
                    clean_text(
                        row.get(
                            "source_record_id"
                        )
                    ),

                "reference_period":
                    clean_text(
                        row.get(
                            "reference_period"
                        )
                    )
                    or str(
                        school_year
                    ),

                "coordinate_source":
                    clean_text(
                        row.get(
                            "coordinate_origin"
                        )
                    ),

                "coordinate_resolution":
                    clean_text(
                        row.get(
                            "coordinate_resolution"
                        )
                    ),

                "confidence":
                    clean_text(
                        row.get(
                            "confidence"
                        )
                    ),

                "resolution_status":
                    (
                        "resolved"
                        if usable
                        else "unresolved"
                    ),

                "usable_for_accessibility":
                    usable,

                "capacity_value":
                    None,

                "capacity_unit":
                    None,

                "provenance_json":
                    compact_json(
                        extra
                    ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=CANONICAL_COLUMNS,
    )


def health_reference_period(row):
    subcategory = clean_text(
        row.get(
            "subcategory"
        )
    )

    if (
        subcategory
        == "pharmacy"
    ):
        value = row.get(
            "reference_date"
        )

        if pd.notna(
            value
        ):
            return pd.Timestamp(
                value
            ).date().isoformat()

    if (
        subcategory
        == "hospital"
    ):
        value = row.get(
            "reference_year"
        )

        if pd.notna(
            value
        ):
            return str(
                int(
                    float(value)
                )
            )

    return None


def map_health(
    health,
):
    rows = []

    canonical_source_columns = {
        "service_site_id",
        "category",
        "subcategory",
        "name",
        "municipality_code",
        "municipality_name",
        "address",
        "postal_code",
        "longitude",
        "latitude",
        "source_system",
        "source_dataset",
        "source_record_id",
        "coordinate_source",
        "coordinate_resolution",
        "confidence",
        "resolution_status",
        "usable_for_accessibility",
    }

    for _, row in health.iterrows():
        usable = bool(
            row.get(
                "usable_for_accessibility",
                False,
            )
        )

        raw_id = row.get(
            "service_site_id"
        )

        subcategory = clean_text(
            row.get(
                "subcategory"
            )
        )

        extra = {
            column:
                row.get(column)
            for column in health.columns
            if column not in canonical_source_columns
            and column != "geometry"
        }

        capacity_value = None
        capacity_unit = None

        if (
            subcategory
            == "hospital"
        ):
            total_beds = clean_float(
                row.get(
                    "total_beds"
                )
            )

            if total_beds is not None:
                capacity_value = (
                    total_beds
                )
                capacity_unit = (
                    "beds"
                )

        rows.append(
            {
                "service_site_id":
                    canonical_id(
                        "health",
                        raw_id,
                    ),

                "domain_service_site_id":
                    clean_text(
                        raw_id
                    ),

                "category":
                    clean_text(
                        row.get(
                            "category"
                        )
                    )
                    or "health",

                "subcategory":
                    subcategory,

                "name":
                    clean_text(
                        row.get(
                            "name"
                        )
                    ),

                "municipality_code":
                    clean_text(
                        row.get(
                            "municipality_code"
                        )
                    ),

                "municipality_name":
                    clean_text(
                        row.get(
                            "municipality_name"
                        )
                    ),

                "address":
                    clean_text(
                        row.get(
                            "address"
                        )
                    ),

                "postal_code":
                    clean_text(
                        row.get(
                            "postal_code"
                        )
                    ),

                "longitude":
                    clean_float(
                        row.get(
                            "longitude"
                        )
                    ),

                "latitude":
                    clean_float(
                        row.get(
                            "latitude"
                        )
                    ),

                "source_system":
                    clean_text(
                        row.get(
                            "source_system"
                        )
                    ),

                "source_dataset":
                    clean_text(
                        row.get(
                            "source_dataset"
                        )
                    ),

                "source_record_id":
                    clean_text(
                        row.get(
                            "source_record_id"
                        )
                    ),

                "reference_period":
                    health_reference_period(
                        row
                    ),

                "coordinate_source":
                    clean_text(
                        row.get(
                            "coordinate_source"
                        )
                    ),

                "coordinate_resolution":
                    clean_text(
                        row.get(
                            "coordinate_resolution"
                        )
                    ),

                "confidence":
                    clean_text(
                        row.get(
                            "confidence"
                        )
                    ),

                "resolution_status":
                    clean_text(
                        row.get(
                            "resolution_status"
                        )
                    )
                    or (
                        "resolved"
                        if usable
                        else "unresolved"
                    ),

                "usable_for_accessibility":
                    usable,

                "capacity_value":
                    capacity_value,

                "capacity_unit":
                    capacity_unit,

                "provenance_json":
                    compact_json(
                        extra
                    ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=CANONICAL_COLUMNS,
    )


def validate_canonical(
    services,
    municipality_code,
):
    if services.empty:
        raise RuntimeError(
            "Canonical Service layer vuoto."
        )

    duplicates = (
        services[
            "service_site_id"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicates.any():
        duplicate_ids = (
            services.loc[
                duplicates,
                "service_site_id",
            ]
            .tolist()
        )

        raise RuntimeError(
            "service_site_id duplicati: "
            f"{duplicate_ids[:20]}"
        )

    wrong_municipality = (
        services[
            "municipality_code"
        ]
        .notna()
        & (
            services[
                "municipality_code"
            ]
            != municipality_code
        )
    )

    if wrong_municipality.any():
        count = int(
            wrong_municipality.sum()
        )

        raise RuntimeError(
            f"{count} servizi appartengono a "
            "un comune diverso da quello richiesto."
        )

    missing_coordinates_usable = (
        services[
            "usable_for_accessibility"
        ]
        & (
            services[
                "longitude"
            ].isna()
            | services[
                "latitude"
            ].isna()
        )
    )

    if (
        missing_coordinates_usable.any()
    ):
        count = int(
            missing_coordinates_usable
            .sum()
        )

        raise RuntimeError(
            f"{count} servizi marcati usable "
            "non hanno coordinate."
        )


def make_geodataframe(
    services,
):
    geometry = []

    for _, row in services.iterrows():
        longitude = row[
            "longitude"
        ]

        latitude = row[
            "latitude"
        ]

        if (
            pd.notna(
                longitude
            )
            and pd.notna(
                latitude
            )
        ):
            geometry.append(
                Point(
                    float(
                        longitude
                    ),
                    float(
                        latitude
                    ),
                )
            )
        else:
            geometry.append(
                None
            )

    return gpd.GeoDataFrame(
        services,
        geometry=geometry,
        crs="EPSG:4326",
    )


def main():
    args = parse_args()

    (
        schools,
        health,
        school_path,
        health_path,
    ) = load_inputs(
        args
    )

    education_services = (
        map_education(
            schools,
            args.school_year,
        )
    )

    health_services = (
        map_health(
            health
        )
    )

    services = pd.concat(
        [
            education_services,
            health_services,
        ],
        ignore_index=True,
    )

    validate_canonical(
        services,
        args.municipality_code,
    )

    services = (
        services.sort_values(
            [
                "category",
                "subcategory",
                "service_site_id",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    gdf = make_geodataframe(
        services
    )

    label = (
        args.health_reference_date
        .strftime("%Y%m%d")
    )

    output_dir = (
        PROCESSED_SERVICES_DIR
        / args.municipality_code
    )

    feature_dir = (
        FEATURES_SERVICES_DIR
        / args.municipality_code
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        output_dir
        / (
            "service_sites_"
            f"{args.school_year}_"
            f"{label}.parquet"
        )
    )

    csv_path = (
        feature_dir
        / (
            "service_sites_"
            f"{args.school_year}_"
            f"{label}.csv"
        )
    )

    manifest_path = (
        feature_dir
        / (
            "service_sites_"
            f"{args.school_year}_"
            f"{label}_manifest.json"
        )
    )

    gdf.to_parquet(
        parquet_path,
        index=False,
    )

    services.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    category_counts = (
        services[
            "category"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    subcategory_counts = (
        services[
            "subcategory"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    confidence_counts = (
        services[
            "confidence"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    coordinate_resolution_counts = (
        services[
            "coordinate_resolution"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    manifest = {
        "generated_at_utc":
            utc_now_iso(),

        "municipality_code":
            args.municipality_code,

        "inputs": {
            "education":
                str(
                    school_path
                ),

            "health":
                str(
                    health_path
                ),
        },

        "total_service_sites":
            int(
                len(
                    services
                )
            ),

        "usable_for_accessibility":
            int(
                services[
                    "usable_for_accessibility"
                ].sum()
            ),

        "category_counts": {
            str(key):
                int(value)
            for key, value
            in category_counts.items()
        },

        "subcategory_counts": {
            str(key):
                int(value)
            for key, value
            in subcategory_counts.items()
        },

        "confidence_counts": {
            str(key):
                int(value)
            for key, value
            in confidence_counts.items()
        },

        "coordinate_resolution_counts": {
            str(key):
                int(value)
            for key, value
            in coordinate_resolution_counts.items()
        },

        "canonical_columns":
            CANONICAL_COLUMNS
            + ["geometry"],

        "notes": [
            (
                "Il canonical Service layer contiene solo gli attributi "
                "comuni necessari agli algoritmi urbani."
            ),
            (
                "Gli attributi specifici dei domini Education e Health "
                "rimangono nei rispettivi Silver e sono inoltre conservati "
                "in provenance_json."
            ),
            (
                "capacity_value è valorizzato attualmente solo quando "
                "esiste una capacità direttamente interpretabile "
                "(es. posti letto ospedalieri)."
            ),
            (
                "La presenza nel canonical layer non implica automaticamente "
                "utilizzabilità: usare sempre usable_for_accessibility."
            ),
        ],
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n===================================="
    )
    print(
        " CANONICAL SERVICE LAYER"
    )
    print(
        "===================================="
    )

    print(
        "Comune: "
        f"{args.municipality_code}"
    )

    print(
        "Education input: "
        f"{len(education_services)}"
    )

    print(
        "Health input: "
        f"{len(health_services)}"
    )

    print(
        "Totale service sites: "
        f"{len(services)}"
    )

    print(
        "Usabili per accessibility: "
        f"{int(services['usable_for_accessibility'].sum())}"
    )

    print(
        "\nCategory:"
    )
    print(
        services[
            "category"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nSubcategory:"
    )
    print(
        services[
            "subcategory"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nCoordinate resolution:"
    )
    print(
        services[
            "coordinate_resolution"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nConfidence:"
    )
    print(
        services[
            "confidence"
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
        f"✓ {parquet_path}"
    )
    print(
        f"✓ {csv_path}"
    )
    print(
        f"✓ {manifest_path}"
    )


if __name__ == "__main__":
    main()
