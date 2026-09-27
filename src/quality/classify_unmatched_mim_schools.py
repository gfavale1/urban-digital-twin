import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
FEATURES_MIM_DIR = ROOT / "data" / "features" / "mim"


MISSING_VALUES = {
    "",
    "NAN",
    "NONE",
    "NULL",
    "N/A",
    "NA",
    "N.D.",
    "ND",
    "NON DISPONIBILE",
    "NON DISP.",
    "-",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classificazione nazionale dei record MIM statali che non trovano "
            "un match esatto nell'Anagrafe dell'edilizia scolastica."
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


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""

    value = str(value).strip()

    if value.upper() in MISSING_VALUES:
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

    value = value.upper()

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def contains_any(text, patterns):
    return any(
        pattern in text
        for pattern in patterns
    )


def classify(row):
    name = normalize_text(
        row.get(
            "school_name"
        )
    )

    characteristics = normalize_text(
        row.get(
            "DESCRIZIONECARATTERISTICASCUOLA"
        )
    )

    grade = normalize_text(
        row.get(
            "grade_description"
        )
    )

    address = normalize_text(
        row.get(
            "school_address"
        )
    )

    combined = " | ".join(
        [
            name,
            characteristics,
            grade,
        ]
    )

    # Restricted-access educational services inside a prison.
    if (
        contains_any(
            combined,
            [
                "CARCERAR",
                "CASA CIRCONDARIALE",
                "ISTITUTO PENITENZIARIO",
            ],
        )
    ):
        return {
            "unmatched_class":
                "restricted_correctional_service",

            "creates_public_school_site":
                False,

            "needs_geolocation":
                False,

            "site_handling":
                "exclude_from_general_school_accessibility",

            "classification_reason":
                (
                    "Servizio scolastico rivolto a utenza carceraria; "
                    "non rappresenta un punto di accesso scolastico ordinario."
                ),
        }

    # Restricted-access educational services inside a hospital.
    if (
        contains_any(
            combined,
            [
                "OSPEDALIER",
                "C O IST OSPEDALIERO",
                "SCUOLA OSPEDALIERA",
            ],
        )
    ):
        return {
            "unmatched_class":
                "restricted_hospital_service",

            "creates_public_school_site":
                False,

            "needs_geolocation":
                False,

            "site_handling":
                "exclude_from_general_school_accessibility",

            "classification_reason":
                (
                    "Servizio scolastico ospedaliero; non rappresenta "
                    "un ordinario sito scolastico accessibile alla popolazione generale."
                ),
        }

    # Evening / second-level pathway: educational offering, not a separate
    # physical site by itself.
    if (
        contains_any(
            combined,
            [
                "CORSO SERALE",
                "PERCORSO II LIVELLO",
                "SECONDO LIVELLO",
            ],
        )
    ):
        return {
            "unmatched_class":
                "non_separate_evening_course",

            "creates_public_school_site":
                False,

            "needs_geolocation":
                False,

            "site_handling":
                "link_to_existing_school_site_if_needed",

            "classification_reason":
                (
                    "Percorso/corso serale: è un'offerta didattica associata "
                    "a una sede esistente, non un nuovo sito fisico."
                ),
        }

    # Default conservative rule: a normal unmatched school record remains a
    # physical-site candidate and must enter the automatic localization pipeline.
    return {
        "unmatched_class":
            "physical_school_candidate",

        "creates_public_school_site":
            True,

        "needs_geolocation":
            True,

        "site_handling":
            "automatic_geolocation_pipeline",

        "classification_reason":
            (
                "Record scolastico ordinario non riconducibile a servizio speciale "
                "o percorso non separato; candidato a vero sito fisico."
            ),
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

    features_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = (
        processed_dir
        / (
            "school_building_unmatched_"
            f"{args.school_year}_from_{args.building_year}.parquet"
        )
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"File unmatched non trovato: {input_path}"
        )

    df = pd.read_parquet(
        input_path
    )

    classifications = (
        df.apply(
            classify,
            axis=1,
            result_type="expand",
        )
    )

    result = pd.concat(
        [
            df.reset_index(
                drop=True
            ),
            classifications.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    output_csv = (
        features_dir
        / (
            "school_unmatched_classification_"
            f"{args.school_year}.csv"
        )
    )

    output_parquet = (
        processed_dir
        / (
            "school_unmatched_classification_"
            f"{args.school_year}.parquet"
        )
    )

    result.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    result.to_parquet(
        output_parquet,
        index=False,
    )

    print(
        "\n===================================="
    )
    print(
        " MIM UNMATCHED CLASSIFICATION"
    )
    print(
        "===================================="
    )

    print(
        f"Rows: {len(result)}"
    )

    print(
        "\nClasses:"
    )

    print(
        result[
            "unmatched_class"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== RESULTS ==="
    )

    print(
        result[
            [
                "school_code",
                "school_name",
                "school_address",
                "DESCRIZIONECARATTERISTICASCUOLA",
                "unmatched_class",
                "creates_public_school_site",
                "needs_geolocation",
                "site_handling",
            ]
        ]
        .sort_values(
            [
                "unmatched_class",
                "school_code",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nPhysical school candidates requiring geolocation:"
    )

    physical = result[
        result[
            "needs_geolocation"
        ]
        == True
    ]

    print(
        f"{len(physical)}"
    )

    if not physical.empty:
        print(
            physical[
                [
                    "school_code",
                    "school_name",
                    "school_address",
                    "institute_reference_code",
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
        "Le regole dipendono esclusivamente da attributi MIM espliciti "
        "e non contengono codici o nomi specifici del comune target."
    )


if __name__ == "__main__":
    main()
