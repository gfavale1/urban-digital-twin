import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "mim"
PROCESSED_DIR = ROOT / "data" / "processed" / "mim"

DEFAULT_SCHOOL_YEAR = "202425"

# The MIM catalogue currently reports the 2024/25 distribution
# as data available "al 31/08/2025".
CATALOG_DATA_AS_OF = {
    "202425": "2025-08-31",
}

# MIM has used more than one URL layout over time.
# We try only official dati.istruzione.it URLs.
URL_PATTERNS = [
    "https://dati.istruzione.it/opendata/opendata/catalog/{filename}",
    "https://dati.istruzione.it/opendata/opendata/catalog/{dataset}/{filename}",
    "https://dati.istruzione.it/opendata/opendata/catalogo/elements1/leaf/{filename}",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scarica e prepara l'Anagrafe MIM delle scuole "
            "statali e paritarie per uno specifico anno scolastico."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--municipality-name",
        required=True,
        help="Nome del comune come riportato da MIM, es. MATERA.",
    )

    parser.add_argument(
        "--school-year",
        default=DEFAULT_SCHOOL_YEAR,
        help="Anno scolastico senza separatore, default 202425.",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Riscarica i file raw anche se già presenti.",
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    args.municipality_name = (
        str(args.municipality_name)
        .strip()
        .upper()
    )

    args.school_year = (
        str(args.school_year)
        .strip()
    )

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "municipality-code deve avere esattamente 6 cifre."
        )

    if not re.fullmatch(
        r"\d{6}",
        args.school_year,
    ):
        raise ValueError(
            "school-year deve avere 6 cifre, es. 202425."
        )

    return args


def candidate_filenames(
    dataset,
    school_year,
):
    """
    MIM filenames normally encode:
      dataset + school year + snapshot date.

    For 2024/25 we first try the final 31/08/2025 distribution
    reported by the current catalogue, then the historical
    01/09/2024 snapshot filename known for this school year.
    """

    if school_year == "202425":
        snapshot_dates = [
            "20250831",
            "20240901",
        ]
    else:
        raise ValueError(
            "Questo script è stato predisposto in modo conservativo "
            "per il 2024/25. Aggiungere esplicitamente gli snapshot "
            "per altri anni prima di usarlo."
        )

    return [
        f"{dataset}{school_year}{snapshot}.csv"
        for snapshot in snapshot_dates
    ]


def download_official_file(
    dataset,
    school_year,
    target_dir,
    refresh,
):
    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filenames = candidate_filenames(
        dataset,
        school_year,
    )

    # Reuse an already downloaded supported file unless refresh requested.
    if not refresh:
        for filename in filenames:
            local_path = (
                target_dir
                / filename
            )

            if (
                local_path.exists()
                and local_path.stat().st_size > 0
            ):
                return (
                    local_path,
                    "existing_local_file",
                )

    headers = {
        "User-Agent": (
            "urban-digital-twin-thesis/1.0 "
            "(academic open-data ingestion)"
        )
    }

    errors = []

    for filename in filenames:
        for pattern in URL_PATTERNS:
            url = pattern.format(
                dataset=dataset,
                filename=filename,
            )

            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=90,
                )

                if response.status_code != 200:
                    errors.append(
                        f"{response.status_code} {url}"
                    )
                    continue

                content = response.content

                # Reject an HTML error/login page accidentally returned as 200.
                prefix = (
                    content[:200]
                    .decode(
                        "utf-8",
                        errors="ignore",
                    )
                    .lower()
                )

                if (
                    "<html" in prefix
                    or "<!doctype html" in prefix
                ):
                    errors.append(
                        f"HTML instead of CSV {url}"
                    )
                    continue

                local_path = (
                    target_dir
                    / filename
                )

                local_path.write_bytes(
                    content
                )

                return (
                    local_path,
                    url,
                )

            except requests.RequestException as exc:
                errors.append(
                    f"{type(exc).__name__}: {url}"
                )

    raise RuntimeError(
        "Impossibile scaricare il dataset MIM da URL ufficiali.\n"
        + "\n".join(
            errors[-12:]
        )
    )


def read_csv_robust(
    path,
):
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding,
                dtype=str,
            )

            if len(
                dataframe.columns
            ) <= 1:
                continue

            return (
                dataframe,
                encoding,
            )

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Impossibile leggere {path}: {last_error}"
    )


def normalize_columns(
    dataframe,
):
    dataframe = dataframe.copy()

    dataframe.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()
        .upper()
        for column in dataframe.columns
    ]

    return dataframe


def first_existing_column(
    dataframe,
    candidates,
):
    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def canonicalize_registry(
    dataframe,
    registry_type,
    municipality_code,
    municipality_name,
    school_year,
    source_path,
    source_url,
):
    dataframe = normalize_columns(
        dataframe
    )

    municipality_column = (
        first_existing_column(
            dataframe,
            [
                "DESCRIZIONECOMUNE",
                "DENOMINAZIONECOMUNE",
                "COMUNE",
            ],
        )
    )

    if municipality_column is None:
        raise RuntimeError(
            "Non trovo la colonna del comune. "
            f"Colonne disponibili: {list(dataframe.columns)}"
        )

    municipality_values = (
        dataframe[
            municipality_column
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    filtered = (
        dataframe.loc[
            municipality_values
            == municipality_name
        ]
        .copy()
    )

    school_code_col = (
        first_existing_column(
            filtered,
            [
                "CODICESCUOLA",
                "CODICESCUOLAPLESSO",
            ],
        )
    )

    school_name_col = (
        first_existing_column(
            filtered,
            [
                "DENOMINAZIONESCUOLA",
                "DENOMINAZIONESCUOLAPLESSO",
            ],
        )
    )

    address_col = (
        first_existing_column(
            filtered,
            [
                "INDIRIZZOSCUOLA",
                "INDIRIZZO",
            ],
        )
    )

    postal_code_col = (
        first_existing_column(
            filtered,
            [
                "CAPSCUOLA",
                "CAP",
            ],
        )
    )

    province_col = (
        first_existing_column(
            filtered,
            [
                "PROVINCIA",
                "DENOMINAZIONEPROVINCIA",
            ],
        )
    )

    grade_col = (
        first_existing_column(
            filtered,
            [
                "DESCRIZIONETIPOLOGIAGRADOISTRUZIONESCUOLA",
                "DESCRIZIONETIPOLOGIAGRADOISTRUZIONE",
            ],
        )
    )

    institute_ref_col = (
        first_existing_column(
            filtered,
            [
                "CODICEISTITUTORIFERIMENTO",
                "CODICEISTITUTO",
            ],
        )
    )

    institute_name_col = (
        first_existing_column(
            filtered,
            [
                "DENOMINAZIONEISTITUTORIFERIMENTO",
                "DENOMINAZIONEISTITUTO",
            ],
        )
    )

    if school_code_col is None:
        raise RuntimeError(
            "CODICESCUOLA non trovato nel file MIM."
        )

    filtered[
        "school_code"
    ] = (
        filtered[
            school_code_col
        ]
        .astype(str)
        .str.strip()
    )

    filtered[
        "school_name"
    ] = (
        filtered[
            school_name_col
        ].astype(str).str.strip()
        if school_name_col
        else None
    )

    filtered[
        "school_address"
    ] = (
        filtered[
            address_col
        ].astype(str).str.strip()
        if address_col
        else None
    )

    filtered[
        "postal_code"
    ] = (
        filtered[
            postal_code_col
        ].astype(str).str.strip()
        if postal_code_col
        else None
    )

    filtered[
        "province_name"
    ] = (
        filtered[
            province_col
        ].astype(str).str.strip()
        if province_col
        else None
    )

    filtered[
        "grade_description"
    ] = (
        filtered[
            grade_col
        ].astype(str).str.strip()
        if grade_col
        else None
    )

    filtered[
        "institute_reference_code"
    ] = (
        filtered[
            institute_ref_col
        ].astype(str).str.strip()
        if institute_ref_col
        else None
    )

    filtered[
        "institute_reference_name"
    ] = (
        filtered[
            institute_name_col
        ].astype(str).str.strip()
        if institute_name_col
        else None
    )

    filtered[
        "registry_type"
    ] = registry_type

    filtered[
        "municipality_code"
    ] = municipality_code

    filtered[
        "municipality_name"
    ] = municipality_name

    filtered[
        "source_school_year"
    ] = school_year

    filtered[
        "source_catalog_data_as_of"
    ] = CATALOG_DATA_AS_OF.get(
        school_year
    )

    filtered[
        "source_filename"
    ] = source_path.name

    filtered[
        "source_url"
    ] = source_url

    filtered[
        "ingested_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    return filtered


def main():
    args = parse_args()

    print(
        "\n===================================="
    )
    print(
        " MIM SCHOOL REGISTRY 2024/25"
    )
    print(
        "===================================="
    )

    raw_year_dir = (
        RAW_DIR
        / "schools"
        / args.school_year
    )

    state_path, state_url = (
        download_official_file(
            dataset="SCUANAGRAFESTAT",
            school_year=args.school_year,
            target_dir=raw_year_dir,
            refresh=args.refresh,
        )
    )

    paritary_path, paritary_url = (
        download_official_file(
            dataset="SCUANAGRAFEPAR",
            school_year=args.school_year,
            target_dir=raw_year_dir,
            refresh=args.refresh,
        )
    )

    print(
        f"✓ Statali raw: {state_path.name}"
    )
    print(
        f"✓ Paritarie raw: {paritary_path.name}"
    )

    state_raw, state_encoding = (
        read_csv_robust(
            state_path
        )
    )

    paritary_raw, paritary_encoding = (
        read_csv_robust(
            paritary_path
        )
    )

    print(
        "\n=== NATIONAL RAW ==="
    )
    print(
        f"Statali: {len(state_raw)}"
    )
    print(
        f"Paritarie: {len(paritary_raw)}"
    )
    print(
        f"Encoding statali: {state_encoding}"
    )
    print(
        f"Encoding paritarie: {paritary_encoding}"
    )

    state_local = canonicalize_registry(
        dataframe=state_raw,
        registry_type="state",
        municipality_code=args.municipality_code,
        municipality_name=args.municipality_name,
        school_year=args.school_year,
        source_path=state_path,
        source_url=state_url,
    )

    paritary_local = canonicalize_registry(
        dataframe=paritary_raw,
        registry_type="paritary",
        municipality_code=args.municipality_code,
        municipality_name=args.municipality_name,
        school_year=args.school_year,
        source_path=paritary_path,
        source_url=paritary_url,
    )

    combined = pd.concat(
        [
            state_local,
            paritary_local,
        ],
        ignore_index=True,
        sort=False,
    )

    duplicate_codes = (
        combined[
            "school_code"
        ]
        .duplicated(
            keep=False
        )
    )

    output_dir = (
        PROCESSED_DIR
        / args.municipality_code
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        output_dir
        / (
            "schools_registry_"
            f"{args.school_year}.parquet"
        )
    )

    csv_path = (
        output_dir
        / (
            "schools_registry_"
            f"{args.school_year}.csv"
        )
    )

    manifest_path = (
        output_dir
        / (
            "schools_registry_"
            f"{args.school_year}_manifest.json"
        )
    )

    combined.to_parquet(
        parquet_path,
        index=False,
    )

    combined.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "municipality_code":
            args.municipality_code,

        "municipality_name":
            args.municipality_name,

        "school_year":
            args.school_year,

        "catalog_data_as_of":
            CATALOG_DATA_AS_OF.get(
                args.school_year
            ),

        "state_raw_filename":
            state_path.name,

        "state_source_url":
            state_url,

        "paritary_raw_filename":
            paritary_path.name,

        "paritary_source_url":
            paritary_url,

        "state_national_rows":
            int(
                len(
                    state_raw
                )
            ),

        "paritary_national_rows":
            int(
                len(
                    paritary_raw
                )
            ),

        "state_local_rows":
            int(
                len(
                    state_local
                )
            ),

        "paritary_local_rows":
            int(
                len(
                    paritary_local
                )
            ),

        "combined_local_rows":
            int(
                len(
                    combined
                )
            ),

        "duplicate_school_code_rows":
            int(
                duplicate_codes.sum()
            ),

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

    print(
        "\n=== MUNICIPALITY ==="
    )
    print(
        f"Comune: {args.municipality_name} "
        f"({args.municipality_code})"
    )
    print(
        f"Statali: {len(state_local)}"
    )
    print(
        f"Paritarie: {len(paritary_local)}"
    )
    print(
        f"Totale: {len(combined)}"
    )
    print(
        "School-code duplicate rows: "
        f"{int(duplicate_codes.sum())}"
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

    print(
        "\nNOTA:"
    )
    print(
        "Il dataset 2026/27 esistente non viene modificato."
    )


if __name__ == "__main__":
    main()
