from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
RAW_MIM_DIR = ROOT / "data" / "raw" / "mim"
PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"

DEFAULT_SCHOOL_YEAR = "202425"
DEFAULT_BUILDING_YEAR = "202425"

# MIM catalogue metadata for the validated thesis snapshot.
CATALOG_DATA_AS_OF = {
    "202425": "2025-08-31",
}

# MIM has used more than one official URL layout over time.
URL_PATTERNS = [
    "https://dati.istruzione.it/opendata/opendata/catalog/{filename}",
    "https://dati.istruzione.it/opendata/opendata/catalog/{dataset}/{filename}",
    "https://dati.istruzione.it/opendata/opendata/catalogo/elements1/leaf/{filename}",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Canonical MIM ingestion for school registry and "
            "state school-building registry."
        )
    )

    parser.add_argument(
        "--step",
        choices=("prepare", "registry", "buildings"),
        default="prepare",
        help=(
            "prepare esegue registry + buildings; "
            "registry o buildings eseguono il singolo step."
        ),
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre.",
    )

    parser.add_argument(
        "--municipality-name",
        default=None,
        help=(
            "Nome del comune come riportato da MIM. "
            "Necessario per registry/prepare."
        ),
    )

    parser.add_argument(
        "--school-year",
        default=DEFAULT_SCHOOL_YEAR,
        help="Anno scolastico senza separatore.",
    )

    parser.add_argument(
        "--building-year",
        default=DEFAULT_BUILDING_YEAR,
        help="Anno dell'Anagrafe edilizia scolastica.",
    )

    parser.add_argument(
        "--building-file",
        default=None,
        help=(
            "CSV edilizia opzionale. Se omesso viene cercato "
            "sotto data/raw/mim/buildings/<building-year>/."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Riscarica i registry scuole raw anche se già presenti.",
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

    if args.municipality_name is not None:
        args.municipality_name = (
            str(args.municipality_name)
            .strip()
            .upper()
        )

    if (
        args.step in {"prepare", "registry"}
        and not args.municipality_name
    ):
        raise ValueError(
            "--municipality-name è richiesto per step prepare/registry."
        )

    for field_name in ("school_year", "building_year"):
        value = str(
            getattr(args, field_name)
        ).strip()

        if not re.fullmatch(r"\d{6}", value):
            raise ValueError(
                f"{field_name} deve avere 6 cifre, es. 202425."
            )

        setattr(args, field_name, value)

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

    errors = []

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
            errors.append(
                f"{encoding}: {exc}"
            )

    raise RuntimeError(
        f"Impossibile leggere {path}\n"
        + "\n".join(
            errors
        )
    )


def normalize_registry_columns(
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
    dataframe = normalize_registry_columns(
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


def normalize_building_columns(
    dataframe,
):
    dataframe = dataframe.copy()

    normalized_columns = []

    for column in dataframe.columns:
        value = (
            str(column)
            .replace("\ufeff", "")
            .strip()
            .upper()
        )

        # MIM historical CSVs are not always perfectly consistent:
        # column names can contain spaces, underscores or punctuation.
        # Canonicalize them so that, for example:
        #
        # "INDIRIZZO EDIFICIO"
        # "INDIRIZZO_EDIFICIO"
        # "INDIRIZZOEDIFICIO"
        #
        # all become INDIRIZZOEDIFICIO.
        value = re.sub(
            r"[^A-Z0-9]+",
            "",
            value,
        )

        normalized_columns.append(
            value
        )

    if len(
        normalized_columns
    ) != len(
        set(
            normalized_columns
        )
    ):
        duplicates = sorted(
            {
                column
                for column in normalized_columns
                if normalized_columns.count(
                    column
                ) > 1
            }
        )

        raise RuntimeError(
            "La normalizzazione dei nomi colonna produce duplicati: "
            + ", ".join(
                duplicates
            )
        )

    dataframe.columns = (
        normalized_columns
    )

    return dataframe


def clean_string_series(
    series,
):
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


def find_building_file(
    building_year,
    explicit_path=None,
):
    if explicit_path:
        path = Path(
            explicit_path
        )

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        return path

    directory = (
        RAW_MIM_DIR
        / "buildings"
        / building_year
    )

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory edilizia non trovata: {directory}"
        )

    candidates = sorted(
        directory.glob(
            "*.csv"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"Nessun CSV trovato in {directory}"
        )

    # Prefer the official MIM school-building registry naming.
    preferred = [
        path
        for path in candidates
        if "EDIANAGRAFESTA" in path.name.upper()
    ]

    if len(
        preferred
    ) == 1:
        return preferred[0]

    if len(
        preferred
    ) > 1:
        # Latest lexical filename normally corresponds to latest snapshot.
        return sorted(
            preferred
        )[-1]

    if len(
        candidates
    ) == 1:
        return candidates[0]

    raise RuntimeError(
        "Più CSV edilizia trovati e nessun file "
        "EDIANAGRAFESTA identificabile automaticamente:\n"
        + "\n".join(
            str(path)
            for path in candidates
        )
    )


def load_school_registry(
    municipality_code,
    school_year,
):
    path = (
        PROCESSED_MIM_DIR
        / municipality_code
        / (
            "schools_registry_"
            f"{school_year}.parquet"
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Registry scuole non trovato: {path}"
        )

    dataframe = pd.read_parquet(
        path
    )

    required = {
        "school_code",
        "registry_type",
        "municipality_code",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Colonne mancanti nel registry scuole: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    dataframe = dataframe.copy()

    dataframe[
        "school_code"
    ] = clean_string_series(
        dataframe[
            "school_code"
        ]
    )

    dataframe[
        "registry_type"
    ] = (
        dataframe[
            "registry_type"
        ]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return (
        dataframe,
        path,
    )


def canonicalize_buildings(
    raw,
    source_path,
    building_year,
):
    raw = normalize_building_columns(
        raw
    )

    school_code_col = (
        first_existing_column(
            raw,
            [
                "CODICESCUOLA",
                "CODICESCUOLAPLESSO",
            ],
        )
    )

    building_code_col = (
        first_existing_column(
            raw,
            [
                "CODICEEDIFICIO",
                "CODICEEDIFICIOSCOLASTICO",
            ],
        )
    )

    municipality_col = (
        first_existing_column(
            raw,
            [
                "COMUNE",
                "DENOMINAZIONECOMUNE",
                "DESCRIZIONECOMUNE",
            ],
        )
    )

    address_col = (
        first_existing_column(
            raw,
            [
                "INDIRIZZO",
                "INDIRIZZOEDIFICIO",
                "INDIRIZZOEDIFICIOSCOLASTICO",
            ],
        )
    )

    address_type_col = (
        first_existing_column(
            raw,
            [
                "TIPOLOGIAINDIRIZZO",
                "TIPOINDIRIZZO",
            ],
        )
    )

    address_name_col = (
        first_existing_column(
            raw,
            [
                "DENOMINAZIONEINDIRIZZO",
                "NOMEINDIRIZZO",
            ],
        )
    )

    civic_number_col = (
        first_existing_column(
            raw,
            [
                "NUMEROCIVICO",
                "CIVICO",
            ],
        )
    )

    postal_code_col = (
        first_existing_column(
            raw,
            [
                "CAP",
                "CAPEDIFICIO",
            ],
        )
    )

    province_col = (
        first_existing_column(
            raw,
            [
                "PROVINCIA",
                "DENOMINAZIONEPROVINCIA",
                "SIGLAPROVINCIA",
            ],
        )
    )

    state_col = (
        first_existing_column(
            raw,
            [
                "STATOEDIFICIO",
            ],
        )
    )

    missing_critical = []

    if school_code_col is None:
        missing_critical.append(
            "CODICESCUOLA"
        )

    if building_code_col is None:
        missing_critical.append(
            "CODICEEDIFICIO"
        )

    if missing_critical:
        raise RuntimeError(
            "Colonne critiche mancanti nel dataset edilizia: "
            + ", ".join(
                missing_critical
            )
            + "\nColonne disponibili:\n"
            + ", ".join(
                raw.columns
            )
        )

    canonical = pd.DataFrame()

    canonical[
        "school_code"
    ] = clean_string_series(
        raw[
            school_code_col
        ]
    )

    canonical[
        "building_code"
    ] = clean_string_series(
        raw[
            building_code_col
        ]
    )

    canonical[
        "building_municipality_name"
    ] = (
        clean_string_series(
            raw[
                municipality_col
            ]
        )
        if municipality_col
        else pd.Series(
            pd.NA,
            index=raw.index,
            dtype="string",
        )
    )

    if address_col:
        canonical[
            "official_building_address"
        ] = clean_string_series(
            raw[
                address_col
            ]
        )

    elif address_name_col:
        address_type = (
            clean_string_series(
                raw[
                    address_type_col
                ]
            )
            if address_type_col
            else pd.Series(
                pd.NA,
                index=raw.index,
                dtype="string",
            )
        )

        address_name = clean_string_series(
            raw[
                address_name_col
            ]
        )

        civic_number = (
            clean_string_series(
                raw[
                    civic_number_col
                ]
            )
            if civic_number_col
            else pd.Series(
                pd.NA,
                index=raw.index,
                dtype="string",
            )
        )

        def compose_address(index):
            parts = []

            for value in [
                address_type.loc[index],
                address_name.loc[index],
                civic_number.loc[index],
            ]:
                if pd.notna(value):
                    value = str(value).strip()

                    if value:
                        parts.append(
                            value
                        )

            return (
                " ".join(
                    parts
                )
                if parts
                else pd.NA
            )

        canonical[
            "official_building_address"
        ] = pd.Series(
            [
                compose_address(
                    index
                )
                for index in raw.index
            ],
            index=raw.index,
            dtype="string",
        )

    else:
        canonical[
            "official_building_address"
        ] = pd.Series(
            pd.NA,
            index=raw.index,
            dtype="string",
        )

    canonical[
        "building_postal_code"
    ] = (
        clean_string_series(
            raw[
                postal_code_col
            ]
        )
        if postal_code_col
        else pd.Series(
            pd.NA,
            index=raw.index,
            dtype="string",
        )
    )

    canonical[
        "building_province_name"
    ] = (
        clean_string_series(
            raw[
                province_col
            ]
        )
        if province_col
        else pd.Series(
            pd.NA,
            index=raw.index,
            dtype="string",
        )
    )

    canonical[
        "building_status_raw"
    ] = (
        clean_string_series(
            raw[
                state_col
            ]
        )
        if state_col
        else pd.Series(
            pd.NA,
            index=raw.index,
            dtype="string",
        )
    )

    canonical[
        "source_building_year"
    ] = building_year

    canonical[
        "source_filename"
    ] = source_path.name

    canonical[
        "source_row_number"
    ] = (
        raw.index
        + 2
    )

    # Remove rows without school/building identifiers.
    canonical = canonical[
        canonical[
            "school_code"
        ].notna()
        & canonical[
            "building_code"
        ].notna()
    ].copy()

    return canonical


def build_join(
    schools,
    buildings,
):
    state_schools = (
        schools[
            schools[
                "registry_type"
            ]
            == "state"
        ]
        .copy()
    )

    paritary_schools = (
        schools[
            schools[
                "registry_type"
            ]
            == "paritary"
        ]
        .copy()
    )

    # Exact school-code join only.
    links = state_schools.merge(
        buildings,
        on="school_code",
        how="left",
        indicator=True,
        suffixes=(
            "_school",
            "_building",
        ),
        validate="one_to_many",
    )

    matched = (
        links[
            links[
                "_merge"
            ]
            == "both"
        ]
        .drop(
            columns=[
                "_merge",
            ]
        )
        .copy()
    )

    unmatched_state = (
        links[
            links[
                "_merge"
            ]
            == "left_only"
        ]
        .drop(
            columns=[
                "_merge",
            ]
        )
        .copy()
    )

    # Keep one row per state-school record in unmatched.
    school_side_columns = [
        column
        for column in state_schools.columns
        if column in unmatched_state.columns
    ]

    unmatched_state = (
        unmatched_state[
            school_side_columns
        ]
        .drop_duplicates(
            subset=[
                "school_code",
            ]
        )
        .copy()
    )

    paritary_not_covered = (
        paritary_schools.copy()
    )

    return (
        matched,
        unmatched_state,
        paritary_not_covered,
    )


def build_physical_buildings(
    matched,
):
    if matched.empty:
        return pd.DataFrame()

    rows = []

    for (
        building_code,
        group,
    ) in matched.groupby(
        "building_code",
        dropna=False,
    ):
        school_codes = sorted(
            {
                str(value)
                for value in group[
                    "school_code"
                ].dropna()
            }
        )

        school_names = []

        if (
            "school_name"
            in group.columns
        ):
            school_names = sorted(
                {
                    str(value)
                    for value in group[
                        "school_name"
                    ].dropna()
                }
            )

        addresses = [
            value
            for value in group[
                "official_building_address"
            ].dropna().unique()
        ]

        municipalities = [
            value
            for value in group[
                "building_municipality_name"
            ].dropna().unique()
        ]

        postal_codes = [
            value
            for value in group[
                "building_postal_code"
            ].dropna().unique()
        ]

        row = {
            "building_code":
                building_code,

            "official_building_address":
                addresses[0]
                if addresses
                else None,

            "building_municipality_name":
                municipalities[0]
                if municipalities
                else None,

            "building_postal_code":
                postal_codes[0]
                if postal_codes
                else None,

            "linked_school_count":
                len(
                    school_codes
                ),

            "linked_school_codes":
                json.dumps(
                    school_codes,
                    ensure_ascii=False,
                ),

            "linked_school_names":
                json.dumps(
                    school_names,
                    ensure_ascii=False,
                ),

            "multiple_official_addresses":
                len(
                    addresses
                ) > 1,

            "multiple_municipality_names":
                len(
                    municipalities
                ) > 1,

            "source_building_year":
                (
                    group[
                        "source_building_year"
                    ]
                    .dropna()
                    .iloc[0]
                    if group[
                        "source_building_year"
                    ].notna().any()
                    else None
                ),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def old_run_summary(
    municipality_code,
):
    directory = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    old_links = (
        directory
        / "school_building_links_202627_from_202425.parquet"
    )

    old_unmatched = (
        directory
        / "school_building_unmatched_202627_from_202425.parquet"
    )

    old_physical = (
        directory
        / "physical_school_buildings_202425.parquet"
    )

    summary = {}

    if old_links.exists():
        dataframe = pd.read_parquet(
            old_links
        )

        summary[
            "old_202627_link_rows"
        ] = int(
            len(
                dataframe
            )
        )

    if old_unmatched.exists():
        dataframe = pd.read_parquet(
            old_unmatched
        )

        summary[
            "old_202627_unmatched_state_rows"
        ] = int(
            len(
                dataframe
            )
        )

    if old_physical.exists():
        dataframe = pd.read_parquet(
            old_physical
        )

        summary[
            "old_202627_physical_buildings"
        ] = int(
            len(
                dataframe
            )
        )

    return summary


def save_outputs(
    municipality_code,
    school_year,
    building_year,
    matched,
    unmatched_state,
    paritary_not_covered,
    physical_buildings,
    school_registry_path,
    building_path,
    building_encoding,
):
    directory = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"{school_year}_from_{building_year}"
    )

    matched_path = (
        directory
        / (
            "school_building_links_"
            f"{prefix}.parquet"
        )
    )

    unmatched_path = (
        directory
        / (
            "school_building_unmatched_"
            f"{prefix}.parquet"
        )
    )

    paritary_path = (
        directory
        / (
            "school_paritary_without_building_registry_"
            f"{prefix}.parquet"
        )
    )

    physical_path = (
        directory
        / (
            "physical_school_buildings_"
            f"{building_year}_from_schools_{school_year}.parquet"
        )
    )

    manifest_path = (
        directory
        / (
            "school_building_join_"
            f"{prefix}_manifest.json"
        )
    )

    matched.to_parquet(
        matched_path,
        index=False,
    )

    unmatched_state.to_parquet(
        unmatched_path,
        index=False,
    )

    paritary_not_covered.to_parquet(
        paritary_path,
        index=False,
    )

    physical_buildings.to_parquet(
        physical_path,
        index=False,
    )

    state_school_count = int(
        matched[
            "school_code"
        ].nunique()
        + unmatched_state[
            "school_code"
        ].nunique()
    )

    matched_school_count = int(
        matched[
            "school_code"
        ].nunique()
    )

    coverage = (
        100.0
        * matched_school_count
        / state_school_count
        if state_school_count
        else 0.0
    )

    manifest = {
        "municipality_code":
            municipality_code,

        "school_year":
            school_year,

        "building_year":
            building_year,

        "school_registry_path":
            str(
                school_registry_path
            ),

        "building_registry_path":
            str(
                building_path
            ),

        "building_registry_encoding":
            building_encoding,

        "state_school_count":
            state_school_count,

        "matched_state_school_count":
            matched_school_count,

        "unmatched_state_school_count":
            int(
                unmatched_state[
                    "school_code"
                ].nunique()
            ),

        "state_school_match_coverage_pct":
            coverage,

        "paritary_school_count":
            int(
                len(
                    paritary_not_covered
                )
            ),

        "school_building_link_rows":
            int(
                len(
                    matched
                )
            ),

        "physical_building_count":
            int(
                len(
                    physical_buildings
                )
            ),

        "buildings_shared_by_multiple_current_schools":
            int(
                (
                    physical_buildings[
                        "linked_school_count"
                    ]
                    > 1
                ).sum()
            )
            if not physical_buildings.empty
            else 0,

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    manifest.update(
        old_run_summary(
            municipality_code
        )
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "matched":
            matched_path,

        "unmatched":
            unmatched_path,

        "paritary":
            paritary_path,

        "physical":
            physical_path,

        "manifest":
            manifest_path,

        "manifest_data":
            manifest,
    }


def run_registry(
    municipality_code,
    municipality_name,
    school_year,
    refresh=False,
):
    print("\n====================================")
    print(" MIM SCHOOL REGISTRY")
    print("====================================")

    raw_year_dir = (
        RAW_MIM_DIR
        / "schools"
        / school_year
    )

    state_path, state_url = download_official_file(
        dataset="SCUANAGRAFESTAT",
        school_year=school_year,
        target_dir=raw_year_dir,
        refresh=refresh,
    )

    paritary_path, paritary_url = download_official_file(
        dataset="SCUANAGRAFEPAR",
        school_year=school_year,
        target_dir=raw_year_dir,
        refresh=refresh,
    )

    print(f"✓ Statali raw: {state_path.name}")
    print(f"✓ Paritarie raw: {paritary_path.name}")

    state_raw, state_encoding = read_csv_robust(
        state_path
    )
    paritary_raw, paritary_encoding = read_csv_robust(
        paritary_path
    )

    print("\n=== NATIONAL RAW ===")
    print(f"Statali: {len(state_raw)}")
    print(f"Paritarie: {len(paritary_raw)}")
    print(f"Encoding statali: {state_encoding}")
    print(f"Encoding paritarie: {paritary_encoding}")

    state_local = canonicalize_registry(
        dataframe=state_raw,
        registry_type="state",
        municipality_code=municipality_code,
        municipality_name=municipality_name,
        school_year=school_year,
        source_path=state_path,
        source_url=state_url,
    )

    paritary_local = canonicalize_registry(
        dataframe=paritary_raw,
        registry_type="paritary",
        municipality_code=municipality_code,
        municipality_name=municipality_name,
        school_year=school_year,
        source_path=paritary_path,
        source_url=paritary_url,
    )

    combined = pd.concat(
        [state_local, paritary_local],
        ignore_index=True,
        sort=False,
    )

    duplicate_codes = (
        combined["school_code"]
        .duplicated(keep=False)
    )

    output_dir = (
        PROCESSED_MIM_DIR
        / municipality_code
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        output_dir
        / f"schools_registry_{school_year}.parquet"
    )
    csv_path = (
        output_dir
        / f"schools_registry_{school_year}.csv"
    )
    manifest_path = (
        output_dir
        / f"schools_registry_{school_year}_manifest.json"
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
        "municipality_code": municipality_code,
        "municipality_name": municipality_name,
        "school_year": school_year,
        "catalog_data_as_of": CATALOG_DATA_AS_OF.get(
            school_year
        ),
        "state_raw_filename": state_path.name,
        "state_source_url": state_url,
        "paritary_raw_filename": paritary_path.name,
        "paritary_source_url": paritary_url,
        "state_national_rows": int(len(state_raw)),
        "paritary_national_rows": int(len(paritary_raw)),
        "state_local_rows": int(len(state_local)),
        "paritary_local_rows": int(len(paritary_local)),
        "combined_local_rows": int(len(combined)),
        "duplicate_school_code_rows": int(
            duplicate_codes.sum()
        ),
        "generated_at_utc": datetime.now(
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

    print("\n=== MUNICIPALITY ===")
    print(
        f"Comune: {municipality_name} "
        f"({municipality_code})"
    )
    print(f"Statali: {len(state_local)}")
    print(f"Paritarie: {len(paritary_local)}")
    print(f"Totale: {len(combined)}")
    print(
        "School-code duplicate rows: "
        f"{int(duplicate_codes.sum())}"
    )

    print("\n=== OUTPUT ===")
    print(f"✓ {parquet_path}")
    print(f"✓ {csv_path}")
    print(f"✓ {manifest_path}")

    return {
        "registry": parquet_path,
        "registry_csv": csv_path,
        "manifest": manifest_path,
        "manifest_data": manifest,
    }


def run_buildings(
    municipality_code,
    school_year,
    building_year,
    building_file=None,
):
    print("\n====================================")
    print(" MIM SCHOOL ↔ BUILDING JOIN")
    print("====================================")
    print(f"Municipality: {municipality_code}")
    print(f"School year: {school_year}")
    print(f"Building year: {building_year}")

    schools, school_registry_path = (
        load_school_registry(
            municipality_code,
            school_year,
        )
    )

    building_path = find_building_file(
        building_year,
        building_file,
    )

    building_raw, building_encoding = (
        read_csv_robust(
            building_path
        )
    )

    buildings = canonicalize_buildings(
        building_raw,
        building_path,
        building_year,
    )

    print("\n=== INPUT ===")
    print(
        f"School registry rows: {len(schools)}"
    )
    print(
        "  state: "
        f"{int((schools['registry_type'] == 'state').sum())}"
    )
    print(
        "  paritary: "
        f"{int((schools['registry_type'] == 'paritary').sum())}"
    )
    print(
        "Building registry national rows: "
        f"{len(buildings)}"
    )
    print(
        f"Building raw file: {building_path.name}"
    )

    (
        matched,
        unmatched_state,
        paritary_not_covered,
    ) = build_join(
        schools,
        buildings,
    )

    physical_buildings = (
        build_physical_buildings(
            matched
        )
    )

    outputs = save_outputs(
        municipality_code=municipality_code,
        school_year=school_year,
        building_year=building_year,
        matched=matched,
        unmatched_state=unmatched_state,
        paritary_not_covered=paritary_not_covered,
        physical_buildings=physical_buildings,
        school_registry_path=school_registry_path,
        building_path=building_path,
        building_encoding=building_encoding,
    )

    manifest = outputs["manifest_data"]

    print("\n=== JOIN RESULTS ===")
    print(
        "State schools: "
        f"{manifest['state_school_count']}"
    )
    print(
        "Exact school-code matched: "
        f"{manifest['matched_state_school_count']}"
    )
    print(
        "Unmatched state schools: "
        f"{manifest['unmatched_state_school_count']}"
    )
    print(
        "State coverage: "
        f"{manifest['state_school_match_coverage_pct']:.2f}%"
    )
    print(
        "Paritary schools outside state-building registry: "
        f"{manifest['paritary_school_count']}"
    )
    print(
        "School ↔ building relation rows: "
        f"{manifest['school_building_link_rows']}"
    )
    print(
        "Unique physical buildings: "
        f"{manifest['physical_building_count']}"
    )
    print(
        "Buildings shared by >1 current school: "
        f"{manifest['buildings_shared_by_multiple_current_schools']}"
    )

    old_unmatched = manifest.get(
        "old_202627_unmatched_state_rows"
    )
    old_physical = manifest.get(
        "old_202627_physical_buildings"
    )

    if (
        old_unmatched is not None
        or old_physical is not None
    ):
        print(
            "\n=== COMPARISON WITH OLD "
            "2026/27 → 2024/25 RUN ==="
        )

        if old_unmatched is not None:
            print(
                "Old unmatched state schools: "
                f"{old_unmatched}"
            )
            print(
                "New unmatched state schools: "
                f"{manifest['unmatched_state_school_count']}"
            )

        if old_physical is not None:
            print(
                "Old unique physical buildings: "
                f"{old_physical}"
            )
            print(
                "New unique physical buildings: "
                f"{manifest['physical_building_count']}"
            )

    print("\n=== OUTPUT ===")
    for key in (
        "matched",
        "unmatched",
        "paritary",
        "physical",
        "manifest",
    ):
        print(f"✓ {outputs[key]}")

    print("\nNOTA METODOLOGICA:")
    print(
        "Il join è esclusivamente exact school_code. "
        "Nessun fuzzy matching viene usato per creare "
        "relazioni amministrative scuola-edificio."
    )
    print(
        "Le paritarie sono mantenute separate perché "
        "l'Anagrafe edilizia usata è relativa alle "
        "scuole statali."
    )

    return outputs


def main():
    args = parse_args()

    if args.step in {"prepare", "registry"}:
        run_registry(
            municipality_code=args.municipality_code,
            municipality_name=args.municipality_name,
            school_year=args.school_year,
            refresh=args.refresh,
        )

    if args.step in {"prepare", "buildings"}:
        run_buildings(
            municipality_code=args.municipality_code,
            school_year=args.school_year,
            building_year=args.building_year,
            building_file=args.building_file,
        )


if __name__ == "__main__":
    main()
