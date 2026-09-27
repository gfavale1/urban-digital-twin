import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_MIM_DIR = ROOT / "data" / "raw" / "mim"
PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"

DEFAULT_SCHOOL_YEAR = "202425"
DEFAULT_BUILDING_YEAR = "202425"


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Join generalizzabile tra Anagrafe scuole MIM e "
            "Anagrafe dell'edilizia scolastica MIM."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--school-year",
        default=DEFAULT_SCHOOL_YEAR,
        help="Anno scolastico anagrafica scuole, default 202425.",
    )

    parser.add_argument(
        "--building-year",
        default=DEFAULT_BUILDING_YEAR,
        help="Anno scolastico anagrafica edilizia, default 202425.",
    )

    parser.add_argument(
        "--building-file",
        default=None,
        help=(
            "Path opzionale al CSV dell'edilizia. "
            "Se omesso, viene cercato automaticamente sotto "
            "data/raw/mim/buildings/<building-year>/."
        ),
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    for field_name in [
        "school_year",
        "building_year",
    ]:
        value = (
            str(
                getattr(
                    args,
                    field_name,
                )
            )
            .strip()
        )

        if not re.fullmatch(
            r"\d{6}",
            value,
        ):
            raise ValueError(
                f"{field_name} deve avere 6 cifre, es. 202425."
            )

        setattr(
            args,
            field_name,
            value,
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

def normalize_columns(
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


def first_existing_column(
    dataframe,
    candidates,
):
    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


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


# ============================================================
# LOAD SCHOOL REGISTRY
# ============================================================

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


# ============================================================
# CANONICALIZE BUILDING REGISTRY
# ============================================================

def canonicalize_buildings(
    raw,
    source_path,
    building_year,
):
    raw = normalize_columns(
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


# ============================================================
# JOIN
# ============================================================

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


# ============================================================
# PHYSICAL BUILDINGS
# ============================================================

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


# ============================================================
# COMPARISON WITH OLD RUN
# ============================================================

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


# ============================================================
# OUTPUT
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    print(
        "\n===================================="
    )
    print(
        " MIM SCHOOL ↔ BUILDING JOIN"
    )
    print(
        "===================================="
    )

    print(
        f"Municipality: {args.municipality_code}"
    )
    print(
        f"School year: {args.school_year}"
    )
    print(
        f"Building year: {args.building_year}"
    )

    (
        schools,
        school_registry_path,
    ) = load_school_registry(
        args.municipality_code,
        args.school_year,
    )

    building_path = find_building_file(
        args.building_year,
        args.building_file,
    )

    building_raw, building_encoding = (
        read_csv_robust(
            building_path
        )
    )

    buildings = canonicalize_buildings(
        building_raw,
        building_path,
        args.building_year,
    )

    print(
        "\n=== INPUT ==="
    )
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
        f"Building registry national rows: {len(buildings)}"
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
        municipality_code=args.municipality_code,
        school_year=args.school_year,
        building_year=args.building_year,
        matched=matched,
        unmatched_state=unmatched_state,
        paritary_not_covered=paritary_not_covered,
        physical_buildings=physical_buildings,
        school_registry_path=school_registry_path,
        building_path=building_path,
        building_encoding=building_encoding,
    )

    manifest = outputs[
        "manifest_data"
    ]

    print(
        "\n=== JOIN RESULTS ==="
    )
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
            "\n=== COMPARISON WITH OLD 2026/27 → 2024/25 RUN ==="
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

    print(
        "\n=== OUTPUT ==="
    )

    for key in [
        "matched",
        "unmatched",
        "paritary",
        "physical",
        "manifest",
    ]:
        print(
            f"✓ {outputs[key]}"
        )

    print(
        "\nNOTA METODOLOGICA:"
    )
    print(
        "Il join è esclusivamente exact school_code. "
        "Nessun fuzzy matching viene usato per creare "
        "relazioni amministrative scuola-edificio."
    )
    print(
        "Le paritarie sono mantenute separate perché "
        "l'Anagrafe edilizia usata è relativa alle scuole statali."
    )


if __name__ == "__main__":
    main()
