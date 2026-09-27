import argparse
import html
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]

RAW_MIM_DIR = ROOT / "data" / "raw" / "mim"
PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"

DEFAULT_SCHOOL_YEAR = "202627"
DEFAULT_BUILDING_YEAR = "202425"

CATALOG_URL = (
    "https://dati.istruzione.it/opendata/opendata/"
    "catalogo/elements1/leaf/"
    "?datasetId=DS0101EDIANAGRAFESTA2021"
)

USER_AGENT = "urban-digital-twin/1.0 (academic research)"

REQUIRED_COLUMNS = {
    "ANNOSCOLASTICO",
    "CODICESCUOLA",
    "CODICEEDIFICIO",
    "CODICECOMUNE",
    "DESCRIZIONECOMUNE",
    "SIGLAPROVINCIA",
    "TIPOLOGIAINDIRIZZO",
    "DENOMINAZIONEINDIRIZZO",
    "NUMEROCIVICO",
    "CAP",
}

MISSING_TEXT_VALUES = {
    "",
    "NAN",
    "NONE",
    "NULL",
    "N/A",
    "NA",
    "N.D.",
    "ND",
    "NON DISPONIBILE",
    "ASSENTE",
    "-",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ingestion MIM Anagrafe dell'edilizia scolastica "
            "e collegamento esatto CodiceScuola -> CodiceEdificio."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--school-year",
        default=DEFAULT_SCHOOL_YEAR,
        help="Anno scolastico anagrafica corrente, default 202627.",
    )

    parser.add_argument(
        "--building-year",
        default=DEFAULT_BUILDING_YEAR,
        help="Anno scolastico edilizia MIM, default 202425.",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Riscarica il CSV ufficiale anche se presente localmente.",
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
            "municipality-code deve avere esattamente 6 cifre."
        )

    for field_name in ["school_year", "building_year"]:
        value = str(
            getattr(args, field_name)
        ).strip()

        if (
            not value.isdigit()
            or len(value) != 6
        ):
            raise ValueError(
                f"{field_name} deve avere 6 cifre, es. 202425."
            )

        setattr(args, field_name, value)

    return args


def clean_text(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip()

    if value.upper() in MISSING_TEXT_VALUES:
        return None

    return value


def normalize_column_name(value):
    value = str(value).strip().upper()

    value = re.sub(
        r"[^A-Z0-9]+",
        "",
        value,
    )

    return value


def read_csv_robust(path):
    encodings = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
        "cp1252",
    ]

    separators = [";", ","]

    last_error = None

    for encoding in encodings:
        for separator in separators:
            try:
                dataframe = pd.read_csv(
                    path,
                    sep=separator,
                    dtype=str,
                    encoding=encoding,
                    low_memory=False,
                )

                if len(dataframe.columns) > 1:
                    return dataframe

            except Exception as exc:
                last_error = exc

    raise RuntimeError(
        f"Impossibile leggere il CSV {path}. "
        f"Ultimo errore: {last_error}"
    )


def registry_path(
    municipality_code,
    school_year,
):
    return (
        PROCESSED_MIM_DIR
        / municipality_code
        / f"schools_registry_{school_year}.parquet"
    )


def load_current_registry(
    municipality_code,
    school_year,
):
    path = registry_path(
        municipality_code,
        school_year,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Registro MIM corrente non trovato: {path}"
        )

    registry = pd.read_parquet(path).copy()

    if "school_code" not in registry.columns:
        raise RuntimeError(
            "school_code mancante nel registro MIM."
        )

    if "school_ownership" not in registry.columns:
        raise RuntimeError(
            "school_ownership mancante nel registro MIM."
        )

    if registry["school_code"].duplicated().any():
        raise RuntimeError(
            "school_code duplicati nel registro MIM corrente."
        )

    registry["school_code"] = (
        registry["school_code"]
        .apply(clean_text)
    )

    registry["school_ownership"] = (
        registry["school_ownership"]
        .apply(clean_text)
    )

    return registry


def raw_building_directory(
    building_year,
):
    directory = (
        RAW_MIM_DIR
        / "buildings"
        / building_year
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def find_existing_csv(
    building_year,
):
    directory = raw_building_directory(
        building_year
    )

    csv_files = sorted(
        directory.glob("*.csv")
    )

    if not csv_files:
        return None

    preferred = [
        path
        for path in csv_files
        if building_year in path.name
    ]

    if preferred:
        return preferred[-1]

    return csv_files[-1]


def extract_csv_urls(
    page_text,
):
    page_text = html.unescape(page_text)

    patterns = [
        r'href\s*=\s*"([^"]+\.csv(?:\?[^"]*)?)"',
        r"href\s*=\s*'([^']+\.csv(?:\?[^']*)?)'",
        r'https?://[^\s"\'<>]+\.csv(?:\?[^\s"\'<>]*)?',
    ]

    urls = []

    for index, pattern in enumerate(patterns):
        matches = re.findall(
            pattern,
            page_text,
            flags=re.IGNORECASE,
        )

        for value in matches:
            if index == 2:
                full_url = value
            else:
                full_url = urljoin(
                    CATALOG_URL,
                    value.strip(),
                )

            if full_url not in urls:
                urls.append(full_url)

    return urls


def discover_csv_url(
    building_year,
):
    print(
        "\nRicerca CSV ufficiale "
        "Anagrafe edilizia MIM..."
    )

    response = requests.get(
        CATALOG_URL,
        headers={
            "User-Agent": USER_AGENT,
        },
        timeout=60,
    )

    response.raise_for_status()

    csv_urls = extract_csv_urls(
        response.text
    )

    if not csv_urls:
        raise RuntimeError(
            "Il catalogo MIM non ha esposto link CSV "
            "direttamente nell'HTML. "
            "Scarica manualmente la distribuzione CSV "
            f"{building_year} dal catalogo MIM e inseriscila in "
            f"{raw_building_directory(building_year)}"
        )

    matching = [
        url
        for url in csv_urls
        if building_year in url
    ]

    if matching:
        return matching[-1]

    raise RuntimeError(
        "Sono stati trovati link CSV nel catalogo MIM, "
        f"ma nessuno contiene l'anno {building_year}. "
        "Scarica manualmente la distribuzione corretta "
        f"e inseriscila in {raw_building_directory(building_year)}"
    )


def download_csv(
    url,
    building_year,
):
    directory = raw_building_directory(
        building_year
    )

    filename = (
        url.split("?")[0]
        .rstrip("/")
        .split("/")[-1]
    )

    if not filename:
        filename = (
            f"EDIANAGRAFESTA_{building_year}.csv"
        )

    path = directory / filename

    print(f"Download: {url}")

    with requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
        },
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()

        with open(path, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)

    return path


def ensure_building_csv(
    building_year,
    refresh=False,
):
    existing = find_existing_csv(
        building_year
    )

    if existing is not None and not refresh:
        print(
            "\nUso CSV edilizia MIM locale:"
        )
        print(f"  {existing}")
        return existing

    url = discover_csv_url(
        building_year
    )

    path = download_csv(
        url,
        building_year,
    )

    print(
        "\n✓ CSV edilizia MIM salvato:"
    )
    print(f"  {path}")

    return path


def load_building_dataset(
    path,
    building_year,
):
    dataframe = read_csv_robust(path)

    dataframe.columns = [
        normalize_column_name(column)
        for column in dataframe.columns
    ]

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Colonne obbligatorie mancanti nel dataset edilizia: "
            + ", ".join(sorted(missing_columns))
        )

    # STATOEDIFICIO è documentato nel tracciato MIM,
    # ma non è presente in alcune distribuzioni CSV recenti.
    # Lo manteniamo quindi come campo opzionale e NON
    # inventiamo il valore "ATTIVO".
    if "STATOEDIFICIO" not in dataframe.columns:
        dataframe["STATOEDIFICIO"] = pd.NA

    for column in dataframe.columns:
        dataframe[column] = (
            dataframe[column]
            .apply(clean_text)
        )

    if "ANNOSCOLASTICO" in dataframe.columns:
        normalized_year = (
            dataframe["ANNOSCOLASTICO"]
            .fillna("")
            .astype(str)
            .str.replace(
                r"\D",
                "",
                regex=True,
            )
        )

        if (
            normalized_year
            .eq(building_year)
            .any()
        ):
            dataframe = dataframe.loc[
                normalized_year.eq(building_year)
            ].copy()

    print(
        "\n=== MIM BUILDING DATASET ==="
    )

    print(
        f"Record nazionali: {len(dataframe)}"
    )

    return dataframe


def compose_building_address(row):
    parts = [
        clean_text(
            row.get("TIPOLOGIAINDIRIZZO")
        ),
        clean_text(
            row.get("DENOMINAZIONEINDIRIZZO")
        ),
        clean_text(
            row.get("NUMEROCIVICO")
        ),
    ]

    value = " ".join(
        part
        for part in parts
        if part
    )

    return value or None


def build_links(
    registry,
    buildings,
    school_year,
    building_year,
):
    state_mask = (
        registry["school_ownership"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("state")
    )

    state_registry = (
        registry.loc[
            state_mask
        ]
        .copy()
    )

    paritary_registry = (
        registry.loc[
            ~state_mask
        ]
        .copy()
    )

    building_subset = (
        buildings[
            [
                "ANNOSCOLASTICO",
                "CODICESCUOLA",
                "CODICEEDIFICIO",
                "CODICECOMUNE",
                "DESCRIZIONECOMUNE",
                "SIGLAPROVINCIA",
                "TIPOLOGIAINDIRIZZO",
                "DENOMINAZIONEINDIRIZZO",
                "NUMEROCIVICO",
                "CAP",
                "STATOEDIFICIO",
            ]
        ]
        .copy()
    )

    building_subset[
        "official_building_address"
    ] = building_subset.apply(
        compose_building_address,
        axis=1,
    )

    building_subset[
        "CODICESCUOLA"
    ] = (
        building_subset[
            "CODICESCUOLA"
        ]
        .apply(clean_text)
    )

    building_subset[
        "CODICEEDIFICIO"
    ] = (
        building_subset[
            "CODICEEDIFICIO"
        ]
        .apply(clean_text)
    )

    matched_building_rows = (
        building_subset[
            building_subset[
                "CODICESCUOLA"
            ]
            .isin(
                set(
                    state_registry[
                        "school_code"
                    ]
                )
            )
        ]
        .copy()
    )

    links = (
        state_registry.merge(
            matched_building_rows,
            left_on="school_code",
            right_on="CODICESCUOLA",
            how="left",
            validate="one_to_many",
        )
    )

    links[
        "current_school_year"
    ] = school_year

    links[
        "building_reference_year"
    ] = building_year

    links[
        "building_match_status"
    ] = (
        links[
            "CODICEEDIFICIO"
        ]
        .notna()
        .map(
            {
                True: "exact_school_code",
                False: "unmatched",
            }
        )
    )

    unmatched = (
        links[
            links[
                "building_match_status"
            ]
            == "unmatched"
        ]
        .copy()
    )

    matched_links = (
        links[
            links[
                "building_match_status"
            ]
            == "exact_school_code"
        ]
        .copy()
    )

    physical_buildings = (
        matched_links[
            [
                "CODICEEDIFICIO",
                "CODICECOMUNE",
                "DESCRIZIONECOMUNE",
                "SIGLAPROVINCIA",
                "TIPOLOGIAINDIRIZZO",
                "DENOMINAZIONEINDIRIZZO",
                "NUMEROCIVICO",
                "CAP",
                "STATOEDIFICIO",
                "official_building_address",
            ]
        ]
        .drop_duplicates(
            subset=["CODICEEDIFICIO"]
        )
        .rename(
            columns={
                "CODICEEDIFICIO":
                    "building_code",
                "CODICECOMUNE":
                    "building_cadastral_municipality_code",
                "DESCRIZIONECOMUNE":
                    "building_municipality_name",
                "SIGLAPROVINCIA":
                    "building_province_code",
                "TIPOLOGIAINDIRIZZO":
                    "address_type",
                "DENOMINAZIONEINDIRIZZO":
                    "address_name",
                "NUMEROCIVICO":
                    "house_number",
                "CAP":
                    "postal_code",
                "STATOEDIFICIO":
                    "building_status",
            }
        )
        .copy()
    )

    school_counts = (
        matched_links.groupby(
            "CODICEEDIFICIO"
        )[
            "school_code"
        ]
        .nunique()
    )

    physical_buildings[
        "linked_current_school_count"
    ] = (
        physical_buildings[
            "building_code"
        ]
        .map(school_counts)
        .fillna(0)
        .astype(int)
    )

    return (
        state_registry,
        paritary_registry,
        matched_links,
        unmatched,
        physical_buildings,
    )


def save_outputs(
    municipality_code,
    school_year,
    building_year,
    matched_links,
    unmatched,
    physical_buildings,
):
    directory = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    links_path = (
        directory
        / (
            "school_building_links_"
            f"{school_year}_from_"
            f"{building_year}.parquet"
        )
    )

    unmatched_path = (
        directory
        / (
            "school_building_unmatched_"
            f"{school_year}_from_"
            f"{building_year}.parquet"
        )
    )

    buildings_path = (
        directory
        / (
            "physical_school_buildings_"
            f"{building_year}.parquet"
        )
    )

    matched_links.to_parquet(
        links_path,
        index=False,
    )

    unmatched.to_parquet(
        unmatched_path,
        index=False,
    )

    physical_buildings.to_parquet(
        buildings_path,
        index=False,
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(f"✓ {links_path}")
    print(f"✓ {unmatched_path}")
    print(f"✓ {buildings_path}")


def print_summary(
    registry,
    state_registry,
    paritary_registry,
    matched_links,
    unmatched,
    physical_buildings,
):
    total_state = len(
        state_registry
    )

    matched_school_codes = int(
        matched_links[
            "school_code"
        ]
        .nunique()
    )

    unmatched_school_codes = int(
        unmatched[
            "school_code"
        ]
        .nunique()
    )

    unique_buildings = int(
        physical_buildings[
            "building_code"
        ]
        .nunique()
    )

    multi_building_school_codes = int(
        (
            matched_links.groupby(
                "school_code"
            )[
                "CODICEEDIFICIO"
            ]
            .nunique()
            > 1
        )
        .sum()
    )

    shared_buildings = int(
        (
            physical_buildings[
                "linked_current_school_count"
            ]
            > 1
        )
        .sum()
    )

    missing_address = int(
        physical_buildings[
            "official_building_address"
        ]
        .isna()
        .sum()
    )

    non_matera_buildings = (
        physical_buildings[
            physical_buildings[
                "building_municipality_name"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
            .ne("MATERA")
        ]
    )

    print(
        "\n===================================="
    )
    print(
        " MIM SCHOOL BUILDING LINK COMPLETATO"
    )
    print(
        "===================================="
    )

    print(
        f"Record MIM correnti: {len(registry)}"
    )
    print(
        f"Scuole statali correnti: {total_state}"
    )
    print(
        "Scuole paritarie correnti "
        "(fuori Anagrafe edilizia statale): "
        f"{len(paritary_registry)}"
    )

    print(
        "\nExact match CodiceScuola:"
    )
    print(
        f"  trovate:     {matched_school_codes}"
    )
    print(
        f"  non trovate: {unmatched_school_codes}"
    )

    coverage = (
        100.0
        * matched_school_codes
        / total_state
        if total_state > 0
        else 0.0
    )

    print(
        f"  coverage:    {coverage:.2f}%"
    )

    print(
        "\nLayer fisico:"
    )
    print(
        f"  edifici unici: {unique_buildings}"
    )
    print(
        "  scuole associate a >1 edificio: "
        f"{multi_building_school_codes}"
    )
    print(
        "  edifici condivisi da >1 scuola corrente: "
        f"{shared_buildings}"
    )
    print(
        "  edifici senza indirizzo ufficiale: "
        f"{missing_address}"
    )
    print(
        "  edifici con Comune != MATERA: "
        f"{len(non_matera_buildings)}"
    )

    print(
        "\n=== SCUOLE STATALI NON TROVATE ==="
    )

    if unmatched.empty:
        print("Nessuna.")
    else:
        display_columns = [
            column
            for column in [
                "school_code",
                "school_name",
                "school_type",
                "address",
            ]
            if column in unmatched.columns
        ]

        print(
            unmatched[
                display_columns
            ]
            .drop_duplicates()
            .to_string(
                index=False
            )
        )

    print(
        "\n=== EDIFICI FUORI MATERA ==="
    )

    if non_matera_buildings.empty:
        print("Nessuno.")
    else:
        print(
            non_matera_buildings[
                [
                    "building_code",
                    "building_municipality_name",
                    "official_building_address",
                    "linked_current_school_count",
                ]
            ]
            .to_string(
                index=False
            )
        )


def main():
    args = parse_args()

    print(
        "\n===================================="
    )
    print(
        " MIM SCHOOL BUILDINGS"
    )
    print(
        "===================================="
    )

    registry = load_current_registry(
        args.municipality_code,
        args.school_year,
    )

    csv_path = ensure_building_csv(
        args.building_year,
        args.refresh,
    )

    buildings = load_building_dataset(
        csv_path,
        args.building_year,
    )

    (
        state_registry,
        paritary_registry,
        matched_links,
        unmatched,
        physical_buildings,
    ) = build_links(
        registry=registry,
        buildings=buildings,
        school_year=args.school_year,
        building_year=args.building_year,
    )

    save_outputs(
        municipality_code=args.municipality_code,
        school_year=args.school_year,
        building_year=args.building_year,
        matched_links=matched_links,
        unmatched=unmatched,
        physical_buildings=physical_buildings,
    )

    print_summary(
        registry=registry,
        state_registry=state_registry,
        paritary_registry=paritary_registry,
        matched_links=matched_links,
        unmatched=unmatched,
        physical_buildings=physical_buildings,
    )


if __name__ == "__main__":
    main()
