import argparse
import html
import os
import re
import shutil
import unicodedata
from pathlib import Path
from urllib.parse import (
    urljoin,
    urlparse,
)

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    text,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_MIM_DIR = (
    ROOT
    / "data"
    / "raw"
    / "mim"
    / "schools"
)

PROCESSED_MIM_DIR = (
    ROOT
    / "data"
    / "processed"
    / "mim"
)


# ============================================================
# MIM ENDPOINTS
# ============================================================

MIM_CATALOG_URL = (
    "https://dati.istruzione.it/"
    "opendata/opendata/catalogo/"
    "elements1/?area=Scuole"
)

MIM_DOWNLOAD_BASE_URL = (
    "https://dati.istruzione.it/"
    "opendata/opendata/catalogo/"
    "elements1/"
)


# ============================================================
# DATASET TYPES
# ============================================================

STANDARD_FLOWS = {
    "state": "SCUANAGRAFESTAT",
    "paritary": "SCUANAGRAFEPAR",
}

AUTONOMOUS_FLOWS = {
    "state": "SCUANAAUTSTAT",
    "paritary": "SCUANAAUTPAR",
}


# ============================================================
# KNOWN RELEASE DATES
# ============================================================
#
# Serve come fallback se il catalogo HTML MIM
# non fosse temporaneamente interrogabile.
#
# yyyy/yy -> data snapshot YYYYMMDD
#
# ============================================================

KNOWN_RELEASE_DATES = {
    "202627": "20260901",
    "202526": "20250901",
    "202425": "20250831",
    "202324": "20240831",
    "202223": "20230831",
}


# ============================================================
# REGION MAPPING
# ============================================================

REGION_NAMES = {
    "01": "Piemonte",
    "02": "Valle d'Aosta",
    "03": "Lombardia",
    "04": "Trentino-Alto Adige",
    "05": "Veneto",
    "06": "Friuli-Venezia Giulia",
    "07": "Liguria",
    "08": "Emilia-Romagna",
    "09": "Toscana",
    "10": "Umbria",
    "11": "Marche",
    "12": "Lazio",
    "13": "Abruzzo",
    "14": "Molise",
    "15": "Campania",
    "16": "Puglia",
    "17": "Basilicata",
    "18": "Calabria",
    "19": "Sicilia",
    "20": "Sardegna",
}

COMMON_REQUIRED_COLUMNS = {
    "ANNOSCOLASTICO",
    "REGIONE",
    "PROVINCIA",
    "CODICESCUOLA",
    "DENOMINAZIONESCUOLA",
    "INDIRIZZOSCUOLA",
    "CAPSCUOLA",
    "CODICECOMUNESCUOLA",
    "DESCRIZIONECOMUNE",
    "DESCRIZIONETIPOLOGIAGRADOISTRUZIONESCUOLA",
}


STATE_REQUIRED_COLUMNS = {
    "CODICEISTITUTORIFERIMENTO",
    "DENOMINAZIONEISTITUTORIFERIMENTO",
}

# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ingestion dell'anagrafica ufficiale "
            "delle scuole MIM."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help=(
            "Codice ISTAT a 6 cifre. "
            "Esempio: 077014 per Matera."
        ),
    )

    parser.add_argument(
        "--school-year",
        default="202627",
        help=(
            "Anno scolastico nel formato YYYYyy. "
            "Default: 202627."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Scarica nuovamente i CSV MIM "
            "anche se già presenti in raw."
        ),
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
            "municipality-code deve avere "
            "esattamente 6 cifre."
        )

    args.school_year = (
        str(args.school_year)
        .strip()
    )

    if (
        not args.school_year.isdigit()
        or len(args.school_year) != 6
    ):
        raise ValueError(
            "school-year deve avere formato "
            "YYYYyy, es. 202627."
        )

    return args


# ============================================================
# DATABASE
# ============================================================

def get_database_engine():
    load_dotenv(
        ROOT / ".env"
    )

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL non definito "
            "nel file .env."
        )

    return create_engine(
        database_url
    )


def load_municipality(
    engine,
    municipality_code,
):
    query = text("""
        SELECT
            id,
            istat_code,
            name,
            province_code,
            region_code

        FROM municipality

        WHERE istat_code =
            :istat_code;
    """)

    with engine.connect() as connection:

        row = connection.execute(
            query,
            {
                "istat_code":
                    municipality_code
            },
        ).mappings().first()

    if row is None:
        raise RuntimeError(
            "\nComune non presente "
            "nel database.\n"
            f"Codice ISTAT: "
            f"{municipality_code}\n\n"
            "Eseguire prima "
            "l'ingestion ISTAT."
        )

    region_code = (
        str(row["region_code"])
        .zfill(2)
    )

    if region_code not in REGION_NAMES:
        raise RuntimeError(
            "Codice regione non riconosciuto: "
            f"{region_code}"
        )

    return {
        "id": row["id"],
        "istat_code":
            row["istat_code"],
        "name":
            row["name"],
        "province_code":
            row["province_code"],
        "region_code":
            region_code,
        "region_name":
            REGION_NAMES[region_code],
    }


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalizzazione robusta per confrontare
    nomi di comuni e regioni provenienti
    da sorgenti differenti.
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    value = str(value).strip()

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(
            character
        )
    )

    value = value.upper()

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


# ============================================================
# FLOW SELECTION
# ============================================================

def get_flows_for_region(
    region_code,
):
    """
    I dataset standard MIM escludono
    Aosta, Trento e Bolzano.

    Valle d'Aosta e Trentino-Alto Adige
    usano quindi i dataset AUT.
    """

    if region_code in {
        "02",
        "04",
    }:
        return AUTONOMOUS_FLOWS

    return STANDARD_FLOWS


# ============================================================
# CATALOG DISCOVERY
# ============================================================

def fetch_catalog_links():
    print(
        "\nRicerca dataset nel catalogo MIM..."
    )

    response = requests.get(
        MIM_CATALOG_URL,
        timeout=60,
        headers={
            "User-Agent": (
                "urban-digital-twin/"
                "1.0 academic-research"
            )
        },
    )

    response.raise_for_status()

    page = html.unescape(
        response.text
    )

    links = re.findall(
        r'href=["\']([^"\']+\.csv)["\']',
        page,
        flags=re.IGNORECASE,
    )

    urls = []

    for link in links:

        url = urljoin(
            MIM_CATALOG_URL,
            link,
        )

        if url not in urls:
            urls.append(url)

    print(
        f"CSV trovati nel catalogo: "
        f"{len(urls)}"
    )

    return urls


def find_dataset_url(
    flow_name,
    school_year,
    catalog_urls,
):
    candidates = []

    expected_prefix = (
        f"{flow_name}"
        f"{school_year}"
    )

    for url in catalog_urls:

        filename = Path(
            urlparse(url).path
        ).name

        if (
            filename.upper()
            .startswith(
                expected_prefix.upper()
            )
            and filename.lower()
            .endswith(".csv")
        ):
            candidates.append(
                url
            )

    if candidates:
        # Se esistessero più release
        # scegliamo quella con data più recente.
        candidates = sorted(
            candidates
        )

        return candidates[-1]

    return None


def build_fallback_url(
    flow_name,
    school_year,
):
    release_date = (
        KNOWN_RELEASE_DATES.get(
            school_year
        )
    )

    if release_date is None:
        return None

    filename = (
        f"{flow_name}"
        f"{school_year}"
        f"{release_date}"
        ".csv"
    )

    return urljoin(
        MIM_DOWNLOAD_BASE_URL,
        filename,
    )


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(
    url,
    destination,
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        destination.with_suffix(
            ".csv.part"
        )
    )

    print(
        "\nDownload:"
    )

    print(
        f"  {url}"
    )

    try:

        with requests.get(
            url,
            stream=True,
            timeout=(20, 180),
            headers={
                "User-Agent": (
                    "urban-digital-twin/"
                    "1.0 academic-research"
                )
            },
        ) as response:

            response.raise_for_status()

            content_type = (
                response.headers.get(
                    "content-type",
                    ""
                )
                .lower()
            )

            if (
                "text/html"
                in content_type
            ):
                raise RuntimeError(
                    "Il server MIM ha restituito "
                    "HTML invece del CSV."
                )

            total = int(
                response.headers.get(
                    "content-length",
                    0,
                )
            )

            downloaded = 0

            with open(
                temporary,
                "wb",
            ) as output:

                for chunk in (
                    response.iter_content(
                        chunk_size=(
                            1024 * 1024
                        )
                    )
                ):

                    if not chunk:
                        continue

                    output.write(
                        chunk
                    )

                    downloaded += (
                        len(chunk)
                    )

                    if total:

                        percentage = (
                            downloaded
                            / total
                            * 100
                        )

                        print(
                            f"\r  "
                            f"{percentage:6.2f}%",
                            end="",
                            flush=True,
                        )

        print()

        temporary.replace(
            destination
        )

    except Exception:

        if temporary.exists():
            temporary.unlink()

        raise


# ============================================================
# ENSURE RAW DATASETS
# ============================================================

def ensure_raw_datasets(
    school_year,
    region_code,
    refresh=False,
):
    flows = get_flows_for_region(
        region_code
    )

    raw_directory = (
        RAW_MIM_DIR
        / school_year
    )

    raw_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # TRY CATALOG DISCOVERY
    # --------------------------------------------------------

    try:
        catalog_urls = (
            fetch_catalog_links()
        )

    except Exception as exc:

        print(
            "\nATTENZIONE: catalogo MIM "
            "non interrogabile automaticamente."
        )

        print(
            f"Motivo: {exc}"
        )

        print(
            "Uso dei pattern URL noti "
            "come fallback."
        )

        catalog_urls = []

    paths = {}

    for ownership, flow_name in (
        flows.items()
    ):

        url = find_dataset_url(
            flow_name,
            school_year,
            catalog_urls,
        )

        if url is None:

            url = build_fallback_url(
                flow_name,
                school_year,
            )

        if url is None:

            raise RuntimeError(
                "Impossibile determinare "
                "l'URL MIM per:\n"
                f"flow={flow_name}\n"
                f"school_year={school_year}"
            )

        filename = Path(
            urlparse(url).path
        ).name

        destination = (
            raw_directory
            / filename
        )

        if (
            destination.exists()
            and not refresh
        ):
            print(
                "\n✓ Dataset MIM "
                "già disponibile:"
            )

            print(
                f"  {destination}"
            )

        else:

            download_file(
                url,
                destination,
            )

        paths[ownership] = (
            destination
        )

    return paths


# ============================================================
# CSV LOADING
# ============================================================

def read_mim_csv(path):
    """
    Prova alcune codifiche comuni.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin1",
        "cp1252",
    ]

    last_error = None

    for encoding in encodings:

        try:

            df = pd.read_csv(
                path,
                encoding=encoding,
                dtype=str,
                low_memory=False,
            )

            print(
                f"✓ Letto {path.name} "
                f"con encoding={encoding}"
            )

            return df

        except UnicodeDecodeError as exc:

            last_error = exc

    raise RuntimeError(
        "Impossibile determinare "
        f"l'encoding di {path}"
    ) from last_error


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_columns(df):
    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .upper()
        for column
        in df.columns
    ]

    return df


# ============================================================
# REQUIRED COLUMNS
# ============================================================

def validate_schema(
    df,
    filename,
    ownership,
):
    """
    Valida il tracciato MIM rispettando
    le differenze ufficiali tra scuole
    statali e paritarie.

    Le scuole statali contengono anche
    i campi relativi all'istituto di
    riferimento.

    Le scuole paritarie non li contengono.
    """

    required_columns = set(
        COMMON_REQUIRED_COLUMNS
    )

    if ownership == "state":
        required_columns.update(
            STATE_REQUIRED_COLUMNS
        )

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "\nSchema MIM inatteso.\n"
            f"File: {filename}\n"
            f"Tipologia: {ownership}\n"
            "Colonne mancanti:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    print(
        f"✓ Schema {ownership} valido"
    )


# ============================================================
# MUNICIPALITY FILTER
# ============================================================

def filter_municipality(
    df,
    municipality,
):
    df = df.copy()

    target_region = (
        normalize_text(
            municipality[
                "region_name"
            ]
        )
    )

    target_municipality = (
        normalize_text(
            municipality[
                "name"
            ]
        )
    )

    df[
        "_region_normalized"
    ] = (
        df["REGIONE"]
        .apply(normalize_text)
    )

    df[
        "_municipality_normalized"
    ] = (
        df[
            "DESCRIZIONECOMUNE"
        ]
        .apply(normalize_text)
    )

    filtered = df[
        (
            df[
                "_region_normalized"
            ]
            == target_region
        )
        &
        (
            df[
                "_municipality_normalized"
            ]
            == target_municipality
        )
    ].copy()

    filtered = filtered.drop(
        columns=[
            "_region_normalized",
            "_municipality_normalized",
        ]
    )

    return filtered


# ============================================================
# CANONICAL REGISTRY
# ============================================================

def build_canonical_registry(
    datasets,
    municipality,
    school_year,
):
    frames = []

    for ownership, df in (
        datasets.items()
    ):

        filtered = (
            filter_municipality(
                df,
                municipality,
            )
        )

        filtered[
            "school_ownership"
        ] = ownership

        # ----------------------------------------------------
        # DIFFERENZA UFFICIALE STATE / PARITARY
        # ----------------------------------------------------

        if ownership == "paritary":

            filtered[
                "CODICEISTITUTORIFERIMENTO"
            ] = pd.NA

            filtered[
                "DENOMINAZIONEISTITUTORIFERIMENTO"
            ] = pd.NA

        frames.append(
            filtered
        )

        print(
            f"{ownership}: "
            f"{len(filtered)} scuole"
        )

    if not frames:
        raise RuntimeError(
            "Nessun dataset MIM disponibile."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined.empty:
        raise RuntimeError(
            "\nNessuna scuola trovata "
            f"per {municipality['name']} "
            f"nell'anno {school_year}."
        )

    canonical = pd.DataFrame(
        {
            "school_year":
                combined[
                    "ANNOSCOLASTICO"
                ],

            "school_code":
                combined[
                    "CODICESCUOLA"
                ],

            "reference_institute_code":
                combined[
                    "CODICEISTITUTORIFERIMENTO"
                ],

            "reference_institute_name":
                combined[
                    "DENOMINAZIONEISTITUTORIFERIMENTO"
                ],

            "school_name":
                combined[
                    "DENOMINAZIONESCUOLA"
                ],

            "school_type":
                combined[
                    "DESCRIZIONETIPOLOGIAGRADOISTRUZIONESCUOLA"
                ],

            "school_ownership":
                combined[
                    "school_ownership"
                ],

            "address":
                combined[
                    "INDIRIZZOSCUOLA"
                ],

            "postal_code":
                combined[
                    "CAPSCUOLA"
                ],

            "mim_cadastral_code":
                combined[
                    "CODICECOMUNESCUOLA"
                ],

            "municipality_name":
                combined[
                    "DESCRIZIONECOMUNE"
                ],

            "province_name":
                combined[
                    "PROVINCIA"
                ],

            "region_name":
                combined[
                    "REGIONE"
                ],
        }
    )

    canonical[
        "municipality_istat_code"
    ] = municipality[
        "istat_code"
    ]

    canonical[
        "source_system"
    ] = "MIM"

    canonical[
        "source_record_id"
    ] = canonical[
        "school_code"
    ]

    # Le coordinate verranno aggiunte
    # nella pipeline di geolocalizzazione.
    canonical[
        "geocoding_status"
    ] = "pending"

    canonical[
        "latitude"
    ] = pd.NA

    canonical[
        "longitude"
    ] = pd.NA

    return canonical


# ============================================================
# DATA QUALITY
# ============================================================

def validate_canonical_registry(
    schools,
):
    print(
        "\n=== DATA QUALITY ==="
    )

    if schools.empty:
        raise RuntimeError(
            "Dataset scuole vuoto."
        )

    missing_codes = (
        schools[
            "school_code"
        ]
        .isna()
        .sum()
    )

    print(
        "Scuole senza codice MIM: "
        f"{missing_codes}"
    )

    if missing_codes:
        raise RuntimeError(
            "Esistono scuole "
            "senza CodiceScuola."
        )

    duplicates = (
        schools[
            "school_code"
        ]
        .duplicated(
            keep=False
        )
    )

    duplicate_count = (
        duplicates.sum()
    )

    print(
        "Codici scuola duplicati: "
        f"{duplicate_count}"
    )

    if duplicate_count:

        print(
            "\nDuplicati:"
        )

        print(
            schools.loc[
                duplicates,
                [
                    "school_code",
                    "school_name",
                    "school_ownership",
                ],
            ]
            .sort_values(
                "school_code"
            )
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "CodiceScuola non univoco "
            "nel canonical dataset."
        )

    missing_address = (
        schools[
            "address"
        ]
        .isna()
        .sum()
    )

    print(
        "Scuole senza indirizzo: "
        f"{missing_address}"
    )

    missing_postal_code = (
        schools[
            "postal_code"
        ]
        .isna()
        .sum()
    )

    print(
        "Scuole senza CAP: "
        f"{missing_postal_code}"
    )

    print(
        "✓ controlli anagrafica "
        "MIM superati"
    )


# ============================================================
# SAVE SILVER
# ============================================================

def save_silver(
    schools,
    municipality_code,
    school_year,
):
    directory = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        directory
        / (
            f"schools_registry_"
            f"{school_year}.parquet"
        )
    )

    schools.to_parquet(
        path,
        index=False,
    )

    print(
        "\n=== SILVER DATASET ==="
    )

    print(
        f"✓ {path}"
    )

    return path


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    schools,
    municipality,
    school_year,
):
    print(
        "\n===================================="
    )

    print(
        " MIM SCHOOL REGISTRY COMPLETATA"
    )

    print(
        "===================================="
    )

    print(
        f"Comune: "
        f"{municipality['name']}"
    )

    print(
        "Codice ISTAT: "
        f"{municipality['istat_code']}"
    )

    print(
        "Anno scolastico: "
        f"{school_year}"
    )

    print(
        f"Scuole totali: "
        f"{len(schools)}"
    )

    print(
        "\nPer titolarità:"
    )

    print(
        schools[
            "school_ownership"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nPer tipologia:"
    )

    print(
        schools[
            "school_type"
        ]
        .value_counts(
            dropna=False
        )
        .head(30)
        .to_string()
    )

    print(
        "\nPrime scuole:"
    )

    print(
        schools[
            [
                "school_code",
                "school_name",
                "school_type",
                "school_ownership",
                "address",
                "postal_code",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    print(
        "\n===================================="
    )

    print(
        " MIM SCHOOL REGISTRY INGESTION"
    )

    print(
        "===================================="
    )

    engine = (
        get_database_engine()
    )

    # --------------------------------------------------------
    # MUNICIPALITY
    # --------------------------------------------------------

    municipality = (
        load_municipality(
            engine,
            args.municipality_code,
        )
    )

    print(
        "\n=== COMUNE ==="
    )

    print(
        f"Nome: "
        f"{municipality['name']}"
    )

    print(
        "Codice ISTAT: "
        f"{municipality['istat_code']}"
    )

    print(
        "Regione: "
        f"{municipality['region_name']}"
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    raw_paths = (
        ensure_raw_datasets(
            school_year=(
                args.school_year
            ),

            region_code=(
                municipality[
                    "region_code"
                ]
            ),

            refresh=args.refresh,
        )
    )

    # --------------------------------------------------------
    # LOAD + VALIDATE RAW
    # --------------------------------------------------------

    datasets = {}

    for ownership, path in (
        raw_paths.items()
    ):

        df = read_mim_csv(
            path
        )

        df = normalize_columns(
            df
        )

        validate_schema(
            df,
            path.name,
            ownership,
        )

        datasets[
            ownership
        ] = df

        print(
            f"{ownership}: "
            f"{len(df)} record nazionali"
        )

    # --------------------------------------------------------
    # CANONICAL
    # --------------------------------------------------------

    print(
        "\n=== FILTRO COMUNALE ==="
    )

    schools = (
        build_canonical_registry(
            datasets,
            municipality,
            args.school_year,
        )
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    validate_canonical_registry(
        schools
    )

    # --------------------------------------------------------
    # SILVER
    # --------------------------------------------------------

    save_silver(
        schools,
        args.municipality_code,
        args.school_year,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_summary(
        schools,
        municipality,
        args.school_year,
    )


if __name__ == "__main__":
    main()