import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from requests.exceptions import SSLError


ROOT = Path(__file__).resolve().parents[2]

RAW_SALUTE_DIR = ROOT / "data" / "raw" / "salute"
FEATURES_SALUTE_DIR = ROOT / "data" / "features" / "salute"


SOURCES = {
    "farmacie": {
        "page_url": "https://www.dati.salute.gov.it/it/dataset/farmacie/",
    },
    "strutture_ospedaliere_2023": {
        "page_url": (
            "https://www.dati.salute.gov.it/it/dataset/"
            "posti-letto-stabilimento-ospedaliero-e-disciplina-2023/"
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download Bronze e audit preliminare dei dataset ufficiali "
            "del Ministero della Salute. In caso di problemi TLS di requests, "
            "usa automaticamente curl come fallback."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Riscarica i file anche se esistono già localmente.",
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
            "--municipality-code deve contenere esattamente 6 cifre."
        )

    return args


def utc_now():
    return datetime.now(timezone.utc)


def get_session():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "urban-digital-twin-thesis/1.0 "
                "(academic research; public open-data ingestion)"
            )
        }
    )
    return session


def curl_fetch(url):
    """
    Fallback per siti che non negoziano correttamente TLS con
    l'OpenSSL usato dall'ambiente Conda/Python.
    """
    command = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--http1.1",
        "--connect-timeout",
        "30",
        "--max-time",
        "180",
        "-A",
        "urban-digital-twin-thesis/1.0 (academic research)",
        url,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            "Fallback curl fallito per:\n"
            f"{url}\n\n{stderr}"
        )

    return result.stdout


def fetch_bytes(session, url, timeout=120):
    """
    Prima prova requests. Se il server del Ministero genera
    SSL handshake failure, usa curl di sistema.
    """
    try:
        response = session.get(
            url,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.content, "requests"

    except SSLError as exc:
        print(
            "  ! TLS handshake fallito con requests; "
            "provo automaticamente con curl..."
        )
        return curl_fetch(url), "curl"


def fetch_text(session, url, timeout=60):
    content, transport = fetch_bytes(
        session,
        url,
        timeout=timeout,
    )

    # Le pagine HTML del portale sono normalmente UTF-8.
    return (
        content.decode(
            "utf-8",
            errors="replace",
        ),
        transport,
    )


def discover_csv_url(session, page_url):
    html, transport = fetch_text(
        session,
        page_url,
        timeout=60,
    )

    hrefs = re.findall(
        r'href\s*=\s*["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )

    candidates = []

    for href in hrefs:
        absolute = urljoin(
            page_url,
            href,
        )

        path = (
            urlparse(absolute)
            .path
            .lower()
        )

        if path.endswith(".csv"):
            candidates.append(
                absolute
            )

    candidates = list(
        dict.fromkeys(
            candidates
        )
    )

    if not candidates:
        raise RuntimeError(
            "Nessun link CSV trovato nella pagina ufficiale:\n"
            f"{page_url}"
        )

    return (
        candidates[0],
        transport,
    )


def download_snapshot(
    session,
    source_name,
    csv_url,
    refresh,
):
    directory = (
        RAW_SALUTE_DIR
        / source_name
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_name = Path(
        urlparse(csv_url).path
    ).name

    if not original_name:
        original_name = (
            f"{source_name}.csv"
        )

    target_path = (
        directory
        / original_name
    )

    if (
        target_path.exists()
        and not refresh
    ):
        return (
            target_path,
            False,
            "cache",
        )

    content, transport = fetch_bytes(
        session,
        csv_url,
        timeout=180,
    )

    target_path.write_bytes(
        content
    )

    return (
        target_path,
        True,
        transport,
    )


def detect_encoding(path):
    raw = path.read_bytes()[:100_000]

    if raw.startswith(
        b"\xef\xbb\xbf"
    ):
        return "utf-8-sig"

    for encoding in [
        "utf-8",
        "cp1252",
        "latin-1",
    ]:
        try:
            raw.decode(
                encoding
            )
            return encoding
        except UnicodeDecodeError:
            continue

    return "latin-1"


def detect_separator(path, encoding):
    sample = path.read_text(
        encoding=encoding,
        errors="replace",
    )[:50_000]

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=[
                ";",
                ",",
                "\t",
                "|",
            ],
        )
        return dialect.delimiter

    except csv.Error:
        counts = {
            delimiter:
                sample.count(delimiter)
            for delimiter in [
                ";",
                ",",
                "\t",
                "|",
            ]
        }

        return max(
            counts,
            key=counts.get,
        )


def read_csv_as_strings(path):
    encoding = detect_encoding(path)
    separator = detect_separator(
        path,
        encoding,
    )

    dataframe = pd.read_csv(
        path,
        sep=separator,
        dtype="string",
        encoding=encoding,
        keep_default_na=False,
        na_values=[],
        low_memory=False,
    )

    dataframe.columns = [
        str(column).strip()
        for column
        in dataframe.columns
    ]

    return (
        dataframe,
        encoding,
        separator,
    )


def normalized_series(series):
    return (
        series
        .astype("string")
        .str.strip()
    )


def normalize_column_name(value):
    value = str(value).strip().lower()
    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    ).strip("_")
    return value


def find_municipality_code_column(
    dataframe,
    municipality_code,
):
    normalized_to_real = {
        normalize_column_name(column):
            column
        for column
        in dataframe.columns
    }

    preferred = [
        "codice_istat_comune",
        "cod_istat_comune",
        "cod_comune",
        "codice_comune",
        "codice_comune_istat",
        "comune_istat",
    ]

    for candidate in preferred:
        real = normalized_to_real.get(
            candidate
        )

        if real is None:
            continue

        series = normalized_series(
            dataframe[real]
        )

        if (
            series
            == municipality_code
        ).any():
            return real

    # Fallback prudente: cerca il codice target nelle colonne,
    # ma evita colonne palesemente testuali molto lunghe.
    for column in dataframe.columns:
        series = normalized_series(
            dataframe[column]
        )

        if (
            series
            == municipality_code
        ).any():
            return column

    return None


def coordinate_coverage(dataframe):
    normalized_to_real = {
        normalize_column_name(column):
            column
        for column
        in dataframe.columns
    }

    lat_col = (
        normalized_to_real.get(
            "latitudine"
        )
        or normalized_to_real.get(
            "latitude"
        )
        or normalized_to_real.get(
            "lat"
        )
    )

    lon_col = (
        normalized_to_real.get(
            "longitudine"
        )
        or normalized_to_real.get(
            "longitude"
        )
        or normalized_to_real.get(
            "lon"
        )
        or normalized_to_real.get(
            "lng"
        )
    )

    if (
        lat_col is None
        or lon_col is None
    ):
        return {
            "latitude_column":
                lat_col,

            "longitude_column":
                lon_col,

            "usable_coordinates":
                0,

            "coordinate_coverage_pct":
                0.0,
        }

    lat = pd.to_numeric(
        dataframe[lat_col]
        .astype("string")
        .str.replace(
            ",",
            ".",
            regex=False,
        ),
        errors="coerce",
    )

    lon = pd.to_numeric(
        dataframe[lon_col]
        .astype("string")
        .str.replace(
            ",",
            ".",
            regex=False,
        ),
        errors="coerce",
    )

    usable = (
        lat.between(
            -90,
            90,
        )
        & lon.between(
            -180,
            180,
        )
    )

    count = int(
        usable.sum()
    )

    total = len(dataframe)

    coverage = (
        100.0
        * count
        / total
        if total
        else 0.0
    )

    return {
        "latitude_column":
            lat_col,

        "longitude_column":
            lon_col,

        "usable_coordinates":
            count,

        "coordinate_coverage_pct":
            coverage,
    }


def summarize_temporal_fields(
    dataframe,
):
    temporal = {}

    for column in dataframe.columns:
        name = normalize_column_name(
            column
        )

        if (
            "anno" not in name
            and "data" not in name
            and "validit" not in name
        ):
            continue

        values = (
            dataframe[column]
            .astype("string")
            .str.strip()
        )

        values = values[
            values.ne("")
        ]

        if values.empty:
            continue

        temporal[
            str(column)
        ] = (
            values
            .drop_duplicates()
            .head(20)
            .tolist()
        )

    return temporal


def audit_source(
    source_name,
    path,
    dataframe,
    municipality_code,
    csv_url,
    page_url,
    downloaded,
    page_transport,
    download_transport,
    encoding,
    separator,
):
    code_column = (
        find_municipality_code_column(
            dataframe,
            municipality_code,
        )
    )

    if code_column:
        municipal = dataframe.loc[
            normalized_series(
                dataframe[code_column]
            )
            == municipality_code
        ].copy()
    else:
        municipal = (
            dataframe.iloc[0:0]
            .copy()
        )

    coord_all = coordinate_coverage(
        dataframe
    )

    coord_municipal = (
        coordinate_coverage(
            municipal
        )
        if not municipal.empty
        else {
            "latitude_column":
                coord_all[
                    "latitude_column"
                ],

            "longitude_column":
                coord_all[
                    "longitude_column"
                ],

            "usable_coordinates":
                0,

            "coordinate_coverage_pct":
                0.0,
        }
    )

    return {
        "source_name":
            source_name,

        "official_page_url":
            page_url,

        "resolved_csv_url":
            csv_url,

        "raw_path":
            str(path),

        "downloaded_now":
            downloaded,

        "page_transport":
            page_transport,

        "download_transport":
            download_transport,

        "encoding":
            encoding,

        "separator":
            repr(separator),

        "national_rows":
            int(len(dataframe)),

        "column_count":
            int(
                len(
                    dataframe.columns
                )
            ),

        "columns":
            list(
                dataframe.columns
            ),

        "municipality_code":
            municipality_code,

        "municipality_code_column":
            code_column,

        "municipality_rows":
            int(len(municipal)),

        "national_coordinate_audit":
            coord_all,

        "municipality_coordinate_audit":
            coord_municipal,

        "temporal_fields_sample":
            summarize_temporal_fields(
                dataframe
            ),

        "municipality_preview":
            municipal.head(10)
            .to_dict(
                orient="records"
            ),
    }


def print_audit(audit):
    print(
        "\n------------------------------------"
    )
    print(
        audit["source_name"].upper()
    )
    print(
        "------------------------------------"
    )

    print(
        f"CSV: {audit['resolved_csv_url']}"
    )

    print(
        f"Raw: {audit['raw_path']}"
    )

    print(
        "Trasporto pagina: "
        f"{audit['page_transport']}"
    )

    print(
        "Trasporto download: "
        f"{audit['download_transport']}"
    )

    print(
        "National rows: "
        f"{audit['national_rows']}"
    )

    print(
        "Municipality code column: "
        f"{audit['municipality_code_column']}"
    )

    print(
        "Municipality rows: "
        f"{audit['municipality_rows']}"
    )

    coordinate_audit = (
        audit[
            "municipality_coordinate_audit"
        ]
    )

    print(
        "Coordinates in municipality: "
        f"{coordinate_audit['usable_coordinates']}"
        "/"
        f"{audit['municipality_rows']} "
        "("
        f"{coordinate_audit['coordinate_coverage_pct']:.2f}%"
        ")"
    )

    print(
        "\nColumns:"
    )

    print(
        audit["columns"]
    )

    if audit[
        "temporal_fields_sample"
    ]:
        print(
            "\nTemporal fields sample:"
        )

        for (
            column,
            values,
        ) in (
            audit[
                "temporal_fields_sample"
            ].items()
        ):
            print(
                f"  {column}: {values}"
            )

    print(
        "\nMunicipality preview:"
    )

    preview = audit[
        "municipality_preview"
    ]

    if preview:
        preview_df = pd.DataFrame(
            preview
        )

        print(
            preview_df.to_string(
                index=False
            )
        )
    else:
        print(
            "  Nessun record trovato per il comune."
        )


def main():
    args = parse_args()
    session = get_session()
    audits = []

    print(
        "\n===================================="
    )
    print(
        " MINISTERO SALUTE - SOURCE AUDIT V2"
    )
    print(
        "===================================="
    )
    print(
        f"Municipality code: {args.municipality_code}"
    )

    for (
        source_name,
        config,
    ) in SOURCES.items():

        page_url = (
            config["page_url"]
        )

        print(
            "\nScoperta link ufficiale: "
            f"{source_name}..."
        )

        (
            csv_url,
            page_transport,
        ) = discover_csv_url(
            session,
            page_url,
        )

        print(
            f"  CSV individuato: {csv_url}"
        )

        (
            path,
            downloaded,
            download_transport,
        ) = download_snapshot(
            session,
            source_name,
            csv_url,
            args.refresh,
        )

        (
            dataframe,
            encoding,
            separator,
        ) = read_csv_as_strings(
            path
        )

        audit = audit_source(
            source_name=source_name,
            path=path,
            dataframe=dataframe,
            municipality_code=(
                args.municipality_code
            ),
            csv_url=csv_url,
            page_url=page_url,
            downloaded=downloaded,
            page_transport=(
                page_transport
            ),
            download_transport=(
                download_transport
            ),
            encoding=encoding,
            separator=separator,
        )

        audits.append(
            audit
        )

        print_audit(
            audit
        )

    output_dir = (
        FEATURES_SALUTE_DIR
        / args.municipality_code
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            "ministero_salute_source_audit_"
            f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}.json"
        )
    )

    output_path.write_text(
        json.dumps(
            {
                "generated_at_utc":
                    utc_now().isoformat(),

                "municipality_code":
                    args.municipality_code,

                "sources":
                    audits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n===================================="
    )
    print(
        " AUDIT COMPLETATO"
    )
    print(
        "===================================="
    )
    print(
        f"✓ {output_path}"
    )


if __name__ == "__main__":
    main()
