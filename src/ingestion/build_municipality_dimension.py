from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_ISTAT_DIR = (
    ROOT
    / "data"
    / "raw"
    / "istat"
)

PROCESSED_ISTAT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "istat"
)

FEATURES_ISTAT_DIR = (
    ROOT
    / "data"
    / "features"
    / "istat"
)

EXPECTED_REGION_CODES = {
    f"{value:02d}"
    for value in range(1, 21)
}

REQUIRED_COLUMNS = {
    "PROCOM",
    "COMUNE",
    "CODPRO",
    "CODREG",
}

OPTIONAL_COLUMNS = {
    "PROVINCIA",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Costruisce una dimensione amministrativa nazionale "
            "dei comuni direttamente dai file regionali del "
            "Censimento permanente ISTAT. La dimensione è "
            "versionata per anno censuario e permette alla "
            "pipeline di risolvere un comune mai processato "
            "senza dipendere preventivamente da PostGIS."
        )
    )

    parser.add_argument(
        "--census-year",
        default="2023",
        help="Anno del dataset censuario ISTAT. Default: 2023.",
    )

    return parser.parse_args()


def utc_now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


def census_directory(
    census_year: str,
) -> Path:
    year = str(census_year).strip()

    return (
        RAW_ISTAT_DIR
        / f"censimento_{year}"
        / f"Dati_regionali_{year}"
    )


def discover_region_files(
    census_year: str,
) -> list[Path]:
    directory = census_directory(
        census_year
    )

    if not directory.exists():
        raise FileNotFoundError(
            "Directory dei dati censuari regionali "
            f"non trovata: {directory}"
        )

    files = sorted(
        directory.glob(
            f"R??_*_{census_year}_sezioni.xlsx"
        )
    )

    if not files:
        raise FileNotFoundError(
            "Nessun file censuario regionale trovato in "
            f"{directory}"
        )

    return files


def normalize_integer_code(
    series: pd.Series,
    width: int,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.isna().any():
        count = int(
            numeric.isna().sum()
        )

        raise RuntimeError(
            "Codice ISTAT non numerico o mancante: "
            f"{count} righe."
        )

    return (
        numeric
        .astype("int64")
        .astype(str)
        .str.zfill(width)
    )


def load_region_dimension(
    path: Path,
    census_year: str,
) -> pd.DataFrame:
    header = pd.read_excel(
        path,
        nrows=0,
        keep_default_na=False,
    )

    available = set(
        header.columns
    )

    missing = (
        REQUIRED_COLUMNS
        - available
    )

    if missing:
        raise RuntimeError(
            f"Colonne mancanti in {path.name}: "
            + ", ".join(
                sorted(missing)
            )
        )

    usecols = list(
        REQUIRED_COLUMNS
        | (
            OPTIONAL_COLUMNS
            & available
        )
    )

    df = pd.read_excel(
        path,
        usecols=usecols,
        keep_default_na=False,
    )

    df = df[
        list(
            REQUIRED_COLUMNS
            | (
                OPTIONAL_COLUMNS
                & set(df.columns)
            )
        )
    ].copy()

    df["istat_code"] = (
        normalize_integer_code(
            df["PROCOM"],
            6,
        )
    )

    df["province_code"] = (
        normalize_integer_code(
            df["CODPRO"],
            3,
        )
    )

    df["region_code"] = (
        normalize_integer_code(
            df["CODREG"],
            2,
        )
    )

    df["name"] = (
        df["COMUNE"]
        .astype("string")
        .str.strip()
    )

    if "PROVINCIA" in df.columns:
        df["province_name"] = (
            df["PROVINCIA"]
            .astype("string")
            .str.strip()
        )
    else:
        df["province_name"] = pd.NA

    # Nei file regionali alcune righe di sezione possono non
    # ripetere la denominazione amministrativa. Non trasformiamo
    # il missing in un valore inventato: usiamo esclusivamente le
    # righe che contengono il nome e verifichiamo che ogni PROCOM
    # presente nel file sia comunque rappresentato.
    all_municipality_codes = set(
        df["istat_code"]
        .dropna()
        .astype(str)
    )

    named_rows = df.loc[
        df["name"].notna()
        & df["name"].ne("")
    ].copy()

    represented_codes = set(
        named_rows["istat_code"]
        .dropna()
        .astype(str)
    )

    missing_metadata_codes = (
        all_municipality_codes
        - represented_codes
    )

    if missing_metadata_codes:
        raise RuntimeError(
            "Comuni senza alcuna denominazione amministrativa "
            f"in {path.name}: "
            + ", ".join(
                sorted(missing_metadata_codes)[:30]
            )
        )

    # Each regional census file contains one row per census
    # section. We collapse the valid administrative metadata to
    # exactly one record per municipality.
    result = (
        named_rows[
            [
                "istat_code",
                "name",
                "province_code",
                "province_name",
                "region_code",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicated_codes = (
        result["istat_code"]
        .duplicated(
            keep=False
        )
    )

    if duplicated_codes.any():
        sample = (
            result.loc[
                duplicated_codes
            ]
            .sort_values(
                "istat_code"
            )
            .head(20)
        )

        raise RuntimeError(
            "Metadati amministrativi non univoci nel file "
            f"{path.name}:\n"
            + sample.to_string(
                index=False
            )
        )

    actual_regions = set(
        result["region_code"]
        .dropna()
        .astype(str)
    )

    if len(actual_regions) != 1:
        raise RuntimeError(
            "Il file regionale contiene CODREG non univoci: "
            f"{path.name} -> {sorted(actual_regions)}"
        )

    result["reference_year"] = (
        str(census_year)
    )

    result["source_system"] = "ISTAT"

    result["source_dataset"] = (
        "Censimento permanente della popolazione "
        f"{census_year} - dati regionali per sezione"
    )

    result["source_file"] = (
        path.name
    )

    result["source_record_id"] = (
        result["istat_code"]
    )

    return result


def validate_national_dimension(
    dimension: pd.DataFrame,
):
    if dimension.empty:
        raise RuntimeError(
            "Dimensione amministrativa vuota."
        )

    if (
        dimension["istat_code"]
        .duplicated()
        .any()
    ):
        duplicates = (
            dimension.loc[
                dimension[
                    "istat_code"
                ].duplicated(
                    keep=False
                )
            ]
            .sort_values(
                "istat_code"
            )
        )

        raise RuntimeError(
            "Codici comunali duplicati nella dimensione "
            "nazionale:\n"
            + duplicates.head(
                30
            ).to_string(
                index=False
            )
        )

    actual_regions = set(
        dimension[
            "region_code"
        ]
        .astype(str)
        .str.zfill(2)
    )

    missing_regions = (
        EXPECTED_REGION_CODES
        - actual_regions
    )

    unexpected_regions = (
        actual_regions
        - EXPECTED_REGION_CODES
    )

    if missing_regions:
        raise RuntimeError(
            "La dimensione non è nazionale. "
            "Regioni ISTAT mancanti: "
            + ", ".join(
                sorted(
                    missing_regions
                )
            )
        )

    if unexpected_regions:
        raise RuntimeError(
            "Codici regione inattesi: "
            + ", ".join(
                sorted(
                    unexpected_regions
                )
            )
        )

    code_prefix = (
        dimension["istat_code"]
        .astype(str)
        .str[:3]
    )

    if not (
        code_prefix
        == dimension[
            "province_code"
        ].astype(str).str.zfill(3)
    ).all():
        bad = dimension.loc[
            code_prefix
            != dimension[
                "province_code"
            ].astype(str).str.zfill(3)
        ]

        raise RuntimeError(
            "Incoerenza tra codice comune e codice provincia:\n"
            + bad.head(
                20
            ).to_string(
                index=False
            )
        )


def main():
    args = parse_args()

    census_year = str(
        args.census_year
    ).strip()

    files = discover_region_files(
        census_year
    )

    print(
        "\n=============================================="
    )
    print(
        " ISTAT NATIONAL MUNICIPALITY DIMENSION"
    )
    print(
        "=============================================="
    )
    print(
        f"Census year: {census_year}"
    )
    print(
        f"File regionali trovati: {len(files)}"
    )

    frames = []

    for index, path in enumerate(
        files,
        start=1,
    ):
        frame = load_region_dimension(
            path,
            census_year,
        )

        frames.append(
            frame
        )

        region_code = (
            frame["region_code"]
            .iloc[0]
        )

        print(
            f"[{index:02d}/{len(files):02d}] "
            f"R{region_code}: "
            f"{len(frame)} comuni "
            f"({path.name})"
        )

    dimension = pd.concat(
        frames,
        ignore_index=True,
    )

    dimension = (
        dimension
        .sort_values(
            [
                "region_code",
                "province_code",
                "istat_code",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    validate_national_dimension(
        dimension
    )

    PROCESSED_ISTAT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURES_ISTAT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        PROCESSED_ISTAT_DIR
        / (
            "municipality_dimension_"
            f"{census_year}.parquet"
        )
    )

    csv_path = (
        FEATURES_ISTAT_DIR
        / (
            "municipality_dimension_"
            f"{census_year}.csv"
        )
    )

    manifest_path = (
        FEATURES_ISTAT_DIR
        / (
            "municipality_dimension_"
            f"{census_year}_manifest.json"
        )
    )

    dimension.to_parquet(
        parquet_path,
        index=False,
    )

    dimension.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "generated_at_utc":
            utc_now_iso(),

        "reference_year":
            census_year,

        "source_system":
            "ISTAT",

        "source_dataset":
            (
                "Censimento permanente della popolazione "
                f"{census_year} - dati regionali per sezione"
            ),

        "regional_source_files": [
            str(path)
            for path in files
        ],

        "regional_source_file_count":
            int(len(files)),

        "municipality_count":
            int(len(dimension)),

        "region_count":
            int(
                dimension[
                    "region_code"
                ].nunique()
            ),

        "province_count":
            int(
                dimension[
                    "province_code"
                ].nunique()
            ),

        "columns":
            list(
                dimension.columns
            ),

        "methodology": {
            "temporal_alignment":
                (
                    "Administrative municipality metadata are derived "
                    "from the same census-year regional ISTAT files used "
                    "by the demographic ingestion, avoiding a current-vs-"
                    "historical administrative-code mismatch."
                ),

            "uniqueness":
                (
                    "Section-level rows are collapsed to exactly one "
                    "administrative record per ISTAT municipality code."
                ),

            "zero_touch":
                (
                    "MunicipalityContext resolves municipality name, "
                    "province code and region code from this versioned "
                    "dimension before any municipality-specific PostGIS "
                    "row needs to exist."
                ),
        },
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
        "\n=== VALIDATION ==="
    )

    print(
        "Regioni: "
        f"{dimension['region_code'].nunique()}"
    )

    print(
        "Province: "
        f"{dimension['province_code'].nunique()}"
    )

    print(
        "Comuni: "
        f"{len(dimension)}"
    )

    print(
        "Codici comunali duplicati: "
        f"{int(dimension['istat_code'].duplicated().sum())}"
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
        "\n✓ Dimensione amministrativa nazionale pronta."
    )


if __name__ == "__main__":
    main()
