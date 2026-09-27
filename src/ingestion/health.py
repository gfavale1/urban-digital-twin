import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[2]

RAW_SALUTE_DIR = ROOT / "data" / "raw" / "salute"
PROCESSED_SALUTE_DIR = ROOT / "data" / "processed" / "salute"
FEATURES_SALUTE_DIR = ROOT / "data" / "features" / "salute"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Costruisce i Silver del Ministero della Salute: "
            "farmacie attive a una data di riferimento e "
            "stabilimenti ospedalieri aggregati per struttura/subcodice."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--reference-date",
        default="2025-06-30",
        help=(
            "Data di riferimento per lo snapshot delle farmacie "
            "(YYYY-MM-DD). Default: 2025-06-30."
        ),
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

    try:
        args.reference_date = pd.Timestamp(
            args.reference_date
        ).normalize()
    except Exception as exc:
        raise ValueError(
            "--reference-date deve essere YYYY-MM-DD."
        ) from exc

    return args


def normalize_text(value):
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


def parse_decimal(series):
    return pd.to_numeric(
        series.astype("string")
        .str.strip()
        .str.replace(
            ",",
            ".",
            regex=False,
        ),
        errors="coerce",
    )


def parse_integer(series):
    return pd.to_numeric(
        series.astype("string")
        .str.strip()
        .str.replace(
            ".",
            "",
            regex=False,
        )
        .str.replace(
            ",",
            ".",
            regex=False,
        ),
        errors="coerce",
    ).round().astype("Int64")


def parse_date(series):
    cleaned = (
        series.astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "-": pd.NA,
                "NULL": pd.NA,
                "None": pd.NA,
            }
        )
    )

    return pd.to_datetime(
        cleaned,
        format="%d/%m/%Y",
        errors="coerce",
    )


def latest_file(directory, pattern="*.csv"):
    files = sorted(
        directory.glob(pattern),
        key=lambda path: (
            path.stat().st_mtime,
            path.name,
        ),
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            f"Nessun file {pattern} trovato in {directory}"
        )

    return files[0]


def read_csv_auto(path):
    last_error = None

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ]:
        for separator in [
            ",",
            ";",
            "\t",
            "|",
        ]:
            try:
                df = pd.read_csv(
                    path,
                    sep=separator,
                    dtype="string",
                    encoding=encoding,
                    keep_default_na=False,
                    low_memory=False,
                )

                if len(df.columns) > 2:
                    df.columns = [
                        str(column).strip()
                        for column in df.columns
                    ]

                    return df

            except Exception as exc:
                last_error = exc

    raise RuntimeError(
        f"Impossibile leggere {path}: {last_error}"
    )


def build_pharmacy_history(raw):
    required = {
        "cod_farmacia",
        "indirizzo",
        "descrizione_farmacia",
        "cap",
        "cod_comune",
        "comune",
        "data_inizio_validita",
        "data_fine_validita",
        "descrizione_tipologia",
        "latitudine",
        "longitudine",
        "localizzazione",
    }

    missing = required - set(raw.columns)

    if missing:
        raise RuntimeError(
            "Colonne farmacia mancanti: "
            + ", ".join(sorted(missing))
        )

    df = raw.copy()

    for column in [
        "cod_farmacia",
        "cod_farmacia_asl",
        "cod_comune",
        "cod_provincia",
        "cod_regione",
        "codice_tipologia",
    ]:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    df["valid_from"] = parse_date(
        df["data_inizio_validita"]
    )

    df["valid_to"] = parse_date(
        df["data_fine_validita"]
    )

    df["latitude"] = parse_decimal(
        df["latitudine"]
    )

    df["longitude"] = parse_decimal(
        df["longitudine"]
    )

    df["coordinate_valid"] = (
        (
            df["latitude"].between(
                -90,
                90,
            )
            & df["longitude"].between(
                -180,
                180,
            )
        )
        .fillna(False)
        .astype(bool)
    )

    df["source_system"] = (
        "Ministero della Salute"
    )

    df["source_dataset"] = (
        "Farmacie"
    )

    df["source_record_id"] = (
        df["cod_farmacia"]
    )

    return df


def snapshot_pharmacies(
    history,
    municipality_code,
    reference_date,
):
    df = history.copy()

    active = (
        df["valid_from"].notna()
        & (
            df["valid_from"]
            <= reference_date
        )
        & (
            df["valid_to"].isna()
            | (
                df["valid_to"]
                >= reference_date
            )
        )
    )

    df = df.loc[
        active
        & (
            df["cod_comune"]
            == municipality_code
        )
    ].copy()

    # Se il registro contiene intervalli sovrapposti per lo stesso codice,
    # preserviamo l'informazione ma scegliamo la versione con valid_from
    # più recente come rappresentazione dello snapshot.
    overlap_counts = (
        df.groupby(
            "cod_farmacia"
        )
        .size()
        .rename(
            "active_version_count"
        )
    )

    df = df.merge(
        overlap_counts,
        left_on="cod_farmacia",
        right_index=True,
        how="left",
    )

    df = (
        df.sort_values(
            [
                "cod_farmacia",
                "valid_from",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "cod_farmacia",
            ],
            keep="first",
        )
        .copy()
    )

    df["reference_date"] = (
        reference_date
    )

    df["service_site_id"] = (
        "SALUTE:PHARMACY:"
        + df["cod_farmacia"]
        .astype(str)
    )

    df["category"] = "health"
    df["subcategory"] = "pharmacy"

    df["name"] = (
        df["descrizione_farmacia"]
        .map(normalize_text)
    )

    df["address"] = (
        df["indirizzo"]
        .map(normalize_text)
    )

    df["municipality_code"] = (
        df["cod_comune"]
    )

    df["municipality_name"] = (
        df["comune"]
        .map(normalize_text)
    )

    df["postal_code"] = (
        df["cap"]
        .map(normalize_text)
    )

    df["coordinate_resolution"] = (
        df["coordinate_valid"]
        .map(
            {
                True: "official_source_coordinate",
                False: "missing",
            }
        )
    )

    df["confidence"] = (
        df["coordinate_valid"]
        .map(
            {
                True: "source_provided",
                False: "unresolved",
            }
        )
    )

    df["needs_geolocation"] = (
        ~df["coordinate_valid"]
    ).astype(bool)

    df["usable_for_accessibility"] = (
        df["coordinate_valid"]
    ).astype(bool)

    keep = [
        "service_site_id",
        "category",
        "subcategory",
        "source_system",
        "source_dataset",
        "source_record_id",
        "cod_farmacia",
        "cod_farmacia_asl",
        "name",
        "address",
        "postal_code",
        "municipality_code",
        "municipality_name",
        "frazione",
        "descrizione_tipologia",
        "codice_tipologia",
        "p_iva",
        "valid_from",
        "valid_to",
        "reference_date",
        "latitude",
        "longitude",
        "coordinate_resolution",
        "confidence",
        "needs_geolocation",
        "usable_for_accessibility",
        "active_version_count",
        "localizzazione",
    ]

    keep = [
        column
        for column in keep
        if column in df.columns
    ]

    return df[keep].copy()


def build_hospital_discipline_history(
    raw,
    hospital_year,
):
    required = {
        "Anno",
        "Codice struttura",
        "Subcodice",
        "Denominazione Struttura/Stabilimento",
        "Indirizzo",
        "Codice Comune",
        "Comune",
        "Codice disciplina",
        "Descrizione disciplina",
        "Totale posti letto",
    }

    missing = required - set(raw.columns)

    if missing:
        raise RuntimeError(
            "Colonne ospedali mancanti: "
            + ", ".join(sorted(missing))
        )

    df = raw.copy()

    df = df.loc[
        df["Anno"]
        .astype("string")
        .str.strip()
        == str(hospital_year)
    ].copy()

    for column in [
        "Codice struttura",
        "Subcodice",
        "Codice Comune",
        "Codice Regione",
        "Codice Azienda",
        "Codice tipo struttura",
        "Codice disciplina",
    ]:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    numeric_columns = [
        "N° Reparti",
        "Posti letto degenza ordinaria",
        "Posti letto degenza a pagamento",
        "Posti letto Day Hospital",
        "Posti letto Day Surgery",
        "Totale posti letto",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = parse_integer(
                df[column]
            )

    df["source_system"] = (
        "Ministero della Salute"
    )

    df["source_dataset"] = (
        "Posti letto per stabilimento "
        "ospedaliero e disciplina"
    )

    return df


def discipline_json(group):
    records = []

    for _, row in (
        group.sort_values(
            "Codice disciplina"
        )
        .iterrows()
    ):
        records.append(
            {
                "discipline_code":
                    normalize_text(
                        row.get(
                            "Codice disciplina"
                        )
                    ),

                "discipline_name":
                    normalize_text(
                        row.get(
                            "Descrizione disciplina"
                        )
                    ),

                "discipline_type":
                    normalize_text(
                        row.get(
                            "Tipo di Disciplina"
                        )
                    ),

                "departments":
                    (
                        int(
                            row["N° Reparti"]
                        )
                        if pd.notna(
                            row.get(
                                "N° Reparti"
                            )
                        )
                        else None
                    ),

                "ordinary_beds":
                    (
                        int(
                            row[
                                "Posti letto degenza ordinaria"
                            ]
                        )
                        if pd.notna(
                            row.get(
                                "Posti letto degenza ordinaria"
                            )
                        )
                        else None
                    ),

                "paid_beds":
                    (
                        int(
                            row[
                                "Posti letto degenza a pagamento"
                            ]
                        )
                        if pd.notna(
                            row.get(
                                "Posti letto degenza a pagamento"
                            )
                        )
                        else None
                    ),

                "day_hospital_beds":
                    (
                        int(
                            row[
                                "Posti letto Day Hospital"
                            ]
                        )
                        if pd.notna(
                            row.get(
                                "Posti letto Day Hospital"
                            )
                        )
                        else None
                    ),

                "day_surgery_beds":
                    (
                        int(
                            row[
                                "Posti letto Day Surgery"
                            ]
                        )
                        if pd.notna(
                            row.get(
                                "Posti letto Day Surgery"
                            )
                        )
                        else None
                    ),

                "total_beds":
                    (
                        int(
                            row[
                                "Totale posti letto"
                            ]
                        )
                        if pd.notna(
                            row.get(
                                "Totale posti letto"
                            )
                        )
                        else None
                    ),
            }
        )

    return json.dumps(
        records,
        ensure_ascii=False,
    )


def first_non_null(group, column):
    for value in group[column]:
        value = normalize_text(value)

        if value is not None:
            return value

    return None


def aggregate_hospitals(
    disciplines,
    municipality_code,
    hospital_year,
):
    df = disciplines.loc[
        disciplines[
            "Codice Comune"
        ]
        == municipality_code
    ].copy()

    rows = []

    grouping = [
        "Codice struttura",
        "Subcodice",
    ]

    for (
        structure_code,
        subcode,
    ), group in df.groupby(
        grouping,
        dropna=False,
        sort=True,
    ):

        total = lambda column: int(
            group[column]
            .fillna(0)
            .astype("Int64")
            .sum()
        )

        rows.append(
            {
                "service_site_id":
                    "SALUTE:HOSPITAL:"
                    f"{structure_code}:"
                    f"{subcode}",

                "category":
                    "health",

                "subcategory":
                    "hospital",

                "source_system":
                    "Ministero della Salute",

                "source_dataset":
                    (
                        "Posti letto per stabilimento "
                        "ospedaliero e disciplina"
                    ),

                "source_record_id":
                    f"{structure_code}:{subcode}",

                "structure_code":
                    structure_code,

                "establishment_subcode":
                    subcode,

                "name":
                    first_non_null(
                        group,
                        "Denominazione Struttura/Stabilimento",
                    ),

                "address":
                    first_non_null(
                        group,
                        "Indirizzo",
                    ),

                "municipality_code":
                    first_non_null(
                        group,
                        "Codice Comune",
                    ),

                "municipality_name":
                    first_non_null(
                        group,
                        "Comune",
                    ),

                "province_code":
                    first_non_null(
                        group,
                        "Sigla Provincia",
                    ),

                "region_code":
                    first_non_null(
                        group,
                        "Codice Regione",
                    ),

                "region_name":
                    first_non_null(
                        group,
                        "Descrizione Regione",
                    ),

                "health_authority_code":
                    first_non_null(
                        group,
                        "Codice Azienda",
                    ),

                "facility_type_code":
                    first_non_null(
                        group,
                        "Codice tipo struttura",
                    ),

                "facility_type":
                    first_non_null(
                        group,
                        "Descrizione tipo struttura",
                    ),

                "reference_year":
                    int(hospital_year),

                "discipline_count":
                    int(
                        group[
                            "Codice disciplina"
                        ]
                        .nunique()
                    ),

                "department_count":
                    total(
                        "N° Reparti"
                    ),

                "ordinary_beds":
                    total(
                        "Posti letto degenza ordinaria"
                    ),

                "paid_beds":
                    total(
                        "Posti letto degenza a pagamento"
                    ),

                "day_hospital_beds":
                    total(
                        "Posti letto Day Hospital"
                    ),

                "day_surgery_beds":
                    total(
                        "Posti letto Day Surgery"
                    ),

                "total_beds":
                    total(
                        "Totale posti letto"
                    ),

                "disciplines_json":
                    discipline_json(
                        group
                    ),

                "latitude":
                    pd.NA,

                "longitude":
                    pd.NA,

                "coordinate_resolution":
                    "missing",

                "confidence":
                    "unresolved",

                "needs_geolocation":
                    True,

                "usable_for_accessibility":
                    False,
            }
        )

    return pd.DataFrame(
        rows
    )


def make_pharmacy_geodataframe(
    pharmacies,
):
    geometry = []

    for _, row in pharmacies.iterrows():
        if bool(
            row[
                "usable_for_accessibility"
            ]
        ):
            geometry.append(
                Point(
                    float(
                        row[
                            "longitude"
                        ]
                    ),
                    float(
                        row[
                            "latitude"
                        ]
                    ),
                )
            )
        else:
            geometry.append(
                None
            )

    return gpd.GeoDataFrame(
        pharmacies,
        geometry=geometry,
        crs="EPSG:4326",
    )


def main():
    args = parse_args()

    pharmacy_raw_path = latest_file(
        RAW_SALUTE_DIR
        / "farmacie",
        "*.csv",
    )

    hospital_raw_path = latest_file(
        RAW_SALUTE_DIR
        / f"strutture_ospedaliere_{args.hospital_year}",
        "*.csv",
    )

    pharmacy_raw = read_csv_auto(
        pharmacy_raw_path
    )

    hospital_raw = read_csv_auto(
        hospital_raw_path
    )

    pharmacy_history = (
        build_pharmacy_history(
            pharmacy_raw
        )
    )

    pharmacies = snapshot_pharmacies(
        pharmacy_history,
        args.municipality_code,
        args.reference_date,
    )

    hospital_disciplines = (
        build_hospital_discipline_history(
            hospital_raw,
            args.hospital_year,
        )
    )

    hospitals = aggregate_hospitals(
        hospital_disciplines,
        args.municipality_code,
        args.hospital_year,
    )

    national_dir = (
        PROCESSED_SALUTE_DIR
        / "national"
    )

    municipality_dir = (
        PROCESSED_SALUTE_DIR
        / args.municipality_code
    )

    feature_dir = (
        FEATURES_SALUTE_DIR
        / args.municipality_code
    )

    for directory in [
        national_dir,
        municipality_dir,
        feature_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    reference_label = (
        args.reference_date
        .strftime(
            "%Y%m%d"
        )
    )

    pharmacy_history_path = (
        national_dir
        / "pharmacy_registry_history.parquet"
    )

    hospital_disciplines_path = (
        national_dir
        / (
            "hospital_disciplines_"
            f"{args.hospital_year}.parquet"
        )
    )

    pharmacy_sites_path = (
        municipality_dir
        / (
            "pharmacy_sites_"
            f"{reference_label}.parquet"
        )
    )

    hospital_sites_path = (
        municipality_dir
        / (
            "hospital_sites_"
            f"{args.hospital_year}.parquet"
        )
    )

    pharmacy_csv_path = (
        feature_dir
        / (
            "pharmacy_sites_"
            f"{reference_label}.csv"
        )
    )

    hospital_csv_path = (
        feature_dir
        / (
            "hospital_sites_"
            f"{args.hospital_year}.csv"
        )
    )

    manifest_path = (
        feature_dir
        / (
            "health_silver_manifest_"
            f"{reference_label}.json"
        )
    )

    pharmacy_history.to_parquet(
        pharmacy_history_path,
        index=False,
    )

    hospital_disciplines.to_parquet(
        hospital_disciplines_path,
        index=False,
    )

    pharmacy_gdf = (
        make_pharmacy_geodataframe(
            pharmacies
        )
    )

    pharmacy_gdf.to_parquet(
        pharmacy_sites_path,
        index=False,
    )

    hospitals.to_parquet(
        hospital_sites_path,
        index=False,
    )

    pharmacies.to_csv(
        pharmacy_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    hospitals.to_csv(
        hospital_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "municipality_code":
            args.municipality_code,

        "pharmacy_reference_date":
            args.reference_date.date().isoformat(),

        "hospital_reference_year":
            int(args.hospital_year),

        "raw_pharmacy_file":
            str(pharmacy_raw_path),

        "raw_hospital_file":
            str(hospital_raw_path),

        "national_pharmacy_history_rows":
            int(
                len(
                    pharmacy_history
                )
            ),

        "active_pharmacy_sites":
            int(
                len(
                    pharmacies
                )
            ),

        "active_pharmacy_sites_with_coordinates":
            int(
                pharmacies[
                    "usable_for_accessibility"
                ].sum()
            ),

        "active_pharmacy_sites_needing_geolocation":
            int(
                pharmacies[
                    "needs_geolocation"
                ].sum()
            ),

        "pharmacy_overlapping_active_versions":
            int(
                (
                    pharmacies[
                        "active_version_count"
                    ]
                    > 1
                ).sum()
            ),

        "hospital_discipline_rows_municipality":
            int(
                (
                    hospital_disciplines[
                        "Codice Comune"
                    ]
                    == args.municipality_code
                ).sum()
            ),

        "unique_hospital_sites":
            int(
                len(
                    hospitals
                )
            ),

        "hospital_sites_needing_geolocation":
            int(
                hospitals[
                    "needs_geolocation"
                ].sum()
            )
            if not hospitals.empty
            else 0,

        "hospital_total_beds":
            int(
                hospitals[
                    "total_beds"
                ].sum()
            )
            if not hospitals.empty
            else 0,

        "notes": [
            (
                "Le farmacie sono filtrate per intervallo di validità "
                "alla data di riferimento, non per data dello snapshot raw."
            ),
            (
                "Gli stabilimenti ospedalieri sono aggregati su "
                "(Codice struttura, Subcodice); le righe per disciplina "
                "non sono trattate come siti distinti."
            ),
            (
                "Le coordinate ospedaliere restano mancanti finché "
                "non viene eseguito un geocoding/matching separato."
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
        " HEALTH SILVER BUILD"
    )
    print(
        "===================================="
    )

    print(
        "\n=== FARMACIE ==="
    )
    print(
        "Reference date: "
        f"{args.reference_date.date().isoformat()}"
    )
    print(
        "Raw municipality rows "
        "(tutte le versioni storiche): "
        f"{int((pharmacy_history['cod_comune'] == args.municipality_code).sum())}"
    )
    print(
        "Farmacie attive distinte: "
        f"{len(pharmacies)}"
    )
    print(
        "Con coordinate: "
        f"{int(pharmacies['usable_for_accessibility'].sum())}"
    )
    print(
        "Da geolocalizzare: "
        f"{int(pharmacies['needs_geolocation'].sum())}"
    )
    print(
        "Codici con più versioni attive sovrapposte: "
        f"{int((pharmacies['active_version_count'] > 1).sum())}"
    )

    if not pharmacies.empty:
        print(
            "\nFarmacie attive:"
        )
        print(
            pharmacies[
                [
                    "cod_farmacia",
                    "name",
                    "address",
                    "valid_from",
                    "valid_to",
                    "latitude",
                    "longitude",
                    "needs_geolocation",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\n=== OSPEDALI ==="
    )
    print(
        "Righe disciplina nel comune: "
        f"{manifest['hospital_discipline_rows_municipality']}"
    )
    print(
        "Stabilimenti ospedalieri distinti: "
        f"{len(hospitals)}"
    )
    print(
        "Posti letto aggregati: "
        f"{manifest['hospital_total_beds']}"
    )
    print(
        "Da geolocalizzare: "
        f"{manifest['hospital_sites_needing_geolocation']}"
    )

    if not hospitals.empty:
        print(
            "\nStabilimenti:"
        )
        print(
            hospitals[
                [
                    "structure_code",
                    "establishment_subcode",
                    "name",
                    "address",
                    "facility_type",
                    "discipline_count",
                    "department_count",
                    "total_beds",
                    "needs_geolocation",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\n=== OUTPUT ==="
    )
    print(
        f"✓ {pharmacy_history_path}"
    )
    print(
        f"✓ {hospital_disciplines_path}"
    )
    print(
        f"✓ {pharmacy_sites_path}"
    )
    print(
        f"✓ {hospital_sites_path}"
    )
    print(
        f"✓ {pharmacy_csv_path}"
    )
    print(
        f"✓ {hospital_csv_path}"
    )
    print(
        f"✓ {manifest_path}"
    )


if __name__ == "__main__":
    main()
