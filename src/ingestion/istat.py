import argparse
from datetime import date
from pathlib import Path
import os

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from sqlalchemy import create_engine, text

from download_istat_boundaries import ensure_region_boundaries


# ============================================================
# PATHS / CONSTANTS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ISTAT_RAW = (
    ROOT
    / "data"
    / "raw"
    / "istat"
)

SHAPE_DIR = (
    ISTAT_RAW
    / "basi_territoriali_2021"
    / "SHP"
)

CENSUS_DIR = (
    ISTAT_RAW
    / "censimento_2023"
    / "Dati_regionali_2023"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "istat"
)

CENSUS_REFERENCE_DATE = date(
    2023,
    12,
    31,
)


# ============================================================
# ISTAT VARIABLE MAPPING
# ============================================================

AGE_0_14 = [
    "P14",
    "P15",
    "P16",
]

AGE_15_64 = [
    "P17",
    "P18",
    "P19",
    "P20",
    "P21",
    "P22",
    "P23",
    "P24",
    "P25",
    "P26",
]

AGE_65_PLUS = [
    "P27",
    "P28",
    "P29",
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ingestion nazionale dei dati censuari ISTAT "
            "per un comune italiano."
        )
    )

    parser.add_argument(
        "--region-code",
        required=True,
        help=(
            "Codice ISTAT della regione a 2 cifre. "
            "Esempio: 17 per Basilicata."
        ),
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help=(
            "Codice ISTAT del comune a 6 cifre. "
            "Esempio: 077014 per Matera."
        ),
    )

    args = parser.parse_args()

    args.region_code = (
        str(args.region_code)
        .strip()
        .zfill(2)
    )

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    if not args.region_code.isdigit():
        raise ValueError(
            "region-code deve contenere solo cifre."
        )

    if len(args.region_code) != 2:
        raise ValueError(
            "region-code deve avere 2 cifre."
        )

    if not args.municipality_code.isdigit():
        raise ValueError(
            "municipality-code deve contenere solo cifre."
        )

    if len(args.municipality_code) != 6:
        raise ValueError(
            "municipality-code deve avere 6 cifre."
        )

    return args


# ============================================================
# HELPERS
# ============================================================

def int_or_none(value):
    if pd.isna(value):
        return None

    return int(value)


def float_or_none(value):
    if pd.isna(value):
        return None

    return float(value)


def to_multipolygon(geometry):
    if geometry is None:
        return None

    if isinstance(
        geometry,
        MultiPolygon,
    ):
        return geometry

    if isinstance(
        geometry,
        Polygon,
    ):
        return MultiPolygon(
            [geometry]
        )

    raise ValueError(
        "Geometria non supportata: "
        f"{geometry.geom_type}"
    )


# ============================================================
# FILE RESOLUTION
# ============================================================

def resolve_istat_files(region_code):
    """
    Risolve automaticamente le sorgenti ISTAT.

    Se le basi territoriali regionali non
    sono presenti, vengono scaricate.
    """

    shapefile = ensure_region_boundaries(
        region_code
    )

    census_matches = list(
        CENSUS_DIR.glob(
            f"R{region_code}_*_2023_sezioni.xlsx"
        )
    )

    if not census_matches:
        raise FileNotFoundError(
            "\nFile censuario 2023 non trovato "
            f"per la regione {region_code}.\n"
            f"Directory cercata:\n{CENSUS_DIR}"
        )

    if len(census_matches) > 1:
        raise RuntimeError(
            "\nSono stati trovati più file censuari "
            f"per la regione {region_code}:\n"
            + "\n".join(
                str(path)
                for path in census_matches
            )
        )

    census_file = census_matches[0]

    return shapefile, census_file


# ============================================================
# EXTRACT
# ============================================================

def load_raw_data(
    shapefile,
    census_file,
):
    print(
        "\nCaricamento geometrie:"
    )

    print(
        f"  {shapefile.name}"
    )

    shapes = gpd.read_file(
        shapefile
    )

    print(
        "\nCaricamento censimento:"
    )

    print(
        f"  {census_file.name}"
    )

    census = pd.read_excel(
        census_file
    )

    return shapes, census


# ============================================================
# MUNICIPALITY FILTER
# ============================================================

def filter_municipality(
    shapes,
    census,
    municipality_code,
    region_code,
):
    """
    Filtra shapefile e censimento per il comune richiesto.

    municipality_code:
        codice ISTAT a 6 cifre, es. 077014.

    Nei dataset ISTAT PROCOM è numerico:
        077014 -> 77014.
    """

    procom = int(
        municipality_code
    )

    census_municipality = census[
        census["PROCOM"] == procom
    ].copy()

    if census_municipality.empty:
        raise RuntimeError(
            "\nComune non trovato "
            "nel censimento ISTAT.\n"
            f"Codice ISTAT: {municipality_code}\n"
            f"Regione richiesta: {region_code}"
        )

    actual_regions = (
        census_municipality[
            "CODREG"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(actual_regions) != 1:
        raise RuntimeError(
            "CODREG non univoco "
            "per il comune selezionato."
        )

    actual_region = str(
        actual_regions[0]
    ).zfill(2)

    if actual_region != region_code:
        raise RuntimeError(
            f"\nIl comune {municipality_code} "
            f"appartiene alla regione "
            f"{actual_region}, "
            f"non alla regione {region_code}."
        )

    shapes_municipality = shapes[
        shapes["PRO_COM"] == procom
    ].copy()

    if shapes_municipality.empty:
        raise RuntimeError(
            "\nComune non trovato "
            "nelle geometrie ISTAT.\n"
            f"PRO_COM cercato: {procom}"
        )

    municipality_names = (
        census_municipality["COMUNE"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    municipality_names = municipality_names[
        municipality_names.ne("")
    ]

    province_names = (
        census_municipality["PROVINCIA"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    province_names = province_names[
        province_names.ne("")
    ]

    if not municipality_names.empty:
        municipality_name = municipality_names.iloc[0]
    else:
        dimension_path = (
            PROCESSED_DIR
            / "municipality_dimension_2023.parquet"
        )

        if not dimension_path.exists():
            raise RuntimeError(
                "Nome comunale assente nel censimento e "
                "dimensione amministrativa nazionale non disponibile."
            )

        dimension = pd.read_parquet(
            dimension_path
        )

        dimension["istat_code"] = (
            dimension["istat_code"]
            .astype(str)
            .str.strip()
            .str.zfill(6)
        )

        dimension_row = dimension.loc[
            dimension["istat_code"]
            == municipality_code
        ]

        if len(dimension_row) != 1:
            raise RuntimeError(
                "Impossibile risolvere univocamente il nome "
                f"del comune {municipality_code}."
            )

        municipality_name = str(
            dimension_row.iloc[0]["name"]
        ).strip()

    if not province_names.empty:
        province_name = province_names.iloc[0]
    else:
        province_name = None

    province_code = str(
        int(
            census_municipality[
                "CODPRO"
            ].iloc[0]
        )
    ).zfill(3)

    metadata = {
        "istat_code":
            municipality_code,

        "procom":
            procom,

        "name":
            municipality_name,

        "province_name":
            province_name,

        "province_code":
            province_code,

        "region_code":
            region_code,
    }

    print(
        "\n=== COMUNE ==="
    )

    print(
        f"Nome: {municipality_name}"
    )

    print(
        f"Codice ISTAT: "
        f"{municipality_code}"
    )

    print(
        f"Provincia: "
        f"{province_name} "
        f"({province_code})"
    )

    print(
        f"Regione: "
        f"{region_code}"
    )

    print(
        "Sezioni territoriali: "
        f"{len(shapes_municipality)}"
    )

    print(
        "Osservazioni censuarie 2023: "
        f"{len(census_municipality)}"
    )

    return (
        shapes_municipality,
        census_municipality,
        metadata,
    )


# ============================================================
# KEY NORMALIZATION
# ============================================================

def normalize_keys(
    shapes,
    census,
):
    shapes = shapes.copy()
    census = census.copy()

    shapes["SEZ21_ID"] = (
        shapes["SEZ21_ID"]
        .astype("int64")
        .astype(str)
    )

    census["SEZ21_ID"] = (
        census["SEZ21_ID"]
        .astype("int64")
        .astype(str)
    )

    return shapes, census


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_demographic_features(
    census,
):
    census = census.copy()

    census["age_0_14"] = (
        census[
            AGE_0_14
        ].sum(axis=1)
    )

    census["age_15_64"] = (
        census[
            AGE_15_64
        ].sum(axis=1)
    )

    census["age_65_plus"] = (
        census[
            AGE_65_PLUS
        ].sum(axis=1)
    )

    return census


# ============================================================
# DATA QUALITY
# ============================================================

def validate_data(
    shapes,
    census,
):
    print(
        "\n=== DATA QUALITY ==="
    )

    # --------------------------------------------------------
    # UNIQUE KEYS
    # --------------------------------------------------------

    if shapes[
        "SEZ21_ID"
    ].duplicated().any():

        raise RuntimeError(
            "SEZ21_ID duplicati "
            "nelle geometrie."
        )

    if census[
        "SEZ21_ID"
    ].duplicated().any():

        raise RuntimeError(
            "SEZ21_ID duplicati "
            "nel censimento."
        )

    print(
        "✓ SEZ21_ID univoci"
    )

    # --------------------------------------------------------
    # GEOMETRY / OBSERVATION MATCH
    # --------------------------------------------------------

    shape_ids = set(
        shapes["SEZ21_ID"]
    )

    census_ids = set(
        census["SEZ21_ID"]
    )

    only_shapes = (
        shape_ids
        - census_ids
    )

    only_census = (
        census_ids
        - shape_ids
    )

    print(
        "Sezioni territoriali senza "
        "osservazione 2023: "
        f"{len(only_shapes)}"
    )

    print(
        "Osservazioni 2023 "
        "senza geometria: "
        f"{len(only_census)}"
    )

    # Un dato demografico senza geometria
    # è considerato un errore.
    if only_census:
        raise RuntimeError(
            "Esistono osservazioni "
            "demografiche senza geometria."
        )

    # Una geometria senza osservazione
    # è invece ammessa.
    if only_shapes:
        print(
            "✓ Le sezioni senza osservazione "
            "vengono conservate senza "
            "dati demografici."
        )

    # --------------------------------------------------------
    # AGE CONSISTENCY
    # --------------------------------------------------------

    age_sum = (
        census["age_0_14"]
        + census["age_15_64"]
        + census["age_65_plus"]
    )

    age_mismatch = (
        age_sum
        != census["P1"]
    ).sum()

    print(
        "Sezioni con somma fasce età "
        "!= popolazione: "
        f"{age_mismatch}"
    )

    if age_mismatch:
        raise RuntimeError(
            "Incoerenza nelle "
            "fasce di età."
        )

    # --------------------------------------------------------
    # NEGATIVE VALUES
    # --------------------------------------------------------

    if (
        census["P1"] < 0
    ).any():

        raise RuntimeError(
            "Trovata popolazione negativa."
        )

    if (
        census["PF1"] < 0
    ).any():

        raise RuntimeError(
            "Trovato numero negativo "
            "di famiglie."
        )

    if (
        census["A8"] < 0
    ).any():

        raise RuntimeError(
            "Trovato numero negativo "
            "di abitazioni."
        )

    print(
        "✓ controlli demografici superati"
    )


# ============================================================
# CANONICAL AREA DATASET
# ============================================================

def build_area_dataset(
    shapes,
):
    """
    Costruisce il layer territoriale.

    Contiene tutte le sezioni ISTAT,
    anche quelle prive di osservazione
    censuaria 2023.
    """

    areas = shapes[
        [
            "SEZ21_ID",
            "COD_TIPO_S",
            "TIPO_LOC",
            "geometry",
        ]
    ].copy()

    areas = gpd.GeoDataFrame(
        areas,
        geometry="geometry",
        crs=shapes.crs,
    )

    # PostGIS usa WGS84
    areas = areas.to_crs(
        epsg=4326
    )

    areas["geometry"] = (
        areas["geometry"]
        .apply(
            to_multipolygon
        )
    )

    areas = areas.rename(
        columns={
            "SEZ21_ID":
                "census_section_code",

            "COD_TIPO_S":
                "section_type_code",

            "TIPO_LOC":
                "locality_type",
        }
    )

    return areas


# ============================================================
# CANONICAL OBSERVATION DATASET
# ============================================================

def build_observation_dataset(
    shapes,
    census,
):
    """
    Costruisce lo snapshot demografico 2023.

    La densità viene calcolata usando
    l'area in EPSG:32632, quindi in metri.
    """

    areas = shapes[
        [
            "SEZ21_ID",
            "geometry",
        ]
    ].copy()

    # CRS ISTAT = EPSG:32632.
    # L'area è quindi espressa in m².
    areas["area_m2"] = (
        areas.geometry.area
    )

    observations = census.merge(
        areas[
            [
                "SEZ21_ID",
                "area_m2",
            ]
        ],
        on="SEZ21_ID",
        how="inner",
        validate="one_to_one",
    )

    observations[
        "area_km2"
    ] = (
        observations[
            "area_m2"
        ]
        / 1_000_000
    )

    # Evitiamo divisioni per zero.
    observations[
        "population_density"
    ] = (
        observations["P1"]
        / observations[
            "area_km2"
        ].replace(
            0,
            pd.NA,
        )
    )

    observations = observations[
        [
            "SEZ21_ID",

            "P1",
            "population_density",

            "PF1",

            "age_0_14",
            "age_15_64",
            "age_65_plus",

            "ST1",

            "P101",

            "A8",
        ]
    ].copy()

    observations = (
        observations.rename(
            columns={
                "SEZ21_ID":
                    "census_section_code",

                "P1":
                    "population",

                "PF1":
                    "families",

                "ST1":
                    "foreign_population",

                "P101":
                    "employed_population",

                "A8":
                    "housing_units",
            }
        )
    )

    # Non abbiamo identificato
    # una variabile ISTAT diretta
    # per la disoccupazione.
    observations[
        "unemployed_population"
    ] = None

    observations[
        "reference_date"
    ] = CENSUS_REFERENCE_DATE

    return observations


# ============================================================
# SILVER DATASETS
# ============================================================

def save_silver_datasets(
    areas,
    observations,
    municipality_code,
):
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    areas_path = (
        PROCESSED_DIR
        / (
            f"{municipality_code}"
            "_census_areas_2021.parquet"
        )
    )

    observations_path = (
        PROCESSED_DIR
        / (
            f"{municipality_code}"
            "_census_observations_2023.parquet"
        )
    )

    areas.to_parquet(
        areas_path,
        index=False,
    )

    observations.to_parquet(
        observations_path,
        index=False,
    )

    print(
        "\n=== SILVER DATASETS ==="
    )

    print(
        f"✓ {areas_path}"
    )

    print(
        f"✓ {observations_path}"
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_database_engine():
    load_dotenv(
        ROOT
        / ".env"
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


# ============================================================
# MUNICIPALITY UPSERT
# ============================================================

def upsert_municipality(
    connection,
    areas,
    metadata,
):
    municipality_geometry = (
        unary_union(
            areas.geometry
        )
    )

    municipality_geometry = (
        to_multipolygon(
            municipality_geometry
        )
    )

    query = text("""
        INSERT INTO municipality (
            istat_code,
            name,
            province_code,
            region_code,
            geometry,
            source_system,
            source_record_id
        )

        VALUES (
            :istat_code,
            :name,
            :province_code,
            :region_code,

            ST_Multi(
                ST_GeomFromText(
                    :geometry,
                    4326
                )
            ),

            'ISTAT',
            :source_record_id
        )

        ON CONFLICT (istat_code)
        DO UPDATE SET

            name =
                EXCLUDED.name,

            province_code =
                EXCLUDED.province_code,

            region_code =
                EXCLUDED.region_code,

            geometry =
                EXCLUDED.geometry,

            source_system =
                EXCLUDED.source_system,

            source_record_id =
                EXCLUDED.source_record_id,

            ingested_at =
                NOW()

        RETURNING id;
    """)

    result = connection.execute(
        query,
        {
            "istat_code":
                metadata[
                    "istat_code"
                ],

            "name":
                metadata[
                    "name"
                ],

            "province_code":
                metadata[
                    "province_code"
                ],

            "region_code":
                metadata[
                    "region_code"
                ],

            "geometry":
                municipality_geometry.wkt,

            "source_record_id":
                str(
                    metadata[
                        "procom"
                    ]
                ),
        },
    )

    return result.scalar_one()


# ============================================================
# CENSUS AREA UPSERT
# ============================================================

def upsert_census_areas(
    connection,
    areas,
    municipality_id,
):
    query = text("""
        INSERT INTO census_area (
            census_section_code,
            municipality_id,

            geometry,

            section_type_code,
            locality_type,

            source_system,
            source_record_id
        )

        VALUES (
            :census_section_code,
            :municipality_id,

            ST_Multi(
                ST_GeomFromText(
                    :geometry,
                    4326
                )
            ),

            :section_type_code,
            :locality_type,

            'ISTAT',
            :source_record_id
        )

        ON CONFLICT (
            census_section_code
        )

        DO UPDATE SET

            municipality_id =
                EXCLUDED.municipality_id,

            geometry =
                EXCLUDED.geometry,

            section_type_code =
                EXCLUDED.section_type_code,

            locality_type =
                EXCLUDED.locality_type,

            source_system =
                EXCLUDED.source_system,

            source_record_id =
                EXCLUDED.source_record_id,

            ingested_at =
                NOW();
    """)

    records = []

    for row in areas.itertuples():

        records.append(
            {
                "census_section_code":
                    row.census_section_code,

                "municipality_id":
                    municipality_id,

                "geometry":
                    row.geometry.wkt,

                "section_type_code":
                    int_or_none(
                        row.section_type_code
                    ),

                "locality_type":
                    int_or_none(
                        row.locality_type
                    ),

                "source_record_id":
                    row.census_section_code,
            }
        )

    connection.execute(
        query,
        records,
    )


# ============================================================
# GET AREA IDS
# ============================================================

def get_census_area_ids(
    connection,
    municipality_id,
):
    result = connection.execute(
        text("""
            SELECT
                id,
                census_section_code

            FROM census_area

            WHERE
                municipality_id =
                    :municipality_id;
        """),
        {
            "municipality_id":
                municipality_id,
        },
    )

    return {
        row.census_section_code:
            row.id
        for row in result
    }


# ============================================================
# OBSERVATION REPLACEMENT
# ============================================================

def replace_census_observations(
    connection,
    observations,
    municipality_id,
):
    """
    Il censimento 2023 viene trattato
    come uno snapshot temporale.

    L'operazione è idempotente:
    eventuali osservazioni 2023 già presenti
    per il comune vengono eliminate
    e ricreate.
    """

    connection.execute(
        text("""
            DELETE FROM
                census_observation co

            USING
                census_area ca

            WHERE
                co.census_area_id =
                    ca.id

                AND ca.municipality_id =
                    :municipality_id

                AND co.reference_date =
                    :reference_date;
        """),
        {
            "municipality_id":
                municipality_id,

            "reference_date":
                CENSUS_REFERENCE_DATE,
        },
    )

    area_ids = get_census_area_ids(
        connection,
        municipality_id,
    )

    query = text("""
        INSERT INTO census_observation (
            census_area_id,

            reference_date,

            population,
            population_density,

            families,

            age_0_14,
            age_15_64,
            age_65_plus,

            foreign_population,

            employed_population,
            unemployed_population,

            housing_units,

            source_system,
            source_record_id
        )

        VALUES (
            :census_area_id,

            :reference_date,

            :population,
            :population_density,

            :families,

            :age_0_14,
            :age_15_64,
            :age_65_plus,

            :foreign_population,

            :employed_population,
            :unemployed_population,

            :housing_units,

            'ISTAT',
            :source_record_id
        );
    """)

    records = []

    for row in observations.itertuples():

        area_id = area_ids.get(
            row.census_section_code
        )

        if area_id is None:
            raise RuntimeError(
                "Sezione non presente "
                "in census_area: "
                f"{row.census_section_code}"
            )

        records.append(
            {
                "census_area_id":
                    area_id,

                "reference_date":
                    CENSUS_REFERENCE_DATE,

                "population":
                    int_or_none(
                        row.population
                    ),

                "population_density":
                    float_or_none(
                        row.population_density
                    ),

                "families":
                    int_or_none(
                        row.families
                    ),

                "age_0_14":
                    int_or_none(
                        row.age_0_14
                    ),

                "age_15_64":
                    int_or_none(
                        row.age_15_64
                    ),

                "age_65_plus":
                    int_or_none(
                        row.age_65_plus
                    ),

                "foreign_population":
                    int_or_none(
                        row.foreign_population
                    ),

                "employed_population":
                    int_or_none(
                        row.employed_population
                    ),

                "unemployed_population":
                    None,

                "housing_units":
                    int_or_none(
                        row.housing_units
                    ),

                "source_record_id":
                    row.census_section_code,
            }
        )

    connection.execute(
        query,
        records,
    )


# ============================================================
# DATABASE WRITE
# ============================================================

def write_to_postgis(
    areas,
    observations,
    metadata,
):
    engine = get_database_engine()

    with engine.begin() as connection:

        municipality_id = (
            upsert_municipality(
                connection,
                areas,
                metadata,
            )
        )

        upsert_census_areas(
            connection,
            areas,
            municipality_id,
        )

        replace_census_observations(
            connection,
            observations,
            municipality_id,
        )

    print(
        "\n✓ Dataset ISTAT caricato "
        "correttamente in PostGIS."
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
        " ISTAT NATIONAL INGESTION"
    )

    print(
        "===================================="
    )

    # --------------------------------------------------------
    # FILE DISCOVERY
    # --------------------------------------------------------

    shapefile, census_file = (
        resolve_istat_files(
            args.region_code
        )
    )

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    shapes, census = (
        load_raw_data(
            shapefile,
            census_file,
        )
    )

    # --------------------------------------------------------
    # MUNICIPALITY FILTER
    # --------------------------------------------------------

    (
        shapes,
        census,
        metadata,
    ) = filter_municipality(
        shapes,
        census,
        args.municipality_code,
        args.region_code,
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    shapes, census = (
        normalize_keys(
            shapes,
            census,
        )
    )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    census = (
        build_demographic_features(
            census
        )
    )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    validate_data(
        shapes,
        census,
    )

    # --------------------------------------------------------
    # CANONICAL MODEL
    # --------------------------------------------------------

    areas = (
        build_area_dataset(
            shapes
        )
    )

    observations = (
        build_observation_dataset(
            shapes,
            census,
        )
    )

    print(
        "\n=== CANONICAL MODEL ==="
    )

    print(
        f"Comune: "
        f"{metadata['name']}"
    )

    print(
        f"Codice ISTAT: "
        f"{metadata['istat_code']}"
    )

    print(
        f"Census areas: "
        f"{len(areas)}"
    )

    print(
        "Census observations 2023: "
        f"{len(observations)}"
    )

    print(
        "\nPrime osservazioni:"
    )

    print(
        observations
        .head()
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SILVER
    # --------------------------------------------------------

    save_silver_datasets(
        areas,
        observations,
        metadata[
            "istat_code"
        ],
    )

    # --------------------------------------------------------
    # POSTGIS
    # --------------------------------------------------------

    write_to_postgis(
        areas,
        observations,
        metadata,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        " INGESTION COMPLETATA"
    )

    print(
        "===================================="
    )

    print(
        f"Comune: "
        f"{metadata['name']}"
    )

    print(
        f"Codice ISTAT: "
        f"{metadata['istat_code']}"
    )

    print(
        f"Sezioni territoriali: "
        f"{len(areas)}"
    )

    print(
        "Osservazioni 2023: "
        f"{len(observations)}"
    )


if __name__ == "__main__":
    main()