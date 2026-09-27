from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

RAW_ISTAT_DIR = RAW_DIR / "istat"
RAW_MIM_DIR = RAW_DIR / "mim"
RAW_SALUTE_DIR = RAW_DIR / "salute"
RAW_OSM_DIR = RAW_DIR / "osm"

PROCESSED_ISTAT_DIR = PROCESSED_DIR / "istat"
PROCESSED_MIM_DIR = PROCESSED_DIR / "mim"
PROCESSED_SALUTE_DIR = PROCESSED_DIR / "salute"
PROCESSED_OSM_DIR = PROCESSED_DIR / "osm"
PROCESSED_SERVICES_DIR = PROCESSED_DIR / "services"

FEATURES_MIM_DIR = FEATURES_DIR / "mim"
FEATURES_SALUTE_DIR = FEATURES_DIR / "salute"
FEATURES_SERVICES_DIR = FEATURES_DIR / "services"
FEATURES_ACCESSIBILITY_DIR = FEATURES_DIR / "accessibility"
FEATURES_COMPARISON_DIR = FEATURES_DIR / "comparison"


def municipality_dir(
    base: Path,
    municipality_code: str,
) -> Path:
    """
    Restituisce la directory relativa a uno specifico comune.
    """
    return base / normalize_municipality_code(
        municipality_code
    )


def normalize_municipality_code(
    municipality_code: str,
) -> str:
    """
    Normalizza e valida il codice ISTAT comunale.
    """
    code = str(
        municipality_code
    ).strip().zfill(6)

    if (
        not code.isdigit()
        or len(code) != 6
    ):
        raise ValueError(
            "Il codice ISTAT comunale deve "
            "contenere esattamente 6 cifre."
        )

    return code
