import argparse
from pathlib import Path
import shutil
import zipfile

import requests


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

BOUNDARIES_DIR = (
    ISTAT_RAW
    / "basi_territoriali_2021"
)

DOWNLOAD_DIR = (
    ISTAT_RAW
    / "downloads"
)

ISTAT_BOUNDARIES_BASE_URL = (
    "https://www.istat.it/storage/cartografia/"
    "basi_territoriali/2021"
)

VALID_REGION_CODES = {
    f"{code:02d}"
    for code in range(1, 21)
}


# ============================================================
# HELPERS
# ============================================================

def normalize_region_code(region_code):
    region_code = str(region_code).strip().zfill(2)

    if region_code not in VALID_REGION_CODES:
        raise ValueError(
            "Codice regione non valido: "
            f"{region_code}. "
            "Sono ammessi i codici da 01 a 20."
        )

    return region_code


def expected_shapefile(region_code):
    return (
        BOUNDARIES_DIR
        / "SHP"
        / f"R{region_code}_21_WGS84.shp"
    )


def expected_sidecars(region_code):
    base = (
        BOUNDARIES_DIR
        / "SHP"
        / f"R{region_code}_21_WGS84"
    )

    return [
        base.with_suffix(".shp"),
        base.with_suffix(".dbf"),
        base.with_suffix(".shx"),
        base.with_suffix(".prj"),
    ]


def region_zip_path(region_code):
    return (
        DOWNLOAD_DIR
        / f"R{region_code}_21.zip"
    )


def region_download_url(region_code):
    return (
        f"{ISTAT_BOUNDARIES_BASE_URL}/"
        f"R{region_code}_21.zip"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_region_files(region_code):
    missing = [
        path
        for path in expected_sidecars(region_code)
        if not path.exists()
    ]

    if missing:
        return False, missing

    return True, []


# ============================================================
# SAFE ZIP EXTRACTION
# ============================================================

def safe_extract_zip(zip_path, destination):
    """
    Estrae lo ZIP impedendo path traversal.
    """

    destination = destination.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:

        for member in archive.infolist():

            target = (
                destination
                / member.filename
            ).resolve()

            try:
                target.relative_to(destination)
            except ValueError:
                raise RuntimeError(
                    "ZIP non sicuro: "
                    f"{member.filename}"
                )

        archive.extractall(destination)


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(url, destination):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = destination.with_suffix(
        destination.suffix + ".part"
    )

    print("\nDownload ISTAT:")
    print(f"  {url}")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, 180),
            headers={
                "User-Agent": (
                    "urban-digital-twin-thesis/1.0"
                )
            },
        ) as response:

            response.raise_for_status()

            total = int(
                response.headers.get(
                    "content-length",
                    0,
                )
            )

            downloaded = 0

            with open(
                temporary_file,
                "wb",
            ) as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    file.write(chunk)

                    downloaded += len(chunk)

                    if total:
                        percentage = (
                            downloaded
                            / total
                            * 100
                        )

                        print(
                            f"\r  {percentage:6.2f}%",
                            end="",
                            flush=True,
                        )

        print()

        temporary_file.replace(
            destination
        )

    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()

        raise


# ============================================================
# MAIN DOWNLOAD LOGIC
# ============================================================

def ensure_region_boundaries(
    region_code,
    force=False,
):
    """
    Garantisce che le basi territoriali ISTAT
    della regione siano disponibili localmente.

    Se sono già presenti:
        non scarica nulla.

    Se mancano:
        scarica RXX_21.zip,
        lo conserva nella directory raw,
        lo estrae e valida i file.
    """

    region_code = normalize_region_code(
        region_code
    )

    valid, missing = validate_region_files(
        region_code
    )

    if valid and not force:
        print(
            "\n✓ Basi territoriali ISTAT "
            f"regione {region_code} "
            "già disponibili."
        )

        return expected_shapefile(
            region_code
        )

    zip_path = region_zip_path(
        region_code
    )

    url = region_download_url(
        region_code
    )

    if force or not zip_path.exists():

        download_file(
            url,
            zip_path,
        )

    else:
        print(
            "\n✓ ZIP ISTAT già presente:"
        )

        print(
            f"  {zip_path}"
        )

    # Verifica preliminare ZIP
    if not zipfile.is_zipfile(
        zip_path
    ):
        raise RuntimeError(
            "Il file scaricato non è "
            f"uno ZIP valido:\n{zip_path}"
        )

    print(
        "\nEstrazione basi territoriali..."
    )

    BOUNDARIES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_extract_zip(
        zip_path,
        BOUNDARIES_DIR,
    )

    valid, missing = validate_region_files(
        region_code
    )

    if not valid:
        missing_text = "\n".join(
            str(path)
            for path in missing
        )

        raise RuntimeError(
            "\nEstrazione completata ma "
            "mancano file obbligatori:\n"
            f"{missing_text}"
        )

    shapefile = expected_shapefile(
        region_code
    )

    print(
        "\n✓ Basi territoriali installate."
    )

    print(
        f"  Regione: {region_code}"
    )

    print(
        f"  Shapefile: {shapefile}"
    )

    return shapefile


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download automatico delle "
            "Basi Territoriali ISTAT 2021."
        )
    )

    parser.add_argument(
        "--region-code",
        required=True,
        help=(
            "Codice ISTAT regione. "
            "Esempio: 15 = Campania."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Scarica nuovamente il file "
            "anche se già disponibile."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ensure_region_boundaries(
        region_code=args.region_code,
        force=args.force,
    )


if __name__ == "__main__":
    main()