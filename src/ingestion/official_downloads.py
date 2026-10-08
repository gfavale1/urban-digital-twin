"""B5B2 verified, opt-in acquisition of pinned official Bronze distributions.

No historical file is overwritten. Every successful output is written atomically
and validated before it becomes visible to downstream ingestion.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import pandas as pd
import requests


ISTAT_2023_REGIONAL_ARCHIVE = (
    "https://esploradati.istat.it/databrowser/DWL/PERMPOP/"
    "SUBCOM/Dati_regionali_2023.zip"
)
MIM_202425_BUILDINGS_URL = (
    "https://dati.istruzione.it/opendata/opendata/catalog/"
    "EDIANAGRAFESTA202120242520250806.csv"
)
SALUTE_PHARMACIES_CATALOGUE = "https://www.dati.salute.gov.it/it/dataset/farmacie/"
SALUTE_HOSPITAL_2023_URL = (
    "https://www.dati.salute.gov.it/sites/default/files/2025-07/"
    "Posti%20letto%20per%20stabilimento%20ospedaliero%20e%20disciplina_2023_0.csv"
)
SALUTE_HOSPITAL_2023_FILENAME = (
    "Posti%20letto%20per%20stabilimento%20ospedaliero%20e%20disciplina_2023_0.csv"
)
USER_AGENT = "urban-digital-twin-thesis/1.0 (academic open-data acquisition)"

# Limits protect against a mistaken webpage, redirect or ZIP bomb.
MAX_ISTAT_ARCHIVE_BYTES = 2 * 1024**3
MAX_ISTAT_REGION_BYTES = 250 * 1024**2
MAX_CSV_BYTES = 500 * 1024**2


@dataclass(frozen=True)
class DownloadedSource:
    path: Path
    source_url: str
    note: str = ""


def _require_official_url(url: str, hosts: frozenset[str]) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in hosts:
        raise ValueError(f"Not an allowed official HTTPS download URL: {url!r}")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Download URL must not contain credentials or fragments.")


def _stream_download(url: str, destination: Path, *, hosts: frozenset[str], limit: int) -> None:
    """Write to a temporary file; validate response destination and body size."""
    _require_official_url(url, hosts)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing Bronze file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        raise FileExistsError(f"Incomplete download already exists: {partial}")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(20, 180),
            headers={"User-Agent": USER_AGENT},
        ) as response:
            response.raise_for_status()
            _require_official_url(response.url, hosts)
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > limit:
                raise ValueError("Official download exceeds maximum allowed size.")
            byte_count = 0
            with partial.open("xb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    byte_count += len(chunk)
                    if byte_count > limit:
                        raise ValueError("Official download exceeds maximum allowed size.")
                    handle.write(chunk)
            if not byte_count:
                raise ValueError("Official download returned an empty file.")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _validate_csv_columns(path: Path, dataset: str) -> None:
    """Validate a CSV header against the existing ingestion contracts."""
    required: dict[str, set[str]] = {
        "mim_buildings": {"codicescuola", "codiceedificio"},
        "salute_pharmacies": {
            "codfarmacia", "codcomune", "datainiziovalidita", "datafinevalidita",
        },
        "salute_hospital": {
            "anno", "codicestruttura", "subcodice", "codicecomune", "codicedisciplina",
        },
    }
    if dataset not in required:
        raise ValueError(f"Unknown CSV contract {dataset!r}.")

    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            columns = pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding,
                nrows=0,
            ).columns
            canonical = {
                re.sub(r"[^a-z0-9]", "", str(column).lower())
                for column in columns
            }
            if required[dataset].issubset(canonical):
                return
            errors.append(f"{encoding}: missing {sorted(required[dataset] - canonical)}")
        except Exception as exc:
            errors.append(f"{encoding}: {type(exc).__name__}")
    raise ValueError(
        f"Downloaded CSV does not match {dataset} ingestion schema: {'; '.join(errors)}"
    )


def _download_csv(url: str, target: Path, *, hosts: frozenset[str], dataset: str) -> None:
    # Destination itself must remain absent if header validation fails.
    # Check before contacting the network to ensure pinned Bronze is never
    # needlessly re-fetched or replaced.
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing Bronze file: {target}")
    temporary = target.with_name(target.name + ".validating")
    if temporary.exists():
        raise FileExistsError(f"Temporary acquisition path already exists: {temporary}")
    try:
        _stream_download(url, temporary, hosts=hosts, limit=MAX_CSV_BYTES)
        _validate_csv_columns(temporary, dataset)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite Bronze file: {target}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def download_mim_buildings(building_year: str, target_dir: Path) -> DownloadedSource:
    if str(building_year) != "202425":
        raise ValueError("MIM building download is pinned only for building_year=202425.")
    target = target_dir / Path(urlsplit(MIM_202425_BUILDINGS_URL).path).name
    _download_csv(
        MIM_202425_BUILDINGS_URL,
        target,
        hosts=frozenset({"dati.istruzione.it"}),
        dataset="mim_buildings",
    )
    return DownloadedSource(
        target,
        MIM_202425_BUILDINGS_URL,
        note="Pinned MIM 2024/25 building distribution published 2025-08-06.",
    )


def _validate_xlsx(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError("ISTAT regional workbook is not a valid XLSX/ZIP file.")
    with zipfile.ZipFile(path) as archive:
        if "[Content_Types].xml" not in archive.namelist():
            raise ValueError("ISTAT regional workbook lacks XLSX content-types metadata.")


def download_istat_region(
    region_code: str,
    census_year: str,
    target_dir: Path,
    archive_cache_dir: Path,
) -> DownloadedSource:
    if str(census_year) != "2023":
        raise ValueError("ISTAT census download is pinned only to 2023.")
    region = str(region_code).zfill(2)
    if not region.isdigit() or region not in {f"{v:02d}" for v in range(1, 21)}:
        raise ValueError(f"Invalid ISTAT region code: {region_code!r}")

    archive_path = archive_cache_dir / "Dati_regionali_2023.zip"
    if not archive_path.exists():
        _stream_download(
            ISTAT_2023_REGIONAL_ARCHIVE,
            archive_path,
            hosts=frozenset({"esploradati.istat.it"}),
            limit=MAX_ISTAT_ARCHIVE_BYTES,
        )
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"ISTAT cached national archive is not a valid ZIP: {archive_path}")

    expected_filename = re.compile(rf"R{region}_.+_2023_sezioni\.xlsx", re.IGNORECASE)
    with zipfile.ZipFile(archive_path) as archive:
        matches = [
            info for info in archive.infolist()
            if not info.is_dir()
            and expected_filename.fullmatch(Path(info.filename).name)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one ISTAT 2023 regional workbook for R{region}; "
                f"found {len(matches)}. No implicit substitution."
            )
        member = matches[0]
        if member.file_size > MAX_ISTAT_REGION_BYTES:
            raise ValueError("ISTAT regional XLSX exceeds uncompressed size limit.")
        target = target_dir / Path(member.filename).name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite Bronze census: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".validating")
        if temp.exists():
            raise FileExistsError(f"Temporary extraction already exists: {temp}")
        try:
            with archive.open(member) as src, temp.open("xb") as dst:
                copied = 0
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > MAX_ISTAT_REGION_BYTES:
                        raise ValueError("ISTAT XLSX extraction exceeded size limit.")
                    dst.write(chunk)
            _validate_xlsx(temp)
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    return DownloadedSource(
        target,
        ISTAT_2023_REGIONAL_ARCHIVE,
        note="Extracted one regional workbook from the official ISTAT 2023 national ZIP; archive cached.",
    )


class _AnchorParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self.hrefs.extend(str(value) for key, value in attrs if key == "href" and value)


def discover_pharmacy_download_url() -> tuple[str, date]:
    allowed_hosts = frozenset({"www.dati.salute.gov.it", "dati.salute.gov.it"})
    with requests.get(
        SALUTE_PHARMACIES_CATALOGUE,
        timeout=(20, 60),
        headers={"User-Agent": USER_AGENT},
    ) as response:
        response.raise_for_status()
        _require_official_url(response.url, allowed_hosts)
        parser = _AnchorParser()
        parser.feed(response.text)

    candidates = set()
    for href in parser.hrefs:
        url = urljoin(SALUTE_PHARMACIES_CATALOGUE, href)
        name = unquote(Path(urlsplit(url).path).name)
        match = re.fullmatch(r"FRM_FARMA_5_(\d{8})\.csv", name, flags=re.IGNORECASE)
        if match:
            _require_official_url(url, allowed_hosts)
            raw_date = match.group(1)
            try:
                release_date = date.fromisoformat(
                    f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
                )
            except ValueError:
                continue
            candidates.add((url, release_date))

    if len(candidates) != 1:
        raise ValueError(
            "Official Salute pharmacy catalogue must expose exactly one dated CSV; "
            f"found {len(candidates)}."
        )
    return next(iter(candidates))


def download_salute_pharmacies(
    reference_date: str,
    target_dir: Path,
    *,
    allow_contemporary_source: bool = False,
) -> DownloadedSource:
    download_url, publication_date = discover_pharmacy_download_url()
    reference = date.fromisoformat(str(reference_date))
    if publication_date > reference and not allow_contemporary_source:
        raise PermissionError(
            f"Current pharmacy publication {publication_date} is newer than requested "
            f"historical reference date {reference}. To allow this explicitly, use "
            "--allow-contemporary-pharmacy-source. Publication date and historical "
            "validity period are NOT interchangeable."
        )
    target = target_dir / unquote(Path(urlsplit(download_url).path).name)
    _download_csv(
        download_url,
        target,
        hosts=frozenset({"www.dati.salute.gov.it", "dati.salute.gov.it"}),
        dataset="salute_pharmacies",
    )
    return DownloadedSource(
        target,
        download_url,
        note=(
            f"Pharmacy publication {publication_date}, requested historical "
            f"reference {reference}; historical completeness is NOT guaranteed by the "
            "validity-date columns."
        ),
    )


def download_salute_hospitals(hospital_year: str, target_dir: Path) -> DownloadedSource:
    if str(hospital_year) != "2023":
        raise ValueError("Salute hospital acquisition is pinned only for hospital_year=2023.")
    target = target_dir / SALUTE_HOSPITAL_2023_FILENAME
    _download_csv(
        SALUTE_HOSPITAL_2023_URL,
        target,
        hosts=frozenset({"www.dati.salute.gov.it", "dati.salute.gov.it"}),
        dataset="salute_hospital",
    )
    return DownloadedSource(
        target,
        SALUTE_HOSPITAL_2023_URL,
        note="Official 2023 hospital establishment/discipline dataset, uploaded July 2025.",
    )
