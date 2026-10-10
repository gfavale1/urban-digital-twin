from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

from ingestion import official_downloads as od
from ingestion import source_acquisition as sa


MIM_BYTES = b"CODICESCUOLA;CODICEEDIFICIO\nX;A\n"
PHARMACY_BYTES = (
    b"cod_farmacia;cod_comune;data_inizio_validita;data_fine_validita\n"
    b"1;034027;01/01/2020;\n"
)
HOSPITAL_BYTES = (
    b"Anno;Codice struttura;Subcodice;Codice Comune;Codice disciplina\n"
    b"2023;11;01;034027;1\n"
)


class FakeResponse:
    def __init__(self, url: str, body: bytes, *, actual_url=None):
        self.url = actual_url or url
        self.body = body
        self.headers = {"Content-Length": str(len(body))}
        self.text = body.decode("utf-8", errors="replace")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1024 * 1024):
        yield self.body


def fake_get_with_body(data: bytes, *, actual_url=None):
    def factory(url, **kwargs):
        return FakeResponse(url, data, actual_url=actual_url)
    return factory


def fake_xlsx():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("xl/workbook.xml", "<workbook/>")
    return buffer.getvalue()


def fake_national_archive(*filenames):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for filename in filenames:
            zf.writestr(f"Dati_regionali_2023/{filename}", fake_xlsx())
    return buffer.getvalue()


class OfficialDownloadsB5B2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_http_or_unknown_host_is_forbidden(self):
        hosts = frozenset({"www.dati.salute.gov.it"})
        for url in (
            "http://www.dati.salute.gov.it/file.csv",
            "https://evil.example/file.csv",
            "https://www.dati.salute.gov.it@evil.example/file.csv",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    od._require_official_url(url, hosts)

    def test_mim_buildings_pinned_download_and_schema(self):
        with patch.object(od.requests, "get", side_effect=fake_get_with_body(MIM_BYTES)) as get:
            acquired = od.download_mim_buildings("202425", self.root / "mim")
        self.assertTrue(acquired.path.is_file())
        self.assertEqual(acquired.path.read_bytes(), MIM_BYTES)
        self.assertIn("20250806", acquired.path.name)
        self.assertEqual(acquired.source_url, od.MIM_202425_BUILDINGS_URL)
        get.assert_called_once()

    def test_mim_invalid_schema_does_not_leave_downloaded_file(self):
        folder = self.root / "mim"
        with patch.object(od.requests, "get", side_effect=fake_get_with_body(b"<html>Error</html>")):
            with self.assertRaises(ValueError):
                od.download_mim_buildings("202425", folder)
        self.assertEqual(list(folder.glob("*")), [])

    def test_existing_bronze_cannot_be_overwritten(self):
        folder = self.root / "mim"
        folder.mkdir()
        target = folder / "EDIANAGRAFESTA202120242520250806.csv"
        target.write_bytes(b"untouched")
        with patch.object(od.requests, "get", side_effect=AssertionError("network forbidden")):
            with self.assertRaises(FileExistsError):
                od.download_mim_buildings("202425", folder)
        self.assertEqual(target.read_bytes(), b"untouched")

    def test_mim_nonpinned_year_fails_before_network(self):
        with patch.object(od.requests, "get", side_effect=AssertionError("network forbidden")):
            with self.assertRaises(ValueError):
                od.download_mim_buildings("202627", self.root / "mim")

    def test_istat_2023_region_extraction_and_archive_cache(self):
        payload = fake_national_archive(
            "R08_Emilia-Romagna_2023_sezioni.xlsx",
            "R15_Campania_2023_sezioni.xlsx",
        )
        with patch.object(od.requests, "get", side_effect=fake_get_with_body(payload)) as get:
            first = od.download_istat_region(
                "08", "2023", self.root / "regions", self.root / "archive"
            )
        get.assert_called_once()
        self.assertTrue(first.path.is_file())
        self.assertTrue((self.root / "archive" / "Dati_regionali_2023.zip").is_file())
        with patch.object(od.requests, "get", side_effect=AssertionError("no re-download")):
            second = od.download_istat_region(
                "15", "2023", self.root / "regions", self.root / "archive"
            )
        self.assertTrue(second.path.is_file())
        self.assertEqual(len(list((self.root / "regions").glob("*.xlsx"))), 2)

    def test_istat_ambiguous_region_is_rejected(self):
        payload = fake_national_archive(
            "R08_Emilia-Romagna_2023_sezioni.xlsx",
            "R08_Emilia_2023_sezioni.xlsx",
        )
        with patch.object(od.requests, "get", side_effect=fake_get_with_body(payload)):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                od.download_istat_region("08", "2023", self.root / "regions", self.root / "archive")
        self.assertFalse((self.root / "regions").exists())

    def test_istat_only_2023_is_supported(self):
        with self.assertRaises(ValueError):
            od.download_istat_region("08", "2024", self.root, self.root)

    def test_istat_fake_workbook_is_rejected_without_partial_file(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("R08_Emilia-Romagna_2023_sezioni.xlsx", b"not a workbook")
        with patch.object(od.requests, "get", side_effect=fake_get_with_body(buffer.getvalue())):
            with self.assertRaises(ValueError):
                od.download_istat_region("08", "2023", self.root / "regions", self.root / "archive")
        self.assertEqual(list((self.root / "regions").glob("*")), [])

    def test_pharmacy_official_catalogue_parses_dated_csv(self):
        html = (
            '<html><a href="/sites/default/files/opendata/'
            'FRM_FARMA_5_20261008.csv">Scarica CSV</a></html>'
        ).encode()
        with patch.object(od.requests, "get", side_effect=fake_get_with_body(html)):
            url, published = od.discover_pharmacy_download_url()
        self.assertEqual(published, date(2026, 10, 8))
        self.assertEqual(
            url, "https://www.dati.salute.gov.it/sites/default/files/opendata/FRM_FARMA_5_20261008.csv"
        )

    def test_pharmacy_historical_snapshot_requires_explicit_consent(self):
        with patch.object(
            od, "discover_pharmacy_download_url",
            return_value=(
                "https://www.dati.salute.gov.it/sites/default/files/opendata/FRM_FARMA_5_20261008.csv",
                date(2026, 10, 8),
            ),
        ):
            with patch.object(od.requests, "get", side_effect=AssertionError("download forbidden")):
                with self.assertRaises(PermissionError):
                    od.download_salute_pharmacies("2025-06-30", self.root / "salute")
        self.assertFalse((self.root / "salute").exists())

    def test_pharmacy_explicit_consent_downloads_and_records_note(self):
        with patch.object(
            od, "discover_pharmacy_download_url",
            return_value=(
                "https://www.dati.salute.gov.it/sites/default/files/opendata/FRM_FARMA_5_20261008.csv",
                date(2026, 10, 8),
            ),
        ):
            with patch.object(od.requests, "get", side_effect=fake_get_with_body(PHARMACY_BYTES)):
                acquired = od.download_salute_pharmacies(
                    "2025-06-30", self.root / "salute", allow_contemporary_source=True
                )
        self.assertTrue(acquired.path.is_file())
        self.assertIn("NOT guaranteed", acquired.note)

    def test_pharmacy_catalogue_rejects_foreign_download(self):
        html = (
            '<a href="https://evil.example/FRM_FARMA_5_20261008.csv">Scarica</a>'
        ).encode()
        with patch.object(od.requests, "get", side_effect=fake_get_with_body(html)):
            with self.assertRaises(ValueError):
                od.discover_pharmacy_download_url()

    def test_hospital_2023_download_uses_pinned_official_url(self):
        with patch.object(od.requests, "get", side_effect=fake_get_with_body(HOSPITAL_BYTES)):
            acquired = od.download_salute_hospitals("2023", self.root / "hospitals")
        self.assertTrue(acquired.path.is_file())
        self.assertEqual(acquired.source_url, od.SALUTE_HOSPITAL_2023_URL)

    def test_hospital_year_not_silently_substituted(self):
        with self.assertRaises(ValueError):
            od.download_salute_hospitals("2024", self.root / "hospitals")

    def test_redirect_to_unapproved_hostname_is_rejected(self):
        with patch.object(
            od.requests, "get",
            side_effect=fake_get_with_body(MIM_BYTES, actual_url="https://example.net/file.csv"),
        ):
            with self.assertRaises(ValueError):
                od.download_mim_buildings("202425", self.root / "mim")
        self.assertFalse((self.root / "mim").exists() and list((self.root / "mim").iterdir()))


class SourceAcquisitionB5B2IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = [
            patch.object(sa, "RAW_ISTAT", self.root / "istat"),
            patch.object(sa, "RAW_MIM", self.root / "mim"),
            patch.object(sa, "RAW_SALUTE", self.root / "salute"),
        ]
        for item in self.stack:
            item.start()
            self.addCleanup(item.stop)

    def fake_acquired(self, filename, url="https://www.dati.salute.gov.it/official.csv"):
        file = self.root / "acquired" / filename
        file.parent.mkdir(exist_ok=True)
        file.write_bytes(b"test,source\n1,2\n")
        return od.DownloadedSource(file, url, "from official catalogue")

    def test_istat_missing_download_includes_sha256(self):
        with patch.object(
            sa.official_downloads,
            "download_istat_region",
            return_value=self.fake_acquired("R08_Emilia-Romagna_2023_sezioni.xlsx"),
        ) as getter:
            row = sa.inspect_istat_census("08", "2023", fetch_supported=True)
        self.assertEqual(row.state, "downloaded")
        self.assertEqual(len(row.sha256), 1)
        getter.assert_called_once()

    def test_cached_istat_never_fetches(self):
        file = (
            self.root / "istat" / "censimento_2023" / "Dati_regionali_2023"
            / "R08_Emilia-Romagna_2023_sezioni.xlsx"
        )
        file.parent.mkdir(parents=True)
        file.write_bytes(b"already-pinned")
        with patch.object(sa.official_downloads, "download_istat_region", side_effect=AssertionError("no fetch")):
            row = sa.inspect_istat_census("08", "2023", fetch_supported=True)
        self.assertEqual(row.state, "cached")
        self.assertEqual(file.read_bytes(), b"already-pinned")

    def test_cached_health_sources_never_use_network_even_if_fetch_enabled(self):
        pharma = self.root / "salute" / "farmacie" / "FRM_FARMA_5_20260927.csv"
        hospital = (
            self.root / "salute" / "strutture_ospedaliere_2023"
            / "Posti%20letto%20per%20stabilimento%20ospedaliero%20e%20disciplina_2023_0.csv"
        )
        for path in (pharma, hospital):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"cached,source\n1,2\n")
        with patch.object(sa.official_downloads, "download_salute_pharmacies", side_effect=AssertionError("no fetch")):
            with patch.object(sa.official_downloads, "download_salute_hospitals", side_effect=AssertionError("no fetch")):
                rows = sa.inspect_health(
                    "2025-06-30", "2023", fetch_supported=True
                )
        self.assertEqual(tuple(row.state for row in rows), ("cached", "cached"))

    def test_historical_pharmacy_refusal_is_explicit_status(self):
        with patch.object(
            sa.official_downloads,
            "download_salute_pharmacies",
            side_effect=PermissionError("historical consent required"),
        ):
            pharma, _ = sa.inspect_health(
                "2025-06-30", "2023", fetch_supported=True
            )
        self.assertEqual(pharma.state, "unsupported_snapshot")
        self.assertIn("consent", pharma.note)

    def test_mim_buildings_fetch_failure_does_not_create_fake_cached_state(self):
        with patch.object(
            sa.official_downloads,
            "download_mim_buildings",
            side_effect=RuntimeError("403 Forbidden"),
        ):
            row = sa.inspect_mim_buildings("202425", fetch_supported=True)
        self.assertEqual(row.state, "acquisition_failed")
        self.assertIn("403", row.note)

    def test_cli_does_not_allow_contemporary_consent_without_fetch(self):
        with self.assertRaises(ValueError):
            sa.main(["--city", "Parma", "--allow-contemporary-pharmacy-source"])


if __name__ == "__main__":
    unittest.main()
