from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.municipality import MunicipalityContext
from ingestion import source_acquisition as sa


class SourceAcquisitionB5B1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.ctx = MunicipalityContext._from_values(
            code="034027", name="Parma", province_code="034",
            region_code="08", census_year="2023",
        )
        self.patches = [
            patch.object(sa, "RAW_ISTAT", self.base / "istat"),
            patch.object(sa, "RAW_MIM", self.base / "mim"),
            patch.object(sa, "RAW_SALUTE", self.base / "salute"),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def write(self, path: Path, data: bytes = b"column1,column2\nfoo,bar\n") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_missing_census_is_explicit(self):
        row = sa.inspect_istat_census("08", "2023")
        self.assertEqual(row.state, "missing")

    def test_census_cached_has_stable_checksum(self):
        path = self.write(self.base / "istat" / "censimento_2023" / "Dati_regionali_2023" / "R08_Emilia_2023_sezioni.xlsx")
        row = sa.inspect_istat_census("08", "2023")
        self.assertEqual(row.state, "cached")
        self.assertEqual(len(row.sha256[path.as_posix()]), 64)
        self.assertEqual(sa.inspect_istat_census("08", "2023").sha256, row.sha256)

    def test_census_ambiguity_rejected(self):
        folder = self.base / "istat" / "censimento_2023" / "Dati_regionali_2023"
        self.write(folder / "R08_A_2023_sezioni.xlsx")
        self.write(folder / "R08_B_2023_sezioni.xlsx")
        row = sa.inspect_istat_census("08", "2023")
        self.assertEqual(row.state, "ambiguous")
        self.assertEqual(len(row.paths), 2)

    def test_census_future_year_not_silently_substituted(self):
        row = sa.inspect_istat_census("08", "2024")
        self.assertEqual(row.state, "unsupported_snapshot")

    def test_mim_cached_and_offline_never_downloads(self):
        filename = sa.mim.candidate_filenames("SCUANAGRAFESTAT", "202425")[0]
        self.write(self.base / "mim" / "schools" / "202425" / filename)
        with patch.object(sa.mim, "download_official_file", side_effect=AssertionError("download should not run")):
            row = sa.inspect_mim_registry("SCUANAGRAFESTAT", "202425", fetch_supported=False)
        self.assertEqual(row.state, "cached")
        self.assertIsNone(row.source_url)

    def test_mim_ambiguous_cache_rejected(self):
        for name in sa.mim.candidate_filenames("SCUANAGRAFESTAT", "202425"):
            self.write(self.base / "mim" / "schools" / "202425" / name)
        row = sa.inspect_mim_registry("SCUANAGRAFESTAT", "202425", fetch_supported=False)
        self.assertEqual(row.state, "ambiguous")

    def test_mim_unsupported_school_year(self):
        row = sa.inspect_mim_registry("SCUANAGRAFESTAT", "202627", fetch_supported=True)
        self.assertEqual(row.state, "unsupported_snapshot")

    def test_mim_downloader_only_when_explicitly_enabled(self):
        filename = sa.mim.candidate_filenames("SCUANAGRAFESTAT", "202425")[0]
        path = self.base / "mim" / "schools" / "202425" / filename

        def fake_download(**kwargs):
            self.write(path)
            return path, "https://dati.istruzione.it/confirmed/file.csv"

        with patch.object(sa.mim, "download_official_file", side_effect=fake_download) as download:
            off = sa.inspect_mim_registry("SCUANAGRAFESTAT", "202425", fetch_supported=False)
            self.assertEqual(off.state, "missing")
            download.assert_not_called()
            on = sa.inspect_mim_registry("SCUANAGRAFESTAT", "202425", fetch_supported=True)
            download.assert_called_once()
            self.assertEqual(on.state, "downloaded")
            self.assertTrue(on.source_url.startswith("https://"))

    def test_mim_buildings_must_have_expected_prefix(self):
        folder = self.base / "mim" / "buildings" / "202425"
        self.write(folder / "some_random.csv")
        self.assertEqual(sa.inspect_mim_buildings("202425").state, "invalid")
        (folder / "some_random.csv").unlink()
        self.write(folder / "EDIANAGRAFESTA20242520250806.csv")
        self.assertEqual(sa.inspect_mim_buildings("202425").state, "cached")

    def test_health_uses_salute_not_health(self):
        self.write(self.base / "salute" / "farmacie" / "farmacie.csv")
        self.write(self.base / "salute" / "strutture_ospedaliere_2023" / "hospitals.csv")
        pharmacy, hospital = sa.inspect_health("2025-06-30", "2023")
        self.assertEqual((pharmacy.state, hospital.state), ("cached", "cached"))
        self.assertTrue(all("/salute/" in p for row in (pharmacy, hospital) for p in row.paths))

    def test_health_ambiguity_rejected(self):
        directory = self.base / "salute" / "farmacie"
        self.write(directory / "a.csv")
        self.write(directory / "b.csv")
        pharmacy, _ = sa.inspect_health("2025-06-30", "2023")
        self.assertEqual(pharmacy.state, "ambiguous")

    def test_nonempty_is_required(self):
        folder = self.base / "salute" / "farmacie"
        self.write(folder / "zero.csv", b"")
        pharmacy, _ = sa.inspect_health("2025-06-30", "2023")
        self.assertEqual(pharmacy.state, "invalid")

    def test_boundaries_dry_run_never_downloads(self):
        with patch.object(sa.download_istat_boundaries, "validate_region_files", return_value=(False, [])):
            with patch.object(sa.download_istat_boundaries, "ensure_region_boundaries", side_effect=AssertionError("no network")):
                row = sa.inspect_istat_boundaries("08", fetch_supported=False)
        self.assertEqual(row.state, "missing")

    def test_boundaries_download_only_when_requested(self):
        folder = self.base / "boundary"
        names = ["R08_21_WGS84" + suffix for suffix in (".shp", ".shx", ".dbf", ".prj")]
        sidecars = tuple(folder / name for name in names)

        def download(_):
            for path in sidecars:
                self.write(path)

        def validate(_):
            ok = all(p.exists() for p in sidecars)
            return ok, [] if ok else list(sidecars)

        with patch.object(sa.download_istat_boundaries, "expected_sidecars", return_value=sidecars):
            with patch.object(sa.download_istat_boundaries, "validate_region_files", side_effect=validate):
                with patch.object(sa.download_istat_boundaries, "ensure_region_boundaries", side_effect=download) as called:
                    row = sa.inspect_istat_boundaries("08", fetch_supported=True)
                    self.assertEqual(row.state, "downloaded")
                    self.assertEqual(len(row.sha256), 4)
                    called.assert_called_once()
                    row2 = sa.inspect_istat_boundaries("08", fetch_supported=False)
                    self.assertEqual(row2.state, "cached")

    def test_manifest_json_is_complete_and_explicitly_not_ready(self):
        report = sa.inspect_sources(
            self.ctx, census_year="2023", school_year="202425",
            building_year="202425", pharmacy_reference_date="2025-06-30",
            hospital_year="2023", fetch_supported=False,
        )
        self.assertFalse(report.ready)
        self.assertEqual(len(report.sources), 7)
        output = self.base / "out" / "manifest.json"
        sa.write_report(report, output)
        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(data["ready"])
        self.assertEqual(len(data["sources"]), 7)
        self.assertIn("notes", data)


# Additional tests keep the CLI contract and strict exit behavior explicit.
class SourceAcquisitionB5B1CliTests(unittest.TestCase):
    def test_cli_parses_name_and_readonly_default(self):
        args = sa.parse_args(["--city", "Parma"])
        self.assertEqual(args.city, "Parma")
        self.assertFalse(args.fetch_supported)
        self.assertFalse(args.strict)

    def test_strict_run_still_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "preflight.json"
            ctx = MunicipalityContext._from_values(
                code="034027", name="Parma", province_code="034",
                region_code="08", census_year="2023",
            )
            report = sa.AcquisitionReport(
                municipality_code="034027",
                municipality_name="Parma",
                census_year="2023",
                school_year="202425",
                building_year="202425",
                hospital_year="2023",
                pharmacy_reference_date="2025-06-30",
                fetch_supported=False,
                generated_at_utc="2026-10-08T00:00:00+00:00",
                sources=(sa.SourceRecord("istanza", "missing", "2023"),),
            )
            with patch.object(sa.MunicipalityContext, "resolve_name", return_value=ctx):
                with patch.object(sa, "inspect_sources", return_value=report) as probe:
                    result = sa.main(["--city", "Parma", "--strict", "--output-path", str(output)])
            self.assertEqual(result, 2)
            self.assertTrue(output.is_file())
            self.assertFalse(json.loads(output.read_text())["ready"])
            self.assertFalse(probe.call_args.kwargs["fetch_supported"])


if __name__ == "__main__":
    unittest.main()
