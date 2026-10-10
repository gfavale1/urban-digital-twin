"""B6A1 synthetic ZIP tests: no network and no live legacy artifacts."""
import csv
import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from ingestion import anncsu

HEADER = [
    "CODICE_COMUNE", "CODICE_ISTAT", "PROGRESSIVO_NAZIONALE", "CODICE_COMUNALE",
    "ODONIMO", "LOCALITA'", "DIZIONE_LINGUA1", "DIZIONE_LINGUA2", "PROGRESSIVO_ACCESSO",
    "CODICE_COMUNALE_ACCESSO", "CIVICO", "ESPONENTE", "SPECIFICITA", "METRICO",
    "PROGRESSIVO_SNC", "COORD_X_COMUNE", "COORD_Y_COMUNE", "QUOTA", "METODO",
]


def row(code="034027", **kwargs):
    data = dict.fromkeys(HEADER, "")
    data.update({
        "CODICE_COMUNE": "G337", "CODICE_ISTAT": code,
        "PROGRESSIVO_NAZIONALE": "795125", "PROGRESSIVO_ACCESSO": "17116393",
        "ODONIMO": "PIAZZA DELLA LIBERTA'", "CIVICO": "1", "ESPONENTE": "A",
        "COORD_X_COMUNE": "10,328412", "COORD_Y_COMUNE": "44,801305", "METODO": "1",
    })
    data.update(kwargs)
    return data


def make_zip(path, rows, *, member="INDIR_EMIL_20260915.csv", header=HEADER, extra=False):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=header, delimiter=";", extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with ZipFile(path, "w") as archive:
        archive.writestr(member, buffer.getvalue().encode("utf-8-sig"))
        if extra:
            archive.writestr("extra.txt", "unknown")


class AnncsuB6A1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "indirizzarioEmilia-romagna20260915.zip"
        self.outputs = self.root / "data"

    def ingest(self, rows):
        make_zip(self.archive, rows)
        return anncsu.import_municipality(self.archive, "034027", output_root=self.outputs)

    def test_semicolon_real_header_and_decimal_commas(self):
        csv_path, _, report, cached = self.ingest([row()])
        self.assertFalse(cached)
        self.assertEqual(report["municipal_rows_selected"], 1)
        with csv_path.open(encoding="utf-8", newline="") as file:
            item = next(csv.DictReader(file))
        self.assertEqual(item["street_normalized"], "PIAZZA DELLA LIBERTA")
        self.assertEqual(item["house_number_normalized"], "1")
        self.assertEqual(item["exponent_normalized"], "A")
        self.assertEqual(item["longitude"], "10.328412")
        self.assertEqual(item["latitude"], "44.801305")
        self.assertEqual(item["method_code"], "1")

    def test_filters_municipality_and_preserves_zero_prefix(self):
        csv_path, _, report, _ = self.ingest([row(code="033001"), row(), row(code="077014")])
        self.assertEqual(report["regional_rows_scanned"], 3)
        self.assertEqual(report["municipal_rows_selected"], 1)
        self.assertNotIn("033001", csv_path.read_text(encoding="utf-8"))
        self.assertEqual(csv_path.read_text(encoding="utf-8").count("034027"), 1)

    def test_wrong_region_has_no_output(self):
        make_zip(self.archive, [row(code="033001")])
        with self.assertRaisesRegex(ValueError, "No ANNCSU records"):
            anncsu.import_municipality(self.archive, "034027", output_root=self.outputs)
        self.assertFalse((self.outputs / "processed" / "anncsu" / "034027" / "addresses_20260915.csv").exists())

    def test_invalid_coordinates_preserve_rows(self):
        _, _, report, _ = self.ingest([
            row(PROGRESSIVO_ACCESSO="1", COORD_X_COMUNE="", COORD_Y_COMUNE=""),
            row(PROGRESSIVO_ACCESSO="2", COORD_X_COMUNE="0", COORD_Y_COMUNE="0"),
            row(PROGRESSIVO_ACCESSO="3", COORD_X_COMUNE="garbage"),
        ])
        self.assertEqual(report["coordinate_status_counts"], {"invalid": 2, "missing": 1})

    def test_different_coordinates_for_same_address_are_ambiguous(self):
        _, _, report, _ = self.ingest([
            row(PROGRESSIVO_ACCESSO="1"),
            row(PROGRESSIVO_ACCESSO="2", COORD_X_COMUNE="10,329"),
            row(PROGRESSIVO_ACCESSO="3", CIVICO="2", ESPONENTE=""),
        ])
        self.assertEqual(report["ambiguous_coordinate_keys"], 1)
        self.assertEqual(report["municipal_unique_street_civico_keys"], 2)

    def test_identical_coordinates_are_not_ambiguous(self):
        _, _, report, _ = self.ingest([row(PROGRESSIVO_ACCESSO="1"), row(PROGRESSIVO_ACCESSO="2")])
        self.assertEqual(report["ambiguous_coordinate_keys"], 0)

    def test_manifest_cache_idempotence(self):
        first = self.ingest([row()])
        second = anncsu.import_municipality(self.archive, "034027", output_root=self.outputs)
        self.assertEqual(first[2], second[2])
        self.assertTrue(second[3])

    def test_changed_archive_does_not_overwrite(self):
        self.ingest([row()])
        make_zip(self.archive, [row(CIVICO="10")])
        with self.assertRaises(FileExistsError):
            anncsu.import_municipality(self.archive, "034027", output_root=self.outputs)

    def test_expected_archive_checksum_guard(self):
        make_zip(self.archive, [row()])
        actual_hash = anncsu.sha256_file(self.archive)
        _, _, _, cached = anncsu.import_municipality(
            self.archive, "034027", output_root=self.outputs, expected_sha256=actual_hash
        )
        self.assertFalse(cached)
        with self.assertRaisesRegex(ValueError, "does not match"):
            anncsu.import_municipality(
                self.archive, "034027", output_root=self.outputs, expected_sha256="a" * 64
            )

    def test_modified_output_does_not_get_overwritten(self):
        csv_path, _, _, _ = self.ingest([row()])
        csv_path.write_text("tampered", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            anncsu.import_municipality(self.archive, "034027", output_root=self.outputs)

    def test_missing_column_rejected(self):
        make_zip(self.archive, [row()], header=[v for v in HEADER if v != "METODO"])
        with self.assertRaisesRegex(ValueError, "columns missing"):
            anncsu.archive_metadata(self.archive)

    def test_ambiguous_or_nested_zip_members_rejected(self):
        make_zip(self.archive, [row()], extra=True)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            anncsu.archive_metadata(self.archive)
        make_zip(self.archive, [row()], member="../INDIR_EMIL_20260915.csv")
        with self.assertRaisesRegex(ValueError, "path-traversing"):
            anncsu.archive_metadata(self.archive)

    def test_snapshot_date_mismatch_rejected(self):
        make_zip(self.archive, [row()], member="INDIR_EMIL_20260914.csv")
        with self.assertRaisesRegex(ValueError, "date differs"):
            anncsu.archive_metadata(self.archive)

    def test_conservative_civic_normalization(self):
        self.assertEqual(anncsu.normalize_house_number("SNC"), "")
        self.assertEqual(anncsu.normalize_house_number("12B"), "")
        self.assertEqual(anncsu.normalize_house_number("002"), "2")
        self.assertEqual(anncsu.normalize_exponent("bis"), "BIS")

    def test_malformed_other_city_row_is_not_ignored(self):
        make_zip(self.archive, [row(code="033001"), row()])
        with ZipFile(self.archive) as z:
            source = z.read("INDIR_EMIL_20260915.csv").decode("utf-8-sig")
        source = source.replace("033001;", "033001;UNEXPECTED;", 1)
        with ZipFile(self.archive, "w") as z:
            z.writestr("INDIR_EMIL_20260915.csv", source)
        with self.assertRaisesRegex(ValueError, "Malformed"):
            anncsu.import_municipality(self.archive, "034027", output_root=self.outputs)

    def test_mutually_exclusive_location(self):
        with self.assertRaises(SystemExit):
            anncsu.parse_args(["--archive", str(self.archive), "--city", "Parma", "--municipality-code", "034027"])

    def test_province_requires_city(self):
        with self.assertRaises(SystemExit):
            anncsu.parse_args(["--archive", str(self.archive), "--municipality-code", "034027", "--province-code", "034"])


if __name__ == "__main__":
    unittest.main()
