"""B6A2 isolated evidence matching tests. No network or live datasets."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ingestion.anncsu import sha256_file
from matching import anncsu_service_match as sut


class AnncsuServiceMatchB6A2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.municipality = "034027"
        self.snapshot = "20260915"
        self.index_path = self.root / "processed/anncsu/034027/addresses_20260915.csv"
        self.manifest_path = self.root / "features/anncsu/034027/manifest_20260915.json"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.anncsu_rows = [self.anncsu("VIA ROMA", "12", access="A1")]
        self.frames = {
            "school_building": pd.DataFrame([{"building_code": "E1", "official_building_address": "Via Roma 12", "building_municipality_name": "Parma", "geocoding_status": "unresolved", "latitude": None, "longitude": None}]),
            "school_registry_record": pd.DataFrame([{"school_code": "S1", "school_address": "Via Roma 12", "municipality_code": "034027", "municipality_name": "Parma"}]),
            "pharmacy": pd.DataFrame([{"service_site_id": "P1", "address": "Via Roma 12", "municipality_code": "034027", "latitude": 44.80, "longitude": 10.32, "coordinate_resolution": "source", "usable_for_accessibility": True}]),
            "hospital": pd.DataFrame([{"service_site_id": "H1", "address": "Via Roma 12", "municipality_code": "034027", "latitude": None, "longitude": None, "usable_for_accessibility": False}]),
        }
        self.populate()

    def anncsu(self, street, number, *, exp="", access="A1", lon="10.328", lat="44.801", quality="valid", method="1"):
        return dict(
            municipality_code="034027", street_normalized=street, house_number_normalized=number,
            exponent_normalized=exp, longitude=lon, latitude=lat, coordinate_status=quality,
            method_code=method, source_snapshot_date="20260915", street_id="ST1", access_id=access,
            source_row_number="2",
        )

    def populate(self):
        with self.index_path.open("w", encoding="utf-8", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=list(self.anncsu_rows[0]))
            writer.writeheader()
            writer.writerows(self.anncsu_rows)
        self.manifest_path.write_text(json.dumps({
            "municipality_code": self.municipality, "snapshot_date": self.snapshot,
            "output_csv_sha256": sha256_file(self.index_path), "archive_sha256": "f" * 64,
        }), encoding="utf-8")
        self.paths = {}
        for kind, parent, template, _, _ in sut.INPUTS:
            vals = dict(building_year="202425", school_year="202425", pharmacy_label="20250630", hospital_year="2023")
            path = self.root / "processed" / parent / self.municipality / template.format(**vals)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(kind, encoding="utf-8")
            self.paths[path.as_posix()] = self.frames[kind]

    def read_fake(self, path):
        return self.paths[Path(path).as_posix()].copy()

    def run_evidence(self):
        with patch.object(sut.pd, "read_parquet", side_effect=self.read_fake):
            return sut.build_evidence(self.root, self.municipality, snapshot=self.snapshot)

    def test_three_civic_formats(self):
        for raw, exp in [("Via Roma 12", ""), ("Via Roma, 12/A", "A"), ("Via Roma 12A", "A"), ("Via Roma 12 BIS", "BIS")]:
            with self.subTest(raw=raw):
                self.assertEqual(sut.parse_service_address(raw), ("VIA ROMA", "12", exp, "parsed"))

    def test_street_number_in_name_does_not_break(self):
        self.assertEqual(sut.parse_service_address("Via XX Settembre 15"), ("VIA XX SETTEMBRE", "15", "", "parsed"))

    def test_missing_civic_fails_closed(self):
        self.assertEqual(sut.parse_service_address("Via Roma")[-1], "missing_or_unparseable_civic")
        self.assertEqual(sut.parse_service_address("Via 25 Aprile")[-1], "missing_or_unparseable_civic")
        self.assertEqual(sut.parse_service_address(None)[-1], "missing_address")

    def test_ranges_and_years_rejected(self):
        self.assertEqual(sut.parse_service_address("Via Roma 12/14")[-1], "unparseable_address")
        self.assertNotEqual(sut.parse_service_address("Via Risorgimento 1860")[-1], "parsed")

    def test_b6a1_index_and_manifest_verified(self):
        rows, report = self.run_evidence()
        self.assertEqual(len(rows), 4)
        self.assertEqual(report["candidate_exact_unique_count"], 4)
        self.assertEqual(report["candidate_without_existing_coordinates"], 3)
        self.assertTrue(all(r["match_status"] == "exact_single_access_candidate" for r in rows))
        self.assertTrue(all(r["coordinate_accepted_for_accessibility"] is False for r in rows))

    def test_existing_coordinate_kept_and_distance_reported(self):
        rows, _ = self.run_evidence()
        p = next(r for r in rows if r["entity_type"] == "pharmacy")
        self.assertEqual(p["existing_latitude"], 44.8)
        self.assertEqual(p["existing_longitude"], 10.32)
        self.assertGreater(p["distance_from_existing_m"], 0)
        self.assertEqual(p["anncsu_latitude"], "44.801")

    def test_no_exact_address_fails_closed(self):
        self.frames["pharmacy"].loc[0, "address"] = "Via Differente 12"
        rows, _ = self.run_evidence()
        p = next(r for r in rows if r["entity_type"] == "pharmacy")
        self.assertEqual(p["match_status"], "no_exact_address_match")
        self.assertEqual(p["anncsu_longitude"], "")

    def test_exponent_not_ignored(self):
        self.frames["pharmacy"].loc[0, "address"] = "Via Roma 12/A"
        rows, _ = self.run_evidence()
        p = next(r for r in rows if r["entity_type"] == "pharmacy")
        self.assertEqual(p["match_status"], "no_exact_address_match")

    def test_multiple_different_positions_review(self):
        self.anncsu_rows.append(self.anncsu("VIA ROMA", "12", access="A2", lon="10.329"))
        self.populate()
        rows, report = self.run_evidence()
        self.assertEqual(report["candidate_exact_unique_count"], 0)
        self.assertTrue(all(r["match_status"] == "ambiguous_multiple_positions" for r in rows))

    def test_duplicate_same_position_still_review(self):
        self.anncsu_rows.append(self.anncsu("VIA ROMA", "12", access="A2"))
        self.populate()
        rows, _ = self.run_evidence()
        self.assertTrue(all(r["match_status"] == "multiple_accesses_review" for r in rows))

    def test_invalid_anncsu_coordinates_not_promoted(self):
        self.anncsu_rows[0].update(coordinate_status="invalid", latitude="", longitude="")
        self.populate()
        rows, _ = self.run_evidence()
        self.assertTrue(all(r["match_status"] == "matched_without_valid_coordinate" for r in rows))

    def test_outside_building_marked_excluded(self):
        self.frames["school_building"].loc[0, "building_municipality_name"] = "Sorbolo"
        rows, _ = self.run_evidence()
        building = next(r for r in rows if r["entity_type"] == "school_building")
        self.assertEqual(building["match_status"], "outside_target_municipality")

    def test_wrong_source_municipality_blocked(self):
        self.frames["hospital"].loc[0, "municipality_code"] = "033001"
        with self.assertRaisesRegex(ValueError, "Unexpected municipality code"):
            self.run_evidence()

    def test_modified_anncsu_index_hash_blocked(self):
        self.index_path.write_text("tampering", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_evidence()

    def test_duplicate_access_id_blocked(self):
        self.anncsu_rows.append(self.anncsu("VIA ROMA", "13", access="A1"))
        self.populate()
        with self.assertRaisesRegex(ValueError, "Duplicate ANNCSU access_id"):
            self.run_evidence()

    def test_wrong_anncsu_snapshot_blocked(self):
        self.anncsu_rows[0]["source_snapshot_date"] = "20250915"
        self.populate()
        with self.assertRaisesRegex(ValueError, "wrong municipality/snapshot"):
            self.run_evidence()

    def test_no_read_parquet_modification_on_save(self):
        rows, report = self.run_evidence()
        path, jpath, cached = sut.save_evidence(self.root, self.municipality, self.snapshot, rows, report)
        self.assertFalse(cached)
        self.assertTrue(path.is_file())
        self.assertTrue(jpath.is_file())
        with path.open(encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)
        self.assertTrue(sut.save_evidence(self.root, self.municipality, self.snapshot, rows, report)[-1])

    def test_changed_inputs_refuse_overwrite(self):
        rows, report = self.run_evidence()
        sut.save_evidence(self.root, self.municipality, self.snapshot, rows, report)
        report2 = dict(report, input_sha256={**report["input_sha256"], "pharmacy": "0" * 64})
        with self.assertRaises(FileExistsError):
            sut.save_evidence(self.root, self.municipality, self.snapshot, rows, report2)

    def test_partial_output_refuse_overwrite(self):
        out = self.root / "features/anncsu/034027/service_address_matches_20260915.csv"
        out.write_text("incomplete", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "Incomplete"):
            sut.save_evidence(self.root, self.municipality, self.snapshot, [], {})

    def test_required_input_absent(self):
        (self.root / "processed/salute/034027/hospital_sites_2023.parquet").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_evidence()

    def test_no_coordinate_mutation_from_record(self):
        original = dict(self.frames["pharmacy"].iloc[0])
        rows, _ = self.run_evidence()
        self.assertEqual(dict(self.frames["pharmacy"].iloc[0]), original)
        self.assertFalse(next(r for r in rows if r["entity_type"] == "pharmacy")["coordinate_accepted_for_accessibility"])

    def test_city_and_municipality_flags_exclusive(self):
        with self.assertRaises(SystemExit):
            sut.parse_args(["--city", "Parma", "--municipality-code", "034027"])


if __name__ == "__main__":
    unittest.main()
