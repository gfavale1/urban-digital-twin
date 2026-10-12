"""No-data, mock-auditor tests for the multi-city V2 quality comparison."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.analysis_spec import TransportMode
from quality.audit_multicity_service_quality import (
    format_report, inspect_municipalities, parse_args, run,
)
from quality.audit_v2_service_boundary import paths_for_snapshot
from quality.preflight_real_service_scenario import CitySnapshot


class MulticityServiceQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _create_files(self, code):
        snapshot = CitySnapshot(code)
        for path in paths_for_snapshot(self.root, snapshot).items().values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic-test-input")

    def _report(self, code):
        return {
            "municipality_code": code, "mode": "walk",
            "services_total": 5, "services_routable": 3, "services_ineligible": 2,
            "primary_routing_reason": {"eligible": 3, "location_not_validated": 2},
            "geography": {"inside_census_footprint": 4, "coordinates_missing": 1},
            "service_types": {
                "pharmacy": {"total": 4, "routable": 2},
                "primary_school": {"total": 1, "routable": 1},
            },
        }

    def test_all_missing_are_not_counted_as_zero(self):
        with patch("quality.audit_multicity_service_quality.inspect_service_quality") as inspector:
            result = inspect_municipalities(["034027", "077014"], root=self.root)
        inspector.assert_not_called()
        self.assertEqual(result["audited_cities"], 0)
        self.assertEqual(result["missing_cities"], 2)
        self.assertFalse(result["all_cities_audited"])
        self.assertTrue(result["cities"]["034027"]["missing_inputs"])
        self.assertIn("NOT AUDITED", format_report(result))

    def test_mixed_presence_is_explicit_and_comparable(self):
        self._create_files("034027")
        with patch("quality.audit_multicity_service_quality.inspect_service_quality", side_effect=lambda snap, root: self._report(snap.municipality_code)) as inspector:
            result = inspect_municipalities(["034027", "077014"], root=self.root)
        self.assertEqual(inspector.call_count, 1)
        self.assertEqual(result["audited_cities"], 1)
        output = format_report(result)
        self.assertIn("034027: AUDITED", output)
        self.assertIn("077014: NOT AUDITED", output)
        self.assertIn("2/4", output)
        self.assertIn("N/A", output)

    def test_both_present_are_audited_independently(self):
        self._create_files("034027")
        self._create_files("077014")
        with patch("quality.audit_multicity_service_quality.inspect_service_quality", side_effect=lambda snap, root: self._report(snap.municipality_code)) as inspector:
            result = inspect_municipalities(["034027", "077014"], root=self.root)
        self.assertEqual(inspector.call_count, 2)
        self.assertTrue(result["all_cities_audited"])

    def test_corrupt_present_city_fails_closed(self):
        self._create_files("034027")
        with patch("quality.audit_multicity_service_quality.inspect_service_quality", side_effect=ValueError("bad graph checksum")):
            result = inspect_municipalities(["034027", "077014"], root=self.root)
        self.assertEqual(result["invalid_cities"], 1)
        self.assertEqual(result["missing_cities"], 1)
        self.assertFalse(result["all_cities_audited"])
        self.assertIn("bad graph checksum", format_report(result))

    def test_non_reconciling_counts_rejected(self):
        self._create_files("034027")
        report = self._report("034027")
        report["services_ineligible"] = 5
        with patch("quality.audit_multicity_service_quality.inspect_service_quality", return_value=report):
            result = inspect_municipalities(["034027"], root=self.root)
        self.assertEqual(result["invalid_cities"], 1)
        self.assertIn("Invalid routing totals", format_report(result))

    def test_wrong_city_from_inspector_rejected(self):
        self._create_files("034027")
        with patch("quality.audit_multicity_service_quality.inspect_service_quality", return_value=self._report("016024")):
            result = inspect_municipalities(["034027"], root=self.root)
        self.assertIn("wrong city", format_report(result))

    def test_type_totals_must_reconcile(self):
        self._create_files("034027")
        report = self._report("034027")
        report["service_types"]["pharmacy"]["routable"] = 5
        with patch("quality.audit_multicity_service_quality.inspect_service_quality", return_value=report):
            result = inspect_municipalities(["034027"], root=self.root)
        self.assertIn("Service-type totals", format_report(result))

    def test_reject_duplicate_and_invalid_codes(self):
        for codes in ([], ["034027", "034027"], ["Parma"], ["034027", "0"]):
            with self.subTest(codes=codes):
                with self.assertRaises(ValueError):
                    inspect_municipalities(codes, root=self.root)

    def test_requires_all_exits_nonzero_when_inputs_missing(self):
        with patch("quality.audit_multicity_service_quality.ROOT", self.root):
            with self.assertRaises(SystemExit) as err:
                run(["--root", str(self.root), "--municipality-codes", "034027", "--require-all"])
        self.assertEqual(err.exception.code, 2)

    def test_invalid_present_snapshot_causes_nonzero_exit(self):
        self._create_files("034027")
        with patch("quality.audit_multicity_service_quality.inspect_service_quality", side_effect=ValueError("checksum mismatch")):
            with self.assertRaises(SystemExit) as caught:
                run(["--root", str(self.root), "--municipality-codes", "034027"])
        self.assertEqual(caught.exception.code, 1)

    def test_no_file_write_on_missing_or_inspected_input(self):
        self._create_files("034027")
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch("quality.audit_multicity_service_quality.inspect_service_quality", return_value=self._report("034027")):
            inspect_municipalities(["034027", "077014"], root=self.root)
        after = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_explicit_mode_and_snapshot_parameters(self):
        parsed = parse_args(["--mode", "drive", "--municipality-codes", "077014"])
        self.assertEqual(parsed.mode, TransportMode.DRIVE.value)
        self.assertEqual(parsed.municipality_codes, ["077014"])


if __name__ == "__main__":
    unittest.main()
