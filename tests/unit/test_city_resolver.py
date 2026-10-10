from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core.municipality import MunicipalityContext, normalize_city_name


class CityResolverB5ATests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "municipality_dimension_2023.parquet"
        self.path.touch()
        self.dimension = pd.DataFrame(
            [
                {"istat_code": "034027", "name": "Parma", "province_code": "034", "region_code": "08"},
                {"istat_code": "063049", "name": "Napoli", "province_code": "063", "region_code": "15"},
                {"istat_code": "066049", "name": "L'Aquila", "province_code": "066", "region_code": "13"},
                {"istat_code": "001099", "name": "San Pietro", "province_code": "001", "region_code": "01"},
                {"istat_code": "002099", "name": "San Pietro", "province_code": "002", "region_code": "02"},
            ]
        )

    def resolve(self, name, **kwargs):
        with patch("core.municipality.municipality_dimension_path", return_value=self.path):
            with patch("core.municipality.pd.read_parquet", return_value=self.dimension.copy()):
                return MunicipalityContext.resolve_name(name, **kwargs)

    def test_normalization_is_deterministic(self):
        self.assertEqual(normalize_city_name("  L’AQUILA "), "l aquila")
        self.assertEqual(normalize_city_name("Città di Castello"), "citta di castello")
        with self.assertRaises(ValueError):
            normalize_city_name("  ")

    def test_resolves_names_without_database(self):
        for name, code in [("PARMA", "034027"), ("naPoLi", "063049"), (" L’AQUILA ", "066049")]:
            with self.subTest(name=name):
                ctx = self.resolve(name)
                self.assertEqual(ctx.code, code)
                self.assertEqual(ctx.procom, int(code))
                self.assertEqual(ctx.census_year, "2023")

    def test_ambiguous_name_lists_candidates(self):
        with self.assertRaisesRegex(ValueError, "Nome comunale ambiguo") as failure:
            self.resolve("San Pietro")
        message = str(failure.exception)
        self.assertIn("001099", message)
        self.assertIn("002099", message)
        self.assertIn("--province-code", message)

    def test_ambiguous_name_disambiguated_by_province(self):
        ctx = self.resolve("san pietro", province_code="002")
        self.assertEqual(ctx.code, "002099")

    def test_ambiguous_name_disambiguated_by_region(self):
        ctx = self.resolve("san pietro", region_code="01")
        self.assertEqual(ctx.code, "001099")

    def test_explicit_unknown_name_does_not_fuzzy_match(self):
        with self.assertRaisesRegex(ValueError, "non trovato"):
            self.resolve("Parmaa")

    def test_invalid_filter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "province_code"):
            self.resolve("Parma", province_code="PR")

    def test_filter_that_does_not_match_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non trovato"):
            self.resolve("Parma", province_code="063")

    def test_corrupt_dimension_duplicate_code_is_rejected(self):
        self.dimension = pd.concat(
            [self.dimension, self.dimension.iloc[[0]]], ignore_index=True
        )
        with self.assertRaisesRegex(RuntimeError, "codice comunale duplicato"):
            self.resolve("Parma")

    def test_missing_dimension_does_not_fall_back_to_postgis(self):
        missing = Path(self.tmp.name) / "missing.parquet"
        with patch("core.municipality.municipality_dimension_path", return_value=missing):
            with patch("core.municipality.get_engine") as get_engine:
                with self.assertRaisesRegex(FileNotFoundError, "B5B"):
                    MunicipalityContext.resolve_name("Parma")
                get_engine.assert_not_called()

    def test_legacy_code_resolver_remains_unchanged(self):
        with patch.object(MunicipalityContext, "_resolve_from_dimension") as resolver:
            ctx = MunicipalityContext._from_values(
                code="034027", name="Parma", province_code="034",
                region_code="08", census_year="2023"
            )
            resolver.return_value = ctx
            result = MunicipalityContext.resolve("034027", "2023")
            self.assertEqual(result.code, "034027")
            resolver.assert_called_once_with(code="034027", census_year="2023")


class CityResolverPipelineCliB5ATests(unittest.TestCase):
    def test_cli_accepts_either_city_or_code_but_not_both(self):
        from pipeline import parse_args

        with patch.object(sys, "argv", ["pipeline.py", "--city", "Parma"]):
            args = parse_args()
            self.assertEqual(args.city, "Parma")
            self.assertIsNone(args.municipality_code)

        with patch.object(sys, "argv", ["pipeline.py", "--municipality-code", "034027"]):
            args = parse_args()
            self.assertEqual(args.municipality_code, "034027")
            self.assertIsNone(args.city)

        with patch.object(sys, "argv", ["pipeline.py", "--city", "Parma", "--municipality-code", "034027"]):
            with patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_filter_requires_city(self):
        from pipeline import parse_args
        with patch.object(sys, "argv", ["pipeline.py", "--municipality-code", "034027", "--region-code", "08"]):
            with patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args()


if __name__ == "__main__":
    unittest.main()
