from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

from core.schema_v2 import ORIGIN_V2
from transformation.build_population_network_origins import (
    add_origin_v2_fields,
    validate_demographic_conservation,
)


ROOT = Path(__file__).resolve().parents[2]


def load_istat_module():
    """Load the script module without performing any network/file acquisition."""
    stub = types.ModuleType("download_istat_boundaries")
    stub.ensure_region_boundaries = lambda region_code: None
    sys.modules.setdefault("download_istat_boundaries", stub)

    path = ROOT / "src" / "ingestion" / "istat.py"
    spec = importlib.util.spec_from_file_location("phase3_istat", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DemandSchemaV2Tests(unittest.TestCase):
    def test_istat_preserves_five_year_age_bands(self):
        istat = load_istat_module()

        row = {
            "P14": 11,
            "P15": 12,
            "P16": 13,
            "P17": 14,
        }
        for index in range(18, 30):
            row[f"P{index}"] = 1

        result = istat.build_demographic_features(pd.DataFrame([row]))

        self.assertEqual(result.loc[0, "age_lt_5"], 11)
        self.assertEqual(result.loc[0, "age_5_9"], 12)
        self.assertEqual(result.loc[0, "age_10_14"], 13)
        self.assertEqual(result.loc[0, "age_15_19"], 14)
        self.assertEqual(result.loc[0, "age_0_14"], 36)

    def test_origin_v2_aliases_keep_legacy_values(self):
        origins = pd.DataFrame(
            {
                "census_section_code": ["S1", "S1"],
                "network_node_id": ["100", "101"],
                "assignment_method": ["equal_internal_nodes"] * 2,
                "assigned_population": [60.0, 40.0],
                "reference_date": ["2023-12-31"] * 2,
                "assigned_age_lt_5": [6.0, 4.0],
                "assigned_age_5_9": [12.0, 8.0],
                "assigned_age_10_14": [9.0, 6.0],
                "assigned_age_15_19": [7.2, 4.8],
                "geometry": [None, None],
            }
        )

        result = add_origin_v2_fields(
            origins,
            municipality_code="034027",
            census_year="2023",
        )

        ORIGIN_V2.validate_columns(result.columns)
        self.assertFalse(result["origin_id"].duplicated().any())
        self.assertEqual(result["municipality_code"].unique().tolist(), ["034027"])
        self.assertEqual(result["geography_version"].unique().tolist(), ["BT2021"])
        self.assertEqual(result["population_total"].sum(), 100.0)
        self.assertEqual(result["population_age_5_9_proxy"].sum(), 20.0)
        self.assertTrue(
            (result["origin_method"] == result["assignment_method"]).all()
        )

    def test_demographic_conservation_passes(self):
        observed = pd.DataFrame(
            {
                "population": [100.0],
                "age_lt_5": [10.0],
                "age_5_9": [20.0],
                "age_10_14": [15.0],
                "age_15_19": [12.0],
            }
        )
        origins = pd.DataFrame(
            {
                "assigned_population": [60.0, 40.0],
                "assigned_age_lt_5": [6.0, 4.0],
                "assigned_age_5_9": [12.0, 8.0],
                "assigned_age_10_14": [9.0, 6.0],
                "assigned_age_15_19": [7.2, 4.8],
            }
        )

        checks = validate_demographic_conservation(observed, origins)
        self.assertAlmostEqual(checks["population"]["difference"], 0.0)
        self.assertAlmostEqual(checks["age_15_19"]["difference"], 0.0)

    def test_demographic_conservation_detects_loss(self):
        observed = pd.DataFrame({"population": [100.0], "age_5_9": [20.0]})
        origins = pd.DataFrame(
            {
                "assigned_population": [100.0],
                "assigned_age_5_9": [19.0],
            }
        )

        with self.assertRaises(RuntimeError):
            validate_demographic_conservation(observed, origins)


if __name__ == "__main__":
    unittest.main()
