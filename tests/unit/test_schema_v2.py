import unittest

from core.analysis_spec import PopulationSelector
from core.schema_v2 import (
    ORIGIN_V2,
    ReachabilityStatus,
    opportunity_count_column,
    population_column_for_selector,
    validate_reachability,
)


class SchemaV2Tests(unittest.TestCase):
    def test_origin_contract_accepts_required_columns(self):
        ORIGIN_V2.validate_columns(ORIGIN_V2.required)

    def test_origin_contract_rejects_missing_columns(self):
        columns = set(ORIGIN_V2.required) - {"population_total"}
        with self.assertRaises(ValueError):
            ORIGIN_V2.validate_columns(columns)

    def test_metric_column_names_are_unambiguous(self):
        self.assertEqual(
            opportunity_count_column(15), "opportunity_count_within_15_min"
        )
        self.assertEqual(
            population_column_for_selector(PopulationSelector.AGE_5_9_PROXY),
            "population_age_5_9_proxy",
        )

    def test_unreachable_uses_null_plus_reason(self):
        validate_reachability(
            nearest_service_time_min=None,
            status=ReachabilityStatus.NO_PATH,
        )
        with self.assertRaises(ValueError):
            validate_reachability(
                nearest_service_time_min=999.0,
                status=ReachabilityStatus.NO_PATH,
            )


if __name__ == "__main__":
    unittest.main()
