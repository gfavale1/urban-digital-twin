import unittest

import pandas as pd

from ingestion.drive_speed_model import (
    assign_free_flow_travel_times,
    parse_maxspeed_kph,
)


class DriveSpeedModelV2Tests(unittest.TestCase):
    def test_parse_numeric_and_mph_maxspeed(self):
        self.assertEqual(parse_maxspeed_kph("50"), 50.0)
        self.assertAlmostEqual(
            parse_maxspeed_kph("30 mph"),
            48.28032,
            places=6,
        )

    def test_multiple_limits_use_conservative_minimum(self):
        self.assertEqual(parse_maxspeed_kph("70;50"), 50.0)
        self.assertEqual(parse_maxspeed_kph(["90", "70"]), 70.0)

    def test_non_numeric_osm_codes_are_left_unresolved(self):
        self.assertIsNone(parse_maxspeed_kph("IT:urban"))
        self.assertIsNone(parse_maxspeed_kph("none"))

    def test_same_highway_class_uses_observed_median(self):
        edges = pd.DataFrame(
            {
                "length_m": [100.0, 100.0, 100.0],
                "road_type": ["residential", "residential", "residential"],
                "maxspeed": ["30", "50", None],
            }
        )

        result, summary = assign_free_flow_travel_times(edges)

        self.assertEqual(result.loc[2, "speed_kph"], 40.0)
        self.assertEqual(
            result.loc[2, "speed_source"],
            "highway_median_osm_maxspeed",
        )
        self.assertEqual(summary.inferred_highway_median, 1)

    def test_unknown_highway_class_uses_graph_global_median(self):
        edges = pd.DataFrame(
            {
                "length_m": [100.0, 100.0, 100.0],
                "road_type": ["primary", "secondary", "service"],
                "maxspeed": ["70", "50", None],
            }
        )

        result, summary = assign_free_flow_travel_times(edges)

        self.assertEqual(result.loc[2, "speed_kph"], 60.0)
        self.assertEqual(
            result.loc[2, "speed_source"],
            "global_median_osm_maxspeed",
        )
        self.assertEqual(summary.inferred_global_median, 1)

    def test_explicit_fallback_is_required_when_graph_has_no_observed_speed(self):
        edges = pd.DataFrame(
            {
                "length_m": [100.0, 200.0],
                "road_type": ["residential", "service"],
                "maxspeed": [None, "IT:urban"],
            }
        )

        with self.assertRaises(RuntimeError):
            assign_free_flow_travel_times(edges)

        result, summary = assign_free_flow_travel_times(
            edges,
            fallback_speed_kph=30.0,
        )

        self.assertTrue((result["speed_kph"] == 30.0).all())
        self.assertTrue((result["speed_source"] == "explicit_fallback").all())
        self.assertEqual(summary.explicit_fallback, 2)

    def test_free_flow_travel_time_uses_speed_kph(self):
        edges = pd.DataFrame(
            {
                "length_m": [1000.0],
                "road_type": ["primary"],
                "maxspeed": ["60"],
            }
        )

        result, _ = assign_free_flow_travel_times(edges)
        self.assertAlmostEqual(
            result.loc[0, "free_flow_travel_time_s"],
            60.0,
            places=10,
        )

    def test_invalid_lengths_are_rejected(self):
        edges = pd.DataFrame(
            {
                "length_m": [0.0],
                "road_type": ["residential"],
                "maxspeed": ["30"],
            }
        )

        with self.assertRaises(ValueError):
            assign_free_flow_travel_times(edges)


if __name__ == "__main__":
    unittest.main()
