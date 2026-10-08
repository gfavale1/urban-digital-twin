import unittest
from types import SimpleNamespace

from core.analysis_spec import (
    AnalysisSpec,
    PopulationSelector,
    ServiceType,
    TransportMode,
)


class AnalysisSpecTests(unittest.TestCase):
    def test_default_spec_is_deterministic(self):
        a = AnalysisSpec.default_for_city(
            "Napoli", analysis_date="2026-10-06", municipality_code="063049"
        )
        b = AnalysisSpec.default_for_city(
            "Napoli", analysis_date="2026-10-06", municipality_code="063049"
        )
        self.assertEqual(a.spec_hash, b.spec_hash)
        self.assertEqual(a.max_threshold_min(TransportMode.WALK), 15)
        self.assertEqual(a.max_threshold_min(TransportMode.DRIVE), 45)

    def test_school_population_selectors_are_explicit_proxies(self):
        spec = AnalysisSpec.default_for_city("Napoli", analysis_date="2026-10-06")
        selectors = {s.service_type: s.population_selector for s in spec.services}
        self.assertEqual(
            selectors[ServiceType.PRIMARY_SCHOOL], PopulationSelector.AGE_5_9_PROXY
        )
        self.assertEqual(
            selectors[ServiceType.LOWER_SECONDARY_SCHOOL],
            PopulationSelector.AGE_10_14_PROXY,
        )

    def test_legacy_bridge_preserves_old_thresholds_and_speed(self):
        legacy = SimpleNamespace(
            walking_speed_m_s=1.4,
            accessibility_thresholds_min=(10, 15, 20),
        )
        spec = AnalysisSpec.from_legacy_pipeline_config(
            city_name="Parma",
            municipality_code="034027",
            analysis_date="2026-10-06",
            config=legacy,
        )
        self.assertEqual(spec.walking_speed_m_s, 1.4)
        self.assertEqual(spec.max_threshold_min(TransportMode.WALK), 20)
        self.assertEqual(spec.execution_profile, "legacy_v1_regression")
        self.assertIn(
            ServiceType.LEGACY_EDUCATION_ALL,
            {service.service_type for service in spec.services},
        )

    def test_analysis_spec_can_be_written_as_json(self):
        import json
        import tempfile
        from pathlib import Path

        spec = AnalysisSpec.default_for_city(
            "Napoli", analysis_date="2026-10-06", municipality_code="063049"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis_spec.json"
            spec.write_json(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["municipality_code"], "063049")
            self.assertEqual(payload["execution_profile"], "methodology_v2")

    def test_invalid_threshold_order_fails(self):
        with self.assertRaises(ValueError):
            from core.analysis_spec import ServiceAnalysisSpec
            ServiceAnalysisSpec(
                service_type=ServiceType.PHARMACY,
                modes=(TransportMode.WALK,),
                thresholds_min_by_mode={"walk": (10, 5, 15)},
                population_selector=PopulationSelector.TOTAL,
            )


if __name__ == "__main__":
    unittest.main()
