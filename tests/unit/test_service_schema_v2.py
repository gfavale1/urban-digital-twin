import json
import unittest

import pandas as pd

from core.analysis_spec import ServiceType
from core.schema_v2 import SERVICE_V2
from transformation.build_service_layer_v2 import (
    classify_school_grade,
    education_relations,
    education_v2_rows,
    health_v2_rows,
)


class ServiceSchemaV2Tests(unittest.TestCase):
    def test_school_grade_classifier_uses_explicit_levels(self):
        cases = {
            "SCUOLA DELL'INFANZIA": ServiceType.PRESCHOOL.value,
            "SCUOLA PRIMARIA": ServiceType.PRIMARY_SCHOOL.value,
            "SCUOLA SECONDARIA DI I GRADO": ServiceType.LOWER_SECONDARY_SCHOOL.value,
            "SCUOLA PRIMO GRADO": ServiceType.LOWER_SECONDARY_SCHOOL.value,
            "SCUOLA SEC. DI 1° GRADO": ServiceType.LOWER_SECONDARY_SCHOOL.value,
            "SCUOLA SECONDARIA DI II GRADO": ServiceType.UPPER_SECONDARY_SCHOOL.value,
            "SCUOLA SECONDO GRADO": ServiceType.UPPER_SECONDARY_SCHOOL.value,
            "LICEO SCIENTIFICO": ServiceType.UPPER_SECONDARY_SCHOOL.value,
            "ISTITUTO TECNICO INDUSTRIALE": ServiceType.UPPER_SECONDARY_SCHOOL.value,
            "IST TEC COMMERCIALE E PER GEOMETRI": ServiceType.UPPER_SECONDARY_SCHOOL.value,
            "ISTITUTO D'ARTE": ServiceType.UPPER_SECONDARY_SCHOOL.value,
            "ISTITUTO MAGISTRALE": ServiceType.UPPER_SECONDARY_SCHOOL.value,
            "IST PROF INDUSTRIA E ARTIGIANATO": ServiceType.UPPER_SECONDARY_SCHOOL.value,
            "IST PROF PER I SERVIZI COMMERCIALI E TURISTICI": ServiceType.UPPER_SECONDARY_SCHOOL.value,
            "IST PROF PER I SERVIZI ALBERGHIERI E RISTORAZIONE": ServiceType.UPPER_SECONDARY_SCHOOL.value,
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                actual, _ = classify_school_grade(label)
                self.assertEqual(actual, expected)

    def test_organisational_school_label_is_not_guessed(self):
        actual, method = classify_school_grade("ISTITUTO COMPRENSIVO")
        self.assertIsNone(actual)
        self.assertEqual(method, "unclassified_grade_description")

    def _legacy_education_site(self):
        return pd.DataFrame(
            [
                {
                    "service_site_id": "EDUCATION::MIMB:001",
                    "category": "education",
                    "subcategory": "state_school_building",
                    "name": "Shared school building",
                    "municipality_code": "034027",
                    "address": "Via Test 1",
                    "longitude": 10.0,
                    "latitude": 44.0,
                    "coordinate_source": "automatic",
                    "coordinate_resolution": "site",
                    "confidence": "high",
                    "usable_for_accessibility": True,
                    "capacity_value": None,
                    "capacity_unit": None,
                    "source_record_id": "001",
                    "provenance_json": json.dumps(
                        {"linked_school_codes": json.dumps(["A", "B", "C"])}
                    ),
                }
            ]
        )

    def _registry(self):
        return pd.DataFrame(
            [
                {
                    "school_code": "A",
                    "grade_description": "SCUOLA PRIMARIA",
                    "registry_type": "state",
                    "source_catalog_data_as_of": "2025-08-31",
                    "ingested_at_utc": "2026-09-27T12:00:00+00:00",
                },
                {
                    "school_code": "B",
                    "grade_description": "SCUOLA PRIMARIA",
                    "registry_type": "state",
                    "source_catalog_data_as_of": "2025-08-31",
                    "ingested_at_utc": "2026-09-27T12:00:00+00:00",
                },
                {
                    "school_code": "C",
                    "grade_description": "SCUOLA SECONDARIA DI I GRADO",
                    "registry_type": "state",
                    "source_catalog_data_as_of": "2025-08-31",
                    "ingested_at_utc": "2026-09-27T12:00:00+00:00",
                },
            ]
        )

    def test_one_site_can_expose_multiple_levels_without_double_counting_same_level(self):
        legacy = self._legacy_education_site()
        relation = education_relations(legacy, self._registry())
        rows = education_v2_rows(legacy, relation, "202425")
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["service_type"] for row in rows},
            {
                ServiceType.PRIMARY_SCHOOL.value,
                ServiceType.LOWER_SECONDARY_SCHOOL.value,
            },
        )
        primary = next(
            row for row in rows if row["service_type"] == ServiceType.PRIMARY_SCHOOL.value
        )
        self.assertEqual(json.loads(primary["source_record_ids_json"]), ["A", "B"])

    def test_health_hospital_semantics_are_establishment_level(self):
        health = pd.DataFrame(
            [
                {
                    "service_site_id": "HEALTH::H1",
                    "category": "health",
                    "subcategory": "hospital",
                    "name": "Hospital establishment",
                    "municipality_code": "034027",
                    "address": "Via Salute 1",
                    "longitude": 10.1,
                    "latitude": 44.1,
                    "coordinate_source": "ministero_salute_confirmed_by_osm",
                    "coordinate_resolution": "site_or_address",
                    "confidence": "high",
                    "usable_for_accessibility": True,
                    "capacity_value": 100,
                    "capacity_unit": "beds",
                    "source_record_id": "H1",
                    "reference_period": "2023",
                    "provenance_json": "{}",
                }
            ]
        )
        rows = health_v2_rows(health)
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["service_type"],
            ServiceType.HOSPITAL_ESTABLISHMENT.value,
        )
        self.assertEqual(rows[0]["capacity_value"], 100.0)
        self.assertIsNone(rows[0]["source_reference_date"])
        self.assertEqual(rows[0]["source_reference_period"], "2023")

    def test_service_v2_contract_accepts_migrated_rows(self):
        legacy = self._legacy_education_site()
        relation = education_relations(legacy, self._registry())
        rows = education_v2_rows(legacy, relation, "202425")
        frame = pd.DataFrame(rows)
        SERVICE_V2.validate_columns(frame.columns)


if __name__ == "__main__":
    unittest.main()
