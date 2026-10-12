"""Regression tests: Health spatial QA evidence must survive finalization.

Do not change the coordinate-selection policy or promote review records.
"""

import unittest

import pandas as pd

from transformation.health import finalize_build_final


def source(service_id='HEALTH::SALUTE:PHARMACY:6496', *, suspicious=False):
    return {
        'service_site_id': service_id,
        'subcategory': 'pharmacy',
        'source_record_id': service_id.rsplit(':', 1)[-1],
        'source_latitude': 44.816433,
        'source_longitude': 9.970429,
        'source_coordinate_present': True,
        'source_coordinate_suspicious': suspicious,
    }


def qa(service_id='HEALTH::SALUTE:PHARMACY:6496', *, suspicious=True):
    return {
        'service_site_id': service_id,
        'source_coordinate_suspicious': suspicious,
        'source_coordinate_inside_municipality': False,
        'source_coordinate_outside_municipality': bool(suspicious),
        'source_coordinate_distance_to_municipality_m': 19263.47,
        'source_coordinate_suspicion_reason': (
            'outside_municipality' if suspicious else None
        ),
        'qa_status': 'review',
    }


def finalize(sources, quality):
    return finalize_build_final(
        pd.DataFrame(sources),
        pd.DataFrame(columns=['service_site_id']),
        pd.DataFrame(quality),
        pd.DataFrame(columns=['subcategory', 'latitude', 'longitude']),
    )


class HealthSpatialQAProvenanceTests(unittest.TestCase):
    def test_municipal_outlier_flag_survives_finalization(self):
        result = finalize([source()], [qa()]).iloc[0]
        self.assertTrue(result['source_coordinate_suspicious'])
        self.assertTrue(result['source_coordinate_outside_municipality'])
        self.assertEqual(result['source_coordinate_suspicion_reason'], 'outside_municipality')
        self.assertEqual(result['source_coordinate_distance_to_municipality_m'], 19263.47)

    def test_outlier_is_not_silently_promoted(self):
        result = finalize([source()], [qa()]).iloc[0]
        self.assertEqual(result['resolution_status'], 'review')
        self.assertEqual(result['coordinate_source'], 'ministero_salute_unvalidated')
        self.assertFalse(result['usable_for_accessibility'])
        self.assertEqual(result['latitude'], 44.816433)
        self.assertEqual(result['longitude'], 9.970429)

    def test_qa_false_does_not_erase_duplicate_suspicion(self):
        result = finalize([source(suspicious=True)], [qa(suspicious=False)]).iloc[0]
        self.assertTrue(result['source_coordinate_suspicious'])
        self.assertFalse(result['usable_for_accessibility'])

    def test_qa_missing_flag_keeps_source_flag(self):
        quality = qa()
        quality.pop('source_coordinate_suspicious')
        result = finalize([source(suspicious=True)], [quality]).iloc[0]
        self.assertTrue(result['source_coordinate_suspicious'])

    def test_invalid_string_flag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'must be boolean'):
            finalize([source()], [qa(suspicious='False')])

    def test_no_qa_record_does_not_create_new_confidence(self):
        result = finalize([source()], []).iloc[0]
        self.assertFalse(result['source_coordinate_suspicious'])
        self.assertEqual(result['resolution_status'], 'review')
        self.assertFalse(result['usable_for_accessibility'])

    def test_join_uses_service_id_not_row_position(self):
        a, b = source('A'), source('B')
        report = finalize([a, b], [qa('B', suspicious=False), qa('A')])
        by_id = report.set_index('service_site_id')
        self.assertTrue(by_id.loc['A', 'source_coordinate_suspicious'])
        self.assertFalse(by_id.loc['B', 'source_coordinate_suspicious'])

    def test_tolerance_is_not_reinterpreted_as_qa_suspicion(self):
        quality = qa(suspicious=False)
        quality['source_coordinate_inside_municipality'] = False
        quality['source_coordinate_distance_to_municipality_m'] = 18.46
        result = finalize([source()], [quality]).iloc[0]
        self.assertFalse(result['source_coordinate_suspicious'])
        self.assertFalse(result['source_coordinate_inside_municipality'])
        self.assertFalse(result['source_coordinate_outside_municipality'])
        self.assertEqual(result['source_coordinate_distance_to_municipality_m'], 18.46)


if __name__ == '__main__':
    unittest.main()
