"""Pure B6B1 evidence audits. No edits to national/municipal source datasets."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quality.audit_anncsu_health_recovery import (
    audit_health_recovery, _finite_coordinate, haversine_m, paths_for,
    read_verified_inputs, run, parse_args, sha256_file, write_evidence,
)

CODE, SNAP = '034027', '20260915'


def fixtures():
    records = []
    health = []
    v2 = []
    targets = [
        ('pharmacy', 'SALUTE:PHARMACY:19961', 'Farmacia Parigi', 'Via Sofia 1', 44.793823, 10.355438, False),
        ('hospital', 'SALUTE:HOSPITAL:080253:', 'Fondazione Don Gnocchi', 'Piazzale dei Servi 3', 44.801009, 10.334596, False),
        ('pharmacy', 'SALUTE:PHARMACY:19577', 'Farmacia Cavagnari', 'Via La Spezia 150/A', 44.784361, 10.293380, True),
    ]
    for kind, sid, name, address, lat, lon, resolved in targets:
        records.append({
            'entity_type': kind, 'entity_id': sid, 'municipality_code': CODE,
            'match_status': 'exact_single_access_candidate',
            'existing_latitude': None, 'existing_longitude': None,
            'source_address': address, 'reference_vintage': '2023' if kind == 'hospital' else '2025-06-30',
            'candidate_count': 1, 'candidate_access_id': sid+'-access',
            'anncsu_latitude': lat, 'anncsu_longitude': lon,
            'anncsu_method_code': 3, 'anncsu_snapshot_date': SNAP,
            'coordinate_accepted_for_accessibility': False,
        })
        health.append({
            'service_site_id': sid, 'name': name, 'subcategory': kind,
            'resolution_status': 'resolved_auto' if resolved else 'unresolved',
            'usable_for_accessibility': resolved,
            'latitude': lat if resolved else None, 'longitude': lon if resolved else None,
            'municipality_code': CODE,
        })
        v2.append({
            'service_id': 'SVC2::HEALTH::'+sid+('::pharmacy' if kind == 'pharmacy' else '::hospital_establishment'),
            'service_type': 'pharmacy' if kind == 'pharmacy' else 'hospital_establishment',
            'legacy_service_site_id': 'HEALTH::'+sid,
            'legacy_usable_for_accessibility': resolved,
            'latitude': lat if resolved else None, 'longitude': lon if resolved else None,
            'municipality_code': CODE,
        })
    # Existing validated sites nearby, but not <150m.
    health.append({
        'service_site_id': 'SALUTE:PHARMACY:OTHER', 'name': 'Altra Farmacia',
        'subcategory': 'pharmacy', 'resolution_status': 'resolved_auto',
        'usable_for_accessibility': True, 'latitude': 44.797, 'longitude': 10.355438,
        'municipality_code': CODE,
    })
    osm = pd.DataFrame([
        {'service_site_id': 'OSM::PHARMACY:1', 'category': 'health', 'subcategory': 'pharmacy',
         'name': 'Other', 'latitude': 44.797, 'longitude': 10.355438, 'municipality_code': CODE},
        {'service_site_id': 'OSM::SCHOOL:1', 'category': 'education', 'subcategory': 'osm_school',
         'name': 'School nearby hospital', 'latitude': 44.801009, 'longitude': 10.334596, 'municipality_code': CODE},
        {'service_site_id': 'OSM::PHARMACY:2', 'category': 'health', 'subcategory': 'pharmacy',
         'name': 'Pharmacy nearby hospital', 'latitude': 44.801009, 'longitude': 10.334596, 'municipality_code': CODE},
    ])
    # A school can generate multiple V2 service types with the same legacy ID.
    v2.extend([
        {'service_id': 'SVC2::SCHOOL::ABC::primary_school', 'service_type': 'primary_school',
         'legacy_service_site_id': 'EDUCATION::ABC', 'legacy_usable_for_accessibility': True,
         'latitude': 44.81, 'longitude': 10.35, 'municipality_code': CODE},
        {'service_id': 'SVC2::SCHOOL::ABC::lower_secondary_school', 'service_type': 'lower_secondary_school',
         'legacy_service_site_id': 'EDUCATION::ABC', 'legacy_usable_for_accessibility': True,
         'latitude': 44.81, 'longitude': 10.35, 'municipality_code': CODE},
    ])
    return dict(matches=pd.DataFrame(records), health_final=pd.DataFrame(health),
                v2_services=pd.DataFrame(v2), osm_only=osm)


class RecoveryAuditTests(unittest.TestCase):
    def setUp(self):
        self.frames = fixtures()

    def audit(self):
        return audit_health_recovery(**self.frames, municipality_code=CODE, anncsu_snapshot=SNAP)

    def test_real_shape_two_unresolved_one_downstream_resolved(self):
        rows, meta = self.audit()
        self.assertEqual(len(rows), 2)
        self.assertEqual(meta['already_usable_downstream'], 1)
        self.assertEqual(meta['exact_health_candidates_missing_silver_coordinates'], 3)
        self.assertEqual(meta['scenario_eligible'], 0)
        self.assertFalse(rows['scenario_eligible'].any())

    def test_medical_classification_keeps_hospital_establishment(self):
        rows, _ = self.audit()
        hospital = rows.set_index('entity_id').loc['SALUTE:HOSPITAL:080253:']
        self.assertEqual(hospital['service_type_v2'], 'hospital_establishment')

    def test_nearby_other_service_category_not_hospital_duplicate(self):
        rows, _ = self.audit()
        hospital = rows.set_index('entity_id').loc['SALUTE:HOSPITAL:080253:']
        self.assertEqual(hospital['osm_same_type_within_150m'], 0)
        self.assertEqual(hospital['nearest_osm_same_type_id'], '')
        self.assertFalse(hospital['independent_identity_verified'])

    def test_nearest_pharmacy(self):
        rows, _ = self.audit()
        pharmacy = rows.set_index('entity_id').loc['SALUTE:PHARMACY:19961']
        self.assertEqual(pharmacy['nearest_osm_same_type_id'], 'OSM::PHARMACY:1')
        self.assertGreater(pharmacy['nearest_osm_same_type_distance_m'], 150)

    def test_same_type_osm_within_150m_warns_without_approval(self):
        self.frames['osm_only'].loc[0, 'latitude'] = 44.793824
        rows, meta = self.audit()
        pharmacy = rows.set_index('entity_id').loc['SALUTE:PHARMACY:19961']
        self.assertEqual(pharmacy['osm_same_type_within_150m'], 1)
        self.assertFalse(pharmacy['scenario_eligible'])
        self.assertEqual(meta['nearby_same_type_osm_review_150m'], 1)

    def test_same_type_final_health_proximity_warns(self):
        self.frames['health_final'].loc[3, 'latitude'] = 44.793824
        rows, meta = self.audit()
        self.assertEqual(meta['nearby_same_type_health_review_150m'], 1)
        self.assertFalse(rows['scenario_eligible'].any())

    def test_v2_health_duplicate_rejected(self):
        row = self.frames['v2_services'].iloc[0].copy()
        row['service_id'] += ':different'
        self.frames['v2_services'] = pd.concat([self.frames['v2_services'], pd.DataFrame([row])], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate health V2'):
            self.audit()

    def test_legacy_v2_flags_disagree_rejected(self):
        self.frames['v2_services'].loc[0, 'legacy_usable_for_accessibility'] = True
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.audit()

    def test_wrong_v2_type_rejected(self):
        self.frames['v2_services'].loc[1, 'service_type'] = 'pharmacy'
        with self.assertRaisesRegex(ValueError, 'Wrong service category'):
            self.audit()

    def test_wrong_city_fails_closed(self):
        self.frames['osm_only'].loc[0, 'municipality_code'] = '016024'
        with self.assertRaisesRegex(ValueError, 'invalid municipality'):
            self.audit()

    def test_wrong_snapshot_fails_closed(self):
        self.frames['matches'].loc[0, 'anncsu_snapshot_date'] = '20260815'
        with self.assertRaisesRegex(ValueError, 'snapshot mismatch'):
            self.audit()

    def test_missing_health_identity_rejected(self):
        self.frames['health_final'] = self.frames['health_final'].iloc[1:]
        with self.assertRaisesRegex(ValueError, 'identity missing'):
            self.audit()

    def test_ambiguous_anncsu_match_not_candidate(self):
        self.frames['matches'].loc[0, 'match_status'] = 'ambiguous_multiple_accesses'
        rows, _ = self.audit()
        self.assertNotIn('SALUTE:PHARMACY:19961', list(rows['entity_id']))

    def test_b6a2_must_not_promote(self):
        self.frames['matches'].loc[0, 'coordinate_accepted_for_accessibility'] = True
        with self.assertRaisesRegex(ValueError, 'unexpectedly promotes'):
            self.audit()

    def test_coordinate_range(self):
        self.assertIsNone(_finite_coordinate(99, 10))
        self.assertIsNone(_finite_coordinate(float('nan'), 10))
        self.assertAlmostEqual(haversine_m((44.8, 10.3), (44.8, 10.3)), 0)

    def test_partial_data_missing_column(self):
        self.frames['health_final'] = self.frames['health_final'].drop(columns=['resolution_status'])
        with self.assertRaisesRegex(ValueError, 'missing required columns'):
            self.audit()

    def test_write_report_cache_and_input_mutation_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence, meta = self.audit()
            meta['input_sha256'] = {'fake': 'a'*64}
            csv_file, json_file, cached = write_evidence(root, CODE, SNAP, '20250630', evidence, meta)
            self.assertFalse(cached)
            self.assertTrue(csv_file.is_file())
            self.assertTrue(json_file.is_file())
            self.assertEqual(json.loads(json_file.read_text())['output_csv_sha256'], sha256_file(csv_file))
            _, _, cached = write_evidence(root, CODE, SNAP, '20250630', evidence, meta)
            self.assertTrue(cached)
            meta['input_sha256'] = {'fake': 'b'*64}
            with self.assertRaises(FileExistsError):
                write_evidence(root, CODE, SNAP, '20250630', evidence, meta)

    def test_cached_report_rejects_mutated_evidence_even_with_same_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence, meta = self.audit()
            meta['input_sha256'] = {'fake': 'a'*64}
            write_evidence(root, CODE, SNAP, '20250630', evidence, meta)
            altered = evidence.copy()
            altered.loc[0, 'source_address'] = 'ALTERED WITHOUT INPUT CHANGE'
            with self.assertRaisesRegex(FileExistsError, 'Existing B6B1 results differ'):
                write_evidence(root, CODE, SNAP, '20250630', altered, meta)

    def test_seven_candidates_five_already_resolved(self):
        extra = self.frames['matches'].loc[2].copy()
        extra_health = self.frames['health_final'].loc[2].copy()
        extra_v2 = self.frames['v2_services'].loc[2].copy()
        for index in range(4):
            item = extra.copy()
            final = extra_health.copy()
            v2 = extra_v2.copy()
            item['entity_id'] = f'SALUTE:PHARMACY:RESOLVED{index}'
            item['candidate_access_id'] = f'ACCESS:{index}'
            final['service_site_id'] = item['entity_id']
            v2['service_id'] = f'SVC2::HEALTH::{item["entity_id"]}::pharmacy'
            v2['legacy_service_site_id'] = f'HEALTH::{item["entity_id"]}'
            self.frames['matches'] = pd.concat([self.frames['matches'], pd.DataFrame([item])], ignore_index=True)
            self.frames['health_final'] = pd.concat([self.frames['health_final'], pd.DataFrame([final])], ignore_index=True)
            self.frames['v2_services'] = pd.concat([self.frames['v2_services'], pd.DataFrame([v2])], ignore_index=True)
        evidence, report = self.audit()
        self.assertEqual(report['exact_health_candidates_missing_silver_coordinates'], 7)
        self.assertEqual(report['already_usable_downstream'], 5)
        self.assertEqual(report['still_unresolved'], 2)
        self.assertEqual(len(evidence), 2)

    def test_hash_mismatch_prevents_using_unverified_anncsu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = paths_for(root, CODE, SNAP, '202425', '2025-06-30')
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('temporary fixture', encoding='utf-8')
            paths['anncsu_manifest'].write_text(json.dumps({
                'municipality_code': CODE, 'anncsu_snapshot_date': SNAP,
                'output_csv_sha256': 'b'*64,
            }), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                read_verified_inputs(paths, CODE, SNAP)

    def test_run_dry_is_read_only(self):
        args = parse_args(['--municipality-code', CODE])
        frames = self.frames
        with tempfile.TemporaryDirectory() as tmp, patch(
            'quality.audit_anncsu_health_recovery.read_verified_inputs',
            return_value=(frames, {'fake': 'a'*64}),
        ):
            root = Path(tmp)
            report = run(args, root=root)
            self.assertEqual(report['still_unresolved'], 2)
            self.assertFalse((root / 'data').exists())


if __name__ == '__main__':
    unittest.main()
