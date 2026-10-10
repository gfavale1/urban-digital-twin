from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point

from quality.audit_anncsu_health_geospatial import (
    build_geospatial_review, census_footprint, explicit_true,
    load_b6b1_review, parse_args, sha256_file, snap_review_points, write_report,
)


def candidates():
    return pd.DataFrame([
        {'entity_id': 'P', 'name': 'Pharmacy', 'service_type_v2': 'pharmacy',
         'anncsu_latitude': 44.8000, 'anncsu_longitude': 10.3200},
        {'entity_id': 'H', 'name': 'Specialist hospital', 'service_type_v2': 'hospital_establishment',
         'anncsu_latitude': 44.8300, 'anncsu_longitude': 10.3600},
    ])


def polygon():
    areas = gpd.GeoDataFrame({'id': [1]}, geometry=[Polygon([
        (10.30, 44.79), (10.34, 44.79), (10.34, 44.81), (10.30, 44.81)
    ])], crs='EPSG:4326')
    return census_footprint(areas)


def attach(kind):
    return pd.DataFrame([
        dict(entity_id='P', snapped=True, snap_distance_m=12.0, node_id='one',
             network_component_id='component_000001', is_largest_component=True),
        dict(entity_id='H', snapped=False, snap_distance_m=2011.0, node_id=None,
             network_component_id=None, is_largest_component=None),
    ])


class B6B2Tests(unittest.TestCase):
    def test_true_must_be_explicit(self):
        self.assertTrue(explicit_true(True))
        self.assertTrue(explicit_true('TRUE'))
        for val in ('false', False, 1, '1', None):
            self.assertFalse(explicit_true(val))

    def test_footprint_covers_inside_not_outside(self):
        p = polygon()
        self.assertTrue(p.covers(Point(10.32, 44.8)))
        self.assertFalse(p.covers(Point(10.4, 44.8)))

    def test_missing_crs_rejected(self):
        areas = gpd.GeoDataFrame(geometry=[polygon()], crs=None)
        with self.assertRaises(ValueError):
            census_footprint(areas)

    def test_non_polygon_rejected(self):
        areas = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs='EPSG:4326')
        with self.assertRaises(ValueError):
            census_footprint(areas)

    def test_build_review_never_promotes(self):
        df, report = build_geospatial_review(candidates(), polygon(),
                                              {'walk': attach('walk'), 'drive': attach('drive')})
        self.assertEqual(len(df), 2)
        self.assertFalse(df['scenario_eligible'].any())
        self.assertFalse(df['municipality_boundary_verified'].any())
        self.assertEqual(report['inside_istat_census_footprint'], 1)
        self.assertEqual(report['walk_snapped'], 1)
        self.assertEqual(report['scenario_eligible'], 0)
        self.assertTrue(df.set_index('entity_id').loc['P', 'walk_snapped'])

    def test_missing_mode_rejected(self):
        with self.assertRaises(ValueError):
            build_geospatial_review(candidates(), polygon(), {'walk': attach('walk')})

    def test_duplicate_candidate_rejected(self):
        dup = pd.concat([candidates(), candidates().iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            build_geospatial_review(dup, polygon(),
                                     {'walk': attach('walk'), 'drive': attach('drive')})

    def test_missing_network_row_rejected(self):
        with self.assertRaises(ValueError):
            build_geospatial_review(candidates(), polygon(),
                {'walk': attach('walk').iloc[:1], 'drive': attach('drive')})

    def test_snapped_without_node_rejected(self):
        bad = attach('walk')
        bad.loc[0, 'node_id'] = None
        with self.assertRaises(ValueError):
            build_geospatial_review(candidates(), polygon(),
                {'walk': bad, 'drive': attach('drive')})

    def test_non_numeric_coordinates_rejected(self):
        bad = candidates()
        bad['anncsu_latitude'] = bad['anncsu_latitude'].astype(object)
        bad.loc[0, 'anncsu_latitude'] = 'broken'
        with self.assertRaises(ValueError):
            build_geospatial_review(bad, polygon(),
                {'walk': attach('walk'), 'drive': attach('drive')})

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            snap_review_points(candidates(), None, None, mode='bike', checksum='abc',
                               max_snap_distance_m=1000)

    def test_network_uses_canonical_attachment_logic(self):
        nodes = gpd.GeoDataFrame({'source_record_id': ['1', '2']},
            geometry=[Point(10.3200, 44.8000), Point(10.3220, 44.8000)], crs='EPSG:4326')
        edges = gpd.GeoDataFrame({'source_osm_node': ['1'], 'target_osm_node': ['2']},
            geometry=[Point(10.321, 44.8)], crs='EPSG:4326')
        subset = candidates().iloc[:1]
        out = snap_review_points(subset, nodes, edges, mode='walk',
                                 checksum='test', max_snap_distance_m=1000)
        self.assertEqual(len(out), 1)
        self.assertTrue(out.iloc[0]['snapped'])
        self.assertEqual(out.iloc[0]['node_id'], '1')
        self.assertEqual(out.iloc[0]['network_component_id'], 'component_000001')

    def test_outlier_attaches_but_not_snapped(self):
        nodes = gpd.GeoDataFrame({'source_record_id': ['1']},
            geometry=[Point(10.3200, 44.8000)], crs='EPSG:4326')
        edges = gpd.GeoDataFrame({'source_osm_node': [], 'target_osm_node': []},
            geometry=[], crs='EPSG:4326')
        far = candidates().iloc[1:]
        out = snap_review_points(far, nodes, edges, mode='drive',
                                 checksum='test', max_snap_distance_m=1000)
        self.assertFalse(out.iloc[0]['snapped'])
        self.assertIsNone(out.iloc[0]['node_id'])

    def test_write_cache_is_idempotent_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df, report = build_geospatial_review(candidates(), polygon(),
                {'walk': attach('walk'), 'drive': attach('drive')})
            report.update(municipality_code='034027', input_sha256={'a': 'hash'})
            out, manifest, cached = write_report(root, '034027', '20260915', '20250630', df, report)
            self.assertFalse(cached)
            self.assertEqual(sha256_file(out), json.loads(manifest.read_text())['output_csv_sha256'])
            _, _, cached = write_report(root, '034027', '20260915', '20250630', df, report)
            self.assertTrue(cached)
            with self.assertRaises(FileExistsError):
                write_report(root, '034027', '20260915', '20250630', df,
                             dict(report, scenario_eligible=1))

    def test_b6b1_load_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv, manifest = root/'review.csv', root/'review.json'
            pd.DataFrame([dict(entity_id='P', name='Parigi', service_type_v2='pharmacy',
                anncsu_latitude=44.79, anncsu_longitude=10.35,
                review_status='manual_verification_required', scenario_eligible=False,
                anncsu_snapshot_date='20260915', health_final_usable=False,
                v2_legacy_usable=False)]).to_csv(csv, index=False)
            payload = dict(policy='b6b1_anncsu_health_recovery_review_v1',
                           municipality_code='034027', anncsu_snapshot_date='20260915',
                           output_csv_sha256=sha256_file(csv))
            manifest.write_text(json.dumps(payload))
            self.assertEqual(len(load_b6b1_review(csv, manifest, '034027', '20260915')), 1)
            csv.write_text(csv.read_text().replace('Parigi', 'Tampered'))
            with self.assertRaises(ValueError):
                load_b6b1_review(csv, manifest, '034027', '20260915')

    def test_b6b1_refuses_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv, manifest = root/'review.csv', root/'review.json'
            pd.DataFrame([dict(entity_id='P', name='Parigi', service_type_v2='pharmacy',
                anncsu_latitude=44.79, anncsu_longitude=10.35,
                review_status='manual_verification_required', scenario_eligible=True,
                anncsu_snapshot_date='20260915', health_final_usable=False,
                v2_legacy_usable=False)]).to_csv(csv, index=False)
            payload = dict(policy='b6b1_anncsu_health_recovery_review_v1',
                           municipality_code='034027', anncsu_snapshot_date='20260915',
                           output_csv_sha256=sha256_file(csv))
            manifest.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_b6b1_review(csv, manifest, '034027', '20260915')

    def test_b6b1_upstream_source_hash_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv, manifest, source = root/'review.csv', root/'review.json', root/'input.csv'
            source.write_text('original')
            pd.DataFrame([dict(entity_id='P', name='Parigi', service_type_v2='pharmacy',
                anncsu_latitude=44.79, anncsu_longitude=10.35,
                review_status='manual_verification_required', scenario_eligible=False,
                anncsu_snapshot_date='20260915', health_final_usable=False,
                v2_legacy_usable=False)]).to_csv(csv, index=False)
            payload = dict(policy='b6b1_anncsu_health_recovery_review_v1',
                           municipality_code='034027', anncsu_snapshot_date='20260915',
                           output_csv_sha256=sha256_file(csv),
                           input_sha256={'source': sha256_file(source)})
            manifest.write_text(json.dumps(payload))
            self.assertEqual(1, len(load_b6b1_review(csv, manifest, '034027', '20260915',
                                                     source_paths={'source': source})))
            source.write_text('changed')
            with self.assertRaises(ValueError):
                load_b6b1_review(csv, manifest, '034027', '20260915',
                                 source_paths={'source': source})

    def test_b6b1_upstream_missing_hashes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv, manifest, source = root/'review.csv', root/'review.json', root/'input.csv'
            source.write_text('original')
            pd.DataFrame([dict(entity_id='P', name='Parigi', service_type_v2='pharmacy',
                anncsu_latitude=44.79, anncsu_longitude=10.35,
                review_status='manual_verification_required', scenario_eligible=False,
                anncsu_snapshot_date='20260915', health_final_usable=False,
                v2_legacy_usable=False)]).to_csv(csv, index=False)
            payload = dict(policy='b6b1_anncsu_health_recovery_review_v1',
                           municipality_code='034027', anncsu_snapshot_date='20260915',
                           output_csv_sha256=sha256_file(csv))
            manifest.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_b6b1_review(csv, manifest, '034027', '20260915',
                                 source_paths={'source': source})

    def test_parse_args(self):
        args = parse_args(['--municipality-code', '034027'])
        self.assertEqual(args.anncsu_snapshot, '20260915')
        with self.assertRaises(SystemExit):
            parse_args(['--municipality-code', 'xxx'])
        with self.assertRaises(SystemExit):
            parse_args(['--municipality-code', '034027', '--max-snap-distance-m', '0'])


if __name__ == '__main__':
    unittest.main()
