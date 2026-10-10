from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pipeline_v2
from core.run_manifest import RunManifest
from ingestion.source_acquisition import AcquisitionReport, SourceRecord


class PipelineV2B5CTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ctx = SimpleNamespace(
            name="Parma", code="034027", region_code="08", province_code="034"
        )
        self.args = pipeline_v2.parse_args(["--city", "Parma", "--analysis-date", "2026-10-10"])
        self.paths = {}
        self.records = []
        for source in sorted(pipeline_v2.EXPECTED_SOURCE_IDS):
            count = 4 if source == 'istat_boundaries_2021' else 1
            found_paths = []
            for i in range(count):
                file = self.root / 'data' / 'raw' / source / (f"item{i}.shp" if count > 1 else 'data.csv')
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(f'{source}:{i}'.encode())
                found_paths.append(file)
            paths = tuple(str(f) for f in found_paths)
            hashes = {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in found_paths}
            self.paths[source] = paths
            year_key = pipeline_v2.EXPECTED_SOURCE_SNAPSHOTS[source]
            snapshot = ('2021' if year_key is None else pipeline_v2.requested_snapshots(self.args)[year_key])
            self.records.append(SourceRecord(
                source_id=source, state='cached', requested_snapshot=snapshot,
                paths=paths, sha256=hashes,
            ))

    def report(self, *, missing=None):
        records = tuple(
            SourceRecord(
                source_id=r.source_id,
                state='missing' if r.source_id == missing else r.state,
                requested_snapshot=r.requested_snapshot,
                paths=() if r.source_id == missing else r.paths,
                sha256={} if r.source_id == missing else r.sha256,
            ) for r in self.records
        )
        return AcquisitionReport(
            municipality_code=self.ctx.code, municipality_name=self.ctx.name,
            census_year='2023', school_year='202425', building_year='202425',
            hospital_year='2023', pharmacy_reference_date='2025-06-30',
            fetch_supported=False, generated_at_utc='2026-10-10T00:00:00Z',
            sources=records,
        )

    def run_with(self, report=None, args=None):
        with patch.object(pipeline_v2, 'resolve_city', return_value=self.ctx), \
             patch.object(pipeline_v2, 'acquire', return_value=(report if report is not None else self.report())) as acquired:
            result = pipeline_v2.run(args or self.args, repo_root=self.root)
        return result, acquired

    def get_manifest(self):
        path = next((self.root/'runs').glob('*/run_manifest.json'))
        return json.loads(path.read_text())

    def test_cli_requires_city_or_code(self):
        with self.assertRaises(SystemExit):
            pipeline_v2.parse_args([])

    def test_cli_rejects_mutually_exclusive_location(self):
        with self.assertRaises(SystemExit):
            pipeline_v2.parse_args(['--city', 'Parma', '--municipality-code', '034027'])

    def test_cli_rejects_province_filter_without_city(self):
        with self.assertRaises(SystemExit):
            pipeline_v2.parse_args(['--municipality-code', '034027', '--province-code', '034'])

    def test_cli_requires_explicit_fetch_for_contemporary_pharmacy(self):
        with self.assertRaises(SystemExit):
            pipeline_v2.parse_args(['--city','Parma','--allow-contemporary-pharmacy-source'])

    def test_success_persists_default_methodology_v2_and_all_sources(self):
        result, acquired = self.run_with()
        self.assertEqual(result, 0)
        acquired.assert_called_once()
        self.assertFalse(acquired.call_args.kwargs['allow_fetch'])
        manifest = self.get_manifest()
        self.assertEqual(manifest['analysis_spec']['execution_profile'], 'methodology_v2')
        self.assertEqual(manifest['analysis_spec']['walking_speed_m_s'], 0.9)
        self.assertEqual(manifest['routing_backend'], 'not_executed_b5c_bootstrap')
        self.assertFalse(manifest['routing_parameters']['routing_executed'])
        self.assertEqual(manifest['stages']['source_acquisition']['status'], 'completed')
        self.assertEqual(set(manifest['source_records']), pipeline_v2.EXPECTED_SOURCE_IDS)
        self.assertEqual(len(manifest['source_checksums']), 7)
        self.assertTrue(next((self.root/'runs').glob('*/source_acquisition.json')).exists())
        for source in manifest['source_records'].values():
            self.assertEqual(source['freshness_status'], 'unknown')
            self.assertIsNone(source['retrieved_at'])

    def test_boundary_bundle_fingerprint_is_stable_and_file_hashes_retained(self):
        self.run_with()
        manifest = self.get_manifest()
        files = manifest['stages']['source_acquisition']['metrics']['file_sha256_by_source']['istat_boundaries_2021']
        self.assertEqual(len(files), 4)
        expected = pipeline_v2._source_fingerprint(next(r for r in self.records if r.source_id == 'istat_boundaries_2021'), self.root)
        self.assertEqual(manifest['source_checksums']['istat_boundaries_2021'], expected)

    def test_dry_run_is_read_only_even_with_fetch_flag(self):
        args = pipeline_v2.parse_args(['--city','Parma','--dry-run','--fetch-supported'])
        result, acquired = self.run_with(args=args)
        self.assertEqual(result, 0)
        self.assertFalse(acquired.call_args.kwargs['allow_fetch'])
        self.assertFalse((self.root/'runs').exists())

    def test_explicit_fetch_is_forwarded(self):
        args = pipeline_v2.parse_args(['--city','Parma','--fetch-supported'])
        _, acquired = self.run_with(args=args)
        self.assertTrue(acquired.call_args.kwargs['allow_fetch'])

    def test_blocked_source_leaves_failed_manifest_and_returns_two(self):
        result, _ = self.run_with(report=self.report(missing='salute_pharmacies'))
        self.assertEqual(result, 2)
        manifest = self.get_manifest()
        self.assertEqual(manifest['stages']['source_acquisition']['status'], 'failed')
        self.assertEqual(manifest['source_records']['salute_pharmacies']['checksum_sha256'], None)
        self.assertIn('salute_pharmacies=missing', manifest['warnings'][-1])

    def test_unexpected_acquisition_failure_is_persisted(self):
        with patch.object(pipeline_v2, 'resolve_city', return_value=self.ctx), \
             patch.object(pipeline_v2, 'acquire', side_effect=RuntimeError('network broken')):
            with self.assertRaisesRegex(RuntimeError, 'network broken'):
                pipeline_v2.run(self.args, repo_root=self.root)
        self.assertEqual(self.get_manifest()['stages']['source_acquisition']['status'], 'failed')

    def test_duplicate_source_id_rejected_with_failure(self):
        r = self.report()
        bad = AcquisitionReport(
            municipality_code=r.municipality_code, municipality_name=r.municipality_name,
            census_year=r.census_year, school_year=r.school_year, building_year=r.building_year,
            hospital_year=r.hospital_year, pharmacy_reference_date=r.pharmacy_reference_date,
            fetch_supported=False, generated_at_utc=r.generated_at_utc,
            sources=(r.sources[0], *r.sources),
        )
        with self.assertRaisesRegex(ValueError, 'Duplicate source ID'):
            self.run_with(report=bad)
        self.assertEqual(self.get_manifest()['stages']['source_acquisition']['status'], 'failed')

    def test_incomplete_inventory_rejected_with_failure(self):
        r = self.report()
        bad = AcquisitionReport(
            municipality_code=r.municipality_code, municipality_name=r.municipality_name,
            census_year=r.census_year, school_year=r.school_year, building_year=r.building_year,
            hospital_year=r.hospital_year, pharmacy_reference_date=r.pharmacy_reference_date,
            fetch_supported=False, generated_at_utc=r.generated_at_utc,
            sources=r.sources[:-1],
        )
        with self.assertRaisesRegex(ValueError, 'Incomplete source inventory'):
            self.run_with(report=bad)
        self.assertEqual(self.get_manifest()['stages']['source_acquisition']['status'], 'failed')

    def test_collision_does_not_overwrite_run_manifest(self):
        self.run_with()
        old = self.get_manifest()
        with patch.object(pipeline_v2, 'resolve_city', return_value=self.ctx), \
             patch.object(pipeline_v2, 'acquire', return_value=self.report()), \
             patch.object(pipeline_v2.RunManifest, 'create', wraps=RunManifest.create) as factory:
            # Only deterministic if both calls happen within one second.
            manifest = RunManifest.create(
                pipeline_v2.AnalysisSpec.default_for_city(
                    self.ctx.name, analysis_date='2026-10-10', municipality_code=self.ctx.code
                ), repo_root=self.root
            )
            existing_dir = next((self.root/'runs').iterdir())
            manifest.run_id = existing_dir.name[:-11]  # remove '_' + 10 hex snapshot digest
            factory.return_value = manifest
            factory.side_effect = None
            with self.assertRaises(FileExistsError):
                pipeline_v2.run(self.args, repo_root=self.root)
        self.assertEqual(self.get_manifest(), old)

    def test_real_preflight_adapter_integration_on_local_cached_sources(self):
        from ingestion import source_acquisition as sa
        from ingestion import download_istat_boundaries as boundaries
        from ingestion import mim
        istat = self.root / 'data' / 'raw' / 'istat'
        mim_root = self.root / 'data' / 'raw' / 'mim'
        salute = self.root / 'data' / 'raw' / 'salute'
        regional = istat/'censimento_2023'/'Dati_regionali_2023'
        regional.mkdir(parents=True)
        (regional/'R08_Emilia-Romagna_2023_sezioni.xlsx').write_bytes(b'cached xlsx payload')
        boundaries_dir = istat/'basi_territoriali_2021'
        shapes_dir = boundaries_dir/'SHP'
        shapes_dir.mkdir(parents=True)
        for suffix in ('.shp','.shx','.dbf','.prj'):
            (shapes_dir/f'R08_21_WGS84{suffix}').write_bytes(b'cached shapefile sidecar')
        school_dir = mim_root/'schools'/'202425'
        school_dir.mkdir(parents=True)
        for dataset in ('SCUANAGRAFESTAT','SCUANAGRAFEPAR'):
            filename = mim.candidate_filenames(dataset,'202425')[0]
            (school_dir/filename).write_bytes(b'cached registry')
        bdir = mim_root/'buildings'/'202425'
        bdir.mkdir(parents=True)
        (bdir/'EDIANAGRAFESTA202120242520250806.csv').write_bytes(b'cached buildings')
        phdir = salute/'farmacie'
        phdir.mkdir(parents=True)
        (phdir/'FRM_FARMA_5_20260927.csv').write_bytes(b'cached pharmacies')
        hosdir = salute/'strutture_ospedaliere_2023'
        hosdir.mkdir(parents=True)
        (hosdir/'Posti_letto_2023.csv').write_bytes(b'cached hospitals')
        with patch.object(pipeline_v2, 'resolve_city', return_value=self.ctx), \
             patch.object(sa, 'RAW_ISTAT', istat), \
             patch.object(sa, 'RAW_MIM', mim_root), \
             patch.object(sa, 'RAW_SALUTE', salute), \
             patch.object(boundaries, 'BOUNDARIES_DIR', boundaries_dir):
            result = pipeline_v2.run(self.args, repo_root=self.root)
        self.assertEqual(result, 0)
        manifest = self.get_manifest()
        self.assertEqual(manifest['stages']['source_acquisition']['status'], 'completed')
        self.assertEqual(len(manifest['source_records']), 7)

    def test_wrong_report_municipality_is_rejected_and_recorded(self):
        r = self.report()
        bad = AcquisitionReport(
            municipality_code='999999', municipality_name=r.municipality_name,
            census_year=r.census_year, school_year=r.school_year, building_year=r.building_year,
            hospital_year=r.hospital_year, pharmacy_reference_date=r.pharmacy_reference_date,
            fetch_supported=False, generated_at_utc=r.generated_at_utc, sources=r.sources,
        )
        with self.assertRaisesRegex(ValueError, 'municipality'):
            self.run_with(report=bad)
        self.assertEqual(self.get_manifest()['stages']['source_acquisition']['status'], 'failed')

    def test_source_vintage_mismatch_is_rejected(self):
        r = self.report()
        old = r.sources[0]
        corrupted = SourceRecord(
            source_id=old.source_id, state=old.state, requested_snapshot='9999',
            paths=old.paths, sha256=old.sha256,
        )
        bad = AcquisitionReport(
            municipality_code=r.municipality_code, municipality_name=r.municipality_name,
            census_year=r.census_year, school_year=r.school_year, building_year=r.building_year,
            hospital_year=r.hospital_year, pharmacy_reference_date=r.pharmacy_reference_date,
            fetch_supported=False, generated_at_utc=r.generated_at_utc,
            sources=(corrupted, *r.sources[1:]),
        )
        with self.assertRaisesRegex(ValueError, 'source vintage mismatch'):
            self.run_with(report=bad)
        self.assertEqual(self.get_manifest()['stages']['source_acquisition']['status'], 'failed')

    def test_invalid_sha256_fails_the_gate(self):
        r = self.report()
        old = next(s for s in r.sources if s.source_id == "mim_buildings")
        bad_checksum = SourceRecord(
            source_id=old.source_id, state=old.state,
            requested_snapshot=old.requested_snapshot,
            paths=old.paths, sha256={old.paths[0]: 'not-a-checksum'},
        )
        bad = AcquisitionReport(
            municipality_code=r.municipality_code, municipality_name=r.municipality_name,
            census_year=r.census_year, school_year=r.school_year, building_year=r.building_year,
            hospital_year=r.hospital_year, pharmacy_reference_date=r.pharmacy_reference_date,
            fetch_supported=False, generated_at_utc=r.generated_at_utc,
            sources=tuple(bad_checksum if s.source_id == old.source_id else s for s in r.sources),
        )
        with self.assertRaisesRegex(ValueError, 'Invalid SHA-256'):
            self.run_with(report=bad)
        self.assertEqual(self.get_manifest()['stages']['source_acquisition']['status'], 'failed')

    def test_snapshot_inputs_recorded_in_manifest(self):
        self.run_with()
        manifest = self.get_manifest()
        self.assertEqual(manifest['routing_parameters']['requested_source_snapshots']['school_year'], '202425')
        self.assertEqual(manifest['stages']['source_acquisition']['metrics']['requested_source_snapshots']['census_year'],'2023')


if __name__ == '__main__':
    unittest.main()
