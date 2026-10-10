"""B6B1: read-only review of *unresolved* health addresses matched to ANNCSU.

No geocoding promotion, source mutation, v2 edit or accessibility calculation.

Usage:
  PYTHONPATH=src python src/quality/audit_anncsu_health_recovery.py \\
    --municipality-code 034027 --anncsu-snapshot 20260915

With --write-report, the evidence CSV and JSON manifest are saved only when
absent, or confirmed byte-for-byte consistent with the existing report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REVIEW_POLICY = 'b6b1_anncsu_health_recovery_review_v1'
DISTANCE_REVIEW_M = 150.0
SOURCE_TYPES = {'pharmacy': 'pharmacy', 'hospital': 'hospital_establishment'}
OUTPUT_COLUMNS = (
    'entity_type', 'entity_id', 'service_id_v2', 'service_type_v2',
    'name', 'source_address', 'health_reference_vintage',
    'anncsu_snapshot_date', 'anncsu_access_id', 'anncsu_method_code',
    'anncsu_latitude', 'anncsu_longitude',
    'health_final_resolution_status', 'health_final_usable',
    'v2_legacy_usable', 'v2_has_coordinates',
    'nearest_osm_same_type_id', 'nearest_osm_same_type_name',
    'nearest_osm_same_type_distance_m', 'osm_same_type_within_150m',
    'nearest_valid_health_same_type_id', 'nearest_valid_health_same_type_distance_m',
    'valid_health_same_type_within_150m',
    'municipality_polygon_checked', 'site_entrance_verified',
    'independent_identity_verified', 'operational_in_reference_period_verified',
    'service_scope_verified', 'review_status', 'scenario_eligible',
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def _missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _truth(value: Any) -> bool:
    """Only explicit booleans and the string 'true' mean true."""
    return (
        (isinstance(value, (bool, np.bool_)) and bool(value))
        or (isinstance(value, str) and value.strip().lower() == 'true')
    )


def _finite_coordinate(lat: Any, lon: Any) -> tuple[float, float] | None:
    try:
        a, b = float(lat), float(lon)
    except (ValueError, TypeError):
        return None
    if not (math.isfinite(a) and math.isfinite(b) and 35 <= a <= 48 and 6 <= b <= 19):
        return None
    return a, b


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * math.asin(min(1.0, math.sqrt(h)))


def _require(df: pd.DataFrame, required: set[str], name: str, id_column: str | None = None) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'{name} missing required columns: {missing}')
    if id_column:
        if df[id_column].isna().any() or df[id_column].astype(str).duplicated().any():
            raise ValueError(f'{name} contains missing or duplicated {id_column}')


def _nearest(points: pd.DataFrame, target: tuple[float, float], *, id_col: str,
             name_col: str = 'name') -> tuple[str, str, float | None, int]:
    distances = []
    for row in points.to_dict(orient='records'):
        coord = _finite_coordinate(row.get('latitude'), row.get('longitude'))
        if coord is None:
            continue
        dist = haversine_m(target, coord)
        distances.append((dist, str(row[id_col]), str(row.get(name_col) or '') if not _missing(row.get(name_col)) else ''))
    if not distances:
        return '', '', None, 0
    distances.sort(key=lambda entry: (entry[0], entry[1]))
    nearest = distances[0]
    return nearest[1], nearest[2], round(nearest[0], 3), sum(d <= DISTANCE_REVIEW_M for d, _, _ in distances)


def audit_health_recovery(
    matches: pd.DataFrame,
    health_final: pd.DataFrame,
    v2_services: pd.DataFrame,
    osm_only: pd.DataFrame,
    *,
    municipality_code: str,
    anncsu_snapshot: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select unresolved matched health records; keep all recommendations gated.

    Only record-level comparisons; never treats the absence of same-category OSM
    as independent proof of a real facility or a correct physical entrance.
    """
    if not re.fullmatch(r'\d{6}', municipality_code):
        raise ValueError('municipality_code must be 6 digits')
    if not re.fullmatch(r'\d{8}', anncsu_snapshot):
        raise ValueError('anncsu_snapshot must be YYYYMMDD')
    datetime.strptime(anncsu_snapshot, '%Y%m%d')
    _require(matches, {
        'entity_type', 'entity_id', 'municipality_code', 'match_status',
        'existing_latitude', 'existing_longitude', 'source_address', 'reference_vintage',
        'candidate_count', 'candidate_access_id', 'anncsu_latitude', 'anncsu_longitude',
        'anncsu_method_code', 'anncsu_snapshot_date', 'coordinate_accepted_for_accessibility',
    }, 'ANNCSU B6A2 evidence')
    _require(health_final, {
        'service_site_id', 'name', 'subcategory', 'resolution_status',
        'usable_for_accessibility', 'latitude', 'longitude', 'municipality_code',
    }, 'Health final', 'service_site_id')
    _require(v2_services, {
        'service_id', 'service_type', 'legacy_service_site_id',
        'legacy_usable_for_accessibility', 'latitude', 'longitude', 'municipality_code',
    }, 'V2 services', 'service_id')
    _require(osm_only, {
        'service_site_id', 'category', 'subcategory', 'name',
        'latitude', 'longitude', 'municipality_code',
    }, 'OSM-only', 'service_site_id')
    for label, df in [('ANNCSU B6A2 evidence', matches), ('Health final', health_final),
                      ('V2 services', v2_services), ('OSM-only', osm_only)]:
        codes = set(df['municipality_code'].dropna().astype(str))
        if codes != {municipality_code}:
            raise ValueError(f'{label} has invalid municipality codes: {codes}')
    if (matches['anncsu_snapshot_date'].dropna().astype(str) != anncsu_snapshot).any():
        raise ValueError('ANNCSU candidate snapshot mismatch')
    if matches.duplicated(subset=['entity_type', 'entity_id']).any():
        raise ValueError('Duplicate ANNCSU source identities')

    candidates = matches.loc[
        matches['entity_type'].isin(SOURCE_TYPES)
        & matches['match_status'].eq('exact_single_access_candidate')
        & matches['existing_latitude'].isna()
        & matches['existing_longitude'].isna()
    ].copy()
    health_lookup = health_final.set_index('service_site_id', drop=False)
    v2_health = v2_services.loc[
        v2_services['legacy_service_site_id'].astype(str).str.startswith('HEALTH::')
    ].copy()
    v2_lookup = v2_health.set_index('legacy_service_site_id', drop=False)
    if v2_lookup.index.has_duplicates:
        raise ValueError('Duplicate health V2 legacy_service_site_id; cannot establish one-to-one identity')

    rows = []
    already_resolved = 0
    for source in candidates.sort_values(['entity_type', 'entity_id']).to_dict(orient='records'):
        sid = str(source['entity_id'])
        kind = str(source['entity_type'])
        if sid not in health_lookup.index:
            raise ValueError(f'ANNCSU source identity missing from health final: {sid}')
        legacy_id = 'HEALTH::' + sid
        if legacy_id not in v2_lookup.index:
            raise ValueError(f'ANNCSU source identity missing from V2: {sid}')
        health = health_lookup.loc[sid]
        v2 = v2_lookup.loc[legacy_id]
        if str(health['subcategory']) != kind or str(v2['service_type']) != SOURCE_TYPES[kind]:
            raise ValueError(f'Wrong service category/classification for {sid}')
        if _truth(source['coordinate_accepted_for_accessibility']):
            raise ValueError('B6A2 evidence unexpectedly promotes candidate coordinates')
        if int(source['candidate_count']) != 1:
            raise ValueError('Non-unique access labeled as exact single candidate')
        coords = _finite_coordinate(source['anncsu_latitude'], source['anncsu_longitude'])
        if coords is None:
            raise ValueError(f'Invalid ANNCSU coordinates: {sid}')
        health_usable = _truth(health['usable_for_accessibility'])
        v2_usable = _truth(v2['legacy_usable_for_accessibility'])
        if health_usable != v2_usable:
            raise ValueError(f'Legacy/V2 usable flags disagree: {sid}')
        if health_usable:
            already_resolved += 1
            continue

        # Do not classify all healthcare POIs as 'hospital': keep like-with-like.
        osm_same_type = osm_only.loc[
            (osm_only['category'].astype(str) == 'health')
            & (osm_only['subcategory'].astype(str) == kind)
        ]
        valid_health_same_type = health_final.loc[
            health_final['subcategory'].astype(str).eq(kind)
            & health_final['service_site_id'].astype(str).ne(sid)
            & health_final['usable_for_accessibility'].map(_truth)
        ]
        osm_id, osm_name, osm_dist, osm_near = _nearest(osm_same_type, coords, id_col='service_site_id')
        nearest_health_id, _, health_dist, health_near = _nearest(valid_health_same_type, coords, id_col='service_site_id')
        rows.append({
            'entity_type': kind, 'entity_id': sid, 'service_id_v2': str(v2['service_id']),
            'service_type_v2': str(v2['service_type']), 'name': str(health['name']),
            'source_address': str(source['source_address']),
            'health_reference_vintage': str(source['reference_vintage']),
            'anncsu_snapshot_date': anncsu_snapshot,
            'anncsu_access_id': str(source['candidate_access_id']),
            'anncsu_method_code': '' if _missing(source['anncsu_method_code']) else str(source['anncsu_method_code']),
            'anncsu_latitude': coords[0], 'anncsu_longitude': coords[1],
            'health_final_resolution_status': str(health['resolution_status']),
            'health_final_usable': health_usable, 'v2_legacy_usable': v2_usable,
            'v2_has_coordinates': _finite_coordinate(v2['latitude'], v2['longitude']) is not None,
            'nearest_osm_same_type_id': osm_id, 'nearest_osm_same_type_name': osm_name,
            'nearest_osm_same_type_distance_m': osm_dist,
            'osm_same_type_within_150m': osm_near,
            'nearest_valid_health_same_type_id': nearest_health_id,
            'nearest_valid_health_same_type_distance_m': health_dist,
            'valid_health_same_type_within_150m': health_near,
            'municipality_polygon_checked': False, 'site_entrance_verified': False,
            'independent_identity_verified': False,
            'operational_in_reference_period_verified': False,
            'service_scope_verified': False,
            'review_status': 'manual_verification_required',
            'scenario_eligible': False,
        })
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    report = {
        'policy': REVIEW_POLICY, 'municipality_code': municipality_code,
        'anncsu_snapshot_date': anncsu_snapshot,
        'exact_health_candidates_missing_silver_coordinates': len(candidates),
        'already_usable_downstream': already_resolved,
        'still_unresolved': len(result),
        'scenario_eligible': 0,
        'unresolved_by_entity_type': {
            t: int((result['entity_type'] == t).sum()) for t in SOURCE_TYPES
        },
        'nearby_same_type_osm_review_150m': int(result['osm_same_type_within_150m'].gt(0).sum()) if len(result) else 0,
        'nearby_same_type_health_review_150m': int(result['valid_health_same_type_within_150m'].gt(0).sum()) if len(result) else 0,
        'notes': [
            'ANNCSU gives municipal civic access coordinates, not a verified service entrance.',
            'ANNCSU 2026 location cannot automatically certify an older health reference date.',
            'The ANNCSU method code is retained without assuming precision or accuracy.',
            'Same-category 150 m checks are duplicate warnings, not evidence of absence of other duplicates.',
            'The OSM-only baseline is not a complete inventory; missing OSM hospitals are not absence of hospitals.',
            'Hospital establishments may have a specialized rehabilitation/care scope, not general hospital services.',
            'No municipal polygon check, operation-date audit or independent verification is performed here.',
            'This tool never promotes coordinates or edits the baseline, v2, ANNCSU inputs or routing.',
        ],
    }
    return result, report


def paths_for(root: Path, code: str, snapshot: str, school_year: str,
              health_date: str) -> dict[str, Path]:
    label = health_date.replace('-', '')
    data = root / 'data'
    return {
        'anncsu_evidence': data / 'features' / 'anncsu' / code / f'service_address_matches_{snapshot}.csv',
        'anncsu_manifest': data / 'features' / 'anncsu' / code / f'service_address_matches_{snapshot}.json',
        'health_final': data / 'processed' / 'salute' / code / f'health_sites_final_{label}.parquet',
        'v2_services': data / 'processed' / 'services' / code / f'service_entities_v2_{school_year}_{label}.parquet',
        'osm_only': data / 'processed' / 'services' / code / 'service_sites_osm_only.parquet',
    }


def read_verified_inputs(paths: dict[str, Path], code: str, snapshot: str) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {name}: {path}')
    record = json.loads(paths['anncsu_manifest'].read_text(encoding='utf-8'))
    expected = record.get('output_csv_sha256')
    if (record.get('municipality_code') != code
        or record.get('anncsu_snapshot_date') != snapshot
        or not re.fullmatch(r'[0-9a-f]{64}', str(expected or ''))):
        raise ValueError('B6A2 evidence manifest has wrong identity or missing checksum')
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes['anncsu_evidence'] != expected:
        raise ValueError('B6A2 evidence CSV checksum mismatch; do not use stale evidence')
    datasets = {
        'matches': pd.read_csv(paths['anncsu_evidence'], dtype={'municipality_code': str, 'anncsu_snapshot_date': str}),
        'health_final': pd.read_parquet(paths['health_final']),
        'v2_services': pd.read_parquet(paths['v2_services']),
        'osm_only': pd.read_parquet(paths['osm_only']),
    }
    return datasets, hashes


def write_evidence(root: Path, code: str, snapshot: str, label: str,
                   evidence: pd.DataFrame, report: dict[str, Any]) -> tuple[Path, Path, bool]:
    directory = root / 'data' / 'features' / 'anncsu' / code
    stem = f'health_recovery_review_{snapshot}_{label}'
    csv_path, json_path = directory / f'{stem}.csv', directory / f'{stem}_manifest.json'
    # Deterministic bytes: a cached report must match freshly computed evidence.
    serialized = evidence.to_csv(index=False, lineterminator='\n').encode('utf-8')
    expected_output_hash = hashlib.sha256(serialized).hexdigest()
    # No overwrite of report generated from a different input snapshot / source.
    if csv_path.exists() or json_path.exists():
        if not csv_path.is_file() or not json_path.is_file():
            raise FileExistsError('Incomplete B6B1 results exist')
        old = json.loads(json_path.read_text(encoding='utf-8'))
        equal = all(old.get(key) == report.get(key) for key in (
            'policy', 'municipality_code', 'anncsu_snapshot_date', 'input_sha256',
            'exact_health_candidates_missing_silver_coordinates',
            'already_usable_downstream', 'still_unresolved',
            'scenario_eligible', 'unresolved_by_entity_type',
            'nearby_same_type_osm_review_150m', 'nearby_same_type_health_review_150m',
        ))
        if (not equal or old.get('output_csv_sha256') != sha256_file(csv_path)
                or old.get('output_csv_sha256') != expected_output_hash):
            raise FileExistsError('Existing B6B1 results differ; refusing silent overwrite')
        return csv_path, json_path, True
    directory.mkdir(parents=True, exist_ok=True)
    tmp_csv, tmp_json = directory / f'{stem}.csv.tmp', directory / f'{stem}_manifest.json.tmp'
    if tmp_csv.exists() or tmp_json.exists():
        raise FileExistsError('Partial temporary B6B1 report exists')
    try:
        tmp_csv.write_bytes(serialized)
        final_report = dict(report,
                            generated_at_utc=datetime.now(timezone.utc).isoformat(),
                            output_csv=str(csv_path),
                            output_csv_sha256=sha256_file(tmp_csv))
        tmp_json.write_text(json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
        tmp_csv.replace(csv_path)
        tmp_json.replace(json_path)
    finally:
        tmp_csv.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
    return csv_path, json_path, False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--municipality-code', required=True)
    parser.add_argument('--anncsu-snapshot', default='20260915')
    parser.add_argument('--school-year', default='202425')
    parser.add_argument('--health-reference-date', default='2025-06-30')
    parser.add_argument('--write-report', action='store_true', help='Explicit opt-in; save evidence CSV and JSON')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'\d{6}', args.municipality_code):
        parser.error('--municipality-code must contain six digits')
    try:
        datetime.strptime(args.anncsu_snapshot, '%Y%m%d')
        datetime.strptime(args.health_reference_date, '%Y-%m-%d')
    except ValueError:
        parser.error('Invalid ANNCSU snapshot or health reference date')
    return args


def run(args: argparse.Namespace, *, root: Path = ROOT) -> dict[str, Any]:
    paths = paths_for(root, args.municipality_code, args.anncsu_snapshot,
                      args.school_year, args.health_reference_date)
    frames, hashes = read_verified_inputs(paths, args.municipality_code, args.anncsu_snapshot)
    evidence, report = audit_health_recovery(**frames,
        municipality_code=args.municipality_code,
        anncsu_snapshot=args.anncsu_snapshot)
    report['input_sha256'] = hashes
    print('=== B6B1 ANNCSU HEALTH RECOVERY REVIEW — EVIDENCE ONLY ===')
    print(f'Municipality: {args.municipality_code} | Snapshot: {args.anncsu_snapshot}')
    for key in ('exact_health_candidates_missing_silver_coordinates',
                'already_usable_downstream', 'still_unresolved', 'scenario_eligible'):
        print(f'{key}: {report[key]}')
    if len(evidence):
        print(evidence[['entity_id', 'name', 'service_type_v2',
                       'nearest_osm_same_type_distance_m',
                       'nearest_valid_health_same_type_distance_m', 'review_status']].to_string(index=False))
    if not args.write_report:
        print('READ-ONLY — no files modified; use --write-report to save evidence')
        return report
    csv_path, json_path, cached = write_evidence(root, args.municipality_code,
        args.anncsu_snapshot, args.health_reference_date.replace('-', ''), evidence, report)
    print(f'Evidence CSV: {csv_path}')
    print(f'Manifest: {json_path} | Cached: {cached}')
    return report


if __name__ == '__main__':
    run(parse_args())
