"""B6B2: conservative geospatial evidence review for ANNCSU health candidates.

A census-section footprint is *not* an authoritative administrative boundary.
Network-node snapping is not evidence that the service exists at that access.
No inputs, canonical services, network attachments, baselines or indicators change.

Usage:
  PYTHONPATH=src python src/quality/audit_anncsu_health_geospatial.py \\
    --municipality-code 034027 --anncsu-snapshot 20260915
  # add --write-report to persist evidence-only outputs
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

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from core.analysis_spec import TransportMode
from quality.audit_anncsu_health_recovery import paths_for as b6b1_input_paths
from transformation.build_network_attachments_v2 import (
    prepare_network_nodes, build_network_components, snap_entities, graph_checksum,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = 'b6b2_anncsu_health_geospatial_evidence_only_v1'
MODES = ('walk', 'drive')
REPORT_COLUMNS = (
    'entity_id', 'name', 'service_type_v2', 'anncsu_latitude', 'anncsu_longitude',
    'municipality_boundary_verified', 'inside_istat_census_footprint',
    'walk_snapped', 'walk_snap_distance_m', 'walk_node_id',
    'walk_network_component_id', 'walk_is_largest_component',
    'drive_snapped', 'drive_snap_distance_m', 'drive_node_id',
    'drive_network_component_id', 'drive_is_largest_component',
    'site_entrance_verified', 'independent_identity_verified',
    'operational_in_reference_period_verified', 'service_scope_verified',
    'review_status', 'scenario_eligible',
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def explicit_true(value: Any) -> bool:
    return ((isinstance(value, (bool, np.bool_)) and bool(value))
            or (isinstance(value, str) and value.strip().lower() == 'true'))


def load_b6b1_review(csv_path: Path, manifest_path: Path, municipality_code: str,
                    anncsu_snapshot: str, *, source_paths: dict[str, Path] | None = None) -> pd.DataFrame:
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError('B6B1 CSV and manifest required; run B6B1 --write-report first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('policy') != 'b6b1_anncsu_health_recovery_review_v1'
            or manifest.get('municipality_code') != municipality_code
            or manifest.get('anncsu_snapshot_date') != anncsu_snapshot
            or manifest.get('output_csv_sha256') != sha256_file(csv_path)):
        raise ValueError('B6B1 evidence identity/checksum mismatch; refusing stale input')
    if source_paths is not None:
        source_hashes = manifest.get('input_sha256')
        if not isinstance(source_hashes, dict) or set(source_hashes) != set(source_paths):
            raise ValueError('B6B1 source manifest missing required input hashes')
        for label, path in source_paths.items():
            if not path.is_file() or sha256_file(path) != source_hashes[label]:
                raise ValueError(f'B6B1 input has changed since audit: {label}')
    df = pd.read_csv(csv_path, dtype={'anncsu_snapshot_date': str})
    needed = {'entity_id', 'name', 'service_type_v2', 'anncsu_latitude',
              'anncsu_longitude', 'review_status', 'scenario_eligible',
              'anncsu_snapshot_date', 'health_final_usable', 'v2_legacy_usable'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'B6B1 CSV lacks required columns: {sorted(missing)}')
    if (df['entity_id'].isna().any() or df['entity_id'].duplicated().any()
            or (df['anncsu_snapshot_date'] != anncsu_snapshot).any()):
        raise ValueError('B6B1 has duplicate/missing IDs or wrong snapshot')
    if (df['scenario_eligible'].map(explicit_true).any()
            or df['health_final_usable'].map(explicit_true).any()
            or df['v2_legacy_usable'].map(explicit_true).any()
            or not df['review_status'].eq('manual_verification_required').all()):
        raise ValueError('B6B1 contains previously promoted/validated services; not an unresolved review')
    if not df['service_type_v2'].isin(['pharmacy', 'hospital_establishment']).all():
        raise ValueError('Unexpected service category')
    return df.sort_values('entity_id', kind='mergesort').reset_index(drop=True)


def census_footprint(areas: gpd.GeoDataFrame):
    """Only a diagnostic footprint; never label this an administrative polygon."""
    if not isinstance(areas, gpd.GeoDataFrame) or areas.crs is None:
        raise ValueError('ISTAT census areas must be a GeoDataFrame with known CRS')
    geom = areas.geometry
    if (len(areas) == 0 or geom.isna().any() or geom.is_empty.any()
            or not geom.geom_type.isin(['Polygon', 'MultiPolygon']).all()
            or not geom.is_valid.all()):
        raise ValueError('ISTAT census geometries missing, empty, invalid, or non-polygonal')
    footprint = areas.to_crs('EPSG:4326').geometry.union_all()
    if footprint.is_empty or not footprint.is_valid:
        raise ValueError('Invalid census footprint')
    return footprint


def _coord(lat: Any, lon: Any) -> tuple[float, float]:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError) as exc:
        raise ValueError('Candidate coordinate missing or non-numeric') from exc
    if not (math.isfinite(lat) and math.isfinite(lon)
            and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError('Candidate coordinate out of WGS84 range')
    return lat, lon


def snap_review_points(candidates: pd.DataFrame, nodes: gpd.GeoDataFrame,
                       edges: gpd.GeoDataFrame, *, mode: str,
                       checksum: str, max_snap_distance_m: float) -> pd.DataFrame:
    if mode not in MODES or not math.isfinite(max_snap_distance_m) or max_snap_distance_m <= 0:
        raise ValueError('Invalid network mode or maximum snap distance')
    if not checksum or not isinstance(checksum, str):
        raise ValueError('Network checksum required')
    prepared = prepare_network_nodes(nodes)
    connected_nodes, _ = build_network_components(prepared, edges)
    points = gpd.GeoDataFrame(
        {'entity_id': candidates['entity_id'].astype(str).to_list()},
        geometry=[Point(_coord(r['anncsu_latitude'], r['anncsu_longitude'])[1],
                        _coord(r['anncsu_latitude'], r['anncsu_longitude'])[0])
                  for _, r in candidates.iterrows()],
        crs='EPSG:4326',
    )
    if points.empty:
        return pd.DataFrame(columns=['entity_id', 'snapped', 'snap_distance_m', 'node_id',
                                     'network_component_id', 'is_largest_component'])
    result = snap_entities(points, connected_nodes, entity_kind='service',
                           mode=TransportMode(mode),
                           max_snap_distance_m=max_snap_distance_m,
                           graph_checksum_value=checksum)
    if len(result) != len(candidates):
        raise RuntimeError('Candidate row-conservation failure')
    return result


def build_geospatial_review(candidates: pd.DataFrame, footprint,
                            attachments: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if set(attachments) != set(MODES):
        raise ValueError('Both walk and drive attachment audits required')
    if candidates['entity_id'].duplicated().any():
        raise ValueError('Duplicate candidate entities')
    result = candidates[['entity_id', 'name', 'service_type_v2',
                         'anncsu_latitude', 'anncsu_longitude']].copy()
    result['municipality_boundary_verified'] = False
    result['inside_istat_census_footprint'] = [
        bool(footprint.covers(Point(_coord(r['anncsu_latitude'], r['anncsu_longitude'])[1],
                                    _coord(r['anncsu_latitude'], r['anncsu_longitude'])[0])))
        for _, r in result.iterrows()
    ]
    for mode in MODES:
        att = attachments[mode]
        required = {'entity_id', 'snapped', 'snap_distance_m', 'node_id',
                    'network_component_id', 'is_largest_component'}
        if required - set(att.columns) or att['entity_id'].duplicated().any():
            raise ValueError(f'Malformed {mode} network attachment result')
        right = att[list(required)].copy()
        right = right.rename(columns={c: f'{mode}_{c}' for c in required - {'entity_id'}})
        result = result.merge(right, on='entity_id', how='left', validate='one_to_one', indicator=True)
        if not result['_merge'].eq('both').all():
            raise ValueError(f'Missing {mode} attachments for some candidates')
        result = result.drop(columns='_merge')
        if result[f'{mode}_snapped'].map(explicit_true).any() and result.loc[
            result[f'{mode}_snapped'].map(explicit_true), f'{mode}_node_id'].isna().any():
            raise ValueError(f'Snapped {mode} destinations with missing node')
    for col in ('site_entrance_verified', 'independent_identity_verified',
                'operational_in_reference_period_verified', 'service_scope_verified',
                'scenario_eligible'):
        result[col] = False
    result['review_status'] = 'manual_verification_required'
    result = result.loc[:, REPORT_COLUMNS].sort_values('entity_id', kind='mergesort').reset_index(drop=True)
    report = {
        'policy': POLICY,
        'total_unresolved_candidates': len(result),
        'inside_istat_census_footprint': int(result['inside_istat_census_footprint'].sum()),
        'walk_snapped': int(result['walk_snapped'].map(explicit_true).sum()),
        'drive_snapped': int(result['drive_snapped'].map(explicit_true).sum()),
        'municipality_boundary_verified': 0,
        'scenario_eligible': 0,
        'notes': [
            'ISTAT census-footprint containment is diagnostic; it is NOT official municipal boundary verification.',
            'ANNCSU coordinates refer to a civic address/access, not necessarily to the health facility entrance.',
            'Graph component IDs are weak connectivity; reachability, especially drive directionality, not certified.',
            'Network snapping does not independently verify facility identity, operation or service scope.',
            'ANNCSU 2026 is later than healthcare vintages; no historical-operation certification performed.',
            'No routing baseline, canonical service, ANNCSU record or network attachment is modified.',
        ],
    }
    return result, report


def paths_for(root: Path, code: str, snapshot: str, year: str,
              health_date: str) -> dict[str, Path]:
    label = health_date.replace('-', '')
    return {
        'b6b1_csv': root / 'data/features/anncsu' / code / f'health_recovery_review_{snapshot}_{label}.csv',
        'b6b1_manifest': root / 'data/features/anncsu' / code / f'health_recovery_review_{snapshot}_{label}_manifest.json',
        'census_footprint': root / 'data/processed/istat' / f'{code}_census_areas_2021.parquet',
        'walk_nodes': root / 'data/processed/osm' / code / 'walk_nodes.parquet',
        'walk_edges': root / 'data/processed/osm' / code / 'walk_edges.parquet',
        'drive_nodes': root / 'data/processed/osm' / code / 'drive_nodes.parquet',
        'drive_edges': root / 'data/processed/osm' / code / 'drive_edges.parquet',
    }


def write_report(root: Path, code: str, snapshot: str, health_label: str,
                 df: pd.DataFrame, report: dict[str, Any]) -> tuple[Path, Path, bool]:
    folder = root / 'data/features/anncsu' / code
    stem = f'health_geospatial_review_{snapshot}_{health_label}'
    csv_path = folder / f'{stem}.csv'
    manifest_path = folder / f'{stem}_manifest.json'
    csv_bytes = df.to_csv(index=False, lineterminator='\n').encode('utf-8')
    expected_hash = hashlib.sha256(csv_bytes).hexdigest()
    if csv_path.exists() or manifest_path.exists():
        if not csv_path.is_file() or not manifest_path.is_file():
            raise FileExistsError('Partial B6B2 audit exists')
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (old.get('output_csv_sha256') != expected_hash
                or sha256_file(csv_path) != expected_hash
                or any(old.get(key) != report.get(key) for key in report)):
            raise FileExistsError('Existing B6B2 report differs; refusing silent overwrite')
        return csv_path, manifest_path, True
    folder.mkdir(parents=True, exist_ok=True)
    tmp_csv = folder / f'{stem}.csv.tmp'
    tmp_json = folder / f'{stem}_manifest.json.tmp'
    if tmp_csv.exists() or tmp_json.exists():
        raise FileExistsError('Temporary B6B2 report exists')
    try:
        tmp_csv.write_bytes(csv_bytes)
        full_manifest = dict(report, generated_at_utc=datetime.now(timezone.utc).isoformat(),
                             output_csv=str(csv_path), output_csv_sha256=expected_hash)
        tmp_json.write_text(json.dumps(full_manifest, indent=2, ensure_ascii=False,
                                       sort_keys=True), encoding='utf-8')
        tmp_csv.replace(csv_path)
        tmp_json.replace(manifest_path)
    finally:
        tmp_csv.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
    return csv_path, manifest_path, False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--municipality-code', required=True)
    parser.add_argument('--anncsu-snapshot', default='20260915')
    parser.add_argument('--school-year', default='202425')
    parser.add_argument('--health-reference-date', default='2025-06-30')
    parser.add_argument('--max-snap-distance-m', type=float, default=1000.0)
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'\d{6}', args.municipality_code):
        parser.error('Invalid municipality code')
    try:
        datetime.strptime(args.anncsu_snapshot, '%Y%m%d')
        datetime.strptime(args.health_reference_date, '%Y-%m-%d')
    except ValueError:
        parser.error('Invalid date')
    if not math.isfinite(args.max_snap_distance_m) or args.max_snap_distance_m <= 0:
        parser.error('--max-snap-distance-m must be positive and finite')
    return args


def run(args: argparse.Namespace, *, root: Path = ROOT) -> dict[str, Any]:
    paths = paths_for(root, args.municipality_code, args.anncsu_snapshot,
                      args.school_year, args.health_reference_date)
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {label}: {path}')
    hashes = {label: sha256_file(path) for label, path in paths.items()}
    upstream = b6b1_input_paths(root, args.municipality_code, args.anncsu_snapshot,
                                args.school_year, args.health_reference_date)
    candidates = load_b6b1_review(paths['b6b1_csv'], paths['b6b1_manifest'],
                                  args.municipality_code, args.anncsu_snapshot,
                                  source_paths=upstream)
    areas = gpd.read_parquet(paths['census_footprint'])
    footprint = census_footprint(areas)
    att: dict[str, pd.DataFrame] = {}
    for mode in MODES:
        nodes = gpd.read_parquet(paths[f'{mode}_nodes'])
        edges = gpd.read_parquet(paths[f'{mode}_edges'])
        graph_hash = graph_checksum(paths[f'{mode}_nodes'], paths[f'{mode}_edges'])
        att[mode] = snap_review_points(candidates, nodes, edges, mode=mode,
                                       checksum=graph_hash,
                                       max_snap_distance_m=args.max_snap_distance_m)
    review, report = build_geospatial_review(candidates, footprint, att)
    report.update(municipality_code=args.municipality_code,
                  anncsu_snapshot_date=args.anncsu_snapshot,
                  health_reference_date=args.health_reference_date,
                  school_year=args.school_year,
                  max_snap_distance_m=args.max_snap_distance_m,
                  input_sha256=hashes)
    print('=== B6B2 ANNCSU HEALTH GEOSPATIAL AUDIT — EVIDENCE ONLY ===')
    for label in ('total_unresolved_candidates', 'inside_istat_census_footprint',
                  'walk_snapped', 'drive_snapped', 'scenario_eligible'):
        print(f'{label}: {report[label]}')
    if len(review):
        print(review[['entity_id', 'inside_istat_census_footprint',
                      'walk_snap_distance_m', 'walk_network_component_id',
                      'drive_snap_distance_m', 'drive_network_component_id',
                      'review_status']].to_string(index=False))
    if args.write_report:
        csv, manifest, cached = write_report(root, args.municipality_code,
            args.anncsu_snapshot, args.health_reference_date.replace('-', ''), review, report)
        print(f'Evidence CSV: {csv}\nManifest: {manifest} | Cached: {cached}')
    else:
        print('READ-ONLY — no files modified; use --write-report to save evidence')
    return report


if __name__ == '__main__':
    run(parse_args())
