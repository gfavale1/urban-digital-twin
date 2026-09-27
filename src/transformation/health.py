# Canonical Health transformation module.
# Behavior-preserving consolidation of spatial QA and finalization.

import argparse
import json
import math
import os
import time
import unicodedata
from pathlib import Path
import pandas as pd
import geopandas as gpd
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz
from shapely import wkt
from sqlalchemy import create_engine, text
qa_ROOT = Path(__file__).resolve().parents[2]
qa_PROCESSED_SALUTE_DIR = qa_ROOT / 'data' / 'processed' / 'salute'
qa_RAW_GEOCODING_DIR = qa_ROOT / 'data' / 'raw' / 'salute' / 'geocoding' / 'nominatim'
qa_FEATURES_SALUTE_DIR = qa_ROOT / 'data' / 'features' / 'salute'
qa_NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search'
qa_REQUEST_DELAY_SECONDS = 1.1
qa_MISSING_TEXT_VALUES = {'', '-', 'nan', 'none', 'null', 'n/a', 'na'}

def qa_parse_args():
    parser = argparse.ArgumentParser(description='QA spaziale dei siti Health. Individua coordinate sorgente sospette (es. duplicati su indirizzi diversi), geocodifica solo i record necessari o, opzionalmente, tutti i record, e confronta coordinate sorgente e candidati Nominatim.')
    parser.add_argument('--municipality-code', required=True, help='Codice ISTAT comunale a 6 cifre.')
    parser.add_argument('--pharmacy-reference-date', default='2025-06-30', help='Snapshot farmacie YYYY-MM-DD. Default: 2025-06-30.')
    parser.add_argument('--hospital-year', default='2023', help='Anno dataset ospedaliero. Default: 2023.')
    parser.add_argument('--validate-all', action='store_true', help='Geocodifica anche i record con coordinate sorgente non sospette. Utile nel PoC; non necessario nella pipeline operativa nazionale.')
    parser.add_argument('--refresh', action='store_true', help='Ignora la cache Nominatim e ripete le richieste.')
    args = parser.parse_args()
    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    if not args.municipality_code.isdigit() or len(args.municipality_code) != 6:
        raise ValueError('--municipality-code deve avere esattamente 6 cifre.')
    args.pharmacy_reference_date = pd.Timestamp(args.pharmacy_reference_date).normalize()
    return args

def qa_normalize_text(value):
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    value = str(value).strip()
    if value.lower() in qa_MISSING_TEXT_VALUES:
        return ''
    return value

def qa_ascii_normalize(value):
    value = qa_normalize_text(value)
    value = unicodedata.normalize('NFKD', value)
    value = ''.join((char for char in value if not unicodedata.combining(char)))
    return ' '.join(value.upper().split())

def qa_haversine_m(lat1, lon1, lat2, lon2):
    if any((pd.isna(value) for value in [lat1, lon1, lat2, lon2])):
        return None
    radius = 6371008.8
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def qa_get_database_engine():
    load_dotenv(qa_ROOT / '.env')
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise RuntimeError('DATABASE_URL non definito nel file .env.')
    return create_engine(database_url)

def qa_load_municipality(engine, municipality_code):
    query = text('\n        SELECT\n            istat_code,\n            name,\n            ST_AsText(geometry) AS geometry_wkt\n        FROM municipality\n        WHERE istat_code = :istat_code;\n        ')
    with engine.connect() as connection:
        row = connection.execute(query, {'istat_code': municipality_code}).mappings().first()
    if row is None:
        raise RuntimeError("Comune non presente in PostGIS. Eseguire prima l'ingestion ISTAT.")
    geometry = wkt.loads(row['geometry_wkt'])
    minx, miny, maxx, maxy = geometry.bounds
    return {'istat_code': row['istat_code'], 'name': str(row['name']).strip(), 'geometry': geometry, 'viewbox': (float(minx), float(maxy), float(maxx), float(miny))}

def qa_load_health_sites(args):
    directory = qa_PROCESSED_SALUTE_DIR / args.municipality_code
    reference_label = args.pharmacy_reference_date.strftime('%Y%m%d')
    pharmacy_path = directory / f'pharmacy_sites_{reference_label}.parquet'
    hospital_path = directory / f'hospital_sites_{args.hospital_year}.parquet'
    if not pharmacy_path.exists():
        raise FileNotFoundError(f'Silver farmacie non trovato: {pharmacy_path}')
    if not hospital_path.exists():
        raise FileNotFoundError(f'Silver ospedali non trovato: {hospital_path}')
    pharmacies = pd.read_parquet(pharmacy_path)
    hospitals = pd.read_parquet(hospital_path)
    return (pharmacies, hospitals)

def qa_add_municipality_coordinate_diagnostics(df, municipality, tolerance_m=500.0):
    """
    Valuta la coerenza spaziale delle coordinate sorgente rispetto
    al comune dichiarato.

    Una coordinata fuori dal comune non viene automaticamente
    scartata: diventa sospetta solo quando la distanza dal territorio
    comunale supera una tolleranza metrica generica.

    Questo evita falsi positivi per strutture immediatamente a ridosso
    del confine amministrativo.
    """
    result = df.copy()
    result['source_coordinate_inside_municipality'] = pd.Series(pd.NA, index=result.index, dtype='boolean')
    result['source_coordinate_distance_to_municipality_m'] = pd.Series(pd.NA, index=result.index, dtype='Float64')
    result['source_coordinate_outside_municipality'] = False
    present = result['source_coordinate_present'].fillna(False).astype(bool)
    present_index = result.index[present]
    if len(present_index) == 0:
        return result
    municipality_geometry = gpd.GeoSeries([municipality['geometry']], crs='EPSG:4326')
    projected_crs = municipality_geometry.estimate_utm_crs()
    if projected_crs is None:
        raise RuntimeError('Impossibile determinare un CRS metrico per il controllo delle coordinate Health.')
    municipality_metric = municipality_geometry.to_crs(projected_crs).iloc[0]
    points = gpd.GeoSeries(gpd.points_from_xy(result.loc[present_index, 'source_longitude'], result.loc[present_index, 'source_latitude']), index=present_index, crs='EPSG:4326')
    points_metric = points.to_crs(projected_crs)
    inside = points_metric.within(municipality_metric) | points_metric.touches(municipality_metric)
    distances = points_metric.distance(municipality_metric)
    result.loc[present_index, 'source_coordinate_inside_municipality'] = inside.astype(bool)
    result.loc[present_index, 'source_coordinate_distance_to_municipality_m'] = distances.astype(float)
    result.loc[present_index, 'source_coordinate_outside_municipality'] = (distances > float(tolerance_m)).astype(bool)
    return result

def qa_detect_suspicious_pharmacy_coordinates(pharmacies, municipality, municipality_tolerance_m=500.0):
    df = pharmacies.copy()
    df['source_latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['source_longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['source_coordinate_present'] = df['source_latitude'].notna() & df['source_longitude'].notna()
    df['normalized_address'] = df['address'].map(qa_ascii_normalize)
    df['coordinate_key'] = None
    present = df['source_coordinate_present']
    df.loc[present, 'coordinate_key'] = df.loc[present, 'source_latitude'].round(6).astype(str) + '|' + df.loc[present, 'source_longitude'].round(6).astype(str)
    group_size = df.loc[present].groupby('coordinate_key').size()
    distinct_addresses = df.loc[present].groupby('coordinate_key')['normalized_address'].nunique()
    df['duplicate_coordinate_group_size'] = df['coordinate_key'].map(group_size).fillna(0).astype(int)
    df['duplicate_coordinate_distinct_addresses'] = df['coordinate_key'].map(distinct_addresses).fillna(0).astype(int)
    duplicate_suspicious = df['source_coordinate_present'] & (df['duplicate_coordinate_group_size'] >= 2) & (df['duplicate_coordinate_distinct_addresses'] >= 2)
    df = qa_add_municipality_coordinate_diagnostics(df, municipality, tolerance_m=municipality_tolerance_m)
    outside_suspicious = df['source_coordinate_outside_municipality'].fillna(False).astype(bool)
    df['source_coordinate_suspicious'] = duplicate_suspicious | outside_suspicious
    reasons = []
    for duplicate, outside in zip(duplicate_suspicious, outside_suspicious):
        if duplicate and outside:
            reason = 'duplicate_and_outside_municipality'
        elif duplicate:
            reason = 'duplicate_coordinate'
        elif outside:
            reason = 'outside_municipality'
        else:
            reason = None
        reasons.append(reason)
    df['source_coordinate_suspicion_reason'] = reasons
    return df

def qa_prepare_hospitals(hospitals, municipality, municipality_tolerance_m=500.0):
    df = hospitals.copy()
    df['source_latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['source_longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['source_coordinate_present'] = df['source_latitude'].notna() & df['source_longitude'].notna()
    df['duplicate_coordinate_group_size'] = 0
    df['duplicate_coordinate_distinct_addresses'] = 0
    df = qa_add_municipality_coordinate_diagnostics(df, municipality, tolerance_m=municipality_tolerance_m)
    df['source_coordinate_suspicious'] = df['source_coordinate_outside_municipality'].fillna(False).astype(bool)
    df['source_coordinate_suspicion_reason'] = None
    df.loc[df['source_coordinate_outside_municipality'].fillna(False).astype(bool), 'source_coordinate_suspicion_reason'] = 'outside_municipality'
    return df

def qa_cache_path(municipality_code):
    directory = qa_RAW_GEOCODING_DIR / municipality_code
    directory.mkdir(parents=True, exist_ok=True)
    return directory / 'health_geocode_cache.json'

def qa_load_cache(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}

def qa_save_cache(path, cache):
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')

def qa_get_session():
    session = requests.Session()
    session.headers.update({'User-Agent': 'urban-digital-twin-thesis/1.0 (academic research; health-site QA)'})
    return session

def qa_build_queries(row, municipality_name):
    name = qa_normalize_text(row.get('name'))
    address = qa_normalize_text(row.get('address'))
    subcategory = qa_normalize_text(row.get('subcategory'))
    queries = []
    if subcategory == 'hospital':
        if name and address:
            queries.append(f'{name}, {address}, {municipality_name}, Italia')
        if address:
            queries.append(f'{address}, {municipality_name}, Italia')
        if name:
            queries.append(f'{name}, {municipality_name}, Italia')
    else:
        if address:
            queries.append(f'{address}, {municipality_name}, Italia')
        if name and address:
            queries.append(f'{name}, {address}, {municipality_name}, Italia')
        if name:
            queries.append(f'{name}, {municipality_name}, Italia')
    return list(dict.fromkeys((query for query in queries if query.strip())))

def qa_query_nominatim(session, query, municipality):
    left, top, right, bottom = municipality['viewbox']
    params = {'q': query, 'format': 'jsonv2', 'limit': 5, 'addressdetails': 1, 'namedetails': 1, 'countrycodes': 'it', 'bounded': 1, 'viewbox': f'{left},{top},{right},{bottom}'}
    response = session.get(qa_NOMINATIM_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()

def qa_municipality_match_score(candidate, municipality_name):
    address = candidate.get('address', {})
    candidate_values = [address.get('city'), address.get('town'), address.get('village'), address.get('municipality')]
    target = qa_ascii_normalize(municipality_name)
    best = 0.0
    for value in candidate_values:
        if value:
            best = max(best, fuzz.ratio(qa_ascii_normalize(value), target))
    return best

def qa_address_match_score(candidate, target_address):
    if not target_address:
        return 0.0
    display_name = qa_normalize_text(candidate.get('display_name'))
    return float(fuzz.token_set_ratio(qa_ascii_normalize(target_address), qa_ascii_normalize(display_name)))

def qa_name_match_score(candidate, target_name):
    if not target_name:
        return 0.0
    namedetails = candidate.get('namedetails', {}) or {}
    possible_names = [namedetails.get('name'), candidate.get('name'), candidate.get('display_name')]
    scores = []
    for value in possible_names:
        if value:
            scores.append(fuzz.token_set_ratio(qa_ascii_normalize(target_name), qa_ascii_normalize(value)))
    return float(max(scores)) if scores else 0.0

def qa_candidate_resolution(candidate):
    addresstype = qa_normalize_text(candidate.get('addresstype')).lower()
    candidate_type = qa_normalize_text(candidate.get('type')).lower()
    detailed = {'house', 'building', 'pharmacy', 'hospital', 'clinic', 'doctors', 'healthcare'}
    if addresstype in detailed or candidate_type in detailed:
        return 'site_or_address_candidate'
    if addresstype in {'road', 'residential', 'pedestrian'} or candidate_type in {'road', 'residential', 'pedestrian'}:
        return 'street_anchor_candidate'
    return 'generic_candidate'

def qa_score_candidate(candidate, row, municipality_name):
    municipality_score = qa_municipality_match_score(candidate, municipality_name)
    address_score = qa_address_match_score(candidate, qa_normalize_text(row.get('address')))
    name_score = qa_name_match_score(candidate, qa_normalize_text(row.get('name')))
    subcategory = qa_normalize_text(row.get('subcategory'))
    if subcategory == 'hospital':
        total = 0.45 * address_score + 0.35 * name_score + 0.2 * municipality_score
    else:
        total = 0.65 * address_score + 0.15 * name_score + 0.2 * municipality_score
    return {'score': float(total), 'municipality_score': float(municipality_score), 'address_score': float(address_score), 'name_score': float(name_score)}

def qa_best_geocoder_candidate(session, cache, cache_file, row, municipality, refresh):
    queries = qa_build_queries(row, municipality['name'])
    all_candidates = []
    for query in queries:
        cache_key = municipality['istat_code'] + '||' + query
        if cache_key in cache and (not refresh):
            results = cache[cache_key]
        else:
            results = qa_query_nominatim(session, query, municipality)
            cache[cache_key] = results
            qa_save_cache(cache_file, cache)
            time.sleep(qa_REQUEST_DELAY_SECONDS)
        for candidate in results:
            scored = qa_score_candidate(candidate, row, municipality['name'])
            all_candidates.append({'query': query, 'candidate': candidate, **scored})
    if not all_candidates:
        return None
    all_candidates.sort(key=lambda item: item['score'], reverse=True)
    return all_candidates[0]

def qa_classify_result(row, best):
    source_present = bool(row['source_coordinate_present'])
    suspicious = bool(row['source_coordinate_suspicious'])
    outside_municipality = bool(row.get('source_coordinate_outside_municipality', False))
    duplicate_suspicious = int(row.get('duplicate_coordinate_group_size', 0) or 0) >= 2 and int(row.get('duplicate_coordinate_distinct_addresses', 0) or 0) >= 2
    if best is None:
        return ('unresolved', 'no_geocoder_candidate')
    candidate = best['candidate']
    geocoder_latitude = float(candidate['lat'])
    geocoder_longitude = float(candidate['lon'])
    distance = None
    if source_present:
        distance = qa_haversine_m(row['source_latitude'], row['source_longitude'], geocoder_latitude, geocoder_longitude)
    resolution = qa_candidate_resolution(candidate)
    strong_candidate = best['municipality_score'] >= 80 and best['address_score'] >= 70 and (best['score'] >= 72)
    if not source_present:
        if strong_candidate:
            return (resolution, 'missing_source_coordinate_with_strong_candidate')
        return ('review', 'missing_source_coordinate_weak_candidate')
    if outside_municipality:
        if strong_candidate:
            return ('source_coordinate_conflict', 'source_coordinate_outside_municipality_with_strong_candidate')
        return ('review', 'source_coordinate_outside_municipality_requires_review')
    if suspicious and duplicate_suspicious:
        if distance is not None and distance > 150 and strong_candidate:
            return ('source_coordinate_conflict', 'duplicate_source_coordinate_conflicts_with_address_geocoder')
        if distance is not None and distance <= 100 and strong_candidate:
            return ('source_coordinate_confirmed', 'duplicate_source_coordinate_but_geocoder_agrees')
        return ('review', 'duplicate_source_coordinate_requires_review')
    if distance is not None and distance <= 100 and strong_candidate:
        return ('source_coordinate_confirmed', 'source_and_geocoder_agree')
    if distance is not None and distance > 250 and strong_candidate:
        return ('source_coordinate_conflict', 'source_and_geocoder_disagree')
    return ('review', 'insufficient_agreement_for_auto_confirmation')

def qa_audit_records(records, municipality, validate_all, refresh):
    cache_file = qa_cache_path(municipality['istat_code'])
    cache = qa_load_cache(cache_file)
    session = qa_get_session()
    outputs = []
    for _, row in records.iterrows():
        should_geocode = validate_all or not bool(row['source_coordinate_present']) or bool(row['source_coordinate_suspicious'])
        base = {'service_site_id': row.get('service_site_id'), 'category': row.get('category'), 'subcategory': row.get('subcategory'), 'source_record_id': row.get('source_record_id'), 'name': row.get('name'), 'address': row.get('address'), 'source_latitude': row.get('source_latitude'), 'source_longitude': row.get('source_longitude'), 'source_coordinate_present': bool(row['source_coordinate_present']), 'duplicate_coordinate_group_size': int(row['duplicate_coordinate_group_size']), 'duplicate_coordinate_distinct_addresses': int(row['duplicate_coordinate_distinct_addresses']), 'source_coordinate_suspicious': bool(row['source_coordinate_suspicious']), 'source_coordinate_inside_municipality': row.get('source_coordinate_inside_municipality'), 'source_coordinate_distance_to_municipality_m': row.get('source_coordinate_distance_to_municipality_m'), 'source_coordinate_outside_municipality': bool(row.get('source_coordinate_outside_municipality', False)), 'source_coordinate_suspicion_reason': row.get('source_coordinate_suspicion_reason'), 'geocoded_for_qa': bool(should_geocode)}
        if not should_geocode:
            outputs.append({**base, 'geocoder_query': None, 'geocoder_display_name': None, 'geocoder_latitude': None, 'geocoder_longitude': None, 'geocoder_type': None, 'geocoder_addresstype': None, 'geocoder_score': None, 'geocoder_address_score': None, 'geocoder_name_score': None, 'geocoder_municipality_score': None, 'source_geocoder_distance_m': None, 'qa_status': 'not_checked', 'qa_reason': 'source_coordinate_not_flagged'})
            continue
        best = qa_best_geocoder_candidate(session=session, cache=cache, cache_file=cache_file, row=row, municipality=municipality, refresh=refresh)
        if best is None:
            status, reason = qa_classify_result(row, best)
            outputs.append({**base, 'geocoder_query': None, 'geocoder_display_name': None, 'geocoder_latitude': None, 'geocoder_longitude': None, 'geocoder_type': None, 'geocoder_addresstype': None, 'geocoder_score': None, 'geocoder_address_score': None, 'geocoder_name_score': None, 'geocoder_municipality_score': None, 'source_geocoder_distance_m': None, 'qa_status': status, 'qa_reason': reason})
            continue
        candidate = best['candidate']
        geocoder_latitude = float(candidate['lat'])
        geocoder_longitude = float(candidate['lon'])
        distance = None
        if bool(row['source_coordinate_present']):
            distance = qa_haversine_m(row['source_latitude'], row['source_longitude'], geocoder_latitude, geocoder_longitude)
        status, reason = qa_classify_result(row, best)
        outputs.append({**base, 'geocoder_query': best['query'], 'geocoder_display_name': candidate.get('display_name'), 'geocoder_latitude': geocoder_latitude, 'geocoder_longitude': geocoder_longitude, 'geocoder_type': candidate.get('type'), 'geocoder_addresstype': candidate.get('addresstype'), 'geocoder_score': round(best['score'], 2), 'geocoder_address_score': round(best['address_score'], 2), 'geocoder_name_score': round(best['name_score'], 2), 'geocoder_municipality_score': round(best['municipality_score'], 2), 'source_geocoder_distance_m': round(distance, 2) if distance is not None else None, 'qa_status': status, 'qa_reason': reason})
    return pd.DataFrame(outputs)

def qa_main():
    args = qa_parse_args()
    engine = qa_get_database_engine()
    municipality = qa_load_municipality(engine, args.municipality_code)
    pharmacies, hospitals = qa_load_health_sites(args)
    pharmacies = qa_detect_suspicious_pharmacy_coordinates(pharmacies, municipality)
    hospitals = qa_prepare_hospitals(hospitals, municipality)
    combined = pd.concat([pharmacies, hospitals], ignore_index=True, sort=False)
    print('\n====================================')
    print(' HEALTH SPATIAL QA')
    print('====================================')
    print(f"Comune: {municipality['name']} ({municipality['istat_code']})")
    print(f'\nFarmacie: {len(pharmacies)}')
    print(f"Farmacie senza coordinate: {int((~pharmacies['source_coordinate_present']).sum())}")
    print(f"Farmacie con coordinate sorgente sospette: {int(pharmacies['source_coordinate_suspicious'].sum())}")
    suspicious_groups = pharmacies.loc[pharmacies['source_coordinate_suspicious'], ['source_latitude', 'source_longitude', 'duplicate_coordinate_group_size', 'duplicate_coordinate_distinct_addresses']].drop_duplicates()
    if not suspicious_groups.empty:
        print('\nGruppi coordinate sospette:')
        print(suspicious_groups.to_string(index=False))
    print(f'\nOspedali/stabilimenti: {len(hospitals)}')
    print(f"Ospedali senza coordinate: {int((~hospitals['source_coordinate_present']).sum())}")
    print('\nEseguo QA Nominatim ' + ('su tutti i record...' if args.validate_all else 'solo sui record mancanti/sospetti...'))
    audit = qa_audit_records(combined, municipality, validate_all=args.validate_all, refresh=args.refresh)
    output_dir = qa_FEATURES_SALUTE_DIR / args.municipality_code
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_label = args.pharmacy_reference_date.strftime('%Y%m%d')
    output_csv = output_dir / f'health_spatial_qa_{reference_label}.csv'
    output_parquet = output_dir / f'health_spatial_qa_{reference_label}.parquet'
    audit.to_csv(output_csv, index=False, encoding='utf-8-sig')
    audit.to_parquet(output_parquet, index=False)
    print('\n=== QA STATUS ===')
    print(audit['qa_status'].value_counts(dropna=False).to_string())
    print('\n=== DETTAGLIO ===')
    columns = ['subcategory', 'source_record_id', 'name', 'address', 'source_coordinate_suspicious', 'duplicate_coordinate_group_size', 'source_latitude', 'source_longitude', 'geocoder_latitude', 'geocoder_longitude', 'source_geocoder_distance_m', 'geocoder_score', 'qa_status', 'qa_reason']
    print(audit[columns].to_string(index=False))
    print('\n=== OUTPUT ===')
    print(f'✓ {output_csv}')
    print(f'✓ {output_parquet}')

import argparse
import json
import math
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
finalize_ROOT = Path(__file__).resolve().parents[2]
finalize_PROCESSED_SALUTE_DIR = finalize_ROOT / 'data' / 'processed' / 'salute'
finalize_FEATURES_SALUTE_DIR = finalize_ROOT / 'data' / 'features' / 'salute'
finalize_AUTO_ADDRESS_SCORE = 90.0
finalize_AUTO_ADDRESS_MARGIN = 12.0
finalize_AUTO_MIN_NAME_SCORE = 40.0
finalize_SOURCE_OSM_CONFIRM_M = 100.0
finalize_SOURCE_OSM_CONFLICT_M = 250.0
finalize_GEOCODER_OSM_CONSENSUS_M = 250.0
finalize_GEOCODER_SITE_MIN_SCORE = 80.0
finalize_GEOCODER_STREET_MIN_SCORE = 80.0

def finalize_parse_args():
    parser = argparse.ArgumentParser(description='Finalizza i siti Health combinando Ministero della Salute, matching OSM e QA/geocoding. Non contiene eccezioni specifiche per singoli comuni: usa regole generiche di evidenza e provenance.')
    parser.add_argument('--municipality-code', required=True, help='Codice ISTAT comunale a 6 cifre.')
    parser.add_argument('--pharmacy-reference-date', default='2025-06-30', help='Snapshot farmacie YYYY-MM-DD. Default: 2025-06-30.')
    parser.add_argument('--hospital-year', default='2023', help='Anno dataset ospedaliero. Default: 2023.')
    args = parser.parse_args()
    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    if not args.municipality_code.isdigit() or len(args.municipality_code) != 6:
        raise ValueError('--municipality-code deve avere esattamente 6 cifre.')
    args.pharmacy_reference_date = pd.Timestamp(args.pharmacy_reference_date).normalize()
    return args

def finalize_haversine_m(lat1, lon1, lat2, lon2):
    values = [lat1, lon1, lat2, lon2]
    if any((pd.isna(value) for value in values)):
        return None
    radius = 6371008.8
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def finalize_read_inputs(args):
    base = finalize_PROCESSED_SALUTE_DIR / args.municipality_code
    feature_base = finalize_FEATURES_SALUTE_DIR / args.municipality_code
    label = args.pharmacy_reference_date.strftime('%Y%m%d')
    pharmacy_path = base / f'pharmacy_sites_{label}.parquet'
    hospital_path = base / f'hospital_sites_{args.hospital_year}.parquet'
    matches_path = base / f'health_osm_matches_{label}.parquet'
    candidates_path = base / 'health_osm_candidates.parquet'
    qa_path = feature_base / f'health_spatial_qa_{label}.parquet'
    for path in [pharmacy_path, hospital_path, matches_path, candidates_path, qa_path]:
        if not path.exists():
            raise FileNotFoundError(f'Input mancante: {path}')
    return {'pharmacies': pd.read_parquet(pharmacy_path), 'hospitals': pd.read_parquet(hospital_path), 'matches': pd.read_parquet(matches_path), 'candidates': gpd.read_parquet(candidates_path), 'qa': pd.read_parquet(qa_path), 'label': label}

def finalize_strengthen_match_status(matches):
    """
    Promuove in automatico alcuni 'review' quando:
    - l'indirizzo è molto forte,
    - il margine sul secondo candidato è ampio,
    - esiste almeno un minimo di coerenza sul nome,
      oppure la coordinata sorgente concorda spazialmente.
    """
    df = matches.copy()
    df['final_match_status'] = df['match_status']
    df['final_match_reason'] = 'original_match_status'
    for index, row in df.iterrows():
        if row['match_status'] != 'review':
            continue
        address_score = row.get('address_score')
        name_score = row.get('name_score')
        margin = row.get('score_margin')
        distance = row.get('source_candidate_distance_m')
        address_strong = pd.notna(address_score) and float(address_score) >= finalize_AUTO_ADDRESS_SCORE
        margin_strong = pd.notna(margin) and float(margin) >= finalize_AUTO_ADDRESS_MARGIN
        name_sufficient = pd.notna(name_score) and float(name_score) >= finalize_AUTO_MIN_NAME_SCORE
        source_agrees = pd.notna(distance) and float(distance) <= finalize_SOURCE_OSM_CONFIRM_M
        if address_strong and margin_strong and (name_sufficient or source_agrees):
            df.loc[index, 'final_match_status'] = 'matched_auto'
            df.loc[index, 'final_match_reason'] = 'promoted_strong_address_margin'
    return df

def finalize_prepare_source_sites(pharmacies, hospitals):
    pharmacy = pharmacies.copy()
    hospital = hospitals.copy()
    pharmacy['source_latitude'] = pd.to_numeric(pharmacy['latitude'], errors='coerce')
    pharmacy['source_longitude'] = pd.to_numeric(pharmacy['longitude'], errors='coerce')
    pharmacy['coordinate_key'] = None
    present = pharmacy['source_latitude'].notna() & pharmacy['source_longitude'].notna()
    pharmacy.loc[present, 'coordinate_key'] = pharmacy.loc[present, 'source_latitude'].round(6).astype(str) + '|' + pharmacy.loc[present, 'source_longitude'].round(6).astype(str)
    pharmacy['normalized_address'] = pharmacy['address'].astype('string').str.upper().str.replace('\\s+', ' ', regex=True).str.strip()
    group_size = pharmacy.loc[present].groupby('coordinate_key').size()
    address_count = pharmacy.loc[present].groupby('coordinate_key')['normalized_address'].nunique()
    pharmacy['source_coordinate_suspicious'] = pharmacy['coordinate_key'].map(group_size).fillna(0).ge(2) & pharmacy['coordinate_key'].map(address_count).fillna(0).ge(2)
    pharmacy['source_coordinate_present'] = present
    hospital['source_latitude'] = pd.to_numeric(hospital['latitude'], errors='coerce')
    hospital['source_longitude'] = pd.to_numeric(hospital['longitude'], errors='coerce')
    hospital['source_coordinate_present'] = hospital['source_latitude'].notna() & hospital['source_longitude'].notna()
    hospital['source_coordinate_suspicious'] = False
    return pd.concat([pharmacy, hospital], ignore_index=True, sort=False)

def finalize_nearest_osm_candidate(candidates, subcategory, latitude, longitude):
    if pd.isna(latitude) or pd.isna(longitude):
        return None
    subset = candidates.loc[candidates['subcategory'] == subcategory]
    best = None
    for _, candidate in subset.iterrows():
        distance = finalize_haversine_m(latitude, longitude, candidate['latitude'], candidate['longitude'])
        if distance is None:
            continue
        item = {'candidate': candidate, 'distance_m': distance}
        if best is None or distance < best['distance_m']:
            best = item
    return best

def finalize_choose_coordinate(source, match, qa, candidates):
    source_present = bool(source.get('source_coordinate_present', False))
    source_suspicious = bool(source.get('source_coordinate_suspicious', False))
    final_match_status = match.get('final_match_status') if match is not None else None
    if final_match_status == 'matched_auto':
        osm_lat = match.get('candidate_latitude')
        osm_lon = match.get('candidate_longitude')
        source_osm_distance = match.get('source_candidate_distance_m')
        if not source_present or source_suspicious:
            return {'latitude': osm_lat, 'longitude': osm_lon, 'coordinate_source': 'osm_matched_site', 'coordinate_resolution': 'site', 'confidence': 'high', 'resolution_status': 'resolved_auto', 'resolution_reason': 'accepted_osm_match_missing_or_suspicious_source'}
        if pd.notna(source_osm_distance) and float(source_osm_distance) <= finalize_SOURCE_OSM_CONFIRM_M:
            return {'latitude': source.get('source_latitude'), 'longitude': source.get('source_longitude'), 'coordinate_source': 'ministero_salute_confirmed_by_osm', 'coordinate_resolution': 'site_or_address', 'confidence': 'high', 'resolution_status': 'resolved_auto', 'resolution_reason': 'source_and_osm_agree'}
        if pd.notna(source_osm_distance) and float(source_osm_distance) > finalize_SOURCE_OSM_CONFLICT_M:
            return {'latitude': osm_lat, 'longitude': osm_lon, 'coordinate_source': 'osm_override_source_conflict', 'coordinate_resolution': 'site', 'confidence': 'high', 'resolution_status': 'resolved_auto', 'resolution_reason': 'accepted_osm_match_source_coordinate_conflict'}
        return {'latitude': source.get('source_latitude'), 'longitude': source.get('source_longitude'), 'coordinate_source': 'ministero_salute_unconfirmed', 'coordinate_resolution': 'source_coordinate', 'confidence': 'medium', 'resolution_status': 'review', 'resolution_reason': 'accepted_osm_match_moderate_coordinate_disagreement'}
    if qa is not None:
        geocoder_lat = qa.get('geocoder_latitude')
        geocoder_lon = qa.get('geocoder_longitude')
        if pd.notna(geocoder_lat) and pd.notna(geocoder_lon):
            nearest = finalize_nearest_osm_candidate(candidates=candidates, subcategory=source.get('subcategory'), latitude=geocoder_lat, longitude=geocoder_lon)
            if nearest is not None and nearest['distance_m'] <= finalize_GEOCODER_OSM_CONSENSUS_M:
                candidate = nearest['candidate']
                return {'latitude': candidate['latitude'], 'longitude': candidate['longitude'], 'coordinate_source': 'geocoder_osm_consensus', 'coordinate_resolution': 'site', 'confidence': 'medium_high', 'resolution_status': 'resolved_auto', 'resolution_reason': 'geocoder_anchor_near_same_category_osm_site'}
            qa_status = qa.get('qa_status')
            qa_score = qa.get('geocoder_score')
            if qa_status == 'site_or_address_candidate' and pd.notna(qa_score) and (float(qa_score) >= finalize_GEOCODER_SITE_MIN_SCORE):
                return {'latitude': geocoder_lat, 'longitude': geocoder_lon, 'coordinate_source': 'nominatim_fallback', 'coordinate_resolution': 'site_or_address', 'confidence': 'medium', 'resolution_status': 'resolved_auto', 'resolution_reason': 'strong_site_or_address_geocoder_candidate'}
            if qa_status == 'street_anchor_candidate' and pd.notna(qa_score) and (float(qa_score) >= finalize_GEOCODER_STREET_MIN_SCORE):
                return {'latitude': geocoder_lat, 'longitude': geocoder_lon, 'coordinate_source': 'nominatim_street_anchor', 'coordinate_resolution': 'street_anchor', 'confidence': 'medium', 'resolution_status': 'resolved_auto', 'resolution_reason': 'street_anchor_fallback'}
    if source_present and (not source_suspicious):
        return {'latitude': source.get('source_latitude'), 'longitude': source.get('source_longitude'), 'coordinate_source': 'ministero_salute_unvalidated', 'coordinate_resolution': 'source_coordinate', 'confidence': 'medium', 'resolution_status': 'review', 'resolution_reason': 'source_coordinate_without_independent_confirmation'}
    return {'latitude': pd.NA, 'longitude': pd.NA, 'coordinate_source': None, 'coordinate_resolution': 'missing', 'confidence': 'unresolved', 'resolution_status': 'unresolved', 'resolution_reason': 'no_reliable_coordinate_evidence'}

def finalize_build_final(sources, matches, qa, candidates):
    matches_by_id = {row['service_site_id']: row for _, row in matches.iterrows()}
    qa_by_id = {row['service_site_id']: row for _, row in qa.iterrows()}
    rows = []
    for _, source in sources.iterrows():
        service_id = source['service_site_id']
        match = matches_by_id.get(service_id)
        qa_row = qa_by_id.get(service_id)
        coordinate = finalize_choose_coordinate(source=source, match=match, qa=qa_row, candidates=candidates)
        base = source.to_dict()
        base['source_latitude'] = source.get('source_latitude')
        base['source_longitude'] = source.get('source_longitude')
        base['source_coordinate_suspicious'] = bool(source.get('source_coordinate_suspicious', False))
        if match is not None:
            base['osm_match_status'] = match.get('final_match_status')
            base['osm_match_reason'] = match.get('final_match_reason')
            base['osm_candidate_id'] = match.get('candidate_id')
            base['osm_candidate_name'] = match.get('candidate_name')
            base['osm_match_score'] = match.get('match_score')
            base['osm_score_margin'] = match.get('score_margin')
            base['osm_address_score'] = match.get('address_score')
            base['osm_name_score'] = match.get('name_score')
        else:
            base['osm_match_status'] = None
            base['osm_match_reason'] = None
            base['osm_candidate_id'] = None
            base['osm_candidate_name'] = None
            base['osm_match_score'] = None
            base['osm_score_margin'] = None
            base['osm_address_score'] = None
            base['osm_name_score'] = None
        if qa_row is not None:
            base['geocoder_qa_status'] = qa_row.get('qa_status')
            base['geocoder_score'] = qa_row.get('geocoder_score')
            base['geocoder_display_name'] = qa_row.get('geocoder_display_name')
        else:
            base['geocoder_qa_status'] = None
            base['geocoder_score'] = None
            base['geocoder_display_name'] = None
        base.update(coordinate)
        base['usable_for_accessibility'] = coordinate['resolution_status'] == 'resolved_auto'
        rows.append(base)
    return pd.DataFrame(rows)

def finalize_make_geodataframe(df):
    geometry = []
    for _, row in df.iterrows():
        if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
            geometry.append(Point(float(row['longitude']), float(row['latitude'])))
        else:
            geometry.append(None)
    return gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

def finalize_main():
    args = finalize_parse_args()
    data = finalize_read_inputs(args)
    sources = finalize_prepare_source_sites(data['pharmacies'], data['hospitals'])
    matches = finalize_strengthen_match_status(data['matches'])
    final = finalize_build_final(sources=sources, matches=matches, qa=data['qa'], candidates=data['candidates'])
    gdf = finalize_make_geodataframe(final)
    output_dir = finalize_PROCESSED_SALUTE_DIR / args.municipality_code
    feature_dir = finalize_FEATURES_SALUTE_DIR / args.municipality_code
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    label = data['label']
    parquet_path = output_dir / f'health_sites_final_{label}.parquet'
    csv_path = feature_dir / f'health_sites_final_{label}.csv'
    manifest_path = feature_dir / f'health_sites_final_{label}_manifest.json'
    gdf.to_parquet(parquet_path, index=False)
    final.to_csv(csv_path, index=False, encoding='utf-8-sig')
    status_counts = final['resolution_status'].value_counts(dropna=False).to_dict()
    resolution_counts = final['coordinate_resolution'].value_counts(dropna=False).to_dict()
    source_counts = final['coordinate_source'].value_counts(dropna=False).to_dict()
    manifest = {'municipality_code': args.municipality_code, 'pharmacy_reference_date': args.pharmacy_reference_date.date().isoformat(), 'hospital_reference_year': int(args.hospital_year), 'total_health_sites': int(len(final)), 'pharmacy_sites': int((final['subcategory'] == 'pharmacy').sum()), 'hospital_sites': int((final['subcategory'] == 'hospital').sum()), 'usable_for_accessibility': int(final['usable_for_accessibility'].sum()), 'resolution_status_counts': {str(key): int(value) for key, value in status_counts.items()}, 'coordinate_resolution_counts': {str(key): int(value) for key, value in resolution_counts.items()}, 'coordinate_source_counts': {str(key): int(value) for key, value in source_counts.items()}, 'notes': ['Il dataset finale non applica patch specifiche per singoli comuni.', 'Coordinate ministeriali duplicate su indirizzi distinti sono considerate sospette e non usate come evidenza positiva.', 'Review OSM possono essere promosse automaticamente solo con indirizzo molto forte, margine sufficiente e ulteriore evidenza sul nome o sulla prossimità.', 'I residui possono essere risolti tramite consenso spaziale tra geocoder e POI OSM della stessa categoria.', 'Gli street anchor restano esplicitamente a risoluzione inferiore e confidence medium.']}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n====================================')
    print(' FINAL HEALTH SITES')
    print('====================================')
    print(f'Totale siti Health: {len(final)}')
    print(f"Farmacie: {int((final['subcategory'] == 'pharmacy').sum())}")
    print(f"Ospedali: {int((final['subcategory'] == 'hospital').sum())}")
    print(f"Usabili per accessibility: {int(final['usable_for_accessibility'].sum())}")
    print('\nResolution status:')
    print(final['resolution_status'].value_counts(dropna=False).to_string())
    print('\nCoordinate source:')
    print(final['coordinate_source'].value_counts(dropna=False).to_string())
    print('\nCoordinate resolution:')
    print(final['coordinate_resolution'].value_counts(dropna=False).to_string())
    print('\n=== DETTAGLIO ===')
    detail_columns = ['subcategory', 'source_record_id', 'name', 'address', 'osm_match_status', 'osm_match_score', 'osm_address_score', 'osm_name_score', 'geocoder_qa_status', 'latitude', 'longitude', 'coordinate_source', 'coordinate_resolution', 'confidence', 'resolution_status', 'resolution_reason']
    print(final[detail_columns].to_string(index=False))
    print('\n=== OUTPUT ===')
    print(f'✓ {parquet_path}')
    print(f'✓ {csv_path}')
    print(f'✓ {manifest_path}')


# -----------------------------------------------------------------------------
# Canonical Health transformation orchestrator
# -----------------------------------------------------------------------------
import sys as _sys
import argparse as _argparse


def _run_legacy_cli(function, argv):
    previous = _sys.argv[:]
    try:
        _sys.argv = [_sys.argv[0], *argv]
        function()
    finally:
        _sys.argv = previous


def canonical_parse_args():
    parser = _argparse.ArgumentParser(
        description=(
            "Canonical Health transformation pipeline: spatial QA and final "
            "Health service-site layer."
        )
    )
    parser.add_argument(
        "--step",
        choices=("prepare", "qa", "finalize"),
        default="prepare",
        help=(
            "prepare esegue QA e finalizzazione; gli altri valori "
            "eseguono un singolo sottostep."
        ),
    )
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--pharmacy-reference-date", default="2025-06-30")
    parser.add_argument("--hospital-year", default="2023")
    parser.add_argument("--validate-all", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    if len(args.municipality_code) != 6 or not args.municipality_code.isdigit():
        raise ValueError("--municipality-code deve avere esattamente 6 cifre.")
    return args


def _common_argv(args):
    return [
        "--municipality-code", args.municipality_code,
        "--pharmacy-reference-date", args.pharmacy_reference_date,
        "--hospital-year", str(args.hospital_year),
    ]


def canonical_main():
    args = canonical_parse_args()
    common = _common_argv(args)

    if args.step in {"prepare", "qa"}:
        qa_argv = list(common)
        if args.validate_all:
            qa_argv.append("--validate-all")
        if args.refresh:
            qa_argv.append("--refresh")
        _run_legacy_cli(qa_main, qa_argv)

    if args.step in {"prepare", "finalize"}:
        _run_legacy_cli(finalize_main, common)


if __name__ == "__main__":
    canonical_main()
