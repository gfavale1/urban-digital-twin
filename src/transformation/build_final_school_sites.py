#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "mim"
FEATURES = ROOT / "data" / "features" / "mim"


def clean(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return None if not s or s.lower() in {"nan", "none", "<na>"} else s


def fnum(v):
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def truth(v):
    if isinstance(v, bool):
        return v
    s = clean(v)
    return bool(s and s.lower() in {"1", "true", "yes", "y", "si", "sì"})


def parse_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [x for x in (clean(i) for i in v) if x]
    try:
        if pd.isna(v):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        try:
            x = json.loads(v)
            if isinstance(x, list):
                return [y for y in (clean(i) for i in x) if y]
        except json.JSONDecodeError:
            pass
    x = clean(v)
    return [x] if x else []


def jlist(values):
    out, seen = [], set()
    for v in values:
        v = clean(v)
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return json.dumps(out, ensure_ascii=False)


def pick_list(row, base):
    for col in (base, f"{base}_x", f"{base}_y"):
        if col in row.index:
            vals = parse_list(row.get(col))
            if vals:
                return vals
    return []


def require(df, cols, label):
    missing = set(cols) - set(df.columns)
    if missing:
        raise RuntimeError(f"{label}: colonne mancanti: {', '.join(sorted(missing))}")


def deterministic_id(prefix, parts):
    raw = "|".join("" if p is None else str(p) for p in parts)
    return f"{prefix}:{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def parse_args():
    p = argparse.ArgumentParser(description="Builder Education generalizzato e zero-touch.")
    p.add_argument("--municipality-code", required=True)
    p.add_argument("--school-year", default="202425")
    p.add_argument("--building-year", default="202425")
    a = p.parse_args()
    a.municipality_code = str(a.municipality_code).strip().zfill(6)
    if len(a.municipality_code) != 6 or not a.municipality_code.isdigit():
        raise ValueError("municipality-code deve avere 6 cifre")
    return a


def load_inputs(a):
    pdir = PROCESSED / a.municipality_code
    fdir = FEATURES / a.municipality_code
    fdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "buildings": pdir / f"physical_school_buildings_{a.building_year}_final_v2.parquet",
        "fallback": fdir / f"school_osm_street_fallback_{a.building_year}.csv",
        "remaining": pdir / f"remaining_school_services_{a.school_year}.parquet",
        "unmatched": pdir / f"school_unmatched_classification_{a.school_year}.parquet",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    buildings = pd.read_parquet(paths["buildings"])
    fallback = pd.read_csv(paths["fallback"], dtype={"building_code": "string"})
    remaining = pd.read_parquet(paths["remaining"])
    unmatched = pd.read_parquet(paths["unmatched"])

    require(buildings, [
        "building_code", "official_building_address", "building_municipality_name",
        "building_postal_code", "final_location_status", "final_geometry_source",
        "final_longitude", "final_latitude", "location_confidence"
    ], "buildings")
    require(fallback, [
        "building_code", "fallback_status", "fallback_confidence",
        "candidate_longitude", "candidate_latitude", "candidate_resolution"
    ], "fallback")
    require(remaining, [
        "school_code", "school_name", "school_address", "registry_type",
        "resolution_status", "resolution_method", "confidence",
        "coordinate_resolution", "longitude", "latitude"
    ], "remaining")
    require(unmatched, [
        "school_code", "unmatched_class", "creates_public_school_site",
        "needs_geolocation", "site_handling"
    ], "unmatched")

    for df, col in [(buildings, "building_code"), (fallback, "building_code"),
                    (remaining, "school_code"), (unmatched, "school_code")]:
        df[col] = df[col].astype("string").str.strip()

    if buildings["building_code"].duplicated().any():
        raise RuntimeError("building_code duplicati in buildings")
    if fallback["building_code"].duplicated().any():
        raise RuntimeError("building_code duplicati in fallback")

    return buildings, fallback, remaining, unmatched, pdir, fdir, paths


def determine_municipality_name(buildings, unmatched):
    vals = buildings["building_municipality_name"].dropna().astype(str).str.strip().unique().tolist()
    vals = [v for v in vals if v]
    if len(vals) == 1:
        return vals[0]
    if "municipality_name" in unmatched.columns:
        vals = unmatched["municipality_name"].dropna().astype(str).str.strip().unique().tolist()
        vals = [v for v in vals if v]
        if len(vals) == 1:
            return vals[0]
    raise RuntimeError("Comune target non determinabile univocamente")


def resolve_building(row):
    status = clean(row.get("final_location_status"))
    lon, lat = fnum(row.get("final_longitude")), fnum(row.get("final_latitude"))
    source = (clean(row.get("final_geometry_source")) or "").lower()
    confidence = clean(row.get("location_confidence"))

    if status in {"validated", "accepted_address"} and lon is not None and lat is not None:
        if "osm" in source:
            resolution, method = "site", "osm_building_match"
        elif status == "accepted_address":
            resolution, method = "address", "nominatim_address"
        else:
            resolution, method = "address", "nominatim_validated_by_osm"
        return True, lon, lat, "resolved_auto", method, resolution, confidence or "high", "fusion"

    fb_status = clean(row.get("fallback_status"))
    lon, lat = fnum(row.get("candidate_longitude")), fnum(row.get("candidate_latitude"))
    fb_conf = clean(row.get("fallback_confidence"))
    if fb_status in {"address_candidate", "street_anchor_candidate"} and lon is not None and lat is not None:
        resolution = "address" if fb_status == "address_candidate" else "street_anchor"
        return True, lon, lat, "resolved_auto", "osm_street_fallback", resolution, fb_conf or ("high" if resolution == "address" else "medium"), "street_fallback"

    return False, None, None, ("review" if status == "review" else "unresolved"), None, None, confidence or "unresolved", "unresolved"


def build_building_rows(buildings, fallback, a, muni):
    use_cols = [c for c in [
        "building_code", "fallback_status", "fallback_confidence",
        "candidate_longitude", "candidate_latitude", "candidate_resolution",
        "candidate_display_name", "matched_osm_street",
        "street_match_score", "street_match_margin"
    ] if c in fallback.columns]
    merged = buildings.merge(fallback[use_cols], on="building_code", how="left", validate="one_to_one")

    rows = []
    for _, row in merged.sort_values("building_code").iterrows():
        code = str(row["building_code"])
        codes = pick_list(row, "linked_school_codes")
        names = pick_list(row, "linked_school_names")
        usable, lon, lat, rstatus, method, cres, conf, stage = resolve_building(row)
        provenance = {
            "source_stage": stage,
            "fusion_status": clean(row.get("final_location_status")),
            "fusion_geometry_source": clean(row.get("final_geometry_source")),
            "osm_match_status": clean(row.get("osm_match_status")),
            "geocoder_status": clean(row.get("geocoder_status")),
            "fallback_status": clean(row.get("fallback_status")),
            "fallback_matched_osm_street": clean(row.get("matched_osm_street")),
        }
        rows.append({
            "school_site_id": f"MIMB:{code}",
            "category": "education",
            "subcategory": "state_school_building",
            "site_record_type": "physical_building",
            "name": names[0] if names else f"School building {code}",
            "municipality_code": a.municipality_code,
            "municipality_name": muni,
            "address": clean(row.get("official_building_address")),
            "postal_code": clean(row.get("building_postal_code")),
            "longitude": lon,
            "latitude": lat,
            "coordinate_origin": "automatic" if usable else None,
            "location_method": method,
            "coordinate_resolution": cres,
            "confidence": conf,
            "resolution_status": rstatus,
            "usable_for_accessibility": usable,
            "source_system": "MIM+OSM/Nominatim",
            "source_dataset": f"MIM buildings {a.building_year} + automatic spatial resolution",
            "source_record_id": code,
            "reference_period": a.school_year,
            "building_code": code,
            "linked_school_codes": jlist(codes),
            "linked_school_names": jlist(names),
            "provenance_json": json.dumps(provenance, ensure_ascii=False, sort_keys=True),
        })
    return rows


def normalize_remaining(row):
    status = clean(row.get("resolution_status"))
    lon, lat = fnum(row.get("longitude")), fnum(row.get("latitude"))
    accepted = {"accepted_osm", "accepted_peer_site", "accepted_address", "accepted_site", "street_anchor_candidate"}
    usable = status in accepted and lon is not None and lat is not None
    if not usable:
        return False, None, None, ("review" if status == "review" else "unresolved"), clean(row.get("resolution_method")), None, clean(row.get("confidence")) or "low"

    if status in {"accepted_osm", "accepted_peer_site", "accepted_site"}:
        resolution = clean(row.get("coordinate_resolution")) or "site"
    elif status == "accepted_address":
        resolution = clean(row.get("coordinate_resolution")) or "address"
    else:
        resolution = "street_anchor"
    return True, lon, lat, "resolved_auto", clean(row.get("resolution_method")), resolution, clean(row.get("confidence")) or ("medium" if resolution == "street_anchor" else "high")


def build_remaining_rows(remaining, a, muni):
    work = remaining.copy()
    norm = [normalize_remaining(row) for _, row in work.iterrows()]
    for i, col in enumerate(["_usable", "_lon", "_lat", "_status", "_method", "_resolution", "_confidence"]):
        work[col] = [x[i] for x in norm]

    keys = []
    for _, row in work.iterrows():
        if truth(row["_usable"]) and row["_resolution"] in {"site", "address"} and str(row["_confidence"]).lower() == "high":
            keys.append(("exact_site", round(float(row["_lon"]), 6), round(float(row["_lat"]), 6)))
        else:
            keys.append(("record", str(row["school_code"])))
    work["_site_group_key"] = keys

    rows = []
    for key, group in work.groupby("_site_group_key", sort=False):
        first = group.iloc[0]
        codes = group["school_code"].dropna().astype(str).str.strip().tolist()
        names = group["school_name"].dropna().astype(str).str.strip().tolist()
        addresses = group["school_address"].dropna().astype(str).str.strip().tolist()
        registry_types = group["registry_type"].dropna().astype(str).str.lower().str.strip().unique().tolist()
        subcategory = "paritary_school" if registry_types and all(x == "paritary" for x in registry_types) else "state_school_without_building_registry"
        school_site_id = deterministic_id("MIMS", [a.municipality_code, key[1], key[2]]) if key[0] == "exact_site" else f"MIMS:{codes[0]}"
        usable = truth(first["_usable"])
        provenance = {
            "original_resolution_statuses": group["resolution_status"].fillna("").astype(str).tolist(),
            "matched_names": group["matched_name"].fillna("").astype(str).tolist() if "matched_name" in group.columns else [],
        }
        rows.append({
            "school_site_id": school_site_id,
            "category": "education",
            "subcategory": subcategory,
            "site_record_type": "school_service_without_building_relation",
            "name": names[0] if len(names) == 1 else " / ".join(names),
            "municipality_code": a.municipality_code,
            "municipality_name": muni,
            "address": addresses[0] if addresses else None,
            "postal_code": None,
            "longitude": fnum(first["_lon"]) if usable else None,
            "latitude": fnum(first["_lat"]) if usable else None,
            "coordinate_origin": "automatic" if usable else None,
            "location_method": clean(first["_method"]),
            "coordinate_resolution": clean(first["_resolution"]),
            "confidence": clean(first["_confidence"]),
            "resolution_status": clean(first["_status"]),
            "usable_for_accessibility": usable,
            "source_system": "MIM+OSM/Nominatim",
            "source_dataset": f"MIM schools {a.school_year} + residual-service resolution",
            "source_record_id": jlist(codes),
            "reference_period": a.school_year,
            "building_code": None,
            "linked_school_codes": jlist(codes),
            "linked_school_names": jlist(names),
            "provenance_json": json.dumps(provenance, ensure_ascii=False, sort_keys=True),
        })
    return rows


def to_geodataframe(df):
    geometry = []
    for lon, lat in zip(df["longitude"], df["latitude"]):
        lon, lat = fnum(lon), fnum(lat)
        geometry.append(Point(lon, lat) if lon is not None and lat is not None else None)
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")


def qa(final, buildings, unmatched):
    if final["school_site_id"].duplicated().any():
        raise RuntimeError("school_site_id duplicati")
    usable = final["usable_for_accessibility"].fillna(False).astype(bool)
    bad = usable & (final["longitude"].isna() | final["latitude"].isna() | final.geometry.isna())
    if bad.any():
        raise RuntimeError("Siti usable_for_accessibility senza coordinate")
    building_rows = final[final["site_record_type"] == "physical_building"]
    if len(building_rows) != len(buildings):
        raise RuntimeError(f"Edifici persi: {len(building_rows)} != {len(buildings)}")

    excluded = set(
        unmatched.loc[~unmatched["creates_public_school_site"].map(truth), "school_code"]
        .dropna().astype(str)
    )
    represented = set()
    for value in final["linked_school_codes"]:
        represented.update(parse_list(value))
    leaked = excluded & represented
    if leaked:
        raise RuntimeError("Servizi speciali esclusi presenti nel layer finale: " + ", ".join(sorted(leaked)))


def main():
    a = parse_args()
    buildings, fallback, remaining, unmatched, pdir, fdir, paths = load_inputs(a)
    muni = determine_municipality_name(buildings, unmatched)

    final = pd.DataFrame(
        build_building_rows(buildings, fallback, a, muni)
        + build_remaining_rows(remaining, a, muni)
    )
    final = to_geodataframe(final)
    qa(final, buildings, unmatched)

    parquet_path = pdir / f"school_sites_{a.school_year}.parquet"
    csv_path = fdir / f"school_sites_{a.school_year}.csv"
    exclusions_path = fdir / f"school_sites_excluded_{a.school_year}.csv"
    manifest_path = fdir / f"school_sites_{a.school_year}_manifest.json"

    final.to_parquet(parquet_path, index=False)
    csv_df = pd.DataFrame(final.drop(columns="geometry"))
    csv_df["geometry_wkt"] = final.geometry.apply(lambda g: g.wkt if g is not None else None)
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    excluded = unmatched.loc[~unmatched["creates_public_school_site"].map(truth)].copy()
    excluded.to_csv(exclusions_path, index=False, encoding="utf-8-sig")

    usable = final["usable_for_accessibility"].fillna(False).astype(bool)
    stats = {
        "municipality_code": a.municipality_code,
        "municipality_name": muni,
        "school_year": a.school_year,
        "building_year": a.building_year,
        "inputs": {k: str(v) for k, v in paths.items()},
        "final_rows": int(len(final)),
        "usable_rows": int(usable.sum()),
        "unusable_rows": int((~usable).sum()),
        "physical_building_rows": int((final["site_record_type"] == "physical_building").sum()),
        "physical_building_usable": int(((final["site_record_type"] == "physical_building") & usable).sum()),
        "remaining_service_rows": int((final["site_record_type"] == "school_service_without_building_relation").sum()),
        "remaining_service_usable": int(((final["site_record_type"] == "school_service_without_building_relation") & usable).sum()),
        "excluded_special_unmatched_records": int(len(excluded)),
        "by_subcategory": {str(k): int(v) for k, v in final["subcategory"].value_counts(dropna=False).items()},
        "by_resolution_status": {str(k): int(v) for k, v in final["resolution_status"].value_counts(dropna=False).items()},
        "by_coordinate_resolution": {str(k): int(v) for k, v in final["coordinate_resolution"].value_counts(dropna=False).items()},
        "by_confidence": {str(k): int(v) for k, v in final["confidence"].value_counts(dropna=False).items()},
    }
    manifest_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n====================================")
    print(" GENERALIZED FINAL EDUCATION BUILDER")
    print("====================================")
    print(f"Comune: {muni} ({a.municipality_code})")
    print(f"Final rows: {len(final)}")
    print(f"Usable: {int(usable.sum())}/{len(final)} ({100 * usable.mean():.2f}%)")
    print(f"Physical buildings: {stats['physical_building_rows']}")
    print(f"Physical buildings usable: {stats['physical_building_usable']}")
    print(f"Remaining-service sites: {stats['remaining_service_rows']}")
    print(f"Remaining-service sites usable: {stats['remaining_service_usable']}")
    print(f"Excluded special unmatched: {stats['excluded_special_unmatched_records']}")

    print("\nResolution status:")
    print(final["resolution_status"].value_counts(dropna=False).to_string())
    print("\nCoordinate resolution:")
    print(final["coordinate_resolution"].value_counts(dropna=False).to_string())
    print("\nSubcategory:")
    print(final["subcategory"].value_counts(dropna=False).to_string())

    print("\n=== OUTPUT ===")
    for path in (parquet_path, csv_path, exclusions_path, manifest_path):
        print(f"✓ {path}")


if __name__ == "__main__":
    main()
