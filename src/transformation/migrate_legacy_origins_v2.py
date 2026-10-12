"""Create an immutable, versioned OriginV2 view from legacy network origins.

The migration preserves row-level geometry and population; it does not
construct a new demand model or alter an existing baseline. Use it only with
legacy files produced by build_population_network_origins.py.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

from core.schema_v2 import ORIGIN_V2
from transformation.build_network_attachments_v2 import sha256_file
from transformation.build_population_network_origins import (
    GEOGRAPHY_VERSION,
    add_origin_v2_fields,
)

ROOT = Path(__file__).resolve().parents[2]
LEGACY_REQUIRED = frozenset({
    "census_section_code", "network_node_id", "assigned_population",
    "reference_date", "assignment_method", "geometry",
})
ALLOWED_METHODS = frozenset({
    "equal_internal_nodes", "representative_point_nearest_node",
})


def migrate_origins(origins: gpd.GeoDataFrame, *, municipality_code: str,
                    census_year: str) -> gpd.GeoDataFrame:
    """Fail-closed schema migration; no geocoding, redistribution or writes."""
    if not re.fullmatch(r"\d{6}", municipality_code):
        raise ValueError("municipality_code must be six ISTAT digits")
    if not re.fullmatch(r"\d{4}", census_year):
        raise ValueError("census_year must be four digits")
    if not isinstance(origins, gpd.GeoDataFrame) or origins.crs is None:
        raise ValueError("Legacy origins must be a GeoDataFrame with an explicit CRS")
    missing = sorted(LEGACY_REQUIRED - set(origins.columns))
    if missing:
        raise ValueError(f"Legacy origins missing columns: {missing}")
    if origins.empty or origins.geometry.isna().any() or (~origins.geom_type.eq("Point")).any():
        raise ValueError("Legacy origins must contain nonempty, non-null Point geometries")
    if any(origins[col].isna().any() for col in (
        "census_section_code", "network_node_id", "reference_date", "assignment_method"
    )):
        raise ValueError("Legacy origins have missing identity/method/reference values")
    if (~origins["assignment_method"].astype(str).isin(ALLOWED_METHODS)).any():
        raise ValueError("Unknown legacy origin assignment method")
    if "municipality_code" in origins.columns and not origins["municipality_code"].astype(str).eq(municipality_code).all():
        raise ValueError("Existing origin municipality_code does not match requested municipality")
    if "census_year" in origins.columns and not origins["census_year"].astype(str).eq(census_year).all():
        raise ValueError("Existing origin census_year does not match requested census year")
    if "origin_id" in origins.columns and origins["origin_id"].notna().any():
        raise ValueError("This migration is only for legacy origins lacking OriginV2 IDs")
    if pd.to_datetime(origins["reference_date"], errors="coerce").isna().any():
        raise ValueError("Legacy origins have invalid population reference dates")
    population = pd.to_numeric(origins["assigned_population"], errors="coerce")
    if population.isna().any() or not population.map(lambda v: math.isfinite(float(v)) and float(v) >= 0).all():
        raise ValueError("Legacy origins have nonfinite or negative population")

    result = add_origin_v2_fields(
        origins, municipality_code=municipality_code, census_year=census_year,
    )
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=origins.crs)
    ORIGIN_V2.validate_columns(result.columns)
    if len(result) != len(origins) or not result.geometry.equals(origins.geometry):
        raise ValueError("Migration changed row count or origin geometry")
    if not result["population_total"].equals(origins["assigned_population"]):
        raise ValueError("Migration changed distributed population")
    if not result["municipality_code"].astype(str).eq(municipality_code).all():
        raise ValueError("Origin municipality identity inconsistent")
    if result["origin_id"].isna().any() or result["origin_id"].duplicated().any():
        raise ValueError("OriginV2 IDs are missing or duplicated")
    return result


def default_paths(root: Path, code: str, year: str) -> tuple[Path, Path]:
    directory = root / "data" / "features" / "accessibility" / code
    source = directory / f"population_network_origins_{year}.parquet"
    migrated = directory / f"population_network_origins_{year}_v2_migrated.parquet"
    return source, migrated


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--write", action="store_true", help="Create a new versioned Parquet and provenance manifest")
    args = parser.parse_args(argv)
    code, year = args.municipality_code.strip(), args.census_year.strip()
    if not re.fullmatch(r"\d{6}", code) or not re.fullmatch(r"\d{4}", year):
        parser.error("Invalid municipality code or census year")
    source, target = default_paths(args.root, code, year)
    manifest = target.with_name(target.stem + "_manifest.json")
    if not source.is_file():
        raise FileNotFoundError(f"Legacy origins missing: {source}")
    if target.exists() or manifest.exists():
        raise FileExistsError(f"Versioned V2 output already exists: {target} or {manifest}")
    source_sha = sha256_file(source)
    legacy = gpd.read_parquet(source)
    if sha256_file(source) != source_sha:
        raise ValueError("Legacy origins changed during migration read")
    migrated = migrate_origins(legacy, municipality_code=code, census_year=year)
    old_population = float(pd.to_numeric(legacy["assigned_population"]).sum())
    new_population = float(pd.to_numeric(migrated["population_total"]).sum())
    if not math.isclose(old_population, new_population, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("Origin migration did not conserve population")
    print(f"=== ORIGIN V2 COMPATIBILITY MIGRATION — {'WRITE' if args.write else 'READ ONLY'} ===")
    print(f"Municipality: {code} | year: {year} | origins: {len(migrated)}")
    print(f"Population: {old_population:.6f} -> {new_population:.6f}")
    print(f"Methods: {migrated['origin_method'].value_counts().to_dict()}")
    print(f"Geography version: {GEOGRAPHY_VERSION}")
    print("Legacy node-distributed geometry is retained; not a new representative-point model.")
    if not args.write:
        print("READ ONLY — use --write to create a new versioned dataset")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or manifest.exists():
        raise FileExistsError("Refusing to overwrite an existing migrated output")
    migrated.to_parquet(target, index=False)
    details = {
        "policy": "origin_v2_legacy_node_distribution_compatibility_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "municipality_code": code, "census_year": year,
        "source_path": str(source.resolve()), "source_sha256": source_sha,
        "target_path": str(target.resolve()), "target_sha256": sha256_file(target),
        "origin_count": len(migrated), "population_total": new_population,
        "geography_version": GEOGRAPHY_VERSION,
        "origin_representation": "legacy_node_distributed_not_representative_point",
        "canonical_origin_replaced": False,
    }
    if sha256_file(source) != source_sha:
        raise ValueError("Source changed while output was being generated")
    manifest.write_text(json.dumps(details, indent=2, sort_keys=True), encoding="utf-8")
    print(f"NEW DERIVED DATASET: {target}")
    print(f"PROVENANCE: {manifest}")
    print("Original origins untouched")


if __name__ == "__main__":
    run()
