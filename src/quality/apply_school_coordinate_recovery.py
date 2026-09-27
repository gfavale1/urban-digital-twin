import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "mim"
FEATURES = ROOT / "data" / "features" / "mim"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--municipality-code", required=True)
    p.add_argument("--building-year", default="202425")
    args = p.parse_args()
    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    return args

def main():
    args = parse_args()

    curated_path = (
        PROCESSED / args.municipality_code /
        f"physical_school_buildings_{args.building_year}_curated.parquet"
    )
    patch_path = (
        FEATURES / args.municipality_code /
        f"school_coordinate_recovery_patch_{args.building_year}.csv"
    )

    if not curated_path.exists():
        raise FileNotFoundError(curated_path)
    if not patch_path.exists():
        raise FileNotFoundError(patch_path)

    curated = pd.read_parquet(curated_path)
    patch = pd.read_csv(patch_path, dtype={"building_code": str})

    if patch["building_code"].duplicated().any():
        raise RuntimeError("building_code duplicati nel patch.")

    patch = patch.rename(columns={
        "recovery_status": "recovery2_status",
        "recovered_longitude": "recovery2_longitude",
        "recovered_latitude": "recovery2_latitude",
        "recovery_source_type": "recovery2_source_type",
        "recovery_confidence": "recovery2_confidence",
        "geometry_resolution": "recovery2_geometry_resolution",
        "primary_source_url": "recovery2_primary_source_url",
        "secondary_source_url": "recovery2_secondary_source_url",
        "recovery_notes": "recovery2_notes",
    })

    out = curated.merge(
        patch,
        on="building_code",
        how="left",
        validate="one_to_one",
    )

    recovered = out["recovery2_status"].eq("recovered")
    has_coords = (
        out["recovery2_longitude"].notna()
        & out["recovery2_latitude"].notna()
    )
    apply_mask = recovered & has_coords

    out.loc[apply_mask, "curated_location_status"] = "validated_recovery2"
    out.loc[apply_mask, "curated_geometry_source"] = out.loc[
        apply_mask, "recovery2_source_type"
    ]
    out.loc[apply_mask, "curated_longitude"] = out.loc[
        apply_mask, "recovery2_longitude"
    ]
    out.loc[apply_mask, "curated_latitude"] = out.loc[
        apply_mask, "recovery2_latitude"
    ]
    out.loc[apply_mask, "curated_confidence"] = out.loc[
        apply_mask, "recovery2_confidence"
    ]
    out.loc[apply_mask, "curated_geometry_resolution"] = out.loc[
        apply_mask, "recovery2_geometry_resolution"
    ]
    out.loc[apply_mask, "curation_reason"] = (
        "Coordinate recuperate nella seconda fase di validazione "
        "documentata; provenance conservata nelle colonne recovery2_*."
    )

    output_parquet = (
        PROCESSED / args.municipality_code /
        f"physical_school_buildings_{args.building_year}_curated_v2.parquet"
    )
    output_csv = (
        FEATURES / args.municipality_code /
        f"physical_school_buildings_{args.building_year}_curated_v2.csv"
    )
    remaining_csv = (
        FEATURES / args.municipality_code /
        f"school_buildings_remaining_after_recovery2_{args.building_year}.csv"
    )

    out.to_parquet(output_parquet, index=False)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    target = out["curated_location_status"].ne("outside_target_municipality")
    usable = out["curated_longitude"].notna() & out["curated_latitude"].notna()

    remaining = out[target & ~usable].copy()
    remaining.to_csv(remaining_csv, index=False, encoding="utf-8-sig")

    target_n = int(target.sum())
    usable_n = int((target & usable).sum())
    coverage = 100 * usable_n / target_n if target_n else 0

    print("\n====================================")
    print(" SCHOOL COORDINATE RECOVERY 2")
    print("====================================")
    print(f"Patch rows: {len(patch)}")
    print(f"Recovered rows applied: {int(apply_mask.sum())}")
    print(f"Matera target: {target_n}")
    print(f"Coordinate utilizzabili: {usable_n}/{target_n}")
    print(f"Coverage spaziale: {coverage:.2f}%")
    print(f"Ancora senza coordinate: {int((target & ~usable).sum())}")
    print("\n=== OUTPUT ===")
    print(f"✓ {output_parquet}")
    print(f"✓ {output_csv}")
    print(f"✓ {remaining_csv}")

if __name__ == "__main__":
    main()
