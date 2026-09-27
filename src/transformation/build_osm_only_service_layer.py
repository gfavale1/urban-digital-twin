import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_OSM_DIR = ROOT / "data" / "processed" / "osm"
PROCESSED_SALUTE_DIR = ROOT / "data" / "processed" / "salute"
PROCESSED_SERVICES_DIR = ROOT / "data" / "processed" / "services"
FEATURES_SERVICES_DIR = ROOT / "data" / "features" / "services"


CANONICAL_COLUMNS = [
    "service_site_id",
    "domain_service_site_id",
    "category",
    "subcategory",
    "name",
    "municipality_code",
    "municipality_name",
    "address",
    "postal_code",
    "longitude",
    "latitude",
    "source_system",
    "source_dataset",
    "source_record_id",
    "reference_period",
    "coordinate_source",
    "coordinate_resolution",
    "confidence",
    "resolution_status",
    "usable_for_accessibility",
    "capacity_value",
    "capacity_unit",
    "provenance_json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Costruisce il canonical Service layer OSM-only indipendente "
            "dai registri istituzionali. Non usa risultati di matching "
            "MIM/Salute per includere o escludere POI."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--reference-period",
        default="2026-09-27",
        help=(
            "Periodo/data dello snapshot OSM usato come baseline. "
            "Default: 2026-09-27."
        ),
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code).strip().zfill(6)
    )

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "--municipality-code deve avere esattamente 6 cifre."
        )

    return args


def utc_now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip()

    if value in {
        "",
        "-",
        "nan",
        "NaN",
        "None",
        "NULL",
    }:
        return None

    return value


def clean_float(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return None


def compact_json(data):
    clean = {}

    for key, value in data.items():
        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except Exception:
            pass

        if isinstance(
            value,
            (
                pd.Timestamp,
                datetime,
            ),
        ):
            value = value.isoformat()

        elif hasattr(
            value,
            "item",
        ):
            try:
                value = value.item()
            except Exception:
                pass

        clean[str(key)] = value

    return json.dumps(
        clean,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def school_input_path(args):
    return (
        PROCESSED_OSM_DIR
        / args.municipality_code
        / "school_sites.parquet"
    )


def health_input_path(args):
    return (
        PROCESSED_SALUTE_DIR
        / args.municipality_code
        / "health_osm_candidates.parquet"
    )


def load_inputs(args):
    school_path = school_input_path(
        args
    )

    health_path = health_input_path(
        args
    )

    if not school_path.exists():
        raise FileNotFoundError(
            f"OSM schools non trovato: {school_path}"
        )

    if not health_path.exists():
        raise FileNotFoundError(
            f"OSM health non trovato: {health_path}"
        )

    schools = gpd.read_parquet(
        school_path
    )

    health = gpd.read_parquet(
        health_path
    )

    return (
        schools,
        health,
        school_path,
        health_path,
    )


def geometry_or_point(
    row,
):
    geometry = row.get(
        "geometry"
    )

    if geometry is not None:
        try:
            if not geometry.is_empty:
                if geometry.geom_type == "Point":
                    return geometry

                return geometry.representative_point()
        except Exception:
            pass

    longitude = clean_float(
        row.get(
            "longitude"
        )
    )

    latitude = clean_float(
        row.get(
            "latitude"
        )
    )

    if (
        longitude is not None
        and latitude is not None
    ):
        return Point(
            longitude,
            latitude,
        )

    return None


def map_osm_schools(
    schools,
    municipality_code,
    reference_period,
):
    rows = []

    for _, row in schools.iterrows():
        raw_id = clean_text(
            row.get(
                "site_id"
            )
        )

        if raw_id is None:
            raise RuntimeError(
                "OSM school site senza site_id."
            )

        point = geometry_or_point(
            row
        )

        longitude = (
            float(point.x)
            if point is not None
            else clean_float(
                row.get(
                    "longitude"
                )
            )
        )

        latitude = (
            float(point.y)
            if point is not None
            else clean_float(
                row.get(
                    "latitude"
                )
            )
        )

        usable = (
            longitude is not None
            and latitude is not None
        )

        provenance = {
            "representative_osm_key":
                row.get(
                    "representative_osm_key"
                ),

            "primary_names":
                row.get(
                    "primary_names"
                ),

            "core_primary_names":
                row.get(
                    "core_primary_names"
                ),

            "operator_names":
                row.get(
                    "operator_names"
                ),

            "addresses":
                row.get(
                    "addresses"
                ),

            "postcodes":
                row.get(
                    "postcodes"
                ),

            "amenities":
                row.get(
                    "amenities"
                ),

            "buildings":
                row.get(
                    "buildings"
                ),

            "member_osm_keys":
                row.get(
                    "member_osm_keys"
                ),

            "member_count":
                row.get(
                    "member_count"
                ),
        }

        rows.append(
            {
                "service_site_id":
                    f"OSM::EDUCATION::{raw_id}",

                "domain_service_site_id":
                    raw_id,

                "category":
                    "education",

                "subcategory":
                    "osm_school",

                "name":
                    clean_text(
                        row.get(
                            "site_name"
                        )
                    ),

                "municipality_code":
                    municipality_code,

                "municipality_name":
                    None,

                "address":
                    clean_text(
                        row.get(
                            "site_address"
                        )
                    ),

                "postal_code":
                    None,

                "longitude":
                    longitude,

                "latitude":
                    latitude,

                "source_system":
                    "OpenStreetMap",

                "source_dataset":
                    "OSM school sites",

                "source_record_id":
                    clean_text(
                        row.get(
                            "representative_osm_key"
                        )
                    )
                    or raw_id,

                "reference_period":
                    reference_period,

                "coordinate_source":
                    "osm",

                "coordinate_resolution":
                    "site",

                "confidence":
                    "source_provided",

                "resolution_status":
                    (
                        "resolved"
                        if usable
                        else "unresolved"
                    ),

                "usable_for_accessibility":
                    usable,

                "capacity_value":
                    None,

                "capacity_unit":
                    None,

                "provenance_json":
                    compact_json(
                        provenance
                    ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=CANONICAL_COLUMNS,
    )


def map_osm_health(
    health,
    municipality_code,
    reference_period,
):
    rows = []

    allowed = {
        "pharmacy",
        "hospital",
    }

    for _, row in health.iterrows():
        subcategory = clean_text(
            row.get(
                "subcategory"
            )
        )

        if subcategory not in allowed:
            continue

        raw_id = clean_text(
            row.get(
                "candidate_id"
            )
        )

        if raw_id is None:
            raise RuntimeError(
                "OSM health candidate senza candidate_id."
            )

        point = geometry_or_point(
            row
        )

        longitude = (
            float(point.x)
            if point is not None
            else clean_float(
                row.get(
                    "longitude"
                )
            )
        )

        latitude = (
            float(point.y)
            if point is not None
            else clean_float(
                row.get(
                    "latitude"
                )
            )
        )

        usable = (
            longitude is not None
            and latitude is not None
        )

        provenance = {
            "amenity":
                row.get(
                    "amenity"
                ),

            "healthcare":
                row.get(
                    "healthcare"
                ),
        }

        rows.append(
            {
                "service_site_id":
                    (
                        "OSM::HEALTH::"
                        f"{subcategory.upper()}::{raw_id}"
                    ),

                "domain_service_site_id":
                    raw_id,

                "category":
                    "health",

                "subcategory":
                    subcategory,

                "name":
                    clean_text(
                        row.get(
                            "name"
                        )
                    ),

                "municipality_code":
                    municipality_code,

                "municipality_name":
                    None,

                "address":
                    clean_text(
                        row.get(
                            "address"
                        )
                    ),

                "postal_code":
                    None,

                "longitude":
                    longitude,

                "latitude":
                    latitude,

                "source_system":
                    "OpenStreetMap",

                "source_dataset":
                    "OSM health POIs",

                "source_record_id":
                    raw_id,

                "reference_period":
                    reference_period,

                "coordinate_source":
                    "osm",

                "coordinate_resolution":
                    "site",

                "confidence":
                    "source_provided",

                "resolution_status":
                    (
                        "resolved"
                        if usable
                        else "unresolved"
                    ),

                "usable_for_accessibility":
                    usable,

                "capacity_value":
                    None,

                "capacity_unit":
                    None,

                "provenance_json":
                    compact_json(
                        provenance
                    ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=CANONICAL_COLUMNS,
    )


def validate_layer(
    services,
    municipality_code,
):
    if services.empty:
        raise RuntimeError(
            "OSM-only Service layer vuoto."
        )

    if (
        services[
            "service_site_id"
        ]
        .duplicated()
        .any()
    ):
        duplicates = (
            services.loc[
                services[
                    "service_site_id"
                ].duplicated(
                    keep=False
                ),
                "service_site_id",
            ]
            .tolist()
        )

        raise RuntimeError(
            "service_site_id duplicati: "
            f"{duplicates[:20]}"
        )

    wrong_code = (
        services[
            "municipality_code"
        ]
        != municipality_code
    )

    if wrong_code.any():
        raise RuntimeError(
            "municipality_code incoerente."
        )

    bad_usable = (
        services[
            "usable_for_accessibility"
        ]
        & (
            services[
                "longitude"
            ].isna()
            | services[
                "latitude"
            ].isna()
        )
    )

    if bad_usable.any():
        raise RuntimeError(
            "Servizi utilizzabili senza coordinate."
        )


def make_geodataframe(
    services,
):
    geometry = []

    for _, row in services.iterrows():
        longitude = row[
            "longitude"
        ]
        latitude = row[
            "latitude"
        ]

        if (
            pd.notna(
                longitude
            )
            and pd.notna(
                latitude
            )
        ):
            geometry.append(
                Point(
                    float(
                        longitude
                    ),
                    float(
                        latitude
                    ),
                )
            )
        else:
            geometry.append(
                None
            )

    return gpd.GeoDataFrame(
        services,
        geometry=geometry,
        crs="EPSG:4326",
    )


def main():
    args = parse_args()

    (
        schools,
        health,
        school_path,
        health_path,
    ) = load_inputs(
        args
    )

    school_services = (
        map_osm_schools(
            schools=schools,
            municipality_code=args.municipality_code,
            reference_period=args.reference_period,
        )
    )

    health_services = (
        map_osm_health(
            health=health,
            municipality_code=args.municipality_code,
            reference_period=args.reference_period,
        )
    )

    services = pd.concat(
        [
            school_services,
            health_services,
        ],
        ignore_index=True,
    )

    services = (
        services.sort_values(
            [
                "category",
                "subcategory",
                "service_site_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    validate_layer(
        services,
        args.municipality_code,
    )

    gdf = make_geodataframe(
        services
    )

    output_dir = (
        PROCESSED_SERVICES_DIR
        / args.municipality_code
    )

    feature_dir = (
        FEATURES_SERVICES_DIR
        / args.municipality_code
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        output_dir
        / "service_sites_osm_only.parquet"
    )

    csv_path = (
        feature_dir
        / "service_sites_osm_only.csv"
    )

    manifest_path = (
        feature_dir
        / "service_sites_osm_only_manifest.json"
    )

    gdf.to_parquet(
        parquet_path,
        index=False,
    )

    services.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    category_counts = (
        services[
            "category"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    subcategory_counts = (
        services[
            "subcategory"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    manifest = {
        "generated_at_utc":
            utc_now_iso(),

        "municipality_code":
            args.municipality_code,

        "reference_period":
            args.reference_period,

        "inputs": {
            "osm_schools":
                str(
                    school_path
                ),

            "osm_health":
                str(
                    health_path
                ),
        },

        "total_service_sites":
            int(
                len(
                    services
                )
            ),

        "usable_for_accessibility":
            int(
                services[
                    "usable_for_accessibility"
                ].sum()
            ),

        "category_counts": {
            str(key):
                int(value)
            for key, value
            in category_counts.items()
        },

        "subcategory_counts": {
            str(key):
                int(value)
            for key, value
            in subcategory_counts.items()
        },

        "methodology": {
            "independence":
                (
                    "The OSM-only baseline is built exclusively from "
                    "OSM-derived school and health POIs. Institutional "
                    "matching results are not used to select, remove, "
                    "or relabel OSM records."
                ),

            "schools":
                (
                    "All consolidated OSM school sites are retained, "
                    "including OSM kindergarten/school semantics already "
                    "present in the source snapshot."
                ),

            "health":
                (
                    "All OSM candidates classified as pharmacy or hospital "
                    "by OSM tags are retained. This intentionally preserves "
                    "OSM-only false positives/semantic differences for the "
                    "subsequent source-comparison experiment."
                ),

            "coordinates":
                (
                    "OSM representative site coordinates are used directly; "
                    "no institutional coordinate correction is applied."
                ),
        },

        "notes": [
            (
                "The OSM-only health layer may include records whose OSM "
                "semantic classification differs from the institutional "
                "registry (e.g. parafarmacia tagged as pharmacy). These are "
                "not filtered here because doing so with institutional "
                "knowledge would contaminate the baseline."
            ),
        ],
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n===================================="
    )
    print(
        " OSM-ONLY CANONICAL SERVICE LAYER"
    )
    print(
        "===================================="
    )

    print(
        f"Comune: {args.municipality_code}"
    )

    print(
        "OSM school sites: "
        f"{len(school_services)}"
    )

    print(
        "OSM health POIs: "
        f"{len(health_services)}"
    )

    print(
        "Totale service sites: "
        f"{len(services)}"
    )

    print(
        "Usabili per accessibility: "
        f"{int(services['usable_for_accessibility'].sum())}"
    )

    print(
        "\nCategory:"
    )
    print(
        services[
            "category"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSubcategory:"
    )
    print(
        services[
            "subcategory"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== OUTPUT ==="
    )
    print(
        f"✓ {parquet_path}"
    )
    print(
        f"✓ {csv_path}"
    )
    print(
        f"✓ {manifest_path}"
    )


if __name__ == "__main__":
    main()
