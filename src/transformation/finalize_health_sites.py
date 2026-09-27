import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_SALUTE_DIR = ROOT / "data" / "processed" / "salute"
FEATURES_SALUTE_DIR = ROOT / "data" / "features" / "salute"


# ---------------------------------------------------------------------
# Thresholds - generic, not municipality-specific
# ---------------------------------------------------------------------

AUTO_ADDRESS_SCORE = 90.0
AUTO_ADDRESS_MARGIN = 12.0
AUTO_MIN_NAME_SCORE = 40.0

SOURCE_OSM_CONFIRM_M = 100.0
SOURCE_OSM_CONFLICT_M = 250.0

GEOCODER_OSM_CONSENSUS_M = 250.0
GEOCODER_SITE_MIN_SCORE = 80.0
GEOCODER_STREET_MIN_SCORE = 80.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Finalizza i siti Health combinando Ministero della Salute, "
            "matching OSM e QA/geocoding. Non contiene eccezioni specifiche "
            "per Matera: usa regole generiche di evidenza e provenance."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--pharmacy-reference-date",
        default="2025-06-30",
        help="Snapshot farmacie YYYY-MM-DD. Default: 2025-06-30.",
    )

    parser.add_argument(
        "--hospital-year",
        default="2023",
        help="Anno dataset ospedaliero. Default: 2023.",
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "--municipality-code deve avere esattamente 6 cifre."
        )

    args.pharmacy_reference_date = (
        pd.Timestamp(
            args.pharmacy_reference_date
        ).normalize()
    )

    return args


def haversine_m(lat1, lon1, lat2, lon2):
    values = [
        lat1,
        lon1,
        lat2,
        lon2,
    ]

    if any(
        pd.isna(value)
        for value in values
    ):
        return None

    radius = 6_371_008.8

    phi1 = math.radians(
        float(lat1)
    )
    phi2 = math.radians(
        float(lat2)
    )

    dphi = math.radians(
        float(lat2) - float(lat1)
    )

    dlambda = math.radians(
        float(lon2) - float(lon1)
    )

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return (
        2
        * radius
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a),
        )
    )


def read_inputs(args):
    base = (
        PROCESSED_SALUTE_DIR
        / args.municipality_code
    )

    feature_base = (
        FEATURES_SALUTE_DIR
        / args.municipality_code
    )

    label = (
        args.pharmacy_reference_date
        .strftime("%Y%m%d")
    )

    pharmacy_path = (
        base
        / f"pharmacy_sites_{label}.parquet"
    )

    hospital_path = (
        base
        / f"hospital_sites_{args.hospital_year}.parquet"
    )

    matches_path = (
        base
        / f"health_osm_matches_{label}.parquet"
    )

    candidates_path = (
        base
        / "health_osm_candidates.parquet"
    )

    qa_path = (
        feature_base
        / f"health_spatial_qa_{label}.parquet"
    )

    for path in [
        pharmacy_path,
        hospital_path,
        matches_path,
        candidates_path,
        qa_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input mancante: {path}"
            )

    return {
        "pharmacies":
            pd.read_parquet(
                pharmacy_path
            ),

        "hospitals":
            pd.read_parquet(
                hospital_path
            ),

        "matches":
            pd.read_parquet(
                matches_path
            ),

        "candidates":
            gpd.read_parquet(
                candidates_path
            ),

        "qa":
            pd.read_parquet(
                qa_path
            ),

        "label":
            label,
    }


def strengthen_match_status(matches):
    """
    Promuove in automatico alcuni 'review' quando:
    - l'indirizzo è molto forte,
    - il margine sul secondo candidato è ampio,
    - esiste almeno un minimo di coerenza sul nome,
      oppure la coordinata sorgente concorda spazialmente.
    """
    df = matches.copy()

    df["final_match_status"] = (
        df["match_status"]
    )

    df["final_match_reason"] = (
        "original_match_status"
    )

    for index, row in df.iterrows():
        if (
            row["match_status"]
            != "review"
        ):
            continue

        address_score = (
            row.get(
                "address_score"
            )
        )

        name_score = (
            row.get(
                "name_score"
            )
        )

        margin = (
            row.get(
                "score_margin"
            )
        )

        distance = (
            row.get(
                "source_candidate_distance_m"
            )
        )

        address_strong = (
            pd.notna(address_score)
            and float(address_score)
            >= AUTO_ADDRESS_SCORE
        )

        margin_strong = (
            pd.notna(margin)
            and float(margin)
            >= AUTO_ADDRESS_MARGIN
        )

        name_sufficient = (
            pd.notna(name_score)
            and float(name_score)
            >= AUTO_MIN_NAME_SCORE
        )

        source_agrees = (
            pd.notna(distance)
            and float(distance)
            <= SOURCE_OSM_CONFIRM_M
        )

        if (
            address_strong
            and margin_strong
            and (
                name_sufficient
                or source_agrees
            )
        ):
            df.loc[
                index,
                "final_match_status",
            ] = "matched_auto"

            df.loc[
                index,
                "final_match_reason",
            ] = (
                "promoted_strong_address_margin"
            )

    return df


def prepare_source_sites(
    pharmacies,
    hospitals,
):
    pharmacy = pharmacies.copy()
    hospital = hospitals.copy()

    pharmacy["source_latitude"] = (
        pd.to_numeric(
            pharmacy["latitude"],
            errors="coerce",
        )
    )

    pharmacy["source_longitude"] = (
        pd.to_numeric(
            pharmacy["longitude"],
            errors="coerce",
        )
    )

    # Suspicious duplicate-coordinate detection repeated here
    # so finalization does not depend on hidden state from earlier scripts.
    pharmacy["coordinate_key"] = None

    present = (
        pharmacy["source_latitude"].notna()
        & pharmacy["source_longitude"].notna()
    )

    pharmacy.loc[
        present,
        "coordinate_key",
    ] = (
        pharmacy.loc[
            present,
            "source_latitude",
        ]
        .round(6)
        .astype(str)
        + "|"
        + pharmacy.loc[
            present,
            "source_longitude",
        ]
        .round(6)
        .astype(str)
    )

    pharmacy["normalized_address"] = (
        pharmacy["address"]
        .astype("string")
        .str.upper()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    group_size = (
        pharmacy.loc[present]
        .groupby(
            "coordinate_key"
        )
        .size()
    )

    address_count = (
        pharmacy.loc[present]
        .groupby(
            "coordinate_key"
        )["normalized_address"]
        .nunique()
    )

    pharmacy[
        "source_coordinate_suspicious"
    ] = (
        pharmacy["coordinate_key"]
        .map(group_size)
        .fillna(0)
        .ge(2)
        & pharmacy["coordinate_key"]
        .map(address_count)
        .fillna(0)
        .ge(2)
    )

    pharmacy["source_coordinate_present"] = (
        present
    )

    hospital["source_latitude"] = (
        pd.to_numeric(
            hospital["latitude"],
            errors="coerce",
        )
    )

    hospital["source_longitude"] = (
        pd.to_numeric(
            hospital["longitude"],
            errors="coerce",
        )
    )

    hospital["source_coordinate_present"] = (
        hospital["source_latitude"].notna()
        & hospital["source_longitude"].notna()
    )

    hospital[
        "source_coordinate_suspicious"
    ] = False

    return pd.concat(
        [
            pharmacy,
            hospital,
        ],
        ignore_index=True,
        sort=False,
    )


def nearest_osm_candidate(
    candidates,
    subcategory,
    latitude,
    longitude,
):
    if (
        pd.isna(latitude)
        or pd.isna(longitude)
    ):
        return None

    subset = candidates.loc[
        candidates[
            "subcategory"
        ]
        == subcategory
    ]

    best = None

    for _, candidate in subset.iterrows():
        distance = haversine_m(
            latitude,
            longitude,
            candidate["latitude"],
            candidate["longitude"],
        )

        if distance is None:
            continue

        item = {
            "candidate":
                candidate,

            "distance_m":
                distance,
        }

        if (
            best is None
            or distance
            < best["distance_m"]
        ):
            best = item

    return best


def choose_coordinate(
    source,
    match,
    qa,
    candidates,
):
    source_present = bool(
        source.get(
            "source_coordinate_present",
            False,
        )
    )

    source_suspicious = bool(
        source.get(
            "source_coordinate_suspicious",
            False,
        )
    )

    final_match_status = (
        match.get(
            "final_match_status"
        )
        if match is not None
        else None
    )

    # ---------------------------------------------------------
    # 1. Accepted OSM match
    # ---------------------------------------------------------
    if (
        final_match_status
        == "matched_auto"
    ):
        osm_lat = match.get(
            "candidate_latitude"
        )

        osm_lon = match.get(
            "candidate_longitude"
        )

        source_osm_distance = (
            match.get(
                "source_candidate_distance_m"
            )
        )

        # Missing or suspicious source: OSM becomes the automatic resolver.
        if (
            not source_present
            or source_suspicious
        ):
            return {
                "latitude":
                    osm_lat,

                "longitude":
                    osm_lon,

                "coordinate_source":
                    "osm_matched_site",

                "coordinate_resolution":
                    "site",

                "confidence":
                    "high",

                "resolution_status":
                    "resolved_auto",

                "resolution_reason":
                    (
                        "accepted_osm_match_"
                        "missing_or_suspicious_source"
                    ),
            }

        # Source coordinate corroborated by OSM.
        if (
            pd.notna(
                source_osm_distance
            )
            and float(
                source_osm_distance
            )
            <= SOURCE_OSM_CONFIRM_M
        ):
            return {
                "latitude":
                    source.get(
                        "source_latitude"
                    ),

                "longitude":
                    source.get(
                        "source_longitude"
                    ),

                "coordinate_source":
                    "ministero_salute_confirmed_by_osm",

                "coordinate_resolution":
                    "site_or_address",

                "confidence":
                    "high",

                "resolution_status":
                    "resolved_auto",

                "resolution_reason":
                    "source_and_osm_agree",
            }

        # Accepted entity/site match but official coordinate conflicts strongly.
        # Use OSM geometry and preserve the conflict in provenance.
        if (
            pd.notna(
                source_osm_distance
            )
            and float(
                source_osm_distance
            )
            > SOURCE_OSM_CONFLICT_M
        ):
            return {
                "latitude":
                    osm_lat,

                "longitude":
                    osm_lon,

                "coordinate_source":
                    "osm_override_source_conflict",

                "coordinate_resolution":
                    "site",

                "confidence":
                    "high",

                "resolution_status":
                    "resolved_auto",

                "resolution_reason":
                    "accepted_osm_match_source_coordinate_conflict",
            }

        # Moderate disagreement: keep source but lower confidence.
        return {
            "latitude":
                source.get(
                    "source_latitude"
                ),

            "longitude":
                source.get(
                    "source_longitude"
                ),

            "coordinate_source":
                "ministero_salute_unconfirmed",

            "coordinate_resolution":
                "source_coordinate",

            "confidence":
                "medium",

            "resolution_status":
                "review",

            "resolution_reason":
                "accepted_osm_match_moderate_coordinate_disagreement",
        }

    # ---------------------------------------------------------
    # 2. Geocoder + OSM spatial consensus
    # ---------------------------------------------------------
    if qa is not None:
        geocoder_lat = qa.get(
            "geocoder_latitude"
        )

        geocoder_lon = qa.get(
            "geocoder_longitude"
        )

        if (
            pd.notna(
                geocoder_lat
            )
            and pd.notna(
                geocoder_lon
            )
        ):
            nearest = nearest_osm_candidate(
                candidates=candidates,
                subcategory=source.get(
                    "subcategory"
                ),
                latitude=geocoder_lat,
                longitude=geocoder_lon,
            )

            if (
                nearest is not None
                and nearest[
                    "distance_m"
                ]
                <= GEOCODER_OSM_CONSENSUS_M
            ):
                candidate = (
                    nearest[
                        "candidate"
                    ]
                )

                return {
                    "latitude":
                        candidate[
                            "latitude"
                        ],

                    "longitude":
                        candidate[
                            "longitude"
                        ],

                    "coordinate_source":
                        "geocoder_osm_consensus",

                    "coordinate_resolution":
                        "site",

                    "confidence":
                        "medium_high",

                    "resolution_status":
                        "resolved_auto",

                    "resolution_reason":
                        (
                            "geocoder_anchor_near_same_category_osm_site"
                        ),
                }

            qa_status = qa.get(
                "qa_status"
            )

            qa_score = qa.get(
                "geocoder_score"
            )

            # Standalone geocoder exact/site candidate.
            if (
                qa_status
                == "site_or_address_candidate"
                and pd.notna(
                    qa_score
                )
                and float(
                    qa_score
                )
                >= GEOCODER_SITE_MIN_SCORE
            ):
                return {
                    "latitude":
                        geocoder_lat,

                    "longitude":
                        geocoder_lon,

                    "coordinate_source":
                        "nominatim_fallback",

                    "coordinate_resolution":
                        "site_or_address",

                    "confidence":
                        "medium",

                    "resolution_status":
                        "resolved_auto",

                    "resolution_reason":
                        "strong_site_or_address_geocoder_candidate",
                }

            # Street anchor remains usable, but explicitly lower resolution.
            if (
                qa_status
                == "street_anchor_candidate"
                and pd.notna(
                    qa_score
                )
                and float(
                    qa_score
                )
                >= GEOCODER_STREET_MIN_SCORE
            ):
                return {
                    "latitude":
                        geocoder_lat,

                    "longitude":
                        geocoder_lon,

                    "coordinate_source":
                        "nominatim_street_anchor",

                    "coordinate_resolution":
                        "street_anchor",

                    "confidence":
                        "medium",

                    "resolution_status":
                        "resolved_auto",

                    "resolution_reason":
                        "street_anchor_fallback",
                }

    # ---------------------------------------------------------
    # 3. Source coordinate only, if present and not suspicious
    # ---------------------------------------------------------
    if (
        source_present
        and not source_suspicious
    ):
        return {
            "latitude":
                source.get(
                    "source_latitude"
                ),

            "longitude":
                source.get(
                    "source_longitude"
                ),

            "coordinate_source":
                "ministero_salute_unvalidated",

            "coordinate_resolution":
                "source_coordinate",

            "confidence":
                "medium",

            "resolution_status":
                "review",

            "resolution_reason":
                "source_coordinate_without_independent_confirmation",
        }

    # ---------------------------------------------------------
    # 4. unresolved
    # ---------------------------------------------------------
    return {
        "latitude":
            pd.NA,

        "longitude":
            pd.NA,

        "coordinate_source":
            None,

        "coordinate_resolution":
            "missing",

        "confidence":
            "unresolved",

        "resolution_status":
            "unresolved",

        "resolution_reason":
            "no_reliable_coordinate_evidence",
    }


def build_final(
    sources,
    matches,
    qa,
    candidates,
):
    matches_by_id = {
        row[
            "service_site_id"
        ]:
            row
        for _, row
        in matches.iterrows()
    }

    qa_by_id = {
        row[
            "service_site_id"
        ]:
            row
        for _, row
        in qa.iterrows()
    }

    rows = []

    for _, source in sources.iterrows():
        service_id = source[
            "service_site_id"
        ]

        match = (
            matches_by_id.get(
                service_id
            )
        )

        qa_row = (
            qa_by_id.get(
                service_id
            )
        )

        coordinate = choose_coordinate(
            source=source,
            match=match,
            qa=qa_row,
            candidates=candidates,
        )

        base = source.to_dict()

        # Preserve original coordinates separately.
        base[
            "source_latitude"
        ] = source.get(
            "source_latitude"
        )

        base[
            "source_longitude"
        ] = source.get(
            "source_longitude"
        )

        base[
            "source_coordinate_suspicious"
        ] = bool(
            source.get(
                "source_coordinate_suspicious",
                False,
            )
        )

        if match is not None:
            base[
                "osm_match_status"
            ] = match.get(
                "final_match_status"
            )

            base[
                "osm_match_reason"
            ] = match.get(
                "final_match_reason"
            )

            base[
                "osm_candidate_id"
            ] = match.get(
                "candidate_id"
            )

            base[
                "osm_candidate_name"
            ] = match.get(
                "candidate_name"
            )

            base[
                "osm_match_score"
            ] = match.get(
                "match_score"
            )

            base[
                "osm_score_margin"
            ] = match.get(
                "score_margin"
            )

            base[
                "osm_address_score"
            ] = match.get(
                "address_score"
            )

            base[
                "osm_name_score"
            ] = match.get(
                "name_score"
            )

        else:
            base[
                "osm_match_status"
            ] = None

            base[
                "osm_match_reason"
            ] = None

            base[
                "osm_candidate_id"
            ] = None

            base[
                "osm_candidate_name"
            ] = None

            base[
                "osm_match_score"
            ] = None

            base[
                "osm_score_margin"
            ] = None

            base[
                "osm_address_score"
            ] = None

            base[
                "osm_name_score"
            ] = None

        if qa_row is not None:
            base[
                "geocoder_qa_status"
            ] = qa_row.get(
                "qa_status"
            )

            base[
                "geocoder_score"
            ] = qa_row.get(
                "geocoder_score"
            )

            base[
                "geocoder_display_name"
            ] = qa_row.get(
                "geocoder_display_name"
            )

        else:
            base[
                "geocoder_qa_status"
            ] = None

            base[
                "geocoder_score"
            ] = None

            base[
                "geocoder_display_name"
            ] = None

        # Final coordinate replaces working lat/lon while provenance remains.
        base.update(
            coordinate
        )

        base[
            "usable_for_accessibility"
        ] = (
            coordinate[
                "resolution_status"
            ]
            == "resolved_auto"
        )

        rows.append(
            base
        )

    return pd.DataFrame(
        rows
    )


def make_geodataframe(df):
    geometry = []

    for _, row in df.iterrows():
        if (
            pd.notna(
                row.get(
                    "latitude"
                )
            )
            and pd.notna(
                row.get(
                    "longitude"
                )
            )
        ):
            geometry.append(
                Point(
                    float(
                        row[
                            "longitude"
                        ]
                    ),
                    float(
                        row[
                            "latitude"
                        ]
                    ),
                )
            )
        else:
            geometry.append(
                None
            )

    return gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:4326",
    )


def main():
    args = parse_args()

    data = read_inputs(
        args
    )

    sources = prepare_source_sites(
        data["pharmacies"],
        data["hospitals"],
    )

    matches = strengthen_match_status(
        data["matches"]
    )

    final = build_final(
        sources=sources,
        matches=matches,
        qa=data["qa"],
        candidates=data["candidates"],
    )

    gdf = make_geodataframe(
        final
    )

    output_dir = (
        PROCESSED_SALUTE_DIR
        / args.municipality_code
    )

    feature_dir = (
        FEATURES_SALUTE_DIR
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

    label = data[
        "label"
    ]

    parquet_path = (
        output_dir
        / f"health_sites_final_{label}.parquet"
    )

    csv_path = (
        feature_dir
        / f"health_sites_final_{label}.csv"
    )

    manifest_path = (
        feature_dir
        / f"health_sites_final_{label}_manifest.json"
    )

    gdf.to_parquet(
        parquet_path,
        index=False,
    )

    final.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    status_counts = (
        final[
            "resolution_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    resolution_counts = (
        final[
            "coordinate_resolution"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    source_counts = (
        final[
            "coordinate_source"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    manifest = {
        "municipality_code":
            args.municipality_code,

        "pharmacy_reference_date":
            args.pharmacy_reference_date
            .date()
            .isoformat(),

        "hospital_reference_year":
            int(
                args.hospital_year
            ),

        "total_health_sites":
            int(
                len(
                    final
                )
            ),

        "pharmacy_sites":
            int(
                (
                    final[
                        "subcategory"
                    ]
                    == "pharmacy"
                ).sum()
            ),

        "hospital_sites":
            int(
                (
                    final[
                        "subcategory"
                    ]
                    == "hospital"
                ).sum()
            ),

        "usable_for_accessibility":
            int(
                final[
                    "usable_for_accessibility"
                ].sum()
            ),

        "resolution_status_counts":
            {
                str(key):
                    int(value)
                for key, value
                in status_counts.items()
            },

        "coordinate_resolution_counts":
            {
                str(key):
                    int(value)
                for key, value
                in resolution_counts.items()
            },

        "coordinate_source_counts":
            {
                str(key):
                    int(value)
                for key, value
                in source_counts.items()
            },

        "notes": [
            (
                "Il dataset finale non applica patch specifiche per Matera."
            ),
            (
                "Coordinate ministeriali duplicate su indirizzi distinti "
                "sono considerate sospette e non usate come evidenza positiva."
            ),
            (
                "Review OSM possono essere promosse automaticamente solo "
                "con indirizzo molto forte, margine sufficiente e ulteriore "
                "evidenza sul nome o sulla prossimità."
            ),
            (
                "I residui possono essere risolti tramite consenso spaziale "
                "tra geocoder e POI OSM della stessa categoria."
            ),
            (
                "Gli street anchor restano esplicitamente a risoluzione "
                "inferiore e confidence medium."
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
        " FINAL HEALTH SITES"
    )
    print(
        "===================================="
    )

    print(
        f"Totale siti Health: {len(final)}"
    )

    print(
        "Farmacie: "
        f"{int((final['subcategory'] == 'pharmacy').sum())}"
    )

    print(
        "Ospedali: "
        f"{int((final['subcategory'] == 'hospital').sum())}"
    )

    print(
        "Usabili per accessibility: "
        f"{int(final['usable_for_accessibility'].sum())}"
    )

    print(
        "\nResolution status:"
    )

    print(
        final[
            "resolution_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nCoordinate source:"
    )

    print(
        final[
            "coordinate_source"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nCoordinate resolution:"
    )

    print(
        final[
            "coordinate_resolution"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n=== DETTAGLIO ==="
    )

    detail_columns = [
        "subcategory",
        "source_record_id",
        "name",
        "address",
        "osm_match_status",
        "osm_match_score",
        "osm_address_score",
        "osm_name_score",
        "geocoder_qa_status",
        "latitude",
        "longitude",
        "coordinate_source",
        "coordinate_resolution",
        "confidence",
        "resolution_status",
        "resolution_reason",
    ]

    print(
        final[
            detail_columns
        ].to_string(
            index=False
        )
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
