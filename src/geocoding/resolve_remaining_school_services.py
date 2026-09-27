import argparse
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from rapidfuzz import fuzz
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
PROCESSED_ISTAT_DIR = ROOT / "data" / "processed" / "istat"
FEATURES_MIM_DIR = ROOT / "data" / "features" / "mim"
RAW_MIM_DIR = ROOT / "data" / "raw" / "mim"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Risoluzione automatica dei record scolastici residui che non sono "
            "coperti dall'Anagrafe edilizia: scuole paritarie + unmatched fisici."
        )
    )

    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--pause-seconds", type=float, default=1.1)
    parser.add_argument("--refresh", action="store_true")

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    return args


def clean_text(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip()
    return value or None


def strip_accents(value):
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


def normalize_text(value):
    value = clean_text(value)

    if value is None:
        return ""

    value = strip_accents(value).upper()
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"[^A-Z0-9']+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_address(value):
    value = normalize_text(value)

    if not value:
        return ""

    replacements = {
        "S N C": "",
        "SNC": "",
        "S N": "",
        "V LE": "VIALE",
        "VLE": "VIALE",
        "P ZA": "PIAZZA",
        "PZZA": "PIAZZA",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"(?<=[A-Z])(?=\d)", " ", value)
    value = re.sub(r"(?<=\d)(?=[A-Z])", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def extract_civic(value):
    value = normalize_address(value)

    if not value:
        return None

    match = re.search(
        r"(?:^|\s)(\d+(?:[A-Z])?(?:[/\-]\d+(?:[A-Z])?)*)\s*$",
        value,
    )

    return match.group(1) if match else None


def normalize_civic(value):
    return normalize_text(value).replace(" ", "")


def school_name_core(value):
    value = normalize_text(value)

    if not value:
        return ""

    generic = {
        "SCUOLA",
        "ISTITUTO",
        "IST",
        "I",
        "C",
        "IC",
        "PLESSO",
        "SEDE",
        "MT",
        "MATERNA",
        "INFANZIA",
        "PRIMARIA",
        "SECONDARIA",
        "PRIMO",
        "SECONDO",
        "GRADO",
        "STATALE",
        "PARITARIA",
        "L",
        "CLAS",
    }

    tokens = [
        token
        for token in value.split()
        if token not in generic
    ]

    return " ".join(tokens) or value


def load_cache(path):
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(path, cache):
    path.write_text(
        json.dumps(
            cache,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_boundary(municipality_code):
    path = (
        PROCESSED_ISTAT_DIR
        / f"{municipality_code}_census_areas_2021.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Boundary ISTAT non trovato: {path}"
        )

    areas = gpd.read_parquet(path).to_crs(4326)

    geometry = areas.geometry.union_all()

    minx, miny, maxx, maxy = geometry.bounds

    return {
        "geometry": geometry,
        "viewbox": f"{minx},{maxy},{maxx},{miny}",
    }


def inside_boundary(boundary_geometry, lon, lat):
    return bool(
        boundary_geometry.covers(
            Point(float(lon), float(lat))
        )
    )


def get_result_road(address):
    if not isinstance(address, dict):
        return None

    return (
        address.get("road")
        or address.get("pedestrian")
        or address.get("residential")
        or address.get("footway")
        or address.get("path")
    )


def geocode_queries(
    session,
    *,
    school_name,
    address,
    municipality_name,
    boundary,
    pause_seconds,
):
    common = {
        "format": "jsonv2",
        "limit": 5,
        "countrycodes": "it",
        "addressdetails": 1,
        "namedetails": 1,
        "bounded": 1,
        "viewbox": boundary["viewbox"],
    }

    variants = []

    if address:
        variants.append(
            (
                "structured_address",
                {
                    **common,
                    "street": address,
                    "city": municipality_name,
                },
            )
        )

    if school_name and address:
        variants.append(
            (
                "name_address",
                {
                    **common,
                    "q": f"{school_name}, {address}, {municipality_name}, Italia",
                },
            )
        )

    if address:
        variants.append(
            (
                "freeform_address",
                {
                    **common,
                    "q": f"{address}, {municipality_name}, Italia",
                },
            )
        )

    all_results = []

    for query_type, params in variants:
        response = session.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            timeout=60,
        )
        response.raise_for_status()

        results = response.json()
        time.sleep(pause_seconds)

        for result in results:
            all_results.append(
                {
                    "query_type": query_type,
                    "result": result,
                }
            )

    return all_results


def score_candidate(
    *,
    school_name,
    school_address,
    candidate,
    boundary_geometry,
):
    raw = candidate["result"]

    try:
        lon = float(raw["lon"])
        lat = float(raw["lat"])
    except Exception:
        return None

    display_name = clean_text(
        raw.get("display_name")
    ) or ""

    address_dict = raw.get("address")
    if not isinstance(address_dict, dict):
        address_dict = {}

    namedetails = raw.get("namedetails")
    if not isinstance(namedetails, dict):
        namedetails = {}

    result_name = (
        namedetails.get("name")
        or raw.get("name")
        or display_name.split(",")[0]
    )

    result_road = get_result_road(
        address_dict
    )

    result_house_number = clean_text(
        address_dict.get("house_number")
    )

    requested_address = normalize_address(
        school_address
    )

    result_address = normalize_address(
        " ".join(
            item
            for item in [
                result_road,
                result_house_number,
            ]
            if item
        )
        or display_name
    )

    address_score = float(
        fuzz.token_set_ratio(
            requested_address,
            result_address,
        )
    )

    requested_name = school_name_core(
        school_name
    )

    result_name_core = school_name_core(
        result_name
    )

    name_score = float(
        fuzz.token_set_ratio(
            requested_name,
            result_name_core,
        )
    ) if requested_name and result_name_core else 0.0

    requested_civic = extract_civic(
        school_address
    )

    civic_match = None

    if requested_civic:
        civic_match = (
            normalize_civic(requested_civic)
            == normalize_civic(result_house_number)
            if result_house_number
            else False
        )

    inside = inside_boundary(
        boundary_geometry,
        lon,
        lat,
    )

    osm_class = clean_text(
        raw.get("class")
    )

    osm_type = clean_text(
        raw.get("type")
    )

    is_school_like = (
        normalize_text(osm_type)
        in {
            "SCHOOL",
            "KINDERGARTEN",
            "COLLEGE",
            "UNIVERSITY",
        }
        or normalize_text(osm_class)
        in {
            "AMENITY",
            "BUILDING",
        }
        and name_score >= 75.0
    )

    if requested_civic:
        if (
            inside
            and civic_match is True
            and address_score >= 85.0
        ):
            resolution = "address"
            status = "accepted_address"
            confidence = "high"
        elif (
            inside
            and is_school_like
            and name_score >= 85.0
            and address_score >= 60.0
        ):
            resolution = "site"
            status = "accepted_site"
            confidence = "high"
        elif (
            inside
            and address_score >= 85.0
        ):
            resolution = "street_anchor"
            status = "street_anchor_candidate"
            confidence = "medium"
        else:
            resolution = None
            status = "review"
            confidence = "low"
    else:
        if (
            inside
            and is_school_like
            and name_score >= 85.0
            and address_score >= 60.0
        ):
            resolution = "site"
            status = "accepted_site"
            confidence = "high"
        elif (
            inside
            and address_score >= 85.0
        ):
            resolution = "street_anchor"
            status = "street_anchor_candidate"
            confidence = "medium"
        else:
            resolution = None
            status = "review"
            confidence = "low"

    return {
        "query_type": candidate["query_type"],
        "display_name": display_name,
        "longitude": lon,
        "latitude": lat,
        "inside_target": inside,
        "result_name": result_name,
        "result_road": result_road,
        "result_house_number": result_house_number,
        "address_score": address_score,
        "name_score": name_score,
        "civic_match": civic_match,
        "osm_class": osm_class,
        "osm_type": osm_type,
        "status": status,
        "confidence": confidence,
        "resolution": resolution,
    }


def candidate_sort_key(candidate):
    status_rank = {
        "accepted_address": 5,
        "accepted_site": 4,
        "street_anchor_candidate": 3,
        "review": 1,
    }

    return (
        status_rank.get(
            candidate["status"],
            0,
        ),
        bool(candidate["inside_target"]),
        candidate["address_score"],
        candidate["name_score"],
    )


def main():
    args = parse_args()

    processed_dir = (
        PROCESSED_MIM_DIR
        / args.municipality_code
    )

    features_dir = (
        FEATURES_MIM_DIR
        / args.municipality_code
    )

    features_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry_path = (
        processed_dir
        / f"schools_registry_{args.school_year}.parquet"
    )

    locations_path = (
        processed_dir
        / f"school_locations_{args.school_year}_v2.parquet"
    )

    unmatched_path = (
        processed_dir
        / f"school_unmatched_classification_{args.school_year}.parquet"
    )

    registry = pd.read_parquet(
        registry_path
    )

    locations = pd.read_parquet(
        locations_path
    )

    unmatched = pd.read_parquet(
        unmatched_path
    )

    for df in [
        registry,
        locations,
        unmatched,
    ]:
        df["school_code"] = (
            df["school_code"]
            .astype("string")
            .str.strip()
        )

    paritary_codes = set(
        registry.loc[
            registry["registry_type"]
            .astype(str)
            .str.lower()
            .str.contains("par"),
            "school_code",
        ]
        .dropna()
        .astype(str)
    )

    physical_unmatched_codes = set(
        unmatched.loc[
            unmatched["needs_geolocation"] == True,
            "school_code",
        ]
        .dropna()
        .astype(str)
    )

    target_codes = (
        paritary_codes
        | physical_unmatched_codes
    )

    target_registry = (
        registry.loc[
            registry["school_code"].isin(
                target_codes
            )
        ]
        .copy()
    )

    municipality_names = (
        target_registry[
            "municipality_name"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if len(municipality_names) != 1:
        raise RuntimeError(
            "Impossibile determinare un unico comune target: "
            + repr(municipality_names)
        )

    municipality_name = (
        municipality_names[0]
    )

    boundary = load_boundary(
        args.municipality_code
    )

    locations_by_code = (
        locations.set_index(
            "school_code"
        )
    )

    # Accepted OSM sites by normalized official address:
    # used only for conservative sibling propagation.
    accepted_by_address = {}

    for _, row in locations.iterrows():
        if not bool(
            row.get(
                "geometry_accepted",
                False,
            )
        ):
            continue

        address = normalize_address(
            row.get("address")
        )

        site_id = clean_text(
            row.get(
                "candidate_site_id"
            )
        )

        if not address:
            continue

        accepted_by_address.setdefault(
            address,
            [],
        ).append(
            {
                "school_code":
                    row["school_code"],
                "candidate_site_id":
                    site_id,
                "longitude":
                    row.get("longitude"),
                "latitude":
                    row.get("latitude"),
                "candidate_site_name":
                    row.get("candidate_site_name"),
            }
        )

    cache_dir = (
        RAW_MIM_DIR
        / "geocoding"
        / args.municipality_code
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        cache_dir
        / f"nominatim_remaining_school_services_{args.school_year}.json"
    )

    cache = load_cache(
        cache_path
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent":
                "urban-digital-twin-thesis/1.0 (academic research)"
        }
    )

    summary_rows = []
    candidate_rows = []

    for _, school in (
        target_registry
        .sort_values("school_code")
        .iterrows()
    ):
        code = str(
            school["school_code"]
        )

        name = clean_text(
            school.get("school_name")
        )

        address = clean_text(
            school.get("school_address")
        )

        registry_type = clean_text(
            school.get("registry_type")
        )

        loc = (
            locations_by_code.loc[code]
            if code in locations_by_code.index
            else None
        )

        if isinstance(
            loc,
            pd.DataFrame,
        ):
            raise RuntimeError(
                f"school_code duplicato nelle locations: {code}"
            )

        # ----------------------------------------------------
        # 1. Existing high-confidence OSM match
        # ----------------------------------------------------
        if (
            loc is not None
            and bool(
                loc.get(
                    "geometry_accepted",
                    False,
                )
            )
        ):
            summary_rows.append(
                {
                    "school_code": code,
                    "school_name": name,
                    "school_address": address,
                    "registry_type": registry_type,
                    "resolution_status": "accepted_osm",
                    "resolution_method": "existing_osm_auto_match",
                    "confidence": "high",
                    "coordinate_resolution": "site",
                    "longitude": loc.get("longitude"),
                    "latitude": loc.get("latitude"),
                    "matched_name": loc.get("candidate_site_name"),
                    "matched_address": loc.get("candidate_site_address"),
                    "notes": "Match OSM già accettato dalla pipeline V2.",
                }
            )
            continue

        # ----------------------------------------------------
        # 2. Conservative same-address + same-OSM-site propagation
        # ----------------------------------------------------
        normalized_address = normalize_address(
            address
        )

        current_candidate_site_id = (
            clean_text(
                loc.get(
                    "candidate_site_id"
                )
            )
            if loc is not None
            else None
        )

        peers = (
            accepted_by_address.get(
                normalized_address,
                [],
            )
            if normalized_address
            else []
        )

        peer = None

        for item in peers:
            if (
                current_candidate_site_id
                and item[
                    "candidate_site_id"
                ]
                == current_candidate_site_id
            ):
                peer = item
                break

        if peer is not None:
            summary_rows.append(
                {
                    "school_code": code,
                    "school_name": name,
                    "school_address": address,
                    "registry_type": registry_type,
                    "resolution_status": "accepted_peer_site",
                    "resolution_method": "same_address_same_osm_site",
                    "confidence": "high",
                    "coordinate_resolution": "site",
                    "longitude": peer["longitude"],
                    "latitude": peer["latitude"],
                    "matched_name": peer["candidate_site_name"],
                    "matched_address": address,
                    "notes": (
                        "Stesso indirizzo ufficiale e stesso candidato OSM "
                        f"di un record già accettato ({peer['school_code']})."
                    ),
                }
            )
            continue

        # ----------------------------------------------------
        # 3. Nominatim bounded fallback
        # ----------------------------------------------------
        cache_key = (
            f"{args.municipality_code}|"
            f"{normalize_text(name)}|"
            f"{normalize_address(address)}"
        )

        if (
            not args.refresh
            and cache_key in cache
        ):
            raw_candidates = cache[
                cache_key
            ]["results"]
        else:
            raw_candidates = geocode_queries(
                session,
                school_name=name,
                address=address,
                municipality_name=municipality_name,
                boundary=boundary,
                pause_seconds=args.pause_seconds,
            )

            cache[
                cache_key
            ] = {
                "school_code": code,
                "school_name": name,
                "school_address": address,
                "queried_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "results": raw_candidates,
            }

            save_cache(
                cache_path,
                cache,
            )

        scored = []

        for raw_candidate in raw_candidates:
            candidate = score_candidate(
                school_name=name,
                school_address=address,
                candidate=raw_candidate,
                boundary_geometry=boundary[
                    "geometry"
                ],
            )

            if candidate is not None:
                scored.append(
                    candidate
                )

        scored.sort(
            key=candidate_sort_key,
            reverse=True,
        )

        for rank, candidate in enumerate(
            scored,
            start=1,
        ):
            candidate_rows.append(
                {
                    "school_code": code,
                    "school_name": name,
                    "school_address": address,
                    "rank": rank,
                    **candidate,
                }
            )

        best = (
            scored[0]
            if scored
            else None
        )

        if best is None:
            summary_rows.append(
                {
                    "school_code": code,
                    "school_name": name,
                    "school_address": address,
                    "registry_type": registry_type,
                    "resolution_status": "unresolved",
                    "resolution_method": "nominatim_bounded",
                    "confidence": "low",
                    "coordinate_resolution": None,
                    "longitude": None,
                    "latitude": None,
                    "matched_name": None,
                    "matched_address": None,
                    "notes": "Nessun candidato geocoding disponibile.",
                }
            )
            continue

        summary_rows.append(
            {
                "school_code": code,
                "school_name": name,
                "school_address": address,
                "registry_type": registry_type,
                "resolution_status": best[
                    "status"
                ],
                "resolution_method": "nominatim_bounded",
                "confidence": best[
                    "confidence"
                ],
                "coordinate_resolution": best[
                    "resolution"
                ],
                "longitude": best[
                    "longitude"
                ],
                "latitude": best[
                    "latitude"
                ],
                "matched_name": best[
                    "display_name"
                ],
                "matched_address": " ".join(
                    item
                    for item in [
                        best[
                            "result_road"
                        ],
                        best[
                            "result_house_number"
                        ],
                    ]
                    if item
                )
                or None,
                "notes": (
                    f"address_score={best['address_score']:.2f}; "
                    f"name_score={best['name_score']:.2f}; "
                    f"civic_match={best['civic_match']}; "
                    f"inside_target={best['inside_target']}"
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    candidates = pd.DataFrame(
        candidate_rows
    )

    output_csv = (
        features_dir
        / f"remaining_school_services_{args.school_year}.csv"
    )

    output_parquet = (
        processed_dir
        / f"remaining_school_services_{args.school_year}.parquet"
    )

    candidates_csv = (
        features_dir
        / f"remaining_school_service_candidates_{args.school_year}.csv"
    )

    summary.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_parquet(
        output_parquet,
        index=False,
    )

    candidates.to_csv(
        candidates_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n===================================="
    )
    print(
        " REMAINING SCHOOL SERVICES"
    )
    print(
        "===================================="
    )

    print(
        f"Paritary schools: {len(paritary_codes)}"
    )

    print(
        "Physical unmatched schools: "
        f"{len(physical_unmatched_codes)}"
    )

    print(
        f"Target records: {len(summary)}"
    )

    print(
        "\n=== RESULTS ==="
    )

    print(
        summary[
            [
                "school_code",
                "school_name",
                "school_address",
                "registry_type",
                "resolution_status",
                "resolution_method",
                "confidence",
                "coordinate_resolution",
                "matched_name",
                "longitude",
                "latitude",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nStatuses:"
    )

    print(
        summary[
            "resolution_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(f"✓ {output_csv}")
    print(f"✓ {output_parquet}")
    print(f"✓ {candidates_csv}")

    print(
        "\nNOTA:"
    )

    print(
        "Lo script seleziona automaticamente paritarie e unmatched fisici; "
        "non contiene codici scuola o regole specifiche di Matera."
    )


if __name__ == "__main__":
    main()
