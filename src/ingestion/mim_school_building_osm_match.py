import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# PATHS / CONSTANTS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = (
    ROOT / "data" / "processed" / "mim"
)

PROCESSED_OSM_DIR = (
    ROOT / "data" / "processed" / "osm"
)

DEFAULT_SCHOOL_YEAR = "202627"
DEFAULT_BUILDING_YEAR = "202425"

# Conservative thresholds.
AUTO_MIN_SCORE = 80.0
AUTO_MIN_MARGIN = 8.0
REVIEW_MIN_SCORE = 60.0

# Spatial evidence is secondary: it must support textual evidence,
# never replace it.
STRONG_DISTANCE_M = 120.0
PLAUSIBLE_DISTANCE_M = 250.0
CONFLICT_DISTANCE_M = 1000.0


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Matching conservativo tra edifici scolastici MIM "
            "e siti scolastici OSM consolidati."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--school-year",
        default=DEFAULT_SCHOOL_YEAR,
        help="Anno scolastico anagrafica corrente, es. 202627.",
    )

    parser.add_argument(
        "--building-year",
        default=DEFAULT_BUILDING_YEAR,
        help="Anno di riferimento edilizia MIM, es. 202425.",
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
            "municipality-code deve avere esattamente 6 cifre."
        )

    return args


# ============================================================
# GENERIC HELPERS
# ============================================================

def is_missing(value):
    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

        if hasattr(result, "item"):
            return bool(result.item())

    except Exception:
        pass

    return False


def clean_text(value):
    if is_missing(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.upper() in {
        "NAN",
        "NONE",
        "NULL",
        "N/A",
        "NA",
        "N.D.",
        "ND",
        "NON DISPONIBILE",
        "-",
    }:
        return None

    return value


def normalize_text(value):
    value = clean_text(value)

    if value is None:
        return None

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = value.upper()

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value or None


def normalize_address(value):
    value = normalize_text(value)

    if value is None:
        return None

    replacements = {
        "PIAZZA GIOVANNI SEMERIA": "PIAZZA SEMERIA",
        "P ZZA": "PIAZZA",
        "P ZA": "PIAZZA",
        "V LE": "VIALE",
        "C DA": "CONTRADA",
        "LOCALITA CONTRADA": "CONTRADA",
        "LOCALITA": "",
        "S N C": "",
        "SNC": "",
        "S N": "",
    }

    for old, new in replacements.items():
        value = value.replace(
            old,
            new,
        )

    # "0" in some official addresses is a missing-house-number marker.
    value = re.sub(
        r"\s+0$",
        "",
        value,
    )

    value = re.sub(
        r"\s+S N$",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value or None


def similarity(
    left,
    right,
):
    if not left or not right:
        return 0.0

    return float(
        max(
            fuzz.ratio(
                left,
                right,
            ),
            fuzz.token_sort_ratio(
                left,
                right,
            ),
        )
    )


def haversine_m(
    lon1,
    lat1,
    lon2,
    lat2,
):
    values = [
        lon1,
        lat1,
        lon2,
        lat2,
    ]

    if any(
        value is None
        for value in values
    ):
        return None

    try:
        values = [
            float(value)
            for value in values
        ]
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not all(
        math.isfinite(value)
        for value in values
    ):
        return None

    (
        lon1,
        lat1,
        lon2,
        lat2,
    ) = values

    radius_m = 6371008.8

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(
        lat2 - lat1
    )

    delta_lambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            delta_phi / 2.0
        ) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(
            delta_lambda / 2.0
        ) ** 2
    )

    c = 2.0 * math.atan2(
        math.sqrt(a),
        math.sqrt(
            max(
                0.0,
                1.0 - a,
            )
        ),
    )

    return radius_m * c


def parse_possible_list(value):
    if is_missing(value):
        return []

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            clean_text(item)
            for item in value
            if clean_text(item)
        ]

    value = str(value).strip()

    if not value:
        return []

    # Several Silver datasets serialize list-valued fields as JSON.
    if (
        value.startswith("[")
        and value.endswith("]")
    ):
        try:
            decoded = json.loads(value)

            if isinstance(decoded, list):
                return [
                    clean_text(item)
                    for item in decoded
                    if clean_text(item)
                ]
        except Exception:
            pass

    return [
        value
    ]


def first_existing_column(
    dataframe,
    candidates,
    required=False,
):
    for column in candidates:
        if column in dataframe.columns:
            return column

    if required:
        raise RuntimeError(
            "Nessuna delle colonne attese è presente: "
            + ", ".join(candidates)
        )

    return None


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs(
    municipality_code,
    school_year,
    building_year,
):
    mim_directory = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    osm_directory = (
        PROCESSED_OSM_DIR
        / municipality_code
    )

    buildings_path = (
        mim_directory
        / (
            "physical_school_buildings_"
            f"{building_year}_geocoded.parquet"
        )
    )

    links_path = (
        mim_directory
        / (
            "school_building_links_"
            f"{school_year}_from_"
            f"{building_year}.parquet"
        )
    )

    sites_path = (
        osm_directory
        / "school_sites.parquet"
    )

    for path in [
        buildings_path,
        links_path,
        sites_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"File richiesto non trovato: {path}"
            )

    buildings = pd.read_parquet(
        buildings_path
    )

    links = pd.read_parquet(
        links_path
    )

    # Compatibility with the canonical schema produced by
    # mim_school_buildings_generalized.py.
    # Keep the legacy alias internally because downstream matching
    # code still refers to CODICEEDIFICIO.
    if (
        "CODICEEDIFICIO" not in links.columns
        and "building_code" in links.columns
    ):
        links["CODICEEDIFICIO"] = links["building_code"]

    sites = pd.read_parquet(
        sites_path
    )

    if (
        "building_code"
        not in buildings.columns
    ):
        raise RuntimeError(
            "building_code mancante nel layer edifici."
        )

    if buildings[
        "building_code"
    ].duplicated().any():
        raise RuntimeError(
            "building_code duplicati nel layer edifici."
        )

    if (
        "school_code"
        not in links.columns
        or "CODICEEDIFICIO"
        not in links.columns
    ):
        raise RuntimeError(
            "Il dataset school_building_links non contiene "
            "school_code / CODICEEDIFICIO."
        )

    return (
        buildings,
        links,
        sites,
    )


# ============================================================
# LINKED SCHOOL NAMES
# ============================================================

def build_linked_school_evidence(
    links,
):
    name_columns = [
        column
        for column in [
            "school_name",
            "reference_institute_name",
            "institute_name",
        ]
        if column in links.columns
    ]

    grouped = {}

    for (
        building_code,
        group,
    ) in links.groupby(
        "CODICEEDIFICIO",
        dropna=True,
    ):
        school_codes = sorted(
            {
                str(value)
                for value
                in group[
                    "school_code"
                ]
                .dropna()
                .tolist()
            }
        )

        names = []

        for column in name_columns:
            for value in group[
                column
            ].tolist():
                cleaned = clean_text(
                    value
                )

                if (
                    cleaned
                    and cleaned
                    not in names
                ):
                    names.append(
                        cleaned
                    )

        grouped[
            str(building_code)
        ] = {
            "linked_school_codes":
                school_codes,

            "linked_school_names":
                names,
        }

    return grouped


# ============================================================
# OSM SITE ADAPTER
# ============================================================

def prepare_osm_sites(
    sites,
):
    site_id_column = (
        first_existing_column(
            sites,
            [
                "site_id",
                "osm_site_id",
                "candidate_site_id",
            ],
            required=True,
        )
    )

    longitude_column = (
        first_existing_column(
            sites,
            [
                "longitude",
                "site_longitude",
                "candidate_longitude",
                "lon",
            ],
            required=True,
        )
    )

    latitude_column = (
        first_existing_column(
            sites,
            [
                "latitude",
                "site_latitude",
                "candidate_latitude",
                "lat",
            ],
            required=True,
        )
    )

    address_columns = [
        column
        for column in [
            "site_address",
            "address",
            "osm_address",
            "candidate_site_address",
            "raw_address",
        ]
        if column in sites.columns
    ]

    scalar_name_columns = [
        column
        for column in [
            "site_name",
            "name",
            "candidate_site_name",
            "primary_name",
            "official_name",
        ]
        if column in sites.columns
    ]

    list_name_columns = [
        column
        for column in [
            "primary_names",
            "core_primary_names",
            "operator_names",
        ]
        if column in sites.columns
    ]

    rows = []

    for _, row in sites.iterrows():
        site_names = []

        for column in scalar_name_columns:
            value = clean_text(
                row.get(column)
            )

            if (
                value
                and value
                not in site_names
            ):
                site_names.append(
                    value
                )

        for column in list_name_columns:
            for value in (
                parse_possible_list(
                    row.get(column)
                )
            ):
                if (
                    value
                    and value
                    not in site_names
                ):
                    site_names.append(
                        value
                    )

        addresses = []

        for column in address_columns:
            values = (
                parse_possible_list(
                    row.get(column)
                )
            )

            for value in values:
                if (
                    value
                    and value
                    not in addresses
                ):
                    addresses.append(
                        value
                    )

        site_name = (
            site_names[0]
            if site_names
            else None
        )

        site_address = (
            addresses[0]
            if addresses
            else None
        )

        rows.append(
            {
                "site_id":
                    str(
                        row[
                            site_id_column
                        ]
                    ),

                "site_name":
                    site_name,

                "site_names":
                    site_names,

                "site_address":
                    site_address,

                "site_addresses":
                    addresses,

                "longitude":
                    row[
                        longitude_column
                    ],

                "latitude":
                    row[
                        latitude_column
                    ],
            }
        )

    output = pd.DataFrame(
        rows
    )

    if output[
        "site_id"
    ].duplicated().any():
        raise RuntimeError(
            "site_id OSM duplicati dopo adattamento."
        )

    return output


# ============================================================
# SCORING
# ============================================================

def best_name_similarity(
    linked_names,
    osm_names,
):
    best_score = 0.0
    best_left = None
    best_right = None

    normalized_linked = [
        (
            value,
            normalize_text(value),
        )
        for value in linked_names
        if normalize_text(value)
    ]

    normalized_osm = [
        (
            value,
            normalize_text(value),
        )
        for value in osm_names
        if normalize_text(value)
    ]

    for (
        original_left,
        normalized_left,
    ) in normalized_linked:
        for (
            original_right,
            normalized_right,
        ) in normalized_osm:
            score = similarity(
                normalized_left,
                normalized_right,
            )

            if score > best_score:
                best_score = score
                best_left = original_left
                best_right = original_right

    return (
        best_score,
        best_left,
        best_right,
    )


def best_address_similarity(
    official_address,
    osm_addresses,
):
    official_normalized = (
        normalize_address(
            official_address
        )
    )

    if not official_normalized:
        return (
            0.0,
            None,
        )

    best_score = 0.0
    best_address = None

    for address in osm_addresses:
        candidate = (
            normalize_address(
                address
            )
        )

        score = similarity(
            official_normalized,
            candidate,
        )

        if score > best_score:
            best_score = score
            best_address = address

    return (
        best_score,
        best_address,
    )


def score_site(
    building,
    linked_names,
    site,
):
    (
        name_score,
        matched_school_name,
        matched_osm_name,
    ) = best_name_similarity(
        linked_names,
        site[
            "site_names"
        ],
    )

    (
        address_score,
        matched_osm_address,
    ) = best_address_similarity(
        building.get(
            "official_building_address"
        ),
        site[
            "site_addresses"
        ],
    )

    geocoder_status = clean_text(
        building.get(
            "geocoding_status"
        )
    )

    geocoder_lon = (
        building.get(
            "longitude"
        )
    )

    geocoder_lat = (
        building.get(
            "latitude"
        )
    )

    distance_m = None

    if geocoder_status in {
        "accepted_candidate",
        "review",
    }:
        distance_m = haversine_m(
            geocoder_lon,
            geocoder_lat,
            site[
                "longitude"
            ],
            site[
                "latitude"
            ],
        )

    # Building-level matching: the official MIM building address is
    # the primary evidence. School names are secondary because the
    # administrative MIM name can differ substantially from the OSM POI.
    score = (
        0.60 * address_score
        + 0.28 * name_score
    )

    spatial_bonus = 0.0

    if distance_m is not None:
        if distance_m <= 60:
            spatial_bonus = 10.0

        elif (
            distance_m
            <= STRONG_DISTANCE_M
        ):
            spatial_bonus = 7.0

        elif (
            distance_m
            <= PLAUSIBLE_DISTANCE_M
        ):
            spatial_bonus = 3.0

        elif (
            distance_m
            >= CONFLICT_DISTANCE_M
        ):
            spatial_bonus = -10.0

    score += spatial_bonus

    # Near-exact official address is very strong physical evidence.
    if address_score >= 98:
        score += 5.0

    # Exact names remain useful as an additional confirmation.
    if name_score >= 97:
        score += 2.0

    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )

    return {
        "site_id":
            site[
                "site_id"
            ],

        "site_name":
            site[
                "site_name"
            ],

        "site_address":
            site[
                "site_address"
            ],

        "site_longitude":
            site[
                "longitude"
            ],

        "site_latitude":
            site[
                "latitude"
            ],

        "name_score":
            name_score,

        "address_score":
            address_score,

        "osm_geocoder_distance_m":
            distance_m,

        "matched_school_name":
            matched_school_name,

        "matched_osm_name":
            matched_osm_name,

        "matched_osm_address":
            matched_osm_address,

        "spatial_bonus":
            spatial_bonus,

        "match_score":
            score,
    }


# ============================================================
# STATUS RULES
# ============================================================

def classify_match(
    best,
    margin,
):
    if best is None:
        return (
            "unresolved",
            "nessun candidato OSM",
        )

    score = best[
        "match_score"
    ]

    name_score = best[
        "name_score"
    ]

    address_score = best[
        "address_score"
    ]

    distance_m = best[
        "osm_geocoder_distance_m"
    ]

    # Rule A: official address is nearly exact and either the independent
    # geocoder confirms the same place or the linked-school name is good.
    # This correctly handles cases such as Marconi, Dante Alighieri and
    # Loperfido-Olivetti without requiring identical administrative names.
    if (
        score >= 82.0
        and address_score >= 97.0
        and margin >= AUTO_MIN_MARGIN
        and (
            (
                distance_m is not None
                and distance_m <= STRONG_DISTANCE_M
            )
            or name_score >= 60.0
        )
    ):
        return (
            "matched_auto",
            "indirizzo ufficiale quasi esatto con conferma spaziale o nominale",
        )

    # Rule B: strong address, good linked-school name and unique candidate.
    if (
        score >= AUTO_MIN_SCORE
        and address_score >= 92.0
        and name_score >= 55.0
        and margin >= AUTO_MIN_MARGIN
        and (
            distance_m is None
            or distance_m <= PLAUSIBLE_DISTANCE_M
        )
    ):
        return (
            "matched_auto",
            "indirizzo forte, nome coerente e candidato univoco",
        )

    # Rule C: very strong school name plus independent spatial convergence.
    if (
        score >= AUTO_MIN_SCORE
        and name_score >= 92.0
        and distance_m is not None
        and distance_m <= STRONG_DISTANCE_M
        and margin >= AUTO_MIN_MARGIN
    ):
        return (
            "matched_auto",
            "nome molto forte e convergenza con geocoder edificio",
        )

    if score >= REVIEW_MIN_SCORE:
        if (
            margin
            < AUTO_MIN_MARGIN
        ):
            return (
                "review",
                "candidato plausibile ma ambiguo rispetto al secondo",
            )

        if (
            distance_m is not None
            and distance_m
            >= CONFLICT_DISTANCE_M
        ):
            return (
                "review",
                "testo plausibile ma conflitto spaziale con geocoder edificio",
            )

        return (
            "review",
            "evidenza plausibile ma insufficiente per auto-match",
        )

    return (
        "unresolved",
        "score sotto soglia",
    )


# ============================================================
# MATCHING
# ============================================================

def match_buildings(
    buildings,
    links,
    sites,
):
    linked_evidence = (
        build_linked_school_evidence(
            links
        )
    )

    prepared_sites = (
        prepare_osm_sites(
            sites
        )
    )

    # Determine the target municipality from the canonical
    # school-building links rather than using municipality-specific
    # hard-coded values.
    if "municipality_name" not in links.columns:
        raise RuntimeError(
            "municipality_name mancante nel dataset school_building_links."
        )

    target_municipalities = {
        (
            clean_text(value)
            or ""
        ).upper()
        for value in links["municipality_name"].dropna()
        if clean_text(value)
    }

    if len(target_municipalities) != 1:
        raise RuntimeError(
            "Impossibile determinare univocamente il comune target "
            "da school_building_links: "
            + ", ".join(sorted(target_municipalities))
        )

    target_municipality_name = next(
        iter(target_municipalities)
    )

    result_rows = []
    ranking_rows = []

    target_buildings = buildings.copy()

    total = len(
        target_buildings
    )

    for position, (
        _,
        building_row,
    ) in enumerate(
        target_buildings.iterrows(),
        start=1,
    ):
        building = (
            building_row.to_dict()
        )

        building_code = str(
            building[
                "building_code"
            ]
        )

        building_municipality = (
            clean_text(
                building.get(
                    "building_municipality_name"
                )
            )
            or ""
        ).upper()

        evidence = linked_evidence.get(
            building_code,
            {
                "linked_school_codes":
                    [],
                "linked_school_names":
                    [],
            },
        )

        linked_school_codes = (
            evidence[
                "linked_school_codes"
            ]
        )

        linked_school_names = (
            evidence[
                "linked_school_names"
            ]
        )

        print(
            f"[{position}/{total}] "
            f"{building_code} "
            f"{building.get('official_building_address')}"
        )

        if (
            building_municipality
            != target_municipality_name
        ):
            result_rows.append(
                {
                    "building_code":
                        building_code,

                    "building_municipality_name":
                        building.get(
                            "building_municipality_name"
                        ),

                    "official_building_address":
                        building.get(
                            "official_building_address"
                        ),

                    "linked_school_codes":
                        json.dumps(
                            linked_school_codes,
                            ensure_ascii=False,
                        ),

                    "linked_school_names":
                        json.dumps(
                            linked_school_names,
                            ensure_ascii=False,
                        ),

                    "osm_match_status":
                        "outside_target_municipality",

                    "status_reason":
                        (
                            "edificio ufficialmente appartenente a comune "
                            f"diverso dal target ({target_municipality_name})"
                        ),

                    "candidate_site_id":
                        None,

                    "candidate_site_name":
                        None,

                    "candidate_site_address":
                        None,

                    "candidate_longitude":
                        None,

                    "candidate_latitude":
                        None,

                    "match_score":
                        None,

                    "name_score":
                        None,

                    "address_score":
                        None,

                    "score_margin":
                        None,

                    "osm_geocoder_distance_m":
                        None,

                    "geometry_accepted":
                        False,

                    "longitude":
                        None,

                    "latitude":
                        None,
                }
            )

            continue

        candidates = []

        for _, site_row in (
            prepared_sites.iterrows()
        ):
            candidate = score_site(
                building=building,
                linked_names=(
                    linked_school_names
                ),
                site=(
                    site_row.to_dict()
                ),
            )

            candidates.append(
                candidate
            )

        candidates.sort(
            key=lambda item:
                item[
                    "match_score"
                ],
            reverse=True,
        )

        top_candidates = (
            candidates[:5]
        )

        for rank, candidate in enumerate(
            top_candidates,
            start=1,
        ):
            ranking_rows.append(
                {
                    "building_code":
                        building_code,

                    "rank":
                        rank,

                    **candidate,
                }
            )

        if not top_candidates:
            best = None
            margin = 0.0

        else:
            best = top_candidates[0]

            second_score = (
                top_candidates[1][
                    "match_score"
                ]
                if len(
                    top_candidates
                ) > 1
                else 0.0
            )

            margin = (
                best[
                    "match_score"
                ]
                - second_score
            )

        (
            status,
            reason,
        ) = classify_match(
            best,
            margin,
        )

        accepted = (
            status
            == "matched_auto"
        )

        result_rows.append(
            {
                "building_code":
                    building_code,

                "building_municipality_name":
                    building.get(
                        "building_municipality_name"
                    ),

                "official_building_address":
                    building.get(
                        "official_building_address"
                    ),

                "building_geocoding_status":
                    building.get(
                        "geocoding_status"
                    ),

                "building_geocoder_longitude":
                    building.get(
                        "longitude"
                    ),

                "building_geocoder_latitude":
                    building.get(
                        "latitude"
                    ),

                "linked_school_codes":
                    json.dumps(
                        linked_school_codes,
                        ensure_ascii=False,
                    ),

                "linked_school_names":
                    json.dumps(
                        linked_school_names,
                        ensure_ascii=False,
                    ),

                "osm_match_status":
                    status,

                "status_reason":
                    reason,

                "candidate_site_id":
                    (
                        best[
                            "site_id"
                        ]
                        if best
                        else None
                    ),

                "candidate_site_name":
                    (
                        best[
                            "site_name"
                        ]
                        if best
                        else None
                    ),

                "candidate_site_address":
                    (
                        best[
                            "site_address"
                        ]
                        if best
                        else None
                    ),

                "candidate_longitude":
                    (
                        best[
                            "site_longitude"
                        ]
                        if best
                        else None
                    ),

                "candidate_latitude":
                    (
                        best[
                            "site_latitude"
                        ]
                        if best
                        else None
                    ),

                "matched_school_name":
                    (
                        best[
                            "matched_school_name"
                        ]
                        if best
                        else None
                    ),

                "matched_osm_name":
                    (
                        best[
                            "matched_osm_name"
                        ]
                        if best
                        else None
                    ),

                "matched_osm_address":
                    (
                        best[
                            "matched_osm_address"
                        ]
                        if best
                        else None
                    ),

                "match_score":
                    (
                        best[
                            "match_score"
                        ]
                        if best
                        else None
                    ),

                "name_score":
                    (
                        best[
                            "name_score"
                        ]
                        if best
                        else None
                    ),

                "address_score":
                    (
                        best[
                            "address_score"
                        ]
                        if best
                        else None
                    ),

                "score_margin":
                    margin,

                "osm_geocoder_distance_m":
                    (
                        best[
                            "osm_geocoder_distance_m"
                        ]
                        if best
                        else None
                    ),

                "geometry_accepted":
                    accepted,

                "longitude":
                    (
                        best[
                            "site_longitude"
                        ]
                        if accepted
                        else None
                    ),

                "latitude":
                    (
                        best[
                            "site_latitude"
                        ]
                        if accepted
                        else None
                    ),
            }
        )

    return (
        pd.DataFrame(
            result_rows
        ),
        pd.DataFrame(
            ranking_rows
        ),
    )


# ============================================================
# SAVE / SUMMARY
# ============================================================

def save_outputs(
    matches,
    rankings,
    municipality_code,
    building_year,
):
    directory = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    matches_path = (
        directory
        / (
            "school_building_osm_matches_"
            f"{building_year}_v2.parquet"
        )
    )

    ranking_path = (
        directory
        / (
            "school_building_osm_candidates_"
            f"{building_year}_v2.parquet"
        )
    )

    matches.to_parquet(
        matches_path,
        index=False,
    )

    rankings.to_parquet(
        ranking_path,
        index=False,
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ {matches_path}"
    )

    print(
        f"✓ {ranking_path}"
    )


def print_summary(
    matches,
):
    print(
        "\n===================================="
    )

    print(
        " MIM BUILDING ↔ OSM MATCH V2 COMPLETATO"
    )

    print(
        "===================================="
    )

    print(
        f"Edifici totali: {len(matches)}"
    )

    print(
        "\nStato matching:"
    )

    print(
        matches[
            "osm_match_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    accepted = (
        matches[
            "osm_match_status"
        ]
        == "matched_auto"
    )

    print(
        "\nMatch automatici accettati: "
        f"{int(accepted.sum())}"
    )

    print(
        "\n=== MATCH AUTOMATICI ==="
    )

    automatic = (
        matches[
            accepted
        ]
    )

    if automatic.empty:
        print(
            "Nessuno."
        )

    else:
        print(
            automatic[
                [
                    "building_code",
                    "official_building_address",
                    "candidate_site_name",
                    "candidate_site_address",
                    "match_score",
                    "name_score",
                    "address_score",
                    "score_margin",
                    "osm_geocoder_distance_m",
                ]
            ]
            .sort_values(
                "match_score",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n=== DA REVISIONARE ==="
    )

    review = (
        matches[
            matches[
                "osm_match_status"
            ]
            == "review"
        ]
    )

    if review.empty:
        print(
            "Nessuno."
        )

    else:
        print(
            review[
                [
                    "building_code",
                    "official_building_address",
                    "candidate_site_name",
                    "candidate_site_address",
                    "match_score",
                    "name_score",
                    "address_score",
                    "score_margin",
                    "osm_geocoder_distance_m",
                    "status_reason",
                ]
            ]
            .sort_values(
                "match_score",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )

    print(
        "\n=== NON RISOLTI ==="
    )

    unresolved = (
        matches[
            matches[
                "osm_match_status"
            ]
            == "unresolved"
        ]
    )

    if unresolved.empty:
        print(
            "Nessuno."
        )

    else:
        print(
            unresolved[
                [
                    "building_code",
                    "official_building_address",
                    "candidate_site_name",
                    "candidate_site_address",
                    "match_score",
                    "name_score",
                    "address_score",
                    "score_margin",
                    "status_reason",
                ]
            ]
            .sort_values(
                "match_score",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )

    shared = (
        matches[
            matches[
                "candidate_site_id"
            ]
            .notna()
        ]
        .groupby(
            "candidate_site_id"
        )[
            "building_code"
        ]
        .nunique()
    )

    shared = shared[
        shared > 1
    ]

    print(
        "\nSiti OSM candidati condivisi da più edifici: "
        f"{len(shared)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    print(
        "\n===================================="
    )

    print(
        " MIM PHYSICAL BUILDING ↔ OSM MATCH V2"
    )

    print(
        "===================================="
    )

    (
        buildings,
        links,
        sites,
    ) = load_inputs(
        municipality_code=(
            args.municipality_code
        ),
        school_year=(
            args.school_year
        ),
        building_year=(
            args.building_year
        ),
    )

    print(
        "\n=== INPUT ==="
    )

    print(
        f"Edifici MIM: {len(buildings)}"
    )

    print(
        f"Relazioni scuola-edificio: {len(links)}"
    )

    print(
        f"Siti OSM consolidati: {len(sites)}"
    )

    (
        matches,
        rankings,
    ) = match_buildings(
        buildings=buildings,
        links=links,
        sites=sites,
    )

    save_outputs(
        matches=matches,
        rankings=rankings,
        municipality_code=(
            args.municipality_code
        ),
        building_year=(
            args.building_year
        ),
    )

    print_summary(
        matches
    )


if __name__ == "__main__":
    main()
