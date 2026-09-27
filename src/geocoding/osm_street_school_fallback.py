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
PROCESSED_OSM_DIR = ROOT / "data" / "processed" / "osm"
PROCESSED_ISTAT_DIR = ROOT / "data" / "processed" / "istat"
FEATURES_MIM_DIR = ROOT / "data" / "features" / "mim"
RAW_MIM_DIR = ROOT / "data" / "raw" / "mim"


ROAD_PREFIXES = {
    "VIA",
    "VIALE",
    "PIAZZA",
    "PIAZZALE",
    "CORSO",
    "LARGO",
    "VICO",
    "VICOLO",
    "CONTRADA",
    "LOCALITA",
    "LOCALITÀ",
    "STRADA",
    "TRAVERSA",
    "SALITA",
    "DISCESA",
    "ROTONDA",
    "LUNGOMARE",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fallback nazionale per edifici scolastici non localizzati: "
            "estrae i nomi stradali dal grafo OSM del comune, effettua "
            "street matching fuzzy e usa il nome OSM selezionato per una "
            "nuova query Nominatim. Nessun alias comunale è hard-coded."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre.",
    )

    parser.add_argument(
        "--school-year",
        default="202425",
    )

    parser.add_argument(
        "--building-year",
        default="202425",
    )

    parser.add_argument(
        "--top-k-streets",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--auto-street-threshold",
        type=float,
        default=90.0,
    )

    parser.add_argument(
        "--review-street-threshold",
        type=float,
        default=75.0,
    )

    parser.add_argument(
        "--min-margin",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.1,
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
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
    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    return "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )


def normalize_text(value):
    value = clean_text(value)

    if value is None:
        return ""

    value = strip_accents(
        value
    )

    value = (
        value.upper()
        .replace("`", "'")
        .replace("’", "'")
    )

    value = re.sub(
        r"[^A-Z0-9']+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def split_official_address(value):
    """
    Split a MIM address into street text and civic number without
    introducing any municipality-specific rewrite.

    Examples:
        Via Mario Rosario Greco 12 -> (Via Mario Rosario Greco, 12)
        Via Petrarca snc            -> (Via Petrarca, None)
        Via Lucana 190/192          -> (Via Lucana, 190/192)
    """
    value = clean_text(value)

    if value is None:
        return None, None

    value = (
        value
        .replace("`", "'")
        .replace("’", "'")
    )

    # Remove common "senza numero civico" variants only at the end.
    value = re.sub(
        r"\s+\bS\s*\.?\s*N\s*\.?\s*C\s*\.?\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\s+\bS\s*\.?\s*N\s*\.?\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\s+SNC\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = value.strip()

    civic_pattern = (
        r"(?:^|\s)"
        r"(\d+(?:[A-Za-z])?"
        r"(?:[/-]\d+(?:[A-Za-z])?)*)"
        r"\s*$"
    )

    match = re.search(
        civic_pattern,
        value,
    )

    if match:
        civic = match.group(1)

        street = (
            value[
                :match.start(1)
            ]
            .strip(" ,")
        )
    else:
        civic = None
        street = value

    return (
        clean_text(street),
        clean_text(civic),
    )


def road_core_tokens(value):
    normalized = normalize_text(
        value
    )

    tokens = normalized.split()

    while (
        tokens
        and tokens[0] in ROAD_PREFIXES
    ):
        tokens = tokens[1:]

    # Remove pure road reference tokens when they are not useful
    # for ordinary municipal street matching.
    return tokens


def initials_signature(tokens):
    if not tokens:
        return ""

    if len(tokens) == 1:
        token = tokens[0]

        if (
            token.isalpha()
            and 1 <= len(token) <= 4
        ):
            return token

        return token[:1]

    return "".join(
        token[0]
        for token in tokens
        if token
    )


def street_similarity(
    official_street,
    osm_street,
):
    """
    General-purpose street-name reconciliation.

    It avoids a known failure mode of token_set_ratio: a short subset
    such as "Via Rosario" must not score 100 against
    "Via Mario Rosario Greco".

    The last core token is treated as a surname / discriminating token
    when available, while abbreviated given names such as
    "Mario Rosario" -> "MR" are handled through initials.
    """
    official_norm = normalize_text(
        official_street
    )

    osm_norm = normalize_text(
        osm_street
    )

    if (
        not official_norm
        or not osm_norm
    ):
        return {
            "street_score":
                0.0,

            "base_score":
                0.0,

            "core_score":
                0.0,

            "surname_score":
                0.0,

            "initials_compatible":
                False,

            "subset_penalty":
                False,
        }

    official_core = road_core_tokens(
        official_street
    )

    osm_core = road_core_tokens(
        osm_street
    )

    official_core_text = " ".join(
        official_core
    )

    osm_core_text = " ".join(
        osm_core
    )

    # Use symmetric similarities as the baseline. Unlike token_set_ratio,
    # these do not assign 100 to a short subset.
    base_score = float(
        fuzz.ratio(
            official_norm,
            osm_norm,
        )
    )

    core_score = float(
        fuzz.token_sort_ratio(
            official_core_text,
            osm_core_text,
        )
    )

    surname_score = 0.0
    initials_compatible = False
    subset_penalty = False

    if (
        official_core
        and osm_core
    ):
        surname_score = float(
            fuzz.ratio(
                official_core[-1],
                osm_core[-1],
            )
        )

        official_before_surname = (
            official_core[:-1]
        )

        osm_before_surname = (
            osm_core[:-1]
        )

        if surname_score >= 90.0:
            official_initials = (
                initials_signature(
                    official_before_surname
                )
                if official_before_surname
                else ""
            )

            osm_initials = (
                initials_signature(
                    osm_before_surname
                )
                if osm_before_surname
                else ""
            )

            if (
                official_initials
                and osm_initials
                and official_initials
                == osm_initials
            ):
                initials_compatible = True

        # A short candidate that omits the discriminating final token
        # must not win merely because it is a token subset.
        if (
            len(
                official_core
            ) >= 2
            and len(
                osm_core
            ) < len(
                official_core
            )
            and surname_score < 70.0
        ):
            subset_penalty = True

    if (
        official_core_text
        and osm_core_text
        and official_core_text
        == osm_core_text
    ):
        street_score = 100.0

    elif (
        surname_score >= 95.0
        and initials_compatible
    ):
        # Example:
        # Mario Rosario Greco <-> MR Greco
        street_score = 98.0

    elif (
        surname_score >= 95.0
        and (
            len(
                official_core
            ) == 1
            or len(
                osm_core
            ) == 1
        )
    ):
        # Handles enrichment such as:
        # Petrarca <-> Francesco Petrarca.
        street_score = max(
            92.0,
            (
                0.45
                * base_score
                + 0.30
                * core_score
                + 0.25
                * surname_score
            ),
        )

    else:
        street_score = (
            0.40
            * base_score
            + 0.35
            * core_score
            + 0.25
            * surname_score
        )

    if subset_penalty:
        street_score = min(
            street_score,
            65.0,
        )

    return {
        "street_score":
            float(
                street_score
            ),

        "base_score":
            base_score,

        "core_score":
            core_score,

        "surname_score":
            surname_score,

        "initials_compatible":
            initials_compatible,

        "subset_penalty":
            subset_penalty,
    }


def parse_attributes(value):
    if isinstance(value, dict):
        return value

    value = clean_text(value)

    if value is None:
        return {}

    try:
        decoded = json.loads(
            value
        )

        if isinstance(
            decoded,
            dict,
        ):
            return decoded
    except Exception:
        pass

    return {}


def extract_names_from_attributes(value):
    attrs = parse_attributes(
        value
    )

    name = attrs.get(
        "name"
    )

    if isinstance(
        name,
        list,
    ):
        return [
            clean_text(item)
            for item in name
            if clean_text(item)
        ]

    if clean_text(name):
        return [
            clean_text(name)
        ]

    return []


def load_pending_buildings(args):
    """
    Load only buildings still requiring automatic spatial resolution
    after the canonical geocoder + OSM fusion stage.

    This replaces the historical dependency on
    *_with_reused_locations.parquet and makes the fallback usable
    for zero-touch municipalities.
    """

    mim_dir = (
        PROCESSED_MIM_DIR
        / args.municipality_code
    )

    source_path = (
        mim_dir
        / (
            "physical_school_buildings_"
            f"{args.building_year}_"
            "final_v2.parquet"
        )
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Dataset fusion edifici non trovato: {source_path}"
        )

    buildings = (
        pd.read_parquet(
            source_path
        )
        .copy()
    )

    required_columns = {
        "building_code",
        "building_municipality_name",
        "official_building_address",
        "final_location_status",
    }

    missing = (
        required_columns
        - set(buildings.columns)
    )

    if missing:
        raise RuntimeError(
            "Colonne mancanti nel dataset fusion: "
            + ", ".join(sorted(missing))
        )

    buildings["building_code"] = (
        buildings["building_code"]
        .astype("string")
        .str.strip()
    )

    # Only genuinely pending buildings enter the national fallback.
    # Strong locations produced by the fusion stage are preserved.
    pending = (
        buildings.loc[
            buildings[
                "final_location_status"
            ].isin(
                [
                    "review",
                    "unresolved",
                ]
            )
        ]
        .copy()
    )

    # Prefer an explicit municipality code when available.
    if (
        "building_municipality_code"
        in pending.columns
    ):
        municipality_codes = (
            pending[
                "building_municipality_code"
            ]
            .astype("string")
            .str.strip()
            .str.zfill(6)
        )

        pending = (
            pending.loc[
                municipality_codes
                == args.municipality_code
            ]
            .copy()
        )

    else:
        # Generic fallback for datasets where building_code embeds
        # the six-digit ISTAT municipality code.
        pending = (
            pending.loc[
                pending[
                    "building_code"
                ]
                .astype(str)
                .str.startswith(
                    args.municipality_code
                )
            ]
            .copy()
        )

    return (
        pending,
        source_path,
    )



def load_osm_street_names(args):
    edges_path = (
        PROCESSED_OSM_DIR
        / args.municipality_code
        / "walk_edges.parquet"
    )

    if not edges_path.exists():
        raise FileNotFoundError(
            f"Rete OSM non trovata: {edges_path}"
        )

    edges = pd.read_parquet(
        edges_path,
        columns=[
            "attributes",
        ],
    )

    names = set()

    for value in edges[
        "attributes"
    ]:
        for name in (
            extract_names_from_attributes(
                value
            )
        ):
            names.add(
                name
            )

    street_names = sorted(
        names,
        key=lambda value: (
            normalize_text(value),
            value,
        ),
    )

    if not street_names:
        raise RuntimeError(
            "Nessun nome stradale trovato negli attributi OSM."
        )

    return (
        street_names,
        edges_path,
    )


def load_municipality_context(
    pending,
    args,
):
    census_path = (
        PROCESSED_ISTAT_DIR
        / (
            f"{args.municipality_code}_"
            "census_areas_2021.parquet"
        )
    )

    if not census_path.exists():
        raise FileNotFoundError(
            f"Dataset ISTAT non trovato: {census_path}"
        )

    census = (
        gpd.read_parquet(
            census_path
        )
        .to_crs(4326)
    )

    geometry = (
        census.geometry.union_all()
    )

    minx, miny, maxx, maxy = (
        geometry.bounds
    )

    municipality_names = (
        pending[
            "building_municipality_name"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if len(
        municipality_names
    ) != 1:
        raise RuntimeError(
            "Impossibile determinare un solo nome di comune: "
            + repr(
                municipality_names
            )
        )

    return {
        "municipality_name":
            municipality_names[0],

        "geometry":
            geometry,

        "viewbox":
            f"{minx},{maxy},{maxx},{miny}",
    }


def rank_osm_streets(
    official_street,
    osm_street_names,
    top_k,
):
    rows = []

    for osm_street in osm_street_names:
        scores = street_similarity(
            official_street,
            osm_street,
        )

        rows.append(
            {
                "osm_street_name":
                    osm_street,

                **scores,
            }
        )

    rows.sort(
        key=lambda row: (
            row[
                "street_score"
            ],
            row[
                "core_score"
            ],
            row[
                "base_score"
            ],
        ),
        reverse=True,
    )

    return rows[
        :top_k
    ]


def classify_street_match(
    candidates,
    auto_threshold,
    review_threshold,
    min_margin,
):
    if not candidates:
        return (
            "unresolved",
            None,
        )

    top_score = (
        candidates[0][
            "street_score"
        ]
    )

    second_score = (
        candidates[1][
            "street_score"
        ]
        if len(
            candidates
        ) > 1
        else 0.0
    )

    margin = (
        top_score
        - second_score
    )

    if (
        top_score >= auto_threshold
        and margin >= min_margin
    ):
        return (
            "matched_auto",
            margin,
        )

    if (
        top_score >= review_threshold
    ):
        return (
            "review",
            margin,
        )

    return (
        "unresolved",
        margin,
    )


def load_cache(path):
    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}


def save_cache(path, cache):
    path.write_text(
        json.dumps(
            cache,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def query_nominatim(
    session,
    matched_street,
    civic,
    context,
    pause_seconds,
):
    url = (
        "https://nominatim.openstreetmap.org/search"
    )

    query_street = (
        f"{matched_street} {civic}"
        if civic
        else matched_street
    )

    common = {
        "format":
            "jsonv2",

        "limit":
            5,

        "countrycodes":
            "it",

        "addressdetails":
            1,

        "bounded":
            1,

        "viewbox":
            context[
                "viewbox"
            ],
    }

    params = {
        **common,
        "street":
            query_street,

        "city":
            context[
                "municipality_name"
            ],
    }

    response = session.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    results = response.json()

    time.sleep(
        pause_seconds
    )

    if not results:
        params = {
            **common,
            "q":
                (
                    f"{query_street}, "
                    f"{context['municipality_name']}, Italia"
                ),
        }

        response = session.get(
            url,
            params=params,
            timeout=60,
        )

        response.raise_for_status()

        results = response.json()

        time.sleep(
            pause_seconds
        )

    return (
        query_street,
        results,
    )


def normalize_civic(value):
    value = normalize_text(
        value
    )

    return value.replace(
        " ",
        "",
    )


def score_geocoder_candidate(
    matched_street,
    civic,
    candidate,
    context,
):
    try:
        longitude = float(
            candidate[
                "lon"
            ]
        )

        latitude = float(
            candidate[
                "lat"
            ]
        )
    except Exception:
        return None

    display_name = (
        clean_text(
            candidate.get(
                "display_name"
            )
        )
        or ""
    )

    inside = bool(
        context[
            "geometry"
        ].covers(
            Point(
                longitude,
                latitude,
            )
        )
    )

    address = candidate.get(
        "address"
    )

    if not isinstance(
        address,
        dict,
    ):
        address = {}

    candidate_road = (
        address.get(
            "road"
        )
        or address.get(
            "pedestrian"
        )
        or address.get(
            "residential"
        )
        or address.get(
            "footway"
        )
    )

    road_score = float(
        fuzz.token_set_ratio(
            normalize_text(
                matched_street
            ),
            normalize_text(
                candidate_road
                or display_name
            ),
        )
    )

    candidate_house_number = (
        clean_text(
            address.get(
                "house_number"
            )
        )
    )

    civic_match = None

    if civic:
        civic_match = (
            normalize_civic(
                civic
            )
            == normalize_civic(
                candidate_house_number
            )
            if candidate_house_number
            else False
        )

    if (
        civic
        and civic_match
    ):
        resolution = (
            "address"
        )
    else:
        resolution = (
            "street"
        )

    return {
        "display_name":
            display_name,

        "longitude":
            longitude,

        "latitude":
            latitude,

        "inside_target":
            inside,

        "candidate_road":
            candidate_road,

        "candidate_house_number":
            candidate_house_number,

        "road_score":
            road_score,

        "civic_match":
            civic_match,

        "resolution":
            resolution,

        "osm_type":
            candidate.get(
                "osm_type"
            ),

        "osm_id":
            candidate.get(
                "osm_id"
            ),

        "place_id":
            candidate.get(
                "place_id"
            ),
    }


def main():
    args = parse_args()

    pending, buildings_path = (
        load_pending_buildings(
            args
        )
    )

    if pending.empty:
        print(
            "Nessun edificio unresolved da processare."
        )
        return

    (
        osm_street_names,
        edges_path,
    ) = load_osm_street_names(
        args
    )

    context = (
        load_municipality_context(
            pending,
            args,
        )
    )

    print(
        "\n===================================="
    )
    print(
        " NATIONAL OSM STREET FALLBACK"
    )
    print(
        "===================================="
    )

    print(
        f"Municipality: "
        f"{context['municipality_name']} "
        f"({args.municipality_code})"
    )

    print(
        f"Pending buildings: "
        f"{len(pending)}"
    )

    print(
        f"Unique named OSM streets: "
        f"{len(osm_street_names)}"
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
        / (
            "nominatim_osm_street_fallback_"
            f"{args.building_year}.json"
        )
    )

    cache = load_cache(
        cache_path
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent":
                (
                    "urban-digital-twin-thesis/1.0 "
                    "(academic research)"
                )
        }
    )

    summary_rows = []
    street_candidate_rows = []
    geocoder_candidate_rows = []

    query_results_memory = {}

    for _, building in (
        pending.sort_values(
            "building_code"
        )
        .iterrows()
    ):
        building_code = (
            building[
                "building_code"
            ]
        )

        official_address = (
            clean_text(
                building.get(
                    "official_building_address"
                )
            )
        )

        (
            parsed_street,
            parsed_civic,
        ) = split_official_address(
            official_address
        )

        street_candidates = (
            rank_osm_streets(
                official_street=(
                    parsed_street
                ),
                osm_street_names=(
                    osm_street_names
                ),
                top_k=(
                    args.top_k_streets
                ),
            )
        )

        (
            street_status,
            street_margin,
        ) = classify_street_match(
            candidates=(
                street_candidates
            ),
            auto_threshold=(
                args.auto_street_threshold
            ),
            review_threshold=(
                args.review_street_threshold
            ),
            min_margin=(
                args.min_margin
            ),
        )

        for rank, candidate in enumerate(
            street_candidates,
            start=1,
        ):
            street_candidate_rows.append(
                {
                    "building_code":
                        building_code,

                    "official_building_address":
                        official_address,

                    "parsed_street":
                        parsed_street,

                    "parsed_civic":
                        parsed_civic,

                    "candidate_rank":
                        rank,

                    **candidate,
                }
            )

        top_street = (
            street_candidates[0]
            if street_candidates
            else None
        )

        query_street = None
        geocoder_candidates = []
        geocoder_source = None

        # Only a high-confidence street reconciliation is allowed to
        # rewrite the query automatically.
        if (
            street_status
            == "matched_auto"
            and top_street
        ):
            matched_osm_street = (
                top_street[
                    "osm_street_name"
                ]
            )

            query_key = (
                f"{args.municipality_code}|"
                f"{normalize_text(matched_osm_street)}|"
                f"{normalize_civic(parsed_civic)}"
            )

            if (
                not args.refresh
                and query_key
                in query_results_memory
            ):
                query_street = (
                    query_results_memory[
                        query_key
                    ][
                        "query_street"
                    ]
                )

                raw_results = (
                    query_results_memory[
                        query_key
                    ][
                        "results"
                    ]
                )

                geocoder_source = (
                    "run_cache"
                )

            elif (
                not args.refresh
                and query_key in cache
            ):
                query_street = (
                    cache[
                        query_key
                    ][
                        "query_street"
                    ]
                )

                raw_results = (
                    cache[
                        query_key
                    ][
                        "results"
                    ]
                )

                geocoder_source = (
                    "disk_cache"
                )

                query_results_memory[
                    query_key
                ] = cache[
                    query_key
                ]

            else:
                (
                    query_street,
                    raw_results,
                ) = query_nominatim(
                    session=session,
                    matched_street=(
                        matched_osm_street
                    ),
                    civic=(
                        parsed_civic
                    ),
                    context=context,
                    pause_seconds=(
                        args.pause_seconds
                    ),
                )

                cache_entry = {
                    "query_street":
                        query_street,

                    "matched_osm_street":
                        matched_osm_street,

                    "parsed_civic":
                        parsed_civic,

                    "queried_at_utc":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),

                    "results":
                        raw_results,
                }

                cache[
                    query_key
                ] = cache_entry

                query_results_memory[
                    query_key
                ] = cache_entry

                save_cache(
                    cache_path,
                    cache,
                )

                geocoder_source = (
                    "nominatim"
                )

            for raw_candidate in (
                raw_results
            ):
                item = (
                    score_geocoder_candidate(
                        matched_street=(
                            matched_osm_street
                        ),
                        civic=(
                            parsed_civic
                        ),
                        candidate=(
                            raw_candidate
                        ),
                        context=context,
                    )
                )

                if item is not None:
                    geocoder_candidates.append(
                        item
                    )

            geocoder_candidates.sort(
                key=lambda row: (
                    row[
                        "inside_target"
                    ],
                    (
                        row[
                            "civic_match"
                        ]
                        is True
                    ),
                    row[
                        "road_score"
                    ],
                ),
                reverse=True,
            )

        for rank, candidate in enumerate(
            geocoder_candidates,
            start=1,
        ):
            geocoder_candidate_rows.append(
                {
                    "building_code":
                        building_code,

                    "query_street":
                        query_street,

                    "candidate_rank":
                        rank,

                    **candidate,
                }
            )

        top_geocoder = (
            geocoder_candidates[0]
            if geocoder_candidates
            else None
        )

        if (
            top_geocoder
            and top_geocoder[
                "inside_target"
            ]
            and top_geocoder[
                "road_score"
            ] >= 85.0
        ):
            if (
                parsed_civic
                and top_geocoder[
                    "civic_match"
                ] is True
            ):
                fallback_status = (
                    "address_candidate"
                )

                fallback_confidence = (
                    "high"
                )
            else:
                fallback_status = (
                    "street_anchor_candidate"
                )

                fallback_confidence = (
                    "medium"
                )
        elif top_geocoder:
            fallback_status = (
                "review"
            )

            fallback_confidence = (
                "low"
            )
        else:
            fallback_status = (
                "unresolved"
            )

            fallback_confidence = (
                "low"
            )

        summary_rows.append(
            {
                "building_code":
                    building_code,

                "official_building_address":
                    official_address,

                "parsed_street":
                    parsed_street,

                "parsed_civic":
                    parsed_civic,

                "street_match_status":
                    street_status,

                "matched_osm_street":
                    (
                        top_street[
                            "osm_street_name"
                        ]
                        if top_street
                        else None
                    ),

                "street_match_score":
                    (
                        top_street[
                            "street_score"
                        ]
                        if top_street
                        else None
                    ),

                "street_match_margin":
                    street_margin,

                "street_initials_compatible":
                    (
                        top_street[
                            "initials_compatible"
                        ]
                        if top_street
                        else None
                    ),

                "geocoder_query":
                    query_street,

                "geocoder_query_source":
                    geocoder_source,

                "fallback_status":
                    fallback_status,

                "fallback_confidence":
                    fallback_confidence,

                "candidate_longitude":
                    (
                        top_geocoder[
                            "longitude"
                        ]
                        if top_geocoder
                        else None
                    ),

                "candidate_latitude":
                    (
                        top_geocoder[
                            "latitude"
                        ]
                        if top_geocoder
                        else None
                    ),

                "candidate_display_name":
                    (
                        top_geocoder[
                            "display_name"
                        ]
                        if top_geocoder
                        else None
                    ),

                "candidate_road":
                    (
                        top_geocoder[
                            "candidate_road"
                        ]
                        if top_geocoder
                        else None
                    ),

                "candidate_house_number":
                    (
                        top_geocoder[
                            "candidate_house_number"
                        ]
                        if top_geocoder
                        else None
                    ),

                "candidate_road_score":
                    (
                        top_geocoder[
                            "road_score"
                        ]
                        if top_geocoder
                        else None
                    ),

                "candidate_civic_match":
                    (
                        top_geocoder[
                            "civic_match"
                        ]
                        if top_geocoder
                        else None
                    ),

                "candidate_inside_target":
                    (
                        top_geocoder[
                            "inside_target"
                        ]
                        if top_geocoder
                        else None
                    ),

                "candidate_resolution":
                    (
                        top_geocoder[
                            "resolution"
                        ]
                        if top_geocoder
                        else None
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    street_candidates_df = pd.DataFrame(
        street_candidate_rows
    )

    geocoder_candidates_df = pd.DataFrame(
        geocoder_candidate_rows
    )

    features_dir = (
        FEATURES_MIM_DIR
        / args.municipality_code
    )

    features_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        features_dir
        / (
            "school_osm_street_fallback_"
            f"{args.building_year}.csv"
        )
    )

    street_candidates_path = (
        features_dir
        / (
            "school_osm_street_fallback_candidates_"
            f"{args.building_year}.csv"
        )
    )

    geocoder_candidates_path = (
        features_dir
        / (
            "school_osm_street_geocoder_candidates_"
            f"{args.building_year}.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    street_candidates_df.to_csv(
        street_candidates_path,
        index=False,
        encoding="utf-8-sig",
    )

    geocoder_candidates_df.to_csv(
        geocoder_candidates_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== RESULTS ==="
    )

    print(
        summary[
            [
                "building_code",
                "official_building_address",
                "parsed_street",
                "parsed_civic",
                "street_match_status",
                "matched_osm_street",
                "street_match_score",
                "street_match_margin",
                "street_initials_compatible",
                "fallback_status",
                "candidate_display_name",
                "candidate_longitude",
                "candidate_latitude",
                "candidate_resolution",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nStreet match statuses:"
    )

    print(
        summary[
            "street_match_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nFallback statuses:"
    )

    print(
        summary[
            "fallback_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ {summary_path}"
    )

    print(
        f"✓ {street_candidates_path}"
    )

    print(
        f"✓ {geocoder_candidates_path}"
    )

    print(
        "\nNOTA METODOLOGICA:"
    )

    print(
        "Il fallback usa esclusivamente l'indirizzo MIM originale, "
        "i nomi stradali del grafo OSM del comune e Nominatim."
    )

    print(
        "Non contiene alias, civici o regole specifiche del comune target."
    )

    print(
        "Un risultato street_anchor_candidate non viene interpretato "
        "come coordinata esatta dell'edificio."
    )


if __name__ == "__main__":
    main()
