import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = ROOT / "data" / "features" / "mim"

DEFAULT_BUILDING_YEAR = "202425"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Review V2: raggruppa i pending solo tramite "
            "evidenza forte di indirizzo ufficiale MIM identico."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--building-year",
        default=DEFAULT_BUILDING_YEAR,
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


def normalize_address(value):
    value = clean_text(value)

    if value is None:
        return None

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    value = value.upper()

    value = re.sub(
        r"(?<=[A-Z])(?=\d)",
        " ",
        value,
    )

    value = re.sub(
        r"(?<=\d)(?=[A-Z])",
        " ",
        value,
    )

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value,
    )

    replacements = {
        "LOCALITA CONTRADA":
            "CONTRADA",

        "LOCALITA":
            "",

        "S N C":
            "",

        "SNC":
            "",

        "S N":
            "",
    }

    for old, new in replacements.items():
        value = value.replace(
            old,
            new,
        )

    # MIM uses terminal 0 in a few rows as a missing civic number.
    value = re.sub(
        r"\s+0$",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value or None


def parse_list_value(value):
    value = clean_text(value)

    if value is None:
        return []

    try:
        decoded = json.loads(
            value
        )

        if isinstance(
            decoded,
            list,
        ):
            return [
                str(item)
                for item in decoded
                if clean_text(item)
            ]
    except Exception:
        pass

    return [value]


def unique_join(values):
    result = []

    for value in values:
        for item in parse_list_value(
            value
        ):
            if item not in result:
                result.append(
                    item
                )

    return json.dumps(
        result,
        ensure_ascii=False,
    )


def load_input(
    municipality_code,
    building_year,
):
    path = (
        FEATURES_DIR
        / municipality_code
        / (
            "school_buildings_manual_review_compact_"
            f"{building_year}.csv"
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path}"
        )

    dataframe = pd.read_csv(
        path,
        dtype={
            "building_code":
                str,
        },
    )

    return (
        dataframe,
        path,
    )


def assign_strong_groups(
    dataframe,
):
    dataframe = dataframe.copy()

    dataframe[
        "normalized_official_address"
    ] = dataframe[
        "official_building_address"
    ].apply(
        normalize_address
    )

    counts = (
        dataframe[
            "normalized_official_address"
        ]
        .value_counts(
            dropna=False
        )
    )

    dataframe[
        "same_official_address_count"
    ] = dataframe[
        "normalized_official_address"
    ].map(
        counts
    ).fillna(
        0
    ).astype(
        int
    )

    group_ids = {}
    next_group = 1

    for address in (
        dataframe[
            "normalized_official_address"
        ]
        .dropna()
        .unique()
    ):
        count = int(
            counts.get(
                address,
                0,
            )
        )

        if count > 1:
            group_ids[
                address
            ] = (
                f"ADDR-{next_group:02d}"
            )
            next_group += 1

    dataframe[
        "strong_address_group"
    ] = dataframe[
        "normalized_official_address"
    ].map(
        group_ids
    )

    dataframe[
        "strong_campus_hint"
    ] = dataframe[
        "strong_address_group"
    ].notna()

    # Shared OSM candidate is retained only as diagnostic information.
    # It is NOT sufficient to create a campus/group.
    if (
        "osm_candidate_site_name"
        in dataframe.columns
    ):
        candidate_counts = (
            dataframe[
                "osm_candidate_site_name"
            ]
            .dropna()
            .value_counts()
        )

        dataframe[
            "shared_osm_candidate_count"
        ] = (
            dataframe[
                "osm_candidate_site_name"
            ]
            .map(
                candidate_counts
            )
            .fillna(0)
            .astype(int)
        )

    else:
        dataframe[
            "shared_osm_candidate_count"
        ] = 0

    return dataframe


def revised_proposal(row):
    current = clean_text(
        row.get(
            "proposal"
        )
    )

    group = clean_text(
        row.get(
            "strong_address_group"
        )
    )

    osm_status = clean_text(
        row.get(
            "osm_match_status"
        )
    )

    geocoder_status = clean_text(
        row.get(
            "geocoder_status"
        )
    )

    osm_address_score = row.get(
        "osm_address_score"
    )

    osm_match_score = row.get(
        "osm_match_score"
    )

    try:
        osm_address_score = float(
            osm_address_score
        )
    except Exception:
        osm_address_score = 0.0

    try:
        osm_match_score = float(
            osm_match_score
        )
    except Exception:
        osm_match_score = 0.0

    # Preserve genuinely strong OSM proposals.
    if current == "accept_osm_candidate":
        return (
            "accept_osm_candidate",
            "high",
            (
                "Candidato OSM supportato da evidenza forte; "
                "review manuale richiesta prima dell'applicazione."
            ),
        )

    # If the row is in an exact official-address group,
    # the correct next action is site-level verification.
    if group:
        return (
            "verify_address_group",
            "medium_high",
            (
                "Più CodiceEdificio condividono lo stesso "
                "indirizzo ufficiale MIM; verificare come "
                "un unico caso territoriale/campus."
            ),
        )

    # The Meucci-like case: geocoder and OSM were already
    # considered spatially coherent by the previous review.
    if current in {
        "compare_sources_prefer_osm",
        "compare_sources_prefer_geocoder",
    }:
        return (
            current,
            "medium",
            (
                "Entrambe le fonti sono plausibili; "
                "verificare manualmente quale rappresenta "
                "il sito scolastico."
            ),
        )

    # A review OSM candidate with very strong address can remain
    # an individual verification case.
    if (
        osm_status == "review"
        and osm_address_score >= 90
        and osm_match_score >= 60
    ):
        return (
            "verify_osm_candidate",
            "medium",
            (
                "Candidato OSM con indirizzo forte, ma evidenza "
                "complessiva insufficiente per accettazione automatica."
            ),
        )

    # Do not promote generic repeated OSM candidates to campuses.
    return (
        "external_verification_required",
        "low",
        (
            "Nessuna evidenza interna sufficientemente forte; "
            "necessaria verifica tramite fonte esterna/ufficiale."
        ),
    )


def build_record_level(
    dataframe,
):
    dataframe = assign_strong_groups(
        dataframe
    )

    proposals = dataframe.apply(
        revised_proposal,
        axis=1,
        result_type="expand",
    )

    proposals.columns = [
        "revised_proposal",
        "revised_confidence",
        "revised_reason",
    ]

    dataframe = pd.concat(
        [
            dataframe,
            proposals,
        ],
        axis=1,
    )

    return dataframe


def build_site_cases(
    dataframe,
):
    rows = []

    used_indices = set()

    # First create one review case for every exact-address group.
    for (
        group_id,
        group,
    ) in dataframe[
        dataframe[
            "strong_address_group"
        ].notna()
    ].groupby(
        "strong_address_group"
    ):
        used_indices.update(
            group.index.tolist()
        )

        rows.append(
            {
                "review_case_id":
                    group_id,

                "case_type":
                    "exact_official_address_group",

                "building_count":
                    len(group),

                "building_codes":
                    json.dumps(
                        group[
                            "building_code"
                        ]
                        .astype(str)
                        .tolist(),
                        ensure_ascii=False,
                    ),

                "official_address":
                    group[
                        "official_building_address"
                    ]
                    .iloc[0],

                "linked_school_names":
                    unique_join(
                        group[
                            "linked_school_names"
                        ].tolist()
                    ),

                "geocoder_statuses":
                    json.dumps(
                        sorted(
                            set(
                                group[
                                    "geocoder_status"
                                ]
                                .dropna()
                                .astype(str)
                                .tolist()
                            )
                        ),
                        ensure_ascii=False,
                    ),

                "osm_candidate_names":
                    json.dumps(
                        sorted(
                            set(
                                group[
                                    "osm_candidate_site_name"
                                ]
                                .dropna()
                                .astype(str)
                                .tolist()
                            )
                        ),
                        ensure_ascii=False,
                    ),

                "recommended_action":
                    (
                        "verify_group_as_single_school_site"
                    ),

                "manual_decision":
                    "",

                "manual_source":
                    "",

                "manual_longitude":
                    "",

                "manual_latitude":
                    "",

                "manual_site_id":
                    "",

                "manual_notes":
                    "",

                "review_completed":
                    False,
            }
        )

    # Then retain every singleton as an independent territorial case.
    singleton_counter = 1

    for index, row in dataframe.iterrows():
        if index in used_indices:
            continue

        rows.append(
            {
                "review_case_id":
                    f"SINGLE-{singleton_counter:02d}",

                "case_type":
                    "single_building",

                "building_count":
                    1,

                "building_codes":
                    json.dumps(
                        [
                            str(
                                row[
                                    "building_code"
                                ]
                            )
                        ],
                        ensure_ascii=False,
                    ),

                "official_address":
                    row[
                        "official_building_address"
                    ],

                "linked_school_names":
                    row.get(
                        "linked_school_names"
                    ),

                "geocoder_statuses":
                    json.dumps(
                        [
                            row[
                                "geocoder_status"
                            ]
                        ]
                        if clean_text(
                            row.get(
                                "geocoder_status"
                            )
                        )
                        else [],
                        ensure_ascii=False,
                    ),

                "osm_candidate_names":
                    json.dumps(
                        [
                            row[
                                "osm_candidate_site_name"
                            ]
                        ]
                        if clean_text(
                            row.get(
                                "osm_candidate_site_name"
                            )
                        )
                        else [],
                        ensure_ascii=False,
                    ),

                "recommended_action":
                    row[
                        "revised_proposal"
                    ],

                "manual_decision":
                    "",

                "manual_source":
                    "",

                "manual_longitude":
                    "",

                "manual_latitude":
                    "",

                "manual_site_id":
                    "",

                "manual_notes":
                    "",

                "review_completed":
                    False,
            }
        )

        singleton_counter += 1

    return pd.DataFrame(
        rows
    )


def save_outputs(
    records,
    cases,
    municipality_code,
    building_year,
):
    directory = (
        FEATURES_DIR
        / municipality_code
    )

    record_path = (
        directory
        / (
            "school_buildings_manual_review_v2_"
            f"{building_year}.csv"
        )
    )

    cases_path = (
        directory
        / (
            "school_site_review_cases_"
            f"{building_year}.csv"
        )
    )

    records.to_csv(
        record_path,
        index=False,
        encoding="utf-8-sig",
    )

    cases.to_csv(
        cases_path,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        record_path,
        cases_path,
    )


def print_summary(
    records,
    cases,
    input_path,
    record_path,
    cases_path,
):
    print(
        "\n===================================="
    )
    print(
        " SCHOOL BUILDING MANUAL REVIEW V2"
    )
    print(
        "===================================="
    )

    print(
        f"Input pending buildings: {len(records)}"
    )

    strong_groups = (
        records[
            "strong_address_group"
        ]
        .dropna()
        .nunique()
    )

    grouped_buildings = int(
        records[
            "strong_campus_hint"
        ]
        .sum()
    )

    print(
        "\nStrong exact-address evidence:"
    )
    print(
        f"  groups: {strong_groups}"
    )
    print(
        f"  buildings in groups: {grouped_buildings}"
    )

    print(
        "\nRevised proposals:"
    )
    print(
        records[
            "revised_proposal"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nTerritorial review cases after grouping:"
    )
    print(
        f"  {len(cases)}"
    )

    print(
        "\nCase types:"
    )
    print(
        cases[
            "case_type"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== OUTPUT ==="
    )
    print(
        f"✓ Record-level V2: {record_path}"
    )
    print(
        f"✓ Site review cases: {cases_path}"
    )

    print(
        "\nNOTE:"
    )
    print(
        "Shared OSM candidates are diagnostic only and no longer "
        "create campus groups."
    )
    print(
        "No final coordinates are modified."
    )


def main():
    args = parse_args()

    (
        dataframe,
        input_path,
    ) = load_input(
        args.municipality_code,
        args.building_year,
    )

    records = build_record_level(
        dataframe
    )

    cases = build_site_cases(
        records
    )

    (
        record_path,
        cases_path,
    ) = save_outputs(
        records,
        cases,
        args.municipality_code,
        args.building_year,
    )

    print_summary(
        records,
        cases,
        input_path,
        record_path,
        cases_path,
    )


if __name__ == "__main__":
    main()
