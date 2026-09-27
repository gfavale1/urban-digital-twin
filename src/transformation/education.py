"""
Canonical Education transformation module.

Responsibilities:
- building-location evidence fusion;
- classification of MIM records without an exact building relation;
- generic OSM street fallback for unresolved buildings;
- residual automatic resolution of school services without building relation;
- construction and QA of the final canonical Education service layer.

The module preserves the validated pilot/transfer algorithms while exposing
one stable transformation entry point for the national pipeline.
"""

from __future__ import annotations

import argparse as _cli_argparse
import sys as _sys
from pathlib import Path as _CliPath

_EDUCATION_ROOT = _CliPath(__file__).resolve().parents[2]
_EDUCATION_SRC = _EDUCATION_ROOT / "src"

if str(_EDUCATION_SRC) not in _sys.path:
    _sys.path.insert(0, str(_EDUCATION_SRC))

from core.config import DEFAULT_CONFIG as _DEFAULT_CONFIG


# ============================================================================
# BUILDING LOCATION FUSION
# ============================================================================

import argparse
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


building_fusion_ROOT = Path(__file__).resolve().parents[2]
building_fusion_PROCESSED_MIM_DIR = building_fusion_ROOT / "data" / "processed" / "mim"

building_fusion_DEFAULT_BUILDING_YEAR = "202425"

building_fusion_GEOCODER_ADDRESS_STRONG = 85.0
building_fusion_GEOCODER_ADDRESS_REVIEW = 70.0
building_fusion_STRONG_AGREEMENT_M = 120.0
building_fusion_PLAUSIBLE_AGREEMENT_M = 250.0


def building_fusion_parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fusion V2 delle localizzazioni degli edifici scolastici "
            "con validazione semantica dell'indirizzo Nominatim."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre.",
    )

    parser.add_argument(
        "--building-year",
        default=building_fusion_DEFAULT_BUILDING_YEAR,
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


def building_fusion_is_missing(value):
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


def building_fusion_clean_text(value):
    if building_fusion_is_missing(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def building_fusion_normalize_address(value):
    value = building_fusion_clean_text(value)

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

    replacements = {
        "LOCALITA CONTRADA": "CONTRADA",
        "LOCALITA": "",
        "P ZZA": "PIAZZA",
        "P ZA": "PIAZZA",
        "V LE": "VIALE",
        "C DA": "CONTRADA",
        "S N C": "",
        "SNC": "",
        "S N": "",
    }

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

    for old, new in replacements.items():
        value = value.replace(
            old,
            new,
        )

    # In several MIM records, terminal 0 means missing house number.
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


def building_fusion_address_similarity(
    official,
    returned,
):
    left = building_fusion_normalize_address(
        official
    )

    right = building_fusion_normalize_address(
        returned
    )

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


def building_fusion_as_float(value):
    if building_fusion_is_missing(value):
        return None

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(value):
        return None

    return value


def building_fusion_haversine_m(
    lon1,
    lat1,
    lon2,
    lat2,
):
    values = [
        building_fusion_as_float(lon1),
        building_fusion_as_float(lat1),
        building_fusion_as_float(lon2),
        building_fusion_as_float(lat2),
    ]

    if any(
        value is None
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

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2.0) ** 2
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


def building_fusion_load_inputs(
    municipality_code,
    building_year,
):
    directory = (
        building_fusion_PROCESSED_MIM_DIR
        / municipality_code
    )

    geocoder_path = (
        directory
        / (
            "physical_school_buildings_"
            f"{building_year}_geocoded.parquet"
        )
    )

    osm_path = (
        directory
        / (
            "school_building_osm_matches_"
            f"{building_year}_v2.parquet"
        )
    )

    for path in [
        geocoder_path,
        osm_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"File richiesto non trovato: {path}"
            )

    geocoder = pd.read_parquet(
        geocoder_path
    )

    osm = pd.read_parquet(
        osm_path
    )

    return (
        geocoder,
        osm,
    )


def building_fusion_prepare_dataset(
    geocoder,
    osm,
):
    geocoder = geocoder.copy()

    geocoder = geocoder.rename(
        columns={
            "geocoding_status":
                "geocoder_status",

            "latitude":
                "geocoder_latitude",

            "longitude":
                "geocoder_longitude",

            "display_name":
                "geocoder_display_name",

            "result_class":
                "geocoder_result_class",

            "result_type":
                "geocoder_result_type",

            "result_address":
                "geocoder_result_address",

            "result_postcode":
                "geocoder_result_postcode",

            "postcode_match":
                "geocoder_postcode_match",

            "inside_municipality":
                "geocoder_inside_municipality",

            "distance_to_municipality_m":
                "geocoder_distance_to_municipality_m",
        }
    )

    geocoder[
        "geocoder_address_score"
    ] = geocoder.apply(
        lambda row:
            building_fusion_address_similarity(
                row.get(
                    "official_building_address"
                ),
                row.get(
                    "geocoder_result_address"
                ),
            ),
        axis=1,
    )

    osm_keep = [
        column
        for column in [
            "building_code",
            "osm_match_status",
            "status_reason",
            "candidate_site_id",
            "candidate_site_name",
            "candidate_site_address",
            "candidate_longitude",
            "candidate_latitude",
            "match_score",
            "name_score",
            "address_score",
            "score_margin",
            "linked_school_codes",
            "linked_school_names",
        ]
        if column in osm.columns
    ]

    osm = (
        osm[
            osm_keep
        ]
        .copy()
        .rename(
            columns={
                "status_reason":
                    "osm_status_reason",

                "candidate_site_id":
                    "osm_candidate_site_id",

                "candidate_site_name":
                    "osm_candidate_site_name",

                "candidate_site_address":
                    "osm_candidate_site_address",

                "candidate_longitude":
                    "osm_candidate_longitude",

                "candidate_latitude":
                    "osm_candidate_latitude",

                "match_score":
                    "osm_match_score",

                "name_score":
                    "osm_name_score",

                "address_score":
                    "osm_address_score",

                "score_margin":
                    "osm_score_margin",
            }
        )
    )

    return geocoder.merge(
        osm,
        on="building_code",
        how="left",
        validate="one_to_one",
    )


def building_fusion_fuse_row(
    row,
    target_municipality_name,
):
    municipality = (
        building_fusion_clean_text(
            row.get(
                "building_municipality_name"
            )
        )
        or ""
    ).upper()

    if municipality != target_municipality_name:
        return {
            "final_location_status":
                "outside_target_municipality",

            "final_geometry_source":
                None,

            "final_longitude":
                None,

            "final_latitude":
                None,

            "location_confidence":
                "excluded",

            "geocoder_address_quality":
                "not_applicable",

            "osm_geocoder_distance_final_m":
                None,

            "final_reason":
                "Edificio ufficialmente esterno al comune target.",
        }

    geocoder_status = row.get(
        "geocoder_status"
    )

    geocoder_address_score = (
        building_fusion_as_float(
            row.get(
                "geocoder_address_score"
            )
        )
        or 0.0
    )

    geocoder_lon = building_fusion_as_float(
        row.get(
            "geocoder_longitude"
        )
    )

    geocoder_lat = building_fusion_as_float(
        row.get(
            "geocoder_latitude"
        )
    )

    osm_status = row.get(
        "osm_match_status"
    )

    osm_lon = building_fusion_as_float(
        row.get(
            "osm_candidate_longitude"
        )
    )

    osm_lat = building_fusion_as_float(
        row.get(
            "osm_candidate_latitude"
        )
    )

    distance_m = building_fusion_haversine_m(
        geocoder_lon,
        geocoder_lat,
        osm_lon,
        osm_lat,
    )

    osm_strong = (
        osm_status
        == "matched_auto"
        and osm_lon is not None
        and osm_lat is not None
    )

    geocoder_has_point = (
        geocoder_status
        == "accepted_candidate"
        and geocoder_lon is not None
        and geocoder_lat is not None
    )

    geocoder_strong = (
        geocoder_has_point
        and geocoder_address_score
        >= building_fusion_GEOCODER_ADDRESS_STRONG
    )

    geocoder_review = (
        geocoder_has_point
        and geocoder_address_score
        >= building_fusion_GEOCODER_ADDRESS_REVIEW
        and not geocoder_strong
    )

    if geocoder_strong:
        address_quality = "strong"
    elif geocoder_review:
        address_quality = "review"
    elif geocoder_has_point:
        address_quality = "weak"
    else:
        address_quality = "unavailable"

    # OSM automatic match is already based on official building address
    # plus linked-school evidence, so it remains the strongest source.
    if osm_strong:
        if (
            geocoder_strong
            and distance_m is not None
            and distance_m
            <= building_fusion_STRONG_AGREEMENT_M
        ):
            reason = (
                "Match OSM forte e geocoder con indirizzo coerente; "
                "le due fonti concordano entro 120 m."
            )
            confidence = "high"
            agreement = "strong"

        elif (
            geocoder_strong
            and distance_m is not None
            and distance_m
            <= building_fusion_PLAUSIBLE_AGREEMENT_M
        ):
            reason = (
                "Match OSM forte e geocoder con indirizzo coerente; "
                "concordanza spaziale entro 250 m."
            )
            confidence = "high"
            agreement = "plausible"

        else:
            reason = (
                "Match edificio-OSM automatico forte; "
                "il geocoder non aggiunge una conferma affidabile."
            )
            confidence = "high"
            agreement = "osm_only"

        return {
            "final_location_status":
                "validated",

            "final_geometry_source":
                "OSM",

            "final_longitude":
                osm_lon,

            "final_latitude":
                osm_lat,

            "location_confidence":
                confidence,

            "source_agreement":
                agreement,

            "geocoder_address_quality":
                address_quality,

            "osm_geocoder_distance_final_m":
                distance_m,

            "final_reason":
                reason,
        }

    # Geocoder can be used only if the returned address itself
    # resembles the official MIM building address.
    if geocoder_strong:
        if (
            osm_status == "review"
            and distance_m is not None
            and distance_m
            <= building_fusion_PLAUSIBLE_AGREEMENT_M
        ):
            return {
                "final_location_status":
                    "validated",

                "final_geometry_source":
                    "Nominatim",

                "final_longitude":
                    geocoder_lon,

                "final_latitude":
                    geocoder_lat,

                "location_confidence":
                    "medium_high",

                "source_agreement":
                    "osm_review_support",

                "geocoder_address_quality":
                    address_quality,

                "osm_geocoder_distance_final_m":
                    distance_m,

                "final_reason":
                    (
                        "Geocoder con indirizzo forte e candidato OSM "
                        "coerente entro 250 m."
                    ),
            }

        return {
            "final_location_status":
                "accepted_address",

            "final_geometry_source":
                "Nominatim",

            "final_longitude":
                geocoder_lon,

            "final_latitude":
                geocoder_lat,

            "location_confidence":
                "medium",

            "source_agreement":
                "geocoder_only",

            "geocoder_address_quality":
                address_quality,

            "osm_geocoder_distance_final_m":
                distance_m,

            "final_reason":
                (
                    "Geocoder nel comune target con indirizzo restituito "
                    "coerente con l'indirizzo ufficiale MIM."
                ),
        }

    if geocoder_review:
        return {
            "final_location_status":
                "review",

            "final_geometry_source":
                None,

            "final_longitude":
                None,

            "final_latitude":
                None,

            "location_confidence":
                "review",

            "source_agreement":
                "insufficient",

            "geocoder_address_quality":
                address_quality,

            "osm_geocoder_distance_final_m":
                distance_m,

            "final_reason":
                (
                    "Geocoder disponibile ma somiglianza dell'indirizzo "
                    "solo intermedia; richiede verifica."
                ),
        }

    if osm_status == "review":
        return {
            "final_location_status":
                "review",

            "final_geometry_source":
                None,

            "final_longitude":
                None,

            "final_latitude":
                None,

            "location_confidence":
                "review",

            "source_agreement":
                "insufficient",

            "geocoder_address_quality":
                address_quality,

            "osm_geocoder_distance_final_m":
                distance_m,

            "final_reason":
                (
                    "Candidato OSM plausibile ma non automatico "
                    "e geocoder non validato semanticamente."
                ),
        }

    return {
        "final_location_status":
            "unresolved",

        "final_geometry_source":
            None,

        "final_longitude":
            None,

        "final_latitude":
            None,

        "location_confidence":
            "unresolved",

        "source_agreement":
            "none",

        "geocoder_address_quality":
            address_quality,

        "osm_geocoder_distance_final_m":
            distance_m,

        "final_reason":
            (
                "Nessuna fonte fornisce una localizzazione "
                "sufficientemente affidabile."
            ),
    }


def building_fusion_build_final_dataset(
    geocoder,
    osm,
):
    # Determine the target municipality from the canonical
    # physical-building dataset. No municipality-specific
    # hard-coded values are allowed.
    if "building_municipality_name" not in geocoder.columns:
        raise RuntimeError(
            "building_municipality_name mancante nel dataset geocoder."
        )

    target_municipalities = {
        (
            building_fusion_clean_text(value)
            or ""
        ).upper()
        for value in geocoder[
            "building_municipality_name"
        ].dropna()
        if building_fusion_clean_text(value)
    }

    if len(target_municipalities) != 1:
        raise RuntimeError(
            "Impossibile determinare univocamente il comune target: "
            + ", ".join(
                sorted(target_municipalities)
            )
        )

    target_municipality_name = next(
        iter(target_municipalities)
    )

    merged = building_fusion_prepare_dataset(
        geocoder,
        osm,
    )

    fusion_rows = []

    for _, row in merged.iterrows():
        fusion_rows.append(
            building_fusion_fuse_row(
                row.to_dict(),
                target_municipality_name,
            )
        )

    return pd.concat(
        [
            merged.reset_index(
                drop=True
            ),
            pd.DataFrame(
                fusion_rows
            ).reset_index(
                drop=True
            ),
        ],
        axis=1,
    )


def building_fusion_save_output(
    dataframe,
    municipality_code,
    building_year,
):
    directory = (
        building_fusion_PROCESSED_MIM_DIR
        / municipality_code
    )

    path = (
        directory
        / (
            "physical_school_buildings_"
            f"{building_year}_final_v2.parquet"
        )
    )

    dataframe.to_parquet(
        path,
        index=False,
    )

    print(
        "\n=== OUTPUT ==="
    )
    print(
        f"✓ {path}"
    )


def building_fusion_print_summary(
    dataframe,
):
    print(
        "\n===================================="
    )
    print(
        " SCHOOL BUILDING LOCATION FUSION V2 COMPLETATA"
    )
    print(
        "===================================="
    )

    print(
        f"Edifici totali: {len(dataframe)}"
    )

    print(
        "\nFinal location status:"
    )
    print(
        dataframe[
            "final_location_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    target = (
        dataframe[
            "final_location_status"
        ]
        != "outside_target_municipality"
    )

    usable = (
        dataframe[
            "final_location_status"
        ]
        .isin(
            [
                "validated",
                "accepted_address",
            ]
        )
    )

    target_count = int(
        target.sum()
    )

    usable_count = int(
        usable.sum()
    )

    validated_count = int(
        (
            dataframe[
                "final_location_status"
            ]
            == "validated"
        )
        .sum()
    )

    coverage = (
        100.0
        * usable_count
        / target_count
        if target_count
        else 0.0
    )

    print(
        "\nEdifici nel comune target:"
    )
    print(
        f"  validati con evidenza forte: {validated_count}"
    )
    print(
        f"  coordinate utilizzabili totali: {usable_count}/{target_count}"
    )
    print(
        f"  coverage utilizzabile: {coverage:.2f}%"
    )

    print(
        "\nQualità indirizzo geocoder:"
    )
    print(
        dataframe[
            "geocoder_address_quality"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n=== COORDINATE UTILIZZABILI ==="
    )

    usable_rows = (
        dataframe[
            usable
        ]
    )

    if usable_rows.empty:
        print(
            "Nessuna."
        )
    else:
        print(
            usable_rows[
                [
                    "building_code",
                    "official_building_address",
                    "geocoder_result_address",
                    "geocoder_address_score",
                    "osm_match_status",
                    "final_location_status",
                    "final_geometry_source",
                    "location_confidence",
                    "final_longitude",
                    "final_latitude",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "\n=== DA REVISIONARE / NON RISOLTI ==="
    )

    pending = (
        dataframe[
            target
            & ~usable
        ]
    )

    if pending.empty:
        print(
            "Nessuno."
        )
    else:
        print(
            pending[
                [
                    "building_code",
                    "official_building_address",
                    "geocoder_status",
                    "geocoder_result_address",
                    "geocoder_address_score",
                    "osm_match_status",
                    "osm_candidate_site_name",
                    "final_location_status",
                ]
            ]
            .to_string(
                index=False
            )
        )


def building_fusion_main():
    args = building_fusion_parse_args()

    print(
        "\n===================================="
    )
    print(
        " MIM SCHOOL BUILDING LOCATION FUSION V2"
    )
    print(
        "===================================="
    )

    (
        geocoder,
        osm,
    ) = building_fusion_load_inputs(
        args.municipality_code,
        args.building_year,
    )

    final = building_fusion_build_final_dataset(
        geocoder=geocoder,
        osm=osm,
    )

    building_fusion_save_output(
        final,
        args.municipality_code,
        args.building_year,
    )

    building_fusion_print_summary(
        final
    )


# ============================================================================
# UNMATCHED MIM SCHOOL CLASSIFICATION
# ============================================================================

import argparse
import re
import unicodedata
from pathlib import Path
import pandas as pd
classification_ROOT = Path(__file__).resolve().parents[2]
classification_PROCESSED_MIM_DIR = classification_ROOT / 'data' / 'processed' / 'mim'
classification_FEATURES_MIM_DIR = classification_ROOT / 'data' / 'features' / 'mim'
classification_MISSING_VALUES = {'', 'NAN', 'NONE', 'NULL', 'N/A', 'NA', 'N.D.', 'ND', 'NON DISPONIBILE', 'NON DISP.', '-'}

def classification_parse_args():
    parser = argparse.ArgumentParser(description="Classificazione nazionale dei record MIM statali che non trovano un match esatto nell'Anagrafe dell'edilizia scolastica.")
    parser.add_argument('--municipality-code', required=True)
    parser.add_argument('--school-year', default='202425')
    parser.add_argument('--building-year', default='202425')
    args = parser.parse_args()
    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    return args

def classification_normalize_text(value):
    if value is None or pd.isna(value):
        return ''
    value = str(value).strip()
    if value.upper() in classification_MISSING_VALUES:
        return ''
    value = unicodedata.normalize('NFKD', value)
    value = ''.join((char for char in value if not unicodedata.combining(char)))
    value = value.upper()
    value = re.sub('[^A-Z0-9]+', ' ', value)
    return re.sub('\\s+', ' ', value).strip()

def classification_contains_any(text, patterns):
    return any((pattern in text for pattern in patterns))

def classification_classify(row):
    name = classification_normalize_text(row.get('school_name'))
    characteristics = classification_normalize_text(row.get('DESCRIZIONECARATTERISTICASCUOLA'))
    grade = classification_normalize_text(row.get('grade_description'))
    address = classification_normalize_text(row.get('school_address'))
    combined = ' | '.join([name, characteristics, grade])
    if classification_contains_any(combined, ['CARCERAR', 'CASA CIRCONDARIALE', 'ISTITUTO PENITENZIARIO']):
        return {'unmatched_class': 'restricted_correctional_service', 'creates_public_school_site': False, 'needs_geolocation': False, 'site_handling': 'exclude_from_general_school_accessibility', 'classification_reason': 'Servizio scolastico rivolto a utenza carceraria; non rappresenta un punto di accesso scolastico ordinario.'}
    if classification_contains_any(combined, ['OSPEDALIER', 'C O IST OSPEDALIERO', 'SCUOLA OSPEDALIERA']):
        return {'unmatched_class': 'restricted_hospital_service', 'creates_public_school_site': False, 'needs_geolocation': False, 'site_handling': 'exclude_from_general_school_accessibility', 'classification_reason': 'Servizio scolastico ospedaliero; non rappresenta un ordinario sito scolastico accessibile alla popolazione generale.'}
    if classification_contains_any(combined, ['CORSO SERALE', 'PERCORSO II LIVELLO', 'SECONDO LIVELLO']):
        return {'unmatched_class': 'non_separate_evening_course', 'creates_public_school_site': False, 'needs_geolocation': False, 'site_handling': 'link_to_existing_school_site_if_needed', 'classification_reason': "Percorso/corso serale: è un'offerta didattica associata a una sede esistente, non un nuovo sito fisico."}
    return {'unmatched_class': 'physical_school_candidate', 'creates_public_school_site': True, 'needs_geolocation': True, 'site_handling': 'automatic_geolocation_pipeline', 'classification_reason': 'Record scolastico ordinario non riconducibile a servizio speciale o percorso non separato; candidato a vero sito fisico.'}

def classification_main():
    args = classification_parse_args()
    processed_dir = classification_PROCESSED_MIM_DIR / args.municipality_code
    features_dir = classification_FEATURES_MIM_DIR / args.municipality_code
    features_dir.mkdir(parents=True, exist_ok=True)
    input_path = processed_dir / f'school_building_unmatched_{args.school_year}_from_{args.building_year}.parquet'
    if not input_path.exists():
        raise FileNotFoundError(f'File unmatched non trovato: {input_path}')
    df = pd.read_parquet(input_path)
    classifications = df.apply(classification_classify, axis=1, result_type='expand')
    result = pd.concat([df.reset_index(drop=True), classifications.reset_index(drop=True)], axis=1)
    output_csv = features_dir / f'school_unmatched_classification_{args.school_year}.csv'
    output_parquet = processed_dir / f'school_unmatched_classification_{args.school_year}.parquet'
    result.to_csv(output_csv, index=False, encoding='utf-8-sig')
    result.to_parquet(output_parquet, index=False)
    print('\n====================================')
    print(' MIM UNMATCHED CLASSIFICATION')
    print('====================================')
    print(f'Rows: {len(result)}')
    print('\nClasses:')
    print(result['unmatched_class'].value_counts().to_string())
    print('\n=== RESULTS ===')
    print(result[['school_code', 'school_name', 'school_address', 'DESCRIZIONECARATTERISTICASCUOLA', 'unmatched_class', 'creates_public_school_site', 'needs_geolocation', 'site_handling']].sort_values(['unmatched_class', 'school_code']).to_string(index=False))
    print('\nPhysical school candidates requiring geolocation:')
    physical = result[result['needs_geolocation'] == True]
    print(f'{len(physical)}')
    if not physical.empty:
        print(physical[['school_code', 'school_name', 'school_address', 'institute_reference_code']].to_string(index=False))
    print('\n=== OUTPUT ===')
    print(f'✓ {output_csv}')
    print(f'✓ {output_parquet}')
    print('\nNOTA:')
    print('Le regole dipendono esclusivamente da attributi MIM espliciti e non contengono codici o nomi specifici del comune target.')

# ============================================================================
# OSM STREET FALLBACK
# ============================================================================

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
street_fallback_ROOT = Path(__file__).resolve().parents[2]
street_fallback_PROCESSED_MIM_DIR = street_fallback_ROOT / 'data' / 'processed' / 'mim'
street_fallback_PROCESSED_OSM_DIR = street_fallback_ROOT / 'data' / 'processed' / 'osm'
street_fallback_PROCESSED_ISTAT_DIR = street_fallback_ROOT / 'data' / 'processed' / 'istat'
street_fallback_FEATURES_MIM_DIR = street_fallback_ROOT / 'data' / 'features' / 'mim'
street_fallback_RAW_MIM_DIR = street_fallback_ROOT / 'data' / 'raw' / 'mim'
street_fallback_ROAD_PREFIXES = {'VIA', 'VIALE', 'PIAZZA', 'PIAZZALE', 'CORSO', 'LARGO', 'VICO', 'VICOLO', 'CONTRADA', 'LOCALITA', 'LOCALITÀ', 'STRADA', 'TRAVERSA', 'SALITA', 'DISCESA', 'ROTONDA', 'LUNGOMARE'}

def street_fallback_parse_args():
    parser = argparse.ArgumentParser(description='Fallback nazionale per edifici scolastici non localizzati: estrae i nomi stradali dal grafo OSM del comune, effettua street matching fuzzy e usa il nome OSM selezionato per una nuova query Nominatim. Nessun alias comunale è hard-coded.')
    parser.add_argument('--municipality-code', required=True, help='Codice ISTAT comunale a 6 cifre.')
    parser.add_argument('--school-year', default='202425')
    parser.add_argument('--building-year', default='202425')
    parser.add_argument('--top-k-streets', type=int, default=5)
    parser.add_argument('--auto-street-threshold', type=float, default=90.0)
    parser.add_argument('--review-street-threshold', type=float, default=75.0)
    parser.add_argument('--min-margin', type=float, default=8.0)
    parser.add_argument('--pause-seconds', type=float, default=1.1)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    if not args.municipality_code.isdigit() or len(args.municipality_code) != 6:
        raise ValueError('municipality-code deve avere esattamente 6 cifre.')
    return args

def street_fallback_clean_text(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    value = str(value).strip()
    return value or None

def street_fallback_strip_accents(value):
    value = unicodedata.normalize('NFKD', value)
    return ''.join((char for char in value if not unicodedata.combining(char)))

def street_fallback_normalize_text(value):
    value = street_fallback_clean_text(value)
    if value is None:
        return ''
    value = street_fallback_strip_accents(value)
    value = value.upper().replace('`', "'").replace('’', "'")
    value = re.sub("[^A-Z0-9']+", ' ', value)
    value = re.sub('\\s+', ' ', value).strip()
    return value

def street_fallback_split_official_address(value):
    """
    Split a MIM address into street text and civic number without
    introducing any municipality-specific rewrite.

    Examples:
        Via Mario Rosario Greco 12 -> (Via Mario Rosario Greco, 12)
        Via Petrarca snc            -> (Via Petrarca, None)
        Via Lucana 190/192          -> (Via Lucana, 190/192)
    """
    value = street_fallback_clean_text(value)
    if value is None:
        return (None, None)
    value = value.replace('`', "'").replace('’', "'")
    value = re.sub('\\s+\\bS\\s*\\.?\\s*N\\s*\\.?\\s*C\\s*\\.?\\s*$', '', value, flags=re.IGNORECASE)
    value = re.sub('\\s+\\bS\\s*\\.?\\s*N\\s*\\.?\\s*$', '', value, flags=re.IGNORECASE)
    value = re.sub('\\s+SNC\\s*$', '', value, flags=re.IGNORECASE)
    value = value.strip()
    civic_pattern = '(?:^|\\s)(\\d+(?:[A-Za-z])?(?:[/-]\\d+(?:[A-Za-z])?)*)\\s*$'
    match = re.search(civic_pattern, value)
    if match:
        civic = match.group(1)
        street = value[:match.start(1)].strip(' ,')
    else:
        civic = None
        street = value
    return (street_fallback_clean_text(street), street_fallback_clean_text(civic))

def street_fallback_road_core_tokens(value):
    normalized = street_fallback_normalize_text(value)
    tokens = normalized.split()
    while tokens and tokens[0] in street_fallback_ROAD_PREFIXES:
        tokens = tokens[1:]
    return tokens

def street_fallback_initials_signature(tokens):
    if not tokens:
        return ''
    if len(tokens) == 1:
        token = tokens[0]
        if token.isalpha() and 1 <= len(token) <= 4:
            return token
        return token[:1]
    return ''.join((token[0] for token in tokens if token))

def street_fallback_street_similarity(official_street, osm_street):
    """
    General-purpose street-name reconciliation.

    It avoids a known failure mode of token_set_ratio: a short subset
    such as "Via Rosario" must not score 100 against
    "Via Mario Rosario Greco".

    The last core token is treated as a surname / discriminating token
    when available, while abbreviated given names such as
    "Mario Rosario" -> "MR" are handled through initials.
    """
    official_norm = street_fallback_normalize_text(official_street)
    osm_norm = street_fallback_normalize_text(osm_street)
    if not official_norm or not osm_norm:
        return {'street_score': 0.0, 'base_score': 0.0, 'core_score': 0.0, 'surname_score': 0.0, 'initials_compatible': False, 'subset_penalty': False}
    official_core = street_fallback_road_core_tokens(official_street)
    osm_core = street_fallback_road_core_tokens(osm_street)
    official_core_text = ' '.join(official_core)
    osm_core_text = ' '.join(osm_core)
    base_score = float(fuzz.ratio(official_norm, osm_norm))
    core_score = float(fuzz.token_sort_ratio(official_core_text, osm_core_text))
    surname_score = 0.0
    initials_compatible = False
    subset_penalty = False
    if official_core and osm_core:
        surname_score = float(fuzz.ratio(official_core[-1], osm_core[-1]))
        official_before_surname = official_core[:-1]
        osm_before_surname = osm_core[:-1]
        if surname_score >= 90.0:
            official_initials = street_fallback_initials_signature(official_before_surname) if official_before_surname else ''
            osm_initials = street_fallback_initials_signature(osm_before_surname) if osm_before_surname else ''
            if official_initials and osm_initials and (official_initials == osm_initials):
                initials_compatible = True
        if len(official_core) >= 2 and len(osm_core) < len(official_core) and (surname_score < 70.0):
            subset_penalty = True
    if official_core_text and osm_core_text and (official_core_text == osm_core_text):
        street_score = 100.0
    elif surname_score >= 95.0 and initials_compatible:
        street_score = 98.0
    elif surname_score >= 95.0 and (len(official_core) == 1 or len(osm_core) == 1):
        street_score = max(92.0, 0.45 * base_score + 0.3 * core_score + 0.25 * surname_score)
    else:
        street_score = 0.4 * base_score + 0.35 * core_score + 0.25 * surname_score
    if subset_penalty:
        street_score = min(street_score, 65.0)
    return {'street_score': float(street_score), 'base_score': base_score, 'core_score': core_score, 'surname_score': surname_score, 'initials_compatible': initials_compatible, 'subset_penalty': subset_penalty}

def street_fallback_parse_attributes(value):
    if isinstance(value, dict):
        return value
    value = street_fallback_clean_text(value)
    if value is None:
        return {}
    try:
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    except Exception:
        pass
    return {}

def street_fallback_extract_names_from_attributes(value):
    attrs = street_fallback_parse_attributes(value)
    name = attrs.get('name')
    if isinstance(name, list):
        return [street_fallback_clean_text(item) for item in name if street_fallback_clean_text(item)]
    if street_fallback_clean_text(name):
        return [street_fallback_clean_text(name)]
    return []

def street_fallback_load_pending_buildings(args):
    """
    Load only buildings still requiring automatic spatial resolution
    after the canonical geocoder + OSM fusion stage.

    This replaces the historical dependency on
    *_with_reused_locations.parquet and makes the fallback usable
    for zero-touch municipalities.
    """
    mim_dir = street_fallback_PROCESSED_MIM_DIR / args.municipality_code
    source_path = mim_dir / f'physical_school_buildings_{args.building_year}_final_v2.parquet'
    if not source_path.exists():
        raise FileNotFoundError(f'Dataset fusion edifici non trovato: {source_path}')
    buildings = pd.read_parquet(source_path).copy()
    required_columns = {'building_code', 'building_municipality_name', 'official_building_address', 'final_location_status'}
    missing = required_columns - set(buildings.columns)
    if missing:
        raise RuntimeError('Colonne mancanti nel dataset fusion: ' + ', '.join(sorted(missing)))
    buildings['building_code'] = buildings['building_code'].astype('string').str.strip()
    pending = buildings.loc[buildings['final_location_status'].isin(['review', 'unresolved'])].copy()
    if 'building_municipality_code' in pending.columns:
        municipality_codes = pending['building_municipality_code'].astype('string').str.strip().str.zfill(6)
        pending = pending.loc[municipality_codes == args.municipality_code].copy()
    else:
        pending = pending.loc[pending['building_code'].astype(str).str.startswith(args.municipality_code)].copy()
    return (pending, source_path)

def street_fallback_load_osm_street_names(args):
    edges_path = street_fallback_PROCESSED_OSM_DIR / args.municipality_code / 'walk_edges.parquet'
    if not edges_path.exists():
        raise FileNotFoundError(f'Rete OSM non trovata: {edges_path}')
    edges = pd.read_parquet(edges_path, columns=['attributes'])
    names = set()
    for value in edges['attributes']:
        for name in street_fallback_extract_names_from_attributes(value):
            names.add(name)
    street_names = sorted(names, key=lambda value: (street_fallback_normalize_text(value), value))
    if not street_names:
        raise RuntimeError('Nessun nome stradale trovato negli attributi OSM.')
    return (street_names, edges_path)

def street_fallback_load_municipality_context(pending, args):
    census_path = street_fallback_PROCESSED_ISTAT_DIR / f'{args.municipality_code}_census_areas_2021.parquet'
    if not census_path.exists():
        raise FileNotFoundError(f'Dataset ISTAT non trovato: {census_path}')
    census = gpd.read_parquet(census_path).to_crs(4326)
    geometry = census.geometry.union_all()
    minx, miny, maxx, maxy = geometry.bounds
    municipality_names = pending['building_municipality_name'].dropna().astype(str).str.strip().unique().tolist()
    if len(municipality_names) != 1:
        raise RuntimeError('Impossibile determinare un solo nome di comune: ' + repr(municipality_names))
    return {'municipality_name': municipality_names[0], 'geometry': geometry, 'viewbox': f'{minx},{maxy},{maxx},{miny}'}

def street_fallback_rank_osm_streets(official_street, osm_street_names, top_k):
    rows = []
    for osm_street in osm_street_names:
        scores = street_fallback_street_similarity(official_street, osm_street)
        rows.append({'osm_street_name': osm_street, **scores})
    rows.sort(key=lambda row: (row['street_score'], row['core_score'], row['base_score']), reverse=True)
    return rows[:top_k]

def street_fallback_classify_street_match(candidates, auto_threshold, review_threshold, min_margin):
    if not candidates:
        return ('unresolved', None)
    top_score = candidates[0]['street_score']
    second_score = candidates[1]['street_score'] if len(candidates) > 1 else 0.0
    margin = top_score - second_score
    if top_score >= auto_threshold and margin >= min_margin:
        return ('matched_auto', margin)
    if top_score >= review_threshold:
        return ('review', margin)
    return ('unresolved', margin)

def street_fallback_load_cache(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}

def street_fallback_save_cache(path, cache):
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding='utf-8')

def street_fallback_query_nominatim(session, matched_street, civic, context, pause_seconds):
    url = 'https://nominatim.openstreetmap.org/search'
    query_street = f'{matched_street} {civic}' if civic else matched_street
    common = {'format': 'jsonv2', 'limit': 5, 'countrycodes': 'it', 'addressdetails': 1, 'bounded': 1, 'viewbox': context['viewbox']}
    params = {**common, 'street': query_street, 'city': context['municipality_name']}
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    results = response.json()
    time.sleep(pause_seconds)
    if not results:
        params = {**common, 'q': f"{query_street}, {context['municipality_name']}, Italia"}
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()
        results = response.json()
        time.sleep(pause_seconds)
    return (query_street, results)

def street_fallback_normalize_civic(value):
    value = street_fallback_normalize_text(value)
    return value.replace(' ', '')

def street_fallback_score_geocoder_candidate(matched_street, civic, candidate, context):
    try:
        longitude = float(candidate['lon'])
        latitude = float(candidate['lat'])
    except Exception:
        return None
    display_name = street_fallback_clean_text(candidate.get('display_name')) or ''
    inside = bool(context['geometry'].covers(Point(longitude, latitude)))
    address = candidate.get('address')
    if not isinstance(address, dict):
        address = {}
    candidate_road = address.get('road') or address.get('pedestrian') or address.get('residential') or address.get('footway')
    road_score = float(fuzz.token_set_ratio(street_fallback_normalize_text(matched_street), street_fallback_normalize_text(candidate_road or display_name)))
    candidate_house_number = street_fallback_clean_text(address.get('house_number'))
    civic_match = None
    if civic:
        civic_match = street_fallback_normalize_civic(civic) == street_fallback_normalize_civic(candidate_house_number) if candidate_house_number else False
    if civic and civic_match:
        resolution = 'address'
    else:
        resolution = 'street'
    return {'display_name': display_name, 'longitude': longitude, 'latitude': latitude, 'inside_target': inside, 'candidate_road': candidate_road, 'candidate_house_number': candidate_house_number, 'road_score': road_score, 'civic_match': civic_match, 'resolution': resolution, 'osm_type': candidate.get('osm_type'), 'osm_id': candidate.get('osm_id'), 'place_id': candidate.get('place_id')}

def street_fallback_write_empty_outputs(args):
    """Write valid empty fallback outputs when no buildings need recovery."""

    features_dir = (
        street_fallback_FEATURES_MIM_DIR
        / args.municipality_code
    )

    features_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        features_dir
        / f"school_osm_street_fallback_{args.building_year}.csv"
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

    summary_columns = [
        "building_code",
        "official_building_address",
        "parsed_street",
        "parsed_civic",
        "street_match_status",
        "matched_osm_street",
        "street_match_score",
        "street_match_margin",
        "street_initials_compatible",
        "geocoder_query",
        "geocoder_query_source",
        "fallback_status",
        "fallback_confidence",
        "candidate_longitude",
        "candidate_latitude",
        "candidate_display_name",
        "candidate_road",
        "candidate_house_number",
        "candidate_road_score",
        "candidate_civic_match",
        "candidate_inside_target",
        "candidate_resolution",
    ]

    street_candidate_columns = [
        "building_code",
        "official_building_address",
        "parsed_street",
        "parsed_civic",
        "candidate_rank",
        "osm_street_name",
        "street_score",
        "base_score",
        "core_score",
        "surname_score",
        "initials_compatible",
        "subset_penalty",
    ]

    geocoder_candidate_columns = [
        "building_code",
        "query_street",
        "candidate_rank",
        "display_name",
        "longitude",
        "latitude",
        "inside_target",
        "candidate_road",
        "candidate_house_number",
        "road_score",
        "civic_match",
        "resolution",
        "osm_type",
        "osm_id",
        "place_id",
    ]

    pd.DataFrame(
        columns=summary_columns
    ).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        columns=street_candidate_columns
    ).to_csv(
        street_candidates_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        columns=geocoder_candidate_columns
    ).to_csv(
        geocoder_candidates_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("✓ Nessun pending: creati output fallback vuoti.")
    print(f"✓ {summary_path}")
    print(f"✓ {street_candidates_path}")
    print(f"✓ {geocoder_candidates_path}")


def street_fallback_main():
    args = street_fallback_parse_args()
    pending, buildings_path = street_fallback_load_pending_buildings(args)
    if pending.empty:
        print('Nessun edificio unresolved da processare.')
        street_fallback_write_empty_outputs(args)
        return
    osm_street_names, edges_path = street_fallback_load_osm_street_names(args)
    context = street_fallback_load_municipality_context(pending, args)
    print('\n====================================')
    print(' NATIONAL OSM STREET FALLBACK')
    print('====================================')
    print(f"Municipality: {context['municipality_name']} ({args.municipality_code})")
    print(f'Pending buildings: {len(pending)}')
    print(f'Unique named OSM streets: {len(osm_street_names)}')
    cache_dir = street_fallback_RAW_MIM_DIR / 'geocoding' / args.municipality_code
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'nominatim_osm_street_fallback_{args.building_year}.json'
    cache = street_fallback_load_cache(cache_path)
    session = requests.Session()
    session.headers.update({'User-Agent': 'urban-digital-twin-thesis/1.0 (academic research)'})
    summary_rows = []
    street_candidate_rows = []
    geocoder_candidate_rows = []
    query_results_memory = {}
    for _, building in pending.sort_values('building_code').iterrows():
        building_code = building['building_code']
        official_address = street_fallback_clean_text(building.get('official_building_address'))
        parsed_street, parsed_civic = street_fallback_split_official_address(official_address)
        street_candidates = street_fallback_rank_osm_streets(official_street=parsed_street, osm_street_names=osm_street_names, top_k=args.top_k_streets)
        street_status, street_margin = street_fallback_classify_street_match(candidates=street_candidates, auto_threshold=args.auto_street_threshold, review_threshold=args.review_street_threshold, min_margin=args.min_margin)
        for rank, candidate in enumerate(street_candidates, start=1):
            street_candidate_rows.append({'building_code': building_code, 'official_building_address': official_address, 'parsed_street': parsed_street, 'parsed_civic': parsed_civic, 'candidate_rank': rank, **candidate})
        top_street = street_candidates[0] if street_candidates else None
        query_street = None
        geocoder_candidates = []
        geocoder_source = None
        if street_status == 'matched_auto' and top_street:
            matched_osm_street = top_street['osm_street_name']
            query_key = f'{args.municipality_code}|{street_fallback_normalize_text(matched_osm_street)}|{street_fallback_normalize_civic(parsed_civic)}'
            if not args.refresh and query_key in query_results_memory:
                query_street = query_results_memory[query_key]['query_street']
                raw_results = query_results_memory[query_key]['results']
                geocoder_source = 'run_cache'
            elif not args.refresh and query_key in cache:
                query_street = cache[query_key]['query_street']
                raw_results = cache[query_key]['results']
                geocoder_source = 'disk_cache'
                query_results_memory[query_key] = cache[query_key]
            else:
                query_street, raw_results = street_fallback_query_nominatim(session=session, matched_street=matched_osm_street, civic=parsed_civic, context=context, pause_seconds=args.pause_seconds)
                cache_entry = {'query_street': query_street, 'matched_osm_street': matched_osm_street, 'parsed_civic': parsed_civic, 'queried_at_utc': datetime.now(timezone.utc).isoformat(), 'results': raw_results}
                cache[query_key] = cache_entry
                query_results_memory[query_key] = cache_entry
                street_fallback_save_cache(cache_path, cache)
                geocoder_source = 'nominatim'
            for raw_candidate in raw_results:
                item = street_fallback_score_geocoder_candidate(matched_street=matched_osm_street, civic=parsed_civic, candidate=raw_candidate, context=context)
                if item is not None:
                    geocoder_candidates.append(item)
            geocoder_candidates.sort(key=lambda row: (row['inside_target'], row['civic_match'] is True, row['road_score']), reverse=True)
        for rank, candidate in enumerate(geocoder_candidates, start=1):
            geocoder_candidate_rows.append({'building_code': building_code, 'query_street': query_street, 'candidate_rank': rank, **candidate})
        top_geocoder = geocoder_candidates[0] if geocoder_candidates else None
        if top_geocoder and top_geocoder['inside_target'] and (top_geocoder['road_score'] >= 85.0):
            if parsed_civic and top_geocoder['civic_match'] is True:
                fallback_status = 'address_candidate'
                fallback_confidence = 'high'
            else:
                fallback_status = 'street_anchor_candidate'
                fallback_confidence = 'medium'
        elif top_geocoder:
            fallback_status = 'review'
            fallback_confidence = 'low'
        else:
            fallback_status = 'unresolved'
            fallback_confidence = 'low'
        summary_rows.append({'building_code': building_code, 'official_building_address': official_address, 'parsed_street': parsed_street, 'parsed_civic': parsed_civic, 'street_match_status': street_status, 'matched_osm_street': top_street['osm_street_name'] if top_street else None, 'street_match_score': top_street['street_score'] if top_street else None, 'street_match_margin': street_margin, 'street_initials_compatible': top_street['initials_compatible'] if top_street else None, 'geocoder_query': query_street, 'geocoder_query_source': geocoder_source, 'fallback_status': fallback_status, 'fallback_confidence': fallback_confidence, 'candidate_longitude': top_geocoder['longitude'] if top_geocoder else None, 'candidate_latitude': top_geocoder['latitude'] if top_geocoder else None, 'candidate_display_name': top_geocoder['display_name'] if top_geocoder else None, 'candidate_road': top_geocoder['candidate_road'] if top_geocoder else None, 'candidate_house_number': top_geocoder['candidate_house_number'] if top_geocoder else None, 'candidate_road_score': top_geocoder['road_score'] if top_geocoder else None, 'candidate_civic_match': top_geocoder['civic_match'] if top_geocoder else None, 'candidate_inside_target': top_geocoder['inside_target'] if top_geocoder else None, 'candidate_resolution': top_geocoder['resolution'] if top_geocoder else None})
    summary = pd.DataFrame(summary_rows)
    street_candidates_df = pd.DataFrame(street_candidate_rows)
    geocoder_candidates_df = pd.DataFrame(geocoder_candidate_rows)
    features_dir = street_fallback_FEATURES_MIM_DIR / args.municipality_code
    features_dir.mkdir(parents=True, exist_ok=True)
    summary_path = features_dir / f'school_osm_street_fallback_{args.building_year}.csv'
    street_candidates_path = features_dir / f'school_osm_street_fallback_candidates_{args.building_year}.csv'
    geocoder_candidates_path = features_dir / f'school_osm_street_geocoder_candidates_{args.building_year}.csv'
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    street_candidates_df.to_csv(street_candidates_path, index=False, encoding='utf-8-sig')
    geocoder_candidates_df.to_csv(geocoder_candidates_path, index=False, encoding='utf-8-sig')
    print('\n=== RESULTS ===')
    print(summary[['building_code', 'official_building_address', 'parsed_street', 'parsed_civic', 'street_match_status', 'matched_osm_street', 'street_match_score', 'street_match_margin', 'street_initials_compatible', 'fallback_status', 'candidate_display_name', 'candidate_longitude', 'candidate_latitude', 'candidate_resolution']].to_string(index=False))
    print('\nStreet match statuses:')
    print(summary['street_match_status'].value_counts(dropna=False).to_string())
    print('\nFallback statuses:')
    print(summary['fallback_status'].value_counts(dropna=False).to_string())
    print('\n=== OUTPUT ===')
    print(f'✓ {summary_path}')
    print(f'✓ {street_candidates_path}')
    print(f'✓ {geocoder_candidates_path}')
    print('\nNOTA METODOLOGICA:')
    print("Il fallback usa esclusivamente l'indirizzo MIM originale, i nomi stradali del grafo OSM del comune e Nominatim.")
    print('Non contiene alias, civici o regole specifiche del comune target.')
    print("Un risultato street_anchor_candidate non viene interpretato come coordinata esatta dell'edificio.")

# ============================================================================
# REMAINING SCHOOL SERVICES
# ============================================================================

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
remaining_services_ROOT = Path(__file__).resolve().parents[2]
remaining_services_PROCESSED_MIM_DIR = remaining_services_ROOT / 'data' / 'processed' / 'mim'
remaining_services_PROCESSED_ISTAT_DIR = remaining_services_ROOT / 'data' / 'processed' / 'istat'
remaining_services_FEATURES_MIM_DIR = remaining_services_ROOT / 'data' / 'features' / 'mim'
remaining_services_RAW_MIM_DIR = remaining_services_ROOT / 'data' / 'raw' / 'mim'

def remaining_services_parse_args():
    parser = argparse.ArgumentParser(description="Risoluzione automatica dei record scolastici residui che non sono coperti dall'Anagrafe edilizia: scuole paritarie + unmatched fisici.")
    parser.add_argument('--municipality-code', required=True)
    parser.add_argument('--school-year', default='202425')
    parser.add_argument('--pause-seconds', type=float, default=1.1)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    return args

def remaining_services_clean_text(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    value = str(value).strip()
    return value or None

def remaining_services_strip_accents(value):
    return ''.join((char for char in unicodedata.normalize('NFKD', value) if not unicodedata.combining(char)))

def remaining_services_normalize_text(value):
    value = remaining_services_clean_text(value)
    if value is None:
        return ''
    value = remaining_services_strip_accents(value).upper()
    value = value.replace('’', "'").replace('`', "'")
    value = re.sub("[^A-Z0-9']+", ' ', value)
    value = re.sub('\\s+', ' ', value).strip()
    return value

def remaining_services_normalize_address(value):
    value = remaining_services_normalize_text(value)
    if not value:
        return ''
    replacements = {'S N C': '', 'SNC': '', 'S N': '', 'V LE': 'VIALE', 'VLE': 'VIALE', 'P ZA': 'PIAZZA', 'PZZA': 'PIAZZA'}
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub('(?<=[A-Z])(?=\\d)', ' ', value)
    value = re.sub('(?<=\\d)(?=[A-Z])', ' ', value)
    value = re.sub('\\s+', ' ', value).strip()
    return value

def remaining_services_extract_civic(value):
    value = remaining_services_normalize_address(value)
    if not value:
        return None
    match = re.search('(?:^|\\s)(\\d+(?:[A-Z])?(?:[/\\-]\\d+(?:[A-Z])?)*)\\s*$', value)
    return match.group(1) if match else None

def remaining_services_normalize_civic(value):
    return remaining_services_normalize_text(value).replace(' ', '')

def remaining_services_school_name_core(value):
    value = remaining_services_normalize_text(value)
    if not value:
        return ''
    generic = {'SCUOLA', 'ISTITUTO', 'IST', 'I', 'C', 'IC', 'PLESSO', 'SEDE', 'MATERNA', 'INFANZIA', 'PRIMARIA', 'SECONDARIA', 'PRIMO', 'SECONDO', 'GRADO', 'STATALE', 'PARITARIA', 'L', 'CLAS'}
    tokens = [token for token in value.split() if token not in generic]
    return ' '.join(tokens) or value

def remaining_services_load_cache(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}

def remaining_services_save_cache(path, cache):
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')

def remaining_services_load_boundary(municipality_code):
    path = remaining_services_PROCESSED_ISTAT_DIR / f'{municipality_code}_census_areas_2021.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Boundary ISTAT non trovato: {path}')
    areas = gpd.read_parquet(path).to_crs(4326)
    geometry = areas.geometry.union_all()
    minx, miny, maxx, maxy = geometry.bounds
    return {'geometry': geometry, 'viewbox': f'{minx},{maxy},{maxx},{miny}'}

def remaining_services_inside_boundary(boundary_geometry, lon, lat):
    return bool(boundary_geometry.covers(Point(float(lon), float(lat))))

def remaining_services_get_result_road(address):
    if not isinstance(address, dict):
        return None
    return address.get('road') or address.get('pedestrian') or address.get('residential') or address.get('footway') or address.get('path')

def remaining_services_geocode_queries(session, *, school_name, address, municipality_name, boundary, pause_seconds):
    common = {'format': 'jsonv2', 'limit': 5, 'countrycodes': 'it', 'addressdetails': 1, 'namedetails': 1, 'bounded': 1, 'viewbox': boundary['viewbox']}
    variants = []
    if address:
        variants.append(('structured_address', {**common, 'street': address, 'city': municipality_name}))
    if school_name and address:
        variants.append(('name_address', {**common, 'q': f'{school_name}, {address}, {municipality_name}, Italia'}))
    if address:
        variants.append(('freeform_address', {**common, 'q': f'{address}, {municipality_name}, Italia'}))
    all_results = []
    for query_type, params in variants:
        response = session.get('https://nominatim.openstreetmap.org/search', params=params, timeout=60)
        response.raise_for_status()
        results = response.json()
        time.sleep(pause_seconds)
        for result in results:
            all_results.append({'query_type': query_type, 'result': result})
    return all_results

def remaining_services_score_candidate(*, school_name, school_address, candidate, boundary_geometry):
    raw = candidate['result']
    try:
        lon = float(raw['lon'])
        lat = float(raw['lat'])
    except Exception:
        return None
    display_name = remaining_services_clean_text(raw.get('display_name')) or ''
    address_dict = raw.get('address')
    if not isinstance(address_dict, dict):
        address_dict = {}
    namedetails = raw.get('namedetails')
    if not isinstance(namedetails, dict):
        namedetails = {}
    result_name = namedetails.get('name') or raw.get('name') or display_name.split(',')[0]
    result_road = remaining_services_get_result_road(address_dict)
    result_house_number = remaining_services_clean_text(address_dict.get('house_number'))
    requested_address = remaining_services_normalize_address(school_address)
    result_address = remaining_services_normalize_address(' '.join((item for item in [result_road, result_house_number] if item)) or display_name)
    address_score = float(fuzz.token_set_ratio(requested_address, result_address))
    requested_name = remaining_services_school_name_core(school_name)
    result_name_core = remaining_services_school_name_core(result_name)
    name_score = float(fuzz.token_set_ratio(requested_name, result_name_core)) if requested_name and result_name_core else 0.0
    requested_civic = remaining_services_extract_civic(school_address)
    civic_match = None
    if requested_civic:
        civic_match = remaining_services_normalize_civic(requested_civic) == remaining_services_normalize_civic(result_house_number) if result_house_number else False
    inside = remaining_services_inside_boundary(boundary_geometry, lon, lat)
    osm_class = remaining_services_clean_text(raw.get('class'))
    osm_type = remaining_services_clean_text(raw.get('type'))
    is_school_like = remaining_services_normalize_text(osm_type) in {'SCHOOL', 'KINDERGARTEN', 'COLLEGE', 'UNIVERSITY'} or (remaining_services_normalize_text(osm_class) in {'AMENITY', 'BUILDING'} and name_score >= 75.0)
    if requested_civic:
        if inside and civic_match is True and (address_score >= 85.0):
            resolution = 'address'
            status = 'accepted_address'
            confidence = 'high'
        elif inside and is_school_like and (name_score >= 85.0) and (address_score >= 60.0):
            resolution = 'site'
            status = 'accepted_site'
            confidence = 'high'
        elif inside and address_score >= 85.0:
            resolution = 'street_anchor'
            status = 'street_anchor_candidate'
            confidence = 'medium'
        else:
            resolution = None
            status = 'review'
            confidence = 'low'
    elif inside and is_school_like and (name_score >= 85.0) and (address_score >= 60.0):
        resolution = 'site'
        status = 'accepted_site'
        confidence = 'high'
    elif inside and address_score >= 85.0:
        resolution = 'street_anchor'
        status = 'street_anchor_candidate'
        confidence = 'medium'
    else:
        resolution = None
        status = 'review'
        confidence = 'low'
    return {'query_type': candidate['query_type'], 'display_name': display_name, 'longitude': lon, 'latitude': lat, 'inside_target': inside, 'result_name': result_name, 'result_road': result_road, 'result_house_number': result_house_number, 'address_score': address_score, 'name_score': name_score, 'civic_match': civic_match, 'osm_class': osm_class, 'osm_type': osm_type, 'status': status, 'confidence': confidence, 'resolution': resolution}

def remaining_services_candidate_sort_key(candidate):
    status_rank = {'accepted_address': 5, 'accepted_site': 4, 'street_anchor_candidate': 3, 'review': 1}
    return (status_rank.get(candidate['status'], 0), bool(candidate['inside_target']), candidate['address_score'], candidate['name_score'])

def remaining_services_main():
    args = remaining_services_parse_args()
    processed_dir = remaining_services_PROCESSED_MIM_DIR / args.municipality_code
    features_dir = remaining_services_FEATURES_MIM_DIR / args.municipality_code
    features_dir.mkdir(parents=True, exist_ok=True)
    registry_path = processed_dir / f'schools_registry_{args.school_year}.parquet'
    locations_path = processed_dir / f'school_locations_{args.school_year}_v2.parquet'
    unmatched_path = processed_dir / f'school_unmatched_classification_{args.school_year}.parquet'
    registry = pd.read_parquet(registry_path)
    locations = pd.read_parquet(locations_path)
    unmatched = pd.read_parquet(unmatched_path)
    for df in [registry, locations, unmatched]:
        df['school_code'] = df['school_code'].astype('string').str.strip()
    paritary_codes = set(registry.loc[registry['registry_type'].astype(str).str.lower().str.contains('par'), 'school_code'].dropna().astype(str))
    physical_unmatched_codes = set(unmatched.loc[unmatched['needs_geolocation'] == True, 'school_code'].dropna().astype(str))
    target_codes = paritary_codes | physical_unmatched_codes
    target_registry = registry.loc[registry['school_code'].isin(target_codes)].copy()
    municipality_names = target_registry['municipality_name'].dropna().astype(str).str.strip().unique().tolist()
    if len(municipality_names) != 1:
        raise RuntimeError('Impossibile determinare un unico comune target: ' + repr(municipality_names))
    municipality_name = municipality_names[0]
    boundary = remaining_services_load_boundary(args.municipality_code)
    locations_by_code = locations.set_index('school_code')
    accepted_by_address = {}
    for _, row in locations.iterrows():
        if not bool(row.get('geometry_accepted', False)):
            continue
        address = remaining_services_normalize_address(row.get('address'))
        site_id = remaining_services_clean_text(row.get('candidate_site_id'))
        if not address:
            continue
        accepted_by_address.setdefault(address, []).append({'school_code': row['school_code'], 'candidate_site_id': site_id, 'longitude': row.get('longitude'), 'latitude': row.get('latitude'), 'candidate_site_name': row.get('candidate_site_name')})
    cache_dir = remaining_services_RAW_MIM_DIR / 'geocoding' / args.municipality_code
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'nominatim_remaining_school_services_{args.school_year}.json'
    cache = remaining_services_load_cache(cache_path)
    session = requests.Session()
    session.headers.update({'User-Agent': 'urban-digital-twin-thesis/1.0 (academic research)'})
    summary_rows = []
    candidate_rows = []
    for _, school in target_registry.sort_values('school_code').iterrows():
        code = str(school['school_code'])
        name = remaining_services_clean_text(school.get('school_name'))
        address = remaining_services_clean_text(school.get('school_address'))
        registry_type = remaining_services_clean_text(school.get('registry_type'))
        loc = locations_by_code.loc[code] if code in locations_by_code.index else None
        if isinstance(loc, pd.DataFrame):
            raise RuntimeError(f'school_code duplicato nelle locations: {code}')
        if loc is not None and bool(loc.get('geometry_accepted', False)):
            summary_rows.append({'school_code': code, 'school_name': name, 'school_address': address, 'registry_type': registry_type, 'resolution_status': 'accepted_osm', 'resolution_method': 'existing_osm_auto_match', 'confidence': 'high', 'coordinate_resolution': 'site', 'longitude': loc.get('longitude'), 'latitude': loc.get('latitude'), 'matched_name': loc.get('candidate_site_name'), 'matched_address': loc.get('candidate_site_address'), 'notes': 'Match OSM già accettato dalla pipeline V2.'})
            continue
        normalized_address = remaining_services_normalize_address(address)
        current_candidate_site_id = remaining_services_clean_text(loc.get('candidate_site_id')) if loc is not None else None
        peers = accepted_by_address.get(normalized_address, []) if normalized_address else []
        peer = None
        for item in peers:
            if current_candidate_site_id and item['candidate_site_id'] == current_candidate_site_id:
                peer = item
                break
        if peer is not None:
            summary_rows.append({'school_code': code, 'school_name': name, 'school_address': address, 'registry_type': registry_type, 'resolution_status': 'accepted_peer_site', 'resolution_method': 'same_address_same_osm_site', 'confidence': 'high', 'coordinate_resolution': 'site', 'longitude': peer['longitude'], 'latitude': peer['latitude'], 'matched_name': peer['candidate_site_name'], 'matched_address': address, 'notes': f"Stesso indirizzo ufficiale e stesso candidato OSM di un record già accettato ({peer['school_code']})."})
            continue
        cache_key = f'{args.municipality_code}|{remaining_services_normalize_text(name)}|{remaining_services_normalize_address(address)}'
        if not args.refresh and cache_key in cache:
            raw_candidates = cache[cache_key]['results']
        else:
            raw_candidates = remaining_services_geocode_queries(session, school_name=name, address=address, municipality_name=municipality_name, boundary=boundary, pause_seconds=args.pause_seconds)
            cache[cache_key] = {'school_code': code, 'school_name': name, 'school_address': address, 'queried_at_utc': datetime.now(timezone.utc).isoformat(), 'results': raw_candidates}
            remaining_services_save_cache(cache_path, cache)
        scored = []
        for raw_candidate in raw_candidates:
            candidate = remaining_services_score_candidate(school_name=name, school_address=address, candidate=raw_candidate, boundary_geometry=boundary['geometry'])
            if candidate is not None:
                scored.append(candidate)
        scored.sort(key=remaining_services_candidate_sort_key, reverse=True)
        for rank, candidate in enumerate(scored, start=1):
            candidate_rows.append({'school_code': code, 'school_name': name, 'school_address': address, 'rank': rank, **candidate})
        best = scored[0] if scored else None
        if best is None:
            summary_rows.append({'school_code': code, 'school_name': name, 'school_address': address, 'registry_type': registry_type, 'resolution_status': 'unresolved', 'resolution_method': 'nominatim_bounded', 'confidence': 'low', 'coordinate_resolution': None, 'longitude': None, 'latitude': None, 'matched_name': None, 'matched_address': None, 'notes': 'Nessun candidato geocoding disponibile.'})
            continue
        summary_rows.append({'school_code': code, 'school_name': name, 'school_address': address, 'registry_type': registry_type, 'resolution_status': best['status'], 'resolution_method': 'nominatim_bounded', 'confidence': best['confidence'], 'coordinate_resolution': best['resolution'], 'longitude': best['longitude'], 'latitude': best['latitude'], 'matched_name': best['display_name'], 'matched_address': ' '.join((item for item in [best['result_road'], best['result_house_number']] if item)) or None, 'notes': f"address_score={best['address_score']:.2f}; name_score={best['name_score']:.2f}; civic_match={best['civic_match']}; inside_target={best['inside_target']}"})
    summary = pd.DataFrame(summary_rows)
    candidates = pd.DataFrame(candidate_rows)
    output_csv = features_dir / f'remaining_school_services_{args.school_year}.csv'
    output_parquet = processed_dir / f'remaining_school_services_{args.school_year}.parquet'
    candidates_csv = features_dir / f'remaining_school_service_candidates_{args.school_year}.csv'
    summary.to_csv(output_csv, index=False, encoding='utf-8-sig')
    summary.to_parquet(output_parquet, index=False)
    candidates.to_csv(candidates_csv, index=False, encoding='utf-8-sig')
    print('\n====================================')
    print(' REMAINING SCHOOL SERVICES')
    print('====================================')
    print(f'Paritary schools: {len(paritary_codes)}')
    print(f'Physical unmatched schools: {len(physical_unmatched_codes)}')
    print(f'Target records: {len(summary)}')
    print('\n=== RESULTS ===')
    print(summary[['school_code', 'school_name', 'school_address', 'registry_type', 'resolution_status', 'resolution_method', 'confidence', 'coordinate_resolution', 'matched_name', 'longitude', 'latitude']].to_string(index=False))
    print('\nStatuses:')
    print(summary['resolution_status'].value_counts(dropna=False).to_string())
    print('\n=== OUTPUT ===')
    print(f'✓ {output_csv}')
    print(f'✓ {output_parquet}')
    print(f'✓ {candidates_csv}')
    print('\nNOTA:')
    print('Lo script seleziona automaticamente paritarie e unmatched fisici; non contiene codici scuola o regole specifiche del comune target.')

# ============================================================================
# FINAL CANONICAL EDUCATION LAYER
# ============================================================================

import argparse
import hashlib
import json
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
final_builder_ROOT = Path(__file__).resolve().parents[2]
final_builder_PROCESSED = final_builder_ROOT / 'data' / 'processed' / 'mim'
final_builder_FEATURES = final_builder_ROOT / 'data' / 'features' / 'mim'

def final_builder_clean(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return None if not s or s.lower() in {'nan', 'none', '<na>'} else s

def final_builder_fnum(v):
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def final_builder_truth(v):
    if isinstance(v, bool):
        return v
    s = final_builder_clean(v)
    return bool(s and s.lower() in {'1', 'true', 'yes', 'y', 'si', 'sì'})

def final_builder_parse_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [x for x in (final_builder_clean(i) for i in v) if x]
    try:
        if pd.isna(v):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        try:
            x = json.loads(v)
            if isinstance(x, list):
                return [y for y in (final_builder_clean(i) for i in x) if y]
        except json.JSONDecodeError:
            pass
    x = final_builder_clean(v)
    return [x] if x else []

def final_builder_jlist(values):
    out, seen = ([], set())
    for v in values:
        v = final_builder_clean(v)
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return json.dumps(out, ensure_ascii=False)

def final_builder_pick_list(row, base):
    for col in (base, f'{base}_x', f'{base}_y'):
        if col in row.index:
            vals = final_builder_parse_list(row.get(col))
            if vals:
                return vals
    return []

def final_builder_require(df, cols, label):
    missing = set(cols) - set(df.columns)
    if missing:
        raise RuntimeError(f"{label}: colonne mancanti: {', '.join(sorted(missing))}")

def final_builder_deterministic_id(prefix, parts):
    raw = '|'.join(('' if p is None else str(p) for p in parts))
    return f'{prefix}:{hashlib.sha1(raw.encode()).hexdigest()[:16]}'

def final_builder_parse_args():
    p = argparse.ArgumentParser(description='Builder Education generalizzato e zero-touch.')
    p.add_argument('--municipality-code', required=True)
    p.add_argument('--school-year', default='202425')
    p.add_argument('--building-year', default='202425')
    a = p.parse_args()
    a.municipality_code = str(a.municipality_code).strip().zfill(6)
    if len(a.municipality_code) != 6 or not a.municipality_code.isdigit():
        raise ValueError('municipality-code deve avere 6 cifre')
    return a

def final_builder_load_inputs(a):
    pdir = final_builder_PROCESSED / a.municipality_code
    fdir = final_builder_FEATURES / a.municipality_code
    fdir.mkdir(parents=True, exist_ok=True)
    paths = {'buildings': pdir / f'physical_school_buildings_{a.building_year}_final_v2.parquet', 'fallback': fdir / f'school_osm_street_fallback_{a.building_year}.csv', 'remaining': pdir / f'remaining_school_services_{a.school_year}.parquet', 'unmatched': pdir / f'school_unmatched_classification_{a.school_year}.parquet'}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    buildings = pd.read_parquet(paths['buildings'])
    fallback = pd.read_csv(paths['fallback'], dtype={'building_code': 'string'})
    remaining = pd.read_parquet(paths['remaining'])
    unmatched = pd.read_parquet(paths['unmatched'])
    final_builder_require(buildings, ['building_code', 'official_building_address', 'building_municipality_name', 'building_postal_code', 'final_location_status', 'final_geometry_source', 'final_longitude', 'final_latitude', 'location_confidence'], 'buildings')
    final_builder_require(fallback, ['building_code', 'fallback_status', 'fallback_confidence', 'candidate_longitude', 'candidate_latitude', 'candidate_resolution'], 'fallback')
    final_builder_require(remaining, ['school_code', 'school_name', 'school_address', 'registry_type', 'resolution_status', 'resolution_method', 'confidence', 'coordinate_resolution', 'longitude', 'latitude'], 'remaining')
    final_builder_require(unmatched, ['school_code', 'unmatched_class', 'creates_public_school_site', 'needs_geolocation', 'site_handling'], 'unmatched')
    for df, col in [(buildings, 'building_code'), (fallback, 'building_code'), (remaining, 'school_code'), (unmatched, 'school_code')]:
        df[col] = df[col].astype('string').str.strip()
    if buildings['building_code'].duplicated().any():
        raise RuntimeError('building_code duplicati in buildings')
    if fallback['building_code'].duplicated().any():
        raise RuntimeError('building_code duplicati in fallback')
    return (buildings, fallback, remaining, unmatched, pdir, fdir, paths)

def final_builder_determine_municipality_name(buildings, unmatched):
    vals = buildings['building_municipality_name'].dropna().astype(str).str.strip().unique().tolist()
    vals = [v for v in vals if v]
    if len(vals) == 1:
        return vals[0]
    if 'municipality_name' in unmatched.columns:
        vals = unmatched['municipality_name'].dropna().astype(str).str.strip().unique().tolist()
        vals = [v for v in vals if v]
        if len(vals) == 1:
            return vals[0]
    raise RuntimeError('Comune target non determinabile univocamente')

def final_builder_resolve_building(row):
    status = final_builder_clean(row.get('final_location_status'))
    lon, lat = (final_builder_fnum(row.get('final_longitude')), final_builder_fnum(row.get('final_latitude')))
    source = (final_builder_clean(row.get('final_geometry_source')) or '').lower()
    confidence = final_builder_clean(row.get('location_confidence'))
    if status in {'validated', 'accepted_address'} and lon is not None and (lat is not None):
        if 'osm' in source:
            resolution, method = ('site', 'osm_building_match')
        elif status == 'accepted_address':
            resolution, method = ('address', 'nominatim_address')
        else:
            resolution, method = ('address', 'nominatim_validated_by_osm')
        return (True, lon, lat, 'resolved_auto', method, resolution, confidence or 'high', 'fusion')
    fb_status = final_builder_clean(row.get('fallback_status'))
    lon, lat = (final_builder_fnum(row.get('candidate_longitude')), final_builder_fnum(row.get('candidate_latitude')))
    fb_conf = final_builder_clean(row.get('fallback_confidence'))
    if fb_status in {'address_candidate', 'street_anchor_candidate'} and lon is not None and (lat is not None):
        resolution = 'address' if fb_status == 'address_candidate' else 'street_anchor'
        return (True, lon, lat, 'resolved_auto', 'osm_street_fallback', resolution, fb_conf or ('high' if resolution == 'address' else 'medium'), 'street_fallback')
    return (False, None, None, 'review' if status == 'review' else 'unresolved', None, None, confidence or 'unresolved', 'unresolved')

def final_builder_build_building_rows(buildings, fallback, a, muni):
    use_cols = [c for c in ['building_code', 'fallback_status', 'fallback_confidence', 'candidate_longitude', 'candidate_latitude', 'candidate_resolution', 'candidate_display_name', 'matched_osm_street', 'street_match_score', 'street_match_margin'] if c in fallback.columns]
    merged = buildings.merge(fallback[use_cols], on='building_code', how='left', validate='one_to_one')
    rows = []
    for _, row in merged.sort_values('building_code').iterrows():
        code = str(row['building_code'])
        codes = final_builder_pick_list(row, 'linked_school_codes')
        names = final_builder_pick_list(row, 'linked_school_names')
        usable, lon, lat, rstatus, method, cres, conf, stage = final_builder_resolve_building(row)
        provenance = {'source_stage': stage, 'fusion_status': final_builder_clean(row.get('final_location_status')), 'fusion_geometry_source': final_builder_clean(row.get('final_geometry_source')), 'osm_match_status': final_builder_clean(row.get('osm_match_status')), 'geocoder_status': final_builder_clean(row.get('geocoder_status')), 'fallback_status': final_builder_clean(row.get('fallback_status')), 'fallback_matched_osm_street': final_builder_clean(row.get('matched_osm_street'))}
        rows.append({'school_site_id': f'MIMB:{code}', 'category': 'education', 'subcategory': 'state_school_building', 'site_record_type': 'physical_building', 'name': names[0] if names else f'School building {code}', 'municipality_code': a.municipality_code, 'municipality_name': muni, 'address': final_builder_clean(row.get('official_building_address')), 'postal_code': final_builder_clean(row.get('building_postal_code')), 'longitude': lon, 'latitude': lat, 'coordinate_origin': 'automatic' if usable else None, 'location_method': method, 'coordinate_resolution': cres, 'confidence': conf, 'resolution_status': rstatus, 'usable_for_accessibility': usable, 'source_system': 'MIM+OSM/Nominatim', 'source_dataset': f'MIM buildings {a.building_year} + automatic spatial resolution', 'source_record_id': code, 'reference_period': a.school_year, 'building_code': code, 'linked_school_codes': final_builder_jlist(codes), 'linked_school_names': final_builder_jlist(names), 'provenance_json': json.dumps(provenance, ensure_ascii=False, sort_keys=True)})
    return rows

def final_builder_normalize_remaining(row):
    status = final_builder_clean(row.get('resolution_status'))
    lon, lat = (final_builder_fnum(row.get('longitude')), final_builder_fnum(row.get('latitude')))
    accepted = {'accepted_osm', 'accepted_peer_site', 'accepted_address', 'accepted_site', 'street_anchor_candidate'}
    usable = status in accepted and lon is not None and (lat is not None)
    if not usable:
        return (False, None, None, 'review' if status == 'review' else 'unresolved', final_builder_clean(row.get('resolution_method')), None, final_builder_clean(row.get('confidence')) or 'low')
    if status in {'accepted_osm', 'accepted_peer_site', 'accepted_site'}:
        resolution = final_builder_clean(row.get('coordinate_resolution')) or 'site'
    elif status == 'accepted_address':
        resolution = final_builder_clean(row.get('coordinate_resolution')) or 'address'
    else:
        resolution = 'street_anchor'
    return (True, lon, lat, 'resolved_auto', final_builder_clean(row.get('resolution_method')), resolution, final_builder_clean(row.get('confidence')) or ('medium' if resolution == 'street_anchor' else 'high'))

def final_builder_build_remaining_rows(remaining, a, muni):
    work = remaining.copy()
    norm = [final_builder_normalize_remaining(row) for _, row in work.iterrows()]
    for i, col in enumerate(['_usable', '_lon', '_lat', '_status', '_method', '_resolution', '_confidence']):
        work[col] = [x[i] for x in norm]
    keys = []
    for _, row in work.iterrows():
        if final_builder_truth(row['_usable']) and row['_resolution'] in {'site', 'address'} and (str(row['_confidence']).lower() == 'high'):
            keys.append(('exact_site', round(float(row['_lon']), 6), round(float(row['_lat']), 6)))
        else:
            keys.append(('record', str(row['school_code'])))
    work['_site_group_key'] = keys
    rows = []
    for key, group in work.groupby('_site_group_key', sort=False):
        first = group.iloc[0]
        codes = group['school_code'].dropna().astype(str).str.strip().tolist()
        names = group['school_name'].dropna().astype(str).str.strip().tolist()
        addresses = group['school_address'].dropna().astype(str).str.strip().tolist()
        registry_types = group['registry_type'].dropna().astype(str).str.lower().str.strip().unique().tolist()
        subcategory = 'paritary_school' if registry_types and all((x == 'paritary' for x in registry_types)) else 'state_school_without_building_registry'
        school_site_id = final_builder_deterministic_id('MIMS', [a.municipality_code, key[1], key[2]]) if key[0] == 'exact_site' else f'MIMS:{codes[0]}'
        usable = final_builder_truth(first['_usable'])
        provenance = {'original_resolution_statuses': group['resolution_status'].fillna('').astype(str).tolist(), 'matched_names': group['matched_name'].fillna('').astype(str).tolist() if 'matched_name' in group.columns else []}
        rows.append({'school_site_id': school_site_id, 'category': 'education', 'subcategory': subcategory, 'site_record_type': 'school_service_without_building_relation', 'name': names[0] if len(names) == 1 else ' / '.join(names), 'municipality_code': a.municipality_code, 'municipality_name': muni, 'address': addresses[0] if addresses else None, 'postal_code': None, 'longitude': final_builder_fnum(first['_lon']) if usable else None, 'latitude': final_builder_fnum(first['_lat']) if usable else None, 'coordinate_origin': 'automatic' if usable else None, 'location_method': final_builder_clean(first['_method']), 'coordinate_resolution': final_builder_clean(first['_resolution']), 'confidence': final_builder_clean(first['_confidence']), 'resolution_status': final_builder_clean(first['_status']), 'usable_for_accessibility': usable, 'source_system': 'MIM+OSM/Nominatim', 'source_dataset': f'MIM schools {a.school_year} + residual-service resolution', 'source_record_id': final_builder_jlist(codes), 'reference_period': a.school_year, 'building_code': None, 'linked_school_codes': final_builder_jlist(codes), 'linked_school_names': final_builder_jlist(names), 'provenance_json': json.dumps(provenance, ensure_ascii=False, sort_keys=True)})
    return rows

def final_builder_to_geodataframe(df):
    geometry = []
    for lon, lat in zip(df['longitude'], df['latitude']):
        lon, lat = (final_builder_fnum(lon), final_builder_fnum(lat))
        geometry.append(Point(lon, lat) if lon is not None and lat is not None else None)
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs='EPSG:4326')

def final_builder_qa(final, buildings, unmatched):
    if final['school_site_id'].duplicated().any():
        raise RuntimeError('school_site_id duplicati')
    usable = final['usable_for_accessibility'].fillna(False).astype(bool)
    bad = usable & (final['longitude'].isna() | final['latitude'].isna() | final.geometry.isna())
    if bad.any():
        raise RuntimeError('Siti usable_for_accessibility senza coordinate')
    building_rows = final[final['site_record_type'] == 'physical_building']
    if len(building_rows) != len(buildings):
        raise RuntimeError(f'Edifici persi: {len(building_rows)} != {len(buildings)}')
    excluded = set(unmatched.loc[~unmatched['creates_public_school_site'].map(final_builder_truth), 'school_code'].dropna().astype(str))
    represented = set()
    for value in final['linked_school_codes']:
        represented.update(final_builder_parse_list(value))
    leaked = excluded & represented
    if leaked:
        raise RuntimeError('Servizi speciali esclusi presenti nel layer finale: ' + ', '.join(sorted(leaked)))

def final_builder_main():
    a = final_builder_parse_args()
    buildings, fallback, remaining, unmatched, pdir, fdir, paths = final_builder_load_inputs(a)
    muni = final_builder_determine_municipality_name(buildings, unmatched)
    final = pd.DataFrame(final_builder_build_building_rows(buildings, fallback, a, muni) + final_builder_build_remaining_rows(remaining, a, muni))
    final = final_builder_to_geodataframe(final)
    final_builder_qa(final, buildings, unmatched)
    parquet_path = pdir / f'school_sites_{a.school_year}.parquet'
    csv_path = fdir / f'school_sites_{a.school_year}.csv'
    exclusions_path = fdir / f'school_sites_excluded_{a.school_year}.csv'
    manifest_path = fdir / f'school_sites_{a.school_year}_manifest.json'
    final.to_parquet(parquet_path, index=False)
    csv_df = pd.DataFrame(final.drop(columns='geometry'))
    csv_df['geometry_wkt'] = final.geometry.apply(lambda g: g.wkt if g is not None else None)
    csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    excluded = unmatched.loc[~unmatched['creates_public_school_site'].map(final_builder_truth)].copy()
    excluded.to_csv(exclusions_path, index=False, encoding='utf-8-sig')
    usable = final['usable_for_accessibility'].fillna(False).astype(bool)
    stats = {'municipality_code': a.municipality_code, 'municipality_name': muni, 'school_year': a.school_year, 'building_year': a.building_year, 'inputs': {k: str(v) for k, v in paths.items()}, 'final_rows': int(len(final)), 'usable_rows': int(usable.sum()), 'unusable_rows': int((~usable).sum()), 'physical_building_rows': int((final['site_record_type'] == 'physical_building').sum()), 'physical_building_usable': int(((final['site_record_type'] == 'physical_building') & usable).sum()), 'remaining_service_rows': int((final['site_record_type'] == 'school_service_without_building_relation').sum()), 'remaining_service_usable': int(((final['site_record_type'] == 'school_service_without_building_relation') & usable).sum()), 'excluded_special_unmatched_records': int(len(excluded)), 'by_subcategory': {str(k): int(v) for k, v in final['subcategory'].value_counts(dropna=False).items()}, 'by_resolution_status': {str(k): int(v) for k, v in final['resolution_status'].value_counts(dropna=False).items()}, 'by_coordinate_resolution': {str(k): int(v) for k, v in final['coordinate_resolution'].value_counts(dropna=False).items()}, 'by_confidence': {str(k): int(v) for k, v in final['confidence'].value_counts(dropna=False).items()}}
    manifest_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n====================================')
    print(' GENERALIZED FINAL EDUCATION BUILDER')
    print('====================================')
    print(f'Comune: {muni} ({a.municipality_code})')
    print(f'Final rows: {len(final)}')
    print(f'Usable: {int(usable.sum())}/{len(final)} ({100 * usable.mean():.2f}%)')
    print(f"Physical buildings: {stats['physical_building_rows']}")
    print(f"Physical buildings usable: {stats['physical_building_usable']}")
    print(f"Remaining-service sites: {stats['remaining_service_rows']}")
    print(f"Remaining-service sites usable: {stats['remaining_service_usable']}")
    print(f"Excluded special unmatched: {stats['excluded_special_unmatched_records']}")
    print('\nResolution status:')
    print(final['resolution_status'].value_counts(dropna=False).to_string())
    print('\nCoordinate resolution:')
    print(final['coordinate_resolution'].value_counts(dropna=False).to_string())
    print('\nSubcategory:')
    print(final['subcategory'].value_counts(dropna=False).to_string())
    print('\n=== OUTPUT ===')
    for path in (parquet_path, csv_path, exclusions_path, manifest_path):
        print(f'✓ {path}')

# ============================================================================
# PUBLIC CLI
# ============================================================================

def parse_args():
    parser = _cli_argparse.ArgumentParser(
        description=(
            "Canonical Education transformation pipeline. "
            "Fusion, unmatched classification, street fallback, "
            "remaining-service resolution and final canonical layer."
        )
    )

    parser.add_argument(
        "--step",
        choices=(
            "prepare",
            "building-fusion",
            "classify-unmatched",
            "street-fallback",
            "remaining-services",
            "finalize",
        ),
        default="prepare",
        help=(
            "prepare esegue l'intera trasformazione Education successiva "
            "al matching; gli altri valori eseguono un singolo sottostep."
        ),
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre.",
    )

    parser.add_argument(
        "--school-year",
        default=_DEFAULT_CONFIG.school_year,
    )

    parser.add_argument(
        "--building-year",
        default=_DEFAULT_CONFIG.building_year,
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
        help="Ignora le cache Nominatim nei sottostep che le supportano.",
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    args.school_year = str(
        args.school_year
    ).strip()

    args.building_year = str(
        args.building_year
    ).strip()

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "municipality-code deve avere esattamente 6 cifre."
        )

    return args


def _run_legacy_cli(
    main_function,
    argv,
):
    previous_argv = list(
        _sys.argv
    )

    try:
        _sys.argv = [
            f"{__file__}:{main_function.__name__}",
            *argv,
        ]
        main_function()
    finally:
        _sys.argv = previous_argv


def _run_building_fusion(args):
    _run_legacy_cli(
        building_fusion_main,
        [
            "--municipality-code",
            args.municipality_code,
            "--building-year",
            args.building_year,
        ],
    )


def _run_classification(args):
    _run_legacy_cli(
        classification_main,
        [
            "--municipality-code",
            args.municipality_code,
            "--school-year",
            args.school_year,
            "--building-year",
            args.building_year,
        ],
    )


def _run_street_fallback(args):
    argv = [
        "--municipality-code",
        args.municipality_code,
        "--school-year",
        args.school_year,
        "--building-year",
        args.building_year,
        "--top-k-streets",
        str(args.top_k_streets),
        "--auto-street-threshold",
        str(args.auto_street_threshold),
        "--review-street-threshold",
        str(args.review_street_threshold),
        "--min-margin",
        str(args.min_margin),
        "--pause-seconds",
        str(args.pause_seconds),
    ]

    if args.refresh:
        argv.append("--refresh")

    _run_legacy_cli(
        street_fallback_main,
        argv,
    )


def _run_remaining_services(args):
    argv = [
        "--municipality-code",
        args.municipality_code,
        "--school-year",
        args.school_year,
        "--pause-seconds",
        str(args.pause_seconds),
    ]

    if args.refresh:
        argv.append("--refresh")

    _run_legacy_cli(
        remaining_services_main,
        argv,
    )


def _run_finalize(args):
    _run_legacy_cli(
        final_builder_main,
        [
            "--municipality-code",
            args.municipality_code,
            "--school-year",
            args.school_year,
            "--building-year",
            args.building_year,
        ],
    )


def main():
    args = parse_args()

    if args.step == "prepare":
        _run_building_fusion(args)
        _run_classification(args)
        _run_street_fallback(args)
        _run_remaining_services(args)
        _run_finalize(args)
        return

    runners = {
        "building-fusion": _run_building_fusion,
        "classify-unmatched": _run_classification,
        "street-fallback": _run_street_fallback,
        "remaining-services": _run_remaining_services,
        "finalize": _run_finalize,
    }

    runners[args.step](args)


if __name__ == "__main__":
    main()
