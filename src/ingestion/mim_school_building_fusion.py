import argparse
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"

DEFAULT_BUILDING_YEAR = "202425"

GEOCODER_ADDRESS_STRONG = 85.0
GEOCODER_ADDRESS_REVIEW = 70.0
STRONG_AGREEMENT_M = 120.0
PLAUSIBLE_AGREEMENT_M = 250.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fusion V2 delle localizzazioni degli edifici scolastici "
            "con validazione semantica dell'indirizzo Nominatim."
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

    return value


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


def address_similarity(
    official,
    returned,
):
    left = normalize_address(
        official
    )

    right = normalize_address(
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


def as_float(value):
    if is_missing(value):
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


def haversine_m(
    lon1,
    lat1,
    lon2,
    lat2,
):
    values = [
        as_float(lon1),
        as_float(lat1),
        as_float(lon2),
        as_float(lat2),
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


def load_inputs(
    municipality_code,
    building_year,
):
    directory = (
        PROCESSED_MIM_DIR
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


def prepare_dataset(
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
            address_similarity(
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


def fuse_row(
    row,
    target_municipality_name,
):
    municipality = (
        clean_text(
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
        as_float(
            row.get(
                "geocoder_address_score"
            )
        )
        or 0.0
    )

    geocoder_lon = as_float(
        row.get(
            "geocoder_longitude"
        )
    )

    geocoder_lat = as_float(
        row.get(
            "geocoder_latitude"
        )
    )

    osm_status = row.get(
        "osm_match_status"
    )

    osm_lon = as_float(
        row.get(
            "osm_candidate_longitude"
        )
    )

    osm_lat = as_float(
        row.get(
            "osm_candidate_latitude"
        )
    )

    distance_m = haversine_m(
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
        >= GEOCODER_ADDRESS_STRONG
    )

    geocoder_review = (
        geocoder_has_point
        and geocoder_address_score
        >= GEOCODER_ADDRESS_REVIEW
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
            <= STRONG_AGREEMENT_M
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
            <= PLAUSIBLE_AGREEMENT_M
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
            <= PLAUSIBLE_AGREEMENT_M
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
                    "Geocoder dentro Matera con indirizzo restituito "
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


def build_final_dataset(
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
            clean_text(value)
            or ""
        ).upper()
        for value in geocoder[
            "building_municipality_name"
        ].dropna()
        if clean_text(value)
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

    merged = prepare_dataset(
        geocoder,
        osm,
    )

    fusion_rows = []

    for _, row in merged.iterrows():
        fusion_rows.append(
            fuse_row(
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


def save_output(
    dataframe,
    municipality_code,
    building_year,
):
    directory = (
        PROCESSED_MIM_DIR
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


def print_summary(
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
        "\nEdifici Matera:"
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


def main():
    args = parse_args()

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
    ) = load_inputs(
        args.municipality_code,
        args.building_year,
    )

    final = build_final_dataset(
        geocoder=geocoder,
        osm=osm,
    )

    save_output(
        final,
        args.municipality_code,
        args.building_year,
    )

    print_summary(
        final
    )


if __name__ == "__main__":
    main()
