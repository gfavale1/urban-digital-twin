from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.accessibility_contracts import (
    compute_accessibility_from_canonical,
    request_from_service_spec,
)
from analysis.network_graph import build_routing_graph
from core.analysis_spec import TransportMode, default_service_specs
from core.schema_v2 import (
    ORIGIN_V2,
    SERVICE_V2,
    has_service_column,
    opportunity_count_column,
    population_column_for_selector,
    population_coverage_column,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test reale dell'accessibilita drive Canonical v2 su rete OSM "
            "con free_flow_travel_time_s. Questo e un integration smoke di Phase B4 "
            "e non ancora un output finale di tesi."
        )
    )
    parser.add_argument("--municipality-code", default="034027")
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument(
        "--connector-speed-kph",
        type=float,
        default=None,
        help=(
            "Velocita tecnica usata solo per convertire le distanze di snapping "
            "in tempo. Se omessa, usa la mediana speed_kph del grafo drive. "
            "Non rappresenta ancora una policy metodologica finale."
        ),
    )
    args = parser.parse_args()
    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    args.health_reference_date = pd.Timestamp(args.health_reference_date).normalize()
    if args.connector_speed_kph is not None:
        if (
            not np.isfinite(args.connector_speed_kph)
            or args.connector_speed_kph <= 0
        ):
            raise ValueError("--connector-speed-kph deve essere finita e > 0.")
    return args


def _input_paths(args):
    health_label = args.health_reference_date.strftime("%Y%m%d")
    return {
        "origins": (
            ROOT
            / "data"
            / "features"
            / "accessibility"
            / args.municipality_code
            / f"population_network_origins_{args.census_year}.parquet"
        ),
        "services": (
            ROOT
            / "data"
            / "processed"
            / "services"
            / args.municipality_code
            / f"service_entities_v2_{args.school_year}_{health_label}.parquet"
        ),
        "attachments": (
            ROOT
            / "data"
            / "processed"
            / "network_attachments"
            / args.municipality_code
            / (
                f"network_attachments_v2_drive_{args.census_year}_"
                f"{args.school_year}_{health_label}.parquet"
            )
        ),
        "nodes": (
            ROOT
            / "data"
            / "processed"
            / "osm"
            / args.municipality_code
            / "drive_nodes.parquet"
        ),
        "edges": (
            ROOT
            / "data"
            / "processed"
            / "osm"
            / args.municipality_code
            / "drive_edges.parquet"
        ),
    }


def _assert_result_invariants(result, origins, request):
    output = result.origins
    thresholds = tuple(request.thresholds_min)

    if len(output) != len(origins):
        raise AssertionError(
            "Origin row count changed during drive accessibility computation."
        )
    if output["origin_id"].duplicated().any():
        raise AssertionError(
            "AccessibilityOriginV2 contains duplicate origin_id values."
        )

    population_column = population_column_for_selector(
        request.population_selector
    )
    expected_population = float(
        pd.to_numeric(origins[population_column], errors="raise").sum()
    )
    actual_population = float(result.summary["target_population_total"])
    if not np.isclose(
        expected_population,
        actual_population,
        atol=1e-9,
        rtol=1e-12,
    ):
        raise AssertionError(
            "Target population mismatch: "
            f"expected={expected_population}, actual={actual_population}"
        )

    previous_counts = None
    previous_has = None
    nearest = pd.to_numeric(
        output["nearest_service_time_min"],
        errors="coerce",
    )

    for threshold in thresholds:
        count_col = opportunity_count_column(threshold)
        has_col = has_service_column(threshold)
        coverage_col = population_coverage_column(threshold)

        counts = output[count_col].to_numpy(dtype=int)
        has_service = output[has_col].astype(bool).to_numpy()

        if (counts < 0).any():
            raise AssertionError(
                f"Negative opportunity counts at threshold={threshold}."
            )
        if not np.array_equal(has_service, counts > 0):
            raise AssertionError(
                f"has_service/count equivalence failed at {threshold} min."
            )

        nearest_implies = (
            nearest.notna() & (nearest <= threshold)
        ).to_numpy()
        if not np.array_equal(nearest_implies, has_service):
            raise AssertionError(
                f"nearest<=threshold iff has_service failed at {threshold} min."
            )

        if previous_counts is not None and (counts < previous_counts).any():
            raise AssertionError(
                "Opportunity counts are not monotonic across thresholds."
            )
        if previous_has is not None and np.any(previous_has & ~has_service):
            raise AssertionError(
                "has_service is not monotonic across thresholds."
            )

        coverage = result.summary[coverage_col]
        if coverage is not None and not (0.0 <= float(coverage) <= 1.0):
            raise AssertionError(
                f"Coverage outside [0,1] at threshold={threshold}."
            )

        target = output["target_population"].to_numpy(dtype=float)
        denominator = float(target.sum())
        expected_coverage = (
            float(target[has_service].sum()) / denominator
            if denominator > 0
            else None
        )

        if expected_coverage is None:
            if coverage is not None:
                raise AssertionError(
                    "Expected null coverage for zero target population."
                )
        elif not np.isclose(
            float(coverage),
            expected_coverage,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise AssertionError(
                f"Coverage recomputation failed at threshold={threshold}."
            )

        previous_counts = counts
        previous_has = has_service

    reachable = output["reachability_status"].astype(str).eq("reachable")
    reachable_times = nearest.loc[reachable]

    if reachable_times.isna().any() or (reachable_times < 0).any():
        raise AssertionError(
            "Reachable origins need non-negative nearest_service_time_min."
        )
    if nearest.loc[~reachable].notna().any():
        raise AssertionError(
            "Non-reachable origins must store null nearest_service_time_min."
        )


def _connector_speed_m_s(edges, explicit_kph):
    if explicit_kph is not None:
        return float(explicit_kph) / 3.6, float(explicit_kph), "cli_override"

    if "speed_kph" not in edges.columns:
        raise RuntimeError(
            "drive_edges.parquet manca speed_kph: "
            "impossibile derivare la velocita tecnica di snapping."
        )

    speed = pd.to_numeric(edges["speed_kph"], errors="coerce")
    valid = speed.loc[np.isfinite(speed) & (speed > 0)]
    if valid.empty:
        raise RuntimeError(
            "drive_edges.parquet non contiene speed_kph valide."
        )

    median_kph = float(valid.median())
    return median_kph / 3.6, median_kph, "graph_median_speed_kph"


def main():
    args = parse_args()
    paths = _input_paths(args)

    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    origins = pd.read_parquet(paths["origins"])
    services_v2 = pd.read_parquet(paths["services"])
    attachments = pd.read_parquet(paths["attachments"])
    nodes = pd.read_parquet(paths["nodes"])
    edges = pd.read_parquet(paths["edges"])

    ORIGIN_V2.validate_columns(origins.columns)
    SERVICE_V2.validate_columns(services_v2.columns)

    graph, graph_summary = build_routing_graph(
        nodes,
        edges,
        travel_time_weight="free_flow_travel_time_s",
        distance_weight="length_m",
    )

    connector_speed_m_s, connector_speed_kph, connector_source = (
        _connector_speed_m_s(
            edges,
            args.connector_speed_kph,
        )
    )

    drive_specs = [
        spec
        for spec in default_service_specs()
        if spec.enabled and TransportMode.DRIVE in spec.modes
    ]
    if not drive_specs:
        raise AssertionError("No enabled drive service specifications found.")

    print("\n====================================")
    print(" CANONICAL DRIVE ACCESSIBILITY V2 SMOKE")
    print("====================================")
    print("Municipality:", args.municipality_code)
    print("Origins:", len(origins))
    print("ServiceV2 entities:", len(services_v2))
    print("Drive graph nodes:", graph.number_of_nodes())
    print("Drive graph edges:", graph.number_of_edges())
    print(
        "Collapsed directed parallel edges:",
        graph_summary["duplicate_directed_pairs_collapsed"],
    )
    print(
        "Connector speed [km/h]:",
        f"{connector_speed_kph:.2f}",
        f"({connector_source})",
    )
    print(
        "NOTE: connector speed is a Phase-B4 integration convention only; "
        "this smoke is not a final thesis output."
    )
    print(
        "Travel-time semantics: OSM-derived free-flow potential accessibility."
    )

    for service_spec in drive_specs:
        request = request_from_service_spec(
            service_spec,
            TransportMode.DRIVE,
            travel_time_weight="free_flow_travel_time_s",
            off_network_speed_m_s=connector_speed_m_s,
            distance_weight="length_m",
        )

        result = compute_accessibility_from_canonical(
            graph,
            origins,
            services_v2,
            attachments,
            request,
        )
        _assert_result_invariants(
            result,
            origins,
            request,
        )

        status_counts = (
            result.origins["reachability_status"]
            .value_counts(dropna=False)
        )
        reachable_count = int(
            status_counts.get("reachable", 0)
        )
        origin_not_snapped_count = int(
            status_counts.get("origin_not_snapped", 0)
        )

        if origin_not_snapped_count == len(origins):
            raise AssertionError(
                f"{service_spec.service_type.value}: all origins are "
                "origin_not_snapped; drive routing smoke would be vacuous."
            )

        if (
            result.summary["service_count_routable"] > 0
            and reachable_count == 0
        ):
            raise AssertionError(
                f"{service_spec.service_type.value}: routable services exist "
                "but no origin is reachable."
            )

        if result.summary["service_count_total"] == 0:
            non_unsnapped = result.origins.loc[
                (
                    result.origins["reachability_status"].astype(str)
                    != "origin_not_snapped"
                ),
                "reachability_status",
            ].astype(str)
            if not non_unsnapped.eq("service_unavailable").all():
                raise AssertionError(
                    f"{service_spec.service_type.value}: absent service type "
                    "must report service_unavailable for snapped origins."
                )

        print(
            "\n---",
            service_spec.service_type.value.upper(),
            "---",
        )
        print(
            "Population selector:",
            service_spec.population_selector.value,
        )
        print(
            "Target population:",
            f"{result.summary['target_population_total']:.6f}",
        )
        print(
            "Services total/routable:",
            (
                f"{result.summary['service_count_total']}/"
                f"{result.summary['service_count_routable']}"
            ),
        )
        print("Reachability status:")
        print(status_counts.to_string())

        nearest = pd.to_numeric(
            result.origins["nearest_service_time_min"],
            errors="coerce",
        ).dropna()
        if not nearest.empty:
            print(
                "Nearest service time [min] "
                "median/mean/max:",
                (
                    f"{nearest.median():.2f} / "
                    f"{nearest.mean():.2f} / "
                    f"{nearest.max():.2f}"
                ),
            )

        for threshold in request.thresholds_min:
            coverage = result.summary[
                population_coverage_column(threshold)
            ]
            coverage_text = (
                "null"
                if coverage is None
                else f"{100.0 * float(coverage):.2f}%"
            )
            print(
                f"{threshold:>2} min coverage:",
                coverage_text,
            )

    print("\n====================================")
    print("RESULT: CANONICAL DRIVE V2 SMOKE PASSED ✅")
    print("====================================")


if __name__ == "__main__":
    main()
