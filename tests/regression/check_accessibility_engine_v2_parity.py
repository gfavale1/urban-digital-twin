"""Real-data regression harness: legacy walking engine vs generic v2 engine.

This is intentionally not named ``test_*.py`` because it depends on locally
materialised processed artifacts and therefore should not run as part of the
portable unit-test suite.
"""

from __future__ import annotations

import argparse
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis import compute_walking_accessibility as legacy
from analysis.accessibility_engine import AccessibilityEngineRequest, compute_accessibility
from core.analysis_spec import PopulationSelector, ServiceType, TransportMode


GROUP_MAPPING = {
    "school": ServiceType.LEGACY_EDUCATION_ALL,
    "pharmacy": ServiceType.PHARMACY,
    "hospital": ServiceType.HOSPITAL_ESTABLISHMENT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy walking accessibility with AccessibilityEngine v2."
    )
    parser.add_argument("--municipality-code", default="034027")
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--walking-speed-m-s", type=float, default=1.4)
    parser.add_argument("--thresholds-min", nargs="+", type=int, default=[10, 15, 20])
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    thresholds = tuple(sorted(set(int(v) for v in cli.thresholds_min)))

    args = SimpleNamespace(
        municipality_code=str(cli.municipality_code).strip().zfill(6),
        census_year=str(cli.census_year),
        school_year=str(cli.school_year),
        health_reference_date=pd.Timestamp(cli.health_reference_date),
        service_layer="enriched",
        walking_speed_m_s=float(cli.walking_speed_m_s),
        thresholds_min=list(thresholds),
    )

    # Known warnings are emitted by the frozen legacy oracle when comparing NaN
    # values. The v2 unit suite separately verifies that the new engine itself
    # does not emit these warnings.
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in less.*",
        category=RuntimeWarning,
        module=r"pandas\.core\.computation\.expressions",
    )

    origins_raw, services_raw, nodes, edges, _ = legacy.load_inputs(args)
    graph, _ = legacy.build_graph(nodes, edges)
    origins = legacy.prepare_origins(origins_raw)
    services = legacy.prepare_services(services_raw)
    legacy.validate_network_membership(graph, origins, services)

    if "origin_id" not in origins.columns:
        raise RuntimeError("origin_id missing: Demand Schema v2 is required.")

    groups = legacy.build_service_groups(services)
    reverse_graph = graph.reverse(copy=False)

    engine_origins = pd.DataFrame(
        {
            "origin_id": origins["origin_id"].astype(str),
            "network_node_id": origins["network_node_id"],
            "origin_snap_distance_m": origins["origin_snap_distance_m"],
            # Deliberately use the legacy denominator: this test isolates the
            # routing/metric refactor from population-schema changes.
            "population_total": origins["assigned_population"].astype(float),
        }
    )

    print("\n====================================")
    print(" ACCESSIBILITY ENGINE LEGACY ↔ V2")
    print("====================================")
    print("Municipality:", args.municipality_code)
    print("Origins:", len(origins))
    print("Services:", len(services))
    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())

    all_ok = True

    for group_name, service_type in GROUP_MAPPING.items():
        group_services = groups[group_name].copy()
        print(f"\n--- {group_name.upper()} ({len(group_services)} services) ---")

        legacy_metrics = legacy.calculate_group_accessibility_full(
            reverse_graph=reverse_graph,
            origins=origins,
            services=group_services,
            thresholds_min=list(thresholds),
            walking_speed_m_s=args.walking_speed_m_s,
        )
        legacy_result = legacy.attach_group_metrics(
            origins=origins,
            group_name=group_name,
            metrics=legacy_metrics,
            thresholds_min=list(thresholds),
        )

        engine_services = pd.DataFrame(
            {
                "service_id": group_services["service_site_id"].astype(str),
                "service_type": service_type.value,
                "network_node_id": group_services["network_node_id"],
                "snap_distance_m": group_services["snap_distance_m"].astype(float),
            }
        )
        request = AccessibilityEngineRequest(
            service_type=service_type,
            mode=TransportMode.WALK,
            thresholds_min=thresholds,
            population_selector=PopulationSelector.TOTAL,
            travel_time_weight="walking_time_s",
            off_network_speed_m_s=args.walking_speed_m_s,
            distance_weight="length_m",
        )
        v2 = compute_accessibility(graph, engine_origins, engine_services, request)
        v2_result = v2.origins

        legacy_reachable = legacy_result[f"{group_name}_reachable"].astype(bool).to_numpy()
        v2_reachable = v2_result["reachability_status"].eq("reachable").to_numpy()
        reachability_equal = np.array_equal(legacy_reachable, v2_reachable)

        legacy_time = pd.to_numeric(
            legacy_result[f"{group_name}_nearest_time_min"], errors="coerce"
        ).to_numpy(dtype=float)
        v2_time = pd.to_numeric(
            v2_result["nearest_service_time_min"], errors="coerce"
        ).to_numpy(dtype=float)
        time_equal = np.allclose(
            legacy_time, v2_time, atol=1e-10, rtol=1e-10, equal_nan=True
        )

        legacy_distance = pd.to_numeric(
            legacy_result[f"{group_name}_nearest_distance_m"], errors="coerce"
        ).to_numpy(dtype=float)
        v2_distance = pd.to_numeric(
            v2_result["nearest_service_distance_m"], errors="coerce"
        ).to_numpy(dtype=float)
        distance_equal = np.allclose(
            legacy_distance, v2_distance, atol=1e-8, rtol=1e-10, equal_nan=True
        )

        legacy_service = (
            legacy_result[f"{group_name}_nearest_service_site_id"]
            .astype("string")
            .fillna("<NA>")
            .to_numpy()
        )
        v2_service = (
            v2_result["nearest_service_id"]
            .astype("string")
            .fillna("<NA>")
            .to_numpy()
        )
        service_equal = np.array_equal(legacy_service, v2_service)

        print("Reachability:", reachability_equal)
        print("Nearest time:", time_equal)
        print("Nearest distance:", distance_equal)
        print("Nearest service:", service_equal)

        if not all((reachability_equal, time_equal, distance_equal, service_equal)):
            all_ok = False

        population = origins["assigned_population"].astype(float).to_numpy()
        total_population = float(population.sum())

        for threshold in thresholds:
            legacy_count = legacy_result[
                f"{group_name}_services_within_{threshold}_min"
            ].to_numpy(dtype=int)
            v2_count = v2_result[
                f"opportunity_count_within_{threshold}_min"
            ].to_numpy(dtype=int)
            counts_equal = np.array_equal(legacy_count, v2_count)

            legacy_has = legacy_result[
                f"{group_name}_within_{threshold}_min"
            ].astype(bool).to_numpy()
            v2_has = v2_result[
                f"has_service_within_{threshold}_min"
            ].astype(bool).to_numpy()
            has_equal = np.array_equal(legacy_has, v2_has)

            legacy_coverage = float(population[legacy_has].sum() / total_population)
            v2_coverage = float(
                v2.summary[f"population_coverage_within_{threshold}_min"]
            )
            coverage_equal = np.isclose(
                legacy_coverage, v2_coverage, atol=1e-12, rtol=1e-12
            )

            print(
                f"{threshold:>2} min | counts={counts_equal} | "
                f"has_service={has_equal} | coverage={coverage_equal}"
            )
            if not all((counts_equal, has_equal, coverage_equal)):
                all_ok = False

    print("\n====================================")
    if not all_ok:
        print("RESULT: REGRESSION DIFFERENCE ❌")
        raise SystemExit(1)
    print("RESULT: EXACT REGRESSION PARITY ✅")
    print("====================================")


if __name__ == "__main__":
    main()
