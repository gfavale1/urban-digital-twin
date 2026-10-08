from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis import compute_walking_accessibility as legacy
from analysis.accessibility_contracts import (
    compute_accessibility_from_canonical,
    request_from_service_spec,
)
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
            "Smoke test reale dei contratti Canonical v2 sul grafo walking legacy. "
            "Non rappresenta ancora il baseline metodologico finale a 0.9 m/s."
        )
    )
    parser.add_argument("--municipality-code", default="034027")
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--walking-speed-m-s", type=float, default=1.4)
    args = parser.parse_args()
    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    args.health_reference_date = pd.Timestamp(args.health_reference_date).normalize()
    return args


def _origin_attachments(origins: pd.DataFrame) -> pd.DataFrame:
    if "fallback_snap_distance_m" not in origins.columns:
        raise RuntimeError(
            "population origins legacy bridge: fallback_snap_distance_m mancante."
        )
    if "network_node_id" not in origins.columns:
        raise RuntimeError("population origins: network_node_id mancante.")

    snap = pd.to_numeric(origins["fallback_snap_distance_m"], errors="coerce").fillna(0.0)
    node = origins["network_node_id"].copy()
    snapped = node.notna()

    if (snap < 0).any():
        raise RuntimeError("Origin legacy bridge con snap distance negativa.")

    return pd.DataFrame(
        {
            "entity_id": origins["origin_id"].astype(str),
            "entity_kind": "origin",
            "mode": TransportMode.WALK.value,
            "node_id": node.where(snapped, None),
            "snapped": snapped.astype(bool),
            "snap_distance_m": snap.where(snapped, np.nan),
            "attachment_quality": "legacy_bridge",
        }
    )


def _service_attachments(
    services_v2: pd.DataFrame,
    legacy_network_services: pd.DataFrame,
) -> pd.DataFrame:
    required_legacy = {"service_site_id", "network_node_id", "snap_distance_m"}
    missing = sorted(required_legacy - set(legacy_network_services.columns))
    if missing:
        raise RuntimeError(f"Legacy service network nodes: colonne mancanti {missing}")

    if legacy_network_services["service_site_id"].astype(str).duplicated().any():
        raise RuntimeError("Legacy service network nodes: service_site_id duplicati.")

    bridge = legacy_network_services[
        ["service_site_id", "network_node_id", "snap_distance_m"]
    ].copy()
    bridge["service_site_id"] = bridge["service_site_id"].astype(str)
    bridge = bridge.rename(columns={"service_site_id": "legacy_service_site_id"})

    service_rows = services_v2[["service_id", "legacy_service_site_id"]].copy()
    service_rows["service_id"] = service_rows["service_id"].astype(str)
    service_rows["legacy_service_site_id"] = service_rows["legacy_service_site_id"].astype("string")
    service_rows = service_rows.merge(
        bridge,
        on="legacy_service_site_id",
        how="left",
        validate="many_to_one",
    )

    snap = pd.to_numeric(service_rows["snap_distance_m"], errors="coerce")
    non_negative_snap = snap.fillna(-1.0).ge(0.0)
    snapped = service_rows["network_node_id"].notna() & snap.notna() & non_negative_snap

    return pd.DataFrame(
        {
            "entity_id": service_rows["service_id"],
            "entity_kind": "service",
            "mode": TransportMode.WALK.value,
            "node_id": service_rows["network_node_id"].where(snapped, None),
            "snapped": snapped.astype(bool),
            "snap_distance_m": snap.where(snapped, np.nan),
            "attachment_quality": np.where(snapped, "legacy_bridge", "legacy_unavailable"),
        }
    )


def _assert_result_invariants(result, origins, service_spec):
    output = result.origins
    thresholds = tuple(service_spec.thresholds_min_by_mode[TransportMode.WALK.value])

    if len(output) != len(origins):
        raise AssertionError("Origin row count changed during canonical accessibility computation.")
    if output["origin_id"].duplicated().any():
        raise AssertionError("AccessibilityOriginV2 contains duplicate origin_id values.")

    population_column = population_column_for_selector(service_spec.population_selector)
    expected_population = float(pd.to_numeric(origins[population_column], errors="raise").sum())
    actual_population = float(result.summary["target_population_total"])
    if not np.isclose(expected_population, actual_population, atol=1e-9, rtol=1e-12):
        raise AssertionError(
            f"Target population mismatch: expected={expected_population}, actual={actual_population}"
        )

    previous_counts = None
    previous_has = None
    nearest = pd.to_numeric(output["nearest_service_time_min"], errors="coerce")

    for threshold in thresholds:
        count_col = opportunity_count_column(threshold)
        has_col = has_service_column(threshold)
        coverage_col = population_coverage_column(threshold)

        counts = output[count_col].to_numpy(dtype=int)
        has_service = output[has_col].astype(bool).to_numpy()

        if (counts < 0).any():
            raise AssertionError(f"Negative opportunity counts at threshold={threshold}.")
        if not np.array_equal(has_service, counts > 0):
            raise AssertionError(f"has_service/count equivalence failed at {threshold} min.")

        nearest_implies = (nearest.notna() & (nearest <= threshold)).to_numpy()
        if not np.array_equal(nearest_implies, has_service):
            raise AssertionError(f"nearest<=threshold iff has_service failed at {threshold} min.")

        if previous_counts is not None and (counts < previous_counts).any():
            raise AssertionError("Opportunity counts are not monotonic across thresholds.")
        if previous_has is not None and np.any(previous_has & ~has_service):
            raise AssertionError("has_service is not monotonic across thresholds.")

        coverage = result.summary[coverage_col]
        if coverage is not None and not (0.0 <= float(coverage) <= 1.0):
            raise AssertionError(f"Coverage outside [0,1] at threshold={threshold}.")

        target = output["target_population"].to_numpy(dtype=float)
        denominator = float(target.sum())
        expected_coverage = (
            float(target[has_service].sum()) / denominator if denominator > 0 else None
        )
        if expected_coverage is None:
            if coverage is not None:
                raise AssertionError("Expected null coverage for zero target population.")
        elif not np.isclose(float(coverage), expected_coverage, atol=1e-12, rtol=1e-12):
            raise AssertionError(f"Coverage recomputation failed at threshold={threshold}.")

        previous_counts = counts
        previous_has = has_service

    reachable = output["reachability_status"].astype(str).eq("reachable")
    reachable_times = nearest.loc[reachable]
    if reachable_times.isna().any() or (reachable_times < 0).any():
        raise AssertionError("Reachable origins need non-negative nearest_service_time_min.")
    if nearest.loc[~reachable].notna().any():
        raise AssertionError("Non-reachable origins must store null nearest_service_time_min.")


def main():
    args = parse_args()
    health_label = args.health_reference_date.strftime("%Y%m%d")

    legacy_args = SimpleNamespace(
        municipality_code=args.municipality_code,
        census_year=args.census_year,
        school_year=args.school_year,
        health_reference_date=args.health_reference_date,
        service_layer="enriched",
        walking_speed_m_s=args.walking_speed_m_s,
        thresholds_min=[10, 15, 20],
    )

    origins, legacy_network_services, nodes, edges, _ = legacy.load_inputs(legacy_args)
    graph, _ = legacy.build_graph(nodes, edges)

    service_v2_path = (
        ROOT
        / "data"
        / "processed"
        / "services"
        / args.municipality_code
        / f"service_entities_v2_{args.school_year}_{health_label}.parquet"
    )
    if not service_v2_path.exists():
        raise FileNotFoundError(service_v2_path)

    services_v2 = pd.read_parquet(service_v2_path)
    ORIGIN_V2.validate_columns(origins.columns)
    SERVICE_V2.validate_columns(services_v2.columns)

    if "legacy_service_site_id" not in services_v2.columns:
        raise RuntimeError("ServiceV2 migration bridge legacy_service_site_id mancante.")

    attachments = pd.concat(
        [
            _origin_attachments(origins),
            _service_attachments(services_v2, legacy_network_services),
        ],
        ignore_index=True,
    )

    origin_attachment_mask = attachments["entity_kind"].astype(str).eq("origin")
    snapped_origin_count = int(
        attachments.loc[origin_attachment_mask, "snapped"].astype(bool).sum()
    )
    if snapped_origin_count == 0:
        raise AssertionError(
            "Canonical smoke bridge produced zero snapped origins; routing smoke would be vacuous."
        )

    print("\n====================================")
    print(" CANONICAL ACCESSIBILITY V2 SMOKE")
    print("====================================")
    print("Municipality:", args.municipality_code)
    print("Origins:", len(origins))
    print("ServiceV2 entities:", len(services_v2))
    print("Walk graph nodes:", graph.number_of_nodes())
    print("Walk graph edges:", graph.number_of_edges())
    print("Compatibility walking speed [m/s]:", args.walking_speed_m_s)
    print("NOTE: compatibility smoke on legacy 1.4 m/s graph; not final 0.9 m/s thesis baseline.")

    walk_specs = [
        spec
        for spec in default_service_specs()
        if spec.enabled and TransportMode.WALK in spec.modes
    ]

    for service_spec in walk_specs:
        request = request_from_service_spec(
            service_spec,
            TransportMode.WALK,
            travel_time_weight="walking_time_s",
            off_network_speed_m_s=args.walking_speed_m_s,
            distance_weight="length_m",
        )
        result = compute_accessibility_from_canonical(
            graph,
            origins,
            services_v2,
            attachments,
            request,
        )
        _assert_result_invariants(result, origins, service_spec)

        status_counts = result.origins["reachability_status"].value_counts(dropna=False)
        reachable_count = int(status_counts.get("reachable", 0))
        origin_not_snapped_count = int(status_counts.get("origin_not_snapped", 0))

        if origin_not_snapped_count == len(origins):
            raise AssertionError(
                f"{service_spec.service_type.value}: all origins are origin_not_snapped; "
                "canonical routing integration is not actually being exercised."
            )

        if result.summary["service_count_routable"] > 0 and reachable_count == 0:
            raise AssertionError(
                f"{service_spec.service_type.value}: routable services exist but no origin is reachable."
            )

        if result.summary["service_count_total"] == 0:
            non_unsnapped = result.origins.loc[
                result.origins["reachability_status"].astype(str) != "origin_not_snapped",
                "reachability_status",
            ].astype(str)
            if not non_unsnapped.eq("service_unavailable").all():
                raise AssertionError(
                    f"{service_spec.service_type.value}: absent service type must report "
                    "service_unavailable for snapped origins."
                )

        print("\n---", service_spec.service_type.value.upper(), "---")
        print("Population selector:", service_spec.population_selector.value)
        print("Target population:", f"{result.summary['target_population_total']:.6f}")
        print(
            "Services total/routable:",
            f"{result.summary['service_count_total']}/{result.summary['service_count_routable']}",
        )
        print("Reachability status:")
        print(status_counts.to_string())
        for threshold in request.thresholds_min:
            coverage = result.summary[population_coverage_column(threshold)]
            if coverage is None:
                coverage_text = "null"
            else:
                coverage_text = f"{100.0 * float(coverage):.2f}%"
            print(f"{threshold:>2} min coverage: {coverage_text}")

    print("\n====================================")
    print("RESULT: CANONICAL V2 SMOKE PASSED ✅")
    print("====================================")


if __name__ == "__main__":
    main()
