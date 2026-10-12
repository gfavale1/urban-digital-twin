"""B8 read-only robustness to temporary removal of verified service destinations.

For each selected, already-routable service, run the unmodified V2 accessibility
engine after masking *only* that service's engine-facing network node. Never
promote unresolved services. The experiment is not a real closure forecast.

Default top-k uses population whose nearest baseline service is the target;
--all examines every routable destination (more expensive). Files untouched.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from analysis.accessibility_contracts import prepare_canonical_accessibility_inputs
from analysis.accessibility_engine import AccessibilityEngineRequest, compute_accessibility
from analysis.analyze_walking_speed_sensitivity import (
    DEFAULT_THRESHOLDS,
    inspect_city_inputs,
    reweight_graph,
    summarize_result,
)
from analysis.network_graph import build_routing_graph
from core.analysis_spec import PopulationSelector, ServiceType, TransportMode
from quality.preflight_real_service_scenario import ROOT

POLICY = "b8_verified_service_leave_one_out_robustness_v1"


def _thresholds(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(t) for t in values)
    if not result or tuple(sorted(set(result))) != result or any(t <= 0 for t in result):
        raise ValueError("Thresholds must be strictly increasing positive integers")
    return result


def _validated_speed(value: float) -> float:
    speed = float(value)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Walking speed must be finite and positive")
    return speed


def _nearest_population(result: Any, valid_ids: set[str]) -> dict[str, float]:
    """Count baseline assigned population without guessing about unreachable origins."""
    output = result.origins
    ids = output["nearest_service_id"]
    population = pd.to_numeric(output["target_population"], errors="raise")
    assigned = {id_: 0.0 for id_ in valid_ids}
    for sid, pop in zip(ids, population):
        if pd.isna(sid):
            continue
        sid = str(sid)
        if sid not in valid_ids:
            raise ValueError(f"Baseline route points to ineligible service {sid}")
        assigned[sid] += float(pop)
    return assigned


def _compare_removal(base: Any, changed: Any, *, thresholds: tuple[int, ...]) -> None:
    """Removing one destination cannot improve nearest times or coverage."""
    lhs = base.origins
    rhs = changed.origins
    if not lhs["origin_id"].equals(rhs["origin_id"]):
        raise ValueError("Removal changed origin order or identifiers")
    if not lhs["target_population"].equals(rhs["target_population"]):
        raise ValueError("Removal changed population")
    baseline_time = pd.to_numeric(lhs["nearest_service_time_min"], errors="coerce").to_numpy(dtype=float)
    scenario_time = pd.to_numeric(rhs["nearest_service_time_min"], errors="coerce").to_numpy(dtype=float)
    was_finite = np.isfinite(baseline_time)
    now_finite = np.isfinite(scenario_time)
    if np.any(now_finite & ~was_finite):
        raise ValueError("Removal made a previously unreachable origin reachable")
    if np.any(scenario_time[now_finite] < baseline_time[now_finite] - 1e-8):
        raise ValueError("Removal improved nearest travel time")
    for threshold in thresholds:
        key = f"opportunity_count_within_{threshold}_min"
        if (rhs[key].to_numpy() > lhs[key].to_numpy()).any():
            raise ValueError("Removal increased the cumulative opportunity count")
        column = f"population_coverage_within_{threshold}_min"
        before = base.summary[column]
        after = changed.summary[column]
        if before is not None and after is not None and after > before + 1e-10:
            raise ValueError("Removal increased coverage")


def analyze_removal_in_memory(
    graph: nx.Graph,
    origins: pd.DataFrame,
    qualified_services: pd.DataFrame,
    attachments: pd.DataFrame,
    *,
    service_type: ServiceType = ServiceType.PHARMACY,
    population_selector: PopulationSelector = PopulationSelector.TOTAL,
    walking_speed_m_s: float = 0.9,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
    top_k: int | None = 3,
) -> dict[str, Any]:
    """Compute baseline + independent one-at-a-time removal scenarios.

    The canonical service rows remain in each engine call; a selected verified
    service is temporarily masked at the routing interface. This is a model
    sensitivity test, not evidence of service closure or a new approval.
    """
    service_type = ServiceType(service_type)
    population_selector = PopulationSelector(population_selector)
    speed = _validated_speed(walking_speed_m_s)
    thresholds = _thresholds(thresholds)
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer or None for all services")
    selected = qualified_services.loc[
        qualified_services["service_type"].astype(str).eq(service_type.value)
    ].copy()
    if selected.empty:
        raise ValueError(f"No canonical services for {service_type.value}")
    selected_ids = set(selected["service_id"].astype(str))
    kinds = attachments["entity_kind"].astype(str)
    scoped = attachments.loc[
        kinds.eq("origin") | (kinds.eq("service") & attachments["entity_id"].astype(str).isin(selected_ids))
    ].copy()
    prepared = prepare_canonical_accessibility_inputs(
        origins, selected, scoped, mode=TransportMode.WALK,
    )
    destination = prepared.services
    eligible_mask = destination["routing_eligible"].eq(True)
    eligible_ids = destination.loc[eligible_mask, "service_id"].astype(str).tolist()
    if not eligible_ids or len(eligible_ids) != len(set(eligible_ids)):
        raise ValueError("Missing or duplicated verified routable service IDs")
    weighted_graph = reweight_graph(graph, speed)
    request = AccessibilityEngineRequest(
        service_type=service_type, mode=TransportMode.WALK,
        thresholds_min=thresholds, population_selector=population_selector,
        travel_time_weight="walking_time_s", off_network_speed_m_s=speed,
        distance_weight="length_m",
    )
    baseline = compute_accessibility(weighted_graph, prepared.origins, destination, request)
    base = summarize_result(baseline, thresholds)
    if base["service_count_routable"] != len(eligible_ids):
        raise ValueError("Routing gate and engine disagree on baseline eligible destinations")
    assigned = _nearest_population(baseline, set(eligible_ids))
    ranking = sorted(eligible_ids, key=lambda sid: (-assigned[sid], sid))
    selected_for_removal = ranking if top_k is None else ranking[:top_k]
    rows = []
    for service_id in selected_for_removal:
        altered = destination.copy(deep=True)
        target = altered["service_id"].astype(str).eq(service_id)
        if target.sum() != 1 or not bool(altered.loc[target, "routing_eligible"].iloc[0]):
            raise ValueError("Experiment can only remove exactly one verified destination")
        altered.loc[target, "network_node_id"] = None
        altered.loc[target, "snap_distance_m"] = float("nan")
        altered.loc[target, "routing_eligible"] = False
        altered.loc[target, "routing_exclusion_reason"] = "experimentally_removed"
        scenario = compute_accessibility(weighted_graph, prepared.origins, altered, request)
        _compare_removal(baseline, scenario, thresholds=thresholds)
        current = summarize_result(scenario, thresholds)
        if (current["service_count_total"] != base["service_count_total"]
                or current["service_count_routable"] != base["service_count_routable"] - 1
                or not math.isclose(current["target_population"], base["target_population"], rel_tol=1e-12, abs_tol=1e-8)):
            raise ValueError("Removal changed the canonical catalog or total population")
        coverage_drop = {}
        for t in thresholds:
            before = base["coverage_by_minutes"][str(t)]
            after = current["coverage_by_minutes"][str(t)]
            coverage_drop[str(t)] = None if before is None or after is None else 100 * (before - after)
        rows.append({
            "removed_service_id": service_id,
            "baseline_nearest_population": assigned[service_id],
            "scenario": current,
            "coverage_loss_percentage_points": coverage_drop,
            "lost_reachable_population": base["reachable_population"] - current["reachable_population"],
        })
    if not (prepared.services["routing_eligible"].eq(True).sum() == len(eligible_ids)):
        raise AssertionError("Baseline eligibility mutated during experimental removals")
    return {
        "policy": POLICY,
        "service_type": service_type.value,
        "population_selector": population_selector.value,
        "walking_speed_m_s": speed,
        "thresholds_min": thresholds,
        "eligible_service_count": len(eligible_ids),
        "excluded_service_count": len(destination) - len(eligible_ids),
        "baseline": base,
        "removals": rows,
        "full_leave_one_out": len(rows) == len(eligible_ids),
        "selection": "all_verified" if top_k is None else "largest_baseline_nearest_population_first",
        "interpretation": "Descriptive robustness to hypothetical single-facility unavailability;"
                          " not a real closure scenario; unverified facilities are never promoted."
                          " Conditional mean excludes unreachable origins and need not be monotonic.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-codes", nargs="+", required=True)
    parser.add_argument("--service-type", choices=[s.value for s in ServiceType], default="pharmacy")
    parser.add_argument("--population-selector", choices=[s.value for s in PopulationSelector], default="population_total")
    parser.add_argument("--walking-speed-m-s", type=float, default=0.9)
    parser.add_argument("--thresholds-min", nargs="+", type=int, default=list(DEFAULT_THRESHOLDS))
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--top-k", type=int, default=3)
    scope.add_argument("--all", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    codes = list(args.municipality_codes)
    if (not codes or len(set(codes)) != len(codes)
            or any(not re.fullmatch(r"\d{6}", code) for code in codes)):
        raise ValueError("Unique six-digit ISTAT municipality codes required")
    speed = _validated_speed(args.walking_speed_m_s)
    thresholds = _thresholds(args.thresholds_min)
    if not args.all and args.top_k < 1:
        raise ValueError("--top-k must be positive")
    record = {}
    print("=== B8 VERIFIED SERVICE REMOVAL ROBUSTNESS — READ ONLY ===", flush=True)
    print(f"Speed: {speed:g} m/s | thresholds: {thresholds} | "
          f"scope: {'all' if args.all else f'top {args.top_k}'}", flush=True)
    for code in codes:
        inspection = inspect_city_inputs(
            args.root.resolve(), code, census_year=args.census_year,
            school_year=args.school_year, health_date=args.health_reference_date,
        )
        summary = inspection["quality"]["service_types"].get(args.service_type)
        if summary is None:
            raise ValueError(f"No audited {args.service_type} catalog in {code}")
        print(f"{code}: PREFLIGHT PASS | origins={len(inspection['origins'])} "
              f"| eligible={summary['routable']}/{summary['total']} | "
              f"origins={inspection['origins_path'].name}", flush=True)
        if args.preflight_only:
            record[code] = {"status": "preflight_pass", "inputs_sha256": inspection["inputs_sha256"]}
            continue
        graph, _ = build_routing_graph(
            inspection["nodes"], inspection["edges"],
            travel_time_weight="walking_time_s", distance_weight="length_m",
        )
        analysis = analyze_removal_in_memory(
            graph, inspection["origins"], inspection["qualified_services"],
            inspection["attachments"], service_type=ServiceType(args.service_type),
            population_selector=PopulationSelector(args.population_selector),
            walking_speed_m_s=speed, thresholds=thresholds,
            top_k=None if args.all else args.top_k,
        )
        if analysis["eligible_service_count"] != summary["routable"]:
            raise ValueError("Leave-one-out routing eligibility differs from B8 independent audit")
        record[code] = {"status": "analyzed", "inputs_sha256": inspection["inputs_sha256"], **analysis}
        baseline = analysis["baseline"]
        desc = " ".join(f"{t}m={100 * baseline['coverage_by_minutes'][str(t)]:.2f}%" for t in thresholds)
        mean = baseline["conditional_population_weighted_mean_min"]
        mean_text = "N/A" if mean is None else f"{mean:.3f} min"
        print(f"  BASELINE | {desc} | conditional_mean={mean_text} | verified={analysis['eligible_service_count']}", flush=True)
        for row in analysis["removals"]:
            desc = " ".join(f"{t}m=-{row['coverage_loss_percentage_points'][str(t)]:.2f} pp" for t in thresholds)
            mean = row["scenario"]["conditional_population_weighted_mean_min"]
            mean_text = "N/A" if mean is None else f"{mean:.3f} min"
            print(f"  REMOVE {row['removed_service_id']} | baseline-nearest-pop={row['baseline_nearest_population']:.2f} "
                  f"| {desc} | conditional_mean={mean_text} "
                  f"| lost-reachable-pop={row['lost_reachable_population']:.2f}", flush=True)
        print(f"  PASS: {len(analysis['removals'])}/{analysis['eligible_service_count']} individual removals; "
              f"complete={analysis['full_leave_one_out']}", flush=True)
    print("READ ONLY — no catalog, Parquet, baseline, scenario report or approval modified", flush=True)
    return {"policy": POLICY, "cities": record, "walking_speed_m_s": speed,
            "preflight_only": args.preflight_only}


if __name__ == "__main__":
    run()
