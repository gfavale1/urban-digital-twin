"""B8 final quality gate for verified, census-qualified V2 walking accessibility.

Reuses existing audits and experiment engines; re-computes the complete pharmacy
walking-speed sensitivity and leave-one-out, never promotes excluded services.
Default is read-only; a new immutable JSON result requires --write --output.
This is a B8 scope gate, not proof of full Italian national coverage.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

from analysis.analyze_service_removal_robustness import analyze_removal_in_memory
from analysis.analyze_walking_speed_sensitivity import analyze_in_memory, inspect_city_inputs
from analysis.network_graph import build_routing_graph
from core.analysis_spec import PopulationSelector, ServiceType
from core.run_manifest import sha256_file
from quality.audit_v2_service_boundary import paths_for_snapshot
from quality.preflight_real_service_scenario import ROOT

POLICY = "b8_final_verified_pharmacy_quality_assessment_v1"
DEFAULT_SPEEDS = (0.7, 0.9, 1.1)
BASELINE_SPEED = 0.9
DEFAULT_THRESHOLDS = (10, 15, 20)
DEFAULT_CITIES = ("034027", "077014", "016024")
SOURCE_MODULES = (
    "src/quality/finalize_accessibility_quality.py",
    "src/quality/audit_service_routing_quality.py",
    "src/analysis/analyze_walking_speed_sensitivity.py",
    "src/analysis/analyze_service_removal_robustness.py",
    "src/analysis/accessibility_engine.py",
    "src/analysis/accessibility_contracts.py",
)
LIMITATIONS = (
    "Routing eligibility describes verified canonical entities; catalog completeness against all operating facilities is unestablished.",
    "Excluded or unresolved destinations are not promoted, and their hypothetical use is not a valid upper bound without location validation.",
    "Territorial qualification uses the ISTAT 2021 census-footprint union, not an administrative municipality boundary.",
    "Matera and Bergamo origins retain the legacy node-distributed population model; they are not representative-point origins.",
    "Population_total is a descriptive all-age denominator for pharmacy access, not evidence of demand, capacity or realized visits.",
    "Walking speeds and hypothetical one-at-a-time removals are model sensitivity tests, not observed travel behavior or real closure forecasts.",
    "Conditional mean excludes origins with no reachable destination and may decrease after a removal.",
    "Historical independent report regression, source coverage across Italy, drive mode, school cohorts and capacities are outside this B8 final gate.",
)


def validate_parameters(codes: Sequence[str], speeds: Sequence[float],
                        thresholds: Sequence[int], baseline: float) -> tuple[list[str], tuple[float, ...], tuple[int, ...]]:
    identifiers = list(codes)
    if not identifiers or any(not isinstance(x, str) or not re.fullmatch(r"\d{6}", x) for x in identifiers):
        raise ValueError("Provide one or more six-digit municipality codes")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate municipality codes")
    velocity = tuple(float(v) for v in speeds)
    if not velocity or any(not math.isfinite(v) or v <= 0 for v in velocity) or velocity != tuple(sorted(set(velocity))):
        raise ValueError("Walking speeds must be finite, positive, strictly increasing and unique")
    if baseline not in velocity:
        raise ValueError("The removal baseline speed must occur in the sensitivity grid")
    minutes = tuple(thresholds)
    if (not minutes or any(type(v) is not int or v <= 0 for v in minutes)
            or minutes != tuple(sorted(set(minutes)))):
        raise ValueError("Thresholds must be strictly increasing positive integers")
    return identifiers, velocity, minutes


def check_quality_denominators(quality: dict[str, Any], category: str) -> dict[str, Any]:
    """Quality reasons are exclusive, overlapping flags must never be summed."""
    total = int(quality["services_total"])
    eligible = int(quality["services_routable"])
    excluded = int(quality["services_ineligible"])
    if total <= 0 or eligible < 0 or excluded < 0 or eligible + excluded != total:
        raise ValueError("Inconsistent canonical service counts")
    types = quality["service_types"]
    if category not in types:
        raise ValueError(f"No audited {category} services")
    if (sum(int(row["total"]) for row in types.values()) != total
            or sum(int(row["routable"]) for row in types.values()) != eligible):
        raise ValueError("Quality service-category totals do not reconcile")
    for row in (quality, *types.values()):
        n = int(row["total"]) if "total" in row else total
        if sum(map(int, row["primary_routing_reason"].values())) != n:
            raise ValueError("Exclusive routing reasons do not sum to denominator")
        if sum(map(int, row["geography"].values())) != n:
            raise ValueError("Geographic classes do not sum to denominator")
    ph = types[category]
    if (int(ph["routable"]) <= 0 or int(ph["routable"]) + int(ph["ineligible"]) != int(ph["total"])):
        raise ValueError("No verified pharmacy destinations, or pharmacy counts inconsistent")
    return {
        "canonical_services_total": total,
        "routable_services_total": eligible,
        "excluded_services_total": excluded,
        "exclusive_exclusion_reasons": quality["primary_routing_reason"],
        "geography": quality["geography"],
        "overlapping_quality_issues": quality["overlapping_issues"],
        "by_service_type": types,
        "focus_service_type": category,
        "focus_services_total": int(ph["total"]),
        "focus_services_routable": int(ph["routable"]),
        "focus_services_excluded": int(ph["ineligible"]),
        "focus_exclusive_reasons": ph["primary_routing_reason"],
        "focus_geography": ph["geography"],
        "focus_overlapping_issues": ph["overlapping_issues"],
    }


def ensure_baseline_equivalence(sensitivity: dict[str, Any], removal: dict[str, Any],
                                *, baseline_speed: float, thresholds: Sequence[int],
                                quality: dict[str, Any]) -> dict[str, Any]:
    """Independent experiments must agree on all aggregate baseline metrics."""
    matches = [x for x in sensitivity["results"] if x["speed_m_s"] == baseline_speed]
    if len(matches) != 1:
        raise ValueError("Missing or duplicated baseline speed in sensitivity results")
    a, b = matches[0], removal["baseline"]
    if not sensitivity["monotonicity_verified"]:
        raise ValueError("Speed monotonicity has not been verified")
    if (not removal["full_leave_one_out"] or removal["selection"] != "all_verified"
            or len(removal["removals"]) != removal["eligible_service_count"]):
        raise ValueError("Leave-one-out is partial; final gate requires all eligible services")
    if (len({r["removed_service_id"] for r in removal["removals"]}) != len(removal["removals"])
            or removal["eligible_service_count"] != quality["focus_services_routable"]
            or removal["excluded_service_count"] != quality["focus_services_excluded"]):
        raise ValueError("Removal destinations disagree with independent eligibility audit")
    for key in ("origin_count", "service_count_total", "service_count_routable"):
        if a[key] != b[key]:
            raise ValueError(f"Baseline mismatch: {key}")
    if a["service_count_total"] != quality["focus_services_total"]:
        raise ValueError("Baseline catalog size disagrees with audited quality")
    for key in ("target_population", "reachable_population", "reachable_population_share",
                "conditional_population_weighted_mean_min"):
        x, y = a[key], b[key]
        if (x is None) != (y is None) or (x is not None and not math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-7)):
            raise ValueError(f"Baseline mismatch: {key}")
    for t in thresholds:
        x, y = a["coverage_by_minutes"][str(t)], b["coverage_by_minutes"][str(t)]
        if (x is None) != (y is None) or (x is not None and not math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError(f"Baseline coverage mismatch: {t} minutes")
    for entry in removal["removals"]:
        if entry["lost_reachable_population"] < -1e-7:
            raise ValueError("Removal increased reachable population")
        if any(value is None or value < -1e-7 for value in entry["coverage_loss_percentage_points"].values()):
            raise ValueError("Removal improved coverage")
    return {"status": "PASS", "baseline_speed_m_s": baseline_speed,
            "verified_eligible_destinations": removal["eligible_service_count"],
            "fully_recomputed_removals": len(removal["removals"])}


def summarize_removal(removal: dict[str, Any], thresholds: Sequence[int]) -> dict[str, Any]:
    worst = {}
    for t in thresholds:
        key = str(t)
        top = max(removal["removals"], key=lambda x: (x["coverage_loss_percentage_points"][key], x["removed_service_id"]))
        worst[key] = {"service_id": top["removed_service_id"],
                      "loss_percentage_points": top["coverage_loss_percentage_points"][key]}
    cases = [{"service_id": row["removed_service_id"], "lost_reachable_population": row["lost_reachable_population"]}
             for row in removal["removals"] if row["lost_reachable_population"] > 1e-7]
    return {
        "status": "PASS", "full_leave_one_out": True,
        "verified_destinations": removal["eligible_service_count"],
        "individual_removals": len(removal["removals"]),
        "baseline": removal["baseline"],
        "worst_coverage_loss_by_threshold": worst,
        "removals_losing_any_reachability": cases,
        "individual_scenarios": removal["removals"],
    }


def check_source_hashes(root: Path, inspection: dict[str, Any]) -> None:
    """Fail if sources changed during the assessment, including census geometry."""
    snapshot = inspection["snapshot"]
    paths = paths_for_snapshot(root, snapshot).items()
    quality_hashes = inspection["quality"]["input_sha256"]
    for key, path in paths.items():
        if quality_hashes[key] != sha256_file(path):
            raise ValueError(f"Input changed during final B8 QA: {key}")
    if inspection["inputs_sha256"]["origins"] != sha256_file(inspection["origins_path"]):
        raise ValueError("Origin source changed during final B8 QA")


def evaluate_city(inspection: dict[str, Any], *, root: Path, speeds: tuple[float, ...],
                  thresholds: tuple[int, ...], baseline_speed: float) -> dict[str, Any]:
    quality = check_quality_denominators(inspection["quality"], "pharmacy")
    graph, _ = build_routing_graph(
        inspection["nodes"], inspection["edges"],
        travel_time_weight="walking_time_s", distance_weight="length_m",
    )
    sensitivity = analyze_in_memory(
        graph, inspection["origins"], inspection["qualified_services"], inspection["attachments"],
        service_type=ServiceType.PHARMACY, population_selector=PopulationSelector.TOTAL,
        speeds=speeds, thresholds=thresholds,
    )
    robustness = analyze_removal_in_memory(
        graph, inspection["origins"], inspection["qualified_services"], inspection["attachments"],
        service_type=ServiceType.PHARMACY, population_selector=PopulationSelector.TOTAL,
        walking_speed_m_s=baseline_speed, thresholds=thresholds, top_k=None,
    )
    agreement = ensure_baseline_equivalence(
        sensitivity, robustness, baseline_speed=baseline_speed,
        thresholds=thresholds, quality=quality,
    )
    check_source_hashes(root, inspection)
    return {
        "status": "PASS", "origin_source": str(inspection["origins_path"].relative_to(root)),
        "input_sha256": inspection["inputs_sha256"],
        "quality_audit_input_sha256": inspection["quality"]["input_sha256"],
        "graph_sha256": inspection["quality"]["graph_sha256"],
        "quality": quality,
        "sensitivity": {"status": "PASS", "monotonicity_verified": True,
                        "scenario_results": sensitivity["results"]},
        "robustness": summarize_removal(robustness, thresholds),
        "cross_experiment_baseline_check": agreement,
    }


def build_assessment(root: Path, codes: Sequence[str] = DEFAULT_CITIES, *,
                     speeds: Sequence[float] = DEFAULT_SPEEDS, thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
                     baseline_speed: float = BASELINE_SPEED,
                     preflight_only: bool = False) -> dict[str, Any]:
    codes, speeds, thresholds = validate_parameters(codes, speeds, thresholds, baseline_speed)
    root = Path(root).resolve()
    implementation = {path: sha256_file(root / path) for path in SOURCE_MODULES}
    cities = {}
    for code in codes:
        inspection = inspect_city_inputs(root, code)
        quality = check_quality_denominators(inspection["quality"], "pharmacy")
        check_source_hashes(root, inspection)
        if preflight_only:
            cities[code] = {"status": "PREFLIGHT_ONLY", "quality": quality,
                            "origin_source": str(inspection["origins_path"].relative_to(root)),
                            "input_sha256": inspection["inputs_sha256"]}
        else:
            cities[code] = evaluate_city(inspection, root=root, speeds=speeds,
                                         thresholds=thresholds, baseline_speed=baseline_speed)
        print(f"{code}: {cities[code]['status']} | services="
              f"{quality['routable_services_total']}/{quality['canonical_services_total']} | pharmacy="
              f"{quality['focus_services_routable']}/{quality['focus_services_total']}", flush=True)
        if not preflight_only:
            worst = cities[code]["robustness"]["worst_coverage_loss_by_threshold"]
            for t in thresholds:
                print(f"  {t}m worst loss: {worst[str(t)]['loss_percentage_points']:.2f} pp "
                      f"[{worst[str(t)]['service_id']}]", flush=True)
            print(f"  Complete removals: {cities[code]['robustness']['individual_removals']}", flush=True)
    if {key: sha256_file(root / key) for key in SOURCE_MODULES} != implementation:
        raise ValueError("Code implementation changed during B8 final assessment")
    return {
        "policy": POLICY,
        "status": "PREFLIGHT_ONLY" if preflight_only else "PASS",
        "scope": {"municipality_codes": codes, "transport_mode": "walk", "service_type": "pharmacy",
                  "population_selector": "population_total", "walking_speeds_m_s": list(speeds),
                  "removal_baseline_speed_m_s": baseline_speed, "thresholds_min": list(thresholds),
                  "all_verified_single_service_removals": not preflight_only},
        "implementation_sha256": implementation,
        "cities": cities,
        "verification": {"input_provenance": "PASS", "routing_quality": "PASS",
                         "speed_sensitivity": "NOT_EXECUTED" if preflight_only else "PASS",
                         "complete_verified_destination_removals": "NOT_EXECUTED" if preflight_only else "PASS",
                         "independent_experiment_baselines_agree": "NOT_EXECUTED" if preflight_only else "PASS",
                         "independent_historical_report_regression": "NOT_EVALUATED",
                         "all_italian_municipalities_covered": "NOT_ESTABLISHED"},
        "methodological_limitations": list(LIMITATIONS),
    }


def write_new_report(output: Path, report: dict[str, Any]) -> Path:
    if report["status"] != "PASS":
        raise ValueError("Refusing to publish preflight-only/incomplete B8 final assessment")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"B8 assessment already exists; refusing overwrite: {output}")
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".b8_quality_", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, output)  # fails atomically if destination exists; no overwrites
    finally:
        Path(tmp).unlink(missing_ok=True)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-codes", nargs="+", default=list(DEFAULT_CITIES))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--write", action="store_true", help="Publish new versioned JSON only after full PASS")
    parser.add_argument("--output", type=Path, help="New JSON path; must not exist")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.write and args.preflight_only:
        raise ValueError("Cannot write a preflight-only report")
    if args.write != (args.output is not None):
        raise ValueError("--write requires --output and --output requires --write")
    if args.output and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {args.output}")
    print("=== B8 FINAL QUALITY ASSESSMENT — " +
          ("PREFLIGHT ONLY" if args.preflight_only else "FULL VALIDATION") + " ===", flush=True)
    assessment = build_assessment(args.root, args.municipality_codes, preflight_only=args.preflight_only)
    print(f"ASSESSMENT: {assessment['status']} | "
          f"cities: {len(assessment['cities'])}/{len(args.municipality_codes)}", flush=True)
    if args.write:
        created = write_new_report(args.output, assessment)
        print(f"NEW IMMUTABLE B8 REPORT: {created}", flush=True)
        print(f"REPORT SHA256: {sha256_file(created)}", flush=True)
    else:
        print("READ ONLY — no source datasets or reports modified", flush=True)
    print("Scope: walk/pharmacy/all-age population; NOT full national/source completeness", flush=True)
    return assessment


if __name__ == "__main__":
    run()
