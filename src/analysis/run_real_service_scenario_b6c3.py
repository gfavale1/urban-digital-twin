"""B6C.3b: first real-data, read-only-by-default service-removal experiment.

Reuses B6C.3a preflight + B6C.0/1/2 scenario contracts, routing and
immutable reporting. The current experiment is intentionally restricted to
walking in the frozen legacy-regression profile (1.4 m/s; 10/15/20 min).
It cannot promote missing or unvalidated ANNCSU services.

Example:
  PYTHONPATH=src python src/analysis/run_real_service_scenario_b6c3.py \\
    --municipality-code 034027 --city-name Parma \\
    --analysis-date 2025-06-30 \\
    --remove-service-id 'SVC2::HEALTH::SALUTE:PHARMACY:19945::pharmacy'

Append --write-report only after checking the read-only result.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.accessibility_contracts import (
    prepare_service_destinations_v2, request_from_service_spec,
)
from analysis.scenario_contracts_v2 import (
    ScenarioAction, ScenarioBaselineRef, ScenarioOperation, ScenarioSpec,
    request_sha256,
)
from analysis.scenario_reporting_v2 import (
    ScenarioInputPaths, _read_table, run_verified_scenario, write_scenario_report,
)
from core.analysis_spec import AnalysisSpec, ServiceType, TransportMode
from core.config import DEFAULT_CONFIG
from quality.preflight_real_scenario_b6c3 import (
    ROOT, CitySnapshot, canonical_paths, inspect_real_snapshot,
)

POLICY = "b6c3b_real_service_removal_experiment_v1"


def _check_legacy_walk_network(edges: pd.DataFrame, speed_m_s: float) -> None:
    """Refuse a misleading 1.4m/s analysis over a different-speed network.

    The legacy walking network materializes walking_time_s=length_m/speed.
    This checks that contract rather than silently rescaling travel times.
    """
    needed = {"length_m", "walking_time_s"}
    if needed - set(edges.columns):
        raise ValueError(f"Walking network lacks columns: {sorted(needed - set(edges.columns))}")
    if edges.empty:
        raise ValueError("Walking network has no edges")
    lengths = pd.to_numeric(edges["length_m"], errors="coerce").to_numpy(dtype=float)
    times = pd.to_numeric(edges["walking_time_s"], errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(lengths).all() or not np.isfinite(times).all()
            or (lengths <= 0).any() or (times <= 0).any()):
        raise ValueError("Invalid walking network lengths or travel times")
    if not np.allclose(times, lengths / speed_m_s, rtol=1e-6, atol=1e-5):
        raise ValueError("Walking network times are inconsistent with the legacy walking speed; do not mix methodologies")


def execute_real_removal(
    *, snapshot: CitySnapshot, paths: ScenarioInputPaths, service_id: str,
    city_name: str, analysis_date: str, output_root: Path | None = None,
    write_report: bool = False,
) -> tuple[dict[str, Any], Path | None]:
    """Validate real canonical files and execute one removal, never modify input.

    The returned report directory is None unless write_report=True.
    Other services of the same type keep their original eligibility status.
    """
    if snapshot.mode is not TransportMode.WALK:
        raise ValueError("B6C.3b legacy-regression experiment currently supports walking only")
    if snapshot.service_type not in (
        ServiceType.PHARMACY, ServiceType.HOSPITAL_ESTABLISHMENT,
        ServiceType.LEGACY_EDUCATION_ALL,
    ):
        raise ValueError("Service type is not supported by the legacy-regression profile")
    if not isinstance(service_id, str) or not service_id.strip() or service_id != service_id.strip():
        raise ValueError("--remove-service-id must be a full canonical service ID")
    if not isinstance(city_name, str) or not city_name.strip():
        raise ValueError("City name must be nonempty")
    if date.fromisoformat(analysis_date).isoformat() != analysis_date:
        raise ValueError("Analysis date must be YYYY-MM-DD")

    # Preflight checks real-world eligibility on the original data, not on a
    # filtered/synthetic dataset. It never changes baseline or service flags.
    inspection = inspect_real_snapshot(snapshot, paths, examples=0)
    if not inspection["selected_eligible_count"]:
        raise ValueError("No eligible services available for a removal scenario")

    services = _read_table(paths.services)
    attachments = _read_table(paths.attachments)
    subset = services.loc[services["service_type"].astype(str).eq(snapshot.service_type.value)]
    # Canonical attachment files contain *all* service types. The B6A3 gate
    # requires exactly one mode attachment for every selected service, and
    # correctly rejects unrelated service IDs as unexpected input.
    selected_ids = set(subset["service_id"].astype(str))
    selected_attachments = attachments.loc[
        attachments["entity_kind"].eq("service")
        & attachments["mode"].eq(snapshot.mode.value)
        & attachments["entity_id"].astype(str).isin(selected_ids)
    ]
    gated = prepare_service_destinations_v2(
        subset, selected_attachments, mode=snapshot.mode,
    )
    eligible = set(gated.loc[gated["routing_eligible"], "service_id"].astype(str))
    if service_id not in eligible:
        raise ValueError(f"The requested service is not an eligible baseline destination: {service_id}")

    spec = AnalysisSpec.from_legacy_pipeline_config(
        city_name=city_name.strip(), municipality_code=snapshot.municipality_code,
        analysis_date=analysis_date, config=DEFAULT_CONFIG,
    )
    service_spec = next(s for s in spec.services if s.service_type is snapshot.service_type)
    request = request_from_service_spec(
        service_spec, snapshot.mode, travel_time_weight="walking_time_s",
        off_network_speed_m_s=spec.walking_speed_m_s,
    )
    # The OSM network edge weights were generated at ingestion time. Check that
    # they match our fixed legacy profile; off-network speed alone is not enough.
    _check_legacy_walk_network(_read_table(paths.edges), spec.walking_speed_m_s)

    baseline = ScenarioBaselineRef(
        municipality_code=snapshot.municipality_code,
        analysis_spec_sha256=spec.spec_hash,
        request_sha256=request_sha256(request),
        origins_sha256=inspection["input_sha256"]["origins"],
        services_sha256=inspection["input_sha256"]["services"],
        attachments_sha256=inspection["input_sha256"]["attachments"],
        graph_sha256=inspection["graph_sha256"],
    )
    scenario = ScenarioSpec(
        baseline=baseline, service_type=snapshot.service_type, mode=snapshot.mode,
        operations=(ScenarioOperation(ScenarioAction.REMOVE_SERVICE, service_id),),
    )
    # B6C.2 verifies file bytes a second time against the preflight hashes.
    result, provenance = run_verified_scenario(
        scenario=scenario, analysis_spec=spec, request=request, inputs=paths,
    )
    if result.metrics["baseline"]["service_count_routable"] != len(eligible):
        raise AssertionError("Baseline routable count differs from B6C.3a Quality Gate")
    if (result.metrics["scenario"]["service_count_routable"]
            != result.metrics["baseline"]["service_count_routable"] - 1):
        raise AssertionError("Removing one eligible service did not remove exactly one destination")
    if (result.metrics["baseline"]["target_population_total"]
            != result.metrics["scenario"]["target_population_total"]):
        raise AssertionError("Scenario changed population total")
    for key, change in result.metrics["delta"].items():
        if (key.startswith("population_coverage_within_")
                or key.startswith("population_weighted_cumulative_opportunities_within_")):
            if change is not None and change > 1e-9:
                raise AssertionError(f"Removing a service improved {key}")
    # Do not assert monotonicity of the conditional mean among REACHABLE origins:
    # that statistic can decrease if previously reachable distant origins drop out.

    # Return a concise, JSON-serializable preview. Full per-origin outputs are
    # persisted only on explicit opt-in below.
    summary = {
        "policy": POLICY,
        "scenario_id": result.scenario_id,
        "municipality_code": snapshot.municipality_code,
        "removed_service_id": service_id,
        "execution_profile": spec.execution_profile,
        "walking_speed_m_s": spec.walking_speed_m_s,
        "thresholds_min": list(request.thresholds_min),
        "origin_count": inspection["origin_count"],
        "preflight_eligible_services": inspection["selected_eligible_count"],
        "metrics": result.metrics,
        "input_sha256": inspection["input_sha256"],
        "graph_sha256": inspection["graph_sha256"],
    }
    dest = None
    if write_report:
        if output_root is None:
            raise ValueError("Explicit output_root is required to write reports")
        provenance["b6c3b_experiment"] = {
            "policy": POLICY, "removed_service_id": service_id,
            "preflight_eligible_services": inspection["selected_eligible_count"],
            "profile": spec.execution_profile,
        }
        dest, cached = write_scenario_report(
            result=result, provenance=provenance, output_root=output_root,
        )
        summary["report_cached"] = cached
        summary["report_directory"] = str(dest)
    return summary, dest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--city-name", default=None)
    parser.add_argument("--analysis-date", required=True,
                        help="Logical analysis date (ISO, e.g. 2025-06-30)")
    parser.add_argument("--census-year", default=DEFAULT_CONFIG.census_year)
    parser.add_argument("--school-year", default=DEFAULT_CONFIG.school_year)
    parser.add_argument("--health-reference-date", default=DEFAULT_CONFIG.health_reference_date)
    parser.add_argument("--service-type", default="pharmacy", choices=[s.value for s in ServiceType])
    parser.add_argument("--mode", default="walk", choices=[m.value for m in TransportMode])
    parser.add_argument("--remove-service-id", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    snap = CitySnapshot(
        municipality_code=args.municipality_code,
        census_year=args.census_year, school_year=args.school_year,
        health_reference_date=args.health_reference_date,
        mode=TransportMode(args.mode), service_type=ServiceType(args.service_type),
    )
    paths = canonical_paths(args.root, snap)
    summary, _ = execute_real_removal(
        snapshot=snap, paths=paths, service_id=args.remove_service_id,
        city_name=args.city_name or f"Comune {snap.municipality_code}",
        analysis_date=args.analysis_date,
        write_report=args.write_report,
        output_root=(args.output_root or args.root / "data/features/scenarios/b6c3b")
            if args.write_report else None,
    )
    print("=== B6C.3b REAL SERVICE REMOVAL — %s ===" %
          ("REPORT SAVED" if args.write_report else "READ ONLY"))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    if not args.write_report:
        print("READ-ONLY: baseline and scenario report directories unchanged")
    return summary


if __name__ == "__main__":
    run()
