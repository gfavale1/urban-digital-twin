"""B8 read-only, one-at-a-time walking-speed sensitivity on qualified V2 services.

Change both network edge times and off-network attachment travel time, leaving
canonical tables, network files, geometry, eligibility and origins unchanged.
An optional preflight checks all input provenance without running routes.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from analysis.accessibility_contracts import prepare_canonical_accessibility_inputs
from analysis.accessibility_engine import AccessibilityEngineRequest, compute_accessibility
from analysis.network_graph import build_routing_graph
from analysis.scenario_reporting_v2 import _read_table
from core.analysis_spec import PopulationSelector, ServiceType, TransportMode
from core.run_manifest import sha256_file
from core.schema_v2 import ORIGIN_V2
from quality.audit_service_routing_quality import inspect_service_quality
from quality.derive_boundary_qualified_services import inspect_boundary_view
from quality.preflight_real_service_scenario import ROOT, CitySnapshot, canonical_paths
from transformation.build_network_attachments_v2 import graph_checksum

POLICY = "walking_speed_sensitivity_census_qualified_v1"
LEGACY_SPEED_M_S = 1.4
DEFAULT_SPEEDS = (0.7, 0.9, 1.1)
DEFAULT_THRESHOLDS = (10, 15, 20)


def _speeds(values: Sequence[float]) -> tuple[float, ...]:
    speeds = tuple(float(v) for v in values)
    if not speeds or any(not math.isfinite(s) or s <= 0 for s in speeds):
        raise ValueError("Walking speeds must be finite and positive")
    if speeds != tuple(sorted(set(speeds))):
        raise ValueError("Walking speeds must be strictly increasing and unique")
    return speeds


def validate_legacy_edge_speeds(edges: pd.DataFrame) -> None:
    required = {"length_m", "walking_time_s"}
    if not required <= set(edges.columns) or edges.empty:
        raise ValueError("Walking edges lack length_m/walking_time_s or are empty")
    length = pd.to_numeric(edges["length_m"], errors="coerce").to_numpy(dtype=float)
    time = pd.to_numeric(edges["walking_time_s"], errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(length).all() or not np.isfinite(time).all()
            or (length <= 0).any() or (time <= 0).any()):
        raise ValueError("Edges must have finite positive length and travel time")
    if not np.isclose(length / time, LEGACY_SPEED_M_S, rtol=1e-3, atol=1e-4).all():
        raise ValueError("Edge times are not consistent with the verified 1.4 m/s legacy network")


def reweight_graph(graph: nx.Graph, speed_m_s: float) -> nx.Graph:
    """Never modify the input graph; recompute *every* walking time from length."""
    speed = _speeds((speed_m_s,))[0]
    result = graph.copy()
    for _, _, attrs in result.edges(data=True):
        distance = float(attrs["length_m"])
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("Graph edge length must be finite and strictly positive")
        attrs["walking_time_s"] = distance / speed
    return result


def summarize_result(result: Any, thresholds: tuple[int, ...]) -> dict[str, Any]:
    frame = result.origins
    population = pd.to_numeric(frame["target_population"], errors="raise").to_numpy(dtype=float)
    nearest = pd.to_numeric(frame["nearest_service_time_min"], errors="coerce").to_numpy(dtype=float)
    reachable = np.isfinite(nearest)
    if (not np.isfinite(population).all() or (population < 0).any()
            or (np.isinf(nearest)).any()):
        raise ValueError("Accessibility results contain invalid populations/times")
    total = float(population.sum())
    reachable_population = float(population[reachable].sum())
    weighted_mean = (
        float(np.dot(nearest[reachable], population[reachable]) / reachable_population)
        if reachable_population > 0 else None
    )
    covered = {}
    for t in thresholds:
        value = result.summary[f"population_coverage_within_{t}_min"]
        if value is not None and (not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
            raise ValueError("Coverage must lie between 0 and 1")
        covered[str(t)] = None if value is None else float(value)
    return {
        "origin_count": int(len(frame)),
        "target_population": total,
        "reachable_population": reachable_population,
        "reachable_population_share": reachable_population / total if total > 0 else None,
        "conditional_population_weighted_mean_min": weighted_mean,
        "coverage_by_minutes": covered,
        "service_count_total": int(result.summary["service_count_total"]),
        "service_count_routable": int(result.summary["service_count_routable"]),
    }


def _assert_monotonic(results: list[tuple[float, Any]], thresholds: tuple[int, ...]) -> None:
    """Uniform speed scaling cannot change connectivity or path distances."""
    prev_speed, prev = results[0]
    previous = prev.origins
    time_ref = pd.to_numeric(previous["nearest_service_time_min"], errors="coerce").to_numpy(dtype=float)
    for speed, result in results[1:]:
        current = result.origins
        if not previous["origin_id"].equals(current["origin_id"]):
            raise ValueError("Origin ordering changed between sensitivity scenarios")
        if not previous["reachability_status"].equals(current["reachability_status"]):
            raise ValueError("Reachability changed despite fixed network and services")
        times = pd.to_numeric(current["nearest_service_time_min"], errors="coerce").to_numpy(dtype=float)
        if not np.array_equal(np.isfinite(time_ref), np.isfinite(times)):
            raise ValueError("Finite reachability changed between speeds")
        finite = np.isfinite(times)
        if not np.allclose(time_ref[finite] * prev_speed, times[finite] * speed,
                           rtol=1e-8, atol=1e-7):
            raise ValueError("Route distance changed under uniform speed scaling")
        if ((times[finite] - time_ref[finite]) > 1e-7).any():
            raise ValueError("Faster walking increased nearest service time")
        for t in thresholds:
            key = f"opportunity_count_within_{t}_min"
            if (current[key].to_numpy() < previous[key].to_numpy()).any():
                raise ValueError("Faster walking reduced cumulative opportunities")
        time_ref, previous, prev_speed = times, current, speed


def analyze_in_memory(
    graph: nx.Graph, origins: pd.DataFrame, qualified_services: pd.DataFrame,
    attachments: pd.DataFrame, *, service_type: ServiceType = ServiceType.PHARMACY,
    population_selector: PopulationSelector = PopulationSelector.TOTAL,
    speeds: Sequence[float] = DEFAULT_SPEEDS,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Execute independent V2 engine runs with identical demand/supply/network."""
    speeds = _speeds(speeds)
    thresholds = tuple(int(t) for t in thresholds)
    if (not thresholds or thresholds != tuple(sorted(set(thresholds)))
            or any(t <= 0 for t in thresholds)):
        raise ValueError("Thresholds must be strictly increasing positive integers")
    service_type = ServiceType(service_type)
    population_selector = PopulationSelector(population_selector)
    selected = qualified_services.loc[
        qualified_services["service_type"].astype(str).eq(service_type.value)
    ].copy()
    if selected.empty:
        raise ValueError(f"No canonical services of type {service_type.value}")
    kind = attachments["entity_kind"].astype(str)
    chosen = attachments.loc[
        kind.eq("origin") | (kind.eq("service") & attachments["entity_id"].astype(str).isin(selected["service_id"].astype(str)))
    ].copy()
    prepared = prepare_canonical_accessibility_inputs(
        origins, selected, chosen, mode=TransportMode.WALK,
    )
    results = []
    rows = []
    for speed in speeds:
        weighted = reweight_graph(graph, speed)
        request = AccessibilityEngineRequest(
            service_type=service_type, mode=TransportMode.WALK,
            thresholds_min=thresholds, population_selector=population_selector,
            travel_time_weight="walking_time_s", off_network_speed_m_s=speed,
            distance_weight="length_m",
        )
        result = compute_accessibility(weighted, prepared.origins, prepared.services, request)
        results.append((speed, result))
        rows.append({"speed_m_s": speed, **summarize_result(result, thresholds)})
    if len(results) > 1:
        _assert_monotonic(results, thresholds)
    first = rows[0]
    if any((r["service_count_routable"] != first["service_count_routable"]
            or r["target_population"] != first["target_population"]
            or r["reachable_population"] != first["reachable_population"])
           for r in rows):
        raise ValueError("Baseline entities/population/reachability changed between speeds")
    return {"service_type": service_type.value, "population_selector": population_selector.value,
            "thresholds_min": thresholds, "results": rows, "monotonicity_verified": True}


def _load_manifest_and_check_origins(root: Path, snapshot: CitySnapshot,
                                     files: dict[str, Path]) -> Path:
    attachments = files["attachments"]
    suffix = f"network_attachments_v2_{snapshot.mode.value}_{snapshot.census_year}_{snapshot.school_year}_{snapshot.health_reference_date.replace('-', '')}"
    manifest_file = attachments.with_name(suffix + "_manifest.json")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Attachment provenance manifest missing: {manifest_file}")
    meta = json.loads(manifest_file.read_text(encoding="utf-8"))
    if meta.get("municipality_code") != snapshot.municipality_code or meta.get("mode") != "walk":
        raise ValueError("Attachment manifest municipality/mode mismatch")
    hashes = meta.get("input_sha256", {})
    paths = meta.get("input_paths", {})
    for key in ("services", "nodes", "edges"):
        if hashes.get(key) != sha256_file(files[key]):
            raise ValueError(f"Attachment {key} input SHA-256 mismatch")
    origin_path_value = paths.get("origins")
    if not isinstance(origin_path_value, str) or not origin_path_value.strip():
        raise ValueError("Attachment manifest lacks an explicit origin input path")
    origin_path = Path(origin_path_value).resolve()
    directory = root / "data/features/accessibility" / snapshot.municipality_code
    canonical = (directory / f"population_network_origins_{snapshot.census_year}.parquet").resolve()
    migrated = (directory / f"population_network_origins_{snapshot.census_year}_v2_migrated.parquet").resolve()
    if origin_path not in (canonical, migrated):
        raise ValueError("Attachment origins are neither canonical nor versioned migrated origins")
    if not origin_path.is_file() or hashes.get("origins") != sha256_file(origin_path):
        raise ValueError("Attachment origin file checksum mismatch")
    if origin_path == migrated:
        provenance_file = migrated.with_name(migrated.stem + "_manifest.json")
        if not provenance_file.is_file():
            raise FileNotFoundError("Migrated origins lack their provenance manifest")
        origin_manifest = json.loads(provenance_file.read_text(encoding="utf-8"))
        if (origin_manifest.get("source_sha256") != sha256_file(canonical)
                or origin_manifest.get("target_sha256") != sha256_file(migrated)
                or origin_manifest.get("canonical_origin_replaced") is not False
                or origin_manifest.get("origin_representation") != "legacy_node_distributed_not_representative_point"):
            raise ValueError("OriginV2 migration provenance mismatch")
        if meta.get("origin_input_selection") != "explicit_v2_origins":
            raise ValueError("Migrated origin manifest does not match attachment selection")
    else:
        if meta.get("origin_input_selection", "default_canonical_origins") != "default_canonical_origins":
            raise ValueError("Canonical origin selection mismatch")
    return origin_path


def inspect_city_inputs(root: Path, code: str, *, census_year: str = "2023",
                        school_year: str = "202425", health_date: str = "2025-06-30") -> dict[str, Any]:
    snapshot = CitySnapshot(municipality_code=code, census_year=census_year,
                            school_year=school_year, health_reference_date=health_date)
    baseline = canonical_paths(root, snapshot)
    files = baseline.items()
    for label, path in files.items():
        if label != "origins" and not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    origin_path = _load_manifest_and_check_origins(root, snapshot, files)
    expected = {key: sha256_file(path) for key, path in files.items() if key != "origins"}
    expected["origins"] = sha256_file(origin_path)
    quality = inspect_service_quality(snapshot, root=root)
    qualified, geographic = inspect_boundary_view(snapshot, root=root)
    origins = _read_table(origin_path)
    ORIGIN_V2.validate_columns(origins.columns)
    if origins.empty or not origins["municipality_code"].astype(str).eq(code).all():
        raise ValueError("Missing or foreign OriginV2 data")
    if origins["origin_id"].isna().any() or origins["origin_id"].astype(str).duplicated().any():
        raise ValueError("Missing or duplicated OriginV2 identifiers")
    attachments = _read_table(files["attachments"])
    nodes = _read_table(files["nodes"])
    edges = _read_table(files["edges"])
    validate_legacy_edge_speeds(edges)
    if graph_checksum(files["nodes"], files["edges"]) != quality["graph_sha256"]:
        raise ValueError("Graph checksum differs from quality audit")
    if not quality["input_sha256"]["services"] == geographic["source_sha256"]["services"]:
        raise ValueError("Qualified service identity mismatch")
    if {key: sha256_file(path) for key, path in files.items() if key != "origins"} != {
            key: value for key, value in expected.items() if key != "origins"}:
        raise ValueError("Canonical files changed during load")
    if sha256_file(origin_path) != expected["origins"]:
        raise ValueError("Origin file changed during load")
    return {"code": code, "snapshot": snapshot, "inputs_sha256": expected,
            "origins_path": origin_path, "quality": quality, "qualified_services": qualified,
            "origins": origins, "attachments": attachments, "nodes": nodes, "edges": edges}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-codes", nargs="+", required=True)
    parser.add_argument("--service-type", choices=[s.value for s in ServiceType], default="pharmacy")
    parser.add_argument("--population-selector", choices=[s.value for s in PopulationSelector], default="population_total")
    parser.add_argument("--thresholds-min", nargs="+", type=int, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--walking-speeds", nargs="+", type=float, default=list(DEFAULT_SPEEDS))
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    codes = list(args.municipality_codes)
    if (not codes or len(set(codes)) != len(codes)
            or any(not re.fullmatch(r"\d{6}", code) for code in codes)):
        raise ValueError("Unique six-digit ISTAT municipality codes are required")
    speeds = _speeds(args.walking_speeds)
    thresholds = tuple(args.thresholds_min)
    if (not thresholds or thresholds != tuple(sorted(set(thresholds)))
            or any(t <= 0 for t in thresholds)):
        raise ValueError("Positive, strictly increasing thresholds required")
    root = args.root.resolve()
    records = {}
    print("=== B8 WALKING SPEED SENSITIVITY — READ ONLY ===")
    print("Speeds (m/s):", speeds, "| minutes:", thresholds)
    for code in codes:
        inspection = inspect_city_inputs(root, code, census_year=args.census_year,
                                         school_year=args.school_year,
                                         health_date=args.health_reference_date)
        selected = inspection["quality"]["service_types"].get(args.service_type)
        if selected is None:
            raise ValueError(f"No audited {args.service_type} services for {code}")
        print(f"{code}: PREFLIGHT PASS | origins={len(inspection['origins'])} | "
              f"{args.service_type} routable={selected['routable']}/{selected['total']} | "
              f"origin_file={inspection['origins_path'].name}", flush=True)
        if args.preflight_only:
            records[code] = {"status": "preflight_pass", "inputs_sha256": inspection["inputs_sha256"]}
            continue
        graph, _ = build_routing_graph(inspection["nodes"], inspection["edges"],
                                       travel_time_weight="walking_time_s", distance_weight="length_m")
        report = analyze_in_memory(graph, inspection["origins"], inspection["qualified_services"],
                                   inspection["attachments"],
                                   service_type=ServiceType(args.service_type),
                                   population_selector=PopulationSelector(args.population_selector),
                                   speeds=speeds, thresholds=thresholds)
        if report["results"][0]["service_count_routable"] != selected["routable"]:
            raise ValueError("Sensitivity routing gate disagrees with independent B8 quality audit")
        records[code] = {"status": "analyzed", "inputs_sha256": inspection["inputs_sha256"], **report}
        for row in report["results"]:
            cover = " ".join(f"{t}m={row['coverage_by_minutes'][str(t)] * 100:.2f}%"
                             if row['coverage_by_minutes'][str(t)] is not None else f"{t}m=N/A"
                             for t in thresholds)
            mean = row["conditional_population_weighted_mean_min"]
            print(f"  {row['speed_m_s']:.1f} m/s | {cover} | "
                  f"conditional_mean={mean:.3f} min" if mean is not None
                  else f"  {row['speed_m_s']:.1f} m/s | {cover} | conditional_mean=N/A",
                  flush=True)
        print("  PASS: eligibility, reachable population, routing distances and monotonicity", flush=True)
    print("READ ONLY — no Parquets, baselines, reports or scenario files modified")
    return {"policy": POLICY, "cities": records, "service_type": args.service_type,
            "population_selector": args.population_selector, "thresholds_min": thresholds,
            "speeds_m_s": speeds, "preflight_only": bool(args.preflight_only),
            "interpretation": "Total population is a descriptive denominator, not a school-age cohort proxy."
                              " Conditional mean excludes unreachable origins."
                              " Footprint is ISTAT-2021 census geography, not an administrative boundary."
                              " No representative-point origin model is implied."}


if __name__ == "__main__":
    run()
