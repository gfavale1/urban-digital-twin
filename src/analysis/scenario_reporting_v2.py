"""B6C.2: verified file-backed service-scenario runner and immutable reports.

Unlike B6C.1 this module reads actual baseline files and verifies hashes.
This is a library runner, not yet the national v2 orchestrator (B10).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from analysis.accessibility_engine import AccessibilityEngineRequest
from analysis.network_graph import build_routing_graph
from analysis.scenario_contracts_v2 import ScenarioSpec, request_sha256
from analysis.service_scenarios_v2 import ScenarioExecutionResult, execute_service_scenario
from core.analysis_spec import AnalysisSpec
from core.run_manifest import sha256_file
from transformation.build_network_attachments_v2 import graph_checksum

POLICY = "b6c2_verified_scenario_report_v1"
OUTPUT_FILES = ("baseline_origins.csv", "scenario_origins.csv", "comparison_origins.csv",
                "metrics.json", "scenario_spec.json")


@dataclass(frozen=True, slots=True)
class ScenarioInputPaths:
    origins: Path
    services: Path
    attachments: Path
    nodes: Path
    edges: Path
    addition_services: Path | None = None
    addition_attachments: Path | None = None

    def __post_init__(self) -> None:
        for name in ("origins", "services", "attachments", "nodes", "edges",
                     "addition_services", "addition_attachments"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if (self.addition_services is None) != (self.addition_attachments is None):
            raise ValueError("Addition service and attachment file paths must both be supplied")

    def items(self) -> dict[str, Path]:
        return {name: value for name in (
            "origins", "services", "attachments", "nodes", "edges",
            "addition_services", "addition_attachments")
            if (value := getattr(self, name)) is not None}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        # Keep administrative/source identifiers stable, especially leading zeros.
        return pd.read_csv(path, dtype={key: str for key in (
            "municipality_code", "origin_id", "service_id", "source_record_id",
            "entity_id", "node_id", "source_osm_node", "target_osm_node",
            "graph_checksum", "source_name")})
    if path.suffix.lower() == ".parquet":
        # GeoParquet stores geometries as WKB. Read them as geometries when
        # available, so addition-row hashes match the B6C.1 canonical contract.
        import pyarrow.parquet as pq
        metadata = pq.read_metadata(path).metadata or {}
        if b"geo" in metadata:
            import geopandas as gpd
            return gpd.read_parquet(path)
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format (expected .csv/.parquet): {path}")


def _bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _canonical_csv(frame: pd.DataFrame) -> bytes:
    if "origin_id" not in frame:
        raise ValueError("Scenario output must contain origin_id")
    if frame["origin_id"].isna().any() or frame["origin_id"].astype(str).duplicated().any():
        raise ValueError("Scenario output has missing or duplicate origin_id")
    return frame.sort_values("origin_id", kind="mergesort").to_csv(
        index=False, lineterminator="\n", na_rep="", float_format="%.17g"
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_verified_scenario(
    *, scenario: ScenarioSpec, analysis_spec: AnalysisSpec,
    request: AccessibilityEngineRequest, inputs: ScenarioInputPaths,
) -> tuple[ScenarioExecutionResult, dict[str, Any]]:
    """Hash/read files, build graph from those files, execute B6C.1 in memory.

    No writes. This checks file identity; it cannot establish that an upstream
    operator truthfully certified legacy_usable_for_accessibility.
    """
    paths = inputs.items()
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing B6C.2 {label} input: {path}")
    if scenario.baseline.analysis_spec_sha256 != analysis_spec.spec_hash:
        raise ValueError("AnalysisSpec does not match scenario baseline")
    if scenario.baseline.request_sha256 != request_sha256(request):
        raise ValueError("Request does not match scenario baseline")
    hashes_before = {key: sha256_file(path) for key, path in paths.items()}
    actual_graph_hash = graph_checksum(inputs.nodes, inputs.edges)
    expected = scenario.baseline
    for name in ("origins", "services", "attachments"):
        if hashes_before[name] != getattr(expected, f"{name}_sha256"):
            raise ValueError(f"Baseline {name} SHA-256 mismatch")
    if actual_graph_hash != expected.graph_sha256:
        raise ValueError("Baseline graph node/edge SHA-256 mismatch")

    tables = {key: _read_table(path) for key, path in paths.items()}
    # Detect edits while files are being read. This cannot defend against a
    # hostile filesystem, but fails closed on ordinary concurrent mutations.
    if hashes_before != {key: sha256_file(path) for key, path in paths.items()}:
        raise ValueError("Input files changed during scenario loading")
    graph, graph_stats = build_routing_graph(
        tables["nodes"], tables["edges"],
        travel_time_weight=request.travel_time_weight,
        distance_weight=request.distance_weight,
    )
    result = execute_service_scenario(
        scenario=scenario, analysis_spec=analysis_spec, request=request, graph=graph,
        origins=tables["origins"], services=tables["services"],
        attachments=tables["attachments"],
        verified_input_sha256={
            "origins": hashes_before["origins"],
            "services": hashes_before["services"],
            "attachments": hashes_before["attachments"],
            "graph": actual_graph_hash,
        },
        addition_services=tables.get("addition_services"),
        addition_attachments=tables.get("addition_attachments"),
    )
    provenance = {
        "policy": POLICY,
        "input_paths": {key: str(path.resolve()) for key, path in paths.items()},
        "input_sha256": hashes_before,
        "graph_sha256": actual_graph_hash,
        "graph_statistics": graph_stats,
        "source_verification": "file_bytes_verified_against_scenario_baseline",
        "eligibility_verification": "upstream_review_not_independently_attested",
        "analysis_spec": analysis_spec.to_dict(),
        "routing_request": {
            "service_type": request.service_type.value,
            "mode": request.mode.value,
            "thresholds_min": list(request.thresholds_min),
            "population_selector": request.population_selector.value,
            "travel_time_weight": request.travel_time_weight,
            "distance_weight": request.distance_weight,
            "off_network_speed_m_s": float(request.off_network_speed_m_s),
        },
        "scenario_spec": scenario.to_dict(),
        "scenario_execution": result.provenance,
    }
    return result, provenance


def _report_metrics(result: ScenarioExecutionResult) -> dict[str, Any]:
    """Preserve absolute deltas, add explicit relative % and coverage point deltas."""
    base, current = result.metrics["baseline"], result.metrics["scenario"]
    relative: dict[str, float | None] = {}
    percentage_points: dict[str, float | None] = {}
    for key, before in base.items():
        after = current[key]
        relative[key] = (
            100.0 * (float(after) - float(before)) / abs(float(before))
            if before is not None and after is not None and float(before) != 0
            else None
        )
        if key.startswith("population_coverage_within_"):
            percentage_points[key] = (
                100.0 * (float(after) - float(before))
                if before is not None and after is not None else None
            )
    return {
        **result.metrics,
        "relative_change_percent": relative,
        "coverage_delta_percentage_points": percentage_points,
    }


def _artifacts(result: ScenarioExecutionResult, provenance: dict[str, Any]) -> dict[str, bytes]:
    return {
        "baseline_origins.csv": _canonical_csv(result.baseline.origins),
        "scenario_origins.csv": _canonical_csv(result.scenario.origins),
        "comparison_origins.csv": _canonical_csv(result.per_origin),
        "metrics.json": _bytes(_report_metrics(result)),
        "scenario_spec.json": _bytes(provenance["scenario_spec"]),
    }


def write_scenario_report(
    *, result: ScenarioExecutionResult, provenance: dict[str, Any],
    output_root: Path,
) -> tuple[Path, bool]:
    """Persist immutable output set; cached only when all bytes and metadata agree.

    Returns (report_folder, cached). Never overwrites conflicting results.
    """
    if provenance.get("policy") != POLICY:
        raise ValueError("Expected B6C.2 verified runner provenance")
    if provenance.get("scenario_execution") != result.provenance:
        raise ValueError("Report provenance does not match execution")
    if result.scenario_id != result.provenance.get("scenario_id"):
        raise ValueError("Scenario ID differs from execution")
    baseline = result.provenance["baseline"]
    output_root = Path(output_root)
    dest = output_root / baseline["municipality_code"] / result.scenario_id
    payloads = _artifacts(result, provenance)
    if set(payloads) != set(OUTPUT_FILES):
        raise ValueError("Output schema unexpectedly changed")
    manifest = {
        "policy": POLICY,
        "scenario_id": result.scenario_id,
        "scenario_sha256": result.provenance["scenario_sha256"],
        "input_provenance": provenance,
        "output_sha256": {key: _sha(value) for key, value in payloads.items()},
        "output_logical_sha256": result.provenance["output_sha256"],
        "artifacts": list(OUTPUT_FILES),
        "immutability": "refuse_conflicting_overwrite",
    }
    manifest_bytes = _bytes(manifest)
    if dest.exists():
        if not dest.is_dir() or set(p.name for p in dest.iterdir()) != set(OUTPUT_FILES) | {"manifest.json"}:
            raise FileExistsError("Incomplete or unexpected B6C.2 output already exists")
        for key, data in {**payloads, "manifest.json": manifest_bytes}.items():
            if (dest / key).read_bytes() != data:
                raise FileExistsError(f"B6C.2 immutable report differs: {key}")
        return dest, True

    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{result.scenario_id}_", dir=dest.parent))
    try:
        for key, data in {**payloads, "manifest.json": manifest_bytes}.items():
            (stage / key).write_bytes(data)
        # Do not replace any complete run; first writer wins. Concurrent writers
        # must be serialized by the orchestrator until B10 introduces locking.
        if dest.exists():
            raise FileExistsError("B6C.2 output appeared during publication")
        os.rename(stage, dest)
        return dest, False
    finally:
        if stage.exists():
            shutil.rmtree(stage)
