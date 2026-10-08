from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point

from core.analysis_spec import TransportMode
from core.schema_v2 import NETWORK_ATTACHMENT_V2, ORIGIN_V2, SERVICE_V2


ROOT = Path(__file__).resolve().parents[2]
FEATURES_ACCESSIBILITY_DIR = ROOT / "data" / "features" / "accessibility"
PROCESSED_SERVICES_DIR = ROOT / "data" / "processed" / "services"
PROCESSED_OSM_DIR = ROOT / "data" / "processed" / "osm"
PROCESSED_ATTACHMENTS_DIR = ROOT / "data" / "processed" / "network_attachments"
DEFAULT_MAX_SNAP_DISTANCE_M = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Costruisce NetworkAttachmentV2 mode-specific per OriginV2 e ServiceV2. "
            "Il routing attachment resta separato dalle entita canoniche."
        )
    )
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--mode", choices=[mode.value for mode in TransportMode], required=True)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument(
        "--max-snap-distance-m",
        type=float,
        default=DEFAULT_MAX_SNAP_DISTANCE_M,
        help="Distanza massima di snapping. Default: 1000 m.",
    )
    args = parser.parse_args()
    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    if not (args.municipality_code.isdigit() and len(args.municipality_code) == 6):
        raise ValueError("--municipality-code deve contenere esattamente 6 cifre.")
    if not np.isfinite(args.max_snap_distance_m) or args.max_snap_distance_m <= 0:
        raise ValueError("--max-snap-distance-m deve essere > 0.")
    args.health_reference_date = pd.Timestamp(args.health_reference_date).normalize()
    return args


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_node_id(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    if not value:
        return None
    if value.endswith(".0"):
        value = value[:-2]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_checksum(nodes_path: Path, edges_path: Path) -> str:
    payload = {
        "nodes_sha256": sha256_file(nodes_path),
        "edges_sha256": sha256_file(edges_path),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    health_label = args.health_reference_date.strftime("%Y%m%d")
    return {
        "origins": (
            FEATURES_ACCESSIBILITY_DIR
            / args.municipality_code
            / f"population_network_origins_{args.census_year}.parquet"
        ),
        "services": (
            PROCESSED_SERVICES_DIR
            / args.municipality_code
            / f"service_entities_v2_{args.school_year}_{health_label}.parquet"
        ),
        "nodes": (
            PROCESSED_OSM_DIR
            / args.municipality_code
            / f"{args.mode}_nodes.parquet"
        ),
        "edges": (
            PROCESSED_OSM_DIR
            / args.municipality_code
            / f"{args.mode}_edges.parquet"
        ),
    }


def load_inputs(args: argparse.Namespace):
    paths = input_paths(args)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Input mancante: {path}")
    origins = gpd.read_parquet(paths["origins"])
    services = gpd.read_parquet(paths["services"])
    nodes = gpd.read_parquet(paths["nodes"])
    edges = gpd.read_parquet(paths["edges"])
    return origins, services, nodes, edges, paths


def _require_known_crs(gdf: gpd.GeoDataFrame, label: str) -> None:
    if gdf.crs is None:
        raise ValueError(f"{label} deve avere un CRS esplicito.")


def prepare_network_nodes(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    required = {"source_record_id", "geometry"}
    missing = sorted(required - set(nodes.columns))
    if missing:
        raise ValueError(f"Network nodes missing columns: {missing}")
    _require_known_crs(nodes, "Network nodes")
    result = nodes[["source_record_id", "geometry"]].copy()
    result["node_id"] = result["source_record_id"].map(normalize_node_id)
    if result["node_id"].isna().any():
        raise ValueError("Network nodes contain invalid source_record_id values.")
    if result["node_id"].duplicated().any():
        raise ValueError("Network node_id values must be unique.")
    if result.geometry.isna().any() or (~result.geometry.geom_type.eq("Point")).any():
        raise ValueError("Network nodes must contain non-null Point geometries.")
    return gpd.GeoDataFrame(
        result[["node_id", "geometry"]],
        geometry="geometry",
        crs=nodes.crs,
    )


def build_network_components(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    required = {"source_osm_node", "target_osm_node"}
    missing = sorted(required - set(edges.columns))
    if missing:
        raise ValueError(f"Network edges missing columns: {missing}")

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["node_id"].astype(str).tolist())
    invalid_endpoints: set[str] = set()
    node_ids = set(graph.nodes)

    for row in edges[["source_osm_node", "target_osm_node"]].itertuples(index=False):
        source = normalize_node_id(row.source_osm_node)
        target = normalize_node_id(row.target_osm_node)
        if source is None or target is None:
            raise ValueError("Network edges contain null/invalid endpoints.")
        if source not in node_ids:
            invalid_endpoints.add(source)
        if target not in node_ids:
            invalid_endpoints.add(target)
        if source in node_ids and target in node_ids:
            graph.add_edge(source, target)

    if invalid_endpoints:
        raise ValueError(
            "Network edges reference nodes absent from the node table. "
            f"Examples: {sorted(invalid_endpoints)[:20]}"
        )

    components = list(nx.weakly_connected_components(graph))
    components.sort(
        key=lambda component: (
            -len(component),
            min(str(node_id) for node_id in component) if component else "",
        )
    )

    component_map: dict[str, str] = {}
    size_map: dict[str, int] = {}
    for index, component in enumerate(components, start=1):
        component_id = f"component_{index:06d}"
        size = len(component)
        for node_id in component:
            component_map[str(node_id)] = component_id
            size_map[str(node_id)] = size

    result = nodes.copy()
    result["network_component_id"] = result["node_id"].map(component_map)
    result["network_component_size"] = result["node_id"].map(size_map).astype(int)
    largest_id = "component_000001" if components else None
    result["is_largest_component"] = result["network_component_id"].eq(largest_id)

    summary = {
        "component_count": len(components),
        "largest_component_id": largest_id,
        "largest_component_size": len(components[0]) if components else 0,
    }
    return result, summary


def prepare_origin_points(origins: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ORIGIN_V2.validate_columns(origins.columns)
    if origins["origin_id"].isna().any() or origins["origin_id"].astype(str).duplicated().any():
        raise ValueError("OriginV2 origin_id must be non-null and unique.")
    _require_known_crs(origins, "OriginV2")
    result = origins[["origin_id", "geometry"]].copy()
    result["entity_id"] = result["origin_id"].astype(str)
    return gpd.GeoDataFrame(
        result[["entity_id", "geometry"]],
        geometry="geometry",
        crs=origins.crs,
    )


def prepare_service_points(services: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    SERVICE_V2.validate_columns(services.columns)
    if services["service_id"].isna().any() or services["service_id"].astype(str).duplicated().any():
        raise ValueError("ServiceV2 service_id must be non-null and unique.")

    # ServiceV2 stores canonical WGS84 latitude/longitude. Prefer a valid
    # GeoParquet geometry when present, but rebuild missing points from those
    # explicit coordinates so unresolved services become explicit unsnapped rows.
    if isinstance(services, gpd.GeoDataFrame) and "geometry" in services.columns:
        if services.crs is None:
            raise ValueError("ServiceV2 geometry is present but CRS is missing.")
        result = services[["service_id", "latitude", "longitude", "geometry"]].copy()
        if result.crs.to_epsg() != 4326:
            result = result.to_crs(4326)
    else:
        result = gpd.GeoDataFrame(
            services[["service_id", "latitude", "longitude"]].copy(),
            geometry=[None] * len(services),
            crs="EPSG:4326",
        )

    lat = pd.to_numeric(result["latitude"], errors="coerce")
    lon = pd.to_numeric(result["longitude"], errors="coerce")

    # Missing coordinates are invalid by definition. Filling them only for
    # the bounds comparison avoids pandas/numexpr RuntimeWarning on NaN.
    valid_lat = lat.notna() & lat.fillna(0.0).between(-90, 90)
    valid_lon = lon.notna() & lon.fillna(0.0).between(-180, 180)
    valid_coords = valid_lat & valid_lon
    missing_geometry = result.geometry.isna()
    fill_mask = missing_geometry & valid_coords
    if fill_mask.any():
        result.loc[fill_mask, "geometry"] = [
            Point(x, y)
            for x, y in zip(lon.loc[fill_mask], lat.loc[fill_mask])
        ]

    bad_nonpoint = result.geometry.notna() & (~result.geometry.geom_type.eq("Point"))
    if bad_nonpoint.any():
        raise ValueError("ServiceV2 network attachment requires Point geometries.")

    result["entity_id"] = result["service_id"].astype(str)
    return gpd.GeoDataFrame(
        result[["entity_id", "geometry"]],
        geometry="geometry",
        crs="EPSG:4326",
    )


def choose_metric_crs(nodes: gpd.GeoDataFrame):
    metric_crs = nodes.estimate_utm_crs()
    if metric_crs is None:
        raise ValueError("Unable to derive a metric CRS for network snapping.")
    return metric_crs


def snap_entities(
    entities: gpd.GeoDataFrame,
    nodes_with_components: gpd.GeoDataFrame,
    *,
    entity_kind: str,
    mode: TransportMode,
    max_snap_distance_m: float,
    graph_checksum_value: str,
) -> pd.DataFrame:
    if entity_kind not in {"origin", "service"}:
        raise ValueError("entity_kind must be 'origin' or 'service'.")
    if entities["entity_id"].isna().any() or entities["entity_id"].astype(str).duplicated().any():
        raise ValueError(f"{entity_kind} entity_id must be non-null and unique.")
    _require_known_crs(entities, entity_kind)
    _require_known_crs(nodes_with_components, "Network nodes")
    if not np.isfinite(max_snap_distance_m) or max_snap_distance_m <= 0:
        raise ValueError("max_snap_distance_m must be finite and > 0.")

    metric_crs = choose_metric_crs(nodes_with_components)
    nodes_metric = nodes_with_components.to_crs(metric_crs).copy()
    entities_metric = entities.to_crs(metric_crs).copy().reset_index(drop=True)
    entities_metric["entity_order"] = np.arange(len(entities_metric), dtype=int)

    valid = entities_metric.loc[entities_metric.geometry.notna()].copy()
    missing = entities_metric.loc[entities_metric.geometry.isna()].copy()

    records: list[dict[str, Any]] = []

    if not valid.empty:
        joined = gpd.sjoin_nearest(
            valid[["entity_id", "geometry", "entity_order"]],
            nodes_metric[
                [
                    "node_id",
                    "network_component_id",
                    "is_largest_component",
                    "geometry",
                ]
            ],
            how="left",
            distance_col="snap_distance_m",
        )
        joined["node_id"] = joined["node_id"].map(normalize_node_id)
        joined["snap_distance_m"] = pd.to_numeric(joined["snap_distance_m"], errors="coerce")
        if joined["node_id"].isna().any() or joined["snap_distance_m"].isna().any():
            raise RuntimeError("Nearest-node snapping returned incomplete matches.")
        joined = (
            joined.sort_values(
                ["entity_order", "snap_distance_m", "node_id"],
                kind="mergesort",
            )
            .drop_duplicates(subset=["entity_order"], keep="first")
        )

        for row in joined.itertuples(index=False):
            distance = float(row.snap_distance_m)
            is_snapped = distance <= max_snap_distance_m
            records.append(
                {
                    "entity_order": int(row.entity_order),
                    "entity_id": str(row.entity_id),
                    "entity_kind": entity_kind,
                    "mode": mode.value,
                    "node_id": str(row.node_id) if is_snapped else None,
                    "snapped": bool(is_snapped),
                    "snap_distance_m": distance,
                    "attachment_quality": "snapped" if is_snapped else "snap_outlier",
                    "network_component_id": (
                        str(row.network_component_id) if is_snapped else None
                    ),
                    "is_largest_component": (
                        bool(row.is_largest_component) if is_snapped else None
                    ),
                    "graph_checksum": graph_checksum_value,
                }
            )

    for row in missing.itertuples(index=False):
        records.append(
            {
                "entity_order": int(row.entity_order),
                "entity_id": str(row.entity_id),
                "entity_kind": entity_kind,
                "mode": mode.value,
                "node_id": None,
                "snapped": False,
                "snap_distance_m": np.nan,
                "attachment_quality": "missing_geometry",
                "network_component_id": None,
                "is_largest_component": None,
                "graph_checksum": graph_checksum_value,
            }
        )

    result = pd.DataFrame.from_records(records).sort_values("entity_order")
    result = result.drop(columns="entity_order").reset_index(drop=True)
    NETWORK_ATTACHMENT_V2.validate_columns(result.columns)

    if len(result) != len(entities):
        raise RuntimeError(
            f"Attachment row conservation failed for {entity_kind}: "
            f"entities={len(entities)}, attachments={len(result)}"
        )
    if result.duplicated(["entity_kind", "entity_id", "mode"]).any():
        raise RuntimeError("Duplicate NetworkAttachmentV2 rows were generated.")
    return result


def build_network_attachments(
    origins: gpd.GeoDataFrame,
    services: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    *,
    mode: TransportMode,
    max_snap_distance_m: float,
    graph_checksum_value: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode = TransportMode(mode)
    prepared_nodes = prepare_network_nodes(nodes)
    nodes_with_components, component_summary = build_network_components(prepared_nodes, edges)
    origin_points = prepare_origin_points(origins)
    service_points = prepare_service_points(services)

    origin_attachments = snap_entities(
        origin_points,
        nodes_with_components,
        entity_kind="origin",
        mode=mode,
        max_snap_distance_m=max_snap_distance_m,
        graph_checksum_value=graph_checksum_value,
    )
    service_attachments = snap_entities(
        service_points,
        nodes_with_components,
        entity_kind="service",
        mode=mode,
        max_snap_distance_m=max_snap_distance_m,
        graph_checksum_value=graph_checksum_value,
    )
    combined = pd.concat([origin_attachments, service_attachments], ignore_index=True)
    NETWORK_ATTACHMENT_V2.validate_columns(combined.columns)
    return combined, component_summary


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    health_label = args.health_reference_date.strftime("%Y%m%d")
    directory = PROCESSED_ATTACHMENTS_DIR / args.municipality_code
    directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"network_attachments_v2_{args.mode}_{args.census_year}_"
        f"{args.school_year}_{health_label}"
    )
    return {
        "parquet": directory / f"{stem}.parquet",
        "csv": directory / f"{stem}.csv",
        "manifest": directory / f"{stem}_manifest.json",
    }


def _snap_summary(table: pd.DataFrame, entity_kind: str) -> dict[str, Any]:
    selected = table.loc[table["entity_kind"] == entity_kind].copy()
    snapped = selected.loc[selected["snapped"]].copy()
    distances = pd.to_numeric(snapped["snap_distance_m"], errors="coerce").dropna()
    return {
        "entities": int(len(selected)),
        "snapped": int(selected["snapped"].sum()),
        "unsnapped": int((~selected["snapped"]).sum()),
        "attachment_quality": {
            str(key): int(value)
            for key, value in selected["attachment_quality"].value_counts(dropna=False).items()
        },
        "snap_distance_m": {
            "min": float(distances.min()) if not distances.empty else None,
            "median": float(distances.median()) if not distances.empty else None,
            "mean": float(distances.mean()) if not distances.empty else None,
            "max": float(distances.max()) if not distances.empty else None,
        },
    }


def main() -> None:
    args = parse_args()
    mode = TransportMode(args.mode)
    origins, services, nodes, edges, paths = load_inputs(args)
    checksum = graph_checksum(paths["nodes"], paths["edges"])

    attachments, component_summary = build_network_attachments(
        origins,
        services,
        nodes,
        edges,
        mode=mode,
        max_snap_distance_m=float(args.max_snap_distance_m),
        graph_checksum_value=checksum,
    )

    outputs = output_paths(args)
    attachments.to_parquet(outputs["parquet"], index=False)
    attachments.to_csv(outputs["csv"], index=False, encoding="utf-8-sig")

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "municipality_code": args.municipality_code,
        "mode": mode.value,
        "census_year": str(args.census_year),
        "school_year": str(args.school_year),
        "health_reference_date": args.health_reference_date.strftime("%Y-%m-%d"),
        "max_snap_distance_m": float(args.max_snap_distance_m),
        "graph_checksum": checksum,
        "input_paths": {key: str(value) for key, value in paths.items()},
        "input_sha256": {key: sha256_file(value) for key, value in paths.items()},
        "network_components": component_summary,
        "origin_attachments": _snap_summary(attachments, "origin"),
        "service_attachments": _snap_summary(attachments, "service"),
        "notes": {
            "attachment_authority": (
                "NetworkAttachmentV2 is the authoritative mode-specific routing attachment; "
                "legacy routing columns on canonical entities are ignored downstream."
            ),
            "current_origin_representation": (
                "Phase B4 compatibility uses the currently migrated OriginV2 geometry. "
                "The methodology-v2 representative-point baseline is a later migration and "
                "these drive attachments are not yet final thesis outputs."
            ),
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n====================================")
    print(" NETWORK ATTACHMENTS V2")
    print("====================================")
    print(f"Comune: {args.municipality_code}")
    print(f"Mode: {mode.value}")
    print(f"Max snap distance: {args.max_snap_distance_m:.0f} m")
    print(f"Network components: {component_summary['component_count']}")
    print(f"Largest component: {component_summary['largest_component_size']} nodes")

    for kind in ("origin", "service"):
        summary = _snap_summary(attachments, kind)
        print(f"\n{kind.upper()}")
        print(f"Entities: {summary['entities']}")
        print(f"Snapped: {summary['snapped']}")
        print(f"Unsnapped: {summary['unsnapped']}")
        print("Attachment quality:")
        print(
            attachments.loc[attachments["entity_kind"] == kind, "attachment_quality"]
            .value_counts(dropna=False)
            .to_string()
        )
        distances = pd.to_numeric(
            attachments.loc[
                (attachments["entity_kind"] == kind) & attachments["snapped"],
                "snap_distance_m",
            ],
            errors="coerce",
        ).dropna()
        if not distances.empty:
            print(
                "Snap distance [m] min/median/mean/max: "
                f"{distances.min():.2f} / {distances.median():.2f} / "
                f"{distances.mean():.2f} / {distances.max():.2f}"
            )

    print("\n=== OUTPUT ===")
    print(f"✓ {outputs['parquet']}")
    print(f"✓ {outputs['csv']}")
    print(f"✓ {outputs['manifest']}")


if __name__ == "__main__":
    main()
