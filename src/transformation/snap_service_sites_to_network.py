import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_OSM_DIR = ROOT / "data" / "processed" / "osm"
PROCESSED_SERVICES_DIR = ROOT / "data" / "processed" / "services"
FEATURES_ACCESSIBILITY_DIR = ROOT / "data" / "features" / "accessibility"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggancia i service sites canonici ai nodi più vicini "
            "della rete pedonale OSM e conserva distanza di snapping "
            "e componente di rete."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre.",
    )

    parser.add_argument(
        "--school-year",
        default="202425",
        help="Anno scolastico del canonical service layer.",
    )

    parser.add_argument(
        "--health-reference-date",
        default="2025-06-30",
        help="Data snapshot Health YYYY-MM-DD.",
    )

    parser.add_argument(
        "--service-layer",
        choices=["enriched", "osm_only"],
        default="enriched",
        help=(
            "Layer dei servizi da agganciare alla rete. "
            "Default: enriched."
        ),
    )

    parser.add_argument(
        "--max-snap-distance-m",
        type=float,
        default=1000.0,
        help=(
            "Distanza oltre la quale il servizio viene marcato "
            "come snap_outlier. Default: 1000 m."
        ),
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code).strip().zfill(6)
    )

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "--municipality-code deve avere esattamente 6 cifre."
        )

    args.health_reference_date = pd.Timestamp(
        args.health_reference_date
    ).normalize()

    if args.max_snap_distance_m <= 0:
        raise ValueError(
            "--max-snap-distance-m deve essere > 0."
        )

    return args


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def normalize_node_id(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


def load_inputs(args):
    label = (
        args.health_reference_date
        .strftime("%Y%m%d")
    )

    if args.service_layer == "osm_only":
        service_path = (
            PROCESSED_SERVICES_DIR
            / args.municipality_code
            / "service_sites_osm_only.parquet"
        )
    else:
        service_path = (
            PROCESSED_SERVICES_DIR
            / args.municipality_code
            / (
                f"service_sites_"
                f"{args.school_year}_"
                f"{label}.parquet"
            )
        )

    nodes_path = (
        PROCESSED_OSM_DIR
        / args.municipality_code
        / "walk_nodes.parquet"
    )

    edges_path = (
        PROCESSED_OSM_DIR
        / args.municipality_code
        / "walk_edges.parquet"
    )

    for path in [
        service_path,
        nodes_path,
        edges_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input mancante: {path}"
            )

    services = gpd.read_parquet(
        service_path
    )

    nodes = gpd.read_parquet(
        nodes_path
    )

    edges = gpd.read_parquet(
        edges_path
    )

    if services.crs is None:
        raise RuntimeError(
            "Canonical service layer senza CRS."
        )

    if nodes.crs is None:
        raise RuntimeError(
            "OSM nodes senza CRS."
        )

    return (
        services,
        nodes,
        edges,
        {
            "services": str(service_path),
            "nodes": str(nodes_path),
            "edges": str(edges_path),
        },
    )


def prepare_nodes(nodes):
    required = {
        "source_record_id",
        "geometry",
    }

    missing = required - set(nodes.columns)

    if missing:
        raise RuntimeError(
            "Colonne OSM nodes mancanti: "
            + ", ".join(sorted(missing))
        )

    nodes = nodes.copy()

    if nodes.empty:
        raise RuntimeError(
            "Dataset OSM nodes vuoto."
        )

    nodes["network_node_id"] = (
        nodes["source_record_id"]
        .map(normalize_node_id)
    )

    if nodes["network_node_id"].isna().any():
        raise RuntimeError(
            "Alcuni nodi non hanno network_node_id valido."
        )

    if nodes["network_node_id"].duplicated().any():
        raise RuntimeError(
            "network_node_id duplicati."
        )

    return nodes[
        [
            "network_node_id",
            "geometry",
        ]
    ].copy()


def add_network_components(
    nodes,
    edges,
):
    required = {
        "source_osm_node",
        "target_osm_node",
    }

    missing = required - set(edges.columns)

    if missing:
        raise RuntimeError(
            "Colonne OSM edges mancanti: "
            + ", ".join(sorted(missing))
        )

    graph = nx.DiGraph()

    node_ids = set(
        nodes[
            "network_node_id"
        ].astype(str)
    )

    graph.add_nodes_from(
        node_ids
    )

    invalid_endpoints = set()

    for _, edge in edges.iterrows():
        source = normalize_node_id(
            edge[
                "source_osm_node"
            ]
        )

        target = normalize_node_id(
            edge[
                "target_osm_node"
            ]
        )

        if (
            source is None
            or target is None
        ):
            continue

        if source not in node_ids:
            invalid_endpoints.add(source)

        if target not in node_ids:
            invalid_endpoints.add(target)

        if (
            source in node_ids
            and target in node_ids
        ):
            graph.add_edge(
                source,
                target,
            )

    if invalid_endpoints:
        sample = sorted(
            invalid_endpoints
        )[:20]

        raise RuntimeError(
            "Gli edge OSM contengono endpoint assenti "
            "dal dataset dei nodi. "
            f"Totale endpoint incoerenti: "
            f"{len(invalid_endpoints)}; "
            f"esempi: {sample}"
        )

    components = list(
        nx.weakly_connected_components(
            graph
        )
    )

    # Deterministic component ordering:
    # largest components first; equal-sized components are ordered
    # by their lexicographically smallest network node ID.
    components.sort(
        key=lambda component: (
            -len(component),
            min(
                str(node_id)
                for node_id in component
            ),
        ),
    )

    component_map = {}
    component_size_map = {}

    for index, component in enumerate(
        components,
        start=1,
    ):
        component_id = (
            f"WCC_{index:04d}"
        )

        for node_id in component:
            component_map[
                node_id
            ] = component_id

            component_size_map[
                node_id
            ] = len(component)

    result = nodes.copy()

    result[
        "network_component_id"
    ] = (
        result[
            "network_node_id"
        ]
        .map(component_map)
    )

    result[
        "network_component_size"
    ] = (
        result[
            "network_node_id"
        ]
        .map(component_size_map)
        .fillna(1)
        .astype(int)
    )

    largest_component_id = (
        "WCC_0001"
        if components
        else None
    )

    result[
        "is_largest_component"
    ] = (
        result[
            "network_component_id"
        ]
        == largest_component_id
    )

    return (
        result,
        {
            "component_count":
                int(
                    len(
                        components
                    )
                ),

            "largest_component_id":
                largest_component_id,

            "largest_component_size":
                int(
                    len(
                        components[0]
                    )
                )
                if components
                else 0,
        },
    )


def choose_metric_crs(gdf):
    metric_crs = (
        gdf.estimate_utm_crs()
    )

    if metric_crs is None:
        raise RuntimeError(
            "Impossibile stimare CRS metrico."
        )

    return metric_crs


def snap_services(
    services,
    nodes,
    metric_crs,
    max_snap_distance_m,
):
    usable = services.loc[
        services[
            "usable_for_accessibility"
        ]
    ].copy()

    unusable = services.loc[
        ~services[
            "usable_for_accessibility"
        ]
    ].copy()

    if usable.empty:
        raise RuntimeError(
            "Nessun service site utilizzabile."
        )

    services_metric = (
        usable.to_crs(
            metric_crs
        )
    )

    nodes_metric = (
        nodes.to_crs(
            metric_crs
        )
    )

    joined = gpd.sjoin_nearest(
        services_metric,
        nodes_metric[
            [
                "network_node_id",
                "network_component_id",
                "network_component_size",
                "is_largest_component",
                "geometry",
            ]
        ],
        how="left",
        distance_col="snap_distance_m",
    )

    joined = (
        joined.sort_values(
            [
                "service_site_id",
                "snap_distance_m",
                "network_node_id",
            ]
        )
        .drop_duplicates(
            subset=[
                "service_site_id",
            ],
            keep="first",
        )
        .copy()
    )

    snap_distance = pd.to_numeric(
        joined["snap_distance_m"],
        errors="coerce",
    )

    if snap_distance.isna().any():
        raise RuntimeError(
            "Alcuni servizi utilizzabili non hanno "
            "snap_distance_m valido."
        )

    if (snap_distance < 0).any():
        raise RuntimeError(
            "Alcuni servizi hanno snap_distance_m negativo."
        )

    joined["snap_distance_m"] = (
        snap_distance.astype(float)
    )

    joined[
        "snap_status"
    ] = "snapped"

    joined.loc[
        joined[
            "snap_distance_m"
        ]
        > max_snap_distance_m,
        "snap_status",
    ] = "snap_outlier"

    joined = joined.to_crs(
        "EPSG:4326"
    )

    # Keep original service geometry but attach node geometry separately.
    node_geometry_map = (
        nodes.set_index(
            "network_node_id"
        )[
            "geometry"
        ]
        .to_dict()
    )

    joined[
        "snapped_node_geometry_wkt"
    ] = (
        joined[
            "network_node_id"
        ]
        .map(
            node_geometry_map
        )
        .map(
            lambda geom:
                geom.wkt
                if geom is not None
                else None
        )
    )

    if not unusable.empty:
        unusable = unusable.copy()

        unusable[
            "network_node_id"
        ] = None

        unusable[
            "network_component_id"
        ] = None

        unusable[
            "network_component_size"
        ] = None

        unusable[
            "is_largest_component"
        ] = False

        unusable[
            "snap_distance_m"
        ] = None

        unusable[
            "snap_status"
        ] = "not_usable"

        unusable[
            "snapped_node_geometry_wkt"
        ] = None

        joined = pd.concat(
            [
                joined,
                unusable,
            ],
            ignore_index=True,
            sort=False,
        )

        joined = gpd.GeoDataFrame(
            joined,
            geometry="geometry",
            crs="EPSG:4326",
        )

    return joined


def validate_results(
    services,
    snapped,
):
    if (
        len(
            snapped
        )
        != len(
            services
        )
    ):
        raise RuntimeError(
            "Numero righe finale diverso dal canonical layer."
        )

    duplicates = (
        snapped[
            "service_site_id"
        ]
        .duplicated()
    )

    if duplicates.any():
        raise RuntimeError(
            "service_site_id duplicati dopo snapping."
        )

    missing_node_for_usable = (
        snapped[
            "usable_for_accessibility"
        ]
        & snapped[
            "network_node_id"
        ]
        .isna()
    )

    if missing_node_for_usable.any():
        raise RuntimeError(
            "Alcuni servizi utilizzabili non sono stati "
            "agganciati a un nodo."
        )


def main():
    args = parse_args()

    (
        services,
        nodes,
        edges,
        input_paths,
    ) = load_inputs(
        args
    )

    nodes = prepare_nodes(
        nodes
    )

    (
        nodes,
        network_summary,
    ) = add_network_components(
        nodes,
        edges,
    )

    metric_crs = choose_metric_crs(
        nodes
    )

    snapped = snap_services(
        services,
        nodes,
        metric_crs,
        args.max_snap_distance_m,
    )

    validate_results(
        services,
        snapped,
    )

    output_dir = (
        FEATURES_ACCESSIBILITY_DIR
        / args.municipality_code
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label = (
        args.health_reference_date
        .strftime("%Y%m%d")
    )

    if args.service_layer == "osm_only":
        output_stem = "service_network_nodes_osm_only"
    else:
        output_stem = (
            "service_network_nodes_"
            f"{args.school_year}_"
            f"{label}"
        )

    parquet_path = (
        output_dir
        / f"{output_stem}.parquet"
    )

    csv_path = (
        output_dir
        / f"{output_stem}.csv"
    )

    manifest_path = (
        output_dir
        / f"{output_stem}_manifest.json"
    )

    snapped.to_parquet(
        parquet_path,
        index=False,
    )

    csv_df = snapped.drop(
        columns=[
            "geometry",
        ],
        errors="ignore",
    )

    csv_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    usable = snapped.loc[
        snapped[
            "usable_for_accessibility"
        ]
    ]

    outlier_count = int(
        (
            usable[
                "snap_status"
            ]
            == "snap_outlier"
        )
        .sum()
    )

    on_lcc = int(
        usable[
            "is_largest_component"
        ]
        .sum()
    )

    manifest = {
        "generated_at_utc":
            utc_now_iso(),

        "municipality_code":
            args.municipality_code,

        "service_layer":
            args.service_layer,

        "inputs":
            input_paths,

        "metric_crs":
            str(
                metric_crs
            ),

        "network":
            network_summary,

        "service_sites_total":
            int(
                len(
                    snapped
                )
            ),

        "service_sites_usable":
            int(
                len(
                    usable
                )
            ),

        "service_sites_on_largest_component":
            on_lcc,

        "service_sites_outside_largest_component":
            int(
                len(
                    usable
                )
                - on_lcc
            ),

        "snap_outlier_count":
            outlier_count,

        "max_snap_distance_m":
            float(
                args.max_snap_distance_m
            ),

        "snap_distance_m": {
            "min":
                float(
                    usable[
                        "snap_distance_m"
                    ]
                    .min()
                ),

            "median":
                float(
                    usable[
                        "snap_distance_m"
                    ]
                    .median()
                ),

            "mean":
                float(
                    usable[
                        "snap_distance_m"
                    ]
                    .mean()
                ),

            "max":
                float(
                    usable[
                        "snap_distance_m"
                    ]
                    .max()
                ),
        },

        "methodology": {
            "snapping":
                (
                    "Each usable service site is snapped to the "
                    "nearest pedestrian-network node in a metric CRS."
                ),

            "components":
                (
                    "The weakly connected component of the snapped node "
                    "is preserved to support explicit reachability checks "
                    "during accessibility analysis."
                ),

            "outliers":
                (
                    "Services farther than max_snap_distance_m from their "
                    "nearest pedestrian node are not silently discarded; "
                    "they are marked as snap_outlier for QA."
                ),
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n===================================="
    )
    print(
        " SERVICE SITES -> NETWORK NODES"
    )
    print(
        "===================================="
    )

    print(
        f"Comune: {args.municipality_code}"
    )

    print(
        f"Service layer: {args.service_layer}"
    )

    print(
        "Service sites totali: "
        f"{len(snapped)}"
    )

    print(
        "Usabili: "
        f"{len(usable)}"
    )

    print(
        "Su largest component: "
        f"{on_lcc}"
    )

    print(
        "Fuori largest component: "
        f"{len(usable) - on_lcc}"
    )

    print(
        "\nSnap distance [m]:"
    )

    print(
        "  min: "
        f"{usable['snap_distance_m'].min():.2f}"
    )

    print(
        "  median: "
        f"{usable['snap_distance_m'].median():.2f}"
    )

    print(
        "  mean: "
        f"{usable['snap_distance_m'].mean():.2f}"
    )

    print(
        "  max: "
        f"{usable['snap_distance_m'].max():.2f}"
    )

    print(
        "Snap outliers > "
        f"{args.max_snap_distance_m:.0f} m: "
        f"{outlier_count}"
    )

    print(
        "\nPer categoria:"
    )

    print(
        usable[
            "category"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nPer sottocategoria:"
    )

    print(
        usable[
            "subcategory"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ {parquet_path}"
    )

    print(
        f"✓ {csv_path}"
    )

    print(
        f"✓ {manifest_path}"
    )


if __name__ == "__main__":
    main()
