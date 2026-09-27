import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_OSM_DIR = ROOT / "data" / "processed" / "osm"
FEATURES_ACCESSIBILITY_DIR = ROOT / "data" / "features" / "accessibility"

DEFAULT_THRESHOLDS_MIN = [10, 15, 20]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calcola l'accessibilità pedonale network-based dai population "
            "origin nodes ai service sites agganciati alla rete. "
            "Le distanze di snapping di origine e servizio sono incluse."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre.",
    )

    parser.add_argument(
        "--census-year",
        default="2023",
        help="Anno osservazioni ISTAT. Default: 2023.",
    )

    parser.add_argument(
        "--school-year",
        default="202425",
        help="Anno scolastico del service layer. Default: 202425.",
    )

    parser.add_argument(
        "--health-reference-date",
        default="2025-06-30",
        help="Data snapshot Health YYYY-MM-DD. Default: 2025-06-30.",
    )

    parser.add_argument(
        "--service-layer",
        choices=["enriched", "osm_only"],
        default="enriched",
        help=(
            "Layer dei servizi da usare nell'analisi. "
            "Default: enriched."
        ),
    )

    parser.add_argument(
        "--walking-speed-m-s",
        type=float,
        default=1.4,
        help=(
            "Velocità pedonale attesa usata per i segmenti di snapping. "
            "La rete deve essere coerente con questo valore. Default: 1.4."
        ),
    )

    parser.add_argument(
        "--thresholds-min",
        nargs="+",
        type=int,
        default=DEFAULT_THRESHOLDS_MIN,
        help="Soglie di accessibilità in minuti. Default: 10 15 20.",
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

    args.health_reference_date = (
        pd.Timestamp(args.health_reference_date).normalize()
    )

    args.thresholds_min = sorted(
        set(args.thresholds_min)
    )

    if (
        not args.thresholds_min
        or any(
            threshold <= 0
            for threshold in args.thresholds_min
        )
    ):
        raise ValueError(
            "--thresholds-min deve contenere valori positivi."
        )

    if (
        not math.isfinite(args.walking_speed_m_s)
        or args.walking_speed_m_s <= 0
    ):
        raise ValueError(
            "--walking-speed-m-s deve essere > 0."
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
    accessibility_dir = (
        FEATURES_ACCESSIBILITY_DIR
        / args.municipality_code
    )

    health_label = (
        args.health_reference_date
        .strftime("%Y%m%d")
    )

    origins_path = (
        accessibility_dir
        / (
            "population_network_origins_"
            f"{args.census_year}.parquet"
        )
    )

    if args.service_layer == "osm_only":
        services_path = (
            accessibility_dir
            / "service_network_nodes_osm_only.parquet"
        )
    else:
        services_path = (
            accessibility_dir
            / (
                "service_network_nodes_"
                f"{args.school_year}_"
                f"{health_label}.parquet"
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
        origins_path,
        services_path,
        nodes_path,
        edges_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input mancante: {path}"
            )

    origins = gpd.read_parquet(
        origins_path
    )

    services = gpd.read_parquet(
        services_path
    )

    nodes = gpd.read_parquet(
        nodes_path
    )

    edges = gpd.read_parquet(
        edges_path
    )

    return (
        origins,
        services,
        nodes,
        edges,
        {
            "origins": str(origins_path),
            "services": str(services_path),
            "nodes": str(nodes_path),
            "edges": str(edges_path),
        },
    )


def build_graph(nodes, edges):
    required_node_columns = {
        "source_record_id",
    }

    missing_nodes = (
        required_node_columns
        - set(nodes.columns)
    )

    if missing_nodes:
        raise RuntimeError(
            "Colonne OSM nodes mancanti: "
            + ", ".join(
                sorted(missing_nodes)
            )
        )

    required_edge_columns = {
        "source_osm_node",
        "target_osm_node",
        "walking_time_s",
        "length_m",
    }

    missing = (
        required_edge_columns
        - set(edges.columns)
    )

    if missing:
        raise RuntimeError(
            "Colonne OSM edges mancanti: "
            + ", ".join(
                sorted(missing)
            )
        )

    node_ids = [
        node_id
        for node_id in (
            nodes["source_record_id"]
            .map(normalize_node_id)
        )
        if node_id is not None
    ]

    if not node_ids:
        raise RuntimeError(
            "Dataset OSM nodes privo di identificativi validi."
        )

    if len(node_ids) != len(set(node_ids)):
        raise RuntimeError(
            "source_record_id duplicati nel dataset OSM nodes."
        )

    node_id_set = set(node_ids)

    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)

    bad_edges = 0
    duplicate_pairs = 0
    invalid_endpoints = set()

    for _, edge in edges.iterrows():
        source = normalize_node_id(
            edge["source_osm_node"]
        )

        target = normalize_node_id(
            edge["target_osm_node"]
        )

        try:
            walking_time_s = float(
                edge["walking_time_s"]
            )
            length_m = float(
                edge["length_m"]
            )
        except Exception:
            bad_edges += 1
            continue

        if (
            source is None
            or target is None
            or not math.isfinite(
                walking_time_s
            )
            or not math.isfinite(
                length_m
            )
            or walking_time_s < 0
            or length_m < 0
        ):
            bad_edges += 1
            continue

        if source not in node_id_set:
            invalid_endpoints.add(source)

        if target not in node_id_set:
            invalid_endpoints.add(target)

        if (
            source not in node_id_set
            or target not in node_id_set
        ):
            continue

        if graph.has_edge(
            source,
            target,
        ):
            duplicate_pairs += 1

            current = graph[
                source
            ][target]

            current_pair = (
                float(
                    current[
                        "walking_time_s"
                    ]
                ),
                float(
                    current[
                        "length_m"
                    ]
                ),
            )

            candidate_pair = (
                walking_time_s,
                length_m,
            )

            if candidate_pair < current_pair:
                current[
                    "walking_time_s"
                ] = walking_time_s

                current[
                    "length_m"
                ] = length_m

        else:
            graph.add_edge(
                source,
                target,
                walking_time_s=walking_time_s,
                length_m=length_m,
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

    if graph.number_of_edges() == 0:
        raise RuntimeError(
            "Grafo pedonale senza archi validi."
        )

    return (
        graph,
        {
            "nodes":
                int(
                    graph.number_of_nodes()
                ),

            "edges":
                int(
                    graph.number_of_edges()
                ),

            "bad_edges_skipped":
                int(
                    bad_edges
                ),

            "duplicate_directed_pairs_collapsed":
                int(
                    duplicate_pairs
                ),
        },
    )


def prepare_origins(origins):
    required = {
        "census_section_code",
        "network_node_id",
        "assigned_population",
        "assignment_method",
    }

    missing = required - set(
        origins.columns
    )

    if missing:
        raise RuntimeError(
            "Colonne origins mancanti: "
            + ", ".join(
                sorted(missing)
            )
        )

    df = origins.copy()

    df["network_node_id"] = (
        df["network_node_id"]
        .map(normalize_node_id)
    )

    if df["network_node_id"].isna().any():
        raise RuntimeError(
            "Alcuni population origins non hanno network_node_id valido."
        )

    assigned_population = pd.to_numeric(
        df["assigned_population"],
        errors="coerce",
    )

    if assigned_population.isna().any():
        raise RuntimeError(
            "Alcuni population origins hanno assigned_population mancante "
            "o non numerica."
        )

    if (assigned_population < 0).any():
        raise RuntimeError(
            "Alcuni population origins hanno assigned_population negativa."
        )

    df["assigned_population"] = (
        assigned_population.astype(float)
    )

    if (
        "fallback_snap_distance_m"
        in df.columns
    ):
        fallback = pd.to_numeric(
            df[
                "fallback_snap_distance_m"
            ],
            errors="coerce",
        )

        fallback_rows = (
            df["assignment_method"]
            == "representative_point_nearest_node"
        )

        invalid_fallback = (
            fallback_rows
            & fallback.isna()
        )

        if invalid_fallback.any():
            raise RuntimeError(
                "Alcuni origins assegnati tramite fallback non hanno "
                "fallback_snap_distance_m valido."
            )

        negative_fallback = (
            fallback.notna()
            & (fallback < 0)
        )

        if negative_fallback.any():
            raise RuntimeError(
                "Alcuni origins hanno fallback_snap_distance_m negativo."
            )

        fallback = fallback.fillna(0.0)

    else:
        fallback_rows = (
            df["assignment_method"]
            == "representative_point_nearest_node"
        )

        if fallback_rows.any():
            raise RuntimeError(
                "fallback_snap_distance_m assente nonostante la presenza "
                "di origins assegnati tramite fallback."
            )

        fallback = pd.Series(
            0.0,
            index=df.index,
        )

    df[
        "origin_snap_distance_m"
    ] = fallback.astype(float)

    return df.reset_index(
        drop=True
    )


def prepare_services(services):
    required = {
        "service_site_id",
        "category",
        "subcategory",
        "network_node_id",
        "snap_distance_m",
        "usable_for_accessibility",
    }

    missing = required - set(
        services.columns
    )

    if missing:
        raise RuntimeError(
            "Colonne services mancanti: "
            + ", ".join(
                sorted(missing)
            )
        )

    df = services.loc[
        services[
            "usable_for_accessibility"
        ]
    ].copy()

    if df.empty:
        raise RuntimeError(
            "Nessun servizio utilizzabile per accessibility."
        )

    df["network_node_id"] = (
        df["network_node_id"]
        .map(normalize_node_id)
    )

    if (
        df["network_node_id"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Alcuni servizi utilizzabili non hanno network_node_id."
        )

    snap_distance = pd.to_numeric(
        df["snap_distance_m"],
        errors="coerce",
    )

    if snap_distance.isna().any():
        raise RuntimeError(
            "Alcuni servizi utilizzabili hanno snap_distance_m "
            "mancante o non numerica."
        )

    if (snap_distance < 0).any():
        raise RuntimeError(
            "Alcuni servizi utilizzabili hanno snap_distance_m negativo."
        )

    df["snap_distance_m"] = (
        snap_distance.astype(float)
    )

    return df.reset_index(
        drop=True
    )


def validate_network_membership(
    graph,
    origins,
    services,
):
    graph_nodes = set(
        graph.nodes
    )

    origin_nodes = set(
        origins["network_node_id"]
        .dropna()
        .astype(str)
    )

    service_nodes = set(
        services["network_node_id"]
        .dropna()
        .astype(str)
    )

    missing_origins = (
        origin_nodes
        - graph_nodes
    )

    if missing_origins:
        raise RuntimeError(
            "Alcuni origin nodes non appartengono al grafo pedonale. "
            f"Totale: {len(missing_origins)}; "
            f"esempi: {sorted(missing_origins)[:20]}"
        )

    missing_services = (
        service_nodes
        - graph_nodes
    )

    if missing_services:
        raise RuntimeError(
            "Alcuni service nodes non appartengono al grafo pedonale. "
            f"Totale: {len(missing_services)}; "
            f"esempi: {sorted(missing_services)[:20]}"
        )


def build_service_groups(services):
    """
    Definizioni analitiche generiche:
    - school: tutti i servizi Education;
    - pharmacy: subcategory pharmacy;
    - hospital: subcategory hospital.
    """
    groups = {
        "school":
            services.loc[
                services[
                    "category"
                ]
                == "education"
            ].copy(),

        "pharmacy":
            services.loc[
                services[
                    "subcategory"
                ]
                == "pharmacy"
            ].copy(),

        "hospital":
            services.loc[
                services[
                    "subcategory"
                ]
                == "hospital"
            ].copy(),
    }

    return groups


def initialize_group_metrics(
    row_count,
    thresholds,
):
    return {
        "nearest_time_s":
            np.full(
                row_count,
                np.inf,
                dtype=float,
            ),

        "nearest_distance_m":
            np.full(
                row_count,
                np.inf,
                dtype=float,
            ),

        "nearest_service_site_id":
            np.full(
                row_count,
                None,
                dtype=object,
            ),

        "counts": {
            threshold:
                np.zeros(
                    row_count,
                    dtype=np.int32,
                )
            for threshold
            in thresholds
        },
    }


def origin_index_by_node(origins):
    mapping = defaultdict(
        list
    )

    for index, node_id in enumerate(
        origins[
            "network_node_id"
        ]
    ):
        mapping[
            node_id
        ].append(
            index
        )

    return mapping


def infer_walking_speed(edges):
    valid = edges.loc[
        pd.to_numeric(
            edges[
                "walking_time_s"
            ],
            errors="coerce",
        )
        > 0
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "Impossibile inferire walking speed."
        )

    lengths = pd.to_numeric(
        valid[
            "length_m"
        ],
        errors="coerce",
    )

    times = pd.to_numeric(
        valid[
            "walking_time_s"
        ],
        errors="coerce",
    )

    speeds = (
        lengths
        / times
    )

    speeds = speeds.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if speeds.empty:
        raise RuntimeError(
            "Walking speed non inferibile."
        )

    median_speed = float(
        speeds.median()
    )

    p05 = float(
        speeds.quantile(
            0.05
        )
    )

    p95 = float(
        speeds.quantile(
            0.95
        )
    )

    if median_speed <= 0:
        raise RuntimeError(
            "Walking speed non valida."
        )

    return (
        median_speed,
        {
            "median_m_s":
                median_speed,

            "p05_m_s":
                p05,

            "p95_m_s":
                p95,
        },
    )


def validate_walking_speed_consistency(
    edges,
    expected_speed_m_s,
    atol=1e-6,
    rtol=1e-6,
):
    lengths = pd.to_numeric(
        edges["length_m"],
        errors="coerce",
    )

    times = pd.to_numeric(
        edges["walking_time_s"],
        errors="coerce",
    )

    valid = (
        lengths.notna()
        & times.notna()
        & np.isfinite(lengths)
        & np.isfinite(times)
        & (lengths >= 0)
        & (times > 0)
    )

    if not valid.any():
        raise RuntimeError(
            "Nessun arco valido per verificare walking speed."
        )

    speeds = (
        lengths.loc[valid]
        / times.loc[valid]
    ).astype(float)

    consistent = np.isclose(
        speeds.to_numpy(),
        float(expected_speed_m_s),
        atol=atol,
        rtol=rtol,
    )

    inconsistent_count = int(
        (~consistent).sum()
    )

    if inconsistent_count:
        deviations = np.abs(
            speeds.to_numpy()
            - float(expected_speed_m_s)
        )

        raise RuntimeError(
            "La rete pedonale non è coerente con "
            f"--walking-speed-m-s={expected_speed_m_s}. "
            f"Archi incoerenti: {inconsistent_count}/{len(speeds)}; "
            f"deviazione massima={float(deviations.max()):.9f} m/s."
        )

    return {
        "expected_m_s":
            float(expected_speed_m_s),

        "validated_edge_count":
            int(len(speeds)),

        "inconsistent_edge_count":
            0,
    }


def calculate_group_accessibility_full(
    reverse_graph,
    origins,
    services,
    thresholds_min,
    walking_speed_m_s,
):
    row_count = len(
        origins
    )

    metrics = (
        initialize_group_metrics(
            row_count,
            thresholds_min,
        )
    )

    origins_by_node = (
        origin_index_by_node(
            origins
        )
    )

    origin_snap_m = (
        origins[
            "origin_snap_distance_m"
        ]
        .to_numpy(
            dtype=float
        )
    )

    for _, service in services.iterrows():
        service_node = (
            service[
                "network_node_id"
            ]
        )

        if service_node not in reverse_graph:
            continue

        service_snap_m = float(
            service[
                "snap_distance_m"
            ]
        )

        service_snap_s = (
            service_snap_m
            / walking_speed_m_s
        )

        service_id = (
            service[
                "service_site_id"
            ]
        )

        time_lengths = (
            nx.single_source_dijkstra_path_length(
                reverse_graph,
                source=service_node,
                weight="walking_time_s",
            )
        )

        distance_lengths = (
            nx.single_source_dijkstra_path_length(
                reverse_graph,
                source=service_node,
                weight="length_m",
            )
        )

        for (
            origin_node,
            row_indices,
        ) in origins_by_node.items():
            if origin_node not in time_lengths:
                continue

            network_time_s = float(
                time_lengths[
                    origin_node
                ]
            )

            network_distance_m = float(
                distance_lengths[
                    origin_node
                ]
            )

            for row_index in row_indices:
                origin_snap_distance_m = float(
                    origin_snap_m[
                        row_index
                    ]
                )

                origin_snap_s = (
                    origin_snap_distance_m
                    / walking_speed_m_s
                )

                total_time_s = (
                    origin_snap_s
                    + network_time_s
                    + service_snap_s
                )

                total_distance_m = (
                    origin_snap_distance_m
                    + network_distance_m
                    + service_snap_m
                )

                if (
                    total_time_s
                    < metrics[
                        "nearest_time_s"
                    ][row_index]
                ):
                    metrics[
                        "nearest_time_s"
                    ][row_index] = (
                        total_time_s
                    )

                    metrics[
                        "nearest_distance_m"
                    ][row_index] = (
                        total_distance_m
                    )

                    metrics[
                        "nearest_service_site_id"
                    ][row_index] = (
                        service_id
                    )

                for threshold in thresholds_min:
                    if (
                        total_time_s
                        <= threshold * 60.0
                    ):
                        metrics[
                            "counts"
                        ][threshold][
                            row_index
                        ] += 1

    return metrics


def attach_group_metrics(
    origins,
    group_name,
    metrics,
    thresholds_min,
):
    result = origins.copy()

    nearest_time = (
        metrics[
            "nearest_time_s"
        ]
    )

    nearest_distance = (
        metrics[
            "nearest_distance_m"
        ]
    )

    reachable = (
        np.isfinite(
            nearest_time
        )
    )

    result[
        f"{group_name}_reachable"
    ] = reachable

    result[
        f"{group_name}_nearest_time_min"
    ] = np.where(
        reachable,
        nearest_time / 60.0,
        np.nan,
    )

    result[
        f"{group_name}_nearest_distance_m"
    ] = np.where(
        reachable,
        nearest_distance,
        np.nan,
    )

    result[
        f"{group_name}_nearest_service_site_id"
    ] = metrics[
        "nearest_service_site_id"
    ]

    for threshold in thresholds_min:
        result[
            (
                f"{group_name}_services_"
                f"within_{threshold}_min"
            )
        ] = metrics[
            "counts"
        ][threshold]

        result[
            (
                f"{group_name}_within_"
                f"{threshold}_min"
            )
        ] = (
            reachable
            & (
                result[
                    f"{group_name}_nearest_time_min"
                ]
                <= threshold
            )
        )

    return result


def weighted_mean(
    values,
    weights,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (
            weights > 0
        )
    )

    if not mask.any():
        return np.nan

    return float(
        np.average(
            values[mask],
            weights=weights[mask],
        )
    )


def build_section_summary(
    origins,
    group_names,
    thresholds_min,
):
    rows = []

    for (
        section_code,
        group,
    ) in origins.groupby(
        "census_section_code",
        sort=True,
    ):
        population = float(
            group[
                "assigned_population"
            ]
            .sum()
        )

        row = {
            "census_section_code":
                section_code,

            "population":
                population,

            "origin_row_count":
                int(
                    len(
                        group
                    )
                ),

            "reference_date":
                group[
                    "reference_date"
                ].iloc[0]
                if (
                    "reference_date"
                    in group.columns
                    and len(
                        group
                    )
                )
                else None,
        }

        weights = (
            group[
                "assigned_population"
            ]
            .to_numpy(
                dtype=float
            )
        )

        for group_name in group_names:
            time_col = (
                f"{group_name}_nearest_time_min"
            )

            reachable_col = (
                f"{group_name}_reachable"
            )

            reachable_mask = (
                group[
                    reachable_col
                ].astype(bool)
                .to_numpy()
            )

            reachable_population = float(
                weights[
                    reachable_mask
                ].sum()
            )

            row[
                f"{group_name}_reachable_population"
            ] = reachable_population

            row[
                f"{group_name}_unreachable_population"
            ] = (
                population
                - reachable_population
            )

            row[
                f"{group_name}_weighted_mean_nearest_time_min"
            ] = weighted_mean(
                group[
                    time_col
                ].to_numpy(),
                weights,
            )

            for threshold in thresholds_min:
                within_col = (
                    f"{group_name}_within_"
                    f"{threshold}_min"
                )

                within_mask = (
                    group[
                        within_col
                    ]
                    .astype(bool)
                    .to_numpy()
                )

                pop_within = float(
                    weights[
                        within_mask
                    ].sum()
                )

                row[
                    (
                        f"{group_name}_population_"
                        f"within_{threshold}_min"
                    )
                ] = pop_within

                row[
                    (
                        f"{group_name}_population_share_"
                        f"within_{threshold}_min"
                    )
                ] = (
                    pop_within
                    / population
                    if population > 0
                    else np.nan
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_municipality_summary(
    origins,
    group_names,
    thresholds_min,
):
    population = float(
        origins[
            "assigned_population"
        ]
        .sum()
    )

    weights = (
        origins[
            "assigned_population"
        ]
        .to_numpy(
            dtype=float
        )
    )

    summary = {
        "population":
            population,
    }

    for group_name in group_names:
        reachable = (
            origins[
                f"{group_name}_reachable"
            ]
            .astype(bool)
            .to_numpy()
        )

        reachable_population = float(
            weights[
                reachable
            ].sum()
        )

        summary[
            f"{group_name}_reachable_population"
        ] = (
            reachable_population
        )

        summary[
            f"{group_name}_unreachable_population"
        ] = (
            population
            - reachable_population
        )

        summary[
            f"{group_name}_reachable_share"
        ] = (
            reachable_population
            / population
            if population > 0
            else None
        )

        summary[
            f"{group_name}_weighted_mean_nearest_time_min"
        ] = weighted_mean(
            origins[
                f"{group_name}_nearest_time_min"
            ].to_numpy(),
            weights,
        )

        for threshold in thresholds_min:
            mask = (
                origins[
                    (
                        f"{group_name}_within_"
                        f"{threshold}_min"
                    )
                ]
                .astype(bool)
                .to_numpy()
            )

            covered_population = float(
                weights[
                    mask
                ].sum()
            )

            summary[
                (
                    f"{group_name}_population_"
                    f"within_{threshold}_min"
                )
            ] = covered_population

            summary[
                (
                    f"{group_name}_population_share_"
                    f"within_{threshold}_min"
                )
            ] = (
                covered_population
                / population
                if population > 0
                else None
            )

    return summary


def main():
    args = parse_args()

    (
        origins,
        services,
        nodes,
        edges,
        input_paths,
    ) = load_inputs(
        args
    )

    origins = prepare_origins(
        origins
    )

    services = prepare_services(
        services
    )

    (
        graph,
        graph_summary,
    ) = build_graph(
        nodes,
        edges,
    )

    validate_network_membership(
        graph,
        origins,
        services,
    )

    reverse_graph = (
        graph.reverse(
            copy=False
        )
    )

    (
        inferred_walking_speed_m_s,
        speed_summary,
    ) = infer_walking_speed(
        edges
    )

    speed_validation = (
        validate_walking_speed_consistency(
            edges,
            args.walking_speed_m_s,
        )
    )

    walking_speed_m_s = float(
        args.walking_speed_m_s
    )

    groups = build_service_groups(
        services
    )

    for group_name, group_df in groups.items():
        if group_df.empty:
            raise RuntimeError(
                f"Nessun servizio nel gruppo '{group_name}'."
            )

    result = origins.copy()

    for (
        group_name,
        group_df,
    ) in groups.items():
        print(
            f"Calcolo accessibility: "
            f"{group_name} "
            f"({len(group_df)} servizi)..."
        )

        metrics = (
            calculate_group_accessibility_full(
                reverse_graph=reverse_graph,
                origins=result,
                services=group_df,
                thresholds_min=args.thresholds_min,
                walking_speed_m_s=walking_speed_m_s,
            )
        )

        result = attach_group_metrics(
            result,
            group_name,
            metrics,
            args.thresholds_min,
        )

    section_summary = (
        build_section_summary(
            result,
            list(
                groups.keys()
            ),
            args.thresholds_min,
        )
    )

    municipality_summary = (
        build_municipality_summary(
            result,
            list(
                groups.keys()
            ),
            args.thresholds_min,
        )
    )

    output_dir = (
        FEATURES_ACCESSIBILITY_DIR
        / args.municipality_code
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    health_label = (
        args.health_reference_date
        .strftime("%Y%m%d")
    )

    if args.service_layer == "osm_only":
        suffix = (
            f"osm_only_{args.census_year}"
        )
    else:
        suffix = (
            f"{args.census_year}_"
            f"{args.school_year}_"
            f"{health_label}"
        )

    origin_parquet = (
        output_dir
        / (
            "walking_accessibility_origins_"
            f"{suffix}.parquet"
        )
    )

    origin_csv = (
        output_dir
        / (
            "walking_accessibility_origins_"
            f"{suffix}.csv"
        )
    )

    section_parquet = (
        output_dir
        / (
            "walking_accessibility_sections_"
            f"{suffix}.parquet"
        )
    )

    section_csv = (
        output_dir
        / (
            "walking_accessibility_sections_"
            f"{suffix}.csv"
        )
    )

    summary_json = (
        output_dir
        / (
            "walking_accessibility_summary_"
            f"{suffix}.json"
        )
    )

    result_gdf = gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=origins.crs,
    )

    result_gdf.to_parquet(
        origin_parquet,
        index=False,
    )

    result.drop(
        columns=["geometry"],
        errors="ignore",
    ).to_csv(
        origin_csv,
        index=False,
        encoding="utf-8-sig",
    )

    section_summary.to_parquet(
        section_parquet,
        index=False,
    )

    section_summary.to_csv(
        section_csv,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "generated_at_utc":
            utc_now_iso(),

        "municipality_code":
            args.municipality_code,

        "service_layer":
            args.service_layer,

        "thresholds_min":
            args.thresholds_min,

        "inputs":
            input_paths,

        "graph":
            graph_summary,

        "walking_speed_requested_m_s":
            walking_speed_m_s,

        "walking_speed_inferred":
            speed_summary,

        "walking_speed_validation":
            speed_validation,

        "service_group_counts": {
            name:
                int(
                    len(
                        group_df
                    )
                )
            for name, group_df
            in groups.items()
        },

        "methodology": {
            "directionality":
                (
                    "Accessibility is computed from origin to service on the "
                    "directed pedestrian graph by running Dijkstra from each "
                    "service on the reversed graph."
                ),

            "weights":
                (
                    "walking_time_s is used for time accessibility and "
                    "length_m for network distance."
                ),

            "snap_segments":
                (
                    "Origin fallback snap distance and service-to-network "
                    "snap distance are added to route length and converted "
                    "to time using the configured walking speed after "
                    "verifying consistency with the network edge weights."
                ),

            "unreachable":
                (
                    "Origins with no directed path to a service group are "
                    "preserved as unreachable rather than dropped."
                ),

            "groups":
                {
                    "school":
                        "all canonical services with category=education",

                    "pharmacy":
                        "canonical services with subcategory=pharmacy",

                    "hospital":
                        "canonical services with subcategory=hospital",
                },
        },

        "municipality_summary":
            municipality_summary,
    }

    summary_json.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n===================================="
    )
    print(
        " WALKING ACCESSIBILITY"
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
        "Origin rows: "
        f"{len(result)}"
    )

    print(
        "Popolazione: "
        f"{municipality_summary['population']:.2f}"
    )

    print(
        "Walking speed configured: "
        f"{walking_speed_m_s:.4f} m/s"
    )

    print(
        "Walking speed inferred: "
        f"{inferred_walking_speed_m_s:.4f} m/s"
    )

    print(
        "\nService groups:"
    )

    for name, group_df in groups.items():
        print(
            f"  {name}: "
            f"{len(group_df)}"
        )

    for group_name in groups:
        print(
            "\n"
            + group_name.upper()
        )

        print(
            "  reachable population: "
            f"{municipality_summary[f'{group_name}_reachable_population']:.2f} "
            f"("
            f"{municipality_summary[f'{group_name}_reachable_share'] * 100:.2f}%"
            f")"
        )

        print(
            "  weighted mean nearest time: "
            f"{municipality_summary[f'{group_name}_weighted_mean_nearest_time_min']:.2f} min"
        )

        for threshold in args.thresholds_min:
            population = municipality_summary[
                (
                    f"{group_name}_population_"
                    f"within_{threshold}_min"
                )
            ]

            share = municipality_summary[
                (
                    f"{group_name}_population_share_"
                    f"within_{threshold}_min"
                )
            ]

            print(
                f"  within {threshold:>2} min: "
                f"{population:.2f} "
                f"({share * 100:.2f}%)"
            )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ {origin_parquet}"
    )

    print(
        f"✓ {origin_csv}"
    )

    print(
        f"✓ {section_parquet}"
    )

    print(
        f"✓ {section_csv}"
    )

    print(
        f"✓ {summary_json}"
    )


if __name__ == "__main__":
    main()
