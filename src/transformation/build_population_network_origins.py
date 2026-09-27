import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_ISTAT_DIR = ROOT / "data" / "processed" / "istat"
PROCESSED_OSM_DIR = ROOT / "data" / "processed" / "osm"

FEATURES_ACCESSIBILITY_DIR = (
    ROOT / "data" / "features" / "accessibility"
)


DEMOGRAPHIC_COLUMNS = [
    "population",
    "families",
    "age_0_14",
    "age_15_64",
    "age_65_plus",
    "foreign_population",
    "employed_population",
    "housing_units",
    "unemployed_population",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Distribuisce le osservazioni ISTAT di sezione sui nodi "
            "della rete pedonale OSM. Per ogni sezione con osservazione, "
            "la popolazione è ripartita uniformemente sui nodi interni. "
            "Se una sezione non contiene nodi, usa il nodo di rete più "
            "vicino al representative point come fallback."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--census-year",
        default="2023",
        help="Anno osservazioni censuarie. Default: 2023.",
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
    areas_path = (
        PROCESSED_ISTAT_DIR
        / f"{args.municipality_code}_census_areas_2021.parquet"
    )

    observations_path = (
        PROCESSED_ISTAT_DIR
        / f"{args.municipality_code}_census_observations_{args.census_year}.parquet"
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
        areas_path,
        observations_path,
        nodes_path,
        edges_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input mancante: {path}"
            )

    areas = gpd.read_parquet(areas_path)
    observations = pd.read_parquet(
        observations_path
    )
    nodes = gpd.read_parquet(nodes_path)
    edges = gpd.read_parquet(edges_path)

    if areas.crs is None:
        raise RuntimeError(
            "Il dataset census areas non ha CRS."
        )

    if nodes.crs is None:
        raise RuntimeError(
            "Il dataset OSM nodes non ha CRS."
        )

    return (
        areas,
        observations,
        nodes,
        edges,
        {
            "areas": str(areas_path),
            "observations": str(observations_path),
            "nodes": str(nodes_path),
            "edges": str(edges_path),
        },
    )


def prepare_areas_and_observations(
    areas,
    observations,
):
    required_area_columns = {
        "census_section_code",
        "geometry",
    }

    required_observation_columns = {
        "census_section_code",
        "population",
        "reference_date",
    }

    missing_areas = (
        required_area_columns
        - set(areas.columns)
    )

    missing_observations = (
        required_observation_columns
        - set(observations.columns)
    )

    if missing_areas:
        raise RuntimeError(
            "Colonne census areas mancanti: "
            + ", ".join(
                sorted(missing_areas)
            )
        )

    if missing_observations:
        raise RuntimeError(
            "Colonne census observations mancanti: "
            + ", ".join(
                sorted(missing_observations)
            )
        )

    areas = areas.copy()
    observations = observations.copy()

    areas["census_section_code"] = (
        areas["census_section_code"]
        .astype("string")
        .str.strip()
    )

    observations["census_section_code"] = (
        observations[
            "census_section_code"
        ]
        .astype("string")
        .str.strip()
    )

    if (
        areas["census_section_code"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "census_section_code duplicati nelle geometrie."
        )

    if (
        observations[
            "census_section_code"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "census_section_code duplicati nelle osservazioni."
        )

    for column in DEMOGRAPHIC_COLUMNS:
        if column in observations.columns:
            observations[column] = (
                pd.to_numeric(
                    observations[column],
                    errors="coerce",
                )
            )

    merged = areas.merge(
        observations,
        on="census_section_code",
        how="left",
        validate="1:1",
        indicator=True,
    )

    merged["has_observation"] = (
        merged["_merge"]
        == "both"
    )

    merged = merged.drop(
        columns=["_merge"]
    )

    return merged


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

    nodes["network_node_id"] = (
        nodes["source_record_id"]
        .map(normalize_node_id)
    )

    if (
        nodes["network_node_id"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Alcuni nodi OSM non hanno identificativo valido."
        )

    if (
        nodes["network_node_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "network_node_id duplicati nel dataset OSM."
        )

    return nodes[
        [
            "network_node_id",
            "geometry",
        ]
    ].copy()


def build_network_components(
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

    node_ids = (
        nodes["network_node_id"]
        .astype(str)
        .tolist()
    )

    graph.add_nodes_from(node_ids)

    for _, edge in edges.iterrows():
        source = normalize_node_id(
            edge["source_osm_node"]
        )
        target = normalize_node_id(
            edge["target_osm_node"]
        )

        if (
            source is None
            or target is None
        ):
            continue

        graph.add_edge(
            source,
            target,
        )

    components = list(
        nx.weakly_connected_components(
            graph
        )
    )

    components.sort(
        key=len,
        reverse=True,
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

        size = len(component)

        for node_id in component:
            component_map[
                node_id
            ] = component_id

            component_size_map[
                node_id
            ] = size

    result = nodes.copy()

    result[
        "network_component_id"
    ] = (
        result["network_node_id"]
        .map(component_map)
    )

    result[
        "network_component_size"
    ] = (
        result["network_node_id"]
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
                int(len(components)),

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


def choose_metric_crs(areas):
    metric_crs = (
        areas.estimate_utm_crs()
    )

    if metric_crs is None:
        raise RuntimeError(
            "Impossibile stimare un CRS metrico."
        )

    return metric_crs


def internal_node_assignments(
    areas_with_obs,
    nodes,
):
    areas_wgs84 = (
        areas_with_obs
        .to_crs("EPSG:4326")
    )

    nodes_wgs84 = (
        nodes.to_crs(
            "EPSG:4326"
        )
    )

    joined = gpd.sjoin(
        nodes_wgs84,
        areas_wgs84[
            [
                "census_section_code",
                "geometry",
            ]
        ],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        return pd.DataFrame(
            columns=[
                "census_section_code",
                "network_node_id",
            ]
        )

    joined = (
        joined[
            [
                "census_section_code",
                "network_node_id",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    return joined


def fallback_assignments(
    fallback_areas,
    nodes,
    metric_crs,
):
    if fallback_areas.empty:
        return pd.DataFrame(
            columns=[
                "census_section_code",
                "network_node_id",
                "fallback_snap_distance_m",
            ]
        )

    areas_metric = (
        fallback_areas
        .to_crs(
            metric_crs
        )
        .copy()
    )

    nodes_metric = (
        nodes
        .to_crs(
            metric_crs
        )
        .copy()
    )

    representative = (
        areas_metric[
            [
                "census_section_code",
                "geometry",
            ]
        ]
        .copy()
    )

    representative[
        "geometry"
    ] = (
        representative[
            "geometry"
        ]
        .representative_point()
    )

    nearest = gpd.sjoin_nearest(
        representative,
        nodes_metric[
            [
                "network_node_id",
                "geometry",
            ]
        ],
        how="left",
        distance_col=(
            "fallback_snap_distance_m"
        ),
    )

    nearest = (
        nearest.sort_values(
            [
                "census_section_code",
                "fallback_snap_distance_m",
                "network_node_id",
            ]
        )
        .drop_duplicates(
            subset=[
                "census_section_code",
            ],
            keep="first",
        )
    )

    return nearest[
        [
            "census_section_code",
            "network_node_id",
            "fallback_snap_distance_m",
        ]
    ].copy()


def build_origin_rows(
    merged_areas,
    nodes,
    metric_crs,
):
    observed = merged_areas.loc[
        merged_areas[
            "has_observation"
        ]
    ].copy()

    internal = internal_node_assignments(
        observed,
        nodes,
    )

    internal_counts = (
        internal.groupby(
            "census_section_code"
        )
        .size()
        .rename(
            "internal_node_count"
        )
    )

    observed = observed.merge(
        internal_counts,
        left_on="census_section_code",
        right_index=True,
        how="left",
    )

    observed[
        "internal_node_count"
    ] = (
        observed[
            "internal_node_count"
        ]
        .fillna(0)
        .astype(int)
    )

    no_internal = observed.loc[
        observed[
            "internal_node_count"
        ]
        == 0
    ].copy()

    fallback = fallback_assignments(
        no_internal,
        nodes,
        metric_crs,
    )

    internal[
        "assignment_method"
    ] = "equal_internal_nodes"

    internal[
        "fallback_snap_distance_m"
    ] = np.nan

    fallback[
        "assignment_method"
    ] = "representative_point_nearest_node"

    assignments = pd.concat(
        [
            internal,
            fallback,
        ],
        ignore_index=True,
    )

    assignment_counts = (
        assignments.groupby(
            "census_section_code"
        )
        .size()
        .rename(
            "assigned_node_count"
        )
    )

    assignments = assignments.merge(
        assignment_counts,
        left_on="census_section_code",
        right_index=True,
        how="left",
    )

    assignments[
        "origin_weight"
    ] = (
        1.0
        / assignments[
            "assigned_node_count"
        ]
    )

    observation_columns = [
        "census_section_code",
        "reference_date",
    ]

    for column in DEMOGRAPHIC_COLUMNS:
        if column in observed.columns:
            observation_columns.append(
                column
            )

    assignments = assignments.merge(
        observed[
            observation_columns
        ],
        on="census_section_code",
        how="left",
        validate="m:1",
    )

    nodes_attributes = (
        nodes[
            [
                "network_node_id",
                "network_component_id",
                "network_component_size",
                "is_largest_component",
                "geometry",
            ]
        ]
        .copy()
    )

    assignments = assignments.merge(
        nodes_attributes,
        on="network_node_id",
        how="left",
        validate="m:1",
    )

    for column in DEMOGRAPHIC_COLUMNS:
        if column not in assignments.columns:
            continue

        assignments[
            f"assigned_{column}"
        ] = (
            assignments[column]
            * assignments[
                "origin_weight"
            ]
        )

    return (
        assignments,
        observed,
    )


def build_section_summary(
    merged_areas,
    origins,
):
    summary_columns = [
        "census_section_code",
        "section_type_code",
        "locality_type",
        "has_observation",
        "population",
        "reference_date",
    ]

    summary_columns = [
        column
        for column in summary_columns
        if column in merged_areas.columns
    ]

    summary = (
        merged_areas[
            summary_columns
        ]
        .copy()
    )

    origin_stats = (
        origins.groupby(
            "census_section_code"
        )
        .agg(
            assigned_node_count=(
                "network_node_id",
                "size",
            ),

            assignment_method=(
                "assignment_method",
                lambda series:
                    ",".join(
                        sorted(
                            set(
                                series.astype(str)
                            )
                        )
                    ),
            ),

            fallback_snap_distance_m=(
                "fallback_snap_distance_m",
                "min",
            ),

            largest_component_origin_count=(
                "is_largest_component",
                "sum",
            ),
        )
        .reset_index()
    )

    summary = summary.merge(
        origin_stats,
        on="census_section_code",
        how="left",
    )

    summary[
        "assigned_node_count"
    ] = (
        summary[
            "assigned_node_count"
        ]
        .fillna(0)
        .astype(int)
    )

    summary[
        "largest_component_origin_count"
    ] = (
        summary[
            "largest_component_origin_count"
        ]
        .fillna(0)
        .astype(int)
    )

    summary[
        "assignment_status"
    ] = np.select(
        [
            ~summary[
                "has_observation"
            ],

            (
                summary[
                    "has_observation"
                ]
                & (
                    summary[
                        "assigned_node_count"
                    ]
                    > 0
                )
            ),
        ],
        [
            "no_observation",
            "assigned",
        ],
        default="unassigned",
    )

    return summary


def validate_population_conservation(
    observed,
    origins,
):
    original_population = float(
        observed[
            "population"
        ]
        .fillna(0)
        .sum()
    )

    assigned_population = float(
        origins[
            "assigned_population"
        ]
        .fillna(0)
        .sum()
    )

    difference = (
        assigned_population
        - original_population
    )

    tolerance = max(
        1e-6,
        abs(
            original_population
        )
        * 1e-10,
    )

    if abs(difference) > tolerance:
        raise RuntimeError(
            "La popolazione non è conservata: "
            f"original={original_population}, "
            f"assigned={assigned_population}, "
            f"difference={difference}"
        )

    return {
        "original_population":
            original_population,

        "assigned_population":
            assigned_population,

        "difference":
            difference,
    }


def make_origins_geodataframe(
    origins,
    crs,
):
    return gpd.GeoDataFrame(
        origins,
        geometry="geometry",
        crs=crs,
    )


def main():
    args = parse_args()

    (
        areas,
        observations,
        nodes,
        edges,
        input_paths,
    ) = load_inputs(
        args
    )

    merged_areas = (
        prepare_areas_and_observations(
            areas,
            observations,
        )
    )

    nodes = prepare_nodes(
        nodes
    )

    (
        nodes,
        network_summary,
    ) = build_network_components(
        nodes,
        edges,
    )

    metric_crs = choose_metric_crs(
        areas
    )

    (
        origins,
        observed,
    ) = build_origin_rows(
        merged_areas,
        nodes,
        metric_crs,
    )

    population_check = (
        validate_population_conservation(
            observed,
            origins,
        )
    )

    section_summary = (
        build_section_summary(
            merged_areas,
            origins,
        )
    )

    origins_gdf = (
        make_origins_geodataframe(
            origins,
            nodes.crs,
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

    origins_parquet = (
        output_dir
        / (
            "population_network_origins_"
            f"{args.census_year}.parquet"
        )
    )

    origins_csv = (
        output_dir
        / (
            "population_network_origins_"
            f"{args.census_year}.csv"
        )
    )

    summary_parquet = (
        output_dir
        / (
            "census_network_assignment_"
            f"{args.census_year}.parquet"
        )
    )

    summary_csv = (
        output_dir
        / (
            "census_network_assignment_"
            f"{args.census_year}.csv"
        )
    )

    manifest_path = (
        output_dir
        / (
            "population_network_origins_"
            f"{args.census_year}_manifest.json"
        )
    )

    origins_gdf.to_parquet(
        origins_parquet,
        index=False,
    )

    origins.drop(
        columns=["geometry"]
    ).to_csv(
        origins_csv,
        index=False,
        encoding="utf-8-sig",
    )

    section_summary.to_parquet(
        summary_parquet,
        index=False,
    )

    section_summary.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    fallback_count = int(
        (
            origins[
                "assignment_method"
            ]
            == "representative_point_nearest_node"
        )
        .sum()
    )

    sections_with_fallback = int(
        origins.loc[
            origins[
                "assignment_method"
            ]
            == "representative_point_nearest_node",
            "census_section_code",
        ]
        .nunique()
    )

    observed_section_count = int(
        merged_areas[
            "has_observation"
        ]
        .sum()
    )

    missing_observation_count = int(
        (
            ~merged_areas[
                "has_observation"
            ]
        )
        .sum()
    )

    origins_on_lcc = int(
        origins[
            "is_largest_component"
        ]
        .sum()
    )

    population_on_lcc = float(
        origins.loc[
            origins[
                "is_largest_component"
            ],
            "assigned_population",
        ]
        .fillna(0)
        .sum()
    )

    manifest = {
        "generated_at_utc":
            utc_now_iso(),

        "municipality_code":
            args.municipality_code,

        "census_year":
            args.census_year,

        "inputs":
            input_paths,

        "crs": {
            "areas":
                str(
                    areas.crs
                ),

            "nodes":
                str(
                    nodes.crs
                ),

            "metric_crs":
                str(
                    metric_crs
                ),
        },

        "census_sections_total":
            int(
                len(
                    merged_areas
                )
            ),

        "census_sections_with_observation":
            observed_section_count,

        "census_sections_without_observation":
            missing_observation_count,

        "origin_rows":
            int(
                len(
                    origins
                )
            ),

        "sections_using_fallback":
            sections_with_fallback,

        "fallback_origin_rows":
            fallback_count,

        "network":
            network_summary,

        "origins_on_largest_component":
            origins_on_lcc,

        "population_on_largest_component":
            population_on_lcc,

        "population_conservation":
            population_check,

        "methodology": {
            "internal_nodes":
                (
                    "For each census section with an observation, "
                    "population and demographic variables are distributed "
                    "uniformly across all pedestrian-network nodes strictly "
                    "within the section."
                ),

            "fallback":
                (
                    "If no pedestrian-network node lies strictly within "
                    "the section, the representative point of the polygon "
                    "is snapped to the nearest network node in a metric CRS."
                ),

            "missing_observations":
                (
                    "Sections without a census observation are preserved "
                    "in the section summary as no_observation and are not "
                    "converted to zero-population demand."
                ),

            "components":
                (
                    "All network components are preserved. Each origin "
                    "stores component ID and whether it belongs to the "
                    "largest weakly connected component, so unreachable "
                    "demand can be handled explicitly downstream rather "
                    "than silently discarded."
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
        " POPULATION -> NETWORK ORIGINS"
    )
    print(
        "===================================="
    )

    print(
        f"Comune: {args.municipality_code}"
    )

    print(
        "Sezioni totali: "
        f"{len(merged_areas)}"
    )

    print(
        "Con osservazione: "
        f"{observed_section_count}"
    )

    print(
        "Senza osservazione: "
        f"{missing_observation_count}"
    )

    print(
        "\nOrigin rows: "
        f"{len(origins)}"
    )

    print(
        "Sezioni con fallback nearest-node: "
        f"{sections_with_fallback}"
    )

    print(
        "\nNetwork components: "
        f"{network_summary['component_count']}"
    )

    print(
        "Largest component size: "
        f"{network_summary['largest_component_size']}"
    )

    print(
        "Origin rows su largest component: "
        f"{origins_on_lcc}"
    )

    print(
        "\nPopolazione originale: "
        f"{population_check['original_population']:.6f}"
    )

    print(
        "Popolazione assegnata: "
        f"{population_check['assigned_population']:.6f}"
    )

    print(
        "Differenza: "
        f"{population_check['difference']:.12f}"
    )

    print(
        "Popolazione su largest component: "
        f"{population_on_lcc:.6f}"
    )

    print(
        "\nAssignment methods:"
    )

    print(
        origins[
            "assignment_method"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSection assignment status:"
    )

    print(
        section_summary[
            "assignment_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== OUTPUT ==="
    )

    print(
        f"✓ {origins_parquet}"
    )

    print(
        f"✓ {origins_csv}"
    )

    print(
        f"✓ {summary_parquet}"
    )

    print(
        f"✓ {summary_csv}"
    )

    print(
        f"✓ {manifest_path}"
    )


if __name__ == "__main__":
    main()
