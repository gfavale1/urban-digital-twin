import argparse
import json
import math
import numbers
import os
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from dotenv import load_dotenv
from shapely import wkt
from sqlalchemy import (
    bindparam,
    create_engine,
    text,
)


# ============================================================
# PATHS / CONSTANTS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_OSM_DIR = (
    ROOT
    / "data"
    / "raw"
    / "osm"
)

PROCESSED_OSM_DIR = (
    ROOT
    / "data"
    / "processed"
    / "osm"
)

DEFAULT_WALKING_SPEED_M_S = 1.4


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ingestion della rete pedonale OpenStreetMap "
            "per un comune già presente nel Digital Twin."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help=(
            "Codice ISTAT a 6 cifre. "
            "Esempio: 077014 per Matera."
        ),
    )

    parser.add_argument(
        "--walking-speed",
        type=float,
        default=DEFAULT_WALKING_SPEED_M_S,
        help=(
            "Velocità pedonale media in m/s. "
            "Default: 1.4."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Ignora lo snapshot GraphML locale "
            "e interroga nuovamente OpenStreetMap."
        ),
    )

    args = parser.parse_args()

    args.municipality_code = (
        str(args.municipality_code)
        .strip()
        .zfill(6)
    )

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "municipality-code deve avere "
            "esattamente 6 cifre."
        )

    if args.walking_speed <= 0:
        raise ValueError(
            "walking-speed deve essere > 0."
        )

    return args


# ============================================================
# GENERIC HELPERS
# ============================================================

def is_missing_scalar(value):
    """
    Restituisce True solo per valori scalari missing:
    None, NaN, pd.NA, NaT, ecc.
    """

    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

        if hasattr(result, "item"):
            return bool(result.item())

    except (
        TypeError,
        ValueError,
        AttributeError,
    ):
        pass

    return False


def to_text(value):
    """
    Converte valori OSM, incluse liste,
    in rappresentazione testuale.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (list, tuple, set),
    ):
        cleaned = [
            str(item)
            for item in value
            if not is_missing_scalar(item)
        ]

        if not cleaned:
            return None

        return ";".join(cleaned)

    if is_missing_scalar(value):
        return None

    return str(value)


def to_bool(value):
    """
    Normalizza valori booleani OSM.

    Importante:
    NaN NON deve diventare True.
    """

    if value is None:
        return None

    if is_missing_scalar(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(
        value,
        numbers.Real,
    ):
        numeric_value = float(value)

        if not math.isfinite(
            numeric_value
        ):
            return None

        return bool(
            numeric_value
        )

    text_value = (
        str(value)
        .strip()
        .lower()
    )

    if text_value in {
        "yes",
        "true",
        "1",
    }:
        return True

    if text_value in {
        "no",
        "false",
        "0",
    }:
        return False

    return None


def json_safe_value(value):
    """
    Converte valori provenienti da
    pandas / NumPy / OSM in JSON standard.

    NaN, pd.NA, NaT, Infinity ecc.
    vengono trasformati in None,
    quindi in JSON null.
    """

    if value is None:
        return None

    # --------------------------------------------------------
    # COLLECTIONS
    # --------------------------------------------------------

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            json_safe_value(item)
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                json_safe_value(item)
            for key, item
            in value.items()
        }

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    if is_missing_scalar(value):
        return None

    # --------------------------------------------------------
    # BOOLEAN
    # --------------------------------------------------------

    # Deve venire prima di Integral:
    # bool è sottoclasse di int.
    if isinstance(
        value,
        bool,
    ):
        return value

    # --------------------------------------------------------
    # INTEGER
    # --------------------------------------------------------

    if isinstance(
        value,
        numbers.Integral,
    ):
        return int(value)

    # --------------------------------------------------------
    # FLOAT / NUMERIC
    # --------------------------------------------------------

    if isinstance(
        value,
        numbers.Real,
    ):
        numeric_value = float(
            value
        )

        if not math.isfinite(
            numeric_value
        ):
            return None

        return numeric_value

    # --------------------------------------------------------
    # STRING
    # --------------------------------------------------------

    if isinstance(
        value,
        str,
    ):
        return value

    # --------------------------------------------------------
    # NUMPY SCALAR
    # --------------------------------------------------------

    if hasattr(
        value,
        "item",
    ):
        try:
            return json_safe_value(
                value.item()
            )
        except Exception:
            pass

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return str(value)


def strict_json_dumps(data):
    """
    Serializza JSON in modalità strict.

    allow_nan=False impedisce che NaN,
    Infinity o -Infinity finiscano nel JSON.
    """

    return json.dumps(
        data,
        ensure_ascii=False,
        allow_nan=False,
    )


# ============================================================
# DATABASE
# ============================================================

def get_database_engine():
    load_dotenv(
        ROOT / ".env"
    )

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL non definito "
            "nel file .env."
        )

    return create_engine(
        database_url
    )


# ============================================================
# MUNICIPALITY
# ============================================================

def load_municipality(
    engine,
    municipality_code,
):
    """
    Recupera dal database il confine
    ufficiale ISTAT del comune.
    """

    query = text("""
        SELECT
            id,
            istat_code,
            name,
            ST_AsText(geometry)
                AS geometry_wkt

        FROM municipality

        WHERE
            istat_code =
                :istat_code;
    """)

    with engine.connect() as connection:

        row = connection.execute(
            query,
            {
                "istat_code":
                    municipality_code
            },
        ).mappings().first()

    if row is None:
        raise RuntimeError(
            "\nComune non presente "
            "nel database.\n"
            f"Codice ISTAT: "
            f"{municipality_code}\n\n"
            "Eseguire prima "
            "l'ingestion ISTAT."
        )

    geometry = wkt.loads(
        row["geometry_wkt"]
    )

    if geometry.geom_type not in {
        "Polygon",
        "MultiPolygon",
    }:
        raise RuntimeError(
            "Il confine comunale "
            "non è Polygon/MultiPolygon."
        )

    metadata = {
        "id":
            row["id"],

        "istat_code":
            row["istat_code"],

        "name":
            row["name"],
    }

    return metadata, geometry


# ============================================================
# OSMNX SETTINGS
# ============================================================

def configure_osmnx():
    """
    Configura OSMnx e conserva alcuni tag
    utili per future analisi di accessibilità.
    """

    extra_way_tags = [
        "foot",
        "surface",
        "smoothness",
        "incline",
        "lit",
        "sidewalk",
        "wheelchair",
        "access",
    ]

    ox.settings.useful_tags_way = list(
        dict.fromkeys(
            list(
                ox.settings.useful_tags_way
            )
            + extra_way_tags
        )
    )

    ox.settings.use_cache = True

    ox.settings.requests_timeout = 180


# ============================================================
# RAW GRAPH SNAPSHOT
# ============================================================

def graphml_path(
    municipality_code,
):
    directory = (
        RAW_OSM_DIR
        / municipality_code
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        / "walk_network.graphml"
    )


def load_or_download_graph(
    boundary,
    municipality_code,
    refresh=False,
):
    """
    Usa lo snapshot locale se disponibile.

    Con --refresh viene invece eseguita
    una nuova query a OpenStreetMap.
    """

    configure_osmnx()

    snapshot = graphml_path(
        municipality_code
    )

    if (
        snapshot.exists()
        and not refresh
    ):
        print(
            "\nCaricamento snapshot "
            "OSM locale:"
        )

        print(
            f"  {snapshot}"
        )

        return ox.io.load_graphml(
            snapshot
        )

    print(
        "\nDownload rete pedonale "
        "da OpenStreetMap..."
    )

    graph = (
        ox.graph.graph_from_polygon(
            boundary,
            network_type="walk",
            simplify=True,
            retain_all=True,
            truncate_by_edge=True,
        )
    )

    ox.io.save_graphml(
        graph,
        filepath=snapshot,
    )

    print(
        "\n✓ Snapshot GraphML salvato:"
    )

    print(
        f"  {snapshot}"
    )

    return graph


# ============================================================
# GRAPH QUALITY
# ============================================================

def validate_graph(
    graph,
):
    print(
        "\n=== OSM GRAPH QUALITY ==="
    )

    nodes_count = (
        graph.number_of_nodes()
    )

    edges_count = (
        graph.number_of_edges()
    )

    print(
        f"Nodi: {nodes_count}"
    )

    print(
        "Archi direzionati: "
        f"{edges_count}"
    )

    if nodes_count == 0:
        raise RuntimeError(
            "Il grafo OSM "
            "non contiene nodi."
        )

    if edges_count == 0:
        raise RuntimeError(
            "Il grafo OSM "
            "non contiene archi."
        )

    # --------------------------------------------------------
    # CONNECTIVITY
    # --------------------------------------------------------

    components = list(
        nx.weakly_connected_components(
            graph
        )
    )

    component_sizes = sorted(
        [
            len(component)
            for component
            in components
        ],
        reverse=True,
    )

    largest = (
        component_sizes[0]
        if component_sizes
        else 0
    )

    largest_pct = (
        largest
        / nodes_count
        * 100
    )

    print(
        "Componenti debolmente "
        "connesse: "
        f"{len(components)}"
    )

    print(
        "Nodi nella componente "
        "principale: "
        f"{largest} "
        f"({largest_pct:.2f}%)"
    )

    if largest_pct < 90:
        print(
            "ATTENZIONE: la rete "
            "presenta una frammentazione "
            "significativa."
        )

    # --------------------------------------------------------
    # LENGTH
    # --------------------------------------------------------

    lengths = [
        data.get("length")
        for _, _, _, data
        in graph.edges(
            keys=True,
            data=True,
        )
    ]

    missing_length = sum(
        value is None
        for value in lengths
    )

    print(
        "Archi senza lunghezza: "
        f"{missing_length}"
    )

    if missing_length:
        raise RuntimeError(
            "Sono presenti archi "
            "senza attributo length."
        )

    invalid_length = sum(
        (
            value is not None
            and float(value) <= 0
        )
        for value in lengths
    )

    print(
        "Archi con lunghezza <= 0: "
        f"{invalid_length}"
    )

    if invalid_length:
        raise RuntimeError(
            "Sono presenti archi "
            "con lunghezza non valida."
        )

    print(
        "✓ controlli grafo superati"
    )


# ============================================================
# CANONICAL MODEL
# ============================================================

def build_canonical_network(
    graph,
    walking_speed,
):
    """
    Converte il MultiDiGraph OSM
    nel nostro canonical model.
    """

    nodes_gdf, edges_gdf = (
        ox.convert.graph_to_gdfs(
            graph,
            nodes=True,
            edges=True,
            node_geometry=True,
            fill_edge_geometry=True,
        )
    )

    # ========================================================
    # NODES
    # ========================================================

    nodes = (
        nodes_gdf
        .reset_index()
        .copy()
    )

    nodes[
        "source_record_id"
    ] = (
        nodes["osmid"]
        .astype(str)
    )

    node_attribute_columns = [
        "highway",
        "street_count",
    ]

    node_attributes = []

    for _, row in nodes.iterrows():

        data = {}

        for column in (
            node_attribute_columns
        ):

            if column not in nodes.columns:
                continue

            value = json_safe_value(
                row[column]
            )

            if value is not None:
                data[column] = value

        node_attributes.append(
            strict_json_dumps(
                data
            )
        )

    nodes["attributes"] = (
        node_attributes
    )

    nodes = gpd.GeoDataFrame(
        nodes[
            [
                "source_record_id",
                "geometry",
                "attributes",
            ]
        ],
        geometry="geometry",
        crs=nodes_gdf.crs,
    )

    if nodes.crs is None:
        nodes = nodes.set_crs(
            epsg=4326
        )

    elif nodes.crs.to_epsg() != 4326:
        nodes = nodes.to_crs(
            epsg=4326
        )

    # ========================================================
    # EDGES
    # ========================================================

    edges = (
        edges_gdf
        .reset_index()
        .copy()
    )

    # u:v:key identifica un arco
    # del MultiDiGraph OSMnx.
    edges[
        "source_record_id"
    ] = (
        edges["u"]
        .astype(str)
        + ":"
        + edges["v"]
        .astype(str)
        + ":"
        + edges["key"]
        .astype(str)
    )

    edges[
        "source_osm_node"
    ] = (
        edges["u"]
        .astype(str)
    )

    edges[
        "target_osm_node"
    ] = (
        edges["v"]
        .astype(str)
    )

    # --------------------------------------------------------
    # LENGTH
    # --------------------------------------------------------

    edges["length_m"] = (
        pd.to_numeric(
            edges["length"],
            errors="coerce",
        )
    )

    if (
        edges["length_m"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Sono presenti archi "
            "senza length_m valida."
        )

    if (
        edges["length_m"] <= 0
    ).any():
        raise RuntimeError(
            "Sono presenti archi "
            "con length_m <= 0."
        )

    # --------------------------------------------------------
    # WALKING TIME
    # --------------------------------------------------------

    edges[
        "walking_time_s"
    ] = (
        edges["length_m"]
        / walking_speed
    )

    # --------------------------------------------------------
    # ROAD TYPE
    # --------------------------------------------------------

    if "highway" in edges.columns:

        edges["road_type"] = (
            edges["highway"]
            .apply(to_text)
        )

    else:
        edges["road_type"] = None

    # --------------------------------------------------------
    # ONEWAY
    # --------------------------------------------------------

    if "oneway" in edges.columns:

        edges[
            "oneway_normalized"
        ] = (
            edges["oneway"]
            .apply(to_bool)
        )

    else:
        edges[
            "oneway_normalized"
        ] = None

    # --------------------------------------------------------
    # OSM WAY IDS
    # --------------------------------------------------------

    if "osmid" in edges.columns:

        edges["osm_way_ids"] = (
            edges["osmid"]
            .apply(to_text)
        )

    else:
        edges["osm_way_ids"] = None

    # --------------------------------------------------------
    # OTHER OSM ATTRIBUTES
    # --------------------------------------------------------

    edge_attribute_columns = [
        "name",
        "highway",
        "surface",
        "smoothness",
        "incline",
        "lit",
        "sidewalk",
        "wheelchair",
        "access",
        "foot",
        "bridge",
        "tunnel",
        "width",
        "ref",
    ]

    edge_attributes = []

    for _, row in edges.iterrows():

        data = {}

        for column in (
            edge_attribute_columns
        ):

            if column not in edges.columns:
                continue

            value = json_safe_value(
                row[column]
            )

            if value is not None:
                data[column] = value

        edge_attributes.append(
            strict_json_dumps(
                data
            )
        )

    edges["attributes"] = (
        edge_attributes
    )

    # --------------------------------------------------------
    # FINAL EDGE DATAFRAME
    # --------------------------------------------------------

    edges = gpd.GeoDataFrame(
        edges[
            [
                "source_record_id",
                "source_osm_node",
                "target_osm_node",

                "geometry",

                "length_m",
                "walking_time_s",

                "road_type",

                "oneway_normalized",

                "osm_way_ids",

                "attributes",
            ]
        ],
        geometry="geometry",
        crs=edges_gdf.crs,
    )

    if edges.crs is None:
        edges = edges.set_crs(
            epsg=4326
        )

    elif edges.crs.to_epsg() != 4326:
        edges = edges.to_crs(
            epsg=4326
        )

    return nodes, edges


# ============================================================
# CANONICAL DATA QUALITY
# ============================================================

def validate_canonical_network(
    nodes,
    edges,
):
    print(
        "\n=== CANONICAL DATA QUALITY ==="
    )

    if nodes[
        "source_record_id"
    ].duplicated().any():

        raise RuntimeError(
            "source_record_id duplicati "
            "nei network nodes."
        )

    print(
        "✓ node source_record_id univoci"
    )

    if edges[
        "source_record_id"
    ].duplicated().any():

        raise RuntimeError(
            "source_record_id duplicati "
            "nei network edges."
        )

    print(
        "✓ edge source_record_id univoci"
    )

    if nodes.geometry.isna().any():
        raise RuntimeError(
            "Nodi con geometria mancante."
        )

    if edges.geometry.isna().any():
        raise RuntimeError(
            "Archi con geometria mancante."
        )

    print(
        "✓ geometrie presenti"
    )

    if (
        edges["length_m"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "length_m contiene NULL."
        )

    if (
        edges["walking_time_s"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "walking_time_s contiene NULL."
        )

    print(
        "✓ lunghezza e walking time validi"
    )

    # Controllo esplicito JSON.
    for value in nodes["attributes"]:
        json.loads(value)

    for value in edges["attributes"]:
        json.loads(value)

    print(
        "✓ JSON attributes validi"
    )


# ============================================================
# SILVER LAYER
# ============================================================

def save_silver_datasets(
    nodes,
    edges,
    municipality_code,
):
    directory = (
        PROCESSED_OSM_DIR
        / municipality_code
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    nodes_path = (
        directory
        / "walk_nodes.parquet"
    )

    edges_path = (
        directory
        / "walk_edges.parquet"
    )

    nodes.to_parquet(
        nodes_path,
        index=False,
    )

    edges.to_parquet(
        edges_path,
        index=False,
    )

    print(
        "\n=== SILVER DATASETS ==="
    )

    print(
        f"✓ {nodes_path}"
    )

    print(
        f"✓ {edges_path}"
    )


# ============================================================
# POSTGIS - NODES
# ============================================================

def upsert_network_nodes(
    connection,
    nodes,
):
    query = text("""
        INSERT INTO network_node (
            geometry,
            source_system,
            source_record_id,
            attributes
        )

        VALUES (
            ST_GeomFromText(
                :geometry,
                4326
            ),

            'OSM',

            :source_record_id,

            CAST(
                :attributes
                AS JSONB
            )
        )

        ON CONFLICT (
            source_system,
            source_record_id
        )

        DO UPDATE SET

            geometry =
                EXCLUDED.geometry,

            attributes =
                EXCLUDED.attributes,

            ingested_at =
                NOW();
    """)

    records = []

    for row in nodes.itertuples():

        records.append(
            {
                "geometry":
                    row.geometry.wkt,

                "source_record_id":
                    row.source_record_id,

                "attributes":
                    row.attributes,
            }
        )

    connection.execute(
        query,
        records,
    )


# ============================================================
# POSTGIS - NODE ID MAP
# ============================================================

def get_network_node_ids(
    connection,
    source_ids,
    chunk_size=5000,
):
    mapping = {}

    statement = (
        text("""
            SELECT
                id,
                source_record_id

            FROM network_node

            WHERE
                source_system = 'OSM'

                AND source_record_id
                    IN :source_ids;
        """)
        .bindparams(
            bindparam(
                "source_ids",
                expanding=True,
            )
        )
    )

    source_ids = list(
        source_ids
    )

    for start in range(
        0,
        len(source_ids),
        chunk_size,
    ):

        chunk = source_ids[
            start:
            start + chunk_size
        ]

        result = connection.execute(
            statement,
            {
                "source_ids":
                    chunk
            },
        )

        for row in result:

            mapping[
                row.source_record_id
            ] = row.id

    return mapping


# ============================================================
# POSTGIS - EDGES
# ============================================================

def upsert_network_edges(
    connection,
    edges,
    node_id_map,
):
    query = text("""
        INSERT INTO network_edge (
            source_node_id,
            target_node_id,

            geometry,

            length_m,
            walking_time_s,

            road_type,

            foot_access,
            vehicle_access,
            oneway,

            source_system,
            source_record_id,

            osm_way_ids,
            attributes
        )

        VALUES (
            :source_node_id,
            :target_node_id,

            ST_GeomFromText(
                :geometry,
                4326
            ),

            :length_m,
            :walking_time_s,

            :road_type,

            TRUE,
            NULL,
            :oneway,

            'OSM',
            :source_record_id,

            :osm_way_ids,

            CAST(
                :attributes
                AS JSONB
            )
        )

        ON CONFLICT (
            source_system,
            source_record_id
        )
        WHERE
            source_record_id
            IS NOT NULL

        DO UPDATE SET

            source_node_id =
                EXCLUDED.source_node_id,

            target_node_id =
                EXCLUDED.target_node_id,

            geometry =
                EXCLUDED.geometry,

            length_m =
                EXCLUDED.length_m,

            walking_time_s =
                EXCLUDED.walking_time_s,

            road_type =
                EXCLUDED.road_type,

            foot_access =
                EXCLUDED.foot_access,

            vehicle_access =
                EXCLUDED.vehicle_access,

            oneway =
                EXCLUDED.oneway,

            osm_way_ids =
                EXCLUDED.osm_way_ids,

            attributes =
                EXCLUDED.attributes,

            ingested_at =
                NOW();
    """)

    records = []

    for row in edges.itertuples():

        source_node_id = (
            node_id_map.get(
                row.source_osm_node
            )
        )

        target_node_id = (
            node_id_map.get(
                row.target_osm_node
            )
        )

        if source_node_id is None:
            raise RuntimeError(
                "Nodo sorgente OSM "
                "non trovato nel DB: "
                f"{row.source_osm_node}"
            )

        if target_node_id is None:
            raise RuntimeError(
                "Nodo destinazione OSM "
                "non trovato nel DB: "
                f"{row.target_osm_node}"
            )

        records.append(
            {
                "source_node_id":
                    source_node_id,

                "target_node_id":
                    target_node_id,

                "geometry":
                    row.geometry.wkt,

                "length_m":
                    float(
                        row.length_m
                    ),

                "walking_time_s":
                    float(
                        row.walking_time_s
                    ),

                "road_type":
                    row.road_type,

                "oneway":
                    (
                        None
                        if is_missing_scalar(
                            row.oneway_normalized
                        )
                        else bool(
                            row.oneway_normalized
                        )
                    ),

                "source_record_id":
                    row.source_record_id,

                "osm_way_ids":
                    row.osm_way_ids,

                "attributes":
                    row.attributes,
            }
        )

    connection.execute(
        query,
        records,
    )


# ============================================================
# DATABASE WRITE
# ============================================================

def write_to_postgis(
    engine,
    nodes,
    edges,
):
    """
    Tutto il caricamento avviene
    in un'unica transazione.

    Se un errore avviene sugli edge,
    anche l'inserimento dei nodi
    viene rollbackato.
    """

    with engine.begin() as connection:

        print(
            "\nCaricamento network nodes..."
        )

        upsert_network_nodes(
            connection,
            nodes,
        )

        print(
            "✓ network nodes caricati"
        )

        node_id_map = (
            get_network_node_ids(
                connection,
                nodes[
                    "source_record_id"
                ].tolist(),
            )
        )

        if (
            len(node_id_map)
            != len(nodes)
        ):
            raise RuntimeError(
                "Non tutti i nodi OSM "
                "sono stati risolti "
                "nel database."
            )

        print(
            "Caricamento network edges..."
        )

        upsert_network_edges(
            connection,
            edges,
            node_id_map,
        )

        print(
            "✓ network edges caricati"
        )

    print(
        "\n✓ Rete pedonale caricata "
        "correttamente in PostGIS."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    print(
        "\n===================================="
    )

    print(
        " OSM PEDESTRIAN NETWORK INGESTION"
    )

    print(
        "===================================="
    )

    engine = get_database_engine()

    # ========================================================
    # MUNICIPALITY
    # ========================================================

    metadata, boundary = (
        load_municipality(
            engine,
            args.municipality_code,
        )
    )

    print(
        "\n=== COMUNE ==="
    )

    print(
        f"Nome: "
        f"{metadata['name']}"
    )

    print(
        "Codice ISTAT: "
        f"{metadata['istat_code']}"
    )

    # ========================================================
    # OSM GRAPH
    # ========================================================

    graph = (
        load_or_download_graph(
            boundary=boundary,

            municipality_code=(
                args.municipality_code
            ),

            refresh=args.refresh,
        )
    )

    # ========================================================
    # RAW GRAPH QUALITY
    # ========================================================

    validate_graph(
        graph
    )

    # ========================================================
    # CANONICAL MODEL
    # ========================================================

    nodes, edges = (
        build_canonical_network(
            graph,
            args.walking_speed,
        )
    )

    print(
        "\n=== CANONICAL NETWORK ==="
    )

    print(
        f"Nodi: "
        f"{len(nodes)}"
    )

    print(
        f"Archi: "
        f"{len(edges)}"
    )

    print(
        "Velocità pedonale: "
        f"{args.walking_speed} m/s"
    )

    # ========================================================
    # CANONICAL QUALITY
    # ========================================================

    validate_canonical_network(
        nodes,
        edges,
    )

    # ========================================================
    # SILVER
    # ========================================================

    save_silver_datasets(
        nodes,
        edges,
        args.municipality_code,
    )

    # ========================================================
    # POSTGIS
    # ========================================================

    write_to_postgis(
        engine,
        nodes,
        edges,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n===================================="
    )

    print(
        " INGESTION OSM COMPLETATA"
    )

    print(
        "===================================="
    )

    print(
        f"Comune: "
        f"{metadata['name']}"
    )

    print(
        "Codice ISTAT: "
        f"{metadata['istat_code']}"
    )

    print(
        f"Nodi: "
        f"{len(nodes)}"
    )

    print(
        f"Archi: "
        f"{len(edges)}"
    )

    print(
        "Walking speed: "
        f"{args.walking_speed} m/s"
    )


if __name__ == "__main__":
    main()