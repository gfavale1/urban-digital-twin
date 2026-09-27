import argparse
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, Point


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_MIM_DIR = ROOT / "data" / "processed" / "mim"
PROCESSED_ISTAT_DIR = ROOT / "data" / "processed" / "istat"
PROCESSED_OSM_DIR = ROOT / "data" / "processed" / "osm"
FEATURES_DIR = ROOT / "data" / "features"

DEFAULT_BUILDING_YEAR = "202425"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "QA V2 degli edifici scolastici: "
            "zoom urbano + mappa dettagliata dei casi pending."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT a 6 cifre, es. 077014.",
    )

    parser.add_argument(
        "--building-year",
        default=DEFAULT_BUILDING_YEAR,
    )

    parser.add_argument(
        "--margin-m",
        type=float,
        default=1200.0,
        help=(
            "Margine in metri attorno all'insieme dei punti "
            "per la mappa zoomata."
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
            "municipality-code deve avere esattamente 6 cifre."
        )

    return args


def load_inputs(
    municipality_code,
    building_year,
):
    mim_dir = (
        PROCESSED_MIM_DIR
        / municipality_code
    )

    buildings_path = (
        mim_dir
        / (
            "physical_school_buildings_"
            f"{building_year}_final_v2.parquet"
        )
    )

    census_path = (
        PROCESSED_ISTAT_DIR
        / (
            f"{municipality_code}_"
            "census_areas_2021.parquet"
        )
    )

    edges_path = (
        PROCESSED_OSM_DIR
        / municipality_code
        / "walk_edges.parquet"
    )

    for path in [
        buildings_path,
        census_path,
        edges_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"File richiesto non trovato: {path}"
            )

    buildings = pd.read_parquet(
        buildings_path
    )

    census = gpd.read_parquet(
        census_path
    )

    edges = gpd.read_parquet(
        edges_path
    )

    return (
        buildings,
        census,
        edges,
    )


def point_layer(
    dataframe,
    lon_col,
    lat_col,
    mask,
):
    subset = (
        dataframe.loc[
            mask
        ]
        .copy()
    )

    subset = subset[
        subset[lon_col].notna()
        & subset[lat_col].notna()
    ].copy()

    if subset.empty:
        return gpd.GeoDataFrame(
            subset,
            geometry=[],
            crs=4326,
        )

    subset["geometry"] = [
        Point(
            float(lon),
            float(lat),
        )
        for lon, lat in zip(
            subset[lon_col],
            subset[lat_col],
        )
    ]

    return gpd.GeoDataFrame(
        subset,
        geometry="geometry",
        crs=4326,
    )


def build_layers(
    buildings,
):
    validated_mask = (
        buildings[
            "final_location_status"
        ]
        == "validated"
    )

    accepted_mask = (
        buildings[
            "final_location_status"
        ]
        == "accepted_address"
    )

    review_mask = (
        buildings[
            "final_location_status"
        ]
        == "review"
    )

    unresolved_mask = (
        buildings[
            "final_location_status"
        ]
        == "unresolved"
    )

    pending_mask = (
        review_mask
        | unresolved_mask
    )

    validated = point_layer(
        buildings,
        "final_longitude",
        "final_latitude",
        validated_mask,
    )

    accepted = point_layer(
        buildings,
        "final_longitude",
        "final_latitude",
        accepted_mask,
    )

    pending_osm = point_layer(
        buildings,
        "osm_candidate_longitude",
        "osm_candidate_latitude",
        pending_mask,
    )

    pending_geocoder = point_layer(
        buildings,
        "geocoder_longitude",
        "geocoder_latitude",
        pending_mask,
    )

    return {
        "validated": validated,
        "accepted": accepted,
        "pending_osm": pending_osm,
        "pending_geocoder": pending_geocoder,
    }


def metric_crs_from(
    census,
):
    metric_crs = (
        census.estimate_utm_crs()
    )

    if metric_crs is None:
        raise RuntimeError(
            "Impossibile stimare un CRS metrico."
        )

    return metric_crs


def all_reference_points(
    layers,
):
    frames = []

    for name in [
        "validated",
        "accepted",
        "pending_osm",
        "pending_geocoder",
    ]:
        layer = layers[name]

        if not layer.empty:
            frames.append(
                layer[
                    ["geometry"]
                ]
            )

    if not frames:
        return None

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return gpd.GeoDataFrame(
        combined,
        geometry="geometry",
        crs=4326,
    )


def calculate_extent(
    reference_points_m,
    margin_m,
):
    minx, miny, maxx, maxy = (
        reference_points_m.total_bounds
    )

    width = maxx - minx
    height = maxy - miny

    dynamic_margin = max(
        margin_m,
        width * 0.08,
        height * 0.08,
    )

    return (
        minx - dynamic_margin,
        maxx + dynamic_margin,
        miny - dynamic_margin,
        maxy + dynamic_margin,
    )


def clip_to_extent(
    layer,
    extent,
):
    if layer.empty:
        return layer

    xmin, xmax, ymin, ymax = extent

    return layer.cx[
        xmin:xmax,
        ymin:ymax,
    ]


def label_points(
    ax,
    layer,
    prefix=None,
):
    if layer.empty:
        return

    offsets = [
        (5, 5),
        (5, -9),
        (-44, 5),
        (-44, -9),
        (8, 12),
        (-50, 12),
    ]

    for idx, (
        _,
        row,
    ) in enumerate(
        layer.iterrows()
    ):
        building_code = str(
            row.get(
                "building_code",
                "",
            )
        )

        label = (
            f"{prefix}{building_code}"
            if prefix
            else building_code
        )

        dx, dy = offsets[
            idx % len(offsets)
        ]

        ax.annotate(
            label,
            (
                row.geometry.x,
                row.geometry.y,
            ),
            xytext=(
                dx,
                dy,
            ),
            textcoords="offset points",
            fontsize=6,
            alpha=0.9,
        )


def build_pending_connection_lines(
    buildings,
):
    pending = (
        buildings[
            buildings[
                "final_location_status"
            ]
            .isin(
                [
                    "review",
                    "unresolved",
                ]
            )
        ]
        .copy()
    )

    rows = []

    for _, row in pending.iterrows():
        geocoder_lon = row.get(
            "geocoder_longitude"
        )
        geocoder_lat = row.get(
            "geocoder_latitude"
        )

        osm_lon = row.get(
            "osm_candidate_longitude"
        )
        osm_lat = row.get(
            "osm_candidate_latitude"
        )

        values = [
            geocoder_lon,
            geocoder_lat,
            osm_lon,
            osm_lat,
        ]

        if any(
            pd.isna(value)
            for value in values
        ):
            continue

        line = LineString(
            [
                (
                    float(geocoder_lon),
                    float(geocoder_lat),
                ),
                (
                    float(osm_lon),
                    float(osm_lat),
                ),
            ]
        )

        rows.append(
            {
                "building_code":
                    row[
                        "building_code"
                    ],
                "geometry":
                    line,
            }
        )

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs=4326,
    )


def make_zoom_map(
    buildings,
    census,
    edges,
    layers,
    metric_crs,
    extent,
    municipality_code,
    building_year,
):
    census_m = census.to_crs(
        metric_crs
    )

    edges_m = edges.to_crs(
        metric_crs
    )

    validated_m = (
        layers[
            "validated"
        ]
        .to_crs(
            metric_crs
        )
    )

    accepted_m = (
        layers[
            "accepted"
        ]
        .to_crs(
            metric_crs
        )
    )

    pending_osm_m = (
        layers[
            "pending_osm"
        ]
        .to_crs(
            metric_crs
        )
    )

    census_clip = clip_to_extent(
        census_m,
        extent,
    )

    edges_clip = clip_to_extent(
        edges_m,
        extent,
    )

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    if not census_clip.empty:
        census_clip.boundary.plot(
            ax=ax,
            linewidth=0.35,
            alpha=0.3,
        )

    if not edges_clip.empty:
        edges_clip.plot(
            ax=ax,
            linewidth=0.35,
            alpha=0.35,
        )

    if not validated_m.empty:
        validated_m.plot(
            ax=ax,
            marker="o",
            markersize=80,
            label="Validated",
            zorder=5,
        )

    if not accepted_m.empty:
        accepted_m.plot(
            ax=ax,
            marker="^",
            markersize=85,
            label="Accepted address",
            zorder=5,
        )

    if not pending_osm_m.empty:
        pending_osm_m.plot(
            ax=ax,
            marker="x",
            markersize=85,
            label="OSM candidate - pending",
            zorder=6,
        )

    label_points(
        ax,
        validated_m,
    )

    label_points(
        ax,
        accepted_m,
    )

    label_points(
        ax,
        pending_osm_m,
        prefix="P:",
    )

    xmin, xmax, ymin, ymax = extent

    ax.set_xlim(
        xmin,
        xmax,
    )

    ax.set_ylim(
        ymin,
        ymax,
    )

    ax.set_title(
        (
            "School buildings QA - urban zoom - "
            f"{municipality_code} - {building_year}"
        )
    )

    ax.set_axis_off()
    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    output_dir = (
        FEATURES_DIR
        / "mim"
        / municipality_code
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_dir
        / (
            "school_buildings_qa_zoom_"
            f"{building_year}.png"
        )
    )

    pdf_path = (
        output_dir
        / (
            "school_buildings_qa_zoom_"
            f"{building_year}.pdf"
        )
    )

    fig.savefig(
        png_path,
        dpi=240,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        png_path,
        pdf_path,
    )


def make_pending_map(
    buildings,
    census,
    edges,
    layers,
    metric_crs,
    municipality_code,
    building_year,
):
    pending_osm = (
        layers[
            "pending_osm"
        ]
    )

    pending_geocoder = (
        layers[
            "pending_geocoder"
        ]
    )

    connection_lines = (
        build_pending_connection_lines(
            buildings
        )
    )

    reference_frames = []

    for layer in [
        pending_osm,
        pending_geocoder,
    ]:
        if not layer.empty:
            reference_frames.append(
                layer[
                    ["geometry"]
                ]
            )

    if not reference_frames:
        return (
            None,
            None,
        )

    reference = gpd.GeoDataFrame(
        pd.concat(
            reference_frames,
            ignore_index=True,
        ),
        geometry="geometry",
        crs=4326,
    ).to_crs(
        metric_crs
    )

    minx, miny, maxx, maxy = (
        reference.total_bounds
    )

    width = max(
        maxx - minx,
        1000.0,
    )

    height = max(
        maxy - miny,
        1000.0,
    )

    margin = max(
        700.0,
        0.10 * width,
        0.10 * height,
    )

    extent = (
        minx - margin,
        maxx + margin,
        miny - margin,
        maxy + margin,
    )

    census_m = census.to_crs(
        metric_crs
    )

    edges_m = edges.to_crs(
        metric_crs
    )

    pending_osm_m = (
        pending_osm.to_crs(
            metric_crs
        )
    )

    pending_geocoder_m = (
        pending_geocoder.to_crs(
            metric_crs
        )
    )

    lines_m = (
        connection_lines.to_crs(
            metric_crs
        )
        if not connection_lines.empty
        else connection_lines
    )

    census_clip = clip_to_extent(
        census_m,
        extent,
    )

    edges_clip = clip_to_extent(
        edges_m,
        extent,
    )

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    if not census_clip.empty:
        census_clip.boundary.plot(
            ax=ax,
            linewidth=0.35,
            alpha=0.3,
        )

    if not edges_clip.empty:
        edges_clip.plot(
            ax=ax,
            linewidth=0.35,
            alpha=0.35,
        )

    if not lines_m.empty:
        lines_m.plot(
            ax=ax,
            linewidth=0.8,
            linestyle="--",
            alpha=0.55,
            label="Geocoder ↔ OSM",
            zorder=3,
        )

    if not pending_geocoder_m.empty:
        pending_geocoder_m.plot(
            ax=ax,
            marker="^",
            markersize=80,
            label="Pending geocoder point",
            zorder=5,
        )

    if not pending_osm_m.empty:
        pending_osm_m.plot(
            ax=ax,
            marker="x",
            markersize=90,
            label="Pending OSM candidate",
            zorder=6,
        )

    label_points(
        ax,
        pending_osm_m,
    )

    xmin, xmax, ymin, ymax = extent

    ax.set_xlim(
        xmin,
        xmax,
    )

    ax.set_ylim(
        ymin,
        ymax,
    )

    ax.set_title(
        (
            "School buildings QA - pending cases - "
            f"{municipality_code} - {building_year}"
        )
    )

    ax.set_axis_off()
    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    output_dir = (
        FEATURES_DIR
        / "mim"
        / municipality_code
    )

    png_path = (
        output_dir
        / (
            "school_buildings_qa_pending_"
            f"{building_year}.png"
        )
    )

    pdf_path = (
        output_dir
        / (
            "school_buildings_qa_pending_"
            f"{building_year}.pdf"
        )
    )

    fig.savefig(
        png_path,
        dpi=240,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        png_path,
        pdf_path,
    )


def save_review_table(
    buildings,
    municipality_code,
    building_year,
):
    pending = (
        buildings[
            buildings[
                "final_location_status"
            ]
            .isin(
                [
                    "review",
                    "unresolved",
                ]
            )
        ]
        .copy()
    )

    columns = [
        column
        for column in [
            "building_code",
            "official_building_address",
            "geocoder_status",
            "geocoder_result_address",
            "geocoder_address_score",
            "geocoder_longitude",
            "geocoder_latitude",
            "osm_match_status",
            "osm_candidate_site_id",
            "osm_candidate_site_name",
            "osm_candidate_site_address",
            "osm_match_score",
            "osm_name_score",
            "osm_address_score",
            "osm_candidate_longitude",
            "osm_candidate_latitude",
            "final_location_status",
            "final_reason",
            "linked_school_codes",
            "linked_school_names",
        ]
        if column in buildings.columns
    ]

    output_dir = (
        FEATURES_DIR
        / "mim"
        / municipality_code
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / (
            "school_buildings_qa_review_"
            f"{building_year}.csv"
        )
    )

    pending[
        columns
    ].to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return (
        path,
        len(pending),
    )


def main():
    args = parse_args()

    print(
        "\n===================================="
    )
    print(
        " SCHOOL BUILDINGS QA V2"
    )
    print(
        "===================================="
    )

    (
        buildings,
        census,
        edges,
    ) = load_inputs(
        args.municipality_code,
        args.building_year,
    )

    layers = build_layers(
        buildings
    )

    metric_crs = metric_crs_from(
        census
    )

    reference_points = (
        all_reference_points(
            layers
        )
    )

    if reference_points is None:
        raise RuntimeError(
            "Nessun punto disponibile per costruire la QA."
        )

    reference_points_m = (
        reference_points.to_crs(
            metric_crs
        )
    )

    extent = calculate_extent(
        reference_points_m,
        args.margin_m,
    )

    (
        zoom_png,
        zoom_pdf,
    ) = make_zoom_map(
        buildings=buildings,
        census=census,
        edges=edges,
        layers=layers,
        metric_crs=metric_crs,
        extent=extent,
        municipality_code=(
            args.municipality_code
        ),
        building_year=(
            args.building_year
        ),
    )

    (
        pending_png,
        pending_pdf,
    ) = make_pending_map(
        buildings=buildings,
        census=census,
        edges=edges,
        layers=layers,
        metric_crs=metric_crs,
        municipality_code=(
            args.municipality_code
        ),
        building_year=(
            args.building_year
        ),
    )

    (
        review_csv,
        pending_count,
    ) = save_review_table(
        buildings,
        args.municipality_code,
        args.building_year,
    )

    print(
        "\n=== INPUT ==="
    )
    print(
        f"Edifici totali: {len(buildings)}"
    )
    print(
        "Validated: "
        f"{len(layers['validated'])}"
    )
    print(
        "Accepted address: "
        f"{len(layers['accepted'])}"
    )
    print(
        "Pending OSM points: "
        f"{len(layers['pending_osm'])}"
    )
    print(
        "Pending geocoder points: "
        f"{len(layers['pending_geocoder'])}"
    )

    print(
        "\n=== OUTPUT ==="
    )
    print(
        f"✓ Zoom PNG: {zoom_png}"
    )
    print(
        f"✓ Zoom PDF: {zoom_pdf}"
    )

    if pending_png:
        print(
            f"✓ Pending PNG: {pending_png}"
        )
        print(
            f"✓ Pending PDF: {pending_pdf}"
        )

    print(
        f"✓ Review CSV: {review_csv}"
    )

    print(
        "\n=== QA SUMMARY ==="
    )
    print(
        f"Casi pending nel dataset: {pending_count}"
    )
    print(
        "Nessuna soglia di matching è stata modificata."
    )


if __name__ == "__main__":
    main()
