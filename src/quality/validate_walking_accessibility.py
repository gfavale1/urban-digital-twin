import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FEATURES_ACCESSIBILITY_DIR = ROOT / "data" / "features" / "accessibility"

GROUPS = ["school", "pharmacy", "hospital"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="QA dei risultati di walking accessibility."
    )
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument(
        "--service-layer",
        choices=["enriched", "osm_only"],
        default="enriched",
    )
    parser.add_argument(
        "--thresholds-min",
        nargs="+",
        type=int,
        default=[10, 15, 20],
    )
    args = parser.parse_args()
    args.municipality_code = str(args.municipality_code).strip().zfill(6)
    args.health_reference_date = pd.Timestamp(
        args.health_reference_date
    ).normalize()
    args.thresholds_min = sorted(set(args.thresholds_min))

    if (
        not args.municipality_code.isdigit()
        or len(args.municipality_code) != 6
    ):
        raise ValueError(
            "--municipality-code deve avere esattamente 6 cifre."
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

    return args


def assert_close(a, b, label, tol=1e-6):
    if not np.isclose(float(a), float(b), atol=tol, rtol=1e-10):
        raise AssertionError(
            f"{label}: {a} != {b}"
        )


def main():
    args = parse_args()

    base = FEATURES_ACCESSIBILITY_DIR / args.municipality_code
    health_label = args.health_reference_date.strftime("%Y%m%d")

    if args.service_layer == "osm_only":
        suffix = f"osm_only_{args.census_year}"
    else:
        suffix = (
            f"{args.census_year}_"
            f"{args.school_year}_"
            f"{health_label}"
        )

    origins_path = base / f"walking_accessibility_origins_{suffix}.parquet"
    sections_path = base / f"walking_accessibility_sections_{suffix}.parquet"
    summary_path = base / f"walking_accessibility_summary_{suffix}.json"

    for path in [origins_path, sections_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    origins = pd.read_parquet(origins_path)
    sections = pd.read_parquet(sections_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    municipality = summary["municipality_summary"]

    failures = []
    checks = []

    if summary.get("service_layer") != args.service_layer:
        failures.append(
            "summary service_layer incompatibile: "
            f"atteso {args.service_layer}, "
            f"trovato {summary.get('service_layer')}"
        )

    summary_thresholds = sorted(
        int(value)
        for value in summary.get(
            "thresholds_min",
            [],
        )
    )

    if summary_thresholds != args.thresholds_min:
        failures.append(
            "summary thresholds_min incompatibili: "
            f"atteso {args.thresholds_min}, "
            f"trovato {summary_thresholds}"
        )

    population = float(origins["assigned_population"].sum())
    summary_population = float(municipality["population"])

    try:
        assert_close(population, summary_population, "population conservation")
        checks.append("population conservation")
    except Exception as exc:
        failures.append(str(exc))

    for group in GROUPS:
        reachable_col = f"{group}_reachable"
        time_col = f"{group}_nearest_time_min"
        dist_col = f"{group}_nearest_distance_m"

        if reachable_col not in origins.columns:
            failures.append(f"{group}: missing {reachable_col}")
            continue

        reachable = origins[reachable_col].astype(bool)

        negative_time = (
            pd.to_numeric(origins.loc[reachable, time_col], errors="coerce") < 0
        ).sum()
        negative_dist = (
            pd.to_numeric(origins.loc[reachable, dist_col], errors="coerce") < 0
        ).sum()

        if negative_time:
            failures.append(f"{group}: {negative_time} negative nearest times")
        else:
            checks.append(f"{group}: nonnegative nearest times")

        if negative_dist:
            failures.append(f"{group}: {negative_dist} negative nearest distances")
        else:
            checks.append(f"{group}: nonnegative nearest distances")

        unreachable_with_time = (
            (~reachable) & origins[time_col].notna()
        ).sum()

        if unreachable_with_time:
            failures.append(
                f"{group}: {unreachable_with_time} unreachable rows have time values"
            )
        else:
            checks.append(f"{group}: unreachable rows keep NA time")

        previous_bool = None
        previous_count = None

        for threshold in args.thresholds_min:
            bool_col = f"{group}_within_{threshold}_min"
            count_col = f"{group}_services_within_{threshold}_min"

            current_bool = origins[bool_col].astype(bool)
            current_count = pd.to_numeric(
                origins[count_col], errors="coerce"
            ).fillna(0)

            if previous_bool is not None:
                violations = (previous_bool & ~current_bool).sum()
                if violations:
                    failures.append(
                        f"{group}: {violations} rows violate threshold monotonicity "
                        f"before {threshold} min"
                    )

            if previous_count is not None:
                violations = (previous_count > current_count).sum()
                if violations:
                    failures.append(
                        f"{group}: {violations} rows violate service-count monotonicity "
                        f"before {threshold} min"
                    )

            previous_bool = current_bool
            previous_count = current_count

            origin_pop = float(
                origins.loc[current_bool, "assigned_population"].sum()
            )
            summary_pop = float(
                municipality[f"{group}_population_within_{threshold}_min"]
            )

            try:
                assert_close(
                    origin_pop,
                    summary_pop,
                    f"{group} population within {threshold} min",
                    tol=1e-5,
                )
                checks.append(
                    f"{group}: population within {threshold} min matches summary"
                )
            except Exception as exc:
                failures.append(str(exc))

            section_col = f"{group}_population_within_{threshold}_min"
            if section_col in sections.columns:
                section_pop = float(sections[section_col].sum())
                try:
                    assert_close(
                        section_pop,
                        origin_pop,
                        f"{group} section/origin aggregation within {threshold} min",
                        tol=1e-5,
                    )
                    checks.append(
                        f"{group}: section aggregation within {threshold} min"
                    )
                except Exception as exc:
                    failures.append(str(exc))

        reachable_pop = float(
            origins.loc[reachable, "assigned_population"].sum()
        )
        summary_reachable = float(
            municipality[f"{group}_reachable_population"]
        )

        try:
            assert_close(
                reachable_pop,
                summary_reachable,
                f"{group} reachable population",
                tol=1e-5,
            )
            checks.append(f"{group}: reachable population matches summary")
        except Exception as exc:
            failures.append(str(exc))

        max_threshold = max(args.thresholds_min)
        within_max_pop = float(
            municipality[
                f"{group}_population_within_{max_threshold}_min"
            ]
        )

        if within_max_pop > summary_reachable + 1e-6:
            failures.append(
                f"{group}: population within {max_threshold} min exceeds reachable"
            )

    print("\n====================================")
    print(" WALKING ACCESSIBILITY QA")
    print("====================================")
    print(f"Comune: {args.municipality_code}")
    print(f"Service layer: {args.service_layer}")
    print(f"Checks passed: {len(checks)}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\n✓ Tutti i controlli superati.")
    print(f"✓ Popolazione verificata: {population:.2f}")

    for group in GROUPS:
        print(f"\n{group.upper()}")
        print(
            "  reachable: "
            f"{municipality[f'{group}_reachable_population']:.2f} "
            f"({municipality[f'{group}_reachable_share'] * 100:.2f}%)"
        )
        for threshold in args.thresholds_min:
            print(
                f"  within {threshold:>2} min: "
                f"{municipality[f'{group}_population_share_within_{threshold}_min'] * 100:.2f}%"
            )


if __name__ == "__main__":
    main()
