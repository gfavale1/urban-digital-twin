"""Read-only comparison of canonical V2 service-routing quality across municipalities.

A missing input means NOT AUDITED, never zero services or a successful QA check.
Each available municipality is audited independently against its own 2021 census
footprint, canonical services, network and attachments. No files are written.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Sequence

from core.analysis_spec import TransportMode
from quality.audit_service_routing_quality import inspect_service_quality
from quality.audit_v2_service_boundary import paths_for_snapshot
from quality.preflight_real_service_scenario import ROOT, CitySnapshot

POLICY = "multicity_canonical_service_quality_comparison_v1"
DEFAULT_MUNICIPALITIES = ("034027", "077014", "016024")  # Parma, Matera, Bergamo: examples only.


def inspect_municipalities(
    municipality_codes: Sequence[str], *, root: Path = ROOT,
    mode: TransportMode = TransportMode.WALK, census_year: str = "2023",
    school_year: str = "202425", health_reference_date: str = "2025-06-30",
) -> dict[str, Any]:
    """Inspect available V2 snapshots; preserve errors in present-but-invalid input.

    No implicit acquisition, no artifacts written and no fallback to legacy data.
    Corrupt or inconsistent existing inputs raise immediately (fail closed).
    """
    codes = list(municipality_codes)
    if not codes or any(not isinstance(c, str) or not re.fullmatch(r"\d{6}", c) for c in codes):
        raise ValueError("Supply one or more six-digit ISTAT municipality codes")
    if len(set(codes)) != len(codes):
        raise ValueError("Municipality codes must be unique")
    root = Path(root)
    mode = TransportMode(mode)
    cities: dict[str, dict[str, Any]] = {}

    for code in codes:
        snapshot = CitySnapshot(
            municipality_code=code, mode=mode, census_year=census_year,
            school_year=school_year, health_reference_date=health_reference_date,
        )
        files = paths_for_snapshot(root, snapshot).items()
        missing = {name: str(path) for name, path in files.items() if not path.is_file()}
        if missing:
            cities[code] = {
                "status": "missing_inputs", "missing_inputs": missing,
                "required_inputs": {name: str(path) for name, path in files.items()},
            }
            continue

        # A corrupt present snapshot is different from a missing one. Report
        # its error, continue inspecting the other cities and fail the CLI.
        # Unexpected programming exceptions are deliberately not suppressed.
        try:
            report = inspect_service_quality(snapshot, root=root)
            if report["municipality_code"] != code or report["mode"] != mode.value:
                raise ValueError(f"Auditor returned the wrong city or transport mode for {code}")
            total = int(report["services_total"])
            routable = int(report["services_routable"])
            excluded = int(report["services_ineligible"])
            if total < 0 or routable < 0 or excluded < 0 or routable + excluded != total:
                raise ValueError(f"Invalid routing totals for {code}")
            service_types = report["service_types"]
            if (sum(int(t["total"]) for t in service_types.values()) != total
                    or sum(int(t["routable"]) for t in service_types.values()) != routable):
                raise ValueError(f"Service-type totals do not reconcile for {code}")
        except (FileNotFoundError, ValueError, KeyError, AssertionError) as exc:
            cities[code] = {"status": "invalid", "error": f"{type(exc).__name__}: {exc}"}
            continue
        cities[code] = {"status": "audited", "report": report}

    audited = sum(city["status"] == "audited" for city in cities.values())
    invalid = sum(city["status"] == "invalid" for city in cities.values())
    missing_count = len(codes) - audited - invalid
    return {
        "policy": POLICY, "mode": mode.value, "municipality_codes": codes,
        "snapshot_parameters": {
            "census_year": census_year, "school_year": school_year,
            "health_reference_date": health_reference_date,
        },
        "audited_cities": audited, "missing_cities": missing_count,
        "invalid_cities": invalid, "all_cities_audited": audited == len(codes), "cities": cities,
        "interpretation": (
            "Not audited means unavailable V2 inputs, never zero facilities. "
            "Invalid means the audit did not pass and the command exits nonzero. "
            "Routing eligibility is not a census of all actually operating sites. "
            "The territorial criterion is the 2021 census footprint, not the official municipal boundary."
        ),
    }


def format_report(result: dict[str, Any]) -> str:
    """Readable comparison; N/A always means an entire city was not audited."""
    codes = result["municipality_codes"]
    cities = result["cities"]
    lines = [
        "=== MULTI-CITY V2 SERVICE QUALITY — READ ONLY ===",
        f"Mode: {result['mode']} | audited: {result['audited_cities']}/{len(codes)} "
        f"| unavailable: {result['missing_cities']} | invalid: {result['invalid_cities']}",
    ]
    for code in codes:
        city = cities[code]
        if city["status"] == "missing_inputs":
            lines.append(f"{code}: NOT AUDITED — {len(city['missing_inputs'])} required input(s) missing")
            for name, path in sorted(city["missing_inputs"].items()):
                lines.append(f"  missing {name}: {path}")
        elif city["status"] == "invalid":
            lines.append(f"{code}: INVALID — {city['error']}")
        else:
            q = city["report"]
            lines.append(f"{code}: AUDITED — {q['services_routable']}/{q['services_total']} routable "
                         f"({q['services_ineligible']} excluded)")
            lines.append(f"  primary exclusion reasons: {q['primary_routing_reason']}")
            lines.append(f"  location breakdown: {q['geography']}")
    categories = sorted({t for city in cities.values() if city["status"] == "audited"
                         for t in city["report"]["service_types"]})
    if categories:
        width = max(13, *(len(t) for t in categories))
        lines.append("\nCATEGORY COMPARISON (routable/total; N/A = missing V2 inputs)")
        lines.append("  " + "Category".ljust(width) + "  " + "  ".join(codes))
        for category in categories:
            values = []
            for code in codes:
                city = cities[code]
                if city["status"] != "audited":
                    values.append("N/A")
                else:
                    cat = city["report"]["service_types"].get(category, {"total": 0, "routable": 0})
                    values.append(f"{cat['routable']}/{cat['total']}")
            lines.append("  " + category.ljust(width) + "  " + "  ".join(values))
    lines.append("Operational census footprint only; not an administrative boundary.")
    lines.append("No datasets or reports modified. Missing/invalid cities are NOT validated.")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-codes", nargs="+", default=list(DEFAULT_MUNICIPALITIES))
    parser.add_argument("--mode", choices=[mode.value for mode in TransportMode], default="walk")
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-all", action="store_true",
                        help="Exit nonzero if any city lacks canonical V2 inputs")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = inspect_municipalities(
        args.municipality_codes, root=args.root, mode=TransportMode(args.mode),
        census_year=args.census_year, school_year=args.school_year,
        health_reference_date=args.health_reference_date,
    )
    print(format_report(result))
    if result["invalid_cities"]:
        raise SystemExit(1)
    if args.require_all and result["missing_cities"]:
        raise SystemExit(2)
    return result


if __name__ == "__main__":
    run()
