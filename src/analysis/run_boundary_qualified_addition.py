"""Run approved service additions against an independently qualified V2 baseline.

The census-footprint/network preflight and the operator's source review are
separate requirements. The review document is an operator-controlled research
record, not a cryptographically signed or independently authenticated claim.
Without that review the runner stops; ANNCSU evidence in `hold` is not approval.
Original catalogues, B6C reports and graph snapshots remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from contextlib import ExitStack
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.accessibility_contracts import request_from_service_spec
from analysis.run_real_service_removal import _check_legacy_walk_network
from analysis.scenario_contracts_v2 import (
    ScenarioAction, ScenarioBaselineRef, ScenarioOperation, ScenarioSpec,
    request_sha256,
)
from analysis.scenario_reporting_v2 import (
    ScenarioInputPaths, _read_table, run_verified_scenario, write_scenario_report,
)
from analysis.service_scenarios_v2 import scenario_row_sha256
from core.analysis_spec import AnalysisSpec, ServiceType, TransportMode
from core.config import DEFAULT_CONFIG
from core.run_manifest import sha256_file
from quality.audit_service_addition_boundaries import inspect_boundary_additions
from quality.preflight_real_service_scenario import (
    ROOT, CitySnapshot, canonical_paths,
)

POLICY = "reviewed_boundary_qualified_addition_v1"
REVIEW_POLICY = "operator_reviewed_service_additions_v1"


def verify_source_review(
    *, review_file: Path, root: Path, snapshot: CitySnapshot,
    addition_services: pd.DataFrame, addition_attachments: pd.DataFrame,
) -> dict[str, Any]:
    """Require per-row, operator-maintained review separate from candidate input.

    This checks structured review provenance, NOT identity/signatures, the
    existence of remote evidence, or whether its conclusions are factually true.
    The review must be maintained independently by the research operator.
    """
    root = Path(root).resolve()
    review_file = Path(review_file).resolve()
    trusted_dir = (root / "config" / "scenario_source_reviews" / snapshot.municipality_code).resolve()
    if not review_file.is_relative_to(trusted_dir) or review_file.suffix != ".json":
        raise ValueError("Source review must reside in the operator-controlled municipality review directory")
    if not review_file.is_file():
        raise FileNotFoundError(f"Independent source review missing: {review_file}")
    original_sha = sha256_file(review_file)
    data = json.loads(review_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("policy") != REVIEW_POLICY:
        raise ValueError("Independent source review policy missing or invalid")
    for key, expected in (
        ("municipality_code", snapshot.municipality_code),
        ("service_type", snapshot.service_type.value),
        ("mode", snapshot.mode.value),
    ):
        if data.get(key) != expected:
            raise ValueError(f"Independent source review {key} mismatch")
    approvals = data.get("reviewed_candidates")
    if not isinstance(approvals, list) or len(approvals) != len(addition_services):
        raise ValueError("Independent review must cover every candidate exactly once")
    by_id: dict[str, dict[str, Any]] = {}
    for record in approvals:
        if not isinstance(record, dict) or not isinstance(record.get("service_id"), str):
            raise ValueError("Invalid source review record")
        sid = record["service_id"]
        if sid in by_id:
            raise ValueError("Duplicate candidate in independent source review")
        by_id[sid] = record
    service_ids = set(addition_services["service_id"].astype(str))
    if set(by_id) != service_ids:
        raise ValueError("Independent review IDs do not exactly match additions")
    att_by_id = addition_attachments.set_index("entity_id", drop=False)
    if not att_by_id.index.is_unique:
        raise ValueError("Candidate attachment IDs are not unique")
    for _, service in addition_services.iterrows():
        sid = str(service["service_id"])
        record = by_id[sid]
        # `approve_for_b6c_review`, `pending` and `hold` do NOT qualify.
        if record.get("decision") != "approved_for_scenario":
            raise ValueError(f"Candidate has no explicit scenario approval: {sid}")
        reviewer = record.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError(f"Named independent reviewer required: {sid}")
        reviewed_on = record.get("reviewed_on")
        if not isinstance(reviewed_on, str) or date.fromisoformat(reviewed_on).isoformat() != reviewed_on:
            raise ValueError(f"ISO review date required: {sid}")
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"Evidence required for scenario approval: {sid}")
        evidence_ids = set()
        for ev in evidence:
            if not isinstance(ev, dict) or any(
                not isinstance(ev.get(field), str) or not ev[field].strip()
                for field in ("source_id", "publisher", "locator")
            ):
                raise ValueError(f"Incomplete source evidence: {sid}")
            if ev["source_id"] in evidence_ids:
                raise ValueError(f"Duplicate source evidence: {sid}")
            evidence_ids.add(ev["source_id"])
        if record.get("service_row_sha256") != scenario_row_sha256(service):
            raise ValueError(f"Approved service row hash mismatch: {sid}")
        if record.get("attachment_row_sha256") != scenario_row_sha256(att_by_id.loc[sid]):
            raise ValueError(f"Approved attachment row hash mismatch: {sid}")
    if sha256_file(review_file) != original_sha:
        raise ValueError("Independent review changed during verification")
    return {
        "policy": REVIEW_POLICY,
        "review_file": str(review_file),
        "review_sha256": original_sha,
        "reviewed_candidate_ids": sorted(service_ids),
        "review_decisions": "all_approved_for_scenario",
        "authentication": "operator_maintained_review_not_cryptographically_signed",
        "source_contents_externally_revalidated": False,
    }



def _materialize_qualified_additions(
    *, candidates: pd.DataFrame, source_path: Path, target_path: Path,
    territorial_hashes: dict[str, str],
) -> None:
    """Derive a distinct, immutable routing view; never rewrite raw candidates.

    The trusted boundary flag is assigned only after the independent B7C.0
    preflight has recomputed the candidates' footprint membership.
    """
    if "inside_operational_footprint" in candidates:
        raise ValueError("Candidate cannot supply its own boundary flag")
    derived = candidates.copy(deep=True)
    derived["inside_operational_footprint"] = True
    derived["operational_footprint_sha256"] = territorial_hashes["footprint"]
    derived["canonical_services_sha256"] = territorial_hashes["canonical_services"]
    derived["territorial_policy"] = "b7c1_independently_qualified_candidate_view_v1"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".parquet":
        derived.to_parquet(target_path, index=False)
    else:
        derived.to_csv(target_path, index=False, lineterminator="\n", float_format="%.17g")


def execute_boundary_qualified_addition(
    *, snapshot: CitySnapshot, canonical: ScenarioInputPaths,
    qualified_services: Path, addition_services: Path, addition_attachments: Path,
    review_file: Path, city_name: str, analysis_date: str,
    root: Path = ROOT, write_report: bool = False,
    output_root: Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Use territorial preflight + source review before any addition routing."""
    if snapshot.mode is not TransportMode.WALK:
        raise ValueError("Qualified addition currently supports the walking regression profile only")
    if snapshot.service_type not in (
        ServiceType.PHARMACY, ServiceType.HOSPITAL_ESTABLISHMENT,
        ServiceType.LEGACY_EDUCATION_ALL,
    ):
        raise ValueError("Service type not supported by the regression profile")
    if not isinstance(city_name, str) or not city_name.strip():
        raise ValueError("Nonempty city name required")
    if date.fromisoformat(analysis_date).isoformat() != analysis_date:
        raise ValueError("ISO analysis date required")
    root = Path(root)
    qualified_services = Path(qualified_services)
    addition_services = Path(addition_services)
    addition_attachments = Path(addition_attachments)
    review_file = Path(review_file)
    if len({p.resolve() for p in (qualified_services, addition_services, addition_attachments, review_file)}) != 4:
        raise ValueError("Qualified view, additions and review must be independent files")
    # This recomputes the independent census-footprint proof, validates all
    # candidate coordinates, graph attachments, identities and file digests.
    territorial = inspect_boundary_additions(
        snapshot=snapshot, canonical=canonical,
        qualified_services=qualified_services,
        addition_services=addition_services,
        addition_attachments=addition_attachments, root=root,
    )
    if territorial.get("scenario_authorized") is not False:
        raise ValueError("Territorial preflight cannot authorize candidate source review")
    cand_hashes = territorial["verified_input_sha256"]
    candidates = _read_table(addition_services)
    attachments = _read_table(addition_attachments)
    source_review = verify_source_review(
        review_file=review_file, root=root, snapshot=snapshot,
        addition_services=candidates, addition_attachments=attachments,
    )
    # Recheck bytes from the independently certified candidate files.
    for key, path in (("candidate_services", addition_services),
                      ("candidate_attachments", addition_attachments),
                      ("qualified_services", qualified_services),
                      ("canonical_services", canonical.services)):
        if sha256_file(path) != cand_hashes[key]:
            raise ValueError(f"Verified input changed after territorial preflight: {key}")
    spec = AnalysisSpec.from_legacy_pipeline_config(
        city_name=city_name.strip(), municipality_code=snapshot.municipality_code,
        analysis_date=analysis_date, config=DEFAULT_CONFIG,
    )
    service_spec = next(s for s in spec.services if s.service_type is snapshot.service_type)
    request = request_from_service_spec(
        service_spec, snapshot.mode, travel_time_weight="walking_time_s",
        off_network_speed_m_s=spec.walking_speed_m_s,
    )
    _check_legacy_walk_network(_read_table(canonical.edges), spec.walking_speed_m_s)
    verified_baseline = replace(canonical, services=qualified_services)
    baseline_hashes = {key: sha256_file(value) for key, value in verified_baseline.items().items()}
    baseline = ScenarioBaselineRef(
        municipality_code=snapshot.municipality_code,
        analysis_spec_sha256=spec.spec_hash, request_sha256=request_sha256(request),
        origins_sha256=baseline_hashes["origins"],
        services_sha256=baseline_hashes["services"],
        attachments_sha256=baseline_hashes["attachments"],
        graph_sha256=cand_hashes["graph"],
    )
    # An addition must carry an operator-derived boundary flag. Concatenating
    # raw candidates (without the flag) into a qualified baseline would leave
    # NA values, which the B7B routing gate correctly excludes. Preserve raw
    # additions and derive an auditable separate file instead.
    stage_identity = {
        "policy": POLICY,
        "candidate_services_sha256": cand_hashes["candidate_services"],
        "candidate_attachments_sha256": cand_hashes["candidate_attachments"],
        "qualified_baseline_sha256": cand_hashes["qualified_services"],
        "footprint_sha256": cand_hashes["footprint"],
        "review_sha256": source_review["review_sha256"],
    }
    stem = hashlib.sha256(json.dumps(stage_identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    with ExitStack() as stage:
        if write_report:
            if output_root is None:
                raise ValueError("Explicit output_root required when writing reports")
            candidate_view = (Path(output_root) / "_verified_additions" /
                              snapshot.municipality_code / f"{stem}{addition_services.suffix.lower()}")
            # Stage independently, verify exact bytes, then publish only if
            # the target is new or byte-identical to this derivation.
            tmp_dir = Path(stage.enter_context(tempfile.TemporaryDirectory(prefix="addition-")))
            staged_candidate = tmp_dir / candidate_view.name
            _materialize_qualified_additions(
                candidates=candidates, source_path=addition_services,
                target_path=staged_candidate, territorial_hashes=cand_hashes,
            )
            candidate_view.parent.mkdir(parents=True, exist_ok=True)
            if candidate_view.exists():
                if candidate_view.read_bytes() != staged_candidate.read_bytes():
                    raise FileExistsError("Immutable qualified addition view differs")
            else:
                # Exclusive creation avoids overwriting a concurrent publisher.
                with candidate_view.open("xb") as out:
                    out.write(staged_candidate.read_bytes())
        else:
            tmp_dir = Path(stage.enter_context(tempfile.TemporaryDirectory(prefix="addition-preview-")))
            candidate_view = tmp_dir / f"{stem}{addition_services.suffix.lower()}"
            _materialize_qualified_additions(
                candidates=candidates, source_path=addition_services,
                target_path=candidate_view, territorial_hashes=cand_hashes,
            )
        qualified_candidates = _read_table(candidate_view)
        attachment_rows = attachments.set_index("entity_id", drop=False)
        operations = tuple(
            ScenarioOperation(
                ScenarioAction.ADD_SERVICE, str(service["service_id"]),
                scenario_row_sha256(service),
                scenario_row_sha256(attachment_rows.loc[str(service["service_id"])]),
            )
            for _, service in qualified_candidates.sort_values("service_id", kind="mergesort").iterrows()
        )
        scenario = ScenarioSpec(
            baseline=baseline, service_type=snapshot.service_type,
            mode=snapshot.mode, operations=operations,
        )
        input_paths = replace(
            verified_baseline, addition_services=candidate_view,
            addition_attachments=addition_attachments,
        )
        view_sha = sha256_file(candidate_view)
        result, provenance = run_verified_scenario(
            scenario=scenario, analysis_spec=spec, request=request, inputs=input_paths,
        )
        if sha256_file(candidate_view) != view_sha:
            raise ValueError("Qualified addition view changed during routing")
    before = result.metrics["baseline"]["service_count_routable"]
    after = result.metrics["scenario"]["service_count_routable"]
    if before != territorial["qualified_baseline_routable"]:
        raise AssertionError("Qualified baseline count changed after preflight")
    if after != before + len(candidates):
        raise AssertionError("Scenario did not add exactly the approved candidates")
    if result.metrics["baseline"]["target_population_total"] != result.metrics["scenario"]["target_population_total"]:
        raise AssertionError("Addition changed population")
    # Reject changed proof/review before publication, not only at load time.
    final_checks = (
        ("candidate_services", addition_services),
        ("candidate_attachments", addition_attachments),
        ("qualified_services", qualified_services),
        ("canonical_services", canonical.services),
    )
    for name, path in final_checks:
        if sha256_file(path) != cand_hashes[name]:
            raise ValueError(f"Scenario input modified during routing: {name}")
    if sha256_file(Path(source_review["review_file"])) != source_review["review_sha256"]:
        raise ValueError("Independent source review modified during routing")
    # Preserve a full immutable proof along with the scenario execution.
    provenance["territorial_qualification"] = territorial
    provenance["source_review"] = source_review
    provenance["addition_derivation"] = {
        **stage_identity, "derived_view_sha256": view_sha,
        "derived_view_path": str(candidate_view.resolve()) if write_report else None,
        "view_persisted": write_report,
    }
    provenance["input_paths"]["addition_services"] = (
        str(candidate_view.resolve()) if write_report else
        "ephemeral_read_only_preview"
    )
    provenance["addition_experiment"] = {"policy": POLICY, "candidate_ids": sorted(candidates["service_id"].astype(str))}
    summary = {
        "policy": POLICY, "scenario_id": result.scenario_id,
        "municipality_code": snapshot.municipality_code,
        "approved_addition_ids": source_review["reviewed_candidate_ids"],
        "qualified_baseline_service_count": before,
        "scenario_service_count": after,
        "metrics": result.metrics,
        "territorial_preflight": territorial,
        "source_review": source_review,
        "derived_addition_view_sha256": view_sha,
        "report_saved": False,
    }
    report_path = None
    if write_report:
        if output_root is None:
            raise ValueError("Explicit output_root required when writing reports")
        report_path, cached = write_scenario_report(
            result=result, provenance=provenance, output_root=output_root,
        )
        summary.update(report_saved=True, report_cached=cached,
                       report_directory=str(report_path))
    return summary, report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipality-code", required=True)
    parser.add_argument("--city-name")
    parser.add_argument("--analysis-date", required=True)
    parser.add_argument("--census-year", default="2023")
    parser.add_argument("--school-year", default="202425")
    parser.add_argument("--health-reference-date", default="2025-06-30")
    parser.add_argument("--service-type", default="pharmacy", choices=[s.value for s in ServiceType])
    parser.add_argument("--mode", default="walk", choices=[m.value for m in TransportMode])
    parser.add_argument("--qualified-services", type=Path, required=True)
    parser.add_argument("--addition-services", type=Path, required=True)
    parser.add_argument("--addition-attachments", type=Path, required=True)
    parser.add_argument("--review-file", type=Path, required=True)
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    snapshot = CitySnapshot(
        municipality_code=args.municipality_code, census_year=args.census_year,
        school_year=args.school_year, health_reference_date=args.health_reference_date,
        service_type=ServiceType(args.service_type), mode=TransportMode(args.mode),
    )
    summary, _ = execute_boundary_qualified_addition(
        snapshot=snapshot, canonical=canonical_paths(args.root, snapshot),
        qualified_services=args.qualified_services,
        addition_services=args.addition_services,
        addition_attachments=args.addition_attachments,
        review_file=args.review_file, root=args.root,
        city_name=args.city_name or f"Comune {snapshot.municipality_code}",
        analysis_date=args.analysis_date, write_report=args.write_report,
        output_root=(args.output_root or args.root / "data/features/scenarios/qualified_additions")
        if args.write_report else None,
    )
    print("=== QUALIFIED SERVICE ADDITION — %s ===" % ("SAVED" if args.write_report else "READ ONLY"))
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return summary


if __name__ == "__main__":
    run()
