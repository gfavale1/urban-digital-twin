from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path

from core.analysis_spec import AnalysisSpec
from core.config import DEFAULT_CONFIG, PipelineConfig
from core.municipality import MunicipalityContext
from core.paths import ROOT
from core.run_manifest import RunManifest


STAGES = (
    "istat",
    "osm",
    "education",
    "health",
    "services",
    "population",
    "accessibility",
    "osm_only",
    "comparison",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Urban Digital Twin pipeline. "
            "Orchestra la pipeline comunale zero-touch "
            "a partire dal codice ISTAT."
        )
    )

    parser.add_argument(
        "--municipality-code",
        required=True,
        help="Codice ISTAT comunale a 6 cifre.",
    )

    parser.add_argument(
        "--analysis-date",
        default=date.today().isoformat(),
        help=(
            "Data logica dell'analisi YYYY-MM-DD. "
            "Default: data odierna. Non modifica i risultati legacy; "
            "serve per provenance e temporal policy."
        ),
    )

    parser.add_argument(
        "--census-year",
        default=DEFAULT_CONFIG.census_year,
    )

    parser.add_argument(
        "--school-year",
        default=DEFAULT_CONFIG.school_year,
    )

    parser.add_argument(
        "--building-year",
        default=DEFAULT_CONFIG.building_year,
    )

    parser.add_argument(
        "--health-reference-date",
        default=DEFAULT_CONFIG.health_reference_date,
    )

    parser.add_argument(
        "--hospital-year",
        default=DEFAULT_CONFIG.hospital_year,
    )

    parser.add_argument(
        "--osm-reference-period",
        default=DEFAULT_CONFIG.osm_reference_period,
    )

    parser.add_argument(
        "--walking-speed",
        type=float,
        default=DEFAULT_CONFIG.walking_speed_m_s,
    )

    parser.add_argument(
        "--thresholds-min",
        type=int,
        nargs="+",
        default=list(
            DEFAULT_CONFIG.accessibility_thresholds_min
        ),
    )

    parser.add_argument(
        "--from-stage",
        choices=STAGES,
        default=STAGES[0],
    )

    parser.add_argument(
        "--to-stage",
        choices=STAGES,
        default=STAGES[-1],
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra i comandi senza eseguirli.",
    )

    return parser.parse_args()


def script_command(
    relative_script: str,
    *args,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / relative_script),
        *[
            str(value)
            for value in args
        ],
    ]


def selected_stages(
    first: str,
    last: str,
) -> tuple[str, ...]:
    first_index = STAGES.index(first)
    last_index = STAGES.index(last)

    if first_index > last_index:
        raise ValueError(
            "--from-stage deve precedere --to-stage."
        )

    return STAGES[
        first_index:
        last_index + 1
    ]


def commands_for_stage(
    stage: str,
    ctx: MunicipalityContext,
    config: PipelineConfig,
) -> list[list[str]]:
    code = ctx.code

    if stage == "istat":
        return [
            script_command(
                "src/ingestion/istat.py",
                "--region-code",
                ctx.region_code,
                "--municipality-code",
                code,
            ),
        ]

    if stage == "osm":
        return [
            script_command(
                "src/ingestion/osm_network.py",
                "--municipality-code",
                code,
                "--walking-speed",
                config.walking_speed_m_s,
            ),
        ]

    if stage == "education":
        return [
            script_command(
                "src/ingestion/mim.py",
                "--step",
                "prepare",
                "--municipality-code",
                code,
                "--municipality-name",
                ctx.name_upper,
                "--school-year",
                config.school_year,
                "--building-year",
                config.building_year,
            ),

            script_command(
                "src/matching/education.py",
                "--step",
                "prepare",
                "--municipality-code",
                code,
                "--school-year",
                config.school_year,
                "--building-year",
                config.building_year,
            ),

            script_command(
                "src/transformation/education.py",
                "--step",
                "prepare",
                "--municipality-code",
                code,
                "--school-year",
                config.school_year,
                "--building-year",
                config.building_year,
            ),
        ]

    if stage == "health":
        return [
            script_command(
                "src/ingestion/health.py",
                "--municipality-code",
                code,
                "--reference-date",
                config.health_reference_date,
                "--hospital-year",
                config.hospital_year,
            ),

            script_command(
                "src/matching/health.py",
                "--municipality-code",
                code,
                "--pharmacy-reference-date",
                config.health_reference_date,
                "--hospital-year",
                config.hospital_year,
            ),

            script_command(
                "src/transformation/health.py",
                "--step",
                "prepare",
                "--municipality-code",
                code,
                "--pharmacy-reference-date",
                config.health_reference_date,
                "--hospital-year",
                config.hospital_year,
            ),
        ]

    if stage == "services":
        return [
            script_command(
                "src/transformation/build_service_layer.py",
                "--municipality-code",
                code,
                "--school-year",
                config.school_year,
                "--health-reference-date",
                config.health_reference_date,
                "--hospital-year",
                config.hospital_year,
            ),
        ]

    if stage == "population":
        return [
            script_command(
                "src/transformation/"
                "build_population_network_origins.py",
                "--municipality-code",
                code,
                "--census-year",
                config.census_year,
            ),
        ]

    if stage == "accessibility":
        return [
            script_command(
                "src/transformation/"
                "snap_service_sites_to_network.py",
                "--municipality-code",
                code,
                "--school-year",
                config.school_year,
                "--health-reference-date",
                config.health_reference_date,
                "--service-layer",
                "enriched",
            ),

            script_command(
                "src/analysis/"
                "compute_walking_accessibility.py",
                "--municipality-code",
                code,
                "--census-year",
                config.census_year,
                "--school-year",
                config.school_year,
                "--health-reference-date",
                config.health_reference_date,
                "--service-layer",
                "enriched",
                "--walking-speed-m-s",
                str(config.walking_speed_m_s),
                "--thresholds-min",
                *config.accessibility_thresholds_min,
            ),

            script_command(
                "src/quality/"
                "validate_walking_accessibility.py",
                "--municipality-code",
                code,
                "--census-year",
                config.census_year,
                "--school-year",
                config.school_year,
                "--health-reference-date",
                config.health_reference_date,
                "--service-layer",
                "enriched",
                "--thresholds-min",
                *config.accessibility_thresholds_min,
            ),
        ]

    if stage == "osm_only":
        return [
            script_command(
                "src/transformation/"
                "build_osm_only_service_layer.py",
                "--municipality-code",
                code,
                "--reference-period",
                config.osm_reference_period,
            ),

            script_command(
                "src/transformation/"
                "snap_service_sites_to_network.py",
                "--municipality-code",
                code,
                "--school-year",
                config.school_year,
                "--health-reference-date",
                config.health_reference_date,
                "--service-layer",
                "osm_only",
            ),

            script_command(
                "src/analysis/"
                "compute_walking_accessibility.py",
                "--municipality-code",
                code,
                "--census-year",
                config.census_year,
                "--school-year",
                config.school_year,
                "--health-reference-date",
                config.health_reference_date,
                "--service-layer",
                "osm_only",
                "--walking-speed-m-s",
                str(config.walking_speed_m_s),
                "--thresholds-min",
                *config.accessibility_thresholds_min,
            ),
            script_command(
                "src/quality/"
                "validate_walking_accessibility.py",
                "--municipality-code",
                code,
                "--census-year",
                config.census_year,
                "--school-year",
                config.school_year,
                "--health-reference-date",
                config.health_reference_date,
                "--service-layer",
                "osm_only",
                "--thresholds-min",
                *config.accessibility_thresholds_min,
            ),
        ]

    if stage == "comparison":
        return [
            script_command(
                "src/analysis/"
                "compare_accessibility_scenarios.py",
                "--municipality-code",
                code,
                "--census-year",
                config.census_year,
                "--school-year",
                config.school_year,
                "--health-reference-date",
                config.health_reference_date,
                "--thresholds-min",
                *config.accessibility_thresholds_min,
            ),
        ]

    raise ValueError(
        f"Stage sconosciuto: {stage}"
    )


def print_context(
    ctx: MunicipalityContext,
    config: PipelineConfig,
    stages: tuple[str, ...],
):
    print(
        "\n=============================================="
    )
    print(
        " URBAN DIGITAL TWIN PIPELINE"
    )
    print(
        "=============================================="
    )

    print(
        f"Comune: {ctx.name}"
    )
    print(
        f"Codice ISTAT: {ctx.code}"
    )
    print(
        f"Provincia ISTAT: {ctx.province_code}"
    )
    print(
        f"Regione ISTAT: {ctx.region_code}"
    )

    print(
        "\nSnapshot:"
    )
    print(
        f"  Census: {config.census_year}"
    )
    print(
        f"  School: {config.school_year}"
    )
    print(
        f"  Buildings: {config.building_year}"
    )
    print(
        f"  Health: {config.health_reference_date}"
    )
    print(
        f"  Hospitals: {config.hospital_year}"
    )
    print(
        f"  OSM reference: "
        f"{config.osm_reference_period}"
    )
    print(
        f"  Walking speed: "
        f"{config.walking_speed_m_s} m/s"
    )
    print(
        f"  Thresholds: "
        f"{list(config.accessibility_thresholds_min)}"
    )

    print(
        "\nStages:"
    )

    for stage in stages:
        print(
            f"  - {stage}"
        )


def build_legacy_analysis_spec(
    ctx: MunicipalityContext,
    config: PipelineConfig,
    analysis_date: str,
) -> AnalysisSpec:
    """Describe the current v1 execution without changing its numerics."""
    return AnalysisSpec.from_legacy_pipeline_config(
        city_name=ctx.name,
        municipality_code=ctx.code,
        analysis_date=analysis_date,
        config=config,
    )


def initialize_run_metadata(
    spec: AnalysisSpec,
) -> tuple[RunManifest, Path, Path, Path]:
    manifest = RunManifest.create(
        spec,
        repo_root=ROOT,
        routing_backend="networkx_legacy_walking",
        routing_parameters={
            "execution_profile": spec.execution_profile,
            "walking_speed_m_s": spec.walking_speed_m_s,
        },
    )

    run_dir = ROOT / "runs" / manifest.run_id
    spec_path = run_dir / "analysis_spec.json"
    manifest_path = run_dir / "run_manifest.json"

    spec.write_json(spec_path)
    manifest.write_json(manifest_path)

    return manifest, run_dir, spec_path, manifest_path


def persist_manifest(
    manifest: RunManifest,
    manifest_path: Path,
) -> None:
    manifest.write_json(manifest_path)


def run_stage(
    *,
    stage: str,
    stage_commands: list[list[str]],
    manifest: RunManifest,
    manifest_path: Path,
) -> None:
    manifest.start_stage(stage)
    stage_record = manifest.stages[stage]
    stage_record.metrics["command_count"] = len(stage_commands)
    stage_record.metrics["commands"] = [
        shlex.join(command)
        for command in stage_commands
    ]
    persist_manifest(manifest, manifest_path)

    try:
        for command in stage_commands:
            run_command(command=command, dry_run=False)
    except subprocess.CalledProcessError as exc:
        manifest.fail_stage(
            stage,
            f"Subprocess failed with return code {exc.returncode}: "
            f"{shlex.join(exc.cmd) if isinstance(exc.cmd, list) else exc.cmd}",
        )
        persist_manifest(manifest, manifest_path)
        raise
    except Exception as exc:
        manifest.fail_stage(stage, f"{type(exc).__name__}: {exc}")
        persist_manifest(manifest, manifest_path)
        raise

    manifest.complete_stage(stage)
    persist_manifest(manifest, manifest_path)


def run_command(
    command: list[str],
    dry_run: bool,
):
    print(
        "\n$ "
        + shlex.join(command)
    )

    if dry_run:
        return

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main():
    args = parse_args()

    config = PipelineConfig(
        census_year=str(args.census_year),
        school_year=str(args.school_year),
        building_year=str(args.building_year),
        health_reference_date=str(
            args.health_reference_date
        ),
        hospital_year=str(args.hospital_year),
        osm_reference_period=str(
            args.osm_reference_period
        ),
        walking_speed_m_s=float(
            args.walking_speed
        ),
        accessibility_thresholds_min=tuple(
            args.thresholds_min
        ),
    )

    ctx = MunicipalityContext.resolve(
        municipality_code=args.municipality_code,
        census_year=config.census_year,
    )

    stages = selected_stages(
        args.from_stage,
        args.to_stage,
    )

    spec = build_legacy_analysis_spec(
        ctx=ctx,
        config=config,
        analysis_date=str(args.analysis_date),
    )

    print_context(
        ctx=ctx,
        config=config,
        stages=stages,
    )

    print(
        f"\nAnalysisSpec profile: {spec.execution_profile}"
    )
    print(
        f"AnalysisSpec hash: {spec.spec_hash}"
    )

    stage_commands = {
        stage: commands_for_stage(
            stage=stage,
            ctx=ctx,
            config=config,
        )
        for stage in stages
    }

    command_count = sum(
        len(commands)
        for commands in stage_commands.values()
    )

    print(
        f"\nComandi da eseguire: {command_count}"
    )

    if args.dry_run:
        for stage in stages:
            for command in stage_commands[stage]:
                run_command(
                    command=command,
                    dry_run=True,
                )

        print(
            "\n✓ Dry-run completato. "
            "Nessun comando eseguito e nessun run artifact creato."
        )
        return

    (
        manifest,
        run_dir,
        spec_path,
        manifest_path,
    ) = initialize_run_metadata(spec)

    print(
        f"\nRun ID: {manifest.run_id}"
    )
    print(
        f"Run directory: {run_dir}"
    )
    print(
        f"AnalysisSpec: {spec_path}"
    )
    print(
        f"RunManifest: {manifest_path}"
    )

    for stage in stages:
        run_stage(
            stage=stage,
            stage_commands=stage_commands[stage],
            manifest=manifest,
            manifest_path=manifest_path,
        )

    print(
        "\n=============================================="
    )
    print(
        " PIPELINE COMPLETATA"
    )
    print(
        "=============================================="
    )
    print(
        f"Run manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()