from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .analysis_spec import AnalysisSpec


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    SKIPPED = "skipped"


class FreshnessStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class SourceRecord:
    source_name: str
    reference_date_or_period: str | None = None
    retrieved_at: str | None = None
    release_id_or_version: str | None = None
    checksum_sha256: str | None = None
    freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    source_url: str | None = None
    license_name: str | None = None
    warning: str | None = None


@dataclass(slots=True)
class StageRecord:
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    warnings: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunManifest:
    run_id: str
    created_at: str
    analysis_spec_hash: str
    analysis_spec: dict[str, Any]
    git_commit: str | None
    git_worktree_dirty: bool | None
    python_version: str
    dependency_lock_hash: str | None
    osmnx_version: str | None
    networkx_version: str | None
    routing_backend: str
    routing_parameters: dict[str, Any]
    crs: str
    graph_checksums: dict[str, str] = field(default_factory=dict)
    source_records: dict[str, SourceRecord] = field(default_factory=dict)
    source_checksums: dict[str, str] = field(default_factory=dict)
    stages: dict[str, StageRecord] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        spec: AnalysisSpec,
        *,
        repo_root: Path,
        routing_backend: str = "networkx",
        routing_parameters: dict[str, Any] | None = None,
    ) -> "RunManifest":
        created_at = utc_now_iso()
        short_hash = spec.spec_hash[:12]
        timestamp = created_at.replace(":", "").replace("-", "")[:15]
        run_id = f"{timestamp}_{short_hash}"

        manifest = cls(
            run_id=run_id,
            created_at=created_at,
            analysis_spec_hash=spec.spec_hash,
            analysis_spec=spec.to_dict(),
            git_commit=detect_git_commit(repo_root),
            git_worktree_dirty=detect_git_worktree_dirty(repo_root),
            python_version=platform.python_version(),
            dependency_lock_hash=detect_dependency_lock_hash(repo_root),
            osmnx_version=package_version("osmnx"),
            networkx_version=package_version("networkx"),
            routing_backend=routing_backend,
            routing_parameters=routing_parameters or {},
            crs=spec.crs_storage,
        )
        if manifest.git_worktree_dirty:
            manifest.warnings.append(
                "Git working tree is dirty; git_commit alone does not fully "
                "identify the executed source tree."
            )
        return manifest

    def register_source(self, key: str, record: SourceRecord) -> None:
        self.source_records[key] = record
        if record.checksum_sha256:
            self.source_checksums[key] = record.checksum_sha256

    def register_graph(self, mode: str, path: Path) -> str:
        checksum = sha256_file(path)
        self.graph_checksums[mode] = checksum
        return checksum

    def start_stage(self, name: str) -> StageRecord:
        stage = self.stages.setdefault(name, StageRecord(name=name))
        stage.status = StageStatus.RUNNING
        stage.started_at = utc_now_iso()
        return stage

    def complete_stage(
        self,
        name: str,
        *,
        outputs: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        degraded: bool = False,
        warnings: list[str] | None = None,
    ) -> None:
        stage = self.stages.setdefault(name, StageRecord(name=name))
        stage.status = StageStatus.DEGRADED if degraded else StageStatus.COMPLETED
        stage.completed_at = utc_now_iso()
        if outputs:
            stage.outputs.extend(outputs)
        if metrics:
            stage.metrics.update(metrics)
        if warnings:
            stage.warnings.extend(warnings)
            self.warnings.extend(f"{name}: {w}" for w in warnings)

    def fail_stage(self, name: str, message: str) -> None:
        stage = self.stages.setdefault(name, StageRecord(name=name))
        stage.status = StageStatus.FAILED
        stage.completed_at = utc_now_iso()
        stage.warnings.append(message)
        self.warnings.append(f"{name}: {message}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_records"] = {
            key: _enum_values(value) for key, value in payload["source_records"].items()
        }
        payload["stages"] = {
            key: _enum_values(value) for key, value in payload["stages"].items()
        }
        return _enum_values(payload)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def detect_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None



def detect_git_worktree_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())

def detect_dependency_lock_hash(repo_root: Path) -> str | None:
    for name in (
        "requirements.lock",
        "requirements.txt",
        "poetry.lock",
        "uv.lock",
        "pyproject.toml",
    ):
        path = repo_root / name
        if path.exists() and path.is_file():
            return sha256_file(path)
    return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _enum_values(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _enum_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_enum_values(v) for v in value]
    if isinstance(value, tuple):
        return [_enum_values(v) for v in value]
    return value
