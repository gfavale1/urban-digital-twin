from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Mapping


METHODOLOGY_VERSION = "2.0.0-design-freeze"


class TransportMode(str, Enum):
    WALK = "walk"
    DRIVE = "drive"


class ServiceType(str, Enum):
    PRESCHOOL = "preschool"
    PRIMARY_SCHOOL = "primary_school"
    LOWER_SECONDARY_SCHOOL = "lower_secondary_school"
    UPPER_SECONDARY_SCHOOL = "upper_secondary_school"
    PHARMACY = "pharmacy"
    COMMUNITY_HOUSE = "community_house"
    HOSPITAL_ESTABLISHMENT = "hospital_establishment"
    LEGACY_EDUCATION_ALL = "legacy_education_all"


class PopulationSelector(str, Enum):
    """Population denominator used by a service accessibility metric.

    School selectors are explicitly proxies because the ISTAT section-level
    dataset exposes five-year age bands rather than exact school-age cohorts.
    """

    TOTAL = "population_total"
    AGE_LT5_PROXY = "population_age_lt5_proxy"
    AGE_5_9_PROXY = "population_age_5_9_proxy"
    AGE_10_14_PROXY = "population_age_10_14_proxy"
    AGE_15_19_PROXY = "population_age_15_19_proxy"


@dataclass(frozen=True, slots=True)
class ServiceAnalysisSpec:
    service_type: ServiceType
    modes: tuple[TransportMode, ...]
    thresholds_min_by_mode: Mapping[str, tuple[int, ...]]
    population_selector: PopulationSelector
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("A service analysis must define at least one mode.")

        normalized: dict[str, tuple[int, ...]] = {}
        for mode in self.modes:
            key = mode.value
            if key not in self.thresholds_min_by_mode:
                raise ValueError(
                    f"Missing thresholds for service={self.service_type.value}, mode={key}."
                )
            values = tuple(int(v) for v in self.thresholds_min_by_mode[key])
            if not values or any(v <= 0 for v in values):
                raise ValueError("Thresholds must be positive integers.")
            if tuple(sorted(set(values))) != values:
                raise ValueError(
                    "Thresholds must be unique and strictly increasing."
                )
            normalized[key] = values

        object.__setattr__(self, "thresholds_min_by_mode", normalized)


@dataclass(frozen=True, slots=True)
class GuardAreaSpec:
    strategy: str = "adaptive_frontier"
    expansion_factor: float = 2.0
    max_iterations: int = 6
    post_convergence_validation: bool = True
    convergence_tolerance: float = 1e-9
    # Intentionally unresolved here: GraphBuilder may derive a deterministic
    # initial envelope per mode. The resolved value must be written to the run manifest.
    initial_buffer_m_by_mode: Mapping[str, float | None] = field(
        default_factory=lambda: {"walk": None, "drive": None}
    )

    def __post_init__(self) -> None:
        if self.strategy != "adaptive_frontier":
            raise ValueError("Only adaptive_frontier is frozen for methodology v2.")
        if self.expansion_factor <= 1:
            raise ValueError("Guard-area expansion_factor must be > 1.")
        if self.max_iterations < 1:
            raise ValueError("Guard-area max_iterations must be >= 1.")
        if self.convergence_tolerance < 0:
            raise ValueError("Guard-area convergence_tolerance must be >= 0.")


@dataclass(frozen=True, slots=True)
class SensitivitySpec:
    enabled: bool = True
    one_at_a_time: bool = True
    walking_speeds_m_s: tuple[float, ...] = (0.7, 0.9, 1.1)
    origin_methods: tuple[str, ...] = (
        "representative_point",
        "geometric_centroid",
    )
    vary_threshold_profile: bool = True
    vary_guard_area: bool = True
    vary_snap_threshold: bool = False
    vary_driving_speed_assumptions: bool = False

    def __post_init__(self) -> None:
        if not self.one_at_a_time:
            raise ValueError(
                "Methodology v2 freezes one-at-a-time sensitivity, not full factorial."
            )
        if any(v <= 0 for v in self.walking_speeds_m_s):
            raise ValueError("Walking sensitivity speeds must be positive.")


@dataclass(frozen=True, slots=True)
class TemporalPolicySpec:
    """Global temporal rule. Source-specific freshness rules live in adapters.

    The method intentionally avoids pretending all datasets share one vintage.
    Each adapter must record its reference date/period and freshness status.
    """

    strategy: str = "latest_official_not_after_analysis_date"
    require_reference_vintage: bool = True
    allow_stale_if_latest_official: bool = True
    community_house_failure_is_non_blocking: bool = True


@dataclass(frozen=True, slots=True)
class AnalysisSpec:
    city_name: str
    analysis_date: str
    methodology_version: str = METHODOLOGY_VERSION
    municipality_code: str | None = None
    services: tuple[ServiceAnalysisSpec, ...] = field(default_factory=tuple)
    walking_speed_m_s: float = 0.9
    guard_area: GuardAreaSpec = field(default_factory=GuardAreaSpec)
    sensitivity: SensitivitySpec = field(default_factory=SensitivitySpec)
    temporal_policy: TemporalPolicySpec = field(default_factory=TemporalPolicySpec)
    max_snap_distance_m: float = 1000.0  # legacy-compatible initial default
    crs_storage: str = "EPSG:4326"
    execution_profile: str = "methodology_v2"

    def __post_init__(self) -> None:
        city = self.city_name.strip()
        if not city:
            raise ValueError("city_name cannot be empty.")
        object.__setattr__(self, "city_name", city)

        try:
            date.fromisoformat(self.analysis_date)
        except ValueError as exc:
            raise ValueError("analysis_date must be YYYY-MM-DD.") from exc

        if self.municipality_code is not None:
            code = str(self.municipality_code).strip().zfill(6)
            if not (code.isdigit() and len(code) == 6):
                raise ValueError("municipality_code must contain exactly 6 digits.")
            object.__setattr__(self, "municipality_code", code)

        if self.walking_speed_m_s <= 0:
            raise ValueError("walking_speed_m_s must be positive.")
        if self.max_snap_distance_m <= 0:
            raise ValueError("max_snap_distance_m must be positive.")
        if not self.services:
            raise ValueError("AnalysisSpec must include at least one service.")

        service_types = [s.service_type for s in self.services if s.enabled]
        if len(service_types) != len(set(service_types)):
            raise ValueError("Duplicate enabled service_type in AnalysisSpec.")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def write_json(self, path: "Path") -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def max_threshold_min(self, mode: TransportMode) -> int:
        values: list[int] = []
        for service in self.services:
            if not service.enabled or mode not in service.modes:
                continue
            values.extend(service.thresholds_min_by_mode[mode.value])
        if not values:
            raise ValueError(f"No enabled thresholds for mode={mode.value}.")
        return max(values)

    @classmethod
    def default_for_city(
        cls,
        city_name: str,
        *,
        analysis_date: str,
        municipality_code: str | None = None,
    ) -> "AnalysisSpec":
        return cls(
            city_name=city_name,
            municipality_code=municipality_code,
            analysis_date=analysis_date,
            services=default_service_specs(),
        )

    @classmethod
    def from_legacy_pipeline_config(
        cls,
        *,
        city_name: str,
        municipality_code: str,
        analysis_date: str,
        config: Any,
    ) -> "AnalysisSpec":
        """Compatibility bridge for regression before pipeline v2 is wired.

        It preserves the legacy walking speed and thresholds so the new
        contracts can be introduced without changing old numerical baselines.
        """
        thresholds = tuple(int(v) for v in config.accessibility_thresholds_min)
        walking_services = tuple(
            ServiceAnalysisSpec(
                service_type=service_type,
                modes=(TransportMode.WALK,),
                thresholds_min_by_mode={"walk": thresholds},
                population_selector=PopulationSelector.TOTAL,
            )
            for service_type in (
                ServiceType.LEGACY_EDUCATION_ALL,
                ServiceType.PHARMACY,
                ServiceType.HOSPITAL_ESTABLISHMENT,
            )
        )
        return cls(
            city_name=city_name,
            municipality_code=municipality_code,
            analysis_date=analysis_date,
            services=walking_services,
            walking_speed_m_s=float(config.walking_speed_m_s),
            execution_profile="legacy_v1_regression",
        )


def default_service_specs() -> tuple[ServiceAnalysisSpec, ...]:
    walk = TransportMode.WALK
    drive = TransportMode.DRIVE

    def spec(
        service_type: ServiceType,
        modes: tuple[TransportMode, ...],
        population_selector: PopulationSelector,
        *,
        walk_thresholds: tuple[int, ...] | None = None,
        drive_thresholds: tuple[int, ...] | None = None,
    ) -> ServiceAnalysisSpec:
        thresholds: dict[str, tuple[int, ...]] = {}
        if walk_thresholds is not None:
            thresholds[walk.value] = walk_thresholds
        if drive_thresholds is not None:
            thresholds[drive.value] = drive_thresholds
        return ServiceAnalysisSpec(
            service_type=service_type,
            modes=modes,
            thresholds_min_by_mode=thresholds,
            population_selector=population_selector,
        )

    return (
        spec(
            ServiceType.PRESCHOOL,
            (walk, drive),
            PopulationSelector.AGE_LT5_PROXY,
            walk_thresholds=(5, 10, 15),
            drive_thresholds=(10, 20, 30),
        ),
        spec(
            ServiceType.PRIMARY_SCHOOL,
            (walk, drive),
            PopulationSelector.AGE_5_9_PROXY,
            walk_thresholds=(5, 10, 15),
            drive_thresholds=(10, 20, 30),
        ),
        spec(
            ServiceType.LOWER_SECONDARY_SCHOOL,
            (walk, drive),
            PopulationSelector.AGE_10_14_PROXY,
            walk_thresholds=(5, 10, 15),
            drive_thresholds=(10, 20, 30),
        ),
        spec(
            ServiceType.UPPER_SECONDARY_SCHOOL,
            (drive,),
            PopulationSelector.AGE_15_19_PROXY,
            drive_thresholds=(10, 20, 30),
        ),
        spec(
            ServiceType.PHARMACY,
            (walk, drive),
            PopulationSelector.TOTAL,
            walk_thresholds=(5, 10, 15),
            drive_thresholds=(10, 20, 30),
        ),
        spec(
            ServiceType.COMMUNITY_HOUSE,
            (walk, drive),
            PopulationSelector.TOTAL,
            walk_thresholds=(5, 10, 15),
            drive_thresholds=(10, 20, 30),
        ),
        spec(
            ServiceType.HOSPITAL_ESTABLISHMENT,
            (drive,),
            PopulationSelector.TOTAL,
            drive_thresholds=(15, 30, 45),
        ),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value
