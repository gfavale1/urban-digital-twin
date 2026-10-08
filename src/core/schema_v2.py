from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .analysis_spec import PopulationSelector, ServiceType, TransportMode


SCHEMA_VERSION = "2.0.0"


class OperationalStatus(str, Enum):
    ACTIVE = "active"
    PLANNED = "planned"
    IN_CONSTRUCTION = "in_construction"
    UNKNOWN = "unknown"


class GeocodeQuality(str, Enum):
    SOURCE_COORDINATE = "source_coordinate"
    ADDRESS_EXACT = "address_exact"
    ADDRESS_APPROXIMATE = "address_approximate"
    FALLBACK = "fallback"
    UNRESOLVED = "unresolved"


class ReachabilityStatus(str, Enum):
    REACHABLE = "reachable"
    NO_PATH = "no_path"
    ORIGIN_NOT_SNAPPED = "origin_not_snapped"
    SERVICE_NOT_SNAPPED = "service_not_snapped"
    SERVICE_UNAVAILABLE = "service_unavailable"
    SERVICE_UNRESOLVED = "service_unresolved"
    DATA_MISSING = "data_missing"


@dataclass(frozen=True, slots=True)
class SchemaContract:
    name: str
    required: frozenset[str]
    optional: frozenset[str] = frozenset()

    def validate_columns(self, columns: Iterable[str]) -> None:
        actual = set(columns)
        missing = sorted(self.required - actual)
        if missing:
            raise ValueError(f"{self.name}: missing required columns: {missing}")


ORIGIN_V2 = SchemaContract(
    name="OriginV2",
    required=frozenset(
        {
            "origin_id",
            "municipality_code",
            "geometry",
            "population_total",
            "population_reference_date",
            "census_year",
            "geography_version",
            "origin_method",
        }
    ),
    optional=frozenset(
        {
            # Five-year section-level age bands used as school target proxies.
            "population_age_lt5_proxy",
            "population_age_5_9_proxy",
            "population_age_10_14_proxy",
            "population_age_15_19_proxy",
            "section_type_code",
            "locality_type",
        }
    ),
)


SERVICE_V2 = SchemaContract(
    name="ServiceV2",
    required=frozenset(
        {
            "service_id",
            "service_type",
            "name",
            "municipality_code",
            "source_name",
            "source_record_id",
            "source_reference_date",
            "retrieved_at",
            "operational_status",
        }
    ),
    optional=frozenset(
        {
            "service_subtype",
            "raw_address",
            "normalized_address",
            "latitude",
            "longitude",
            "crs",
            "coordinate_source",
            "geocode_method",
            "geocode_quality",
            "status_date",
            "capacity_value",
            "capacity_unit",
            "source_release_id_or_version",
            "source_checksum",
            "source_reference_period",
            "legacy_service_site_id",
            "legacy_category",
            "legacy_subcategory",
            "legacy_usable_for_accessibility",
            "migration_status",
            "source_record_ids_json",
            "source_grade_descriptions_json",
            "school_ownerships_json",
            "classification_method",
        }
    ),
)


NETWORK_ATTACHMENT_V2 = SchemaContract(
    name="NetworkAttachmentV2",
    required=frozenset(
        {
            "entity_id",
            "entity_kind",  # origin | service
            "mode",
            "node_id",
            "snapped",
            "snap_distance_m",
            "attachment_quality",
        }
    ),
    optional=frozenset(
        {
            "network_component_id",
            "is_largest_component",
            "graph_checksum",
        }
    ),
)


ACCESSIBILITY_ORIGIN_V2 = SchemaContract(
    name="AccessibilityOriginV2",
    required=frozenset(
        {
            "origin_id",
            "service_type",
            "mode",
            "population_selector",
            "target_population",
            "nearest_service_time_min",
            "reachability_status",
        }
    ),
    optional=frozenset(
        {
            "nearest_service_id",
            "nearest_service_distance_m",
            # Threshold-specific columns are added dynamically using helpers below.
        }
    ),
)


def opportunity_count_column(threshold_min: int) -> str:
    _validate_threshold(threshold_min)
    return f"opportunity_count_within_{int(threshold_min)}_min"


def has_service_column(threshold_min: int) -> str:
    _validate_threshold(threshold_min)
    return f"has_service_within_{int(threshold_min)}_min"


def population_coverage_column(threshold_min: int) -> str:
    _validate_threshold(threshold_min)
    return f"population_coverage_within_{int(threshold_min)}_min"


def population_column_for_selector(selector: PopulationSelector | str) -> str:
    return PopulationSelector(selector).value


def validate_service_type(value: str) -> ServiceType:
    return ServiceType(value)


def validate_transport_mode(value: str) -> TransportMode:
    return TransportMode(value)


def validate_reachability(
    *,
    nearest_service_time_min: float | None,
    status: ReachabilityStatus | str,
) -> None:
    status = ReachabilityStatus(status)
    if status == ReachabilityStatus.REACHABLE:
        if nearest_service_time_min is None or nearest_service_time_min < 0:
            raise ValueError("Reachable rows need a non-negative nearest-service time.")
    else:
        # At data level we prefer null + explicit reason over serialising +inf to Parquet/JSON.
        if nearest_service_time_min is not None:
            raise ValueError(
                "Unreachable/non-computable rows must store nearest_service_time_min=null "
                "and an explicit reachability_status."
            )


def _validate_threshold(value: int) -> None:
    if int(value) <= 0:
        raise ValueError("threshold_min must be positive.")
