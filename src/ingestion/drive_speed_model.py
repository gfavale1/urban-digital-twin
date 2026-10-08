from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_MPH_TO_KPH = 1.609344
_NUMERIC_TOKEN = re.compile(r"(?<![A-Za-z])([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True, slots=True)
class DriveSpeedSummary:
    edge_count: int
    parsed_osm_maxspeed: int
    inferred_highway_median: int
    inferred_global_median: int
    explicit_fallback: int
    min_speed_kph: float
    median_speed_kph: float
    max_speed_kph: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "edge_count": self.edge_count,
            "parsed_osm_maxspeed": self.parsed_osm_maxspeed,
            "inferred_highway_median": self.inferred_highway_median,
            "inferred_global_median": self.inferred_global_median,
            "explicit_fallback": self.explicit_fallback,
            "min_speed_kph": self.min_speed_kph,
            "median_speed_kph": self.median_speed_kph,
            "max_speed_kph": self.max_speed_kph,
        }


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _flatten_tokens(value: Any) -> list[str]:
    if _is_missing(value):
        return []

    if isinstance(value, (list, tuple, set)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_flatten_tokens(item))
        return tokens

    text = str(value).strip()
    if not text:
        return []

    # OSM values may be semicolon-separated and GraphML round-trips may
    # preserve a list-like representation. Splitting on semicolons is the
    # conservative deterministic case we need here.
    return [part.strip() for part in text.split(";") if part.strip()]


def parse_maxspeed_kph(value: Any) -> float | None:
    """Parse an OSM ``maxspeed`` value into km/h.

    Numeric values are interpreted as km/h unless the token explicitly
    contains ``mph``. When multiple parseable limits are present (for example
    a semicolon-separated value), the minimum is used conservatively.

    Non-numeric OSM conventions such as ``none`` or country-specific implicit
    codes are deliberately left unresolved and handled by the documented
    imputation hierarchy instead of being silently guessed here.
    """

    parsed: list[float] = []

    for token in _flatten_tokens(value):
        lower = token.lower()
        match = _NUMERIC_TOKEN.search(lower)
        if match is None:
            continue

        speed = float(match.group(1))
        if "mph" in lower:
            speed *= _MPH_TO_KPH

        if math.isfinite(speed) and speed > 0:
            parsed.append(speed)

    if not parsed:
        return None

    return float(min(parsed))


def normalize_highway_class(value: Any) -> str | None:
    """Return a deterministic primary highway class for imputation groups."""

    tokens = _flatten_tokens(value)
    if not tokens:
        return None

    token = tokens[0].strip().lower()
    return token or None


def assign_free_flow_travel_times(
    edges: pd.DataFrame,
    *,
    fallback_speed_kph: float | None = None,
) -> tuple[pd.DataFrame, DriveSpeedSummary]:
    """Assign traceable free-flow speeds and travel times to drive edges.

    Hierarchy, in order:
    1. parseable OSM ``maxspeed`` on the edge;
    2. median observed OSM maxspeed for the same highway class;
    3. median observed OSM maxspeed over the whole graph;
    4. an explicit caller-provided fallback, only if the graph contains no
       usable OSM maxspeed observations at all.

    This deliberately avoids embedding an undocumented road-class speed table.
    The caller can later freeze an explicit fallback profile in AnalysisSpec if
    empirical validation shows it is needed nationally.
    """

    required = {"length_m"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(
            "Drive edge table missing required columns: "
            + ", ".join(sorted(missing))
        )

    if fallback_speed_kph is not None:
        fallback_speed_kph = float(fallback_speed_kph)
        if not math.isfinite(fallback_speed_kph) or fallback_speed_kph <= 0:
            raise ValueError("fallback_speed_kph must be finite and > 0.")

    result = edges.copy()

    length_m = pd.to_numeric(result["length_m"], errors="coerce")
    if length_m.isna().any() or (~np.isfinite(length_m.to_numpy(dtype=float))).any():
        raise ValueError("length_m must be finite for every drive edge.")
    if (length_m <= 0).any():
        raise ValueError("length_m must be > 0 for every drive edge.")
    result["length_m"] = length_m.astype(float)

    if "road_type" in result.columns:
        highway_source = result["road_type"]
    elif "highway" in result.columns:
        highway_source = result["highway"]
    else:
        highway_source = pd.Series(None, index=result.index, dtype=object)

    result["speed_highway_class"] = highway_source.map(normalize_highway_class)

    if "maxspeed" in result.columns:
        maxspeed_raw = result["maxspeed"]
    elif "maxspeed_raw" in result.columns:
        maxspeed_raw = result["maxspeed_raw"]
    else:
        maxspeed_raw = pd.Series(None, index=result.index, dtype=object)

    result["maxspeed_raw"] = maxspeed_raw.map(
        lambda value: None if _is_missing(value) else str(value)
    )
    parsed = maxspeed_raw.map(parse_maxspeed_kph).astype(float)

    result["speed_kph"] = parsed
    result["speed_source"] = np.where(
        parsed.notna(),
        "osm_maxspeed",
        None,
    )

    observed_mask = parsed.notna()

    if observed_mask.any():
        observed = result.loc[
            observed_mask & result["speed_highway_class"].notna(),
            ["speed_highway_class", "speed_kph"],
        ]

        highway_medians = (
            observed.groupby("speed_highway_class", dropna=True)["speed_kph"]
            .median()
            .to_dict()
        )

        missing_speed = result["speed_kph"].isna()
        same_class_speed = result["speed_highway_class"].map(highway_medians)
        same_class_mask = missing_speed & same_class_speed.notna()
        result.loc[same_class_mask, "speed_kph"] = same_class_speed.loc[
            same_class_mask
        ].astype(float)
        result.loc[same_class_mask, "speed_source"] = "highway_median_osm_maxspeed"

        global_median = float(parsed.loc[observed_mask].median())
        global_mask = result["speed_kph"].isna()
        result.loc[global_mask, "speed_kph"] = global_median
        result.loc[global_mask, "speed_source"] = "global_median_osm_maxspeed"

    else:
        if fallback_speed_kph is None:
            raise RuntimeError(
                "No parseable OSM maxspeed values are available in the drive graph. "
                "Provide an explicit fallback_speed_kph rather than silently guessing."
            )

        result["speed_kph"] = float(fallback_speed_kph)
        result["speed_source"] = "explicit_fallback"

    speeds = pd.to_numeric(result["speed_kph"], errors="coerce")
    if speeds.isna().any() or (~np.isfinite(speeds.to_numpy(dtype=float))).any():
        raise RuntimeError("Drive speed assignment produced missing/non-finite values.")
    if (speeds <= 0).any():
        raise RuntimeError("Drive speed assignment produced non-positive values.")

    result["speed_kph"] = speeds.astype(float)
    result["free_flow_travel_time_s"] = (
        result["length_m"] / (result["speed_kph"] * 1000.0 / 3600.0)
    )

    travel = result["free_flow_travel_time_s"].to_numpy(dtype=float)
    if (~np.isfinite(travel)).any() or (travel < 0).any():
        raise RuntimeError("Invalid free_flow_travel_time_s values were produced.")

    source_counts = result["speed_source"].value_counts().to_dict()
    summary = DriveSpeedSummary(
        edge_count=int(len(result)),
        parsed_osm_maxspeed=int(source_counts.get("osm_maxspeed", 0)),
        inferred_highway_median=int(
            source_counts.get("highway_median_osm_maxspeed", 0)
        ),
        inferred_global_median=int(
            source_counts.get("global_median_osm_maxspeed", 0)
        ),
        explicit_fallback=int(source_counts.get("explicit_fallback", 0)),
        min_speed_kph=float(result["speed_kph"].min()),
        median_speed_kph=float(result["speed_kph"].median()),
        max_speed_kph=float(result["speed_kph"].max()),
    )

    return result, summary
