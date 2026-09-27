from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configurazione temporale canonica della pipeline nazionale.

    I valori rappresentano lo snapshot multi-source validato
    su Matera e Parma. Devono essere sovrascrivibili dalla CLI,
    ma definiti in un solo punto del codice.
    """

    census_year: str = "2023"
    school_year: str = "202425"
    building_year: str = "202425"
    health_reference_date: str = "2025-06-30"
    hospital_year: str = "2023"
    osm_reference_period: str = "2026-09-27"

    walking_speed_m_s: float = 1.4

    accessibility_thresholds_min: tuple[int, ...] = (
        10,
        15,
        20,
    )


DEFAULT_CONFIG = PipelineConfig()
