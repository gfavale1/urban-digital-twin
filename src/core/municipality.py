from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

from .database import get_engine
from .paths import normalize_municipality_code


@dataclass(frozen=True)
class MunicipalityContext:
    code: str
    procom: int
    name: str
    name_upper: str
    province_code: str
    region_code: str
    census_year: str

    @classmethod
    def resolve(
        cls,
        municipality_code: str,
        census_year: str = "2023",
    ) -> "MunicipalityContext":
        code = normalize_municipality_code(
            municipality_code
        )

        engine = get_engine()

        query = text(
            """
            SELECT
                istat_code,
                name,
                province_code,
                region_code
            FROM municipality
            WHERE istat_code = :istat_code
            """
        )

        with engine.connect() as connection:
            rows = (
                connection.execute(
                    query,
                    {
                        "istat_code": code,
                    },
                )
                .mappings()
                .all()
            )

        if not rows:
            raise RuntimeError(
                "\nComune non presente nella dimensione "
                "municipality di PostGIS.\n"
                f"Codice ISTAT: {code}\n\n"
                "Per Matera e Parma il contesto è già "
                "disponibile. Prima del run nazionale "
                "verrà introdotto il bootstrap della "
                "dimensione amministrativa nazionale ISTAT."
            )

        if len(rows) != 1:
            raise RuntimeError(
                "Il codice ISTAT comunale non è "
                "univoco nella tabella municipality: "
                f"{code}"
            )

        row = rows[0]

        name = str(
            row["name"]
        ).strip()

        province_code = str(
            row["province_code"]
        ).strip().zfill(3)

        region_code = str(
            row["region_code"]
        ).strip().zfill(2)

        if not name:
            raise RuntimeError(
                f"Nome del comune mancante per {code}."
            )

        if (
            not province_code.isdigit()
            or len(province_code) != 3
        ):
            raise RuntimeError(
                "Codice provincia non valido "
                f"per il comune {code}: "
                f"{province_code}"
            )

        if (
            not region_code.isdigit()
            or len(region_code) != 2
        ):
            raise RuntimeError(
                "Codice regione non valido "
                f"per il comune {code}: "
                f"{region_code}"
            )

        return cls(
            code=code,
            procom=int(code),
            name=name,
            name_upper=name.upper(),
            province_code=province_code,
            region_code=region_code,
            census_year=str(
                census_year
            ).strip(),
        )
