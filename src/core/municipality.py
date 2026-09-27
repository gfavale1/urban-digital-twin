from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from .database import get_engine
from .paths import ROOT, normalize_municipality_code


def municipality_dimension_path(
    census_year: str,
) -> Path:
    year = str(census_year).strip()

    return (
        ROOT
        / "data"
        / "processed"
        / "istat"
        / f"municipality_dimension_{year}.parquet"
    )


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
    def _from_values(
        cls,
        *,
        code: str,
        name,
        province_code,
        region_code,
        census_year: str,
    ) -> "MunicipalityContext":
        name = str(name).strip()

        province_code = (
            str(province_code)
            .strip()
            .zfill(3)
        )

        region_code = (
            str(region_code)
            .strip()
            .zfill(2)
        )

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

    @classmethod
    def _resolve_from_dimension(
        cls,
        *,
        code: str,
        census_year: str,
    ) -> "MunicipalityContext | None":
        path = municipality_dimension_path(
            census_year
        )

        if not path.exists():
            return None

        dimension = pd.read_parquet(
            path,
            columns=[
                "istat_code",
                "name",
                "province_code",
                "region_code",
            ],
        )

        dimension = dimension.copy()

        dimension["istat_code"] = (
            dimension["istat_code"]
            .astype("string")
            .str.strip()
            .str.zfill(6)
        )

        rows = dimension.loc[
            dimension["istat_code"] == code
        ]

        if rows.empty:
            raise RuntimeError(
                "\nComune non presente nella dimensione "
                f"amministrativa ISTAT {census_year}.\n"
                f"Codice ISTAT: {code}\n"
                f"Dataset: {path}"
            )

        if len(rows) != 1:
            raise RuntimeError(
                "Il codice ISTAT comunale non è "
                "univoco nella dimensione nazionale: "
                f"{code}"
            )

        row = rows.iloc[0]

        return cls._from_values(
            code=code,
            name=row["name"],
            province_code=row["province_code"],
            region_code=row["region_code"],
            census_year=census_year,
        )

    @classmethod
    def _resolve_from_database(
        cls,
        *,
        code: str,
        census_year: str,
    ) -> "MunicipalityContext | None":
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
            return None

        if len(rows) != 1:
            raise RuntimeError(
                "Il codice ISTAT comunale non è "
                "univoco nella tabella municipality: "
                f"{code}"
            )

        row = rows[0]

        return cls._from_values(
            code=code,
            name=row["name"],
            province_code=row["province_code"],
            region_code=row["region_code"],
            census_year=census_year,
        )

    @classmethod
    def resolve(
        cls,
        municipality_code: str,
        census_year: str = "2023",
    ) -> "MunicipalityContext":
        code = normalize_municipality_code(
            municipality_code
        )

        year = str(census_year).strip()

        # Primary source for zero-touch execution:
        # the versioned national ISTAT administrative dimension.
        context = cls._resolve_from_dimension(
            code=code,
            census_year=year,
        )

        if context is not None:
            return context

        # Backward-compatible fallback for repositories that have not
        # built the national dimension yet.
        context = cls._resolve_from_database(
            code=code,
            census_year=year,
        )

        if context is not None:
            return context

        path = municipality_dimension_path(
            year
        )

        raise RuntimeError(
            "\nComune non risolvibile automaticamente.\n"
            f"Codice ISTAT: {code}\n"
            f"Dimensione attesa: {path}\n\n"
            "Costruire prima la dimensione nazionale con:\n"
            "python src/ingestion/build_municipality_dimension.py "
            f"--census-year {year}"
        )
