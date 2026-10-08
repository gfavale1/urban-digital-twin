from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

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


def normalize_city_name(value: str) -> str:
    """Normalize spelling, accents and punctuation; never fuzzy-match names."""
    if not isinstance(value, str):
        raise ValueError("Il nome del comune deve essere una stringa.")

    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", without_accents.casefold())
    result = " ".join(normalized.split())

    if not result:
        raise ValueError("Il nome del comune non può essere vuoto.")
    return result


def _optional_administrative_code(
    value: str | None,
    *,
    width: int,
    label: str,
) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().zfill(width)
    if len(normalized) != width or not normalized.isdigit():
        raise ValueError(
            f"{label} deve essere un codice ISTAT numerico a {width} cifre: "
            f"{value!r}"
        )
    return normalized


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


    @classmethod
    def resolve_name(
        cls,
        city_name: str,
        census_year: str = "2023",
        *,
        province_code: str | None = None,
        region_code: str | None = None,
    ) -> "MunicipalityContext":
        """Resolve one city by its official ISTAT name, without PostGIS.

        Matching ignores case, accents and punctuation, but it is NOT fuzzy.
        Multiple matching municipalities must be disambiguated explicitly.
        A versioned national dimension must exist before a name-based run.
        """
        city_key = normalize_city_name(city_name)
        year = str(census_year).strip()
        province = _optional_administrative_code(
            province_code, width=3, label="province_code"
        )
        region = _optional_administrative_code(
            region_code, width=2, label="region_code"
        )

        path = municipality_dimension_path(year)
        if not path.exists():
            raise FileNotFoundError(
                "Dimensione nazionale ISTAT necessaria per --city non trovata: "
                f"{path}. Prepararla con: "
                "python src/ingestion/build_municipality_dimension.py "
                f"--census-year {year}. "
                "Il bootstrap automatico delle fonti è previsto in B5B."
            )

        columns = [
            "istat_code", "name", "province_code", "region_code",
        ]
        dimension = pd.read_parquet(path, columns=columns).copy()
        if dimension.empty:
            raise RuntimeError(f"Dimensione ISTAT vuota: {path}")

        dimension["istat_code"] = (
            dimension["istat_code"]
            .astype("string")
            .str.strip()
            .str.zfill(6)
        )
        dimension["province_code"] = (
            dimension["province_code"]
            .astype("string")
            .str.strip()
            .str.zfill(3)
        )
        dimension["region_code"] = (
            dimension["region_code"]
            .astype("string")
            .str.strip()
            .str.zfill(2)
        )

        names = (
            dimension["name"]
            .astype("string")
            .fillna("")
            .map(lambda name: normalize_city_name(name) if name.strip() else "")
        )
        matched = dimension.loc[names == city_key].copy()
        if province is not None:
            matched = matched.loc[matched["province_code"] == province]
        if region is not None:
            matched = matched.loc[matched["region_code"] == region]

        if matched.empty:
            qualifiers = []
            if province is not None:
                qualifiers.append(f"provincia ISTAT {province}")
            if region is not None:
                qualifiers.append(f"regione ISTAT {region}")
            suffix = f" ({', '.join(qualifiers)})" if qualifiers else ""
            raise ValueError(
                f"Comune {city_name!r}{suffix} non trovato nella dimensione "
                f"ISTAT {year}. Nessuna corrispondenza approssimata applicata."
            )

        if matched["istat_code"].duplicated().any():
            raise RuntimeError(
                "Dimensione ISTAT incoerente: codice comunale duplicato "
                f"tra i match per {city_name!r}."
            )

        if len(matched) != 1:
            choices = "\n".join(
                f"  - {row.name}: {row.istat_code} "
                f"(provincia {row.province_code}, regione {row.region_code})"
                for row in matched.sort_values("istat_code").itertuples(index=False)
            )
            raise ValueError(
                f"Nome comunale ambiguo: {city_name!r} ({year}).\n"
                f"Candidati:\n{choices}\n"
                "Usare --province-code NNN, --region-code NN oppure "
                "--municipality-code NNNNNN."
            )

        row = matched.iloc[0]
        return cls._from_values(
            code=str(row["istat_code"]),
            name=row["name"],
            province_code=row["province_code"],
            region_code=row["region_code"],
            census_year=year,
        )
