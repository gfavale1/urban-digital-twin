from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

from .paths import ROOT


load_dotenv(
    ROOT / ".env"
)


def database_url() -> str:
    url = os.getenv(
        "DATABASE_URL"
    )

    if not url:
        raise RuntimeError(
            "DATABASE_URL non definita. "
            "Controllare il file .env."
        )

    return url


def get_engine():
    return create_engine(
        database_url(),
        future=True,
    )
