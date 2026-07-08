"""Configuração central do projeto (paths e conexão de banco).

Tudo que depende de ambiente sai daqui. A URL do banco vem da env var
`DATABASE_URL`; se ausente, usa SQLite local — assim testes e CI rodam sem
precisar de um PostgreSQL de verdade.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RAW_CSV = DATA_RAW / "credit.csv"
PROCESSED_PARQUET = DATA_PROCESSED / "credit.parquet"

TABLE_NAME = "credit"
TARGET = "default"


def get_database_url() -> str:
    """Retorna a URL do banco. Default: SQLite em data/processed/credit.db."""
    default = f"sqlite:///{DATA_PROCESSED / 'credit.db'}"
    return os.environ.get("DATABASE_URL", default)
