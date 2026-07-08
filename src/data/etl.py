"""Etapa 2 — ETL: extrai o CSV bruto, limpa/valida e carrega no banco + parquet.

Fluxo:
    extract  -> lê data/raw/credit.csv (gera se não existir)
    transform-> tipa colunas, valida faixas, remove inconsistências
    load     -> grava na tabela `credit` (PostgreSQL ou SQLite) e em parquet

Uso:
    python -m src.data.etl
"""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine

from src import config
from src.data.generate_synthetic import generate_credit

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("etl")

# Faixas aceitáveis para validação (defensivo).
NUMERIC_BOUNDS = {
    "age": (18, 100),
    "income": (0, 5_000_000),
    "loan_amount": (0, 1_000_000),
    "interest_rate": (0, 60),
    "loan_percent_income": (0, 5),
    "debt_to_income": (0, 5),
}
CATEGORICAL = {
    "home_ownership": {"RENT", "MORTGAGE", "OWN"},
    "loan_intent": {"PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                    "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"},
}


def extract() -> pd.DataFrame:
    """Lê o CSV bruto; se não existir, gera os dados sintéticos primeiro."""
    if not config.RAW_CSV.exists():
        logger.info("CSV bruto ausente — gerando dados sintéticos...")
        df = generate_credit()
        config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.RAW_CSV, index=False)
    df = pd.read_csv(config.RAW_CSV)
    logger.info("Extraídas %d linhas de %s", len(df), config.RAW_CSV.name)
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Tipa, valida faixas e remove linhas inconsistentes."""
    df = df.drop_duplicates(subset="customer_id").copy()

    # Descarta linhas fora das faixas plausíveis.
    n0 = len(df)
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col in df:
            df = df[df[col].between(lo, hi)]
    for col, allowed in CATEGORICAL.items():
        if col in df:
            df = df[df[col].isin(allowed)]

    df = df.dropna()
    df[config.TARGET] = df[config.TARGET].astype(int)
    removed = n0 - len(df)
    if removed:
        logger.info("Removidas %d linhas inconsistentes na validação", removed)
    logger.info("Transform ok: %d linhas válidas", len(df))
    return df.reset_index(drop=True)


def load(df: pd.DataFrame) -> None:
    """Grava no banco (tabela `credit`) e em parquet."""
    engine = create_engine(config.get_database_url())
    df.to_sql(config.TABLE_NAME, engine, if_exists="replace", index=False)
    config.PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED_PARQUET, index=False)
    logger.info("Carregado no banco (%s) e em %s",
                engine.url.get_backend_name(), config.PROCESSED_PARQUET.name)


def run_etl() -> pd.DataFrame:
    """Executa o pipeline completo e retorna o DataFrame processado."""
    df = transform(extract())
    load(df)
    return df


if __name__ == "__main__":
    run_etl()
