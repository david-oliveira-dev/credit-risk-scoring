"""Etapa 2 — ETL: baixa o dataset real, limpa/valida e carrega no banco + parquet.

Fluxo:
    extract  -> data/raw/uci_credit_default.csv (baixa do UCI se não existir)
    transform-> resolve os problemas de qualidade do dado real (abaixo)
    load     -> grava na tabela `credit` (PostgreSQL ou SQLite) e em parquet

## Os problemas reais desta base

Diferente de um dado gerado, este veio com inconsistências entre o arquivo e a
documentação oficial. Cada uma é tratada explicitamente aqui:

1. **`education` tem os códigos 0, 5 e 6** (345 registros), mas a documentação
   define apenas 1=pós, 2=universitário, 3=médio, 4=outros. Os três são
   colapsados em 4 ("outros") — tratamento padrão na literatura sobre esta base,
   e evita categorias com dezenas de exemplos que só virariam ruído no one-hot.
2. **`marriage` tem o código 0** (54 registros), fora dos documentados
   1=casado, 2=solteiro, 3=outros. Vai para 3.
3. **As colunas `pay_*` têm os códigos -2 e 0**, também não documentados (a doc
   descreve -1 = pago em dia e 1..9 = meses de atraso). O consenso é que -2 =
   sem consumo no mês e 0 = crédito rotativo (pagou o mínimo). **Eles não são
   removidos nem recodificados**: a taxa de inadimplência por código mostra que
   carregam sinal real e *não monótono* (0 → 12,8%, -1 → 16,8%), então tratar a
   coluna como um inteiro ordinal seria errado. Isso é resolvido na feature
   engineering, não aqui.
4. **35 linhas são duplicatas exatas** (desconsiderando o `ID`) e são removidas.
5. **`bill_amt*` pode ser negativo** (590 casos na primeira fatura): é saldo a
   favor do cliente, não erro — fica como está.

Uso:
    python -m src.data.etl
"""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine

from src import config
from src.data.download import obter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("etl")

PAY_COLS = [f"pay_{i}" for i in range(1, 7)]
BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]

# Códigos válidos segundo a documentação do dataset.
EDUCATION_VALIDOS = {1, 2, 3, 4}
MARRIAGE_VALIDOS = {1, 2, 3}
EDUCATION_OUTROS = 4
MARRIAGE_OUTROS = 3

NUMERIC_BOUNDS = {
    "age": (18, 100),
    "limit_bal": (0, 2_000_000),
}


def extract() -> pd.DataFrame:
    """Lê o CSV bruto; se não existir, baixa do UCI."""
    df = obter()
    logger.info("Extraídas %d linhas", len(df))
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa as inconsistências documentadas no topo do módulo."""
    df = df.copy()
    n0 = len(df)

    duplicadas = int(df.duplicated().sum())
    if duplicadas:
        df = df.drop_duplicates()
        logger.info("Removidas %d linhas duplicadas", duplicadas)

    # Categorias fora da documentação -> "outros".
    fora_edu = int((~df["education"].isin(EDUCATION_VALIDOS)).sum())
    df.loc[~df["education"].isin(EDUCATION_VALIDOS), "education"] = EDUCATION_OUTROS

    fora_civil = int((~df["marriage"].isin(MARRIAGE_VALIDOS)).sum())
    df.loc[~df["marriage"].isin(MARRIAGE_VALIDOS), "marriage"] = MARRIAGE_OUTROS

    if fora_edu or fora_civil:
        logger.info("Recodificados p/ 'outros': %d em education, %d em marriage",
                    fora_edu, fora_civil)

    # Faixas implausíveis (defensivo — a base real não tem, mas a API pode receber).
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        df = df[df[col].between(lo, hi)]

    df = df.dropna()
    df[config.TARGET] = df[config.TARGET].astype(int)

    removidas = n0 - len(df)
    if removidas:
        logger.info("Total removido na validação: %d linhas", removidas)
    logger.info("Transform ok: %d linhas válidas (%.2f%% de default)",
                len(df), 100 * df[config.TARGET].mean())
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
