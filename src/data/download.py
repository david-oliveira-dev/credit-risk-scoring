"""Etapa 1 — Download do dataset real (UCI).

Fonte: **Default of Credit Card Clients** (Yeh & Lien, 2009), UCI Machine
Learning Repository — 30.000 clientes de cartão de crédito de Taiwan, com o
histórico de pagamento de abril a setembro de 2005 e o desfecho de
inadimplência no mês seguinte.

    https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

O arquivo original é um `.xls` dentro de um zip, com a primeira linha ocupada
por um cabeçalho de grupo (por isso `header=1`). Aqui ele é baixado uma única
vez, convertido para CSV e cacheado em `data/raw/` — que não é versionado.

Uso:
    python -m src.data.download
"""
from __future__ import annotations

import io
import logging
import urllib.request
import zipfile

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("download")

# Renomeia para snake_case. `PAY_0` vira `pay_1`: a numeração original pula o 1,
# o que é uma pegadinha conhecida deste dataset.
COLUNAS = {
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_1",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "default payment next month": config.TARGET,
}
for i in range(1, 7):
    COLUNAS[f"BILL_AMT{i}"] = f"bill_amt{i}"
    COLUNAS[f"PAY_AMT{i}"] = f"pay_amt{i}"


def baixar(url: str = config.UCI_URL, timeout: int = 120) -> pd.DataFrame:
    """Baixa o zip do UCI e devolve o DataFrame bruto, já com nomes normalizados."""
    logger.info("Baixando dataset do UCI...")
    with urllib.request.urlopen(url, timeout=timeout) as resposta:  # noqa: S310
        conteudo = resposta.read()

    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        nome_xls = next(n for n in z.namelist() if n.lower().endswith(".xls"))
        with z.open(nome_xls) as f:
            # header=1: a linha 0 do arquivo é um cabeçalho de grupo, não os nomes.
            df = pd.read_excel(io.BytesIO(f.read()), header=1)

    df = df.drop(columns=["ID"], errors="ignore").rename(columns=COLUNAS)
    logger.info("Baixadas %d linhas x %d colunas", *df.shape)
    return df


def obter(forcar: bool = False) -> pd.DataFrame:
    """Devolve o dataset bruto, baixando só se o cache local não existir."""
    if config.RAW_CSV.exists() and not forcar:
        logger.info("Usando cache local: %s", config.RAW_CSV.name)
        return pd.read_csv(config.RAW_CSV)

    df = baixar()
    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_CSV, index=False)
    logger.info("Salvo em %s", config.RAW_CSV)
    return df


if __name__ == "__main__":
    obter(forcar=True)
