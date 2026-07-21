"""Fixtures compartilhadas.

Os testes **não baixam o dataset do UCI**: seria lento, dependeria de rede no CI
e tornaria a suíte não determinística. Em vez disso, montam um DataFrame com o
mesmo esquema de colunas e com sinal plantado, o suficiente para exercitar ETL,
features, treino e API.

Quem valida o dataset real de verdade é o `etl.transform` rodando em produção
(`python -m src.data.etl`) e as regras de limpeza testadas aqui com os mesmos
códigos inválidos que a base real contém.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.download import COLUNAS

# Colunas finais esperadas depois do download (sem o ID, já renomeadas).
COLUNAS_UCI = [c for c in COLUNAS.values()]

PAY_COLS = [f"pay_{i}" for i in range(1, 7)]
BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]


def montar_amostra(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """DataFrame no esquema do UCI, com relação real entre atraso e default.

    O alvo vem de um modelo logístico sobre atraso e utilização do limite, então
    os testes de treino conseguem verificar que o pipeline aprende algo — sem
    isso, um teste de ROC-AUC seria só ruído.
    """
    rng = np.random.default_rng(seed)

    limit_bal = rng.choice([20_000, 50_000, 100_000, 200_000, 500_000], size=n)
    atraso_base = rng.choice([-2, -1, 0, 1, 2], size=n, p=[0.1, 0.2, 0.5, 0.13, 0.07])

    dados = {
        "limit_bal": limit_bal.astype(float),
        "sex": rng.choice([1, 2], size=n),
        "education": rng.choice([1, 2, 3, 4], size=n),
        "marriage": rng.choice([1, 2, 3], size=n),
        "age": rng.integers(21, 70, size=n),
    }
    for i in range(1, 7):
        dados[f"pay_{i}"] = np.clip(atraso_base + rng.integers(-1, 2, size=n), -2, 8)
        dados[f"bill_amt{i}"] = (limit_bal * rng.uniform(0, 0.95, size=n)).round(0)
        dados[f"pay_amt{i}"] = (limit_bal * rng.uniform(0, 0.1, size=n)).round(0)

    df = pd.DataFrame(dados)
    utilizacao = df["bill_amt1"] / df["limit_bal"]
    logit = -2.2 + 0.9 * np.maximum(atraso_base, 0) + 1.6 * utilizacao
    df["default"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


@pytest.fixture
def amostra() -> pd.DataFrame:
    """Amostra pequena e rápida para os testes de features/ETL."""
    return montar_amostra(n=500, seed=42)


@pytest.fixture
def amostra_grande() -> pd.DataFrame:
    """Amostra maior, para os testes que treinam um modelo de verdade."""
    return montar_amostra(n=3000, seed=7)
