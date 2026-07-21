"""Etapa 4 — Feature Engineering.

Cria features derivadas de risco e monta o pré-processador (encoding + scaling)
como um ColumnTransformer do scikit-learn, reaproveitado no treino e na API
(garante que treino e produção transformam os dados do mesmo jeito).

## Por que as features derivadas existem

As colunas cruas do dataset são valores absolutos (`bill_amt*`, `pay_amt*`) e
códigos de status (`pay_*`). Nenhum dos dois é comparável entre clientes: uma
fatura de NT$ 50.000 significa coisas opostas para quem tem limite de 60.000 e
para quem tem 500.000. As derivadas normalizam isso e traduzem os 6 meses de
histórico em comportamento:

- **Utilização do limite** (`utilization_*`) — o driver clássico de risco em
  cartão: quanto do limite disponível já está comprometido.
- **Taxa de pagamento** (`payment_ratio_*`) — quanto o cliente efetivamente
  pagou da fatura do mês anterior. Distingue quem quita de quem rola a dívida.
- **Contagens de status** (`meses_em_atraso`, `meses_rotativo`,
  `meses_sem_consumo`) — resolvem o fato de os códigos de `pay_*` **não serem
  ordinais**. Como -2 (sem consumo), -1 (em dia) e 0 (rotativo) têm taxas de
  inadimplência que não crescem em ordem, usar o número cru como escala seria
  errado; contar ocorrências de cada regime preserva o sinal sem impor ordem.
- **Tendência da dívida** (`bill_trend`) — a fatura cresceu ou encolheu ao longo
  dos 6 meses, relativa ao limite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

PAY_COLS = [f"pay_{i}" for i in range(1, 7)]
BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]

# sex/education/marriage são códigos inteiros sem ordem — entram como categóricas.
CATEGORICAL_FEATURES = ["sex", "education", "marriage"]

NUMERIC_FEATURES = [
    "limit_bal", "age",
    *PAY_COLS, *BILL_COLS, *PAY_AMT_COLS,
    # derivadas:
    "limit_log",
    "utilization_last", "utilization_mean", "utilization_max",
    "payment_ratio_mean", "payment_ratio_last",
    "meses_em_atraso", "meses_rotativo", "meses_sem_consumo",
    "atraso_maximo", "teve_atraso",
    "bill_trend",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona as features derivadas descritas no topo do módulo."""
    df = df.copy()

    # O limite é divisor de quase tudo — protegido contra zero.
    limite = df["limit_bal"].clip(lower=1)
    df["limit_log"] = np.log1p(df["limit_bal"].clip(lower=0))

    faturas = df[BILL_COLS]
    df["utilization_last"] = df["bill_amt1"] / limite
    df["utilization_mean"] = faturas.mean(axis=1) / limite
    df["utilization_max"] = faturas.max(axis=1) / limite

    # Quanto pagou da fatura do mês anterior: pay_amt_i quita bill_amt_{i+1}.
    # Fatura <= 0 (saldo a favor) não tem o que pagar -> razão neutra de 1.
    razoes = []
    for i in range(1, 6):
        fatura_anterior = df[f"bill_amt{i + 1}"]
        razao = np.where(
            fatura_anterior > 0,
            df[f"pay_amt{i}"] / fatura_anterior.clip(lower=1),
            1.0,
        )
        razoes.append(np.clip(razao, 0, 2))  # corta quitações atípicas
    razoes = np.column_stack(razoes)
    df["payment_ratio_mean"] = razoes.mean(axis=1)
    df["payment_ratio_last"] = razoes[:, 0]

    # Contagens por regime de pagamento (ver docstring: pay_* não é ordinal).
    status = df[PAY_COLS]
    df["meses_em_atraso"] = (status >= 1).sum(axis=1)
    df["meses_rotativo"] = (status == 0).sum(axis=1)
    df["meses_sem_consumo"] = (status == -2).sum(axis=1)
    df["atraso_maximo"] = status.max(axis=1)
    df["teve_atraso"] = (df["atraso_maximo"] >= 1).astype(int)

    # Dívida crescendo (positivo) ou encolhendo (negativo) nos 6 meses.
    df["bill_trend"] = (df["bill_amt1"] - df["bill_amt6"]) / limite

    return df


def build_preprocessor() -> ColumnTransformer:
    """Monta o ColumnTransformer: OneHot nas categóricas, scaling nas numéricas."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica as features derivadas e separa X (features) e y (alvo)."""
    df = add_engineered_features(df)
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[config.TARGET].astype(int)
    return x, y
