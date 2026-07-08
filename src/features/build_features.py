"""Etapa 4 — Feature Engineering.

Cria features derivadas de risco e monta o pré-processador (encoding + scaling)
como um ColumnTransformer do scikit-learn, para ser reaproveitado no treino e na
API (garante que treino e produção transformam os dados do mesmo jeito).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

CATEGORICAL_FEATURES = ["home_ownership", "loan_intent"]

# Numéricas originais + as derivadas criadas em add_engineered_features().
NUMERIC_FEATURES = [
    "age", "income", "employment_length", "loan_amount", "interest_rate",
    "loan_percent_income", "debt_to_income", "credit_history_length",
    "num_credit_lines", "past_delinquencies",
    # derivadas:
    "income_per_credit_line", "installment_pressure", "has_delinquency",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas com significado de risco de crédito."""
    df = df.copy()
    df["income_per_credit_line"] = df["income"] / (df["num_credit_lines"] + 1)
    # "pressão da parcela": juros pesando sobre o comprometimento de renda
    df["installment_pressure"] = df["loan_percent_income"] * df["interest_rate"]
    df["has_delinquency"] = (df["past_delinquencies"] > 0).astype(int)
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
