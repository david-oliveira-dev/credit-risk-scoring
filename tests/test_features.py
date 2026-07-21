"""Testes de feature engineering (Etapa 4)."""
import numpy as np
import pandas as pd

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_engineered_features,
    build_preprocessor,
    make_xy,
)


def test_cria_todas_as_features_derivadas(amostra):
    df = add_engineered_features(amostra)
    derivadas = [
        "limit_log", "utilization_last", "utilization_mean", "utilization_max",
        "payment_ratio_mean", "payment_ratio_last", "meses_em_atraso",
        "meses_rotativo", "meses_sem_consumo", "atraso_maximo", "teve_atraso",
        "bill_trend",
    ]
    for col in derivadas:
        assert col in df.columns, f"faltou {col}"


def test_nao_altera_o_dataframe_original(amostra):
    colunas_antes = list(amostra.columns)
    add_engineered_features(amostra)
    assert list(amostra.columns) == colunas_antes


def test_utilizacao_e_fatura_sobre_limite():
    df = pd.DataFrame([{
        "limit_bal": 100_000.0, "sex": 1, "education": 2, "marriage": 1, "age": 30,
        **{f"pay_{i}": 0 for i in range(1, 7)},
        **{f"bill_amt{i}": 25_000.0 for i in range(1, 7)},
        **{f"pay_amt{i}": 1_000.0 for i in range(1, 7)},
    }])
    out = add_engineered_features(df)
    assert out["utilization_last"].iloc[0] == 0.25
    assert out["utilization_mean"].iloc[0] == 0.25


def test_limite_zero_nao_estoura_divisao():
    """Guarda contra divisão por zero — a API pode receber limite 0."""
    df = pd.DataFrame([{
        "limit_bal": 0.0, "sex": 1, "education": 2, "marriage": 1, "age": 30,
        **{f"pay_{i}": 0 for i in range(1, 7)},
        **{f"bill_amt{i}": 100.0 for i in range(1, 7)},
        **{f"pay_amt{i}": 10.0 for i in range(1, 7)},
    }])
    out = add_engineered_features(df)
    assert np.isfinite(out["utilization_last"].iloc[0])
    assert np.isfinite(out["bill_trend"].iloc[0])


def test_contagens_de_status_de_pagamento():
    """As contagens tratam -2/-1/0 como regimes, não como escala ordinal."""
    linha = {
        "limit_bal": 50_000.0, "sex": 1, "education": 2, "marriage": 1, "age": 40,
        "pay_1": 2, "pay_2": 1, "pay_3": 0, "pay_4": 0, "pay_5": -1, "pay_6": -2,
        **{f"bill_amt{i}": 10_000.0 for i in range(1, 7)},
        **{f"pay_amt{i}": 500.0 for i in range(1, 7)},
    }
    out = add_engineered_features(pd.DataFrame([linha])).iloc[0]

    assert out["meses_em_atraso"] == 2     # pay_1=2 e pay_2=1
    assert out["meses_rotativo"] == 2      # pay_3 e pay_4
    assert out["meses_sem_consumo"] == 1   # pay_6
    assert out["atraso_maximo"] == 2
    assert out["teve_atraso"] == 1


def test_cliente_sem_atraso_nenhum():
    linha = {
        "limit_bal": 50_000.0, "sex": 2, "education": 1, "marriage": 2, "age": 33,
        **{f"pay_{i}": -1 for i in range(1, 7)},
        **{f"bill_amt{i}": 1_000.0 for i in range(1, 7)},
        **{f"pay_amt{i}": 1_000.0 for i in range(1, 7)},
    }
    out = add_engineered_features(pd.DataFrame([linha])).iloc[0]
    assert out["meses_em_atraso"] == 0
    assert out["teve_atraso"] == 0


def test_payment_ratio_fica_no_intervalo(amostra):
    out = add_engineered_features(amostra)
    assert out["payment_ratio_mean"].between(0, 2).all()


def test_bill_trend_positivo_quando_divida_cresce():
    linha = {
        "limit_bal": 100_000.0, "sex": 1, "education": 2, "marriage": 1, "age": 45,
        **{f"pay_{i}": 0 for i in range(1, 7)},
        "bill_amt1": 60_000.0, "bill_amt2": 50_000.0, "bill_amt3": 40_000.0,
        "bill_amt4": 30_000.0, "bill_amt5": 20_000.0, "bill_amt6": 10_000.0,
        **{f"pay_amt{i}": 1_000.0 for i in range(1, 7)},
    }
    out = add_engineered_features(pd.DataFrame([linha])).iloc[0]
    assert out["bill_trend"] > 0


def test_make_xy_separa_alvo(amostra):
    x, y = make_xy(amostra)
    assert len(x) == len(y) == len(amostra)
    assert "default" not in x.columns
    assert list(x.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_preprocessor_expande_categoricas(amostra):
    x, _ = make_xy(amostra)
    xt = build_preprocessor().fit_transform(x)
    assert xt.shape[0] == len(amostra)
    assert xt.shape[1] > len(NUMERIC_FEATURES)  # one-hot adiciona colunas


def test_preprocessor_tolera_categoria_nova(amostra):
    """`handle_unknown='ignore'`: uma categoria inédita não pode quebrar a API."""
    x, _ = make_xy(amostra)
    pre = build_preprocessor().fit(x)

    novo = x.head(1).copy()
    novo["education"] = 99  # código que nunca apareceu no treino
    assert pre.transform(novo).shape[0] == 1
