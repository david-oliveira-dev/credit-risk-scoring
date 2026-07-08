"""Testes de feature engineering (Etapa 4)."""
from src.data.generate_synthetic import generate_credit
from src.features.build_features import (
    NUMERIC_FEATURES,
    add_engineered_features,
    build_preprocessor,
    make_xy,
)


def test_engineered_features_created():
    df = add_engineered_features(generate_credit(n=200, seed=1))
    for col in ("income_per_credit_line", "installment_pressure", "has_delinquency"):
        assert col in df.columns
    assert set(df["has_delinquency"].unique()) <= {0, 1}


def test_make_xy_shapes():
    x, y = make_xy(generate_credit(n=300, seed=2))
    assert len(x) == len(y) == 300
    assert "default" not in x.columns


def test_preprocessor_fit_transform():
    x, _ = make_xy(generate_credit(n=300, seed=3))
    pre = build_preprocessor()
    xt = pre.fit_transform(x)
    # linhas preservadas; colunas expandem pelo OneHot
    assert xt.shape[0] == 300
    assert xt.shape[1] >= len(NUMERIC_FEATURES)
