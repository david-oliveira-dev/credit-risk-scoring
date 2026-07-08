"""Testes do gerador de dados sintéticos de crédito (Etapa 1)."""
from src.data.generate_synthetic import generate_credit

EXPECTED_COLUMNS = {
    "customer_id", "age", "income", "employment_length", "home_ownership",
    "loan_intent", "loan_amount", "interest_rate", "loan_percent_income",
    "debt_to_income", "credit_history_length", "num_credit_lines",
    "past_delinquencies", "default",
}


def test_shape_and_columns():
    df = generate_credit(n=600, seed=1)
    assert len(df) == 600
    assert set(df.columns) == EXPECTED_COLUMNS
    assert df["customer_id"].is_unique


def test_default_rate_is_imbalanced():
    # Precisa ser desbalanceado (minoritária) para justificar SMOTE.
    df = generate_credit(n=8000, seed=42)
    rate = df["default"].mean()
    assert 0.05 <= rate <= 0.25, f"taxa de default fora do esperado: {rate:.2%}"


def test_reproducible_with_seed():
    a = generate_credit(n=400, seed=7)
    b = generate_credit(n=400, seed=7)
    assert a.equals(b)


def test_signal_high_loan_ratio_defaults_more():
    # Sanidade de negócio: parcela alta sobre a renda -> mais default.
    df = generate_credit(n=12000, seed=3)
    high = df[df["loan_percent_income"] > 0.4]["default"].mean()
    low = df[df["loan_percent_income"] < 0.1]["default"].mean()
    assert high > low
