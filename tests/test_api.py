"""Testes da API (Etapa 6).

Testamos a lógica das rotas chamando as funções diretamente e a validação de
entrada pelo Pydantic — sem depender de um cliente HTTP de teste (evita
incompatibilidades de versão do Starlette/httpx no CI).
"""
import pytest
from pydantic import ValidationError
from xgboost import XGBClassifier

from app import main
from app.main import CreditApplication, health, predict
from src.data.generate_synthetic import generate_credit
from src.features.build_features import make_xy
from src.models.train import _build_pipeline

SAMPLE = {
    "age": 35, "income": 60000, "employment_length": 5,
    "home_ownership": "RENT", "loan_intent": "PERSONAL",
    "loan_amount": 12000, "interest_rate": 11.5, "loan_percent_income": 0.2,
    "debt_to_income": 0.35, "credit_history_length": 8,
    "num_credit_lines": 4, "past_delinquencies": 0,
}


@pytest.fixture
def patched_model(monkeypatch):
    x, y = make_xy(generate_credit(n=1500, seed=1))
    model = XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss", n_jobs=1)
    pipe = _build_pipeline(model).fit(x, y)
    monkeypatch.setattr(main, "get_model", lambda: pipe)
    return pipe


def test_health():
    body = health()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_returns_valid_schema(patched_model):
    result = predict(CreditApplication(**SAMPLE))
    assert 0.0 <= result.default_probability <= 1.0
    assert result.risk_band in {"BAIXO", "MEDIO", "ALTO"}
    assert isinstance(result.will_default, bool)


def test_input_validation_rejects_underage():
    with pytest.raises(ValidationError):
        CreditApplication(**{**SAMPLE, "age": 5})  # < 18 viola o Field(ge=18)
