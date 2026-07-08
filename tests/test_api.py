"""Testes da API FastAPI (Etapa 6)."""
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from app import main
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
def client(monkeypatch):
    x, y = make_xy(generate_credit(n=1500, seed=1))
    model = XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss", n_jobs=1)
    pipe = _build_pipeline(model).fit(x, y)
    monkeypatch.setattr(main, "get_model", lambda: pipe)
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_schema(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_band"] in {"BAIXO", "MEDIO", "ALTO"}
    assert isinstance(body["will_default"], bool)


def test_predict_rejects_bad_input(client):
    bad = {**SAMPLE, "age": 5}  # < 18 viola a validação
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
