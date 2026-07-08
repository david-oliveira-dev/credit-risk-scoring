"""Etapa 6 — API FastAPI para servir o modelo de risco de crédito.

Endpoints:
    GET  /health   -> status do serviço e se o modelo está carregado
    POST /predict  -> recebe os dados de uma solicitação e devolve a
                      probabilidade de default + a faixa de risco.

Uso:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_engineered_features,
)
from src.models.train import MODEL_PATH

app = FastAPI(title="Credit Risk Scoring API", version="1.0.0")


class CreditApplication(BaseModel):
    """Dados brutos de uma solicitação de crédito (antes das features derivadas)."""
    age: int = Field(ge=18, le=100, examples=[35])
    income: float = Field(gt=0, examples=[60000])
    employment_length: int = Field(ge=0, examples=[5])
    home_ownership: str = Field(examples=["RENT"])
    loan_intent: str = Field(examples=["PERSONAL"])
    loan_amount: float = Field(gt=0, examples=[12000])
    interest_rate: float = Field(gt=0, examples=[11.5])
    loan_percent_income: float = Field(ge=0, examples=[0.2])
    debt_to_income: float = Field(ge=0, examples=[0.35])
    credit_history_length: int = Field(ge=0, examples=[8])
    num_credit_lines: int = Field(ge=0, examples=[4])
    past_delinquencies: int = Field(ge=0, examples=[0])


class Prediction(BaseModel):
    default_probability: float
    risk_band: str
    will_default: bool


@lru_cache(maxsize=1)
def get_model():
    """Carrega o Pipeline treinado (uma vez). Erro claro se não existir."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {MODEL_PATH}. Rode 'python -m src.models.train'."
        )
    return joblib.load(MODEL_PATH)


def _risk_band(p: float) -> str:
    if p < 0.10:
        return "BAIXO"
    if p < 0.30:
        return "MEDIO"
    return "ALTO"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(application: CreditApplication) -> Prediction:
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df = add_engineered_features(pd.DataFrame([application.model_dump()]))
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    proba = float(model.predict_proba(x)[:, 1][0])
    return Prediction(
        default_probability=round(proba, 4),
        risk_band=_risk_band(proba),
        will_default=proba >= 0.5,
    )
