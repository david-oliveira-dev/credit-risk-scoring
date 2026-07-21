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

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_engineered_features,
)
from src.models.train import MODEL_PATH, carregar_bundle

app = FastAPI(title="Credit Risk Scoring API", version="1.0.0")


class CreditApplication(BaseModel):
    """Cadastro + 6 meses de histórico de um cliente, como vêm da base do UCI.

    As features derivadas (utilização do limite, taxa de pagamento, contagens de
    atraso) são calculadas aqui pelo mesmo código do treino — o cliente da API
    manda só o dado cru.
    """

    # Cadastro
    limit_bal: float = Field(gt=0, examples=[200000], description="Limite de crédito (NT$)")
    sex: int = Field(ge=1, le=2, examples=[2], description="1=masculino, 2=feminino")
    education: int = Field(ge=1, le=4, examples=[2],
                           description="1=pós, 2=universitário, 3=médio, 4=outros")
    marriage: int = Field(ge=1, le=3, examples=[1],
                          description="1=casado, 2=solteiro, 3=outros")
    age: int = Field(ge=18, le=100, examples=[35])

    # Status de pagamento dos 6 meses (set→abr/2005).
    # -2=sem consumo, -1=pago em dia, 0=rotativo, 1..9=meses de atraso.
    pay_1: int = Field(ge=-2, le=9, examples=[0])
    pay_2: int = Field(ge=-2, le=9, examples=[0])
    pay_3: int = Field(ge=-2, le=9, examples=[0])
    pay_4: int = Field(ge=-2, le=9, examples=[0])
    pay_5: int = Field(ge=-2, le=9, examples=[0])
    pay_6: int = Field(ge=-2, le=9, examples=[0])

    # Valor da fatura de cada mês (pode ser negativo: saldo a favor).
    bill_amt1: float = Field(examples=[50000])
    bill_amt2: float = Field(examples=[48000])
    bill_amt3: float = Field(examples=[46000])
    bill_amt4: float = Field(examples=[44000])
    bill_amt5: float = Field(examples=[42000])
    bill_amt6: float = Field(examples=[40000])

    # Valor efetivamente pago em cada mês.
    pay_amt1: float = Field(ge=0, examples=[3000])
    pay_amt2: float = Field(ge=0, examples=[3000])
    pay_amt3: float = Field(ge=0, examples=[3000])
    pay_amt4: float = Field(ge=0, examples=[3000])
    pay_amt5: float = Field(ge=0, examples=[3000])
    pay_amt6: float = Field(ge=0, examples=[3000])


class Prediction(BaseModel):
    default_probability: float
    risk_band: str
    will_default: bool
    threshold: float


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    """Carrega o bundle treinado (pipeline + limiar) uma única vez."""
    try:
        return carregar_bundle()
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc


def _risk_band(p: float, limiar: float) -> str:
    """Faixa de risco ancorada no limiar de decisão, não em cortes redondos.

    ALTO é justamente quem o modelo marcaria como default no limiar calibrado;
    BAIXO fica abaixo de metade dele. Assim a faixa e a decisão contam a mesma
    história — se o limiar for recalibrado, as faixas acompanham.
    """
    if p < limiar / 2:
        return "BAIXO"
    if p < limiar:
        return "MEDIO"
    return "ALTO"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(application: CreditApplication) -> Prediction:
    try:
        bundle = get_bundle()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    limiar = bundle["threshold"]
    df = add_engineered_features(pd.DataFrame([application.model_dump()]))
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    proba = float(bundle["pipeline"].predict_proba(x)[:, 1][0])
    return Prediction(
        default_probability=round(proba, 4),
        risk_band=_risk_band(proba, limiar),
        will_default=proba >= limiar,
        threshold=limiar,
    )
