"""Testes da API (Etapa 6).

Testamos a lógica das rotas chamando as funções diretamente e a validação de
entrada pelo Pydantic — sem depender de um cliente HTTP de teste (evita
incompatibilidades de versão do Starlette/httpx no CI).
"""
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from xgboost import XGBClassifier

from app import main
from app.main import CreditApplication, health, predict
from src.features.build_features import make_xy
from src.models.train import _build_pipeline

SAMPLE = {
    "limit_bal": 200_000, "sex": 2, "education": 2, "marriage": 1, "age": 35,
    **{f"pay_{i}": 0 for i in range(1, 7)},
    **{f"bill_amt{i}": 50_000 for i in range(1, 7)},
    **{f"pay_amt{i}": 3_000 for i in range(1, 7)},
}


@pytest.fixture
def bundle_mock(monkeypatch, amostra_grande):
    """Substitui o bundle carregado do disco por um treinado na hora."""
    x, y = make_xy(amostra_grande)
    model = XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss", n_jobs=1)
    pipe = _build_pipeline(model).fit(x, y)
    bundle = {"pipeline": pipe, "threshold": 0.35, "model": "xgboost"}
    monkeypatch.setattr(main, "get_bundle", lambda: bundle)
    return bundle


def test_health():
    body = health()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_devolve_schema_valido(bundle_mock):
    result = predict(CreditApplication(**SAMPLE))
    assert 0.0 <= result.default_probability <= 1.0
    assert result.risk_band in {"BAIXO", "MEDIO", "ALTO"}
    assert isinstance(result.will_default, bool)
    assert result.threshold == 0.35


def test_decisao_usa_o_limiar_calibrado_e_nao_o_meio(bundle_mock):
    """`will_default` segue o limiar do bundle, não o 0.5 implícito."""
    result = predict(CreditApplication(**SAMPLE))
    assert result.will_default == (result.default_probability >= 0.35)


def test_faixas_de_risco_acompanham_o_limiar():
    assert main._risk_band(0.05, 0.35) == "BAIXO"    # < limiar/2
    assert main._risk_band(0.25, 0.35) == "MEDIO"    # entre limiar/2 e limiar
    assert main._risk_band(0.60, 0.35) == "ALTO"     # >= limiar


def test_sem_modelo_responde_503(monkeypatch):
    """Erro acionável em vez de 500 genérico quando o modelo não foi treinado."""
    def sem_modelo():
        raise RuntimeError("Modelo não encontrado. Rode 'python -m src.models.train'.")

    monkeypatch.setattr(main, "get_bundle", sem_modelo)
    with pytest.raises(HTTPException) as exc:
        predict(CreditApplication(**SAMPLE))
    assert exc.value.status_code == 503


def test_validacao_rejeita_menor_de_idade():
    with pytest.raises(ValidationError):
        CreditApplication(**{**SAMPLE, "age": 5})


def test_validacao_rejeita_status_de_pagamento_invalido():
    """`pay_*` só aceita -2..9, a faixa documentada do dataset."""
    with pytest.raises(ValidationError):
        CreditApplication(**{**SAMPLE, "pay_1": -5})


def test_validacao_rejeita_limite_negativo():
    with pytest.raises(ValidationError):
        CreditApplication(**{**SAMPLE, "limit_bal": -1000})


def test_aceita_fatura_negativa():
    """Saldo a favor do cliente é válido e não pode ser barrado na entrada."""
    app = CreditApplication(**{**SAMPLE, "bill_amt1": -2_500})
    assert app.bill_amt1 == -2_500
