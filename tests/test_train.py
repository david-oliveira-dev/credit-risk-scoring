"""Testes do treino/avaliação (Etapa 5). Usa um modelo leve para ser rápido."""
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.build_features import make_xy
from src.models.train import _build_pipeline, evaluate, melhor_limiar

METRIC_KEYS = {
    "roc_auc", "pr_auc", "ks", "threshold",
    "recall_default", "precision_default", "f1_default",
    "recall_calibrado", "precision_calibrada", "f1_calibrado",
}


def _modelo_leve() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=60, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", random_state=0, n_jobs=1,
    )


def test_pipeline_treina_e_metricas_sao_validas(amostra_grande):
    x, y = make_xy(amostra_grande)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=0
    )
    pipe = _build_pipeline(_modelo_leve()).fit(x_tr, y_tr)

    metrics = evaluate(pipe, x_te, y_te)

    assert set(metrics) == METRIC_KEYS
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    # A fixture tem sinal plantado, então o pipeline precisa superar o acaso.
    # A barra é baixa de propósito: isto verifica que o encadeamento
    # features -> SMOTE -> modelo funciona, não a qualidade do modelo real
    # (essa está em reports/metrics.json, medida sobre os 30 mil clientes).
    assert metrics["roc_auc"] > 0.60


def test_limiar_calibrado_captura_mais_que_o_padrao(amostra_grande):
    """O ponto da calibração: mais recall na classe minoritária que o 0.5."""
    x, y = make_xy(amostra_grande)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=0
    )
    pipe = _build_pipeline(_modelo_leve()).fit(x_tr, y_tr)

    metrics = evaluate(pipe, x_te, y_te)

    assert metrics["f1_calibrado"] >= metrics["f1_default"]
    if metrics["threshold"] < 0.5:
        assert metrics["recall_calibrado"] >= metrics["recall_default"]


def test_melhor_limiar_fica_no_intervalo_da_grade():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.15, 0.6])
    t = melhor_limiar(y, proba)
    assert 0.05 <= t <= 0.95


def test_smote_nao_altera_o_numero_de_previsoes(amostra_grande):
    """No pipeline do imblearn o SMOTE só age no fit — nunca na inferência."""
    x, y = make_xy(amostra_grande)
    pipe = _build_pipeline(_modelo_leve()).fit(x, y)
    assert len(pipe.predict(x)) == len(x)
