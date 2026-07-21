"""Etapa 5 — Treino, balanceamento (SMOTE) e comparação de modelos.

Pontos-chave:
- SMOTE entra DENTRO de um Pipeline do imbalanced-learn, então só reamostra
  durante o `fit` (dados de treino). No teste/predict ele é ignorado — evita
  vazamento de dados, o erro clássico de quem aplica SMOTE no dataset inteiro.
- Base desbalanceada => avaliamos com ROC-AUC, PR-AUC, KS, recall e F1 na
  classe default; acurácia sozinha enganaria.
- Compara XGBoost e CatBoost, registra tudo no MLflow e salva o melhor modelo
  (o Pipeline inteiro, já com o pré-processador) em models/.

Uso:
    python -m src.models.train
"""
from __future__ import annotations

import json
import logging

import joblib
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src import config
from src.data.etl import run_etl
from src.features.build_features import build_preprocessor, make_xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")

RANDOM_STATE = 42
MODEL_PATH = config.MODELS_DIR / "model.joblib"
METRICS_PATH = config.REPORTS_DIR / "metrics.json"


def _get_data() -> pd.DataFrame:
    """Carrega o parquet processado; se não existir, roda o ETL."""
    if config.PROCESSED_PARQUET.exists():
        return pd.read_parquet(config.PROCESSED_PARQUET)
    logger.info("Parquet ausente — rodando ETL...")
    return run_etl()


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Estatística KS = separação máxima entre TPR e FPR."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def carregar_bundle() -> dict:
    """Carrega o artefato salvo: ``{"pipeline", "threshold", "model"}``.

    Ponto único de leitura do modelo — usado pela API, pelo dashboard e pelo
    SHAP, para que os três enxerguem o mesmo limiar.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. Rode 'python -m src.models.train'."
        )
    return joblib.load(MODEL_PATH)


def melhor_limiar(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Limiar que maximiza o F1 na classe default.

    O padrão de 0.5 é arbitrário e, numa base com 22% de positivos, custa caro:
    o modelo fica conservador demais e deixa passar a maioria dos inadimplentes.
    Aqui o corte é escolhido pelos dados. Em produção o critério seria o custo
    real do negócio (perda de um default vs. margem perdida numa recusa), e este
    F1 é o substituto neutro enquanto esses números não estão na mesa.
    """
    grade = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, (proba >= t).astype(int)) for t in grade]
    return float(grade[int(np.argmax(f1s))])


def evaluate(pipe: ImbPipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Métricas apropriadas para base desbalanceada, em dois limiares.

    As métricas de ranqueamento (ROC-AUC, PR-AUC, KS) não dependem de limiar.
    Já recall/precisão/F1 dependem — por isso são reportadas tanto no 0.5 padrão
    quanto no limiar calibrado, que é o que a API usa.
    """
    proba = pipe.predict_proba(x_test)[:, 1]
    y = y_test.to_numpy()
    limiar = melhor_limiar(y, proba)

    pred_05 = (proba >= 0.5).astype(int)
    pred_cal = (proba >= limiar).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "ks": round(_ks_statistic(y, proba), 4),
        "threshold": round(limiar, 3),
        # limiar padrão (0.5)
        "recall_default": round(recall_score(y, pred_05), 4),
        "precision_default": round(precision_score(y, pred_05), 4),
        "f1_default": round(f1_score(y, pred_05), 4),
        # limiar calibrado
        "recall_calibrado": round(recall_score(y, pred_cal), 4),
        "precision_calibrada": round(precision_score(y, pred_cal), 4),
        "f1_calibrado": round(f1_score(y, pred_cal), 4),
    }


def _build_pipeline(model) -> ImbPipeline:
    """Pré-processador -> SMOTE (só no treino) -> classificador."""
    return ImbPipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", model),
    ])


def _candidate_models() -> dict:
    return {
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.08,
            random_seed=RANDOM_STATE, verbose=0,
        ),
    }


def train_and_compare() -> dict:
    """Treina os candidatos, escolhe o melhor por PR-AUC e persiste."""
    df = _get_data()
    x, y = make_xy(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Treino=%d  Teste=%d  (default no treino: %.1f%%)",
                len(x_train), len(x_test), 100 * y_train.mean())

    # Backend SQLite (o file store foi descontinuado nas versões novas do MLflow).
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("credit-risk")

    results, best = {}, None
    for name, model in _candidate_models().items():
        with mlflow.start_run(run_name=name):
            pipe = _build_pipeline(model)
            pipe.fit(x_train, y_train)
            metrics = evaluate(pipe, x_test, y_test)
            results[name] = metrics
            mlflow.log_param("model", name)
            mlflow.log_metrics(metrics)
            logger.info("%s -> %s", name, metrics)
            if best is None or metrics["pr_auc"] > best[1]["pr_auc"]:
                best = (name, metrics, pipe)

    best_name, best_metrics, best_pipe = best
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Salva o bundle: o pipeline sozinho não basta, o limiar calibrado faz parte
    # da decisão e precisa viajar junto para a API não voltar ao 0.5 implícito.
    joblib.dump(
        {"pipeline": best_pipe, "threshold": best_metrics["threshold"], "model": best_name},
        MODEL_PATH,
    )
    payload = {"best_model": best_name, "metrics": results}
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Melhor modelo: %s (PR-AUC=%.4f) salvo em %s",
                best_name, best_metrics["pr_auc"], MODEL_PATH.name)
    return payload


if __name__ == "__main__":
    train_and_compare()
