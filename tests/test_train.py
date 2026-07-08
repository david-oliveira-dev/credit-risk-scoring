"""Testes do treino/avaliação (Etapa 5). Usa um modelo leve para ser rápido."""
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from src.data.generate_synthetic import generate_credit
from src.features.build_features import make_xy
from src.models.train import _build_pipeline, evaluate

METRIC_KEYS = {"roc_auc", "pr_auc", "ks", "recall_default", "f1_default"}


def test_pipeline_trains_and_metrics_are_valid():
    x, y = make_xy(generate_credit(n=3000, seed=42))
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=0
    )
    model = XGBClassifier(
        n_estimators=60, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", random_state=0, n_jobs=1,
    )
    pipe = _build_pipeline(model)
    pipe.fit(x_tr, y_tr)

    metrics = evaluate(pipe, x_te, y_te)
    assert set(metrics) == METRIC_KEYS
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    # o dataset tem sinal real: o modelo deve superar o acaso com folga
    assert metrics["roc_auc"] > 0.65


def test_smote_only_affects_training():
    # No pipeline do imblearn, o SMOTE não altera o nº de amostras previstas.
    x, y = make_xy(generate_credit(n=1500, seed=7))
    model = XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss", n_jobs=1)
    pipe = _build_pipeline(model).fit(x, y)
    preds = pipe.predict(x)
    assert len(preds) == len(x)
