"""Etapa 5 (interpretabilidade) — explicações SHAP do modelo campeão.

Gera um summary plot de SHAP mostrando quais features mais influenciam a
previsão de inadimplência. Salva a figura em reports/shap_summary.png.

Uso:
    python -m src.models.explain
"""
from __future__ import annotations

import logging

import joblib
import matplotlib
import shap

matplotlib.use("Agg")  # backend sem display (roda em servidor/CI)
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src.models.train import MODEL_PATH, _get_data  # noqa: E402
from src.features.build_features import make_xy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("explain")

SHAP_PLOT = config.REPORTS_DIR / "shap_summary.png"


def explain(sample_size: int = 500) -> None:
    """Calcula SHAP num recorte dos dados e salva o summary plot."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Rode o treino antes.")

    pipe = joblib.load(MODEL_PATH)
    pre = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["classifier"]

    x, _ = make_xy(_get_data())
    x = x.sample(min(sample_size, len(x)), random_state=42)
    x_t = pre.transform(x)
    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_t)

    shap.summary_plot(shap_values, x_t, feature_names=feature_names, show=False)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary salvo em %s", SHAP_PLOT)


if __name__ == "__main__":
    explain()
