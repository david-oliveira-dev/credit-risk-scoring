"""Gera as figuras de resultado usadas no README.

Salva em `reports/figures/`:

- `tradeoff_limiar.png` — recall, precisão e F1 em função do limiar de decisão,
  com o 0.5 padrão e o limiar calibrado marcados. É o achado central do projeto:
  a escolha do corte pesa mais que a escolha entre CatBoost e XGBoost;
- `curva_pr.png` — curva precisão–recall no holdout, com os dois pontos de
  operação destacados.

Uso:
    python -m src.viz.plot_results
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # backend sem display: roda em servidor e no CI
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from src.features.build_features import make_xy  # noqa: E402
from src.models.train import RANDOM_STATE, _get_data, carregar_bundle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("plot_results")

ROOT = Path(__file__).resolve().parents[2]
FIGURAS = ROOT / "reports" / "figures"


def _scores_do_holdout() -> tuple[np.ndarray, np.ndarray, float]:
    """Recompõe o mesmo split do treino e devolve (y, proba, limiar calibrado)."""
    bundle = carregar_bundle()
    x, y = make_xy(_get_data())
    _, x_te, _, y_te = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    proba = bundle["pipeline"].predict_proba(x_te)[:, 1]
    return y_te.to_numpy(), proba, bundle["threshold"]


def plot_tradeoff_limiar() -> Path:
    y, proba, limiar = _scores_do_holdout()

    grade = np.linspace(0.05, 0.95, 181)
    recalls, precisoes, f1s = [], [], []
    for t in grade:
        pred = (proba >= t).astype(int)
        recalls.append(recall_score(y, pred))
        precisoes.append(precision_score(y, pred, zero_division=0))
        f1s.append(f1_score(y, pred))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(grade, recalls, color="#c44e52", lw=2, label="recall (inadimplentes capturados)")
    ax.plot(grade, precisoes, color="#4c72b0", lw=2, label="precisão")
    ax.plot(grade, f1s, color="#55a868", lw=2, ls="--", label="F1")

    for t, cor, rotulo in [(0.5, "#8c8c8c", "padrão 0.50"), (limiar, "k", f"calibrado {limiar:.3f}")]:
        ax.axvline(t, color=cor, lw=1.2, ls=":")
        r = recall_score(y, (proba >= t).astype(int))
        ax.plot([t], [r], "o", color=cor, ms=7)
        ax.annotate(f"{rotulo}\nrecall {r:.3f}", xy=(t, r), xytext=(t + 0.03, r + 0.08),
                    fontsize=9, color=cor)

    ax.set_xlabel("limiar de decisão")
    ax.set_ylabel("valor da métrica")
    ax.set_title("O limiar decide mais que o modelo\n"
                 "sair de 0.50 para o corte calibrado leva o recall de 0,396 a 0,587")
    ax.legend(loc="center right")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()

    FIGURAS.mkdir(parents=True, exist_ok=True)
    destino = FIGURAS / "tradeoff_limiar.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    logger.info("Salvo %s", destino.name)
    return destino


def plot_curva_pr() -> Path:
    y, proba, limiar = _scores_do_holdout()
    precisao, recall, _ = precision_recall_curve(y, proba)
    pr_auc = average_precision_score(y, proba)
    taxa_base = y.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precisao, color="#c44e52", lw=2, label=f"CatBoost (PR-AUC {pr_auc:.3f})")
    ax.axhline(taxa_base, ls="--", c="k", lw=1, label=f"acaso ({taxa_base:.1%} de default)")

    for t, cor, rotulo in [(0.5, "#8c8c8c", "0.50"), (limiar, "k", f"{limiar:.3f}")]:
        pred = (proba >= t).astype(int)
        ax.plot([recall_score(y, pred)], [precision_score(y, pred, zero_division=0)],
                "o", color=cor, ms=8, label=f"limiar {rotulo}")

    ax.set_xlabel("recall"); ax.set_ylabel("precisão")
    ax.set_title("Curva precisão–recall no holdout (5.993 clientes)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    FIGURAS.mkdir(parents=True, exist_ok=True)
    destino = FIGURAS / "curva_pr.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    logger.info("Salvo %s", destino.name)
    return destino


def main() -> None:
    plot_tradeoff_limiar()
    plot_curva_pr()


if __name__ == "__main__":
    main()
