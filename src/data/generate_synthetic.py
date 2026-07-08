"""Gerador de dados sintéticos para scoring de risco de crédito.

Assim como no projeto de churn, o alvo (`default`) não é sorteado à toa: ele vem
de um *modelo logístico latente* baseado nas features de crédito. Isso dá **sinal
real** — quem tem alta relação parcela/renda, muita dívida sobre a renda, histórico
curto e passado de atrasos tende a inadimplir — e o dataset nasce **desbalanceado**
(~12% de default), que é justamente o cenário onde o SMOTE faz sentido.

Uso:
    python -m src.data.generate_synthetic --n 12000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

HOME = ["RENT", "MORTGAGE", "OWN"]
INTENT = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_credit(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    """Gera `n` solicitações de crédito sintéticas com alvo `default` (~12%)."""
    rng = np.random.default_rng(seed)

    # --- Perfil do solicitante ---
    age = rng.integers(21, 70, size=n)
    income = np.round(rng.lognormal(mean=10.8, sigma=0.5, size=n)).clip(12000, 400000)
    emp_length = rng.integers(0, 30, size=n).clip(0, np.maximum(age - 18, 0))
    home = rng.choice(HOME, size=n, p=[0.50, 0.38, 0.12])
    intent = rng.choice(INTENT, size=n, p=[0.22, 0.15, 0.15, 0.15, 0.15, 0.18])

    credit_hist_length = rng.integers(1, 30, size=n).clip(1, np.maximum(age - 18, 1))
    num_credit_lines = rng.integers(1, 15, size=n)
    past_delinquencies = rng.poisson(0.4, size=n)

    # --- Empréstimo ---
    loan_amount = np.round(rng.uniform(1000, 45000, size=n)).astype(int)
    interest_rate = np.round(rng.normal(11, 3.2, size=n), 2).clip(5.0, 24.0)
    loan_percent_income = np.round(loan_amount / income, 3).clip(0.01, 1.5)
    # dívida total sobre renda (inclui outras linhas de crédito)
    debt_to_income = np.round(
        loan_percent_income + num_credit_lines * rng.uniform(0.01, 0.04, size=n), 3
    ).clip(0.02, 2.0)

    # --- Modelo logístico latente para o default ---
    logit = (
        -3.0                                   # intercepto -> base desbalanceada
        + 2.6 * loan_percent_income            # parcela pesada na renda -> risco
        + 1.4 * (debt_to_income - 0.3)         # endividamento -> risco
        + 0.10 * (interest_rate - 11)          # juros alto (cliente mais arriscado)
        + 0.45 * past_delinquencies            # atrasos passados -> risco forte
        - 0.030 * (income / 10000)             # renda maior -> menos risco
        - 0.05 * credit_hist_length            # histórico longo -> menos risco
        - 0.03 * emp_length                    # estabilidade no emprego -> menos risco
        + 0.4 * (home == "RENT").astype(float)  # aluguel -> risco um pouco maior
    )
    prob = _sigmoid(logit)
    default = rng.binomial(1, prob)

    df = pd.DataFrame({
        "customer_id": [f"A{200000 + i}" for i in range(n)],
        "age": age,
        "income": income.astype(int),
        "employment_length": emp_length,
        "home_ownership": home,
        "loan_intent": intent,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "loan_percent_income": loan_percent_income,
        "debt_to_income": debt_to_income,
        "credit_history_length": credit_hist_length,
        "num_credit_lines": num_credit_lines,
        "past_delinquencies": past_delinquencies,
        "default": default,
    })
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dados sintéticos de risco de crédito.")
    parser.add_argument("--n", type=int, default=12000, help="nº de solicitações")
    parser.add_argument("--seed", type=int, default=42, help="semente aleatória")
    parser.add_argument("--out", type=Path, default=RAW_DIR / "credit.csv")
    args = parser.parse_args()

    df = generate_credit(n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    rate = df["default"].mean()
    print(f"Gerados {len(df)} pedidos de crédito -> {args.out}")
    print(f"Taxa de default: {rate:.1%}  (classe positiva minoritária -> SMOTE)")


if __name__ == "__main__":
    main()
