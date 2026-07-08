"""Etapa 7 — Dashboard Streamlit de risco de crédito.

Três blocos:
  1. KPIs da carteira (volume, taxa de default, ticket médio)
  2. EDA visual (default por segmento, distribuição de comprometimento de renda)
  3. Simulador de score: preencha uma solicitação e veja a probabilidade de default

O simulador usa o modelo treinado localmente (models/model.joblib). Rode antes:
    python -m src.models.train

Uso:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_engineered_features,
)

st.set_page_config(page_title="Credit Risk Scoring", page_icon="💳", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    if config.PROCESSED_PARQUET.exists():
        return pd.read_parquet(config.PROCESSED_PARQUET)
    return pd.read_csv(config.RAW_CSV)


@st.cache_resource
def load_model():
    import joblib
    if config.MODELS_DIR.joinpath("model.joblib").exists():
        return joblib.load(config.MODELS_DIR / "model.joblib")
    return None


st.title("💳 Credit Risk Scoring")

df = load_data()

# --- 1. KPIs ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Solicitações", f"{len(df):,}")
c2.metric("Taxa de default", f"{df['default'].mean():.1%}")
c3.metric("Empréstimo médio", f"R$ {df['loan_amount'].mean():,.0f}")
c4.metric("Juros médio", f"{df['interest_rate'].mean():.1f}%")

st.divider()

# --- 2. EDA ---
left, right = st.columns(2)
with left:
    st.subheader("Default por tipo de moradia")
    st.bar_chart(df.groupby("home_ownership")["default"].mean())
with right:
    st.subheader("Default por finalidade do empréstimo")
    st.bar_chart(df.groupby("loan_intent")["default"].mean())

st.subheader("Comprometimento de renda (loan_percent_income)")
st.bar_chart(df["loan_percent_income"].round(1).value_counts().sort_index())

st.divider()

# --- 3. Simulador ---
st.subheader("🎯 Simulador de score")
model = load_model()
if model is None:
    st.warning("Modelo não encontrado. Rode `python -m src.models.train` primeiro.")
else:
    a, b, c = st.columns(3)
    entrada = {
        "age": a.number_input("Idade", 18, 100, 35),
        "income": a.number_input("Renda anual", 12000, 400000, 60000, step=1000),
        "employment_length": a.number_input("Anos de emprego", 0, 40, 5),
        "home_ownership": b.selectbox("Moradia", ["RENT", "MORTGAGE", "OWN"]),
        "loan_intent": b.selectbox(
            "Finalidade",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
             "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        ),
        "loan_amount": b.number_input("Valor do empréstimo", 1000, 45000, 12000, step=500),
        "interest_rate": c.slider("Juros (%)", 5.0, 24.0, 11.5),
        "loan_percent_income": c.slider("Parcela/renda", 0.01, 1.5, 0.2),
        "debt_to_income": c.slider("Dívida/renda", 0.02, 2.0, 0.35),
        "credit_history_length": a.number_input("Histórico de crédito (anos)", 1, 30, 8),
        "num_credit_lines": b.number_input("Linhas de crédito", 1, 15, 4),
        "past_delinquencies": c.number_input("Atrasos passados", 0, 10, 0),
    }
    if st.button("Calcular risco", type="primary"):
        x = add_engineered_features(pd.DataFrame([entrada]))[
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ]
        proba = float(model.predict_proba(x)[:, 1][0])
        band = "BAIXO" if proba < 0.10 else "MEDIO" if proba < 0.30 else "ALTO"
        st.metric("Probabilidade de default", f"{proba:.1%}", help=f"Faixa de risco: {band}")
        (st.success if band == "BAIXO" else st.warning if band == "MEDIO" else st.error)(
            f"Risco {band}"
        )
