"""Etapa 7 — Dashboard Streamlit de risco de crédito.

Três blocos:
  1. KPIs da carteira (clientes, taxa de default, limite médio, utilização)
  2. EDA visual (default por status de pagamento, escolaridade e utilização)
  3. Simulador de score: monte um perfil e veja a probabilidade de default

O simulador usa o modelo treinado localmente (models/model.joblib). Rode antes:
    python -m src.models.train

Uso:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permite rodar via `streamlit run app/dashboard.py`: garante a raiz do projeto
# no sys.path (senão o pacote `src` não é encontrado, pois o cwd fica em app/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_engineered_features,
)

st.set_page_config(page_title="Credit Risk Scoring", page_icon="💳", layout="wide")

ESCOLARIDADE = {1: "Pós-graduação", 2: "Universitário", 3: "Ensino médio", 4: "Outros"}
ESTADO_CIVIL = {1: "Casado", 2: "Solteiro", 3: "Outros"}
STATUS_PAGAMENTO = {
    -2: "-2 · sem consumo",
    -1: "-1 · pago em dia",
    0: "0 · rotativo (mínimo)",
    1: "1 · 1 mês de atraso",
    2: "2 · 2 meses de atraso",
    3: "3 · 3 meses de atraso",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    if config.PROCESSED_PARQUET.exists():
        return pd.read_parquet(config.PROCESSED_PARQUET)
    return pd.read_csv(config.RAW_CSV)


@st.cache_resource
def load_bundle():
    """Carrega o bundle (pipeline + limiar); None se o modelo não foi treinado."""
    from src.models.train import carregar_bundle

    try:
        return carregar_bundle()
    except FileNotFoundError:
        return None


st.title("💳 Credit Risk Scoring")
st.caption(
    "Dados reais: **Default of Credit Card Clients** (UCI) — 30.000 clientes de "
    "cartão em Taiwan, histórico de abr–set/2005."
)

df = load_data()
com_features = add_engineered_features(df)

# --- 1. KPIs ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes", f"{len(df):,}".replace(",", "."))
c2.metric("Taxa de default", f"{df['default'].mean():.1%}")
c3.metric("Limite médio", f"NT$ {df['limit_bal'].mean():,.0f}".replace(",", "."))
c4.metric("Utilização média do limite", f"{com_features['utilization_mean'].mean():.0%}")

st.divider()

# --- 2. EDA ---
left, right = st.columns(2)
with left:
    st.subheader("Default por status do último pagamento")
    st.caption("Códigos negativos e o zero não são documentados — e não são monótonos.")
    st.bar_chart(df.groupby("pay_1")["default"].mean())
with right:
    st.subheader("Default por escolaridade")
    por_escolaridade = df.groupby("education")["default"].mean()
    por_escolaridade.index = [ESCOLARIDADE.get(i, str(i)) for i in por_escolaridade.index]
    st.bar_chart(por_escolaridade)

st.subheader("Default por faixa de utilização do limite")
faixas = pd.cut(
    com_features["utilization_mean"],
    bins=[-float("inf"), 0.1, 0.3, 0.6, 0.9, float("inf")],
    labels=["até 10%", "10–30%", "30–60%", "60–90%", "acima de 90%"],
)
st.bar_chart(com_features.groupby(faixas, observed=True)["default"].mean())

st.divider()

# --- 3. Simulador ---
st.subheader("🎯 Simulador de score")
bundle = load_bundle()
if bundle is None:
    st.warning("Modelo não encontrado. Rode `python -m src.models.train` primeiro.")
else:
    limiar = bundle["threshold"]
    a, b, c = st.columns(3)

    limite = a.number_input("Limite de crédito (NT$)", 10_000, 1_000_000, 200_000, step=10_000)
    idade = a.number_input("Idade", 18, 100, 35)
    escolaridade = a.selectbox("Escolaridade", list(ESCOLARIDADE), format_func=ESCOLARIDADE.get)
    estado_civil = b.selectbox("Estado civil", list(ESTADO_CIVIL), format_func=ESTADO_CIVIL.get)
    sexo = b.selectbox("Sexo", [1, 2], format_func=lambda v: "Masculino" if v == 1 else "Feminino")
    status = b.selectbox(
        "Status do último pagamento", list(STATUS_PAGAMENTO), index=2,
        format_func=STATUS_PAGAMENTO.get,
    )
    fatura = c.number_input("Fatura do mês (NT$)", 0, 1_000_000, 50_000, step=5_000)
    pagamento = c.number_input("Valor pago no mês (NT$)", 0, 1_000_000, 3_000, step=1_000)
    meses_atraso = c.slider("Meses em atraso nos últimos 6", 0, 6, 0)

    if st.button("Calcular risco", type="primary"):
        # Replica o histórico de 6 meses a partir dos controles do simulador:
        # os `meses_atraso` primeiros meses recebem o status de atraso.
        historico = {}
        for i in range(1, 7):
            historico[f"pay_{i}"] = max(status, 1) if i <= meses_atraso else status
            historico[f"bill_amt{i}"] = fatura
            historico[f"pay_amt{i}"] = pagamento

        entrada = {
            "limit_bal": limite, "sex": sexo, "education": escolaridade,
            "marriage": estado_civil, "age": idade, **historico,
        }
        x = add_engineered_features(pd.DataFrame([entrada]))[
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ]
        proba = float(bundle["pipeline"].predict_proba(x)[:, 1][0])
        faixa = "BAIXO" if proba < limiar / 2 else "MEDIO" if proba < limiar else "ALTO"

        st.metric(
            "Probabilidade de default", f"{proba:.1%}",
            help=f"Limiar de decisão calibrado: {limiar:.1%}",
        )
        (st.success if faixa == "BAIXO" else st.warning if faixa == "MEDIO" else st.error)(
            f"Risco {faixa} — o limiar de decisão está em {limiar:.1%}"
        )
