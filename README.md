# Credit Risk Scoring System

Sistema ponta a ponta de **avaliação de risco de crédito**: prevê a
probabilidade de inadimplência (default) de um empréstimo — de dados a modelo
servido em API, com dashboard e monitoramento. Projeto de portfólio para vaga
de **Cientista de Dados Pleno**.

> ✅ **Projeto completo** (9 etapas). Roteiro em [`BUILD_BRIEF.md`](BUILD_BRIEF.md),
> arquitetura e relatório técnico em [`reports/`](reports/).

## Resultados
Comparação XGBoost vs CatBoost (base desbalanceada, ~21% default):

| Modelo | ROC-AUC | PR-AUC | KS | Recall (default) | F1 (default) |
|---|---|---|---|---|---|
| XGBoost | 0.863 | 0.665 | 0.578 | 0.579 | 0.621 |
| **CatBoost** ✅ | **0.865** | **0.675** | **0.589** | 0.573 | **0.624** |

Interpretabilidade via SHAP em [`reports/shap_summary.png`](reports/shap_summary.png).

## Stack
Python 3.12 · Pandas/NumPy · scikit-learn · XGBoost · CatBoost ·
imbalanced-learn (SMOTE) · SHAP · MLflow · FastAPI · Streamlit ·
SQLAlchemy/PostgreSQL · Docker · pytest

## Arquitetura (alvo)
```
Dados sintéticos (desbalanceados) → ETL → PostgreSQL → Feature Engineering
    → SMOTE (treino) → XGBoost / CatBoost + SHAP → MLflow
    → API (FastAPI) → Dashboard (Streamlit) → Monitoramento
```

## Como rodar (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Etapa 1 — gerar dados sintéticos
python -m src.data.generate_synthetic --n 12000 --seed 42

# Testes
pytest -q
```

## Como servir (API + dashboard)
```bash
python -m src.data.etl          # carrega dados
python -m src.models.train      # treina e salva o melhor modelo
python -m src.models.explain    # gera o SHAP

uvicorn app.main:app --reload            # API em http://localhost:8000/docs
streamlit run app/dashboard.py           # dashboard em http://localhost:8501
# ou tudo via Docker:
docker compose up --build
```

## Status das etapas
- [x] 1 — Geração de dados sintéticos
- [x] 2 — ETL + carga no banco
- [x] 3 — EDA
- [x] 4 — Feature Engineering
- [x] 5 — Treino (SMOTE) e comparação de modelos
- [x] 6 — API FastAPI
- [x] 7 — Dashboard Streamlit
- [x] 8 — Docker + Compose
- [x] 9 — Testes, CI e documentação
