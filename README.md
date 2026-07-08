# Credit Risk Scoring System

Sistema ponta a ponta de **avaliação de risco de crédito**: prevê a
probabilidade de inadimplência (default) de um empréstimo — de dados a modelo
servido em API, com dashboard e monitoramento. Projeto de portfólio para vaga
de **Cientista de Dados Pleno**.

> 🚧 **Em construção incremental.** Roteiro completo em
> [`BUILD_BRIEF.md`](BUILD_BRIEF.md). Etapa atual: **1 — geração de dados**.

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

## Status das etapas
- [x] 1 — Geração de dados sintéticos
- [ ] 2 — ETL + carga no banco
- [ ] 3 — EDA
- [ ] 4 — Feature Engineering
- [ ] 5 — Treino (SMOTE) e comparação de modelos
- [ ] 6 — API FastAPI
- [ ] 7 — Dashboard Streamlit
- [ ] 8 — Docker + Compose
- [ ] 9 — Testes, CI e documentação
