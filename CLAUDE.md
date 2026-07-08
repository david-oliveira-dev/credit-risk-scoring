# CLAUDE.md — Credit Risk Scoring System

Contexto e convenções para qualquer sessão do Claude Code neste repositório.

## O que é
Projeto de portfólio (Cientista de Dados Pleno): sistema ponta a ponta de
scoring de risco de crédito (previsão de inadimplência). O roteiro está em
**`BUILD_BRIEF.md`** — siga-o etapa a etapa.

## Como trabalhar aqui
- Construção **incremental**: uma etapa do BUILD_BRIEF por vez, commit ao fim.
- Explique decisões técnicas (é portfólio). **Commits sem `Co-Authored-By`.**
- Não comitar dados grandes nem segredos. Configs por variável de ambiente.

## Stack
Python 3.12, Pandas/NumPy, scikit-learn, XGBoost, CatBoost, imbalanced-learn
(SMOTE), SHAP, MLflow, FastAPI, Streamlit, SQLAlchemy + PostgreSQL (fallback
SQLite), Docker, pytest.

## Cuidados de modelagem
- Base **desbalanceada** (~12% default): SMOTE só no treino, nunca no teste.
- Avaliar com ROC-AUC, PR-AUC, KS, recall e F1 na classe default — nunca só acurácia.

## Ambiente
- Use `venv`; o `pip` global da máquina do dono é bloqueado (PEP 668).
- Testes/CI devem rodar **sem** PostgreSQL (fallback SQLite via `DATABASE_URL`).
