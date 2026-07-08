# Build Brief — Credit Risk Scoring System

> Roteiro de construção do projeto. Qualidade de **portfólio para vaga de
> Cientista de Dados Pleno**. Construção **incremental**: uma etapa por vez,
> commit ao final de cada uma.

## Objetivo
Sistema de **avaliação de risco de crédito**: prever a probabilidade de um
cliente entrar em **inadimplência (default)** num empréstimo — ponta a ponta,
pronto para produção.

## Decisões já tomadas (não reabrir)
- **Dados:** sintéticos, gerados por código, **desbalanceados** (~12% de default)
  — é o que justifica o uso de **SMOTE** no treino.
- **Balanceamento:** SMOTE (imbalanced-learn), aplicado só no treino (nunca no teste).
- **Modelos:** comparar **XGBoost** e **CatBoost**; interpretar com **SHAP**.
- **Dashboard:** Streamlit. **Banco:** PostgreSQL com fallback SQLite.
- **Tracking:** MLflow (local em `mlruns/`).

## Etapas (commit ao fim de cada uma)

### Etapa 1 — Geração de dados sintéticos ✅ (feita no scaffold)
- `src/data/generate_synthetic.py`: gera clientes com features de crédito e um
  alvo `default` derivado de modelo logístico latente (desbalanceado ~12%).
- Salva `data/raw/credit.csv`. Testes em `tests/test_generate_synthetic.py`.

### Etapa 2 — ETL + carga no banco
- `src/data/etl.py`: limpeza, tipagem, validação; grava em PostgreSQL
  (fallback SQLite via `DATABASE_URL`) e em `data/processed/credit.parquet`.

### Etapa 3 — EDA
- `notebooks/01-eda.ipynb`: perfil de risco, distribuições, taxa de default por
  segmento, insights de negócio em markdown.

### Etapa 4 — Feature Engineering
- `src/features/build_features.py`: encoding, scaling, razões de risco
  (ex.: `loan_percent_income`, `debt_to_income`). Persistir o preprocessor.

### Etapa 5 — Treino e comparação de modelos
- `src/models/train.py`: aplica **SMOTE só no treino**, treina XGBoost e CatBoost,
  valida com **ROC-AUC, PR-AUC, KS, recall e F1 na classe default** (nunca só
  acurácia — base desbalanceada). Registra no MLflow, salva o melhor em `models/`.
- `src/models/explain.py`: SHAP em `reports/`.

### Etapa 6 — API FastAPI
- `app/main.py`: `/health` e `/predict` (features → probabilidade de default +
  faixa de risco). Carrega modelo + preprocessor.

### Etapa 7 — Dashboard Streamlit
- `app/dashboard.py`: KPIs de risco, gráficos da EDA, simulador de score que
  chama a API.

### Etapa 8 — Docker + Compose
- `Dockerfile` (API) e `docker-compose.yml` (API + Postgres + dashboard).

### Etapa 9 — Testes, CI, docs
- Cobertura nas peças críticas; CI verde. README profissional, diagrama de
  arquitetura, fluxograma e relatório técnico em `reports/`.

## Padrões
- Código modular e tipado; docstrings. Configs por env var; sem segredos no repo.
- Métricas apropriadas para dados desbalanceados. **Commits sem `Co-Authored-By`.**
