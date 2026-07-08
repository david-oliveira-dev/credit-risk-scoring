# Arquitetura — Credit Risk Scoring System

## Diagrama de arquitetura

```mermaid
flowchart LR
    A[Gerador sintético<br/>src/data/generate_synthetic] --> B[ETL<br/>src/data/etl]
    B --> DB[(PostgreSQL / SQLite)]
    B --> Pq[(credit.parquet)]
    Pq --> F[Feature Engineering<br/>src/features/build_features]
    F --> S[SMOTE só no treino]
    S --> T[Treino XGBoost / CatBoost<br/>src/models/train]
    T --> ML[(MLflow<br/>mlruns/)]
    T --> M[(model.joblib)]
    T --> X[SHAP<br/>src/models/explain]
    M --> API[FastAPI<br/>app/main]
    M --> DASH[Streamlit<br/>app/dashboard]
    API --> DASH
```

## Fluxograma da predição

```mermaid
flowchart TD
    R[Solicitação de crédito] --> V{Validação<br/>Pydantic}
    V -- inválida --> E[HTTP 422]
    V -- válida --> FE[Features derivadas]
    FE --> P[Pipeline: preprocessor + modelo]
    P --> Pr[Probabilidade de default]
    Pr --> BD{Faixa de risco}
    BD --> BX[BAIXO < 10%]
    BD --> MD[MÉDIO 10–30%]
    BD --> AL[ALTO > 30%]
```

## Componentes

| Camada | Arquivo | Responsabilidade |
|---|---|---|
| Dados | `src/data/generate_synthetic.py` | Gera base sintética desbalanceada |
| ETL | `src/data/etl.py` | Extrai, valida e carrega no banco + parquet |
| Config | `src/config.py` | Paths e `DATABASE_URL` (fallback SQLite) |
| Features | `src/features/build_features.py` | Features derivadas + ColumnTransformer |
| Treino | `src/models/train.py` | SMOTE + XGBoost/CatBoost + MLflow |
| Interpretação | `src/models/explain.py` | SHAP summary |
| API | `app/main.py` | `/health`, `/predict` |
| Dashboard | `app/dashboard.py` | KPIs, EDA, simulador |
| Deploy | `Dockerfile`, `docker-compose.yml` | API + Postgres + dashboard |
