# Arquitetura — Credit Risk Scoring System

## Diagrama de arquitetura

```mermaid
flowchart LR
    A[Download UCI<br/>src/data/download] --> B[ETL + limpeza<br/>src/data/etl]
    B --> DB[(PostgreSQL / SQLite)]
    B --> Pq[(credit.parquet)]
    Pq --> F[Feature Engineering<br/>src/features/build_features]
    F --> S[SMOTE só no treino]
    S --> T[Treino XGBoost / CatBoost<br/>src/models/train]
    T --> ML[(MLflow<br/>mlruns/)]
    T --> C[Calibração do limiar<br/>por F1]
    C --> M[(model.joblib<br/>pipeline + threshold)]
    T --> X[SHAP<br/>src/models/explain]
    M --> API[FastAPI<br/>app/main]
    M --> DASH[Streamlit<br/>app/dashboard]
    API --> DASH
```

## Fluxograma da predição

```mermaid
flowchart TD
    R[Dados do cliente<br/>cadastro + 6 meses] --> V{Validação<br/>Pydantic}
    V -- inválida --> E[HTTP 422]
    V -- válida --> FE[Features derivadas<br/>utilização · regimes · tendência]
    FE --> P[Pipeline: preprocessor + modelo]
    P --> Pr[Probabilidade de default]
    Pr --> BD{Faixa de risco<br/>ancorada no limiar t}
    BD --> BX["BAIXO: p &lt; t/2"]
    BD --> MD["MÉDIO: t/2 ≤ p &lt; t"]
    BD --> AL["ALTO: p ≥ t"]
```

O limiar `t` não é fixo em 0.5: é calibrado no treino, salvo junto do modelo e
devolvido na resposta da API. Ver o README para o impacto dessa escolha.

## Componentes

| Camada | Arquivo | Responsabilidade |
|---|---|---|
| Dados | `src/data/download.py` | Baixa e cacheia o dataset real do UCI |
| ETL | `src/data/etl.py` | Limpa os códigos fora da documentação, deduplica, carrega |
| Config | `src/config.py` | Paths, URL do dataset e `DATABASE_URL` (fallback SQLite) |
| Features | `src/features/build_features.py` | Derivadas de crédito + ColumnTransformer |
| Treino | `src/models/train.py` | SMOTE + XGBoost/CatBoost + calibração de limiar + MLflow |
| Interpretação | `src/models/explain.py` | SHAP summary |
| API | `app/main.py` | `/health`, `/predict` |
| Dashboard | `app/dashboard.py` | KPIs, EDA, simulador |
| Deploy | `Dockerfile`, `docker-compose.yml` | API + Postgres + dashboard |
