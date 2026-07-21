# Credit Risk Scoring System

Sistema **ponta a ponta** de **avaliação de risco de crédito**: estima a
probabilidade de inadimplência (*default*) de uma solicitação de empréstimo — de
dados a score servido em API, com explicabilidade por SHAP, dashboard e
containerização. Projeto de portfólio para **Cientista de Dados Pleno**, com foco
em rigor sob classe desbalanceada (PR-AUC e KS, SMOTE sem vazamento) e em decisão
de crédito (o que fazer com cada faixa de risco).

![CI](https://github.com/david-oliveira-dev/credit-risk-scoring/actions/workflows/ci.yml/badge.svg)

> ⚠️ **Sobre os dados.** A base é **sintética**, gerada por
> [`src/data/generate_synthetic.py`](src/data/generate_synthetic.py) a partir de um
> modelo logístico latente. A escolha é deliberada: o objetivo é a **engenharia
> ponta a ponta** — SMOTE sem vazamento, comparação de modelos, explicabilidade,
> serviço e monitoramento — e não a descoberta de um sinal novo. Como o gerador
> planta os fatores de risco que os modelos recuperam, **as métricas abaixo não são
> comparáveis a benchmarks de dado real** (German Credit, Give Me Some Credit);
> leia-as como validação de que o pipeline funciona fim a fim.

## Contexto de negócio
Conceder crédito é decidir sob incerteza com **erros assimétricos**: aprovar quem
vai inadimplir vira perda direta do principal; negar um bom pagador vira receita
perdida e cliente na concorrência. O trabalho do modelo não é "acertar mais" — é
**ordenar risco bem o suficiente** para que a política de crédito possa cortar onde
o negócio quiser cortar. Por isso a saída aqui não é um sim/não, e sim uma
**probabilidade de default** mais uma **faixa de risco** (BAIXO / MÉDIO / ALTO)
que a área de crédito consegue traduzir em aprovar, negar, ajustar juros ou
reduzir limite. Público-alvo: bancos e fintechs de crédito.

## Stack
Python 3.12 · Pandas/NumPy · scikit-learn · XGBoost · CatBoost ·
imbalanced-learn (SMOTE) · SHAP · MLflow · FastAPI · Streamlit ·
SQLAlchemy/PostgreSQL · Docker · pytest

## Arquitetura
```
 generate_synthetic ─▶ ETL ─▶ PostgreSQL / parquet
   (12.000 solicitações,      │              │
    21% default)              ▼              ▼
          Feature Engineering ─▶ Split treino/teste estratificado
          (income_per_credit_line,   │
           installment_pressure,     │   SMOTE ─▶ XGBoost · CatBoost
           has_delinquency,          │   (dentro do ImbPipeline:
           one-hot + scaling)        │    só reamostra no fit)
                                     │        └─▶ MLflow (ROC-AUC/PR-AUC/KS/recall)
                                     ▼
                        melhor modelo por PR-AUC ─▶ SHAP (explicabilidade)
                                     │
                ┌────────────────────┴────────────────────┐
                ▼                                          ▼
          API FastAPI  ◀── /predict ──   Dashboard Streamlit
     (/health /predict)                  (KPIs · EDA · simulador)
```
Detalhes e trade-offs em [`reports/ARQUITETURA.md`](reports/ARQUITETURA.md) e
[`reports/RELATORIO_TECNICO.md`](reports/RELATORIO_TECNICO.md).

## Resultados

Base de **12.000 solicitações com 20,97% de default** — desbalanceada de
propósito, como é o caso real em crédito. Avaliação em holdout estratificado.
Métrica principal: **PR-AUC**, acompanhada do **KS** (separação entre bons e maus
pagadores), a estatística que a indústria de crédito de fato usa. Acurácia é
ignorada: aprovar todo mundo já "acerta" 79%.

| Modelo | ROC-AUC | PR-AUC | KS | Recall (default) | F1 (default) |
|---|---|---|---|---|---|
| **CatBoost** ✅ | **0.865** | **0.675** | **0.589** | 0.573 | **0.624** |
| XGBoost | 0.863 | 0.665 | 0.578 | 0.579 | 0.621 |

**CatBoost venceu, mas por margem estreita** — 0.002 de ROC-AUC e 0.010 de PR-AUC.
Honestamente: os dois modelos são equivalentes neste dado, e a escolha por CatBoost
se justifica mais pelo tratamento nativo de variáveis categóricas (menos
pré-processamento a manter em produção) do que pela diferença numérica. Um **KS de
0,59** indica boa separação entre as populações. O **recall de 0,57 na classe
default** é o número que a área de crédito discutiria: no limiar padrão de 0.5, o
modelo captura pouco mais da metade dos inadimplentes — em produção esse limiar
seria calibrado pelo custo real (perda de um default vs. margem de uma recusa),
não deixado em 0.5.

### Explicabilidade
O SHAP ([`reports/shap_summary.png`](reports/shap_summary.png)) confirma os fatores
esperados: **comprometimento de renda, endividamento e atrasos passados** empurram
o risco para cima; **renda e histórico de crédito longo** puxam para baixo. Essa
coerência não é detalhe estético — em crédito, explicabilidade é **exigência
regulatória**: é preciso justificar a negativa ao solicitante.

## Decisões de projeto
- **SMOTE dentro do `Pipeline` do imbalanced-learn**, nunca antes do split. Assim
  ele só reamostra durante o `fit`; no teste e na inferência é ignorado. Aplicar
  SMOTE no dataset inteiro é o erro clássico que infla toda métrica.
- **PR-AUC e KS em vez de acurácia**, porque a classe de interesse é a minoritária.
- **Pipeline inteiro serializado** (pré-processador + modelo), então a mesma
  transformação vale no treino e na API — sem divergência treino/inferência.
- **Faixas de risco** (BAIXO < 0.10 ≤ MÉDIO < 0.30 ≤ ALTO) para entregar decisão,
  não só número.

## Como executar

### Local (venv)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data.generate_synthetic --n 12000 --seed 42   # 1) gera a base
python -m src.data.etl                                      # 2) limpa e persiste
python -m src.models.train                                  # 3) treina e compara
python -m src.models.explain                                # 4) gera o SHAP

uvicorn app.main:app --reload              # API em http://localhost:8000/docs
streamlit run app/dashboard.py             # dashboard em http://localhost:8501
pytest -q                                  # testes
```

### Docker
```bash
docker compose up --build    # sobe Postgres + API (8000) + dashboard (8501)
```
> Treine o modelo antes (`python -m src.models.train`) para gerar `models/model.joblib`.

## Exemplo de uso da API
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age": 35, "income": 60000, "employment_length": 5,
  "home_ownership": "RENT", "loan_intent": "PERSONAL",
  "loan_amount": 12000, "interest_rate": 11.5,
  "loan_percent_income": 0.2, "debt_to_income": 0.35,
  "credit_history_length": 8, "num_credit_lines": 4, "past_delinquencies": 0
}'
# → {"default_probability": ..., "risk_band": "MEDIO", "will_default": false}
```
`GET /health` informa o status do serviço e se o modelo foi carregado; se o
`model.joblib` não existir, a API responde **503 com mensagem acionável** em vez
de estourar um erro genérico.

## Estrutura
```
src/
  data/       generate_synthetic.py · etl.py
  features/   build_features.py
  models/     train.py · explain.py
  config.py
app/          main.py (FastAPI) · dashboard.py (Streamlit)
tests/        5 arquivos de teste (dados, ETL, features, modelo, API)
reports/      ARQUITETURA.md · RELATORIO_TECNICO.md · metrics.json · shap_summary.png
```

## Melhorias futuras
- **Calibrar o limiar por custo de negócio** em vez de usar 0.5 — o ganho prático
  mais imediato, dado o recall de 0,57.
- Validar o pipeline contra uma **base pública real** (German Credit, Give Me Some
  Credit) para ter métricas comparáveis.
- **Monitoramento de drift** e recalibração periódica.
- Otimização de hiperparâmetros (Optuna) e **calibração de probabilidades**
  (Platt/isotônica) — importante quando a probabilidade vira preço de juros.

---
Projeto **02** do portfólio de Data Science Pleno.
