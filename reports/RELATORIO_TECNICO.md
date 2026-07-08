# Relatório Técnico — Credit Risk Scoring System

## 1. Problema de negócio
Instituições financeiras e fintechs precisam decidir a quem conceder crédito.
Um erro caro: aprovar quem vai inadimplir (perda direta) ou negar bom pagador
(receita perdida). Este sistema estima a **probabilidade de default** de uma
solicitação, permitindo decisões baseadas em risco (aprovar, negar, ajustar juros
ou limite).

## 2. Dados
Base **sintética** gerada por um modelo logístico latente (`generate_synthetic.py`),
com ~21% de default — deliberadamente **desbalanceada**, refletindo a realidade do
crédito. Features: renda, idade, tempo de emprego, moradia, finalidade, valor e
juros do empréstimo, comprometimento de renda, endividamento, histórico e atrasos.

## 3. Pipeline
`ETL → validação → feature engineering → SMOTE (só no treino) → modelo`.
O SMOTE fica **dentro de um Pipeline do imbalanced-learn**, garantindo que o
reamostramento ocorra apenas no treino — evitando o vazamento de dados que
inflaria as métricas.

Features derivadas com sentido de risco: `income_per_credit_line`,
`installment_pressure` (juros × comprometimento de renda) e `has_delinquency`.

## 4. Modelagem e avaliação
Comparamos **XGBoost** e **CatBoost**. Por ser base desbalanceada, priorizamos
**PR-AUC, KS e recall na classe default** em vez de acurácia.

| Modelo | ROC-AUC | PR-AUC | KS | Recall (default) | F1 (default) |
|---|---|---|---|---|---|
| XGBoost | 0.863 | 0.665 | 0.578 | 0.579 | 0.621 |
| **CatBoost** ✅ | **0.865** | **0.675** | **0.589** | 0.573 | **0.624** |

**Campeão: CatBoost** (maior PR-AUC). Todo o experimento é rastreado no **MLflow**.

## 5. Interpretabilidade (SHAP)
O `shap_summary.png` confirma os fatores de risco esperados: comprometimento de
renda, endividamento e atrasos passados puxam o risco para cima; renda e histórico
de crédito longos puxam para baixo. Coerência entre o que o modelo aprendeu e a
intuição de negócio — essencial em crédito (explicabilidade regulatória).

## 6. Deploy
Modelo servido via **FastAPI** (`/predict` retorna probabilidade + faixa de risco
BAIXO/MÉDIO/ALTO) e **dashboard Streamlit** com KPIs, EDA e simulador. Empacotado
em **Docker** (`docker-compose` sobe API + PostgreSQL + dashboard).

## 7. Trade-offs e melhorias futuras
- **Dados sintéticos** permitem portfólio reprodutível, mas não capturam toda a
  complexidade real; próximo passo seria validar com uma base pública (ex.: German
  Credit, Give Me Some Credit).
- **Threshold de 0.5** é um ponto de partida; em produção seria calibrado por
  custo de negócio (custo de um default vs. de uma recusa).
- Adicionar **monitoramento de drift** e **recalibração** periódica.
- Otimização de hiperparâmetros (Optuna) e calibração de probabilidades.
