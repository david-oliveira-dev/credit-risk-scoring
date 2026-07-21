# Relatório Técnico — Credit Risk Scoring System

## 1. Problema de negócio
Instituições financeiras e fintechs precisam antecipar quem vai deixar de pagar.
O erro é assimétrico: não identificar um inadimplente vira perda direta do
principal; barrar um bom pagador vira receita perdida. Este sistema estima a
**probabilidade de default** no mês seguinte, permitindo decisões baseadas em
risco (aprovar, revisar limite, suspender).

## 2. Dados
**Default of Credit Card Clients** (Yeh & Lien, 2009), UCI Machine Learning
Repository: **30.000 clientes** de cartão de crédito de Taiwan, com histórico de
abril a setembro de 2005 e o desfecho no mês seguinte. Após limpeza:
**29.965 clientes, 22,13% de inadimplência** — desbalanceada, como é o caso real.

Variáveis: limite, sexo, escolaridade, estado civil, idade, e para cada um dos 6
meses o status de pagamento (`pay_*`), o valor da fatura (`bill_amt*`) e o valor
pago (`pay_amt*`).

### 2.1 Qualidade do dado
O arquivo contradiz a documentação oficial em três pontos, tratados no ETL:

| Problema | Registros | Decisão |
|---|---:|---|
| `education` com códigos 0, 5, 6 (doc: 1–4) | 345 | Colapsados em 4 (outros) |
| `marriage` com código 0 (doc: 1–3) | 54 | Colapsado em 3 (outros) |
| Duplicatas exatas | 35 | Removidas |
| `bill_amt*` negativo | 590 | Mantidos (saldo a favor) |
| `pay_*` com códigos -2 e 0 (não documentados) | ~17.500 | Mantidos — ver §3 |

## 3. Feature engineering: o achado que a definiu
Medindo a inadimplência por código de `pay_1`, a relação **não é monótona**:

| Código | Leitura | Clientes | Default |
|---|---|---:|---:|
| -2 | sem consumo | 2.750 | 13,2% |
| -1 | pagou a fatura inteira | 5.682 | **16,8%** |
| 0 | crédito rotativo | 14.737 | **12,8%** |
| 1 | 1 mês de atraso | 3.667 | 34,0% |
| 2 | 2 meses de atraso | 2.666 | 69,1% |

Quem quita a fatura toda inadimple mais do que quem rola no rotativo. Usar a
coluna como inteiro numa escala linear ensinaria ao modelo uma ordem inexistente.

A solução foi **contar regimes**: `meses_em_atraso`, `meses_rotativo` e
`meses_sem_consumo`. A primeira é a feature mais correlacionada com o alvo
(**0,398**) e é monótona: de **11,7%** de default com zero atrasos a **70,3%**
com seis.

Demais derivadas, todas normalizadas pelo limite (valores absolutos não são
comparáveis entre clientes): utilização do limite (última, média e máxima), taxa
de pagamento da fatura anterior e tendência da dívida nos 6 meses.

## 4. Pipeline
`download → ETL → feature engineering → SMOTE (só no treino) → modelo`.
O SMOTE fica **dentro de um Pipeline do imbalanced-learn**, garantindo que o
reamostramento ocorra apenas no `fit` — evitando o vazamento que inflaria todas
as métricas.

## 5. Modelagem e avaliação
Comparamos **XGBoost** e **CatBoost**, em holdout estratificado de 20% (5.993
clientes). Por ser base desbalanceada, priorizamos **PR-AUC e KS**.

| Modelo | ROC-AUC | PR-AUC | KS |
|---|---:|---:|---:|
| **CatBoost** ✅ | **0.7727** | **0.5545** | **0.4191** |
| XGBoost | 0.7679 | 0.5445 | 0.3985 |

Um ROC-AUC de ~0,77 está alinhado com a literatura publicada para este dataset.
Números muito acima disso, nesta base, indicariam vazamento. Todo o experimento é
rastreado no **MLflow**.

## 6. Calibração do limiar
A diferença entre os dois modelos é de 0,009 em PR-AUC. A escolha do **limiar**
vale muito mais:

| Limiar | Recall | Precisão | F1 |
|---|---:|---:|---:|
| 0.50 (padrão) | 0.396 | 0.624 | 0.484 |
| **0.335 (calibrado por F1)** | **0.587** | 0.485 | **0.531** |

No corte padrão o modelo captura menos de 40% dos inadimplentes. O limiar é
calibrado no treino e **persistido junto do modelo** (`{pipeline, threshold}`),
para que API e dashboard não voltem ao 0.5 implícito. As faixas de risco são
ancoradas nele: BAIXO abaixo de `t/2`, ALTO a partir de `t`.

## 7. Interpretabilidade (SHAP)
O `shap_summary.png` confirma a EDA: **comportamento recente de pagamento
domina**, cadastro quase não importa (`age` correlaciona 0,014 com o alvo).

Duas variáveis aparecem como protetoras e merecem cautela: `meses_rotativo`
(−0,154) e `limit_bal` (−0,154). O limite é **endógeno** — é consequência de um
bom histórico, já que o banco concedeu confiança ao cliente. Usá-lo para decidir
crédito realimenta a política que o gerou.

## 8. Deploy
Modelo servido via **FastAPI** (`/predict` retorna probabilidade, faixa de risco
e o limiar usado) e **dashboard Streamlit** com KPIs, EDA e simulador. Empacotado
em **Docker** (`docker-compose` sobe API + PostgreSQL + dashboard). Suíte de
**32 testes** que não dependem de rede.

## 9. Trade-offs e melhorias futuras
- **Trocar o F1 por uma função de custo real** na calibração do limiar — o ganho
  prático mais imediato, já que o corte pesa mais que a escolha do modelo.
- **Tratar a endogeneidade de `limit_bal`**, avaliando o modelo sem ela para
  medir quanto do desempenho vem da política de crédito já existente.
- **Monitoramento de drift** e recalibração periódica.
- Otimização de hiperparâmetros (Optuna) e calibração de probabilidades
  (Platt/isotônica), relevante quando a probabilidade vira preço.
