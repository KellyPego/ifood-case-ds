# Case Técnico de Data Science - iFood

Este repositório contém a solução para o case técnico de Data Science proposto pelo iFood. O objetivo é desenvolver um modelo de propensão para prever a probabilidade de um cliente completar uma oferta, viabilizando segmentação baseada em dados e otimização do orçamento de campanha. A abordagem combina análise exploratória, engenharia de features point-in-time e LightGBM calibrado para ranquear clientes e simular ROI sob restrições de budget.

## 🗂️ Visualização dos Notebooks

Os notebooks estão versionados sem outputs. Para visualizar a análise completa com todos os gráficos e tabelas, baixe os exports em HTML:

- [`notebooks/1_data_processing.html`](notebooks/1_data_processing.html) — EDA e pipeline de dados
- [`notebooks/2_modeling.html`](notebooks/2_modeling.html) — Feature engineering, treinamento e avaliação

Baixe o arquivo e abra em qualquer navegador — sem necessidade de configurar ambiente.

## 🚀 Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir clareza e reprodutibilidade:

```
ifood-case/
├── data/
│   └── raw/                      # offers.json, profile.json, transactions.json
├── notebooks/
│   ├── 1_data_processing.ipynb   # EDA + pipeline de dados
│   └── 2_modeling.ipynb          # Feature engineering + modelo + avaliação
├── src/
│   └── ifood_case/               # Pacote Python com o código fonte modularizado
│       ├── config.py
│       ├── schemas.py
│       ├── data_quality.py
│       ├── data_processing.py
│       ├── feature_engineering.py
│       ├── eda.py
│       ├── utils.py
│       ├── model_trainer.py
│       └── evaluator.py
├── case/
│   └── Case_Técnico_Data_Science_iFood.pdf
├── requirements.txt
└── README.md
```

## 🛠️ Stack de Tecnologias

* **Processamento e Análise:** PySpark, Pandas, NumPy (Databricks Serverless)
* **Visualização de Dados:** Matplotlib, Seaborn
* **Modelagem e Machine Learning:** Scikit-learn, LightGBM
* **Otimização de Hiperparâmetros:** Optuna
* **Interpretabilidade do Modelo:** SHAP

## ⚙️ Ambiente — Databricks (Serverless)

Este projeto roda no **Databricks Free Tier** com Unity Catalog e compute Serverless.

> Acesso ao filesystem local e ao DBFS root público (`/FileStore`) estão desabilitados neste ambiente. Todos os dados são armazenados em Unity Catalog Volumes. Todas as tabelas processadas são salvas como Delta tables no schema `default`.

### Passo a Passo

1. **Crie um Unity Catalog Volume**

   No Databricks, acesse **Catalog → workspace → default → Create → Volume** e nomeie como `ifood_case_raw`. Isso cria o caminho:
   ```
   /Volumes/workspace/default/ifood_case_raw/
   ```

2. **Faça upload dos arquivos brutos**

   Envie os três arquivos de `data/raw/` para o volume criado:
   ```
   /Volumes/workspace/default/ifood_case_raw/
   ├── offers.json
   ├── profile.json
   └── transactions.json
   ```

3. **Conecte o repositório**

   Em **Workspace → Git → Clone Repository**, aponte para este repositório. O caminho do `src/` já está configurado de forma relativa no topo de cada notebook:
   ```python
   sys.path.insert(0, os.path.abspath("../src"))
   ```
   Nenhuma alteração necessária.

4. **Instale as dependências**

   Execute a célula de instalação no topo de `2_modeling.ipynb` (comentada por padrão):
   ```python
   %pip install lightgbm shap scikit-learn optuna optuna-integration[sklearn]
   ```
   > PySpark, pandas, numpy, matplotlib e seaborn já vêm pré-instalados nos clusters Databricks.

## 🚀 Como Reproduzir os Experimentos

Execute os notebooks na seguinte ordem:

1. **`notebooks/1_data_processing.ipynb`** — carrega os dados brutos, executa EDA completa e salva as Delta tables processadas.
2. **`notebooks/2_modeling.ipynb`** — carrega as Delta tables, constrói a ABT, treina o modelo com tuning via Optuna e avalia com métricas técnicas e de negócio.

## 📊 Resultados

| Métrica | Valor |
|--------|-------|
| ROC-AUC | 0,8527 (+0,35 sobre baseline) |
| PR-AUC (principal) | 0,8593 (+0,30 sobre baseline) |
| KS Statistic | 0,548 no threshold 0,663 |
| Threshold ótimo | 0,06 (maximização de lucro, calibration set) |
| Top features SHAP | `customer_tenure_days`, `avg_ticket_before` |
| Uplift financeiro | +BRL 2.120 (+0,9%) vs. enviar-para-todos |
| Oportunidades excluídas | 3.494 de 33.988 (10,3%) com perda mínima de conversão |
| Simulação de orçamento (BRL 10k) | 7.852 conversões — 78,5% precisão — ROI 13,26× |
| Decis 1–3 | 47% de todas as conversões segmentando 30% da base |
