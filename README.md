# Tech Challenge Fase 3 — MLOps & Produção

Sistema de triagem automática de laudos médicos e resumos clínicos (NLP) desenvolvido para o Tech Challenge Fase 3.

---

## 🏗️ Etapa 1: Decisão Arquitetural e Modelagem

### 1. Estratégia de Inferência e Arquitetura Cloud (AWS)
Para a triagem clínica, adotamos a **Inferência Real-Time (Online)** via API REST síncrona. O processamento em *Batch* foi descartado pois introduziria atrasos inaceitáveis para casos classificados como *Urgente*.
* **Serviço de Inferência:** AWS ECS com AWS Fargate para escalabilidade serverless.
* **Armazenamento de Artefactos:** Amazon S3 para versionamento do modelo (`.joblib`) e dados históricos.
* **Orquestração e Monitorização:** Amazon MWAA (Airflow) para retreino periódico e Prometheus/Grafana para observabilidade.

### 2. Decisões de Modelagem (NLP)
* **Dataset:** Medical Abstracts TC Corpus.
* **Filtro Single-Label:** Remoção de abstracts multi-rótulo conflitantes para evitar ruído e vazamento de dados entre treino e teste (6.600+ amostras limpas).
* **Modelo Escolhido:** Pipeline com **TF-IDF Vectorizer + Linear SVM**.
* **Métricas de Qualidade no Teste:** Acurácia: `81.79%` | Macro-F1: `0.8149`.

---

## 🔁 Etapa 2: CI/CD e Pipeline de Treino Automatizado

### 1. CI/CD (GitHub Actions)
O workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) roda a cada push e pull request, em três jobs encadeados:

| Job | O que verifica |
|---|---|
| `lint` | `ruff check` e `ruff format --check` em todo o código |
| `test` | `pytest` (17 testes): filtro single-label, treino, exportação ONNX (concordância e locale), rotas da API, validação de payload, métricas Prometheus e integridade da DAG |
| `build` | Só roda se `lint` e `test` passarem: treina o modelo executando a DAG do Airflow, constrói a imagem Docker com esses artefatos, sobe o container e roda [`scripts/smoke_test.py`](scripts/smoke_test.py) contra ele |

Os testes usam um corpus sintético e diretórios temporários, então rodam em segundos e sem rede. Localmente: `uv sync --group airflow && uv run pytest`.

### 2. DAG de Treino (Airflow)
A DAG [`dags/medical_triage_training_dag.py`](dags/medical_triage_training_dag.py) (`medical_triage_training`, agendamento semanal) orquestra o retreino chamando as mesmas funções usadas pela linha de comando (`src.data.io`, `src.models.train`, `src.models.convert_to_onnx`):

```text
ingest_data → train_model → quality_gate → export_onnx → promote_model
```

| Task | O que faz |
|---|---|
| `ingest_data` | Baixa o corpus, aplica o filtro single-label e grava `data/processed/train.csv` e `test.csv` |
| `train_model` | Lê os CSVs, treina TF-IDF + LogisticRegression, registra o run `airflow_retrain` no MLflow e salva o candidato em `models/candidate/` |
| `quality_gate` | Reprova a execução (sem retry) se o Macro-F1 no teste for menor que `0.70` |
| `export_onnx` | Converte o candidato para ONNX e exige ≥ 97% de concordância com o scikit-learn no conjunto de teste |
| `promote_model` | Move os artefatos aprovados para `models/linear_svm_baseline.joblib` e `models/model_optimized.onnx`, que a API carrega |

* **Modelo servido protegido:** os artefatos da API só são substituídos se o candidato passar pelas duas travas; uma execução reprovada mantém a versão atual.
* **Execuções em série:** `max_active_runs=1` impede que um disparo manual e o agendado disputem os mesmos arquivos.
* **Mesmo ambiente da API:** o Airflow roda no ambiente `uv` do projeto (grupo `airflow`), com as mesmas versões de scikit-learn e skl2onnx da imagem Docker, então o modelo retreinado carrega na API sem incompatibilidade de versão.

**Como executar** (Linux, macOS ou WSL — o Airflow não roda nativamente no Windows):
```bash
uv sync --group airflow
./scripts/airflow.sh dags test medical_triage_training   # execução completa, sem scheduler (~30 s)
./scripts/airflow.sh standalone                          # UI em http://localhost:8080
```
No modo `standalone`, o usuário é `admin` e a senha fica em `.airflow/simple_auth_manager_passwords.json.generated`. Como a API carrega o modelo na inicialização, após uma execução bem-sucedida reconstrua o container para servir a nova versão: `docker compose up -d --build api`.

---

## 📈 Etapa 3: Monitoramento e Observabilidade (Dashboard as Code)

A API foi instrumentada com o `prometheus_client` expondo a rota `/metrics`. A infraestrutura de observabilidade foi implementada seguindo a prática de **Dashboard as Code (Provisioning)**:
* **Automação:** O Grafana é provisionado automaticamente ao subir os containers, com o Datasource do Prometheus e o Dashboard já mapeados via volumes.
* **Métricas Obrigatórias:** O dashboard padrão conta com os três painéis exigidos:
  1. Total de Requisições.
  2. Latência P95 (Tempo de resposta em ms).
  3. Taxa de Erros HTTP 5xx (%).

---

## 🚀 Como Executar o Projeto (Guia do Avaliador)

### Pré-requisitos
Certifique-se de ter o [uv](https://github.com/astral-sh/uv) e o **Docker Desktop** (com docker-compose) instalados.

### 1. Instalar Dependências e Treinar o Modelo Baseline
Este comando fará o download dos dados, treinará o modelo SVM, registrará no MLflow e salvará o artefato:
```bash
uv sync
uv run python -m src.models.train