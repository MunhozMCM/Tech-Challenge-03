# Tech Challenge Fase 3 — MLOps & Produção

Sistema de triagem automática de laudos médicos e resumos clínicos (NLP) desenvolvido para o Tech Challenge Fase 3.

---

## 🏗️ Etapa 1: Decisão Arquitetural e Modelagem

### 1. Estratégia de Inferência e Arquitetura Cloud (AWS)
Para a triagem clínica, adotamos a **Inferência Real-Time (Online)** via API REST síncrona. O processamento em *Batch* foi descartado pois introduziria atrasos inaceitáveis para casos classificados como *Urgente*.
* **Serviço de Inferência:** AWS ECS com AWS Fargate para escalabilidade serverless.
* **Armazenamento de Artefactos:** Amazon S3 para versionamento do modelo (`.joblib`) e dados históricos.
* **Orquestração e Monitorização:** Amazon MWAA (Airflow) para retreino periódico e Prometheus/Grafana para observabilidade.
* **Baseline de latência local** (API no Docker, rota `/predict` com scikit-learn): `3,2 ms` de tempo de resposta (p50) e `3,6 ms` (p95), dos quais `1,5 ms` são o modelo. Medição e comparação com o ONNX na [Etapa 4](#-etapa-4-otimização-de-latência-onnx).

### 2. Decisões de Modelagem (NLP)
* **Dataset:** Medical Abstracts TC Corpus.
* **Filtro Single-Label:** Remoção de abstracts multi-rótulo conflitantes para evitar ruído e vazamento de dados entre treino e teste (6.600+ amostras limpas).
* **Modelo em Produção:** Pipeline com **TF-IDF Vectorizer (unigramas + bigramas) + Linear SVM calibrado** (`CalibratedClassifierCV`, sigmoide). O TF-IDF é a etapa de NLP (tokenização, stop words, n-gramas e ponderação dos termos); o SVM classifica o vetor resultante.
* **Métricas de Qualidade no Teste:** Acurácia: `82.91%` | Macro-F1: `0.8279` (validação cruzada no treino: `0.805`).
* **Por que calibrado:** o Linear SVM foi o melhor classificador no notebook, mas o skl2onnx 1.20 não converte um `LinearSVC` puro dentro de um Pipeline, e a Etapa 4 depende do ONNX. Envolvido na calibração, ele converte, ganha probabilidades por classe e mantém o mesmo tamanho (5,6 MB). A versão intermediária com Regressão Logística ficava em Macro-F1 `0.7443`. A análise completa está em [`notebooks/NLP_experiments_decisions.md`](notebooks/NLP_experiments_decisions.md).

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
| `train_model` | Lê os CSVs, treina TF-IDF + Linear SVM calibrado, registra o run `airflow_retrain` no MLflow e salva o candidato em `models/candidate/` |
| `quality_gate` | Reprova a execução (sem retry) se o Macro-F1 no teste for menor que `0.78` (o modelo atual faz `0.828`; a versão antiga com Regressão Logística, `0.744`, seria barrada) |
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
* **Métricas Obrigatórias:** O dashboard [`API de Triagem Médica`](http://localhost:3000/d/triage-api) conta com os três painéis exigidos, todos restritos às rotas de triagem (sem os scrapes do Prometheus em `/metrics` nem o `/health`):
  1. Total de Requisições de Triagem.
  2. Latência P95 por Rota (ms) — uma linha para `/predict` (scikit-learn) e outra para `/predict/onnx` (ONNX Runtime). O histograma usa buckets de 0,5 ms até 5 ms, pois as respostas levam de 1 a 4 ms (com os buckets padrão, que começam em 5 ms, o P95 não distinguiria as rotas).
  3. Taxa de Erros HTTP (%) — erros 5xx (falha do servidor) e 4xx (payload inválido); exibe 0% quando não há erros.
* **Tráfego para o dashboard:** `uv run python -m src.api.traffic` envia ~10 req/s por 5 minutos, alternando as duas rotas com abstracts reais de teste e ~5% de payloads inválidos (HTTP 422), o que alimenta os três painéis. O P95 só aparece enquanto chegam requisições, porque é calculado sobre janelas de 1 minuto.

---

## ⚡ Etapa 4: Otimização de Latência (ONNX)

* **Técnica:** o pipeline TF-IDF + Linear SVM calibrado é exportado para **ONNX** ([`src/models/convert_to_onnx.py`](src/models/convert_to_onnx.py)) e servido pelo **ONNX Runtime** na rota `/predict/onnx`, ao lado da rota original `/predict` (scikit-learn).
* **Qualidade preservada:** o modelo ONNX concorda com o scikit-learn em 98,3% dos abstracts de teste. As divergências são quase-empates entre duas classes (TF-IDF em float32), e o Macro-F1 do ONNX é `0.833`, contra `0.828` do original. A DAG reprova a publicação se a concordância cair abaixo de 97%.
* **Sessão com uma thread:** a API cria a sessão do ONNX Runtime com `intra_op_num_threads=1`. Com o pool padrão (uma thread por núcleo), as threads ficam em espera ativa após cada inferência e disputam CPU com o servidor. Com requisições espaçadas em 100 ms, o p95 HTTP da rota ONNX era de 11,0 ms, contra 5,5 ms do scikit-learn; com uma thread, caiu para 3,9 ms.
* **Latência** (`uv run python -m src.api.benchmark`: 1.000 abstracts de teste por rota, com as rotas intercaladas para que a variação da máquina afete as duas igualmente; API no Docker local):

| Rota | Modelo p50 | Modelo p95 | HTTP p50 | HTTP p95 |
|---|---|---|---|---|
| `/predict` (scikit-learn) | 1,53 ms | 1,90 ms | 3,18 ms | 3,55 ms |
| `/predict/onnx` (ONNX Runtime) | **0,54 ms** | **0,67 ms** | **2,14 ms** | **2,38 ms** |
| Ganho do ONNX | **−65%** | **−65%** | **−33%** | **−33%** |

O tempo de inferência do modelo cai a cerca de um terço. No tempo de resposta completo (HTTP) o ganho é menor porque cerca de 1,6 ms de cada requisição vem da rede e do FastAPI, e não do modelo.

---

## 🎬 Vídeo de Apresentação (STAR)

🔗 _Link a ser adicionado._

---

## 🚀 Como Executar o Projeto (Guia do Avaliador)

### Pré-requisitos
* [uv](https://github.com/astral-sh/uv) e **Docker Desktop** (com Docker Compose).
* Linux, macOS ou WSL para o Airflow (ele não roda nativamente no Windows).

### 1. Instalar as dependências
```bash
uv sync --group airflow
```

### 2. Treinar o modelo (Etapa 2)
A DAG baixa os dados, treina, aplica as travas de qualidade, exporta para ONNX e publica os artefatos em `models/`:
```bash
./scripts/airflow.sh dags test medical_triage_training
```
Alternativa sem Airflow: `uv run python -m src.models.train && uv run python -m src.models.convert_to_onnx`.

### 3. Subir API + Prometheus + Grafana (Etapas 1 e 3)
```bash
docker compose up -d --build
```

#### 🌐 Endereçamento e Portas dos Serviços

Os serviços podem ser validados tanto **localmente via Docker Compose** quanto pelos **links públicos ao vivo** (expostos via túnel seguro Cloudflare com HTTPS):

| Serviço | Porta Local | Endereço Local (Docker) | Endereço Público Online | Descrição & Observações |
|---|:---:|---|---|---|
| **API REST (FastAPI / Swagger)** | `8000` | [http://localhost:8000](http://localhost:8000) | [https://api.lucasmunhoz.dev.br](https://api.lucasmunhoz.dev.br) | Documentação interativa Swagger (redireciona automaticamente da raiz `/` para `/docs`). |
| **Health Check da API** | `8000` | [http://localhost:8000/health](http://localhost:8000/health) | [https://api.lucasmunhoz.dev.br/health](https://api.lucasmunhoz.dev.br/health) | Retorna status de prontidão e confirma que os modelos scikit-learn e ONNX estão carregados. |
| **Grafana (Dashboard de Triagem)** | `3000` | [http://localhost:3000/d/triage-api](http://localhost:3000/d/triage-api) | [https://grafana.lucasmunhoz.dev.br/d/triage-api](https://grafana.lucasmunhoz.dev.br/d/triage-api) | Dashboard com os 3 painéis provisionados. Acesso anônimo ativo (abre sem exigir senha; login opcional `admin`/`admin`). |
| **Prometheus (Métricas brutas)** | `9090` | [http://localhost:9090](http://localhost:9090) | *Rede interna Docker* | Coleta métricas da rota `/metrics` da API a cada 5 segundos. |
| **Airflow UI (Orquestração de Retreino)** | `8080` | [http://localhost:8080](http://localhost:8080) | *Execução local sob demanda* | Iniciar com `./scripts/airflow.sh standalone` (credenciais em `.airflow/simple_auth_manager_passwords.json.generated`). |

### 4. Validar a API e gerar tráfego (Etapas 1, 3 e 4)
```bash
uv run python scripts/smoke_test.py    # health, as duas rotas de predição, validação e /metrics
uv run python -m src.api.benchmark     # latência scikit-learn × ONNX (tabela da Etapa 4)
uv run python -m src.api.traffic       # ~10 req/s por 5 min para ver o dashboard ao vivo
```
Exemplo de requisição:
```bash
curl -X POST http://localhost:8000/predict/onnx -H "Content-Type: application/json" \
  -d '{"medical_abstract": "Acute myocardial infarction with severe left ventricular dysfunction."}'
```

### 5. Testes e lint (o mesmo que o CI executa)
```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Para encerrar a stack: `docker compose down`.
