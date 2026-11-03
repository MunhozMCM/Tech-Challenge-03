# Tech Challenge Fase 3 — MLOps & Produção

Sistema de triagem automática de laudos médicos e resumos clínicos (NLP). Focado em inferência Real-Time, pipeline de retreino orquestrado e observabilidade da API em produção.

## 🚀 Tecnologias Utilizadas
- **Python 3.12+**
- **uv**: Gerenciador de dependências ultrarrápido.
- **FastAPI**: API REST síncrona para inferência em tempo real.
- **scikit-learn & ONNX Runtime**: Pipeline de NLP (TF-IDF + Linear SVM) e otimização de latência na inferência.
- **Apache Airflow**: Orquestração do pipeline de retreino e aprovação do modelo (Quality Gate).
- **Prometheus & Grafana**: Monitoramento de latência, tráfego e erros (Dashboard as Code).
- **Docker**: Containerização de toda a infraestrutura de servimento e monitoramento.

## 📂 Estrutura do Projeto

```text
├── .github/workflows/       # Pipeline de CI/CD (lint, test, build automático)
├── dags/                    # DAGs do Airflow (pipeline de retreino)
├── data/                    # Datasets (Medical Abstracts TC Corpus)
├── models/                  # Artefatos aprovados (.joblib e .onnx) consumidos pela API
├── notebooks/               # Logs de decisão arquitetural e EDA
├── scripts/                 # Scripts auxiliares (Airflow, smoke tests)
├── src/
│   ├── api/                 # Servidor FastAPI, gerador de tráfego e benchmark
│   ├── data/                # Ingestão e processamento de dados
│   └── models/              # Lógica de treinamento e exportação para ONNX
├── tests/                   # Testes unitários e de integração (pytest)
└── docker-compose.yml       # Infraestrutura da API + Grafana + Prometheus
```

## ⚙️ Como Executar o Pipeline de Treino (Local)

### 1. Pré-requisitos
Instale o `uv` no seu sistema. Se for executar o Airflow, certifique-se de estar em um ambiente Linux, macOS ou WSL (Windows Subsystem for Linux).

### 2. Instalação das dependências
Com o `uv` instalado, rode o comando abaixo na raiz do projeto para criar o ambiente (incluindo dependências do Airflow):
```bash
uv sync --group airflow
```

### 3. Executando o Pipeline de Treino (Airflow)
A DAG de treino fará o download dos dados, treinará o modelo, aplicará as regras de validação (Quality Gate) e exportará a versão otimizada em ONNX:
```bash
./scripts/airflow.sh dags test medical_triage_training
```
> **Nota para usuários de Windows (sem WSL):** O Airflow não roda nativamente no Windows. Para treinar o modelo ignorando o Airflow, rode os scripts diretamente:
> ```powershell
> uv run python -m src.models.train ; uv run python -m src.models.convert_to_onnx
> ```

## 🐳 Como Executar via Docker (API & Monitoramento)

Com o modelo treinado na pasta `models/`, inicialize a API e a stack de observabilidade:
```bash
docker compose up -d --build
```
Isso irá inicializar 3 serviços locais:
1. **API REST (FastAPI):** Disponível na porta **8000** (Documentação interativa em `http://localhost:8000/docs`).
2. **Grafana (Dashboard as Code):** Disponível na porta **3000** (Acesse `http://localhost:3000/d/triage-api` - login opcional `admin`/`admin`).
3. **Prometheus:** Disponível na porta **9090** (Coletando dados silenciosamente).

### Simulando Tráfego e Validando
Para popular o dashboard do Grafana com dados em tempo real, abra um novo terminal e rode o gerador de tráfego (ele envia ~10 requisições/segundo por 5 minutos):
```bash
uv run python -m src.api.traffic
```
Para comparar a diferença de latência entre as rotas scikit-learn e ONNX:
```bash
uv run python -m src.api.benchmark
```

## ✅ Testes e Linting
Para rodar a verificação de Clean Code (Ruff) e a suíte completa de testes (pytest):
```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
