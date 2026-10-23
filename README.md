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