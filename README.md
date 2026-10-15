# Tech Challenge Fase 3 — MLOps & Produção

Sistema de triagem automática de laudos médicos e resumos clínicos (NLP) desenvolvido para o Tech Challenge Fase 3[cite: 2].

---

## 🏗️ Etapa 1: Decisão Arquitetural e API Inicial

### 1. Estratégia de Inferência: Real-Time vs. Batch
Para o cenário de triagem clínica de laudos médicos, a estratégia escolhida é a **Inferência Real-Time (Online)** via API REST síncrona.
* **Justificativa Clínica:** Pacientes em estado crítico necessitam de priorização imediata assim que o laudo é emitido. O processamento em lote (*Batch*) introduziria um atraso inaceitável para casos classificados como *Urgente*[cite: 2].
* **Requisito Técnico:** Utilização de um modelo leve de NLP com baixa latência empacotado em contentor Docker para respostas em milissegundos[cite: 2].

### 2. Proposta de Arquitetura na Nuvem (AWS)
* **Serviço de Inferência:** AWS ECS com AWS Fargate para contentores serverless escaláveis[cite: 4].
* **Armazenamento de Artefactos:** Amazon S3 para versionamento do modelo (`.joblib`) e dados históricos[cite: 4].
* **Orquestração e Monitorização:** Amazon MWAA (Airflow) para retreino e Prometheus/Grafana para observabilidade[cite: 4].

---

## 📊 Decisões de Modelagem e NLP (Decision Summary)
* **Dataset:** Medical Abstracts TC Corpus.
* **Filtro Single-Label:** Remoção de abstracts multi-rótulo conflitantes para evitar ruído e vazamento de dados entre treino e teste (6.600+ amostras limpas).
* **Modelo Escolhido:** Pipeline com **TF-IDF Vectorizer + Linear SVM**.
* **Métricas de Qualidade no Teste:**
  * **Acurácia:** `81.79%`
  * **Macro-F1:** `0.8149`

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o [uv](https://github.com/astral-sh/uv) e o Docker instalados.

### 1. Instalar Dependências e Treinar o Modelo Baseline
```bash
uv sync
uv run python -m src.models.train