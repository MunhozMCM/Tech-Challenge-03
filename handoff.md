# Handoff — Tech Challenge Fase 3: Triagem de Laudos com NLP

Material para quem vai gravar o vídeo STAR. Ele reúne as decisões de modelagem e de engenharia, com os números medidos, e o que vale dizer e mostrar em 5 minutos.

Estado de referência: `main` em `b553155` (31/10/2026). Todos os números abaixo foram medidos nesse estado.

---

## 1. O projeto em 30 segundos

Um hospital quer triar automaticamente textos clínicos em **normal / atenção / urgente**. Entregamos:

- um classificador de texto leve (**TF-IDF + Linear SVM calibrado**), exportado para **ONNX**;
- uma API **FastAPI** em **Docker**, com uma rota scikit-learn (`/predict`) e uma rota ONNX (`/predict/onnx`);
- uma **DAG do Airflow** que retreina o modelo e só o publica depois de passar por travas de qualidade;
- **CI no GitHub Actions** (lint → testes → treino + build + smoke test do container);
- **Prometheus + Grafana** com dashboard provisionado como código.

| Indicador | Valor |
|---|---|
| Qualidade (teste) | Macro-F1 **0,828**, acurácia **82,9%** |
| Triagem de urgência | **88,1%** de acerto; **85,8%** dos urgentes reconhecidos; só **0,8%** dos urgentes classificados como "normal" |
| ONNX vs. scikit-learn | modelo **−65%** de latência; resposta HTTP **−33%** |

## 2. Onde está cada coisa

| O quê | Onde |
|---|---|
| Análise de dados e comparação de modelos | `notebooks/NLP_experiments.ipynb` + `notebooks/NLP_experiments_decisions.md` (Decisões 1–5) |
| Treino / exportação ONNX | `src/models/train.py`, `src/models/convert_to_onnx.py` |
| API + métricas Prometheus | `src/api/main.py` |
| DAG de retreino | `dags/medical_triage_training_dag.py` (rodar via `scripts/airflow.sh`) |
| CI | `.github/workflows/ci.yml` |
| Dashboard | `monitoring/grafana/dashboards/triage_dashboard.json` |
| Benchmark / gerador de tráfego / smoke test | `src/api/benchmark.py`, `src/api/traffic.py`, `scripts/smoke_test.py` |
| Constantes (caminhos, travas, mapeamento de urgência) | `src/config.py` |

---

## 3. Decisões sobre o modelo

### D1 — Dataset: Medical Abstracts TC Corpus

- **Contexto:** o PDF sugere o Medical Abstracts TC Corpus, recortes do MIMIC-III ou outro dataset com texto + rótulo e ≥ 2.000 amostras. O MIMIC exige credenciamento.
- **Decisão:** usar o Medical Abstracts (Schopf et al., 2023): 14.438 linhas, split oficial treino/teste, 5 classes de condição (neoplasias, sistema digestivo, sistema nervoso, cardiovascular, condições patológicas gerais).
- **Urgência:** o dataset não tem rótulo de urgência. O grupo definiu um mapeamento em `src/config.py`:

| Condição | Urgência | Justificativa usada |
|---|---|---|
| Sistema nervoso | urgente | risco neurológico agudo |
| Cardiovascular | urgente | risco hemodinâmico/isquêmico agudo |
| Neoplasias | atenção | investigação oncológica prioritária |
| Condições patológicas gerais | atenção | avaliação clínica geral |
| Sistema digestivo | normal | acompanhamento ambulatorial |

  ⚠️ É uma **heurística do grupo**, não uma regra clínica validada. Os textos são **resumos de artigos médicos**, usados como substitutos de laudos.

### D2 — Limpeza: só abstracts com um único rótulo

- **Problema:** o corpus é multi-rótulo "achatado". Um abstract com *k* condições aparece *k* vezes, uma por rótulo.
  - Dos 11.227 abstracts únicos, 2.929 têm 2 a 4 rótulos.
  - Isso afeta 42,8% das linhas de treino e 41,4% das de teste.
  - **35% das linhas de teste repetem um texto do treino, sempre com outro rótulo.**
- **Evidência:** o modelo de produção tem **49% de acurácia** no split oficial e **83% nos dados limpos**. A queda vem dos rótulos conflitantes, não do modelo. Treinar com as linhas brutas também custa 5,6 pontos de Macro-F1 (0,772 contra 0,828).
- **Decisão:** remover todo abstract com mais de um rótulo. Sobram **6.607 de treino e 1.691 de teste**, sem nenhum texto repetido entre os dois. A proporção das classes se mantém.

### D3 — Métrica e protocolo

- **Métrica: Macro-F1.** As classes são desbalanceadas (digestivo ≈ 8%, gerais ≈ 29%), e o Macro-F1 dá peso igual a cada condição.
- **Seleção:** o modelo é escolhido por validação cruzada estratificada de 5 folds no treino. O teste serve só para reportar o resultado final.

### D4 — Representação: TF-IDF

Todos os classificadores usam o mesmo TF-IDF: minúsculas, stop words em inglês, unigramas + bigramas, termos em ≥ 2 documentos e TF sublinear. São **81.472 termos**. Isso **é** a parte de NLP: o texto vira um vetor de pesos de termos, e o classificador decide a partir desse vetor.

### D5 — Classificador: do notebook à produção

**Comparação no notebook** (dados limpos, mesmo TF-IDF):

| Modelo | Macro-F1 (teste) | Latência p50 | Tamanho |
|---|---|---|---|
| Linear SVM | **0,815** | 0,95 ms | 5,6 MB |
| Regressão Logística (`C=1`) | 0,744 | 0,62 ms | 5,6 MB |
| Random Forest (exemplo do PDF) | 0,705 | **22,8 ms** | **126 MB** |
| Naive Bayes | 0,464 | 1,06 ms | 8,8 MB |
| Dummy (classe mais frequente) | 0,088 | — | — |

O Random Forest sugerido no PDF foi **descartado**: perde em qualidade e é cerca de 24× mais lento e 23× maior que o Linear SVM. As latências variam dezenas de por cento entre rodadas; compare os modelos dentro da mesma rodada.

**Caminho até produção:**

1. O Linear SVM venceu, mas o **skl2onnx 1.20 não converte um `LinearSVC` puro** dentro de um Pipeline (bug na opção `zipmap`).
2. Para destravar o ONNX, a produção passou temporariamente para a Regressão Logística (0,744).
3. **Solução final:** o SVM envolvido em calibração, `CalibratedClassifierCV(LinearSVC, cv=5, ensemble=False)`. Ele converte para ONNX, passa a dar probabilidades e mantém o mesmo tamanho:

| Candidato | CV F1 | Teste F1 | ONNX F1 | p50 sklearn | p50 ONNX | Tamanho |
|---|---|---|---|---|---|---|
| Regressão Logística `C=1` | 0,725 | 0,744 | 0,752 | 0,69 ms | 0,47 ms | 5,6 MB |
| Regressão Logística `C=30` | 0,791 | 0,803 | 0,807 | 0,62 ms | 0,46 ms | 5,6 MB |
| SVM calibrado (ensemble) | 0,803 | 0,823 | 0,829 | 4,41 ms | 0,86 ms | 18,6 MB |
| **SVM calibrado (único) ✅** | **0,805** | **0,828** | **0,833** | 1,77 ms | 0,62 ms | 5,6 MB |

**Desempenho do modelo em produção** (teste, 1.691 abstracts):

| Classe | Precisão | Recall | F1 |
|---|---|---|---|
| Neoplasias | 0,869 | 0,898 | 0,883 |
| Sistema digestivo | 0,833 | 0,851 | 0,842 |
| Sistema nervoso | 0,817 | 0,756 | 0,785 |
| Cardiovascular | 0,874 | 0,882 | 0,878 |
| Condições gerais | 0,755 | 0,747 | 0,751 |

**No nível de urgência**, que é o que importa para a triagem: **88,1%** de acerto.

| Real \ Previsto | urgente | atenção | normal |
|---|---|---|---|
| **urgente** (633) | **543 (85,8%)** | 85 | **5 (0,8%)** |
| **atenção** (917) | 72 | 826 | 19 |
| **normal** (141) | 5 | 16 | 120 |

O erro mais grave, um caso urgente classificado como normal, acontece em 5 de 633. Os pontos fracos são "sistema nervoso" (recall 0,76) e a classe genérica "condições gerais", cujo vocabulário ("acute", "chronic", "postoperative") se sobrepõe ao das outras.

### D6 — Otimização: ONNX

- **Conversão:** o pipeline inteiro (TF-IDF + SVM calibrado) vai para ONNX e é servido pelo ONNX Runtime em `/predict/onnx`.
- **Concordância:** **98,3%** com o scikit-learn. As divergências são quase-empates entre duas classes (o TF-IDF do ONNX é calculado em float32), e o Macro-F1 do ONNX até sobe um pouco (0,833).
- **Locale:** a etapa de minúsculas do ONNX exige um locale do sistema. O `en_US.UTF-8` não existe no WSL nem em imagens slim, então exportamos com `C.UTF-8`, que existe em qualquer Linux.
- **Uma thread:** a sessão do ONNX Runtime roda com **uma thread**. Com o pool padrão, uma thread por núcleo, as threads ficam em espera ativa depois de cada inferência e disputam CPU com o servidor. O problema apareceu no dashboard: com requisições espaçadas, o p95 da rota ONNX era **11,0 ms** (pior que os 5,5 ms do sklearn). Com uma thread, caiu para **3,9 ms**.
- **Latência** (benchmark com 1.000 abstracts por rota, rotas intercaladas, API no Docker):

| Rota | Modelo p50 | Modelo p95 | HTTP p50 | HTTP p95 |
|---|---|---|---|---|
| `/predict` (scikit-learn) | 1,53 ms | 1,90 ms | 3,18 ms | 3,55 ms |
| `/predict/onnx` (ONNX Runtime) | **0,54 ms** | **0,67 ms** | **2,14 ms** | **2,38 ms** |
| Ganho | **−65%** | **−65%** | **−33%** | **−33%** |

  No HTTP o ganho é menor porque cerca de 1,6 ms de cada requisição é rede + FastAPI, não modelo.

  > Medir uma rota depois da outra dava resultados enganosos (o ONNX chegou a parecer mais lento). Por isso o benchmark intercala as rotas.

### D7 — Retreino seguro (DAG do Airflow)

```text
ingest_data → train_model → quality_gate → export_onnx → promote_model
```

- **Candidato primeiro:** o modelo novo vai para `models/candidate/` e só substitui o modelo da API se passar por duas travas: **Macro-F1 ≥ 0,78** e **concordância ONNX ≥ 97%**.
  - A trava de 0,78 fica cerca de 5 pontos abaixo do modelo atual (0,828). Ela barraria a Regressão Logística antiga (0,744).
- **Teste da trava:** com o mínimo forçado em 0,99, a execução falhou com `Candidate macro-F1 0.8279 < 0.99; served model kept`, e os arquivos servidos ficaram idênticos (md5).
- **Execuções em série:** `max_active_runs=1`. Sem isso, um disparo manual simultâneo ao agendado apagava o candidato do outro; achamos esse bug testando no scheduler real.
- **Tempo:** do zero (download + treino + travas + ONNX + publicação), cerca de **50 s**.

---

## 4. Decisões de engenharia

| Tema | Decisão | Por quê |
|---|---|---|
| Arquitetura cloud (README, Etapa 1) | Inferência **real-time** em AWS ECS Fargate; S3 para artefatos; MWAA (Airflow) para retreino; Prometheus/Grafana | Em lote, um caso urgente esperaria pela próxima janela de processamento |
| Airflow | Roda no ambiente `uv` do projeto, como grupo opcional | Usa as mesmas versões de scikit-learn/skl2onnx da imagem da API, então o modelo retreinado carrega sem conflito; a imagem Docker não leva o Airflow |
| `svcs<26` | Versão fixada no grupo `airflow` | O svcs 26 quebra o `airflow dags test` do Airflow 3.2 (`KeyError: 'svcs_registry'`) |
| CI | 3 jobs: `lint` (~9 s) → `test` (~31 s, 17 testes) → `build` (~92 s) | O `build` treina pela DAG, constrói a imagem com esses artefatos, sobe o container e roda o smoke test: o mesmo caminho da produção |
| Testes | Corpus sintético + diretórios temporários | Rodam em segundos, sem rede e sem tocar em `data/` e `models/` |
| Dashboard | Painéis filtrados às rotas de triagem | Sem o filtro, os scrapes do Prometheus em `/metrics` inflavam o total (~12/min com a API parada) |
| Histograma | Buckets de 0,5 ms até 5 ms | Os buckets padrão começam em 5 ms, e as respostas levam 1–4 ms: o P95 ficava preso em ~4,75 ms para as duas rotas |
| Painel de erros | Séries 5xx **e** 4xx | A API não tem caminho que gere 5xx; os 4xx (payload inválido) mostram o painel reagindo |

---

## 5. Roteiro STAR (≤ 5 min)

Cerca de 140 palavras por minuto. Os números em **negrito** são os que valem a pena falar.

### S — Situação (0:00–0:40)

**Na tela:** README (título + Etapa 1).

- Um hospital de referência recebe um grande volume de textos clínicos, e alguém precisa decidir o que é **urgente**. Triagem manual é lenta, e um caso urgente na fila errada custa caro.
- A proposta: classificar automaticamente cada texto em **normal / atenção / urgente**, em tempo real, via API.
- Diga com clareza: usamos o **Medical Abstracts Corpus**, com resumos médicos como substitutos de laudos e um **mapeamento de condição para urgência definido pelo grupo**.

### T — Tarefa (0:40–1:20)

**Na tela:** lista de requisitos ou o README.

- Requisitos: modelo de NLP **leve**, API REST em Docker, **CI/CD** no GitHub Actions, **retreino orquestrado** no Airflow, **monitoramento** com Prometheus + Grafana e **uma técnica de otimização de latência**.
- Decisão de arquitetura: inferência **real-time** (ECS Fargate na AWS), porque em lote um caso urgente esperaria pela próxima janela de processamento.

### A — Ação (1:20–3:50)

**Na tela, nesta ordem:** notebook (tabela de modelos) → código do `train.py` → Airflow (grafo da DAG) → GitHub Actions (run verde) → Grafana.

1. **Dados (~30 s).** "Ao explorar os dados, descobrimos que 35% do teste repetia textos do treino com outro rótulo, porque o corpus é multi-rótulo. Isso derrubava a acurácia para menos de 50%. Filtramos para abstracts com um único rótulo e ficamos com 6,6 mil para treino."
2. **Modelo (~35 s).** Comparamos 5 classificadores sobre o mesmo TF-IDF, escolhendo por validação cruzada e Macro-F1. O **Random Forest** sugerido no PDF ficou cerca de **24× mais lento** e pior que o **Linear SVM**. O SVM puro não convertia para ONNX; **calibrado**, ele converte, dá probabilidades e chega a **Macro-F1 0,828**.
3. **Pipeline (~35 s).** A DAG faz ingestão → treino → trava de qualidade → ONNX → publicação. **O modelo da API só é trocado se o novo passar nas duas travas.** O CI roda lint, 17 testes e um build que **treina pela DAG, constrói a imagem e testa o container**.
4. **Monitoramento e otimização (~50 s).** Exportamos para **ONNX Runtime**. O próprio dashboard revelou um problema: com tráfego espaçado, a rota ONNX era **mais lenta** que a original, porque o pool de threads do ONNX ficava em espera ativa e disputava CPU com o servidor. Com **uma thread**, o p95 caiu de **11 para 3,9 ms**. Também ajustamos os buckets do histograma, porque os padrões não enxergavam a diferença em milissegundos.

### R — Resultado (3:50–5:00)

**Na tela:** Grafana com tráfego (P95 por rota com o ONNX abaixo), Swagger com uma predição e a tabela de latência do README.

- **Qualidade:** Macro-F1 **0,828**. Na triagem, **88%** de acerto, **86%** dos casos urgentes reconhecidos e só **0,8%** dos urgentes classificados como normais.
- **Latência:** o ONNX reduz o tempo do modelo em **65%** e o tempo de resposta em **33%** (de **3,2 para 2,1 ms**).
- **Ciclo de vida:** retreino em ~**50 s** com travas; CI verde em ~**2 min**; dashboard provisionado como código.
- **Lições aprendidas** (escolha 2):
  - Olhar os dados antes do modelo: o maior ganho veio de entender o rótulo, não de trocar o algoritmo.
  - Benchmark tem que ser justo: medindo uma rota depois da outra, o ONNX parecia mais lento.
  - Observabilidade serve para achar problemas: o dashboard revelou o problema de threads do ONNX.
  - O exemplo do enunciado (Random Forest) não era a melhor escolha para texto esparso.

---

## 6. Preparação da gravação

```bash
# 1. Docker Desktop aberto. Na raiz do repo:
uv sync --group airflow
./scripts/airflow.sh dags test medical_triage_training      # gera models/ (se ainda não existir)
docker compose up -d --build                                 # API + Prometheus + Grafana
uv run python scripts/smoke_test.py                          # confere se está tudo de pé

# 2. Uns 3 minutos antes de mostrar o Grafana, num terminal separado:
uv run python -m src.api.traffic --minutes 15                # ~10 req/s, ~5% de payloads inválidos

# 3. Airflow com interface (outro terminal), para mostrar o grafo da DAG:
./scripts/airflow.sh standalone                              # http://localhost:8080
#    usuário admin; a senha está em .airflow/simple_auth_manager_passwords.json.generated

# 4. Para a tabela de latência ao vivo (opcional, ~30 s):
uv run python -m src.api.benchmark
```

| Abrir no navegador | Endereço |
|---|---|
| Swagger (fazer uma predição ao vivo) | http://localhost:8000/docs |
| Grafana | http://localhost:3000/d/triage-api (admin / admin) |
| Airflow | http://localhost:8080 → DAG `medical_triage_training` → aba Graph |
| GitHub Actions | https://github.com/MunhozMCM/Tech-Challenge-03/actions |

Exemplo para o Swagger: `"Acute ST-elevation myocardial infarction with severe left ventricular dysfunction requiring immediate catheterization."` → cardiovascular / **urgente**.

**Dicas:**

- O P95 só aparece **enquanto há tráfego** (janelas de 1 min). Deixe o gerador rodando.
- Não rode o `benchmark` com o gerador ligado: um interfere na medição do outro.
- Depois de um retreino, a API só passa a servir o novo modelo após `docker compose up -d --build api`.

---

## 7. Cuidados no que dizer

| ❌ Evite | ✅ Diga |
|---|---|
| "Laudos reais de hospital" | "Resumos médicos, como substitutos de laudos" |
| "Mapeamento de urgência validado clinicamente" | "Mapeamento definido pelo grupo; em produção precisaria de validação médica" |
| "Usamos Random Forest" / "deep learning" / "BERT" | "TF-IDF + Linear SVM calibrado; o Random Forest foi testado e descartado" |
| "O ONNX deixou a API 3× mais rápida" | "−65% no tempo do modelo, −33% no tempo de resposta" |
| "Está rodando na AWS" | "A arquitetura proposta é ECS Fargate; a entrega roda localmente em Docker" |
| "A taxa de erro 5xx é zero porque o sistema é perfeito" | "Não houve 5xx nos testes; a série 4xx mostra o painel reagindo a payloads inválidos" |

---

## 8. Perguntas prováveis

- **Por que não um transformer (BERT/BioBERT)?** O desafio pede um modelo leve, com foco em latência. Um transformer seria centenas de vezes mais pesado. Com TF-IDF + SVM chegamos a 0,83 de Macro-F1 com cerca de 0,5 ms de inferência no ONNX.
- **Por que Macro-F1 e não acurácia?** As classes são desbalanceadas, e a acurácia esconderia um desempenho ruim na classe pequena (digestivo, 8%).
- **Por que remover os abstracts multi-rótulo?** O mesmo texto com rótulos diferentes é ruído que nenhum classificador de rótulo único resolve, e 35% do teste vazava do treino com outro rótulo.
- **E se o retreino gerar um modelo pior?** A DAG reprova a execução e mantém o modelo atual (trava de F1 ≥ 0,78 + concordância ONNX ≥ 97%). Testamos isso forçando a trava.
- **Por que uma thread no ONNX?** O modelo é pequeno; o paralelismo interno só gerava espera ativa e disputa de CPU. Medimos: p95 de 11,0 → 3,9 ms com tráfego espaçado.
- **Como o modelo novo chega à API?** A DAG publica os artefatos em `models/`, e a imagem é reconstruída. Na nuvem, isso seria um novo deploy da task no ECS a partir do S3.

---

## 9. Pendências conhecidas

- **Link do vídeo** no README (seção "🎬 Vídeo de Apresentação" com espaço reservado).
- **Print do dashboard** para a entrega (o JSON já está no repo).
- **Branch `feat/etapa-2-ci-airflow`:** está toda contida na `main` e pode ser apagada no GitHub.
- **Imagem Docker:** tem 1,66 GB, porque leva o MLflow e outras dependências que a API não usa. Dá para separar se houver tempo.
- **Probabilidades:** a API ainda não devolve a probabilidade da classe, embora o modelo calibrado já a calcule.
