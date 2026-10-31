# NLP Experiments — Decision Log

All numbers below were measured on the single-label version of the official Medical Abstracts TC Corpus split (6,607 train / 1,691 test abstracts), `SEED = 42`. Model selection uses 5-fold stratified CV on train; the test set is only used for reporting.

## Decision 1: Dropping every abstract tagged with more than one condition

### Problem

The corpus is a multi-label dataset flattened into rows: an abstract tagged with *k* conditions appears *k* times, once per label. There are no exact `(text, label)` duplicates, so every repeated text has conflicting targets:

| Fact | Value |
|---|---|
| Unique abstracts (train + test) | 11,227 |
| Abstracts with 2 / 3 / 4 labels | 2,653 / 270 / 6 |
| Train rows whose text is multi-label | 4,943 (42.8%) |
| Test rows whose text is multi-label | 1,197 (41.4%) |
| Test rows whose text is also in train | 1,010 (35.0%), **always with a different label** |

The dominant combinations pair a specific condition with the catch-all class: cardiovascular + general pathological (738), neoplasms + general (486), digestive + general (475), nervous + general (446).

A single-label classifier cannot fit identical inputs with different targets, and the official test split penalises it for learning the train label: 35% of test rows are abstracts the model has already seen in train, with another label.

### Evidence

The best pipeline (TF-IDF + Linear SVM) trained on the raw official train split:

| Train → Test | Accuracy | F1 (macro) |
|---|---|---|
| raw → raw | 0.505 | 0.501 |
| raw → clean | 0.811 | 0.809 |
| clean → clean | **0.818** | **0.815** |

Training on the raw rows barely hurts (0.809 vs 0.815). The raw test split is the problem: it cuts the measured accuracy from ~0.81 to ~0.50 because of label conflicts, not because the model is weak.

### Decision

Drop every abstract with more than one distinct label, from both splits. This keeps 6,607 / 1,691 rows (well above the challenge's 2,000-sample minimum), keeps the class proportions of the original splits, and removes all train/test text overlap. The notebook asserts that no text overlap and no duplicate texts remain.

### Alternatives kept open

- **Priority collapse:** when an abstract has one specific condition plus "general pathological conditions", keep the specific one. This would recover ~2,100 abstracts. It is a labelling rule we would have to justify, so it is not adopted yet.
- **Multi-label (one-vs-rest):** this matches the data more faithfully, but it does not fit an API that returns one triage class per report.

---

## Decision 2: Macro-F1 as the selection metric

The classes are imbalanced (digestive ≈ 8%, general pathological ≈ 29% of single-label train). Accuracy and weighted-F1 would reward a model that neglects the small classes. Macro-F1 weights every condition equally, which suits triage: a missed digestive case is as bad as a missed cardiovascular one. Accuracy, weighted-F1 and macro precision/recall are still reported and logged to MLflow.

---

## Decision 3: One shared TF-IDF representation for every classifier

All candidates use the same vectorizer: lowercase, English stop words, unigrams + bigrams, `min_df=2`, `sublinear_tf=True`. The vocabulary has 81,472 terms. Holding the representation fixed means the comparison isolates the classifier. Tuning the vectorizer (`max_features`, `max_df`, character n-grams) is a later step, and it also affects model size and latency.

---

## Decision 4: TF-IDF + Linear SVM as the leading candidate

| Model | CV F1 macro | Test F1 macro | Test accuracy | Latency p50 / p95 (ms) | Size (MB) |
|---|---|---|---|---|---|
| **Linear SVM** | **0.798 ± 0.008** | **0.815** | **0.818** | 0.60 / 0.81 | 5.6 |
| Logistic Regression | 0.725 ± 0.004 | 0.744 | 0.769 | 0.62 / 0.91 | 5.6 |
| Random Forest (300 trees) | 0.699 ± 0.004 | 0.705 | 0.740 | 20.3 / 25.9 | 125.8 |
| Multinomial NB | 0.473 ± 0.004 | 0.464 | 0.661 | 1.04 / 1.38 | 8.8 |
| Dummy (most frequent) | 0.090 | 0.088 | 0.280 | 0.48 / 0.70 | 2.3 |

Latency is the wall time of `predict([one_abstract])` over 300 test abstracts, single-threaded. That is how the API will serve requests. It was measured in-process, without HTTP or Docker overhead.

- **Linear SVM** has the best quality and is among the cheapest models. Its CV and test scores agree (0.798 vs 0.815), so the choice is not a lucky test split.
- **Random Forest**, the challenge's example (TF-IDF + RF), is worse on every axis: −0.11 macro-F1, ~34× slower and ~23× larger. Hundreds of deep trees over 81k sparse features are the wrong tool here.
- **Logistic Regression and Multinomial NB** run on library defaults (`C=1.0`, `alpha=1.0`). Both look under-tuned, so their scores are lower bounds, not verdicts.

Weakest spots of the Linear SVM: *nervous system diseases* (recall 0.71, with 21% of its abstracts predicted as *general pathological*) and the catch-all *general pathological conditions* (F1 0.74). That is expected for a residual class whose vocabulary ("acute", "chronic", "postoperative") overlaps the others.

### Open points for the next iterations

- `LinearSVC` has no `predict_proba`. If the API must return a confidence score, wrap it in `CalibratedClassifierCV` or tune `C` for Logistic Regression and compare again.
- Hyperparameter search (`C`, `alpha`, vectorizer settings) under the same CV protocol.
- Etapa 4: confirm the chosen pipeline converts with `skl2onnx` (TF-IDF with bigrams and `sublinear_tf`). Then measure ONNX Runtime against sklearn with this same single-request protocol. The sklearn baseline is already sub-millisecond, so the comparison should also report p95 and the latency measured through the API.

---

## Decision 5: Production model — calibrated Linear SVM (resolves the open points above)

### What happened after the notebook

1. `src/models/train.py` first shipped the Linear SVM. A bare `LinearSVC` inside a Pipeline does not convert with skl2onnx 1.20: its parser asks every classifier for the `zipmap` option, and the `LinearSVC` converter rejects it (`NameError: Option 'zipmap' not in [...] for class 'LinearSVC'`). The conversion fails both through `to_onnx` and through `convert_sklearn`.
2. To unblock Etapa 4 (ONNX), the pipeline was switched to `LogisticRegression` on defaults (`C=1.0`), which converts but scores only 0.744 macro-F1.
3. Re-evaluated with the same TF-IDF, splits and 5-fold CV. The ONNX columns come from the converted model on the test set; latency is single-request p50, in-process:

| Classifier | CV F1 macro | Test F1 macro | ONNX F1 | ONNX agreement | sklearn p50 | ONNX p50 | joblib / ONNX size |
|---|---|---|---|---|---|---|---|
| LogisticRegression `C=1` | 0.725 | 0.744 | 0.752 | 98.5% | 0.69 ms | 0.47 ms | 5.6 / 4.8 MB |
| LogisticRegression `C=30` | 0.791 | 0.803 | 0.807 | 98.4% | 0.62 ms | 0.46 ms | 5.6 / 4.8 MB |
| Calibrated LinearSVC, `ensemble=True` | 0.803 | 0.823 | 0.829 | 98.0% | 4.41 ms | 0.86 ms | 18.6 / 12.9 MB |
| **Calibrated LinearSVC, `ensemble=False`** | **0.805** | **0.828** | **0.833** | 98.3% | 1.77 ms | 0.62 ms | 5.6 / 4.8 MB |

### Decision

Serve `CalibratedClassifierCV(LinearSVC(), cv=5, ensemble=False)` (Platt/sigmoid calibration on out-of-fold scores, then one SVM refit on all training data):

- It has the best quality, in CV and on test. Its test result is accuracy 0.829 and macro-F1 0.828, with macro precision 0.830 and macro recall 0.827.
- The calibration wrapper converts to ONNX, so the Linear SVM chosen here is now the model that actually ships.
- It returns class probabilities, which closes the first open point.
- `ensemble=False` keeps a single SVM. The model is as small as the LogisticRegression, while the default `ensemble=True` is 3× larger and slower with no quality gain.

Its sklearn latency (1.8 ms) is higher than the LogisticRegression's, because calibration adds a step in Python. That cost disappears in ONNX (0.6 ms), which is the optimized serving path.

The Airflow quality gate was raised from 0.70 to **0.78** macro-F1, about 5 points below the served model. A regression to the old LogisticRegression (0.744) would now be rejected.
