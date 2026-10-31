"""Training pipeline for the NLP classifier (TF-IDF + calibrated Linear SVM)."""

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    BASELINE_MODEL_PATH,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    SEED,
    TARGET,
    TEXT_COLUMN,
)
from src.data.io import load_clean_splits


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + calibrated Linear SVM pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            # Linear SVM had the best macro-F1 in the notebook. Wrapping it in Platt calibration
            # (sigmoid, fitted on 5-fold out-of-fold scores) adds class probabilities and makes the
            # pipeline convertible to ONNX: skl2onnx 1.20 rejects a bare LinearSVC in a Pipeline.
            # ensemble=False keeps a single SVM, so the model is as small and fast as LogReg.
            (
                "clf",
                CalibratedClassifierCV(LinearSVC(random_state=SEED), cv=5, ensemble=False),
            ),
        ]
    )


def train_and_evaluate(
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    model_path: Path = BASELINE_MODEL_PATH,
    run_name: str = "calibrated_svm_baseline",
) -> dict[str, float]:
    """Train the baseline model, log metrics to MLflow, persist the artifact and return the metrics.

    Without splits it loads the clean single-label corpus itself (CLI usage); the Airflow DAG
    passes the CSVs written by its ingest task and a candidate ``model_path``.
    """
    if train_df is None or test_df is None:
        train_df, test_df = load_clean_splits()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train = train_df[TEXT_COLUMN], train_df[TARGET]
    x_test, y_test = test_df[TEXT_COLUMN], test_df[TARGET]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        pipeline = build_pipeline()
        pipeline.fit(x_train, y_train)

        preds = pipeline.predict(x_test)
        metrics = {
            "test_accuracy": float(accuracy_score(y_test, preds)),
            "test_f1_macro": float(f1_score(y_test, preds, average="macro")),
            "test_f1_weighted": float(f1_score(y_test, preds, average="weighted")),
            "test_precision_macro": float(precision_score(y_test, preds, average="macro")),
            "test_recall_macro": float(recall_score(y_test, preds, average="macro")),
        }

        mlflow.log_params(
            {
                "vectorizer": "TfidfVectorizer",
                "ngram_range": "(1, 2)",
                "min_df": 2,
                "sublinear_tf": True,
                "classifier": "CalibratedClassifierCV(LinearSVC, sigmoid, cv=5)",
                "seed": SEED,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
            }
        )
        mlflow.log_metrics(metrics)

        joblib.dump(pipeline, model_path)
        print(f"Modelo salvo em: {model_path}")
        print(f"Métricas no teste: {metrics}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
