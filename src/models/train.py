"""Training pipeline for the baseline NLP classifier (TF-IDF + LinearSVC)."""

import joblib
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    BASELINE_MODEL_PATH,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    SEED,
    TARGET,
    TEXT_COLUMN,
)
from src.data.io import load_clean_splits


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + LinearSVC pipeline defined in Decisions 3 and 4."""
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
            ("clf", LinearSVC(random_state=SEED)),
        ]
    )


def train_and_evaluate() -> Pipeline:
    """Train the baseline model, log metrics to MLflow, and persist the artifact."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_clean_splits()

    x_train, y_train = train_df[TEXT_COLUMN], train_df[TARGET]
    x_test, y_test = test_df[TEXT_COLUMN], test_df[TARGET]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="linear_svm_baseline"):
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
                "classifier": "LinearSVC",
                "seed": SEED,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
            }
        )
        mlflow.log_metrics(metrics)

        joblib.dump(pipeline, BASELINE_MODEL_PATH)
        print(f"Modelo salvo em: {BASELINE_MODEL_PATH}")
        print(f"Métricas no teste: {metrics}")

    return pipeline


if __name__ == "__main__":
    train_and_evaluate()
