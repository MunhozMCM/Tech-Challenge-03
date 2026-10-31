"""Airflow DAG: weekly retraining of the medical-triage text classifier.

ingest_data → train_model → quality_gate → export_onnx → promote_model

Tasks call the same functions as the CLI (src.data.io, src.models.train,
src.models.convert_to_onnx), so a model trained here is identical to one trained by hand.
A new model is written to models/candidate/ first; the joblib + ONNX artifacts the API
serves are only replaced after it clears both quality gates, so a failed run leaves the
current production model untouched.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowFailException

# Airflow only puts the dags folder on sys.path; tasks also need the repo root to import `src`.
# Heavy imports (pandas, sklearn, onnx) stay inside the tasks to keep DAG parsing fast.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dag(
    dag_id="medical_triage_training",
    description="Ingest Medical Abstracts → train TF-IDF + LogisticRegression → gate → ONNX",
    schedule="@weekly",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    # Runs share data/processed/ and models/candidate/: two at once would promote each
    # other's files, so a manual trigger waits for the scheduled run (and vice versa)
    max_active_runs=1,
    default_args={"owner": "tech-challenge-3", "retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["tech-challenge-3", "nlp", "training"],
)
def medical_triage_training() -> None:
    @task
    def ingest_data() -> dict[str, int]:
        """Download the corpus, keep single-label abstracts and write the clean splits as CSV."""
        from src.config import PROCESSED_TEST_PATH, PROCESSED_TRAIN_PATH
        from src.data.io import load_clean_splits, save_dataset

        train_df, test_df = load_clean_splits()
        save_dataset(train_df, PROCESSED_TRAIN_PATH)
        save_dataset(test_df, PROCESSED_TEST_PATH)
        return {"train_rows": len(train_df), "test_rows": len(test_df)}

    @task
    def train_model(dataset: dict[str, int]) -> dict[str, float]:
        """Read the clean CSVs, train the pipeline, log the run to MLflow, save a candidate."""
        from src.config import CANDIDATE_MODEL_PATH, PROCESSED_TEST_PATH, PROCESSED_TRAIN_PATH
        from src.data.io import read_dataset
        from src.models.train import train_and_evaluate

        print(
            f"Training on {dataset['train_rows']:,} abstracts, testing on {dataset['test_rows']:,}"
        )
        return train_and_evaluate(
            read_dataset(PROCESSED_TRAIN_PATH),
            read_dataset(PROCESSED_TEST_PATH),
            model_path=CANDIDATE_MODEL_PATH,
            run_name="airflow_retrain",
        )

    @task
    def quality_gate(metrics: dict[str, float]) -> float:
        """Fail the run (no retries) when the candidate's test macro-F1 is below the floor."""
        from src.config import MIN_F1_MACRO

        f1_macro = metrics["test_f1_macro"]
        if f1_macro < MIN_F1_MACRO:
            raise AirflowFailException(
                f"Candidate macro-F1 {f1_macro:.4f} < {MIN_F1_MACRO}; served model kept"
            )
        return f1_macro

    @task
    def export_onnx() -> float:
        """Convert the candidate to ONNX and check it predicts like the scikit-learn pipeline."""
        from src.config import (
            CANDIDATE_MODEL_PATH,
            CANDIDATE_ONNX_PATH,
            MIN_ONNX_AGREEMENT,
            PROCESSED_TEST_PATH,
            TEXT_COLUMN,
        )
        from src.data.io import read_dataset
        from src.models import convert_to_onnx

        convert_to_onnx.export_onnx(CANDIDATE_MODEL_PATH, CANDIDATE_ONNX_PATH)
        texts = read_dataset(PROCESSED_TEST_PATH)[TEXT_COLUMN].tolist()
        agreement = convert_to_onnx.onnx_agreement(CANDIDATE_MODEL_PATH, CANDIDATE_ONNX_PATH, texts)
        if agreement < MIN_ONNX_AGREEMENT:
            raise AirflowFailException(
                f"ONNX agrees with scikit-learn on {agreement:.2%} of test abstracts "
                f"(< {MIN_ONNX_AGREEMENT:.0%}); served model kept"
            )
        return agreement

    @task
    def promote_model() -> dict[str, str]:
        """Move the candidate artifacts onto the paths the API loads (atomic renames)."""
        from src.config import (
            BASELINE_MODEL_PATH,
            CANDIDATE_MODEL_PATH,
            CANDIDATE_ONNX_PATH,
            ONNX_MODEL_PATH,
        )

        CANDIDATE_MODEL_PATH.replace(BASELINE_MODEL_PATH)
        CANDIDATE_ONNX_PATH.replace(ONNX_MODEL_PATH)
        # The API loads models at startup: rebuild/restart it to serve the new version
        return {"joblib": str(BASELINE_MODEL_PATH), "onnx": str(ONNX_MODEL_PATH)}

    metrics = train_model(ingest_data())
    quality_gate(metrics) >> export_onnx() >> promote_model()


medical_triage_training()
