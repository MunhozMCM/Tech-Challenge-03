"""Shared fixtures: a tiny synthetic corpus and model artifacts trained on it.

Tests never touch data/, models/ or mlflow.db and need no network: everything they write
goes under pytest's tmp paths.
"""

import os
import random
import tempfile

# Airflow creates its home (config, logs) on first import — keep it out of ~/airflow
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="tc3-airflow-"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.config import CLASS_NAMES, TARGET, TEXT_COLUMN  # noqa: E402

# Distinct vocabulary per condition, so the classifier can separate the classes reliably
CLASS_VOCAB: dict[int, list[str]] = {
    1: ["tumor", "carcinoma", "malignant", "metastasis", "chemotherapy", "lymphoma", "biopsy"],
    2: ["liver", "cirrhosis", "gastric", "bowel", "colitis", "hepatitis", "pancreatitis"],
    3: ["brain", "seizure", "epilepsy", "neuropathy", "migraine", "cerebral", "dementia"],
    4: ["coronary", "myocardial", "hypertension", "arrhythmia", "aortic", "ventricular", "cardiac"],
    5: ["infection", "inflammation", "fever", "sepsis", "postoperative", "wound", "bleeding"],
}
FILLER = ["patients", "study", "clinical", "results", "treatment", "group", "evaluated"]
SAMPLES_PER_CLASS, TEST_PER_CLASS = 30, 6


def make_abstract(rng: random.Random, label: int) -> str:
    words = rng.choices(CLASS_VOCAB[label], k=10) + rng.choices(FILLER, k=6)
    rng.shuffle(words)
    return " ".join(words).capitalize() + "."


@pytest.fixture(scope="session")
def corpus() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (train, test) splits in the same schema as the Medical Abstracts CSVs."""
    rng = random.Random(42)
    train_rows, test_rows = [], []
    for label in CLASS_NAMES:
        rows = [
            {TARGET: label, TEXT_COLUMN: make_abstract(rng, label)}
            for _ in range(SAMPLES_PER_CLASS)
        ]
        train_rows += rows[TEST_PER_CLASS:]
        test_rows += rows[:TEST_PER_CLASS]
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory, corpus) -> dict:
    """Train with the production code path, then export to ONNX, all under a tmp dir."""
    from src.models import convert_to_onnx, train

    out = tmp_path_factory.mktemp("models")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train, "MLFLOW_TRACKING_URI", f"sqlite:///{out / 'mlflow.db'}")
        metrics = train.train_and_evaluate(
            *corpus, model_path=out / "model.joblib", run_name="pytest"
        )
    onnx_path = convert_to_onnx.export_onnx(out / "model.joblib", out / "model.onnx")
    return {"model": out / "model.joblib", "onnx": onnx_path, "metrics": metrics}
