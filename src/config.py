"""Central project configuration: paths, constants and MLflow settings."""

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

# Medical Abstracts TC Corpus (Schopf et al., 2023) — ships with an official train/test split
DATASET_URL: Final[str] = (
    "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
)
TRAIN_DATASET_PATH: Final[Path] = RAW_DATA_DIR / "medical_tc_train.csv"
TEST_DATASET_PATH: Final[Path] = RAW_DATA_DIR / "medical_tc_test.csv"
LABELS_PATH: Final[Path] = RAW_DATA_DIR / "medical_tc_labels.csv"

# Single-label splits written by the Airflow ingest task and read by its train task
PROCESSED_TRAIN_PATH: Final[Path] = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_PATH: Final[Path] = PROCESSED_DATA_DIR / "test.csv"

TEXT_COLUMN: Final[str] = "medical_abstract"
TARGET: Final[str] = "condition_label"
CLASS_NAMES: Final[dict[int, str]] = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

SEED: Final[int] = 42

MLFLOW_TRACKING_URI: Final[str] = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
EXPERIMENT_NAME: Final[str] = "medical-abstracts-nlp"

# Mapeamento clínico de urgência para triagem hospitalar
URGENCY_MAPPING: Final[dict[int, str]] = {
    1: "atencao",  # neoplasms (investigação oncológica prioritária)
    2: "normal",  # digestive system diseases (acompanhamento ambulatorial padrão)
    3: "urgente",  # nervous system diseases (risco neurológico agudo)
    4: "urgente",  # cardiovascular diseases (risco hemodinâmico/isquêmico agudo)
    5: "atencao",  # general pathological conditions (avaliação clínica geral)
}

BASELINE_MODEL_PATH: Final[Path] = MODELS_DIR / "linear_svm_baseline.joblib"
ONNX_MODEL_PATH: Final[Path] = MODELS_DIR / "model_optimized.onnx"

# ONNX StringNormalizer locale — C.UTF-8 ships with every glibc Linux, so the exported model
# runs without installing language packs (en_US.UTF-8 is missing on WSL/slim images)
ONNX_LOCALE: Final[str] = "C.UTF-8"

# Retraining (dags/medical_triage_training_dag.py): a new model is written here first and only
# replaces the served artifacts above after passing both quality gates
CANDIDATE_MODEL_DIR: Final[Path] = MODELS_DIR / "candidate"
CANDIDATE_MODEL_PATH: Final[Path] = CANDIDATE_MODEL_DIR / "model.joblib"
CANDIDATE_ONNX_PATH: Final[Path] = CANDIDATE_MODEL_DIR / "model.onnx"

MIN_F1_MACRO: Final[float] = 0.70  # current LogisticRegression baseline: 0.744 on the clean test
# ONNX vs scikit-learn label agreement on the test set; 0.985 observed, the misses are near-ties
# (float32 TF-IDF) and ONNX macro-F1 is on par, so this only catches a broken conversion
MIN_ONNX_AGREEMENT: Final[float] = 0.97
