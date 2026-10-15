"""Central project configuration: paths, constants and MLflow settings."""

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

# Medical Abstracts TC Corpus (Schopf et al., 2023) — ships with an official train/test split
DATASET_URL: Final[str] = (
    "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
)
TRAIN_DATASET_PATH: Final[Path] = RAW_DATA_DIR / "medical_tc_train.csv"
TEST_DATASET_PATH: Final[Path] = RAW_DATA_DIR / "medical_tc_test.csv"
LABELS_PATH: Final[Path] = RAW_DATA_DIR / "medical_tc_labels.csv"

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
    2: "normal",   # digestive system diseases (acompanhamento ambulatorial padrão)
    3: "urgente",  # nervous system diseases (risco neurológico agudo)
    4: "urgente",  # cardiovascular diseases (risco hemodinâmico/isquêmico agudo)
    5: "atencao",  # general pathological conditions (avaliação clínica geral)
}

BASELINE_MODEL_PATH: Final[Path] = MODELS_DIR / "linear_svm_baseline.joblib"
