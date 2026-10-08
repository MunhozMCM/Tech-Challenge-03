"""Dataset input/output helpers."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from src.config import (
    DATASET_URL,
    LABELS_PATH,
    RAW_DATA_DIR,
    TEST_DATASET_PATH,
    TRAIN_DATASET_PATH,
)


def download_medical_abstracts(force: bool = False) -> None:
    """Fetch the Medical Abstracts TC Corpus CSVs into data/raw, skipping files already present."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (TRAIN_DATASET_PATH, TEST_DATASET_PATH, LABELS_PATH):
        if force or not path.exists():
            urlretrieve(f"{DATASET_URL}/{path.name}", path)


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset into a DataFrame."""
    return pd.read_csv(path)
