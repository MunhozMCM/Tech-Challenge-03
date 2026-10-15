"""Dataset input/output and preprocessing helpers."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from src.config import (
    DATASET_URL,
    LABELS_PATH,
    RAW_DATA_DIR,
    TARGET,
    TEST_DATASET_PATH,
    TEXT_COLUMN,
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


def load_clean_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test splits keeping only single-label abstracts (Decision 1)."""
    download_medical_abstracts()
    train_df = read_dataset(TRAIN_DATASET_PATH)
    test_df = read_dataset(TEST_DATASET_PATH)

    combined = pd.concat([train_df, test_df], ignore_index=True)
    label_counts = combined.groupby(TEXT_COLUMN)[TARGET].nunique()
    single_label_texts = set(label_counts[label_counts == 1].index)

    clean_train = train_df[train_df[TEXT_COLUMN].isin(single_label_texts)].reset_index(drop=True)
    clean_test = test_df[test_df[TEXT_COLUMN].isin(single_label_texts)].reset_index(drop=True)

    return clean_train, clean_test
