"""Data layer: single-label filtering (Decision 1) and CSV persistence."""

import pandas as pd

from src.config import TARGET, TEXT_COLUMN
from src.data import io


def test_load_clean_splits_drops_every_multilabel_abstract(tmp_path, monkeypatch):
    train = pd.DataFrame(
        {
            TARGET: [1, 5, 3, 2],
            TEXT_COLUMN: [
                "tagged twice in train",
                "tagged twice in train",
                "only in train",
                "in both",
            ],
        }
    )
    test = pd.DataFrame({TARGET: [4, 1], TEXT_COLUMN: ["in both", "only in test"]})
    paths = {
        "TRAIN_DATASET_PATH": tmp_path / "train.csv",
        "TEST_DATASET_PATH": tmp_path / "test.csv",
        "LABELS_PATH": tmp_path / "labels.csv",
    }
    for name, path in paths.items():
        monkeypatch.setattr(io, name, path)
    monkeypatch.setattr(io, "RAW_DATA_DIR", tmp_path)
    train.to_csv(paths["TRAIN_DATASET_PATH"], index=False)
    test.to_csv(paths["TEST_DATASET_PATH"], index=False)
    paths["LABELS_PATH"].write_text("condition_label,condition_name\n")  # present → no download

    clean_train, clean_test = io.load_clean_splits()

    assert clean_train[TEXT_COLUMN].tolist() == ["only in train"]
    assert clean_test[TEXT_COLUMN].tolist() == ["only in test"]


def test_save_dataset_round_trips_and_creates_parent_dirs(tmp_path, corpus):
    train, _ = corpus
    destination = tmp_path / "processed" / "nested" / "train.csv"

    io.save_dataset(train, destination)

    pd.testing.assert_frame_equal(io.read_dataset(destination), train)
