"""Training and ONNX export, run through the same functions the CLI and the DAG use."""

import joblib
import onnx

from src.config import CLASS_NAMES, MIN_F1_MACRO, MIN_ONNX_AGREEMENT, ONNX_LOCALE, TEXT_COLUMN
from src.models.convert_to_onnx import onnx_agreement


def test_train_and_evaluate_returns_test_metrics(artifacts):
    metrics = artifacts["metrics"]

    assert set(metrics) == {
        "test_accuracy",
        "test_f1_macro",
        "test_f1_weighted",
        "test_precision_macro",
        "test_recall_macro",
    }
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    # A separable corpus must clear the same gate the Airflow DAG applies
    assert metrics["test_f1_macro"] >= MIN_F1_MACRO


def test_saved_pipeline_knows_every_condition(artifacts):
    pipeline = joblib.load(artifacts["model"])

    assert set(pipeline.classes_) == set(CLASS_NAMES)


def test_onnx_model_predicts_like_sklearn(artifacts, corpus):
    _, test = corpus

    agreement = onnx_agreement(artifacts["model"], artifacts["onnx"], test[TEXT_COLUMN].tolist())

    assert agreement >= MIN_ONNX_AGREEMENT


def test_onnx_model_uses_portable_locale(artifacts):
    # en_US.UTF-8 is missing on WSL and slim images; ONNX Runtime refuses to load the model there
    graph = onnx.load(artifacts["onnx"]).graph
    normalizers = [node for node in graph.node if node.op_type == "StringNormalizer"]
    locales = {
        attr.s.decode() for node in normalizers for attr in node.attribute if attr.name == "locale"
    }

    assert locales == {ONNX_LOCALE}
