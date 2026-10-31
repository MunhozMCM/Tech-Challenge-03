"""Convert the Scikit-Learn baseline pipeline to ONNX and check it predicts like the original."""

from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import BASELINE_MODEL_PATH, ONNX_LOCALE, ONNX_MODEL_PATH, TEXT_COLUMN


def export_onnx(model_path: Path = BASELINE_MODEL_PATH, onnx_path: Path = ONNX_MODEL_PATH) -> Path:
    """Convert a saved TF-IDF pipeline to ONNX; the API feeds it a [n, 1] string tensor."""
    pipeline = joblib.load(model_path)
    initial_type = [(TEXT_COLUMN, StringTensorType([None, 1]))]

    # O LogisticRegression é 100% suportado nativamente pelo skl2onnx
    onx = to_onnx(
        pipeline,
        initial_types=initial_type,
        target_opset=12,
        options={TfidfVectorizer: {"locale": ONNX_LOCALE}},
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onx.SerializeToString())
    return onnx_path


def onnx_agreement(model_path: Path, onnx_path: Path, texts: list[str]) -> float:
    """Share of texts where ONNX Runtime and scikit-learn predict the same label."""
    pipeline = joblib.load(model_path)
    session = ort.InferenceSession(str(onnx_path))
    inputs = {TEXT_COLUMN: np.array(texts, dtype=object).reshape(-1, 1)}
    onnx_preds = session.run(None, inputs)[0]
    return float(np.mean(onnx_preds == pipeline.predict(texts)))


def main() -> None:
    print(f"Convertendo {BASELINE_MODEL_PATH} para ONNX...")
    onnx_path = export_onnx()
    print(f"Modelo ONNX salvo com sucesso em: {onnx_path}")


if __name__ == "__main__":
    main()
