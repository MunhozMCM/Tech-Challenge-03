"""FastAPI application for real-time medical report triage."""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException

from src.api.schemas import TriageRequest, TriageResponse
from src.config import BASELINE_MODEL_PATH, CLASS_NAMES, URGENCY_MAPPING

model_registry: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the classification pipeline into memory on startup."""
    if not BASELINE_MODEL_PATH.exists():
        msg = (
            f"Modelo não encontrado em {BASELINE_MODEL_PATH}. "
            "Execute `uv run python -m src.models.train` antes de iniciar a API."
        )
        raise RuntimeError(msg)
    model_registry["pipeline"] = joblib.load(BASELINE_MODEL_PATH)
    yield
    model_registry.clear()


app = FastAPI(
    title="Medical Triage NLP API",
    description="API de triagem automática de laudos médicos (Tech Challenge Fase 3)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Verify service health and model availability."""
    return {"status": "ok", "model_loaded": "pipeline" in model_registry}


@app.post("/predict", response_model=TriageResponse)
def predict(payload: TriageRequest) -> TriageResponse:
    """Classify a medical abstract and return condition and urgency level."""
    pipeline = model_registry.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    start = time.perf_counter()
    pred_label = int(pipeline.predict([payload.medical_abstract])[0])
    latency_ms = (time.perf_counter() - start) * 1000.0

    return TriageResponse(
        condition_label=pred_label,
        condition_name=CLASS_NAMES[pred_label],
        urgency=URGENCY_MAPPING[pred_label],
        latency_ms=round(latency_ms, 4),
        model_backend="sklearn-linearsvc",
    )
