"""FastAPI application for real-time medical report triage."""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.schemas import TriageRequest, TriageResponse
from src.config import BASELINE_MODEL_PATH, CLASS_NAMES, URGENCY_MAPPING

ONNX_MODEL_PATH = "models/model_optimized.onnx"

REQUEST_COUNT = Counter(
    "api_requests_total", 
    "Total de requisicoes", 
    ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", 
    "Tempo de resposta da API em segundos", 
    ["method", "endpoint"]
)

model_registry: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load both standard and ONNX classification pipelines into memory on startup."""
    if BASELINE_MODEL_PATH.exists():
        model_registry["pipeline"] = joblib.load(BASELINE_MODEL_PATH)
    
    # Carregando o modelo otimizado com ONNX Runtime
    model_registry["onnx_session"] = ort.InferenceSession(ONNX_MODEL_PATH)
    
    yield
    model_registry.clear()


app = FastAPI(
    title="Medical Triage NLP API",
    description="API de triagem automática de laudos médicos (Tech Challenge Fase 3)",
    version="0.1.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    method = request.method
    endpoint = request.url.path
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency = time.perf_counter() - start_time
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=str(status_code)).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

@app.get("/metrics")
def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok", 
        "sklearn_loaded": "pipeline" in model_registry,
        "onnx_loaded": "onnx_session" in model_registry
    }

# Rota Baseline (Scikit-Learn)
@app.post("/predict", response_model=TriageResponse)
def predict(payload: TriageRequest) -> TriageResponse:
    pipeline = model_registry.get("pipeline")
    start = time.perf_counter()
    
    pred_label = int(pipeline.predict([payload.medical_abstract])[0])
    latency_ms = (time.perf_counter() - start) * 1000.0

    return TriageResponse(
        condition_label=pred_label,
        condition_name=CLASS_NAMES[pred_label],
        urgency=URGENCY_MAPPING[pred_label],
        latency_ms=round(latency_ms, 4),
        model_backend="sklearn-logistic-regression",
    )

# Nova Rota Otimizada (ONNX)
@app.post("/predict/onnx", response_model=TriageResponse)
def predict_onnx(payload: TriageRequest) -> TriageResponse:
    session = model_registry.get("onnx_session")
    if not session:
        raise HTTPException(status_code=503, detail="ONNX Runtime não carregado.")
        
    start = time.perf_counter()
    
    # ONNX exige o input como um array 2D de strings do tipo object
    inputs = {"medical_abstract": np.array([[payload.medical_abstract]], dtype=object)}
    
    # Executa a inferência em C++
    ort_outs = session.run(None, inputs)
    pred_label = int(ort_outs[0][0])
    
    latency_ms = (time.perf_counter() - start) * 1000.0

    return TriageResponse(
        condition_label=pred_label,
        condition_name=CLASS_NAMES[pred_label],
        urgency=URGENCY_MAPPING[pred_label],
        latency_ms=round(latency_ms, 4),
        model_backend="onnxruntime",
    )