"""Inference API: both backends, payload validation and Prometheus instrumentation."""

import pytest
from fastapi.testclient import TestClient

from src.config import CLASS_NAMES, TEXT_COLUMN, URGENCY_MAPPING

CARDIAC_REPORT = (
    "Coronary myocardial infarction with ventricular arrhythmia and severe hypertension "
    "observed in cardiac patients."
)


@pytest.fixture
def client(monkeypatch, artifacts):
    from src.api import main

    monkeypatch.setattr(main, "BASELINE_MODEL_PATH", artifacts["model"])
    monkeypatch.setattr(main, "ONNX_MODEL_PATH", artifacts["onnx"])
    with TestClient(main.app) as test_client:  # runs the lifespan that loads both models
        yield test_client


def test_health_reports_both_models_loaded(client):
    response = client.get("/health")

    assert response.json() == {"status": "ok", "sklearn_loaded": True, "onnx_loaded": True}


@pytest.mark.parametrize(
    ("route", "backend"),
    [("/predict", "sklearn-logistic-regression"), ("/predict/onnx", "onnxruntime")],
)
def test_predict_returns_condition_and_urgency(client, route, backend):
    response = client.post(route, json={TEXT_COLUMN: CARDIAC_REPORT})

    assert response.status_code == 200
    body = response.json()
    assert body["condition_label"] == 4
    assert body["condition_name"] == CLASS_NAMES[4]
    assert body["urgency"] == URGENCY_MAPPING[4] == "urgente"
    assert body["model_backend"] == backend
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize("payload", [{TEXT_COLUMN: "too short"}, {}, {"text": CARDIAC_REPORT}])
def test_predict_rejects_invalid_payloads(client, payload):
    assert client.post("/predict/onnx", json=payload).status_code == 422


def test_metrics_endpoint_exposes_request_count_and_latency(client):
    client.post("/predict/onnx", json={TEXT_COLUMN: CARDIAC_REPORT})

    lines = client.get("/metrics").text.splitlines()

    assert any(
        line.startswith("api_requests_total{")
        and 'endpoint="/predict/onnx"' in line
        and 'http_status="200"' in line
        for line in lines
    )
    assert any(line.startswith("api_request_latency_seconds_bucket{") for line in lines)
