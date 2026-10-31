"""Benchmark scikit-learn vs ONNX latency against the running API (container or compose stack).

    uv run python -m src.api.benchmark        # API_BASE_URL defaults to http://127.0.0.1:8000

Requests alternate between /predict and /predict/onnx (flipping the order every abstract),
so network or machine drift during the run hits both routes equally.
"""

import os
import statistics
import time

import httpx

from src.config import TEXT_COLUMN
from src.data.io import load_clean_splits

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
SKLEARN_ROUTE, ONNX_ROUTE = "/predict", "/predict/onnx"
ROUTES = (SKLEARN_ROUTE, ONNX_ROUTE)


def percentile(values: list[float], pct: int) -> float:
    return statistics.quantiles(values, n=100)[pct - 1]


def run_benchmark(n_samples: int = 1000) -> None:
    """Measure p50/p95 model and end-to-end latency of both routes over n_samples abstracts."""
    _, test_df = load_clean_splits()
    samples = test_df[TEXT_COLUMN].iloc[:n_samples].tolist()

    model_ms: dict[str, list[float]] = {route: [] for route in ROUTES}
    http_ms: dict[str, list[float]] = {route: [] for route in ROUTES}

    with httpx.Client(base_url=API_BASE_URL, timeout=10.0) as client:
        for text in samples[:20]:  # warm-up
            for route in ROUTES:
                client.post(route, json={TEXT_COLUMN: text}).raise_for_status()

        for i, text in enumerate(samples):
            for route in ROUTES if i % 2 == 0 else ROUTES[::-1]:
                start = time.perf_counter()
                response = client.post(route, json={TEXT_COLUMN: text})
                http_ms[route].append((time.perf_counter() - start) * 1000.0)
                response.raise_for_status()
                model_ms[route].append(float(response.json()["latency_ms"]))

    print(f"=== Benchmark ({len(samples)} requisições por rota, intercaladas) ===")
    print(f"{'Rota':<15}{'Modelo p50':>12}{'Modelo p95':>12}{'HTTP p50':>12}{'HTTP p95':>12}  (ms)")
    for route in ROUTES:
        cells = [percentile(series[route], p) for series in (model_ms, http_ms) for p in (50, 95)]
        print(f"{route:<15}" + "".join(f"{cell:>12.3f}" for cell in cells))

    for label, series in (("modelo", model_ms), ("HTTP", http_ms)):
        gain = 1 - statistics.median(series[ONNX_ROUTE]) / statistics.median(series[SKLEARN_ROUTE])
        print(f"ONNX vs scikit-learn ({label}, p50): {gain:.0%} mais rápido")


if __name__ == "__main__":
    run_benchmark()
