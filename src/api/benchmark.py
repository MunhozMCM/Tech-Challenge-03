"""Benchmark HTTP + inference latency against the running Docker container."""

import os
import statistics
import time

import httpx

from src.config import TEXT_COLUMN
from src.data.io import load_clean_splits

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")


def run_benchmark(n_samples: int = 300) -> None:
    """Measure p50 and p95 latency over n_samples test abstracts."""
    _, test_df = load_clean_splits()
    samples = test_df[TEXT_COLUMN].iloc[:n_samples].tolist()

    http_latencies: list[float] = []
    model_latencies: list[float] = []

    with httpx.Client(timeout=10.0) as client:
        # Warm-up
        client.post(API_URL, json={"medical_abstract": samples[0]})

        for text in samples:
            start = time.perf_counter()
            response = client.post(API_URL, json={"medical_abstract": text})
            total_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()

            data = response.json()
            http_latencies.append(total_ms)
            model_latencies.append(float(data["latency_ms"]))

    p50_model = statistics.median(model_latencies)
    p95_model = statistics.quantiles(model_latencies, n=20)[18]
    p50_http = statistics.median(http_latencies)
    p95_http = statistics.quantiles(http_latencies, n=20)[18]

    print(f"=== Benchmark ({n_samples} requisições sequenciais) ===")
    print(f"In-process (Model) | p50: {p50_model:.3f} ms | p95: {p95_model:.3f} ms")
    print(f"End-to-End (HTTP)  | p50: {p50_http:.3f} ms | p95: {p95_http:.3f} ms")


if __name__ == "__main__":
    run_benchmark()
