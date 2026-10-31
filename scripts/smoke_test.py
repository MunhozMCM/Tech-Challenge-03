"""Smoke test a running inference API (single container or the docker compose stack).

    uv run python scripts/smoke_test.py [base_url]     # default: http://localhost:8000

Waits for /health, checks that both prediction routes classify a clear-cut cardiac report
the same way, that invalid payloads are rejected and that /metrics counts the calls.
Exits non-zero on the first failure.
"""

import sys
import time

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
CARDIAC_REPORT = (
    "Acute ST-elevation myocardial infarction with severe left ventricular dysfunction and "
    "coronary artery occlusion requiring immediate cardiac catheterization."
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def wait_until_healthy(client: httpx.Client, timeout_s: float = 90.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = client.get("/health")
            if response.status_code == 200:
                return response.json()
        except httpx.TransportError:
            pass
        time.sleep(1)
    raise SystemExit(f"FAIL: API at {BASE_URL} not healthy after {timeout_s:.0f}s")


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        health = wait_until_healthy(client)
        check(health.get("sklearn_loaded") and health.get("onnx_loaded"), f"health: {health}")

        labels = {}
        for route in ("/predict", "/predict/onnx"):
            response = client.post(route, json={"medical_abstract": CARDIAC_REPORT})
            check(response.status_code == 200, f"{route} returned {response.status_code}")
            body = response.json()
            labels[route] = body["condition_label"]
            print(
                f"{route:<14} → {body['condition_name']} / {body['urgency']} "
                f"({body['latency_ms']:.2f} ms, {body['model_backend']})"
            )
            check(body["urgency"] == "urgente", f"{route} triaged a cardiac report as {body}")
        check(len(set(labels.values())) == 1, f"sklearn and ONNX disagree: {labels}")

        status = client.post("/predict/onnx", json={"medical_abstract": "short"}).status_code
        check(status == 422, f"short payload returned {status}, expected 422")

        metrics = client.get("/metrics").text
        check('endpoint="/predict/onnx"' in metrics, "/metrics has no request counter")
        check("api_request_latency_seconds_bucket" in metrics, "/metrics has no latency histogram")

    print(f"Smoke test passed against {BASE_URL}")


if __name__ == "__main__":
    main()
