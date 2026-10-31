"""Send steady traffic to the API so the Grafana dashboard has live data (Etapa 3 demo).

    uv run python -m src.api.traffic                         # 5 min at ~10 req/s
    uv run python -m src.api.traffic --minutes 2 --rps 20 --invalid-ratio 0.1

Alternates /predict and /predict/onnx with real test abstracts. A small share of requests
carries a too-short text, which the API rejects with HTTP 422, so the error panel shows a
4xx series next to the 5xx one. API_BASE_URL defaults to http://127.0.0.1:8000.
"""

import argparse
import os
import random
import time
from collections import Counter

import httpx

from src.config import SEED, TEXT_COLUMN
from src.data.io import load_clean_splits

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
ROUTES = ("/predict", "/predict/onnx")
INVALID_TEXT = "curto"  # below the 10-character minimum → 422
REPORT_EVERY_S = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=5.0, help="how long to run")
    parser.add_argument("--rps", type=float, default=10.0, help="target requests per second")
    parser.add_argument(
        "--invalid-ratio", type=float, default=0.05, help="share of invalid payloads (HTTP 422)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, test_df = load_clean_splits()
    texts = test_df[TEXT_COLUMN].tolist()
    rng = random.Random(SEED)

    interval_s = 1.0 / args.rps
    deadline = time.monotonic() + args.minutes * 60
    next_report = time.monotonic() + REPORT_EVERY_S
    statuses: Counter[str] = Counter()
    sent = 0

    print(f"Enviando ~{args.rps:g} req/s para {API_BASE_URL} por {args.minutes:g} min (Ctrl+C)")
    with httpx.Client(base_url=API_BASE_URL, timeout=10.0) as client:
        try:
            while (started := time.monotonic()) < deadline:
                route = ROUTES[sent % len(ROUTES)]
                invalid = rng.random() < args.invalid_ratio
                text = INVALID_TEXT if invalid else rng.choice(texts)
                try:
                    statuses[str(client.post(route, json={TEXT_COLUMN: text}).status_code)] += 1
                except httpx.TransportError:
                    statuses["sem conexão"] += 1
                sent += 1

                if started >= next_report:
                    print(f"  {sent} requisições | status: {dict(sorted(statuses.items()))}")
                    next_report += REPORT_EVERY_S
                time.sleep(max(0.0, interval_s - (time.monotonic() - started)))
        except KeyboardInterrupt:
            pass

    print(f"Fim: {sent} requisições | status: {dict(sorted(statuses.items()))}")


if __name__ == "__main__":
    main()
