#!/usr/bin/env bash
# Run any Airflow CLI command against this repo's DAGs, inside the project's uv environment
# (same scikit-learn / skl2onnx versions as the API image, so retrained models load there).
#
#   ./scripts/airflow.sh standalone                          # UI at http://localhost:8080
#   ./scripts/airflow.sh dags test medical_triage_training   # one full run, no scheduler
#
# Airflow state (config, sqlite metadata DB, logs, UI password) lives in .airflow/ (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AIRFLOW_HOME="${AIRFLOW_HOME:-$ROOT/.airflow}"
export AIRFLOW__CORE__DAGS_FOLDER="$ROOT/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export MLFLOW_DISABLE_AGENT_HINT=1

# First use: create the sqlite metadata DB (`standalone` does it itself, `dags test` does not)
if [[ ! -f "$AIRFLOW_HOME/airflow.db" ]]; then
    uv run --project "$ROOT" --group airflow airflow db migrate >/dev/null 2>&1
fi

exec uv run --project "$ROOT" --group airflow airflow "$@"
