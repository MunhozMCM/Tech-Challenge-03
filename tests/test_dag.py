"""Airflow DAG integrity: parses cleanly, runs the stages in order, gates weak models."""

import pytest

pytest.importorskip("airflow", reason="Airflow group not installed (uv sync --group airflow)")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402
from airflow.sdk.exceptions import AirflowFailException  # noqa: E402

from src.config import MIN_F1_MACRO, PROJECT_ROOT  # noqa: E402

DAG_ID = "medical_triage_training"


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder=str(PROJECT_ROOT / "dags"), include_examples=False)


@pytest.fixture(scope="module")
def dag(dagbag):
    return dagbag.dags[DAG_ID]


def test_dags_folder_imports_without_errors(dagbag):
    assert dagbag.import_errors == {}
    assert DAG_ID in dagbag.dags


def test_tasks_form_the_training_pipeline_in_order(dag):
    downstream = {task.task_id: task.downstream_task_ids for task in dag.tasks}

    assert downstream == {
        "ingest_data": {"train_model"},
        "train_model": {"quality_gate"},
        "quality_gate": {"export_onnx"},
        "export_onnx": {"promote_model"},
        "promote_model": set(),
    }


def test_runs_weekly_one_at_a_time(dag):
    assert dag.schedule == "@weekly"
    assert dag.catchup is False
    assert dag.max_active_runs == 1  # concurrent runs would promote each other's candidates


def test_quality_gate_blocks_models_below_the_f1_floor(dag):
    gate = dag.get_task("quality_gate").python_callable

    assert gate({"test_f1_macro": MIN_F1_MACRO + 0.01}) == pytest.approx(MIN_F1_MACRO + 0.01)
    with pytest.raises(AirflowFailException):
        gate({"test_f1_macro": MIN_F1_MACRO - 0.01})
