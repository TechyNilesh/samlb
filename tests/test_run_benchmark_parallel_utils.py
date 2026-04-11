import pytest

import run_benchmark as rb
from samlb.evaluation.results import RunResult


def test_resolve_seeds_default():
    assert rb._resolve_seeds(4, None) == [0, 1, 2, 3]


def test_resolve_seeds_validates_length():
    with pytest.raises(ValueError, match="len\\(seeds\\)"):
        rb._resolve_seeds(3, [1, 2])


def test_auto_workers_uses_cpu_utilization(monkeypatch):
    monkeypatch.setattr(rb.os, "cpu_count", lambda: 8)
    cpu_total, workers = rb._auto_workers(0.8, None)
    assert cpu_total == 8
    assert workers == 6


def test_auto_workers_honors_cap(monkeypatch):
    monkeypatch.setattr(rb.os, "cpu_count", lambda: 8)
    _, workers = rb._auto_workers(0.8, 4)
    assert workers == 4


def test_run_single_seed_forwards_progress_to_queue(monkeypatch):
    recorded = []

    class DummyQueue:
        def put(self, payload):
            recorded.append(payload)

    class DummySuite:
        def __init__(self, **kwargs):
            pass

        def run(self, progress_callback=None):
            result = RunResult(
                framework_name="ASML",
                dataset_name="electricity",
                task="classification",
                n_samples=10,
                metrics={
                    "accuracy": 0.9,
                    "f1": 0.9,
                    "precision": 0.9,
                    "recall": 0.9,
                },
                windowed_metrics={"accuracy": [0.9]},
                total_runtime_s=0.25,
                runtime_per_instance_ms=[],
            )
            progress_callback(
                {
                    "event": "task_started",
                    "task_index": 1,
                    "task_total": 4,
                    "run_id": 0,
                    "seed": 7,
                    "framework_name": "ASML",
                    "dataset_name": "electricity",
                }
            )
            progress_callback(
                {
                    "event": "task_finished",
                    "task_index": 1,
                    "task_total": 4,
                    "run_id": 0,
                    "seed": 7,
                    "framework_name": "ASML",
                    "dataset_name": "electricity",
                    "result": result,
                }
            )
            return [result]

    monkeypatch.setattr(rb, "BenchmarkSuite", DummySuite)
    monkeypatch.setattr(rb, "_build_models", lambda seed: {"ASML": object()})

    results = rb._run_single_seed(
        run_id=3,
        seed=7,
        datasets=["electricity"],
        max_samples=100,
        progress_queue=DummyQueue(),
    )

    assert len(results) == 1
    assert recorded[0]["event"] == "task_started"
    assert recorded[1]["event"] == "task_finished"
    assert recorded[1]["status"] == "ok"
    assert recorded[1]["metric"] == pytest.approx(0.9)
    assert recorded[1]["seed"] == 7
