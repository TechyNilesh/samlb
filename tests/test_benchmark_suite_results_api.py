import pytest

from samlb.benchmark import BenchmarkSuite
from samlb.evaluation.results import RunResult


def _make_result(task="classification", run_id=0):
    return RunResult(
        framework_name="FW",
        dataset_name="electricity",
        task=task,
        n_samples=100,
        metrics={"accuracy": 0.8, "f1": 0.7, "precision": 0.7, "recall": 0.7},
        windowed_metrics={"accuracy": [0.8], "f1": [0.7], "precision": [0.7], "recall": [0.7]},
        total_runtime_s=1.0,
        runtime_per_instance_ms=[1.0],
        run_id=run_id,
        seed=run_id,
        error=None,
    )


def _suite():
    return BenchmarkSuite(
        models={"Dummy": object()},
        datasets=["electricity"],
        task="classification",
        n_runs=1,
        verbose=False,
    )


def test_load_results_replaces_existing_results():
    suite = _suite()
    r0 = _make_result(run_id=0)
    r1 = _make_result(run_id=1)

    suite.load_results([r0])
    assert len(suite.results) == 1
    assert suite.results[0].run_id == 0

    suite.load_results([r1])
    assert len(suite.results) == 1
    assert suite.results[0].run_id == 1


def test_merge_results_appends():
    suite = _suite()
    r0 = _make_result(run_id=0)
    r1 = _make_result(run_id=1)
    suite.merge_results([r0, r1])
    assert [r.run_id for r in suite.results] == [0, 1]


def test_merge_results_validates_task():
    suite = _suite()
    bad = _make_result(task="regression")
    with pytest.raises(ValueError, match="does not match suite task"):
        suite.merge_results([bad])


def test_merge_results_validates_type():
    suite = _suite()
    with pytest.raises(TypeError, match="RunResult"):
        suite.merge_results([object()])  # type: ignore[list-item]
