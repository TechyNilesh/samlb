import numpy as np
import pytest

import samlb.datasets as datasets_mod
from samlb.evaluation.evaluator import PrequentialEvaluator
from samlb.evaluation.results import RunResult, aggregate_runs
from samlb.framework.classification.asml import AutoStreamClassifier


def test_stream_normalize_is_online_not_full_dataset(monkeypatch):
    X = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int32)
    meta = {"feature_names": ["f0"]}

    def fake_load(name, task="classification", max_samples=None):
        return X, y, meta

    monkeypatch.setattr(datasets_mod, "load", fake_load)

    values = [x["f0"] for x, _ in datasets_mod.stream("dummy", normalize=True)]
    assert values == [0.0, 1.0, 1.0]


def test_evaluator_validates_runtime_sampling_interval():
    with pytest.raises(ValueError, match="sample_runtime_every"):
        PrequentialEvaluator(task="classification", sample_runtime_every=0)


def test_evaluator_keeps_partial_tail_window():
    class ConstantModel:
        def predict_one(self, x):
            return 0

        def learn_one(self, x, y):
            return None

    result = PrequentialEvaluator(
        task="classification",
        window_size=128,
        max_samples=300,
    ).run(ConstantModel(), "electricity", "constant")

    assert result.n_samples == 300
    assert len(result.windowed_metrics["accuracy"]) == 3


def test_aggregate_runs_excludes_failed_runs():
    ok = RunResult(
        framework_name="fw",
        dataset_name="ds",
        task="classification",
        n_samples=100,
        metrics={"accuracy": 0.8, "f1": 0.7, "precision": 0.7, "recall": 0.7},
        windowed_metrics={"accuracy": [0.8], "f1": [0.7], "precision": [0.7], "recall": [0.7]},
        total_runtime_s=1.0,
        runtime_per_instance_ms=[1.0],
        run_id=0,
        seed=1,
        error=None,
    )
    failed = RunResult(
        framework_name="fw",
        dataset_name="ds",
        task="classification",
        n_samples=10,
        metrics={"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0},
        windowed_metrics={"accuracy": [], "f1": [], "precision": [], "recall": []},
        total_runtime_s=0.1,
        runtime_per_instance_ms=[1.0],
        run_id=1,
        seed=2,
        error="boom",
    )

    agg = aggregate_runs([ok, failed])

    assert agg["n_runs"] == 2
    assert agg["n_successful_runs"] == 1
    assert agg["n_failed_runs"] == 1
    assert agg["metrics"]["accuracy"]["mean"] == pytest.approx(0.8)
    assert len(agg["windowed_metrics"]["accuracy"]["mean"]) == 1


def test_asml_snapshots_do_not_alias_live_pipelines():
    model = AutoStreamClassifier(seed=42, prediction_mode="ensemble")

    pipeline_ids = {id(pipe) for pipe in model._pipelines}
    snapshot_ids = [id(snap) for snap in model._snapshots]

    assert all(sid not in pipeline_ids for sid in snapshot_ids)


def test_asml_best_model_tracks_active_pool_after_exploration():
    model = AutoStreamClassifier(seed=42, prediction_mode="best")

    model._counter = model.exploration_window
    model._maybe_explore()

    pipeline_ids = {id(pipe) for pipe in model._pipelines}
    assert id(model.best_model) in pipeline_ids
    assert model._best_idx == 0
