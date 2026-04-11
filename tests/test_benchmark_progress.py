from samlb.benchmark.suite import BenchmarkSuite
from samlb.evaluation.results import RunResult


class DummyModel:
    def __init__(self):
        self.seed = None
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class FakeEvaluator:
    def __init__(self, **kwargs):
        self.task = kwargs["task"]

    def run(self, model, dataset_name, framework_name):
        return RunResult(
            framework_name=framework_name,
            dataset_name=dataset_name,
            task=self.task,
            n_samples=25,
            metrics={
                "accuracy": 0.75,
                "f1": 0.74,
                "precision": 0.76,
                "recall": 0.73,
            },
            windowed_metrics={"accuracy": [0.75]},
            total_runtime_s=0.12,
            runtime_per_instance_ms=[],
        )


def test_benchmark_suite_emits_task_progress(monkeypatch):
    monkeypatch.setattr("samlb.benchmark.suite.PrequentialEvaluator", FakeEvaluator)

    model = DummyModel()
    events = []
    suite = BenchmarkSuite(
        models={"Dummy": model},
        datasets=["electricity", "shuttle"],
        task="classification",
        n_runs=2,
        seeds=[11, 22],
        verbose=False,
    )

    results = suite.run(progress_callback=events.append)

    assert len(results) == 4
    assert model.reset_calls == 4

    started = [event for event in events if event["event"] == "task_started"]
    finished = [event for event in events if event["event"] == "task_finished"]

    assert len(started) == 4
    assert len(finished) == 4
    assert started[0]["task_index"] == 1
    assert started[-1]["task_total"] == 4
    assert finished[0]["result"].framework_name == "Dummy"
    assert finished[-1]["seed"] == 22
