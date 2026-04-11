"""
samlb.evaluation
~~~~~~~~~~~~~~~~
Framework-agnostic prequential evaluation primitives.

Any model with predict_one / learn_one / reset can be evaluated here.

    from samlb.evaluation import PrequentialEvaluator, WindowedEvaluator, RunResult

    ev = PrequentialEvaluator(task="classification")
    result = ev.run(model, dataset_name="electricity", framework_name="ASML")

    result.metrics                        # {'accuracy': 0.90, 'f1': ...}
    result.windowed_metrics["accuracy"]   # per-window learning curve list
    result.total_runtime_s
"""
from .evaluator import PrequentialEvaluator, WindowedEvaluator
from .results   import RunResult, aggregate_runs
from .metrics   import classification_metrics, regression_metrics, metrics_for_task

__all__ = [
    "PrequentialEvaluator",
    "WindowedEvaluator",
    "RunResult",
    "aggregate_runs",
    "classification_metrics",
    "regression_metrics",
    "metrics_for_task",
]
