"""
samlb.evaluation.metrics
~~~~~~~~~~~~~~~~~~~~~~~~
Factory functions that produce fresh River metric dicts per task type.
One dict of live metric objects is maintained per run; all are updated
together in the inner prequential loop.
"""
from __future__ import annotations

from typing import Dict
from river import metrics as _m


def classification_metrics() -> Dict[str, object]:
    """Fresh River metric objects for classification.
    Keys: accuracy, f1, precision, recall  (F1/precision/recall are Macro).
    """
    return {
        "accuracy":  _m.Accuracy(),
        "f1":        _m.MacroF1(),
        "precision": _m.MacroPrecision(),
        "recall":    _m.MacroRecall(),
    }


def regression_metrics() -> Dict[str, object]:
    """Fresh River metric objects for regression.
    Keys: mae, rmse, r2
    """
    return {
        "mae":  _m.MAE(),
        "rmse": _m.RMSE(),
        "r2":   _m.R2(),
    }


def metrics_for_task(task: str) -> Dict[str, object]:
    if task == "classification":
        return classification_metrics()
    if task == "regression":
        return regression_metrics()
    raise ValueError(f"Unknown task {task!r}. Choose 'classification' or 'regression'.")


def snapshot(metric_dict: Dict[str, object]) -> Dict[str, float]:
    """Read the current .get() value from every metric in the dict."""
    return {k: float(m.get()) for k, m in metric_dict.items()}
