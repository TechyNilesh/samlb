"""
samlb.metrics
~~~~~~~~~~~~~
Streaming metrics and drift detectors, backed by C++.

Replaces the ``river.metrics`` / ``river.drift`` surface SAMLB used. The
classes keep River's calling convention — ``update(y_true, y_pred)`` and
``get()`` — so framework code reads the same as before.
"""
from __future__ import annotations

import samlb._samlb_core as _core


class _Metric:
    """Shared plumbing: a C++ handle plus River's update/get interface."""

    _cpp_cls = None
    bigger_is_better = True

    def __init__(self):
        self._cpp = self._cpp_cls()

    def update(self, y_true, y_pred):
        self._cpp.update(y_true, y_pred)
        return self

    def get(self) -> float:
        return self._cpp.get()

    def reset(self):
        self._cpp.reset()
        return self

    def is_better_than(self, other) -> bool:
        """True when this metric beats *other* (direction-aware)."""
        if other is None:
            return True
        return self.get() > other.get() if self.bigger_is_better else self.get() < other.get()

    def __repr__(self) -> str:
        return f"{type(self).__name__}: {self.get():.6f}"


class _ClassificationMetric(_Metric):
    """Labels are ints on the C++ side; accept integral floats too, since a
    model (or a caller) may hand back e.g. 1.0 for class 1."""

    def update(self, y_true, y_pred):
        self._cpp.update(int(y_true), int(y_pred))
        return self


class Accuracy(_ClassificationMetric):
    _cpp_cls = _core.Accuracy


class MacroF1(_ClassificationMetric):
    _cpp_cls = _core.MacroF1


class MacroPrecision(_ClassificationMetric):
    _cpp_cls = _core.MacroPrecision


class MacroRecall(_ClassificationMetric):
    _cpp_cls = _core.MacroRecall


class MAE(_Metric):
    _cpp_cls = _core.MAE
    bigger_is_better = False


class RMSE(_Metric):
    _cpp_cls = _core.RMSE
    bigger_is_better = False


class R2(_Metric):
    _cpp_cls = _core.R2


class WindowMetric:
    """Accuracy / macro-F1 over a sliding window of predictions (C++ backend).

    Used by StreamingAutoGluon to weight each stacked learner's vote.
    """

    METRIC_ACCURACY = 0
    METRIC_F1 = 1

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._cpp = _core.WindowMetric(window_size=window_size)

    def update(self, predicted, actual):
        self._cpp.update(int(predicted), int(actual))
        return self

    def get(self, metric: int = METRIC_ACCURACY) -> float:
        return self._cpp.get(metric)

    def accuracy(self) -> float:
        return self._cpp.accuracy()

    def macro_f1(self) -> float:
        return self._cpp.macro_f1()

    @property
    def size(self) -> int:
        return self._cpp.size

    def reset(self):
        self._cpp.reset()
        return self


class WindowRegressionMetric:
    """MAE / RMSE over a sliding window of predictions (C++ backend).

    The regression counterpart of :class:`WindowMetric`; used by
    StreamingAutoGluonRegressor to weight each stacked regressor's prediction.
    """

    METRIC_MAE = 0
    METRIC_RMSE = 1

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._cpp = _core.WindowRegressionMetric(window_size=window_size)

    def update(self, y_true, y_pred):
        self._cpp.update(float(y_true), float(y_pred))
        return self

    def get(self, metric: int = METRIC_MAE) -> float:
        return self._cpp.get(metric)

    def mae(self) -> float:
        return self._cpp.mae()

    def rmse(self) -> float:
        return self._cpp.rmse()

    @property
    def size(self) -> int:
        return self._cpp.size

    def reset(self):
        self._cpp.reset()
        return self


class ADWIN:
    """Adaptive windowing drift detector (C++ backend)."""

    def __init__(self, delta: float = 0.002, clock: int = 32, max_buckets: int = 5,
                 min_window_length: int = 5, grace_period: int = 10):
        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        self.grace_period = grace_period
        self._cpp = _core.ADWIN(delta=delta, clock=clock, max_buckets=max_buckets,
                                min_window_length=min_window_length, grace_period=grace_period)

    def update(self, value):
        self._cpp.update(float(value))
        return self

    @property
    def drift_detected(self) -> bool:
        return self._cpp.drift_detected

    @property
    def estimation(self) -> float:
        return self._cpp.estimation

    @property
    def width(self) -> float:
        return self._cpp.width

    def reset(self):
        self._cpp.reset()
        return self


class EDDM:
    """Early Drift Detection Method (C++ backend). ``update(1)`` = error."""

    def __init__(self, warm_start: int = 30, alpha: float = 0.95, beta: float = 0.9):
        self.warm_start = warm_start
        self.alpha = alpha
        self.beta = beta
        self._cpp = _core.EDDM(warm_start=warm_start, alpha=alpha, beta=beta)

    def update(self, x):
        self._cpp.update(int(x))
        return self

    @property
    def drift_detected(self) -> bool:
        return self._cpp.drift_detected

    @property
    def warning_detected(self) -> bool:
        return self._cpp.warning_detected

    def reset(self):
        self._cpp.reset()
        return self


__all__ = [
    "Accuracy", "MacroF1", "MacroPrecision", "MacroRecall",
    "MAE", "RMSE", "R2", "WindowMetric", "WindowRegressionMetric",
    "ADWIN", "EDDM",
]
