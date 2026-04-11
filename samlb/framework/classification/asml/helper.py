"""Helper utilities for ASML Classification."""
from river import metrics
import numpy as np


def range_gen(min_n, max_n, step=1, float_n=False):
    """Generate a list of evenly-spaced values for hyperparameter search spaces."""
    if float_n:
        return [min_n + i * step for i in range(int((max_n - min_n) / step) + 1)]
    return list(range(min_n, max_n + 1, step))


class WindowClassificationPerformanceEvaluator:
    """
    Window-wise classification performance tracker.

    Resets the metric at the end of each window and records the score,
    giving a time-series of windowed accuracy (or any other metric).

    Parameters
    ----------
    metric : river metric, optional
        Classification metric.  Defaults to ``metrics.Accuracy()``.
    window_width : int
        Window size in instances.
    print_every : int
        Print progress every N instances (0 = silent).
    """

    def __init__(self, metric=None, window_width: int = 1000, print_every: int = 0):
        self.window_width = window_width
        self.metric = metric if metric is not None else metrics.Accuracy()
        self.print_every = print_every
        self.counter = 0
        self.scores_list: list = []

    def __repr__(self) -> str:
        val = np.mean(self.get()) * 100 if self.scores_list else 0.0
        return f"{type(self).__name__}({type(self.metric).__name__}): {val:.2f}%"

    def update(self, y_pred, y, sample_weight: float = 1.0):
        self.metric.update(y, y_pred)
        self.counter += 1
        if self.print_every and self.counter % self.print_every == 0:
            print(f"[{self.counter}] {self.metric}")
        if self.counter % self.window_width == 0:
            self.scores_list.append(self.metric.get())
            self.metric = type(self.metric)()

    def get(self) -> list:
        return self.scores_list
