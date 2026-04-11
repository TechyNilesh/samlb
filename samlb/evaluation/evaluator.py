"""
samlb.evaluation.evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~
PrequentialEvaluator — the core predict-then-learn evaluation loop.
WindowedEvaluator    — same loop, verbose=True by default (alias for intent).
"""
from __future__ import annotations

import math
import time
import traceback
from typing import Any, Dict, List, Optional

from samlb.datasets import stream as _stream
from .metrics import metrics_for_task, snapshot
from .results import RunResult


class _OnlineTargetScaler:
    """Welford running mean/variance for online target normalisation.

    Transforms targets to zero-mean, unit-variance using statistics
    observed *so far* (no future leakage).  Predictions are
    denormalised before metric evaluation so reported scores remain
    in the original target space.
    """

    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def std(self) -> float:
        if self._n < 2:
            return 1.0
        return math.sqrt(self._m2 / self._n)

    def transform(self, y: float) -> float:
        """Normalise *y* using current running statistics."""
        s = self.std
        if s == 0:
            return 0.0
        return (y - self._mean) / s

    def inverse_transform(self, y_norm: float) -> float:
        """Map a normalised prediction back to the original scale."""
        return y_norm * self.std + self._mean

    def update(self, y: float) -> None:
        """Update running mean/variance with a new observation."""
        self._n += 1
        delta = y - self._mean
        self._mean += delta / self._n
        delta2 = y - self._mean
        self._m2 += delta * delta2


class PrequentialEvaluator:
    """Standard prequential (test-then-train) evaluation loop.

    For each instance the model predicts *before* learning, so every
    prediction is made on unseen data.  All metrics are updated
    incrementally; a windowed snapshot is stored at every ``window_size``
    boundary so learning curves can be plotted afterward.

    Parameters
    ----------
    task : str
        "classification" or "regression".
    window_size : int
        Instances per evaluation window.  Default 1000.
    max_samples : int | None
        Optional row cap — useful for quick smoke tests.
    normalize : bool
        Passed to samlb.datasets.stream(); min-max scales features.
    normalize_target : bool
        If True (default for regression), apply online z-score
        normalisation to targets before ``learn_one`` and
        denormalise predictions before metric evaluation.  This
        prevents SGD-based learners from diverging on datasets
        with large or small target ranges.
    sample_runtime_every : int
        Record per-instance wall-clock time every N instances (default 100).
        Keeps the runtime list size bounded without adding overhead every call.
    verbose : bool
        Print a progress line at each window boundary.

    Example
    -------
        ev = PrequentialEvaluator(task="classification")
        result = ev.run(model, "electricity", "ASML")
        print(result.metrics)
        print(result.windowed_metrics["accuracy"])   # learning curve
    """

    def __init__(
        self,
        task: str,
        window_size: int = 1000,
        max_samples: Optional[int] = None,
        normalize: bool = False,
        normalize_target: Optional[bool] = None,
        sample_runtime_every: int = 100,
        verbose: bool = False,
    ):
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got {task!r}"
            )
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        if sample_runtime_every < 1:
            raise ValueError(
                f"sample_runtime_every must be >= 1, got {sample_runtime_every}."
            )
        self.task = task
        self.window_size = window_size
        self.max_samples = max_samples
        self.normalize = normalize
        # Default: normalise targets for regression, skip for classification
        self.normalize_target = (
            normalize_target if normalize_target is not None
            else (task == "regression")
        )
        self.sample_runtime_every = sample_runtime_every
        self.verbose = verbose

    def run(
        self,
        model: Any,
        dataset_name: str,
        framework_name: str,
    ) -> RunResult:
        """Execute a full prequential run and return a RunResult.

        The model is NOT reset here — BenchmarkSuite calls model.reset()
        before every run.  If you call this directly, ensure the model
        is in a fresh (untrained) state first.

        Exceptions inside the loop are caught; the RunResult carries the
        traceback in ``error`` and the suite continues to the next run.
        """
        cumulative  = metrics_for_task(self.task)   # final cumulative metrics
        window_live = metrics_for_task(self.task)   # reset at each window boundary

        windowed: Dict[str, List[float]] = {k: [] for k in cumulative}
        runtime_samples: List[float] = []

        t_total = time.perf_counter()
        n_samples = 0
        window_count = 0
        error_str: Optional[str] = None

        primary_key = "accuracy" if self.task == "classification" else "r2"

        # Online target normalisation for regression
        y_scaler = _OnlineTargetScaler() if self.normalize_target else None

        try:
            for x, y in _stream(
                dataset_name,
                task=self.task,
                max_samples=self.max_samples,
                normalize=self.normalize,
            ):
                t_inst = time.perf_counter()

                # Original y for metrics; normalised y for model training
                y_original = y
                if y_scaler is not None:
                    y_scaler.update(y_original)
                    y_norm = y_scaler.transform(y_original)
                else:
                    y_norm = y

                y_pred = model.predict_one(x)
                model.learn_one(x, y_norm)

                # None predictions come from OAML's warm-up phase —
                # we still count the instance but skip metric updates.
                if y_pred is not None:
                    # Denormalise prediction back to original scale
                    if y_scaler is not None:
                        y_pred = y_scaler.inverse_transform(float(y_pred))

                    # Clip predictions to a safe float range to prevent
                    # OverflowError in squared-error metrics.
                    _SAFE = 1e150
                    if isinstance(y_pred, (int, float)):
                        y_pred = max(-_SAFE, min(_SAFE, float(y_pred)))
                    for m in cumulative.values():
                        m.update(y_original, y_pred)
                    for m in window_live.values():
                        m.update(y_original, y_pred)

                n_samples += 1

                if n_samples % self.sample_runtime_every == 0:
                    runtime_samples.append(
                        round((time.perf_counter() - t_inst) * 1000, 4)
                    )

                if n_samples % self.window_size == 0:
                    window_count += 1
                    snap = snapshot(window_live)
                    for k, v in snap.items():
                        windowed[k].append(v)
                    window_live = metrics_for_task(self.task)   # fresh window
                    if self.verbose:
                        val = snap.get(primary_key, float("nan"))
                        print(
                            f"  [{framework_name}|{dataset_name}] "
                            f"w={window_count}  n={n_samples:,}  "
                            f"{primary_key}={val:.4f}"
                        )

            # Keep the final partial window so learning curves include tail behavior.
            if n_samples > 0 and (n_samples % self.window_size) != 0:
                snap = snapshot(window_live)
                for k, v in snap.items():
                    windowed[k].append(v)

        except Exception:
            error_str = traceback.format_exc()

        return RunResult(
            framework_name=framework_name,
            dataset_name=dataset_name,
            task=self.task,
            n_samples=n_samples,
            metrics=snapshot(cumulative),
            windowed_metrics=windowed,
            total_runtime_s=round(time.perf_counter() - t_total, 4),
            runtime_per_instance_ms=runtime_samples,
            error=error_str,
        )


class WindowedEvaluator(PrequentialEvaluator):
    """Prequential evaluator with verbose=True by default.

    Use this when you specifically want to observe the learning curve
    as the run progresses.  The returned RunResult is identical to
    PrequentialEvaluator — ``result.windowed_metrics`` holds the series.

    Example
    -------
        ev = WindowedEvaluator(task="classification", window_size=1000)
        result = ev.run(model, "electricity", "ASML")
        # result.windowed_metrics["accuracy"] → list of per-window scores
    """

    def __init__(self, task: str, window_size: int = 1000, **kwargs):
        kwargs.setdefault("verbose", True)
        super().__init__(task=task, window_size=window_size, **kwargs)
