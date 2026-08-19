"""
StreamingAutoGluon Regression — online stacking of k-fold cross-validated
stream regressors.

Regression counterpart of
:class:`samlb.framework.classification.sag.StreamingAutoGluon`. The Java
reference is classification-only; the structure carries over unchanged and only
the two class-specific pieces are restated for a continuous target:

* **Meta features.** Classification contributes one feature per (type, class)
  — the averaged class probabilities of that type's fold learners. Regression
  has no classes, so each type contributes a single feature: the average
  prediction of its ``k`` fold learners.
* **Vote weighting.** Classification weights each stacked learner by its
  windowed accuracy or macro F1 (higher is better). Regression weights by
  ``1 / (eps + windowed error)`` with MAE or RMSE, so a lower error earns a
  larger share, and the final prediction is the weighted mean of the stacked
  predictions.

Everything else is identical: ``k`` fold learners per type trained with online
k-fold cross-validation (the learner whose index equals ``instances_seen % k``
is held out), one stacked learner per type trained on the augmented instance,
fold learners predicting before any of them trains so the meta features stay
out-of-fold, and the per-instance vote cache.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from samlb.base import Pipeline
from samlb.framework.base._framework import BaseStreamFramework
from samlb.metrics import WindowRegressionMetric

from .config import SAG_REG_BASE_LEARNERS, SAG_REG_SCALERS

_METRICS = {
    "mae": WindowRegressionMetric.METRIC_MAE,
    "rmse": WindowRegressionMetric.METRIC_RMSE,
}

# Guards 1/error when a stacked regressor is (near) perfect on the window.
_EPS = 1e-9


class StreamingAutoGluonRegressor(BaseStreamFramework):
    """Online stacking of k-fold cross-validated stream regressors.

    Parameters
    ----------
    learners : list, optional
        Base learner prototypes, one per type. Defaults to
        :data:`SAG_REG_BASE_LEARNERS` (the shared C++ regression pool).
    scalers : list, estimator or None
        Preprocessors, assigned round-robin across the base types and fused
        into each learner's pipeline. Defaults to :data:`SAG_REG_SCALERS`;
        ``None`` feeds raw features.
    n_folds : int
        Number of cross-validation folds ``k`` (default 3).
    metric : {"mae", "rmse"}
        Windowed error used to weight the stacked regressors.
    window_size : int
        Sliding window over which that error is estimated.
    clip : bool
        Clip predictions to the target range seen so far. Matches the option
        the other SAMLB regression frameworks expose.
    seed : int or None
        Kept for API symmetry; the method itself is deterministic.
    """

    exploration_window: int = 1000

    def __init__(
        self,
        learners: Optional[List] = None,
        scalers: Any = "default",
        n_folds: int = 3,
        metric: str = "mae",
        window_size: int = 1000,
        clip: bool = True,
        seed: Optional[int] = 42,
    ):
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {sorted(_METRICS)}, got {metric!r}.")

        self.learners = learners if learners is not None else SAG_REG_BASE_LEARNERS
        if isinstance(scalers, str) and scalers == "default":
            scalers = SAG_REG_SCALERS
        if scalers is None:
            self.scalers: List = []
        elif isinstance(scalers, (list, tuple)):
            self.scalers = list(scalers)
        else:
            self.scalers = [scalers]
        self.n_folds = n_folds
        self.metric = metric
        self.window_size = window_size
        self.clip = clip
        self.seed = seed

        if not self.learners:
            raise ValueError("StreamingAutoGluonRegressor requires at least one base type.")

        self.reset()

    # ── construction ─────────────────────────────────────────────────────────

    @staticmethod
    def _unique_names(learners) -> List[str]:
        """Distinct name per base type — meta-feature keys are built from these."""
        names, counts = [], {}
        for learner in learners:
            base = type(learner).__name__
            counts[base] = counts.get(base, 0) + 1
            names.append(base if counts[base] == 1 else f"{base}{counts[base]}")
        return names

    def _build(self, prototype, type_index: int):
        learner = prototype.clone()
        if not self.scalers:
            return learner
        return Pipeline(self.scalers[type_index % len(self.scalers)].clone(), learner)

    def reset(self) -> None:
        self._names = self._unique_names(self.learners)
        self._n_types = len(self.learners)

        self._fold_learners = [
            [self._build(p, t) for _ in range(self.n_folds)]
            for t, p in enumerate(self.learners)
        ]
        self._stacked_learners = [
            self._build(p, t) for t, p in enumerate(self.learners)
        ]
        self._stacked_metrics = [
            WindowRegressionMetric(self.window_size) for _ in self.learners
        ]

        self._seen = 0
        self._ymin: Optional[float] = None
        self._ymax: Optional[float] = None
        self._clear_cache()

    # ── meta features ────────────────────────────────────────────────────────

    def _clear_cache(self) -> None:
        self._cached_x = None
        self._cached_votes = None

    def _fold_predictions(self, x: dict) -> List[List[float]]:
        """Predictions of every fold learner, cached per instance."""
        if self._cached_votes is not None and self._cached_x is x:
            return self._cached_votes

        preds = []
        for row in self._fold_learners:
            per_fold = []
            for learner in row:
                value = learner.predict_one(x)
                per_fold.append(float(value) if value is not None else 0.0)
            preds.append(per_fold)

        self._cached_x = x
        self._cached_votes = preds
        return preds

    def _aggregate(self, fold_predictions) -> List[float]:
        """Average, per type, the predictions of that type's k fold learners."""
        return [sum(per_fold) / self.n_folds for per_fold in fold_predictions]

    def _augment(self, x: dict, aggregated: List[float]) -> dict:
        """Raw features plus one meta feature per base type."""
        out = dict(x)
        for name, value in zip(self._names, aggregated):
            out[f"P_{name}"] = value
        return out

    # ── BaseStreamFramework interface ────────────────────────────────────────

    def learn_one(self, x: dict, y: Any) -> None:
        y = float(y)
        self._ymin = y if self._ymin is None else min(self._ymin, y)
        self._ymax = y if self._ymax is None else max(self._ymax, y)

        held_out = self._seen % self.n_folds

        # 1) Fold learners predict before being trained on the instance.
        fold_predictions = self._fold_predictions(x)

        # 2) Stacked learners are scored, then trained, on the augmented instance.
        augmented = self._augment(x, self._aggregate(fold_predictions))
        for t in range(self._n_types):
            prediction = self._stacked_learners[t].predict_one(augmented)
            if prediction is not None:
                self._stacked_metrics[t].update(y, float(prediction))
            self._stacked_learners[t].learn_one(augmented, y)

        # 3) Fold learners train on the raw instance, except the held-out one.
        for row in self._fold_learners:
            for i, learner in enumerate(row):
                if i != held_out:
                    learner.learn_one(x, y)

        self._seen += 1
        self._clear_cache()

    def predict_one(self, x: dict) -> float:
        augmented = self._augment(x, self._aggregate(self._fold_predictions(x)))
        metric = _METRICS[self.metric]

        weighted = 0.0
        total_weight = 0.0
        plain: List[float] = []

        for t in range(self._n_types):
            prediction = self._stacked_learners[t].predict_one(augmented)
            if prediction is None:
                continue
            prediction = float(prediction)
            plain.append(prediction)

            window = self._stacked_metrics[t]
            if window.size > 0:
                # Lower error earns a larger share.
                weight = 1.0 / (_EPS + window.get(metric))
                weighted += prediction * weight
                total_weight += weight

        if total_weight > 0.0:
            prediction = weighted / total_weight
        elif plain:
            # No error estimate yet: average uniformly, as the classification
            # version votes uniformly.
            prediction = sum(plain) / len(plain)
        else:
            prediction = 0.0

        if self.clip and self._ymin is not None:
            prediction = max(self._ymin, min(self._ymax, prediction))
        return prediction

    # ── diagnostics ──────────────────────────────────────────────────────────

    def stacked_weights(self) -> Dict[str, float]:
        """Normalised weight currently given to each stacked regressor."""
        metric = _METRICS[self.metric]
        raw = {
            name: (1.0 / (_EPS + m.get(metric)) if m.size > 0 else 0.0)
            for name, m in zip(self._names, self._stacked_metrics)
        }
        total = sum(raw.values())
        return {k: (v / total if total > 0 else 0.0) for k, v in raw.items()}

    def stacked_errors(self) -> Dict[str, float]:
        """Windowed error of each stacked regressor."""
        metric = _METRICS[self.metric]
        return {
            name: m.get(metric) for name, m in zip(self._names, self._stacked_metrics)
        }

    def __repr__(self) -> str:
        return (
            f"StreamingAutoGluonRegressor(types={self._names}, n_folds={self.n_folds}, "
            f"metric={self.metric!r}, window_size={self.window_size}, "
            f"scalers={[type(s).__name__ for s in self.scalers]})"
        )
