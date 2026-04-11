"""
ASML Regression — AutoStreamRegressor.

Adaptive Streaming Machine Learning for regression tasks.
Uses ARDNS pipeline search with ensemble or best-model prediction.

Improvements:
  - ADWIN drift detection on prediction error
  - Prediction clipping to prevent divergence
  - Recency-weighted ensemble (recent snapshots count more)
  - Adaptive exploration budget on performance drop
"""
from __future__ import annotations

import math
import random

import numpy as np
from river import metrics
from river.drift import ADWIN

from samlb.framework.base._framework import BaseStreamFramework
from .search import PipelineSearch


_NORM_CLIP = 1e6  # in normalised space: if prediction > 1M std devs, it's diverged


class _RunningNorm:
    """Welford online mean/std for internal target normalisation."""
    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self._m2 / self._n) if self._n >= 2 else 1.0

    def update(self, y: float):
        self._n += 1
        d = y - self._mean
        self._mean += d / self._n
        self._m2 += d * (y - self._mean)

    def normalize(self, y: float) -> float:
        s = self.std
        return (y - self._mean) / s if s > 0 else 0.0

    def denormalize(self, y_norm: float) -> float:
        return y_norm * self.std + self._mean


class AutoStreamRegressor(BaseStreamFramework):
    """
    Automated streaming regressor with ARDNS pipeline search and drift detection.

    Parameters
    ----------
    config_dict : dict, optional
        Custom search space dict.
    metric : river metric
        Metric used to compare pipelines (default: ``RMSE``).
    exploration_window : int
        Instances per search cycle.
    budget : int
        Total pipelines to track simultaneously.
    ensemble_size : int
        Number of model snapshots for ensemble prediction.
    prediction_mode : str
        ``'ensemble'`` (default) or ``'best'``.
    feature_selection : bool
        Include a feature-selection step in pipelines.
    verbose : bool
        Print per-window diagnostics.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        config_dict=None,
        metric=None,
        exploration_window: int = 1000,
        budget: int = 10,
        ensemble_size: int = 3,
        prediction_mode: str = "ensemble",
        feature_selection: bool = True,
        verbose: bool = False,
        seed: int | None = 42,
    ):
        if prediction_mode not in ("ensemble", "best"):
            raise ValueError("prediction_mode must be 'ensemble' or 'best'")

        self.config_dict = config_dict
        self.metric = metric if metric is not None else metrics.RMSE()
        self.exploration_window = exploration_window
        self._base_exploration_window = exploration_window
        self.budget = budget
        self.ensemble_size = ensemble_size
        self.prediction_mode = prediction_mode
        self.feature_selection = feature_selection
        self.verbose = verbose
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self._counter = 0
        self._drift_detector = ADWIN()
        self._drift_mode = False
        self._prev_best_score = None
        self._y_norm = _RunningNorm()
        self._dead_pipes: set = set()  # indices of diverged pipelines

        self._pipe_search = PipelineSearch(
            config_dict=self.config_dict,
            feature_selection=self.feature_selection,
            budget=self.budget - 1,
            seed=self.seed,
        )
        self._pipelines = self._pipe_search._create_pipelines()
        self._metrics = [type(self.metric)() for _ in self._pipelines]
        self._best_idx = np.random.randint(len(self._pipelines))
        self.best_model = self._pipelines[self._best_idx]

        if self.prediction_mode == "ensemble":
            self._snapshots = [
                self._pipelines[np.random.randint(len(self._pipelines))]
                for _ in range(self.ensemble_size)
            ]
            self._snap_metrics = [type(self.metric)() for _ in self._snapshots]

    # ── public interface ──────────────────────────────────────────────────────

    @staticmethod
    def _clip(v: float) -> float:
        """Clip prediction to prevent divergence."""
        if not math.isfinite(v):
            return 0.0
        return max(-_NORM_CLIP, min(_NORM_CLIP, v))

    @staticmethod
    def _is_diverged(v: float) -> bool:
        """Check if a normalised prediction is divergent."""
        if not math.isfinite(v):
            return True
        return abs(v) > _NORM_CLIP

    def predict_one(self, x: dict) -> float:
        if self.prediction_mode == "ensemble":
            preds = []
            n = len(self._snapshots)
            weights = []
            for i, reg in enumerate(self._snapshots):
                try:
                    p = reg.predict_one(x)
                    if self._is_diverged(p):
                        continue  # skip diverged snapshot
                    preds.append(self._y_norm.denormalize(p))
                    weights.append((i + 1) / n)
                except Exception:
                    pass
            if not preds:
                return self._y_norm._mean  # fallback to running mean
            w_sum = sum(weights)
            return self._clip(sum(p * w for p, w in zip(preds, weights)) / w_sum)
        else:
            try:
                p = self.best_model.predict_one(x)
                if self._is_diverged(p):
                    return self._y_norm._mean
                return self._clip(self._y_norm.denormalize(p))
            except Exception:
                return self._y_norm._mean

    def learn_one(self, x: dict, y: float) -> None:
        # Update running normaliser and get normalised target
        self._y_norm.update(y)
        y_n = self._y_norm.normalize(y)

        for idx, pipe in enumerate(self._pipelines):
            if idx in self._dead_pipes:
                continue  # permanently disabled
            try:
                y_pred_n = pipe.predict_one(x)
                if self._is_diverged(y_pred_n):
                    self._dead_pipes.add(idx)
                    continue
                y_pred = self._y_norm.denormalize(y_pred_n)
                self._metrics[idx].update(y, y_pred)
                pipe.learn_one(x, y_n)
                if self._metrics[idx].is_better_than(self._metrics[self._best_idx]):
                    self._best_idx = idx
            except Exception:
                self._dead_pipes.add(idx)

        # Feed drift detector with bounded squared error
        try:
            best_pred_n = self._clip(self._pipelines[self._best_idx].predict_one(x))
            best_pred = self._y_norm.denormalize(best_pred_n)
            err = min((y - best_pred) ** 2, 1e10)
            self._drift_detector.update(err)
        except Exception:
            pass

        if self.prediction_mode == "ensemble":
            for idx, snap in enumerate(self._snapshots):
                try:
                    y_pred_n = snap.predict_one(x)
                    if self._is_diverged(y_pred_n):
                        continue
                    y_pred = self._y_norm.denormalize(y_pred_n)
                    self._snap_metrics[idx].update(y, y_pred)
                    snap.learn_one(x, y_n)
                except Exception:
                    pass
        else:
            try:
                self.best_model.learn_one(x, y_n)
            except Exception:
                pass

        self._counter += 1

        if self._drift_detector.drift_detected:
            self._on_drift()

        self._maybe_explore()

    def reset(self) -> None:
        self.__init__(
            config_dict=self.config_dict,
            metric=type(self.metric)(),
            exploration_window=self._base_exploration_window,
            budget=self.budget,
            ensemble_size=self.ensemble_size,
            prediction_mode=self.prediction_mode,
            feature_selection=self.feature_selection,
            verbose=self.verbose,
            seed=self.seed,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _on_drift(self):
        """React to detected concept drift: boost exploration."""
        self._drift_mode = True
        self.exploration_window = max(200, self._base_exploration_window // 2)
        self._pipe_search.random_budget = int(self._pipe_search.budget * 0.8)
        self._pipe_search.ardns_budget = self._pipe_search.budget - self._pipe_search.random_budget
        self.best_model = self._pipelines[self._best_idx]
        self._pipelines = self._pipe_search.select_and_update_pipelines(self.best_model)
        self._reset_cycle()

    def _reset_cycle(self):
        self._metrics = [type(self.metric)() for _ in self._pipelines]
        self._dead_pipes = set()  # new pipelines get a fresh chance
        self._best_idx = np.random.randint(len(self._pipelines))
        if self.prediction_mode == "ensemble":
            self._snap_metrics = [type(self.metric)() for _ in self._snapshots]

    def _is_better(self, m_new, m_old) -> bool:
        name = type(self.metric).__name__
        if name in ("R2",):
            return m_new.get() > m_old.get()
        return m_new.get() < m_old.get()

    def _worst_snapshot_idx(self) -> int:
        name = type(self.metric).__name__
        scores = [m.get() for m in self._snap_metrics]
        return int(np.argmin(scores) if name in ("R2",) else np.argmax(scores))

    def _maybe_explore(self):
        if self._counter % self.exploration_window != 0:
            return

        self.best_model = self._pipelines[self._best_idx]
        current_score = self._metrics[self._best_idx].get()

        # Adaptive budget on performance drop
        if self._prev_best_score is not None:
            is_r2 = type(self.metric).__name__ in ("R2",)
            if is_r2:
                dropped = current_score < self._prev_best_score * 0.9
            else:
                dropped = current_score > self._prev_best_score * 1.1
            if dropped:
                self._pipe_search.random_budget = int(self._pipe_search.budget * 0.7)
                self._pipe_search.ardns_budget = self._pipe_search.budget - self._pipe_search.random_budget
            elif self._drift_mode:
                self._drift_mode = False
                self.exploration_window = self._base_exploration_window
                self._pipe_search.random_budget = (self._pipe_search.budget + 1) // 2
                self._pipe_search.ardns_budget = self._pipe_search.budget - self._pipe_search.random_budget
        self._prev_best_score = current_score

        if self.prediction_mode == "ensemble":
            if len(self._snapshots) >= self.ensemble_size:
                worst = self._worst_snapshot_idx()
                self._snapshots.pop(worst)
                self._snap_metrics.pop(worst)
            self._snapshots.append(self.best_model)
            self._snap_metrics.append(type(self.metric)())

        if self.verbose:
            drift_str = " [DRIFT]" if self._drift_mode else ""
            print(f"[ASML-REG] step={self._counter}  best={self.best_model}{drift_str}")
            print("-" * 70)

        self._pipelines = self._pipe_search.select_and_update_pipelines(self.best_model)
        self._reset_cycle()
