"""
ASML Classification — AutoStreamClassifier.

Adaptive Streaming Machine Learning for classification tasks.
Uses the ARDNS (Adaptive Random Drift Nearby Search) pipeline search strategy
to continuously adapt the best-performing pipeline as the data stream evolves.

Improvements over the base ARDNS design:
  - ADWIN-based drift detection triggers aggressive re-exploration
  - Recency-weighted ensemble voting (recent snapshots count more)
  - Adaptive random/ARDNS budget split based on performance drop
"""
from __future__ import annotations

import random
from collections import Counter

import numpy as np
from river import metrics
from river.drift import ADWIN

from samlb.framework.base._framework import BaseStreamFramework
from .search import PipelineSearch


class AutoStreamClassifier(BaseStreamFramework):
    """
    Automated streaming classifier with ARDNS pipeline search and drift detection.

    At every ``exploration_window`` instance, the classifier:
      1. Identifies the best-performing pipeline so far.
      2. Generates ARDNS neighbours of the best pipeline.
      3. Adds fresh random pipelines to maintain diversity.
      4. In ensemble mode, replaces the worst snapshot with the new best.

    When the ADWIN drift detector fires, the classifier:
      - Temporarily increases the random pipeline fraction to 80%
      - Halves the exploration window for faster adaptation
      - Restores normal settings after one stable window

    Parameters
    ----------
    config_dict : dict, optional
        Custom search-space dict with keys ``models``, ``preprocessors``,
        ``features``, ``hyperparameters``.  Defaults to the ASML config.
    metric : river metric
        Metric used to compare pipelines (default: ``Accuracy``).
    exploration_window : int
        Instances per search cycle.
    budget : int
        Total pipelines to track simultaneously.
    ensemble_size : int
        Number of model snapshots kept for ensemble prediction.
    prediction_mode : str
        ``'ensemble'`` (default) or ``'best'``.
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
        verbose: bool = False,
        seed: int | None = 42,
    ):
        if prediction_mode not in ("ensemble", "best"):
            raise ValueError("prediction_mode must be 'ensemble' or 'best'")

        self.config_dict = config_dict
        self.metric = metric if metric is not None else metrics.Accuracy()
        self.exploration_window = exploration_window
        self._base_exploration_window = exploration_window
        self.budget = budget
        self.ensemble_size = ensemble_size
        self.prediction_mode = prediction_mode
        self.verbose = verbose
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self._counter = 0
        self._drift_detector = ADWIN()
        self._drift_mode = False
        self._prev_best_score = 0.0

        self._pipe_search = PipelineSearch(
            config_dict=self.config_dict,
            budget=self.budget - 1,
            seed=self.seed,
        )
        self._pipelines = self._pipe_search._create_pipelines()
        self._metrics = [type(self.metric)() for _ in self._pipelines]
        self._best_idx = np.random.randint(len(self._pipelines))
        self.best_model = self._pipelines[self._best_idx]

        if self.prediction_mode == "ensemble":
            self._snapshots = [
                self._pipelines[np.random.randint(len(self._pipelines))].clone()
                for _ in range(self.ensemble_size)
            ]
            self._snap_metrics = [type(self.metric)() for _ in self._snapshots]

    # ── public interface ──────────────────────────────────────────────────────

    def predict_one(self, x: dict):
        if self.prediction_mode == "ensemble":
            votes: Counter = Counter()
            n = len(self._snapshots)
            for i, clf in enumerate(self._snapshots):
                # Recency weight: newer snapshots count more
                weight = (i + 1) / n
                try:
                    pred = clf.predict_one(x)
                    votes[pred] += weight
                except Exception:
                    pass
            return votes.most_common(1)[0][0] if votes else 0
        else:
            try:
                return self._pipelines[self._best_idx].predict_one(x)
            except Exception:
                return 0

    def learn_one(self, x: dict, y) -> None:
        try:
            best_pred = self._pipelines[self._best_idx].predict_one(x)
        except Exception:
            best_pred = None

        # Update all competing pipelines
        for idx, pipe in enumerate(self._pipelines):
            try:
                y_pred = pipe.predict_one(x)
                self._metrics[idx].update(y, y_pred)
                pipe.learn_one(x, y)
                if self._metrics[idx].is_better_than(self._metrics[self._best_idx]):
                    self._best_idx = idx
            except Exception:
                pass

        # Feed drift detector with best pipeline's correctness
        if best_pred is not None:
            self._drift_detector.update(int(best_pred == y))
        self.best_model = self._pipelines[self._best_idx]

        # Update ensemble snapshots
        if self.prediction_mode == "ensemble":
            for idx, snap in enumerate(self._snapshots):
                try:
                    y_pred = snap.predict_one(x)
                    self._snap_metrics[idx].update(y, y_pred)
                    snap.learn_one(x, y)
                except Exception:
                    pass

        self._counter += 1

        # Check for drift (skip first 2 windows to let pipelines stabilise)
        if self._drift_detector.drift_detected and self._counter > 2 * self._base_exploration_window:
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
            verbose=self.verbose,
            seed=self.seed,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _on_drift(self):
        """React to detected concept drift: boost exploration."""
        self._drift_mode = True
        # Temporarily halve exploration window for faster adaptation
        self.exploration_window = max(200, self._base_exploration_window // 2)
        # Shift budget toward random pipelines (80% random)
        self._pipe_search.random_budget = int(self._pipe_search.budget * 0.8)
        self._pipe_search.ardns_budget = self._pipe_search.budget - self._pipe_search.random_budget
        # Force immediate re-exploration
        self.best_model = self._pipelines[self._best_idx]
        self._pipelines = self._pipe_search.select_and_update_pipelines(self.best_model)
        self._reset_cycle()

    def _reset_cycle(self):
        self._metrics = [type(self.metric)() for _ in self._pipelines]
        self._best_idx = 0
        self.best_model = self._pipelines[self._best_idx]
        if self.prediction_mode == "ensemble":
            self._snap_metrics = [type(self.metric)() for _ in self._snapshots]

    def _maybe_explore(self):
        if self._counter % self.exploration_window != 0:
            return

        self.best_model = self._pipelines[self._best_idx]
        current_score = self._metrics[self._best_idx].get()

        # Adaptive budget: if performance dropped, increase random fraction
        if self._prev_best_score > 0 and current_score < self._prev_best_score * 0.95:
            self._pipe_search.random_budget = int(self._pipe_search.budget * 0.7)
            self._pipe_search.ardns_budget = self._pipe_search.budget - self._pipe_search.random_budget
        elif self._drift_mode:
            # Restore normal budget after one stable window post-drift
            self._drift_mode = False
            self.exploration_window = self._base_exploration_window
            self._pipe_search.random_budget = (self._pipe_search.budget + 1) // 2
            self._pipe_search.ardns_budget = self._pipe_search.budget - self._pipe_search.random_budget
        self._prev_best_score = current_score

        if self.prediction_mode == "ensemble":
            if len(self._snapshots) >= self.ensemble_size:
                worst = int(np.argmin([m.get() for m in self._snap_metrics]))
                self._snapshots.pop(worst)
                self._snap_metrics.pop(worst)
            self._snapshots.append(self.best_model.clone())
            self._snap_metrics.append(type(self.metric)())

        if self.verbose:
            self._print_info()

        self._pipelines = self._pipe_search.select_and_update_pipelines(self.best_model)
        self._reset_cycle()

    def _print_info(self):
        drift_str = " [DRIFT]" if self._drift_mode else ""
        print(f"[ASML-CLS] step={self._counter}  best={self.best_model}{drift_str}")
        print("-" * 70)
