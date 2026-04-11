"""
OAML — Online AutoML Classifier.

Inspired by:
  Gama, J. et al. "OAML: Online AutoML for Data Streams."
  https://github.com/gama-platform/OAML

Architecture
------------
Phase 1 — Warm-up / Initial Search
  Buffer the first ``initial_batch_size`` instances.  Then run a random
  search: sample ``budget`` (scaler | classifier) pipelines, train each on
  the first 80 % of the buffer, evaluate on the remaining 20 %.  Deploy the
  best pipeline found (highest accuracy on hold-out split).

Phase 2 — Online Streaming
  For every new instance:
    1. Predict using the current best pipeline.
    2. Learn (update scaler + classifier).
    3. Update the EDDM drift detector with the binary classification error.
    4. If drift detected → trigger re-search on the sliding window buffer.

Phase 3 — Drift-triggered Re-search
  When drift is detected, re-run the same random search on the current
  sliding window of ``window_size`` instances and deploy the new best
  pipeline.  The drift detector is reset.

Key differences from the original OAML
---------------------------------------
  * GAMA (genetic programming) is replaced by a lightweight uniform random
    search — no external AutoML dependency.
  * skmultiflow is replaced by ``river.drift.EDDM``.
  * The pipeline search space uses SAMLB C++ wrappers instead of scikit-learn.
"""
from __future__ import annotations

import collections
import copy
from typing import Any, Deque, Optional, Tuple

import numpy as np
from river.drift.binary import EDDM as _EDDM

from samlb.framework.base._framework import BaseStreamFramework
from .config import OAML_CLASSIFIERS, OAML_SCALERS


# ── Lightweight pipeline ──────────────────────────────────────────────────────

class _Pipeline:
    """A (scaler | classifier) pipeline."""

    def __init__(self, scaler, classifier):
        self.scaler = copy.deepcopy(scaler)
        self.classifier = copy.deepcopy(classifier)

    def predict_one(self, x: dict) -> Any:
        x_t = self.scaler.transform_one(x)
        return self.classifier.predict_one(x_t)

    def predict_proba_one(self, x: dict) -> dict:
        x_t = self.scaler.transform_one(x)
        return self.classifier.predict_proba_one(x_t)

    def learn_one(self, x: dict, y: Any) -> "_Pipeline":
        self.scaler.learn_one(x)
        x_t = self.scaler.transform_one(x)
        self.classifier.learn_one(x_t, y)
        return self


# ── OAML Classifier ───────────────────────────────────────────────────────────

class OAMLClassifier(BaseStreamFramework):
    """Online AutoML Classifier with drift-triggered pipeline search.

    Parameters
    ----------
    initial_batch_size : int
        Number of instances buffered before the first search.
    budget : int
        Number of random pipelines evaluated per search round.
    window_size : int
        Size of the sliding window kept for drift-triggered re-search.
    train_split : float
        Fraction of the buffer used for training during search (rest = eval).
    seed : int
        Random seed for reproducibility.
    """

    exploration_window: int = 200
    budget: int = 20

    def __init__(
        self,
        initial_batch_size: int = 200,
        budget: int = 20,
        window_size: int = 500,
        train_split: float = 0.8,
        seed: int = 42,
        scalers: Optional[list] = None,
        classifiers: Optional[list] = None,
    ):
        self.initial_batch_size = initial_batch_size
        self.budget = budget
        self.window_size = window_size
        self.train_split = train_split
        self.seed = seed
        self.scalers     = scalers     if scalers     is not None else OAML_SCALERS
        self.classifiers = classifiers if classifiers is not None else OAML_CLASSIFIERS

        self._rng = np.random.RandomState(seed)
        self._warm_buffer: list = []
        self._sliding_window: Deque[Tuple] = collections.deque(maxlen=window_size)
        self._current: Optional[_Pipeline] = None
        self._drift_detector = _EDDM()
        self._warmed_up: bool = False

    # ── internal helpers ──────────────────────────────────────────────────────

    def _random_pipeline(self) -> _Pipeline:
        scaler     = self.scalers[self._rng.randint(0, len(self.scalers))]
        classifier = self.classifiers[self._rng.randint(0, len(self.classifiers))]
        return _Pipeline(scaler, classifier)

    def _run_search(self, data: list) -> Optional[_Pipeline]:
        """Evaluate ``budget`` random pipelines on *data*; return the best."""
        n = len(data)
        if n < 4:
            return None

        split = max(1, int(n * self.train_split))
        train_data = data[:split]
        eval_data = data[split:]

        best_acc = -1.0
        best_pipeline: Optional[_Pipeline] = None

        for _ in range(self.budget):
            pipeline = self._random_pipeline()

            # Train on training split
            for x, y in train_data:
                pipeline.learn_one(x, y)

            # Evaluate on hold-out split
            correct = sum(
                1 for x, y in eval_data if pipeline.predict_one(x) == y
            )
            acc = correct / len(eval_data)

            if acc > best_acc:
                best_acc = acc
                best_pipeline = pipeline

        return best_pipeline

    def _trigger_search(self) -> None:
        """Run a drift-triggered re-search on the sliding window."""
        data = list(self._sliding_window)
        new_pipeline = self._run_search(data)
        if new_pipeline is not None:
            self._current = new_pipeline
        # Always reset drift detector after a search
        self._drift_detector = _EDDM()

    # ── BaseStreamFramework interface ─────────────────────────────────────────

    def predict_one(self, x: dict) -> Any:
        """Predict with the current best pipeline (None during warm-up)."""
        if self._current is None:
            return None
        return self._current.predict_one(x)

    def learn_one(self, x: dict, y: Any) -> None:
        self._sliding_window.append((x, y))

        if not self._warmed_up:
            # ── Warm-up phase ─────────────────────────────────────────────────
            self._warm_buffer.append((x, y))

            if len(self._warm_buffer) >= self.initial_batch_size:
                # Run initial search on warm-up buffer
                best = self._run_search(self._warm_buffer)
                if best is not None:
                    self._current = best
                self._warmed_up = True
                self._warm_buffer = []   # free memory
        else:
            # ── Online phase ─────────────────────────────────────────────────
            y_pred = self._current.predict_one(x) if self._current else None
            self._current.learn_one(x, y)

            # Update drift detector with classification error
            if y_pred is not None:
                error = 0 if y_pred == y else 1
                self._drift_detector.update(error)
                if self._drift_detector.drift_detected:
                    self._trigger_search()

    def reset(self) -> None:
        """Reset to the initial (untrained) state."""
        self._rng = np.random.RandomState(self.seed)
        self._warm_buffer = []
        self._sliding_window = collections.deque(maxlen=self.window_size)
        self._current = None
        self._drift_detector = _EDDM()
        self._warmed_up = False
