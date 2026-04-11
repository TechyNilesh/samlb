"""
EvoAutoML Classification — EvolutionaryBaggingClassifier.

Algorithm (Poisson Oza Bagging + evolutionary mutation):
  1. Initialise a population of ``population_size`` (scaler | classifier) pipelines,
     each randomly sampled from ``param_grid``.
  2. Every ``sampling_rate`` instances:
       * Score each pipeline with its running Accuracy metric.
       * Deepcopy the *best* pipeline and mutate one component (Scaler or Classifier).
       * Replace the *worst* pipeline with the mutant; reset its metric.
  3. For each instance learn_one is called:
       * Update every pipeline's metric using its *current* prediction.
       * Train each pipeline with Poisson(6) passes (Oza bagging noise).
  4. predict_one: average class probabilities across all population members.

Reference
---------
Kulbach, Cedric et al. "EvOAutoML: Evolutionary Automated Machine Learning
for Evolving Data Streams." (2022).
"""
from __future__ import annotations

import collections
import copy
from typing import Any, Dict, List, Optional

import numpy as np
from river import metrics

from samlb.framework.base._framework import BaseStreamFramework
from .config import EAML_CLF_PARAM_GRID


# ── Lightweight pipeline ──────────────────────────────────────────────────────

class _Pipeline:
    """A (scaler | classifier) pipeline supporting evolutionary mutation."""

    def __init__(self, scaler, classifier):
        self.scaler = copy.deepcopy(scaler)
        self.classifier = copy.deepcopy(classifier)

    def _set_params(self, params: dict) -> None:
        """Replace scaler or classifier component (mutation step)."""
        if "Scaler" in params:
            self.scaler = copy.deepcopy(params["Scaler"])
        if "Classifier" in params:
            self.classifier = copy.deepcopy(params["Classifier"])

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


# ── Evolutionary Bagging Classifier ──────────────────────────────────────────

class EvolutionaryBaggingClassifier(BaseStreamFramework):
    """Evolutionary Bagging Classifier (SAMLB implementation of EvOAutoML).

    Each population member is a lightweight ``(scaler | classifier)``
    pipeline.  The evolutionary selection-mutation-replacement loop uses
    running Accuracy as fitness; Oza-style Poisson(6) bagging provides
    diversity during training.

    Parameters
    ----------
    population_size : int
        Number of pipelines in the population / ensemble.
    sampling_size : int
        Kept for API compatibility; only one mutation per update step is used.
    sampling_rate : int
        Number of instances between evolutionary update steps.
    seed : int
        Random seed for reproducibility.
    param_grid : dict or None
        Custom parameter grid.  Keys must be "Scaler" and "Classifier";
        values are lists of instances.  Defaults to EAML_CLF_PARAM_GRID.
    """

    exploration_window: int = 1000

    def __init__(
        self,
        population_size: int = 10,
        sampling_size: int = 1,
        sampling_rate: int = 1000,
        seed: int = 42,
        param_grid: Optional[Dict[str, List]] = None,
    ):
        self.population_size = population_size
        self.sampling_size = sampling_size
        self.sampling_rate = sampling_rate
        self.seed = seed
        self.param_grid = param_grid if param_grid is not None else EAML_CLF_PARAM_GRID

        self._rng = np.random.RandomState(seed)
        self._population: List[_Pipeline] = [
            self._random_pipeline() for _ in range(population_size)
        ]
        self._pop_metrics: List[metrics.Accuracy] = [
            metrics.Accuracy() for _ in range(population_size)
        ]
        self._i: int = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _random_pipeline(self) -> _Pipeline:
        scaler = self.param_grid["Scaler"][
            self._rng.randint(0, len(self.param_grid["Scaler"]))
        ]
        classifier = self.param_grid["Classifier"][
            self._rng.randint(0, len(self.param_grid["Classifier"]))
        ]
        return _Pipeline(scaler, classifier)

    def _mutate_pipeline(self, pipeline: _Pipeline) -> _Pipeline:
        """Deepcopy *pipeline*, then randomly swap one component."""
        child = copy.deepcopy(pipeline)
        key = self._rng.choice(["Scaler", "Classifier"])
        options = self.param_grid[key]
        value = options[self._rng.randint(0, len(options))]
        child._set_params({key: value})
        return child

    # ── BaseStreamFramework interface ─────────────────────────────────────────

    def predict_proba_one(self, x: dict) -> dict:
        """Average class probabilities across all population members."""
        y_pred: collections.Counter = collections.Counter()
        for pipeline in self._population:
            probas = pipeline.predict_proba_one(x)
            if probas:
                y_pred.update(probas)
        total = sum(y_pred.values())
        if total > 0:
            return {label: prob / total for label, prob in y_pred.items()}
        return dict(y_pred)

    def predict_one(self, x: dict) -> Any:
        """Predict the class with the highest averaged probability."""
        probas = self.predict_proba_one(x)
        if not probas:
            return None
        return max(probas, key=probas.get)

    def learn_one(self, x: dict, y: Any) -> None:
        # Evolutionary update step
        if self._i > 0 and self._i % self.sampling_rate == 0:
            scores = [m.get() for m in self._pop_metrics]
            idx_best = scores.index(max(scores))
            idx_worst = scores.index(min(scores))
            child = self._mutate_pipeline(self._population[idx_best])
            self._population[idx_worst] = child
            self._pop_metrics[idx_worst] = metrics.Accuracy()

        # Oza bagging: each pipeline receives Poisson(6) training passes
        for idx, pipeline in enumerate(self._population):
            y_pred = pipeline.predict_one(x)
            if y_pred is not None:
                self._pop_metrics[idx].update(y_true=y, y_pred=y_pred)
            k = int(self._rng.poisson(6))
            for _ in range(k):
                pipeline.learn_one(x, y)

        self._i += 1

    def reset(self) -> None:
        """Reset the ensemble to its initial (untrained) state."""
        self._rng = np.random.RandomState(self.seed)
        self._population = [self._random_pipeline() for _ in range(self.population_size)]
        self._pop_metrics = [metrics.Accuracy() for _ in range(self.population_size)]
        self._i = 0
