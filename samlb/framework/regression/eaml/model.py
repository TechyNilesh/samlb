"""
EvoAutoML Regression — EvolutionaryBaggingRegressor.

Same Poisson-bagging + evolutionary mutation algorithm as the classifier
variant, but prediction is the **mean** of all population members' outputs
and the fitness metric is Mean Absolute Error (lower is better → we negate
scores before argmax).

Reference
---------
Kulbach, Cedric et al. "EvOAutoML: Evolutionary Automated Machine Learning
for Evolving Data Streams." (2022).
"""
from __future__ import annotations

import copy
import statistics
from typing import Any, Dict, List, Optional

import numpy as np
from river import metrics

from samlb.framework.base._framework import BaseStreamFramework
from .config import EAML_REG_PARAM_GRID


# ── Lightweight pipeline ──────────────────────────────────────────────────────

class _Pipeline:
    """A (scaler | regressor) pipeline supporting evolutionary mutation."""

    def __init__(self, scaler, regressor):
        self.scaler = copy.deepcopy(scaler)
        self.regressor = copy.deepcopy(regressor)

    def _set_params(self, params: dict) -> None:
        if "Scaler" in params:
            self.scaler = copy.deepcopy(params["Scaler"])
        if "Regressor" in params:
            self.regressor = copy.deepcopy(params["Regressor"])

    def predict_one(self, x: dict) -> Optional[float]:
        x_t = self.scaler.transform_one(x)
        return self.regressor.predict_one(x_t)

    def learn_one(self, x: dict, y: float) -> "_Pipeline":
        self.scaler.learn_one(x)
        x_t = self.scaler.transform_one(x)
        self.regressor.learn_one(x_t, y)
        return self


# ── Evolutionary Bagging Regressor ────────────────────────────────────────────

class EvolutionaryBaggingRegressor(BaseStreamFramework):
    """Evolutionary Bagging Regressor (SAMLB implementation of EvOAutoML).

    Each population member is a lightweight ``(scaler | regressor)``
    pipeline.  Fitness is tracked by MAE (lower is better).  The best
    pipeline (lowest MAE) is mutated; the worst is replaced.

    Parameters
    ----------
    population_size : int
        Number of pipelines in the population / ensemble.
    sampling_size : int
        Kept for API compatibility; one mutation per update step is used.
    sampling_rate : int
        Number of instances between evolutionary update steps.
    seed : int
        Random seed for reproducibility.
    param_grid : dict or None
        Custom parameter grid.  Keys must be "Scaler" and "Regressor";
        values are lists of instances.  Defaults to EAML_REG_PARAM_GRID.
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
        self.param_grid = param_grid if param_grid is not None else EAML_REG_PARAM_GRID

        self._rng = np.random.RandomState(seed)
        self._population: List[_Pipeline] = [
            self._random_pipeline() for _ in range(population_size)
        ]
        # MAE — lower is better; we negate when selecting best
        self._pop_metrics: List[metrics.MAE] = [
            metrics.MAE() for _ in range(population_size)
        ]
        self._i: int = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _random_pipeline(self) -> _Pipeline:
        scaler = self.param_grid["Scaler"][
            self._rng.randint(0, len(self.param_grid["Scaler"]))
        ]
        regressor = self.param_grid["Regressor"][
            self._rng.randint(0, len(self.param_grid["Regressor"]))
        ]
        return _Pipeline(scaler, regressor)

    def _mutate_pipeline(self, pipeline: _Pipeline) -> _Pipeline:
        """Deepcopy *pipeline*, then randomly swap one component."""
        child = copy.deepcopy(pipeline)
        key = self._rng.choice(["Scaler", "Regressor"])
        options = self.param_grid[key]
        value = options[self._rng.randint(0, len(options))]
        child._set_params({key: value})
        return child

    # ── BaseStreamFramework interface ─────────────────────────────────────────

    def predict_one(self, x: dict) -> float:
        """Return the mean prediction across all population members."""
        preds = [p.predict_one(x) for p in self._population]
        valid = [v for v in preds if v is not None]
        return statistics.mean(valid) if valid else 0.0

    def learn_one(self, x: dict, y: float) -> None:
        # Evolutionary update step (MAE: lower is better → negate for argmax)
        if self._i > 0 and self._i % self.sampling_rate == 0:
            scores = [-m.get() for m in self._pop_metrics]  # negate MAE
            idx_best = scores.index(max(scores))
            idx_worst = scores.index(min(scores))
            child = self._mutate_pipeline(self._population[idx_best])
            self._population[idx_worst] = child
            self._pop_metrics[idx_worst] = metrics.MAE()

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
        self._pop_metrics = [metrics.MAE() for _ in range(self.population_size)]
        self._i = 0
