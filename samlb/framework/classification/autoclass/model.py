"""
AutoClass Classification — Genetic Algorithm AutoML.

Maintains a population of streaming classifiers and evolves them via
genetic mutation guided by a meta-regressor (AdaptiveRandomForest).
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from river import metrics, forest

from samlb.framework.base._framework import BaseStreamFramework
from .config import default_config_dict


class AutoClass(BaseStreamFramework):
    """
    Genetic-algorithm AutoML for streaming classification.

    Each exploration cycle:
      1. Selects a parent model by fitness-proportionate selection.
      2. Mutates its hyperparameters using a truncated-normal distribution.
      3. Predicts the mutant's fitness with a surrogate meta-regressor (ARF).
      4. Replaces the worst population member if the mutant is predicted better.

    Parameters
    ----------
    config_dict : dict, optional
        Custom search space with keys ``algorithms`` and ``hyperparameters``.
    metric : river metric
        Fitness metric (default: ``Accuracy``).
    exploration_window : int
        Instances per evolution cycle.
    population_size : int
        Number of models in the population.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        config_dict=None,
        metric=None,
        exploration_window: int = 1000,
        population_size: int = 10,
        seed: int | None = 42,
    ):
        self.config_dict = config_dict
        self.metric = metric if metric is not None else metrics.Accuracy()
        self.exploration_window = exploration_window
        self.population_size = population_size
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        cfg = config_dict if config_dict else default_config_dict
        self._algorithms = cfg["algorithms"]
        self._hyperparameters = cfg["hyperparameters"]

        self._population = [
            self._random_model(random_hyper=True)
            for _ in range(self.population_size)
        ]
        self._metrics = [type(self.metric)() for _ in self._population]
        self._best_idx = 0
        self._counter = 0
        self._max_params = max(
            (len(v) for v in self._hyperparameters.values()), default=1
        )
        self._meta = forest.ARFRegressor(seed=seed or 0)

    # ── public interface ──────────────────────────────────────────────────────

    def predict_one(self, x: dict) -> Any:
        try:
            return self._population[self._best_idx].predict_one(x)
        except Exception:
            return 0

    def learn_one(self, x: dict, y: Any) -> None:
        for idx, model in enumerate(self._population):
            try:
                y_pred = model.predict_one(x)
                self._metrics[idx].update(y, y_pred)
                # Poisson bagging: each model sees a random number of copies
                for _ in range(np.random.poisson(6)):
                    model.learn_one(x, y)
                if self._metrics[idx].is_better_than(self._metrics[self._best_idx]):
                    self._best_idx = idx
            except Exception:
                pass

        self._counter += 1
        if self._counter % self.exploration_window == 0:
            self._evolve()

    def reset(self) -> None:
        self.__init__(
            config_dict=self.config_dict,
            metric=type(self.metric)(),
            exploration_window=self.exploration_window,
            population_size=self.population_size,
            seed=self.seed,
        )

    # ── genetic operators ─────────────────────────────────────────────────────

    def _get_current_params(self, model) -> dict:
        name = type(model).__name__
        space = self._hyperparameters.get(name, {})
        cur = model._get_params()
        return {k: cur[k] for k in space if k in cur}

    def _random_model(self, random_hyper: bool = False):
        model = random.choice(self._algorithms)
        if not random_hyper:
            return model.clone()
        name = type(model).__name__
        space = self._hyperparameters.get(name, {})
        params = {k: random.choice(v) for k, v in space.items()}
        return model.clone(new_params=params) if params else model.clone()

    def _encode_model(self, model) -> dict:
        """Encode model + hyperparams as a fixed-length numeric vector for the meta-regressor."""
        name = type(model).__name__
        params = self._get_current_params(model)
        space = self._hyperparameters.get(name, {})
        vec = [0.0] * self._max_params
        for i, (k, v) in enumerate(params.items()):
            if i >= self._max_params:
                break
            s = space.get(k, [v])
            if isinstance(v, bool):
                vec[i] = 1.0 if v else 2.0
            elif isinstance(v, (int, float)):
                vec[i] = float(v)
            elif v in s:
                vec[i] = float(s.index(v) + 1)
            else:
                vec[i] = 1.0
        return {str(i): vec[i] for i in range(self._max_params)}

    def _mutate(self, model):
        """Return a clone of model with mutation-perturbed hyperparameters."""
        name = type(model).__name__
        space = self._hyperparameters.get(name, {})
        cur = self._get_current_params(model)
        mutated: dict = {}
        for k, v in cur.items():
            vals = space.get(k, [v])
            if isinstance(v, bool):
                mutated[k] = not v if np.random.rand() < 0.5 else v
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                lo, hi = min(vals), max(vals)
                sd = max((hi - lo) / 6, 1e-9)
                new_v = np.clip(np.random.normal(v, sd * 0.7), lo, hi)
                mutated[k] = int(round(new_v)) if isinstance(v, int) else float(new_v)
            elif isinstance(v, str) and v in vals:
                idx = vals.index(v)
                probs = np.ones(len(vals)) / len(vals)
                probs[idx] *= 1.5
                probs /= probs.sum()
                mutated[k] = np.random.choice(vals, p=probs)
            else:
                mutated[k] = random.choice(vals)
        return model.clone(new_params=mutated) if mutated else model.clone()

    def _select_parent(self):
        scores = [max(m.get(), 1e-9) for m in self._metrics]
        total = sum(scores)
        probs = [s / total for s in scores]
        return random.choices(self._population, probs)[0]

    def _evolve(self):
        """One genetic evolution step."""
        # Update meta-regressor with current population fitness
        for idx, model in enumerate(self._population):
            x_meta = self._encode_model(model)
            y_meta = round(self._metrics[idx].get() * 100, 2)
            self._meta.learn_one(x_meta, y_meta)

        parent = self._select_parent()
        mutant = self._mutate(parent)

        predicted_score = self._meta.predict_one(self._encode_model(mutant))
        worst_idx = int(np.argmin([m.get() for m in self._metrics]))

        if predicted_score >= self._metrics[worst_idx].get() * 100:
            self._population.pop(worst_idx)
            self._metrics.pop(worst_idx)
            self._population.append(mutant)
            self._metrics.append(type(self.metric)())
            # Re-calibrate best index
            self._best_idx = int(
                np.argmax([m.get() for m in self._metrics])
            )
