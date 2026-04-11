"""
ASML Classification — Pipeline search with ARDNS strategy.

ARDNS = Adaptive Random Drift Nearby Search.
Budget is split 50/50 between:
  - ARDNS: generate nearby pipelines by perturbing the best pipeline's hyperparams
  - Random: generate fresh random pipelines from the search space
"""
from __future__ import annotations

import math
import random

import numpy as np

from .config import default_config_dict


class PipelineSearch:
    """
    Creates and manages a pool of River pipelines (preprocessor | [feature_sel] | model).

    Parameters
    ----------
    config_dict : dict, optional
        Custom search space.  Falls back to :data:`default_config_dict`.
    no_fs_models : list
        Model class names that skip the feature-selection step.
    budget : int
        Total number of pipelines to maintain (excluding the best carried over).
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        config_dict=None,
        no_fs_models: list | None = None,
        budget: int = 10,
        seed: int | None = 42,
    ):
        self.config_dict = config_dict
        self.no_fs_models = no_fs_models or []
        self.budget = budget
        self.seed = seed

        self.pipeline_list: list = []

        self.random_budget = math.ceil(self.budget / 2)
        self.ardns_budget = self.budget - self.random_budget

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        cfg = config_dict if config_dict else default_config_dict
        self.hyperparameters = cfg.get("hyperparameters", {})
        self.algorithms = [
            self._init_params(m, "default") for m in cfg.get("models", [])
        ]
        self.preprocessing_steps = [
            self._init_params(p, "default") for p in cfg.get("preprocessors", [])
        ]
        self.feature_selection_methods = [
            self._init_params(f, "default") for f in cfg.get("features", [])
        ]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _init_params(self, model, mode: str = "random"):
        """Return model with hyperparams set to default or random values."""
        name = type(model).__name__
        space = self.hyperparameters.get(name, {})
        if not space:
            return model.clone()

        params = {}
        if mode == "random":
            for k, vals in space.items():
                params[k] = random.choice(vals)
        else:  # "default" — use model's current value if in space, else first
            cur = model._get_params()
            for k, vals in space.items():
                params[k] = cur.get(k, vals[0]) if cur.get(k) in vals else vals[0]

        return model.clone(new_params=params)

    def _get_current_params(self, model) -> dict:
        name = type(model).__name__
        space = self.hyperparameters.get(name, {})
        cur = model._get_params()
        return {k: cur[k] for k in space if k in cur}

    def _ardns(self, current_value, values_list):
        """ARDNS: pick same / upper-neighbour / lower-neighbour / random."""
        choice = np.random.choice(["same", "upper", "lower", "random"])
        if choice == "same":
            return current_value
        try:
            idx = values_list.index(current_value)
        except ValueError:
            return random.choice(values_list)
        if choice == "upper":
            return values_list[min(idx + 1, len(values_list) - 1)]
        if choice == "lower":
            return values_list[max(idx - 1, 0)]
        return random.choice(values_list)

    def _suggest_nearby(self, model):
        """Return model clone with ARDNS-perturbed hyperparameters."""
        name = type(model).__name__
        space = self.hyperparameters.get(name, {})
        cur = self._get_current_params(model)
        suggested = {}
        for param, val in cur.items():
            vals = space.get(param, [val])
            new_val = self._ardns(val, vals)
            if isinstance(val, int):
                suggested[param] = int(new_val)
            elif isinstance(val, float):
                suggested[param] = float(new_val)
            else:
                suggested[param] = new_val
        return model.clone(new_params=suggested) if suggested else model.clone()

    # ── pipeline creation ─────────────────────────────────────────────────────

    def _create_pipelines(self) -> list:
        """Enumerate all (preprocessor × [feature_sel] × model) combos."""
        for pre in self.preprocessing_steps:
            for model in self.algorithms:
                self.pipeline_list.append((pre | model).clone())
                if type(model).__name__ not in self.no_fs_models:
                    for fs in self.feature_selection_methods:
                        self.pipeline_list.append((pre | fs | model).clone())
        return self.pipeline_list

    def _random_pipeline(self, random_hyper: bool = False):
        """Build one random pipeline (random component choices)."""
        algo = random.choice(self.algorithms)
        pre = random.choice(self.preprocessing_steps)
        fs = random.choice(self.feature_selection_methods) if self.feature_selection_methods else None

        if random_hyper:
            algo = self._init_params(algo, "random")
            pre = self._init_params(pre, "random")
            if fs:
                fs = self._init_params(fs, "random")

        use_fs = fs is not None and type(algo).__name__ not in self.no_fs_models and random.random() > 0.5
        return (pre | fs | algo).clone() if use_fs else (pre | algo).clone()

    # ── ARDNS update ──────────────────────────────────────────────────────────

    def next_nearby(self, pipeline):
        """Return a pipeline with ARDNS-perturbed components."""
        steps = list(pipeline.steps.values())
        new_pre = self._suggest_nearby(steps[0])
        new_algo = self._suggest_nearby(steps[-1])
        if len(steps) > 2:
            new_fs = self._suggest_nearby(steps[1])
            return new_pre | new_fs | new_algo
        return new_pre | new_algo

    def select_and_update_pipelines(self, best_pipeline) -> list:
        """
        Rebuild the pipeline pool after each exploration window:
          - 1 copy of the current best
          - ``ardns_budget`` ARDNS neighbours of the best
          - ``random_budget`` fresh random pipelines
        """
        base = best_pipeline.clone()
        ardns = []
        cursor = base
        for _ in range(self.ardns_budget):
            nb = self.next_nearby(cursor)
            ardns.append(nb)
            cursor = nb.clone()

        randoms = [self._random_pipeline(random_hyper=False).clone()
                   for _ in range(self.random_budget)]

        self.pipeline_list = [base] + ardns + randoms
        return self.pipeline_list
