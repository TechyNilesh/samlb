"""
samlb.framework.random_search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Random-search baseline for streaming AutoML (classification and regression).

A deliberately simple reference point that performs *no* intelligent search.
It maintains the full shared learner pool as a set of (scaler | model)
pipelines, keeps every pipeline warm by updating all of them online, and at
each exploration window randomly selects one pipeline from the pool to serve
predictions for the next window.  It therefore isolates the value added by an
AutoML search strategy: any framework that does not beat random per-window
selection over the same warm pool is not exploiting its search.

For regression, predictions are clipped to the running [min, max] target range
seen so far (as in ASML's regression variant), bounding the occasional
diverging pipeline that random selection may deploy.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

from samlb.framework.base._framework import BaseStreamFramework


class _Pipe:
    """A lightweight (scaler | model) pipeline kept warm online."""

    def __init__(self, scaler, model):
        self.scaler = copy.deepcopy(scaler)
        self.model = copy.deepcopy(model)

    def predict_one(self, x: Dict[str, float]) -> Any:
        return self.model.predict_one(self.scaler.transform_one(x))

    def learn_one(self, x: Dict[str, float], y: Any) -> None:
        self.scaler.learn_one(x)
        self.model.learn_one(self.scaler.transform_one(x), y)


class RandomSearch(BaseStreamFramework):
    """Random per-window selection over a warm shared pool.

    Parameters
    ----------
    scalers : list
        Candidate preprocessor instances (shared pool).
    models : list
        Candidate model instances (shared pool).
    exploration_window : int
        Number of instances between random re-selections of the active model.
    clip : bool
        If True (regression), clip predictions to the running target range.
    seed : int or None
        Random seed; the suite sets this before each reset().
    """

    def __init__(self, scalers: List, models: List,
                 exploration_window: int = 1000, clip: bool = False,
                 seed: Optional[int] = 42, **kwargs: Any) -> None:
        self._scalers = scalers
        self._models = models
        self.exploration_window = exploration_window
        self.clip = clip
        self.seed = seed
        self.reset()

    # ── required interface ────────────────────────────────────────────────────

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._pool = [_Pipe(self._rng.choice(self._scalers), m)
                      for m in self._models]
        self._active = self._rng.randrange(len(self._pool))
        self._seen = 0
        self._ymin = None
        self._ymax = None

    def predict_one(self, x: Dict[str, float]) -> Any:
        pred = self._pool[self._active].predict_one(x)
        if self.clip and self._ymin is not None and isinstance(pred, (int, float)):
            if pred < self._ymin:
                pred = self._ymin
            elif pred > self._ymax:
                pred = self._ymax
        return pred

    def learn_one(self, x: Dict[str, float], y: Any) -> None:
        if self.clip and isinstance(y, (int, float)):
            self._ymin = y if self._ymin is None else min(self._ymin, y)
            self._ymax = y if self._ymax is None else max(self._ymax, y)
        for p in self._pool:
            p.learn_one(x, y)
        self._seen += 1
        if self._seen % self.exploration_window == 0:
            self._active = self._rng.randrange(len(self._pool))
