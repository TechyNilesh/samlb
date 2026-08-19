"""
samlb.base
~~~~~~~~~~
Estimator and Pipeline primitives.

This replaces the parts of ``river.base`` / ``river.compose`` that SAMLB relied
on: hyperparameter introspection (``_get_params`` / ``clone``), the ``|``
pipeline operator, and ``pipeline.steps``.

Pipelines are *fused*: the components are stitched together on the C++ side, so
one instance crosses the Python/C++ boundary exactly once per ``learn_one`` /
``predict_one`` instead of once per stage.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import samlb._samlb_core as _core


class Estimator:
    """Base for every SAMLB component.

    Subclasses store their hyperparameters as attributes named exactly like
    their ``__init__`` arguments; that convention is what makes ``_get_params``
    and ``clone`` work without any per-class boilerplate.
    """

    # Set by subclasses; used to pick the right fused pipeline type.
    _kind: str = "estimator"       # "transformer" | "classifier" | "regressor"

    def _get_params(self) -> Dict[str, Any]:
        params = {}
        for name, param in inspect.signature(self.__init__).parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)
        return params

    def clone(self, new_params: Optional[Dict[str, Any]] = None) -> "Estimator":
        """A fresh, untrained copy — same hyperparameters, no learned state."""
        params = {**self._get_params(), **(new_params or {})}
        return type(self)(**params)

    def _set_params(self, new_params: Optional[Dict[str, Any]] = None) -> "Estimator":
        return self.clone(new_params)

    def __or__(self, other) -> "Pipeline":
        return Pipeline(self, other)

    def __deepcopy__(self, memo):
        # C++ handles cannot be pickled; a deepcopy of an estimator means a
        # fresh one with identical hyperparameters, matching river's clone().
        new = self.clone()
        memo[id(self)] = new
        return new

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._get_params().items())
        return f"{type(self).__name__}({args})"

    def __str__(self) -> str:
        return type(self).__name__


class Pipeline:
    """``transformer | ... | learner``, executed as one C++ object.

    Accepts the same shapes river's pipeline did: ``scaler | model``,
    ``scaler | selector | model``, and ``(scaler | selector) | model``.
    ``None`` steps are skipped so callers can pass an optional selector.
    """

    _kind = "pipeline"

    def __init__(self, *steps):
        flat: List[Estimator] = []
        for s in steps:
            if s is None:
                continue
            if isinstance(s, Pipeline):
                flat.extend(s._steps)
            else:
                flat.append(s)
        if not flat:
            raise ValueError("A pipeline needs at least one step.")

        self._steps: List[Estimator] = flat

        # `scaler | selector | model` binds left-to-right, so `scaler | selector`
        # is built first and is not yet runnable. Such a partial chain stays
        # composable; it only gains a C++ object once a learner is appended.
        self._partial = flat[-1]._kind == "transformer"
        if self._partial:
            self._transformers = flat
            self._learner = None
            self._cpp = None
        else:
            self._transformers = flat[:-1]
            self._learner = flat[-1]
            if any(t._kind != "transformer" for t in self._transformers):
                raise TypeError("Every step but the last must be a transformer.")
            cpp_steps = [t._cpp for t in self._transformers]
            if self._learner._kind == "classifier":
                self._cpp = _core.ClassificationPipeline(cpp_steps, self._learner._cpp)
            else:
                self._cpp = _core.RegressionPipeline(cpp_steps, self._learner._cpp)

        self._order_declared = False

    def _require_runnable(self) -> None:
        if self._partial:
            raise TypeError("This pipeline has no learner; append one with `| model`.")

    # ── river-compatible surface ─────────────────────────────────────────────

    @property
    def steps(self) -> Dict[str, Estimator]:
        """Ordered {name: step} mapping, as river's Pipeline exposed."""
        out: Dict[str, Estimator] = {}
        for s in self._steps:
            name = type(s).__name__
            if name in out:
                i = 2
                while f"{name}{i}" in out:
                    i += 1
                name = f"{name}{i}"
            out[name] = s
        return out

    def clone(self, new_params: Optional[Dict[str, Any]] = None) -> "Pipeline":
        return Pipeline(*[s.clone() for s in self._steps])

    def __or__(self, other) -> "Pipeline":
        return Pipeline(self, other)

    def __deepcopy__(self, memo):
        new = self.clone()
        memo[id(self)] = new
        return new

    # ── learning ─────────────────────────────────────────────────────────────

    def _declare_order(self, x: dict) -> None:
        # Feature order is lost inside C++ (unordered_map) but SelectKBest
        # breaks ties by it, so hand it over once, from the first instance.
        self._cpp.set_feature_order(list(x.keys()))
        self._order_declared = True

    def learn_one(self, x: dict, y) -> "Pipeline":
        self._require_runnable()
        if not self._order_declared:
            self._declare_order(x)
        self._cpp.learn_one(x, y)
        return self

    def predict_one(self, x: dict):
        self._require_runnable()
        if not self._order_declared:
            self._declare_order(x)
        return self._cpp.predict_one(x)

    def predict_proba_one(self, x: dict) -> dict:
        self._require_runnable()
        if not self._order_declared:
            self._declare_order(x)
        return self._cpp.predict_proba_one(x)

    def reset(self) -> None:
        if self._cpp is not None:
            self._cpp.reset()

    def __repr__(self) -> str:
        return " | ".join(repr(s) for s in self._steps)

    def __str__(self) -> str:
        return " | ".join(str(s) for s in self._steps)
