"""
samlb.framework.adapters._river
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run any `River <https://riverml.xyz>`_ estimator inside a SAMLB benchmark.

River is an optional dependency — SAMLB's own learners are pure C++ — so the
import is deferred until an adapter is actually constructed. Check for the
backend with :meth:`RiverClassifier.is_available` before adding one to a suite.

    from river import forest, preprocessing
    from samlb.framework.adapters import RiverClassifier

    model = RiverClassifier(
        preprocessing.StandardScaler() | forest.ARFClassifier(seed=42),
        name="river-ARF",
    )
"""
from __future__ import annotations

import importlib.util
from typing import Any, Callable, Dict, Optional, Union

from samlb.framework.base._framework import BaseStreamFramework


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_river() -> None:
    if not _has_module("river"):
        raise ImportError(
            "River adapters require the river package. "
            "Install with `pip install river`."
        )


#: Either a River estimator (it is cloned, never trained in place) or a
#: zero-argument callable returning a fresh one.
RiverSpec = Union[Any, Callable[[], Any]]


class _RiverAdapter(BaseStreamFramework):
    """Shared plumbing for the River classification / regression adapters."""

    _task = "estimator"

    def __init__(
        self,
        model: RiverSpec,
        name: Optional[str] = None,
        seed: int = 42,
    ):
        _require_river()

        self.model = model
        self.name = name
        self.seed = seed

        # A callable without the learner surface is a factory; anything else is
        # a prototype we clone. Either way the object handed in stays untrained.
        self._is_factory = callable(model) and not hasattr(model, "learn_one")
        self._model = self._new_model()

        for method in ("learn_one", "predict_one"):
            if not hasattr(self._model, method):
                raise TypeError(
                    f"{type(self).__name__} needs a River estimator exposing "
                    f"{method}(); got {type(self._model).__name__}."
                )
        self._check_task(self._model)

    # ── construction ─────────────────────────────────────────────────────────

    @staticmethod
    def is_available() -> bool:
        """True when River is importable."""
        return _has_module("river")

    def _new_model(self) -> Any:
        if self._is_factory:
            return self.model()
        clone = getattr(self.model, "clone", None)
        if clone is not None:
            return clone()
        import copy
        return copy.deepcopy(self.model)

    def _check_task(self, model: Any) -> None:
        """Warn-free task check: River's own inspector, when it has one.

        ``river.utils.inspect`` understands pipelines and wrappers, which a
        bare ``isinstance`` against ``river.base`` would not.
        """
        try:
            from river.utils import inspect as river_inspect
            check = getattr(river_inspect, self._inspect_fn)
        except (ImportError, AttributeError):
            return
        if not check(model):
            raise TypeError(
                f"{type(self).__name__} expects a River {self._task}; "
                f"{type(model).__name__} is not one."
            )

    # ── SAMLB framework interface ────────────────────────────────────────────

    def learn_one(self, x: Dict[str, float], y: Any) -> None:
        self._model.learn_one(x, y)

    def predict_one(self, x: Dict[str, float]) -> Any:
        return self._model.predict_one(x)

    def reset(self) -> None:
        self._model = self._new_model()

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def river_model(self) -> Any:
        """The live River estimator — handy for post-run inspection."""
        return self._model

    def __str__(self) -> str:
        return self.name or f"River({type(self._model).__name__})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._model!r})"


class RiverClassifier(_RiverAdapter):
    """A River classifier (or pipeline ending in one) as a SAMLB framework.

    Parameters
    ----------
    model : River estimator or callable
        The estimator to benchmark. An instance is cloned before every run, so
        the object you pass in is never trained; pass a callable instead when a
        model cannot be cloned.
    name : str, optional
        Display name, used by ``str()``. Defaults to ``River(<class name>)``.
    seed : int
        Kept for interface symmetry with the other frameworks. River seeds its
        estimators at construction, so set the seed on the estimator itself.

    Notes
    -----
    River classifiers return ``None`` until they have seen a label; the
    prequential evaluator counts those instances but skips the metric update,
    exactly as it does for OAML's warm-up.
    """

    _task = "classifier"
    _inspect_fn = "isclassifier"


class RiverRegressor(_RiverAdapter):
    """A River regressor (or pipeline ending in one) as a SAMLB framework.

    Parameters are identical to :class:`RiverClassifier`. Note the benchmark
    normalises regression targets online, so the estimator sees standardised
    ``y`` and its predictions are mapped back before scoring.
    """

    _task = "regressor"
    _inspect_fn = "isregressor"

    def predict_one(self, x: Dict[str, float]) -> float:
        pred = self._model.predict_one(x)
        return 0.0 if pred is None else float(pred)
