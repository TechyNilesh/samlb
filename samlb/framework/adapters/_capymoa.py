"""
samlb.framework.adapters._capymoa
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run any `CapyMOA <https://capymoa.org>`_ (MOA) learner inside a SAMLB benchmark.

CapyMOA learners are built against a fixed MOA ``Schema``, and they consume
``Instance`` objects rather than the ``{feature: value}`` dicts SAMLB streams.
The adapter bridges both gaps: it derives the schema from the first instance it
sees, builds the learner lazily at that point, and converts every instance to
the dense array MOA expects.

CapyMOA is an optional dependency (it needs a JVM), so the import is deferred
until an adapter is constructed. Check :meth:`CapyMOAClassifier.is_available`
before adding one to a suite.

    from samlb.framework.adapters import CapyMOAClassifier

    model = CapyMOAClassifier("AdaptiveRandomForestClassifier", ensemble_size=10)
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
from typing import Any, Dict, Iterable, List, Optional, Union

from samlb.framework.base._framework import BaseStreamFramework

#: Nominal class slots reserved when the label set is not known up front.
DEFAULT_MAX_CLASSES = 100


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_capymoa() -> None:
    if not _has_module("capymoa"):
        raise ImportError(
            "CapyMOA adapters require the capymoa package (and a JVM). "
            "Install with `pip install capymoa`."
        )


class _CapyMOAAdapter(BaseStreamFramework):
    """Shared plumbing for the CapyMOA classification / regression adapters."""

    _task = "estimator"
    _module = ""          # capymoa submodule a string learner is resolved against

    def __init__(
        self,
        learner: Union[str, type],
        name: Optional[str] = None,
        seed: int = 42,
        **learner_kwargs: Any,
    ):
        _require_capymoa()

        self.learner = learner
        self.name = name
        self.seed = seed
        self.learner_kwargs = learner_kwargs

        self._learner_cls = self._resolve(learner)
        self._reset_state()

    # ── construction ─────────────────────────────────────────────────────────

    @staticmethod
    def is_available() -> bool:
        """True when CapyMOA is importable (which also starts its JVM)."""
        if not _has_module("capymoa"):
            return False
        try:
            importlib.import_module("capymoa")
        except Exception:
            # A missing or unusable JVM surfaces here, not at find_spec time.
            return False
        return True

    def _resolve(self, learner: Union[str, type]) -> type:
        """Accept a class, or a name looked up in ``capymoa.<module>``."""
        if isinstance(learner, type):
            return learner
        if not isinstance(learner, str):
            raise TypeError(
                f"{type(self).__name__} takes a CapyMOA class or its name; "
                f"an already-built {type(learner).__name__} cannot be used "
                "because CapyMOA learners need their schema at construction."
            )
        module = importlib.import_module(f"capymoa.{self._module}")
        path = learner.rsplit(".", 1)[-1]
        if not hasattr(module, path):
            available = ", ".join(sorted(n for n in dir(module) if n[0].isupper()))
            raise ValueError(
                f"capymoa.{self._module} has no learner {path!r}. "
                f"Available: {available}"
            )
        return getattr(module, path)

    def _reset_state(self) -> None:
        self._model = None
        self._schema = None
        self._features: List[str] = []

    def _build_learner(self, schema: Any) -> Any:
        kwargs = dict(self.learner_kwargs)
        params = inspect.signature(self._learner_cls.__init__).parameters
        if "random_seed" in params and "random_seed" not in kwargs:
            kwargs["random_seed"] = self.seed
        return self._learner_cls(schema=schema, **kwargs)

    def _ensure_built(self, x: Dict[str, float]) -> None:
        if self._model is not None:
            return
        # Feature order is fixed by the first instance and every later instance
        # is projected onto it, so the dense array always matches the schema.
        self._features = list(x)
        self._schema = self._make_schema(self._features)
        self._model = self._build_learner(self._schema)

    def _to_array(self, x: Dict[str, float]):
        import numpy as np
        return np.array([float(x.get(f, 0.0)) for f in self._features], dtype=np.float64)

    def _make_schema(self, features: List[str]) -> Any:
        raise NotImplementedError

    # ── SAMLB framework interface ────────────────────────────────────────────

    def reset(self) -> None:
        self._reset_state()

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def capymoa_model(self) -> Any:
        """The live CapyMOA learner, or None before the first instance."""
        return self._model

    def __str__(self) -> str:
        return self.name or f"CapyMOA({self._learner_cls.__name__})"

    def __repr__(self) -> str:
        args = "".join(f", {k}={v!r}" for k, v in self.learner_kwargs.items())
        return f"{type(self).__name__}({self._learner_cls.__name__}{args})"


class CapyMOAClassifier(_CapyMOAAdapter):
    """A CapyMOA classifier as a SAMLB framework.

    Parameters
    ----------
    learner : type or str
        A CapyMOA classifier class, or its name in ``capymoa.classifier``
        (e.g. ``"HoeffdingTree"``). An already-constructed learner cannot be
        used, because CapyMOA binds a learner to its schema at construction.
    classes : iterable, optional
        The label set, when it is known. Leave it unset to discover labels as
        they arrive; the schema then reserves ``max_classes`` nominal slots.
    max_classes : int
        Slots reserved for discovered labels. Only the memory MOA allocates for
        per-class statistics scales with it — unseen labels never get predicted
        because they carry no counts.
    name : str, optional
        Display name, used by ``str()``.
    seed : int
        Passed as ``random_seed`` to learners that accept one.
    **learner_kwargs
        Forwarded verbatim to the CapyMOA learner.

    Notes
    -----
    MOA works in label *indices*; the adapter keeps the mapping and returns the
    original SAMLB labels. Before the first label is seen ``predict_one``
    returns ``None``, which the prequential evaluator counts but does not score.
    """

    _task = "classifier"
    _module = "classifier"

    def __init__(
        self,
        learner: Union[str, type],
        classes: Optional[Iterable[Any]] = None,
        max_classes: int = DEFAULT_MAX_CLASSES,
        name: Optional[str] = None,
        seed: int = 42,
        **learner_kwargs: Any,
    ):
        if max_classes < 2:
            raise ValueError(f"max_classes must be >= 2, got {max_classes}.")
        self.classes = list(classes) if classes is not None else None
        self.max_classes = len(self.classes) if self.classes else max_classes
        if self.classes and len(self.classes) < 2:
            raise ValueError("classes must name at least two labels.")
        super().__init__(learner, name=name, seed=seed, **learner_kwargs)

    def _reset_state(self) -> None:
        super()._reset_state()
        # Position in this list is the MOA class index.
        self._labels: List[Any] = list(self.classes) if self.classes else []

    def _make_schema(self, features: List[str]) -> Any:
        from capymoa.stream import Schema
        values = (
            [str(c) for c in self.classes]
            if self.classes
            else [str(i) for i in range(self.max_classes)]
        )
        target = "_samlb_target"
        return Schema.from_custom(
            features=[*features, target],
            target=target,
            categories={target: values},
            name="samlb",
        )

    def _label_index(self, y: Any) -> int:
        try:
            return self._labels.index(y)
        except ValueError:
            pass
        if self.classes is not None:
            raise ValueError(
                f"Label {y!r} is not in the declared classes {self.classes!r}."
            )
        if len(self._labels) >= self.max_classes:
            raise ValueError(
                f"Stream has more than max_classes={self.max_classes} labels. "
                "Raise max_classes, or pass classes= explicitly."
            )
        self._labels.append(y)
        return len(self._labels) - 1

    def learn_one(self, x: Dict[str, float], y: Any) -> None:
        self._ensure_built(x)
        from capymoa.instance import LabeledInstance
        index = self._label_index(y)
        self._model.train(LabeledInstance.from_array(self._schema, self._to_array(x), index))

    def predict_one(self, x: Dict[str, float]) -> Any:
        self._ensure_built(x)
        from capymoa.instance import Instance
        index = self._model.predict(Instance.from_array(self._schema, self._to_array(x)))
        if index is None or not 0 <= int(index) < len(self._labels):
            # No label seen yet, or MOA picked a reserved-but-unused slot.
            return None
        return self._labels[int(index)]


class CapyMOARegressor(_CapyMOAAdapter):
    """A CapyMOA regressor as a SAMLB framework.

    Parameters
    ----------
    learner : type or str
        A CapyMOA regressor class, or its name in ``capymoa.regressor``
        (e.g. ``"AdaptiveRandomForestRegressor"``).
    name : str, optional
        Display name, used by ``str()``.
    seed : int
        Passed as ``random_seed`` to learners that accept one.
    **learner_kwargs
        Forwarded verbatim to the CapyMOA learner.
    """

    _task = "regressor"
    _module = "regressor"

    def _make_schema(self, features: List[str]) -> Any:
        from capymoa.stream import Schema
        target = "_samlb_target"
        # No categories for the target ⇒ numeric ⇒ MOA treats it as regression.
        return Schema.from_custom(
            features=[*features, target],
            target=target,
            name="samlb",
        )

    def learn_one(self, x: Dict[str, float], y: Any) -> None:
        self._ensure_built(x)
        from capymoa.instance import RegressionInstance
        self._model.train(
            RegressionInstance.from_array(self._schema, self._to_array(x), float(y))
        )

    def predict_one(self, x: Dict[str, float]) -> float:
        self._ensure_built(x)
        from capymoa.instance import Instance
        pred = self._model.predict(Instance.from_array(self._schema, self._to_array(x)))
        return 0.0 if pred is None else float(pred)
