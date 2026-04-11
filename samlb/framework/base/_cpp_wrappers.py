"""
samlb.framework.base._cpp_wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
River-compatible wrappers around the SAMLB C++ algorithms.

Inheriting from river.base.Classifier / river.base.Regressor gives us:
  - _get_params() / _set_params() / clone() automatically (River introspects __init__)
  - Compatibility with River's Pipeline ( | operator )
  - Compatibility with River's preprocessing and feature selection steps
"""
from __future__ import annotations

import math

from river.base import Classifier as _Clf, Regressor as _Reg
import samlb._samlb_core as _core


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_cpp_cls(cpp_cls, init_kwargs):
    """Create a fresh C++ classifier instance with keyword arguments."""
    return cpp_cls(**{k: v for k, v in init_kwargs.items()})


def _sanitize(x: dict) -> dict:
    """Normalise a feature dict for C++ wrappers (legacy, slow path).

    - Converts all keys to ``str`` (C++ expects ``Mapping[str, float]``).
    - Drops NaN values (C++ cannot handle them).
    """
    return {str(k): v for k, v in x.items() if v == v}  # v==v is False for NaN


# Fast-path: skip str() conversion (keys are already str from dataset loader)
# and use math.isnan for NaN check only when needed.
_isnan = math.isnan


def _fast_sanitize(x: dict) -> dict:
    """Fast sanitize: only drop NaN values, assume keys are already str."""
    # In the common case (no NaNs), just return x directly.
    # Only build a new dict if we find a NaN.
    for v in x.values():
        if v != v:  # NaN detected — fall back to filtered dict
            return {k: v for k, v in x.items() if v == v}
    return x


class _CppDeepCopyMixin:
    """Mixin that makes C++ wrapper objects deepcopy-safe.

    C++ extension objects cannot be pickled, so Python's ``copy.deepcopy``
    fails on them.  This mixin intercepts deepcopy and recreates the wrapper
    from its River-introspected ``__init__`` parameters, producing a
    *fresh* (untrained) copy with the same hyperparameters.
    """

    def __deepcopy__(self, memo: dict):
        new = type(self)(**self._get_params())
        memo[id(self)] = new
        return new


# ── Classification wrappers ───────────────────────────────────────────────────

class NaiveBayes(_CppDeepCopyMixin, _Clf):
    """Gaussian Naïve Bayes (C++ backend)."""

    def __init__(self):
        self._cpp = _core.NaiveBayes()

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))

    def predict_proba_one(self, x):
        return self._cpp.predict_proba_one(_fast_sanitize(x))


class Perceptron(_CppDeepCopyMixin, _Clf):
    """Multiclass Perceptron (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self._cpp = _core.Perceptron(learning_rate=learning_rate)

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))

    def predict_proba_one(self, x):
        return self._cpp.predict_proba_one(_fast_sanitize(x))


class LogisticRegression(_CppDeepCopyMixin, _Clf):
    """One-vs-Rest Logistic Regression with SGD (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.l2 = l2
        self._cpp = _core.LogisticRegressionClassifier(
            learning_rate=learning_rate, l2=l2
        )

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))

    def predict_proba_one(self, x):
        return self._cpp.predict_proba_one(_fast_sanitize(x))


class PassiveAggressiveClassifier(_CppDeepCopyMixin, _Clf):
    """PA-I Classifier (C++ backend)."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self._cpp = _core.PassiveAggressiveClassifier(C=C)

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))

    def predict_proba_one(self, x):
        return self._cpp.predict_proba_one(_fast_sanitize(x))


class SoftmaxRegression(_CppDeepCopyMixin, _Clf):
    """Multiclass Softmax Regression (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.l2 = l2
        self._cpp = _core.SoftmaxRegression(learning_rate=learning_rate, l2=l2)

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))

    def predict_proba_one(self, x):
        return self._cpp.predict_proba_one(_fast_sanitize(x))


class KNNClassifier(_CppDeepCopyMixin, _Clf):
    """K-Nearest Neighbours Classifier with sliding window (C++ backend)."""

    def __init__(self, n_neighbors: int = 5, window_size: int = 1000, p: int = 2):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.p = p
        self._cpp = _core.KNNClassifier(
            n_neighbors=n_neighbors, window_size=window_size, p=p
        )

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))

    def predict_proba_one(self, x):
        return self._cpp.predict_proba_one(_fast_sanitize(x))


class HoeffdingTreeClassifier(_CppDeepCopyMixin, _Clf):
    """Very Fast Decision Tree — VFDT (C++ backend)."""

    def __init__(
        self,
        grace_period: int = 200,
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        nb_threshold: int = 0,
        max_depth: int = 20,
        split_criterion: str = "info_gain",
    ):
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.nb_threshold = nb_threshold
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        self._cpp = _core.HoeffdingTreeClassifier(
            grace_period=grace_period,
            split_confidence=split_confidence,
            tie_threshold=tie_threshold,
            nb_threshold=nb_threshold,
            max_depth=max_depth,
            split_criterion=split_criterion,
        )

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))

    def predict_proba_one(self, x):
        return self._cpp.predict_proba_one(_fast_sanitize(x))


class SGTClassifier(_CppDeepCopyMixin, _Clf):
    """Stochastic Gradient Tree Classifier (C++ backend)."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        lambda_: float = 0.1,
        grace_period: int = 200,
        max_depth: int = 6,
    ):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.grace_period = grace_period
        self.max_depth = max_depth
        self._cpp = _core.SGTClassifier(
            learning_rate=learning_rate,
            lambda_=lambda_,
            grace_period=grace_period,
            max_depth=max_depth,
        )

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))

    def predict_proba_one(self, x):
        return self._cpp.predict_proba_one(_fast_sanitize(x))


# ── Regression wrappers ───────────────────────────────────────────────────────

class LinearRegression(_CppDeepCopyMixin, _Reg):
    """Online SGD Linear Regression (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.l2 = l2
        self._cpp = _core.LinearRegression(learning_rate=learning_rate, l2=l2)

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))


class BayesianLinearRegression(_CppDeepCopyMixin, _Reg):
    """Incremental Bayesian Linear Regression (C++ backend)."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self._cpp = _core.BayesianLinearRegression(alpha=alpha, beta=beta)

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))


class PassiveAggressiveRegressor(_CppDeepCopyMixin, _Reg):
    """PA-I Regressor with epsilon-insensitive loss (C++ backend)."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.1):
        self.C = C
        self.epsilon = epsilon
        self._cpp = _core.PassiveAggressiveRegressor(C=C, epsilon=epsilon)

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))


class KNNRegressor(_CppDeepCopyMixin, _Reg):
    """K-Nearest Neighbours Regressor with sliding window (C++ backend)."""

    def __init__(self, n_neighbors: int = 5, window_size: int = 1000, p: int = 2):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.p = p
        self._cpp = _core.KNNRegressor(
            n_neighbors=n_neighbors, window_size=window_size, p=p
        )

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))


class HoeffdingTreeRegressor(_CppDeepCopyMixin, _Reg):
    """Hoeffding Tree Regressor with SDR splitting (C++ backend)."""

    def __init__(
        self,
        grace_period: int = 200,
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        max_depth: int = 20,
        learning_rate: float = 0.01,
    ):
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._cpp = _core.HoeffdingTreeRegressor(
            grace_period=grace_period,
            split_confidence=split_confidence,
            tie_threshold=tie_threshold,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )

    def learn_one(self, x, y):
        self._cpp.learn_one(_fast_sanitize(x), y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(_fast_sanitize(x))
