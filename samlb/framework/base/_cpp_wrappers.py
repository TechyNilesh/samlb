"""
samlb.framework.base._cpp_wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thin Python wrappers around the SAMLB C++ components.

Every wrapper holds a ``_cpp`` handle and stores its hyperparameters as
attributes named after its ``__init__`` arguments. That convention is all
:class:`samlb.base.Estimator` needs to provide ``_get_params`` / ``clone``,
and the ``_cpp`` handle is what lets :class:`samlb.base.Pipeline` fuse the
whole chain into a single C++ object.

There is deliberately no dict sanitising here: the C++ side takes the feature
mapping directly, and NaN handling lives in the dataset loader.
"""
from __future__ import annotations

import samlb._samlb_core as _core

from samlb.base import Estimator


# ── Classification ────────────────────────────────────────────────────────────

class _Classifier(Estimator):
    _kind = "classifier"

    def learn_one(self, x, y):
        self._cpp.learn_one(x, y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(x)

    def predict_proba_one(self, x):
        return self._cpp.predict_proba_one(x)

    def reset(self):
        self._cpp.reset()
        return self


class NaiveBayes(_Classifier):
    """Gaussian Naïve Bayes (C++ backend)."""

    def __init__(self):
        self._cpp = _core.NaiveBayes()


class Perceptron(_Classifier):
    """Multiclass Perceptron (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self._cpp = _core.Perceptron(learning_rate=learning_rate)


class LogisticRegression(_Classifier):
    """One-vs-Rest Logistic Regression with SGD (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.l2 = l2
        self._cpp = _core.LogisticRegressionClassifier(learning_rate=learning_rate, l2=l2)


class PassiveAggressiveClassifier(_Classifier):
    """Passive-Aggressive Classifier (C++ backend)."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self._cpp = _core.PassiveAggressiveClassifier(C=C)


class SoftmaxRegression(_Classifier):
    """Multinomial Softmax Regression (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.l2 = l2
        self._cpp = _core.SoftmaxRegression(learning_rate=learning_rate, l2=l2)


class KNNClassifier(_Classifier):
    """K-Nearest Neighbours Classifier with sliding window (C++ backend)."""

    def __init__(self, n_neighbors: int = 5, window_size: int = 1000, p: int = 2):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.p = p
        self._cpp = _core.KNNClassifier(n_neighbors=n_neighbors, window_size=window_size, p=p)


class HoeffdingTreeClassifier(_Classifier):
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


class EFDTClassifier(_Classifier):
    """Extremely Fast Decision Tree (C++ backend)."""

    def __init__(
        self,
        grace_period: int = 200,
        split_confidence: float = 1e-5,
        tie_threshold: float = 0.05,
        nb_threshold: int = 0,
        max_depth: int = 20,
    ):
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.nb_threshold = nb_threshold
        self.max_depth = max_depth
        self._cpp = _core.EFDTClassifier(
            grace_period=grace_period,
            split_confidence=split_confidence,
            tie_threshold=tie_threshold,
            nb_threshold=nb_threshold,
            max_depth=max_depth,
        )


class SGTClassifier(_Classifier):
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


class ARFClassifier(_Classifier):
    """Adaptive Random Forest (C++ backend).

    Gomes et al., Machine Learning 2017 — the standard streaming baseline.
    Online bagging over Hoeffding trees, each resampling a random feature
    subspace at every split, with a warning/drift detector pair per tree and
    accuracy-weighted voting.
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int = 0,
        lambda_value: float = 6.0,
        drift_delta: float = 0.001,
        warning_delta: float = 0.01,
        grace_period: int = 50,
        max_depth: int = 20,
        split_confidence: float = 0.01,
        subspace_size: int = -1,
    ):
        self.n_models = n_models
        self.seed = seed
        self.lambda_value = lambda_value
        self.drift_delta = drift_delta
        self.warning_delta = warning_delta
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.split_confidence = split_confidence
        self.subspace_size = subspace_size
        self._cpp = _core.ARFClassifier(
            n_models=n_models, seed=seed, lambda_value=lambda_value,
            drift_delta=drift_delta, warning_delta=warning_delta,
            grace_period=grace_period, max_depth=max_depth,
            split_confidence=split_confidence, subspace_size=subspace_size,
        )


class SRPClassifier(_Classifier):
    """Streaming Random Patches (C++ backend).

    Gomes et al., ECML PKDD 2019. The difference from :class:`ARFClassifier` is
    where the feature randomisation happens: ARF resamples a subspace at every
    split attempt inside the tree, while SRP draws one random feature subset per
    ensemble member and keeps it for that member's whole life, so the base
    learner is an unmodified tree fed a projected instance. Combined with
    Poisson resampling that gives a random *patch* — a subset of features by a
    subset of instances.

    Parameters
    ----------
    training_method : str
        ``"patches"`` (default) — subspace and resampling, the paper's RP.
        ``"subspaces"`` — subspace only; every member sees every instance once.
        ``"resampling"`` — resampling only, over all features.
    subspace_fraction : float
        Share of the features each member is given.
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int = 0,
        lambda_value: float = 6.0,
        drift_delta: float = 0.001,
        warning_delta: float = 0.01,
        grace_period: int = 50,
        max_depth: int = 20,
        split_confidence: float = 0.01,
        subspace_fraction: float = 0.6,
        training_method: str = "patches",
    ):
        if training_method not in ("patches", "subspaces", "resampling"):
            raise ValueError(
                "training_method must be 'patches', 'subspaces' or 'resampling', "
                f"got {training_method!r}."
            )
        self.n_models = n_models
        self.seed = seed
        self.lambda_value = lambda_value
        self.drift_delta = drift_delta
        self.warning_delta = warning_delta
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.split_confidence = split_confidence
        self.subspace_fraction = subspace_fraction
        self.training_method = training_method
        self._cpp = _core.SRPClassifier(
            n_models=n_models, seed=seed, lambda_value=lambda_value,
            drift_delta=drift_delta, warning_delta=warning_delta,
            grace_period=grace_period, max_depth=max_depth,
            split_confidence=split_confidence,
            subspace_fraction=subspace_fraction,
            training_method=training_method,
        )


# ── Regression ────────────────────────────────────────────────────────────────

class _Regressor(Estimator):
    _kind = "regressor"

    def learn_one(self, x, y):
        self._cpp.learn_one(x, y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(x)

    def reset(self):
        self._cpp.reset()
        return self


class LinearRegression(_Regressor):
    """Linear Regression with SGD (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.l2 = l2
        self._cpp = _core.LinearRegression(learning_rate=learning_rate, l2=l2)


class BayesianLinearRegression(_Regressor):
    """Bayesian Linear Regression (C++ backend)."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self._cpp = _core.BayesianLinearRegression(alpha=alpha, beta=beta)


class PassiveAggressiveRegressor(_Regressor):
    """Passive-Aggressive Regressor (C++ backend)."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.1):
        self.C = C
        self.epsilon = epsilon
        self._cpp = _core.PassiveAggressiveRegressor(C=C, epsilon=epsilon)


class KNNRegressor(_Regressor):
    """K-Nearest Neighbours Regressor with sliding window (C++ backend)."""

    def __init__(self, n_neighbors: int = 5, window_size: int = 1000, p: int = 2):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.p = p
        self._cpp = _core.KNNRegressor(n_neighbors=n_neighbors, window_size=window_size, p=p)


class HoeffdingTreeRegressor(_Regressor):
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


class ARFRegressor(_Regressor):
    """Adaptive Random Forest Regressor (C++ backend).

    Used by AutoClass as a surrogate fitness model.
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int = 0,
        lambda_value: float = 6.0,
        drift_delta: float = 0.001,
        warning_delta: float = 0.01,
        grace_period: int = 200,
        max_depth: int = 20,
        learning_rate: float = 0.01,
    ):
        self.n_models = n_models
        self.seed = seed
        self.lambda_value = lambda_value
        self.drift_delta = drift_delta
        self.warning_delta = warning_delta
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._cpp = _core.ARFRegressor(
            n_models=n_models,
            seed=seed,
            lambda_value=lambda_value,
            drift_delta=drift_delta,
            warning_delta=warning_delta,
            grace_period=grace_period,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )


# ── Preprocessing / feature selection ─────────────────────────────────────────

class _Transformer(Estimator):
    _kind = "transformer"
    _supervised = False

    def __deepcopy__(self, memo):
        # Unlike learners (which cannot be deep-copied at all and so clone
        # fresh), River's preprocessors carried their fitted statistics through
        # a deepcopy. EvoAutoML's mutation step relies on that, so the copy
        # keeps the running statistics.
        new = self.clone()
        new._cpp = self._cpp.clone_state()
        memo[id(self)] = new
        return new

    def learn_one(self, x):
        self._cpp.learn_one(x)
        return self

    def transform_one(self, x):
        return self._cpp.transform_one(x)

    def reset(self):
        self._cpp.reset()
        return self


class StandardScaler(_Transformer):
    """Zero-mean / unit-variance scaling (C++ backend)."""

    def __init__(self, with_std: bool = True):
        self.with_std = with_std
        self._cpp = _core.StandardScaler(with_std=with_std)


class MinMaxScaler(_Transformer):
    """Min-max scaling to [0, 1] (C++ backend)."""

    def __init__(self):
        self._cpp = _core.MinMaxScaler()


class MaxAbsScaler(_Transformer):
    """Absolute-max scaling to [-1, 1] (C++ backend)."""

    def __init__(self):
        self._cpp = _core.MaxAbsScaler()


class VarianceThreshold(_Transformer):
    """Drop features whose running variance stays below ``threshold``."""

    def __init__(self, threshold: float = 0.0, min_samples: int = 2):
        self.threshold = threshold
        self.min_samples = min_samples
        self._cpp = _core.VarianceThreshold(threshold=threshold, min_samples=min_samples)


class SelectKBest(_Transformer):
    """Keep the ``k`` features most correlated with the target (Pearson).

    Supervised: ``learn_one`` takes ``(x, y)``.
    """

    _supervised = True

    def __init__(self, k: int = 10, use_abs: bool = False):
        self.k = k
        self.use_abs = use_abs
        self._cpp = _core.SelectKBest(k=k, use_abs=use_abs)
        self._order_declared = False

    def __deepcopy__(self, memo):
        new = super().__deepcopy__(memo)
        new._order_declared = self._order_declared
        return new

    def learn_one(self, x, y):
        if not self._order_declared:
            # Tie-break order, see samlb.base.Pipeline._declare_order.
            self._cpp.set_feature_order(list(x.keys()))
            self._order_declared = True
        self._cpp.learn_one(x, float(y))
        return self
