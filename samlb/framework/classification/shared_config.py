"""
samlb.framework.classification.shared_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Single source of truth for the C++ algorithm pool used by ALL classification
frameworks (ASML, AutoClass, EvoAutoML, OAML).

Keeping this in one place guarantees a fair comparison:  the only difference
between benchmark results is the AutoML *strategy*, not the algorithm set.

Only SAMLB C++ wrappers are used — no River Python classifiers.

The ``ClassificationConfig`` dataclass is the single object you pass to
``run_classification_benchmark(config=...)`` to change the algorithm pool
for all four frameworks at once.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List

from samlb.framework.base import MaxAbsScaler, MinMaxScaler, StandardScaler

from samlb.framework.base._cpp_wrappers import (
    ARFClassifier,
    HoeffdingAdaptiveTreeClassifier,
    HoeffdingTreeClassifier,
    KNNClassifier,
    LeveragingBaggingClassifier,
    LogisticRegression,
    SGBTClassifier,
    NaiveBayes,
    PassiveAggressiveClassifier,
    Perceptron,
    SGTClassifier,
    SoftmaxRegression,
    SRPClassifier,
)
from samlb.framework.classification.asml.helper import range_gen

# ── Preprocessors ─────────────────────────────────────────────────────────────
# Same for all frameworks.

SHARED_PREPROCESSORS = [
    MinMaxScaler(),
    StandardScaler(),
    MaxAbsScaler(),
]

# ── Base classifier pool (one default instance per type) ──────────────────────
# Used by ASML and AutoClass (which pair this with SHARED_HYPERPARAMETERS).

SHARED_MODEL_POOL = [
    NaiveBayes(),
    Perceptron(),
    LogisticRegression(),
    PassiveAggressiveClassifier(),
    SoftmaxRegression(),
    HoeffdingTreeClassifier(),
    SGTClassifier(),
    KNNClassifier(),
]

# ── Ensemble baseline pool ──────────────────────────────────────────────────
# Drift-adaptive ensembles (ARF, SRP, Leveraging Bagging, Hoeffding Adaptive
# Tree) instead of the plain single-model pool above. Pass this to
# ``get_classification_config(pool="ensemble")`` when a search framework
# should pick among strong streaming baselines rather than tune simple models
# from scratch — e.g. to see whether search still adds value once every
# candidate is already drift-adaptive on its own.

ENSEMBLE_MODEL_POOL = [
    ARFClassifier(),
    SRPClassifier(),
    LeveragingBaggingClassifier(),
    HoeffdingAdaptiveTreeClassifier(),
    # The only boosting method in the pool; everything above is bagging-family.
    # At the paper's 100 boosting iterations, cost per instance is 100 x the
    # class count in trees, each touched twice — by a wide margin the most
    # expensive candidate here. Drop n_models if a search framework has to fit
    # it many times.
    SGBTClassifier(),
]

# ── Hyperparameter search spaces ──────────────────────────────────────────────
# Keyed by class name — used by ASML (ARDNS) and AutoClass (genetic).

SHARED_HYPERPARAMETERS = {
    "NaiveBayes": {},
    "Perceptron": {
        "learning_rate": range_gen(0.001, 0.1, step=0.005, float_n=True),
    },
    "LogisticRegression": {
        "learning_rate": range_gen(0.001, 0.1, step=0.005, float_n=True),
        "l2":            range_gen(0.0, 0.01, step=0.001, float_n=True),
    },
    "PassiveAggressiveClassifier": {
        "C": range_gen(0.1, 10.0, step=0.5, float_n=True),
    },
    "SoftmaxRegression": {
        "learning_rate": range_gen(0.001, 0.1, step=0.005, float_n=True),
        "l2":            range_gen(0.0, 0.01, step=0.001, float_n=True),
    },
    "HoeffdingTreeClassifier": {
        "grace_period":     range_gen(50, 500, step=50),
        "split_confidence": [1e-9, 1e-7, 1e-4, 1e-2],
        "tie_threshold":    range_gen(0.02, 0.08, step=0.01, float_n=True),
        "nb_threshold":     range_gen(0, 50, step=10),
        "max_depth":        range_gen(10, 100, step=10),
        "split_criterion":  ["info_gain", "gini"],
    },
    "SGTClassifier": {
        "learning_rate": range_gen(0.01, 0.5, step=0.05, float_n=True),
        "grace_period":  range_gen(50, 500, step=50),
        "max_depth":     range_gen(3, 12, step=1),
    },
    "KNNClassifier": {
        "n_neighbors": range_gen(3, 15, step=2),
        "window_size": range_gen(200, 2000, step=200),
        "p":           [1, 2],
    },
    "ARFClassifier": {
        "n_models":         range_gen(5, 30, step=5),
        "lambda_value":     range_gen(1.0, 10.0, step=1.0, float_n=True),
        "grace_period":     range_gen(50, 500, step=50),
        "split_confidence": [1e-9, 1e-7, 1e-4, 1e-2],
        "max_depth":        range_gen(10, 100, step=10),
    },
    "SRPClassifier": {
        "n_models":          range_gen(5, 30, step=5),
        "lambda_value":      range_gen(1.0, 10.0, step=1.0, float_n=True),
        "subspace_fraction": range_gen(0.2, 0.9, step=0.1, float_n=True),
        "training_method":   ["patches", "subspaces", "resampling"],
    },
    "SGBTClassifier": {
        "n_models":               range_gen(10, 100, step=10),
        "learning_rate":          [0.0125, 0.05, 0.1, 0.25],
        "percentage_of_features": range_gen(50, 100, step=10),
        "bag_size":               [1, 5, 10],
        "skip_training":          [1, 2, 4],
    },
    "LeveragingBaggingClassifier": {
        "n_models":     range_gen(5, 30, step=5),
        "lambda_value": range_gen(1.0, 10.0, step=1.0, float_n=True),
        "drift_delta":  [1e-4, 1e-3, 1e-2, 5e-2],
    },
    "HoeffdingAdaptiveTreeClassifier": {
        "grace_period":  range_gen(50, 500, step=50),
        "max_depth":     range_gen(10, 100, step=10),
        "drift_delta":   [1e-4, 1e-3, 1e-2, 5e-2],
        "warning_delta": [1e-3, 1e-2, 5e-2, 1e-1],
    },
}

# ── Pre-configured instances (all hyper combos) ───────────────────────────────
# Used by EvoAutoML and OAML (which need a flat list of ready-to-use instances).

SHARED_CLASSIFIER_INSTANCES = [
    NaiveBayes(),
    # Perceptron
    Perceptron(learning_rate=0.001),
    Perceptron(learning_rate=0.01),
    Perceptron(learning_rate=0.05),
    Perceptron(learning_rate=0.1),
    # Logistic Regression
    LogisticRegression(learning_rate=0.001),
    LogisticRegression(learning_rate=0.01),
    LogisticRegression(learning_rate=0.01, l2=0.001),
    LogisticRegression(learning_rate=0.05),
    # Passive Aggressive
    PassiveAggressiveClassifier(C=0.5),
    PassiveAggressiveClassifier(C=1.0),
    PassiveAggressiveClassifier(C=5.0),
    # Softmax Regression
    SoftmaxRegression(learning_rate=0.001),
    SoftmaxRegression(learning_rate=0.01),
    SoftmaxRegression(learning_rate=0.05),
    # Hoeffding Tree
    HoeffdingTreeClassifier(grace_period=50,  max_depth=10),
    HoeffdingTreeClassifier(grace_period=100, max_depth=20),
    HoeffdingTreeClassifier(grace_period=200, max_depth=30),
    HoeffdingTreeClassifier(grace_period=500, max_depth=50),
    HoeffdingTreeClassifier(grace_period=200, max_depth=20, split_criterion="gini"),
    # SGT
    SGTClassifier(learning_rate=0.1,  max_depth=6),
    SGTClassifier(learning_rate=0.01, max_depth=10),
    SGTClassifier(learning_rate=0.05, max_depth=8),
    # KNN
    KNNClassifier(n_neighbors=3,  window_size=500),
    KNNClassifier(n_neighbors=5,  window_size=1000),
    KNNClassifier(n_neighbors=10, window_size=2000),
    KNNClassifier(n_neighbors=5,  window_size=500,  p=1),
]

# Ensemble-baseline counterpart of SHARED_CLASSIFIER_INSTANCES, for EvoAutoML
# (param_grid) / OAML (classifiers=) when the pool should be drift-adaptive
# ensembles rather than plain single models.

ENSEMBLE_CLASSIFIER_INSTANCES = [
    # ARF
    ARFClassifier(n_models=10),
    ARFClassifier(n_models=20, lambda_value=8.0),
    ARFClassifier(n_models=10, grace_period=200),
    # SRP
    SRPClassifier(n_models=10),
    SRPClassifier(n_models=10, training_method="subspaces"),
    SRPClassifier(n_models=10, subspace_fraction=0.4),
    # Leveraging Bagging
    LeveragingBaggingClassifier(n_models=10),
    LeveragingBaggingClassifier(n_models=20, lambda_value=8.0),
    # Hoeffding Adaptive Tree
    HoeffdingAdaptiveTreeClassifier(),
    HoeffdingAdaptiveTreeClassifier(grace_period=100, drift_delta=0.01),
    # SGBT — cheaper than the paper's 100 iterations, so a search framework can
    # afford to fit it alongside the rest.
    SGBTClassifier(n_models=25),
    SGBTClassifier(n_models=25, learning_rate=0.05),
    SGBTClassifier(n_models=50, bag_size=5),
]


# ── ClassificationConfig — unified config object ──────────────────────────────

@dataclasses.dataclass
class ClassificationConfig:
    """Single config object passed to all four classification frameworks.

    Pass one instance to ``run_classification_benchmark(config=...)`` to
    change the algorithm pool for *every* framework at once — the only
    difference between results is then the AutoML strategy, not the pool.

    Attributes
    ----------
    scalers : list
        Preprocessor instances (MinMaxScaler, StandardScaler, …).
        Used by all four frameworks.
    model_pool : list
        One default instance per algorithm type.
        Used by ASML (ARDNS search) and AutoClass (genetic mutation).
    hyperparameters : dict
        Hyperparameter search spaces keyed by class name.
        Used by ASML and AutoClass for mutation/ARDNS neighbourhood search.
    classifier_instances : list
        Pre-configured instances at various hyperparameter settings.
        Used by EvoAutoML (param_grid) and OAML (random search pool).

    Example — drop KNN, add only HoeffdingTree + Perceptron
    --------------------------------------------------------
        from samlb.framework.classification.shared_config import ClassificationConfig
        from samlb.framework.base._cpp_wrappers import HoeffdingTreeClassifier, Perceptron
        from samlb.framework.base import MaxAbsScaler, MinMaxScaler, StandardScaler

        cfg = ClassificationConfig(
            scalers=[MinMaxScaler(), StandardScaler()],
            model_pool=[HoeffdingTreeClassifier(), Perceptron()],
            hyperparameters={
                "HoeffdingTreeClassifier": {"grace_period": [100, 200, 500]},
                "Perceptron":              {"learning_rate": [0.01, 0.1]},
            },
            classifier_instances=[
                HoeffdingTreeClassifier(grace_period=100),
                HoeffdingTreeClassifier(grace_period=500),
                Perceptron(learning_rate=0.01),
                Perceptron(learning_rate=0.1),
            ],
        )

        from samlb.benchmark import BenchmarkSuite
        from samlb.framework.classification.asml      import AutoStreamClassifier
        from samlb.framework.classification.autoclass import AutoClass
        from samlb.framework.classification.eaml      import EvolutionaryBaggingClassifier
        from samlb.framework.classification.oaml      import OAMLClassifier

        suite = BenchmarkSuite(
            models={
                "ASML":      AutoStreamClassifier(config_dict=cfg.asml_config_dict(), seed=42),
                "AutoClass": AutoClass(config_dict=cfg.autoclass_config_dict(), seed=42),
                "EvoAutoML": EvolutionaryBaggingClassifier(param_grid=cfg.eaml_param_grid(), seed=42),
                "OAML":      OAMLClassifier(scalers=cfg.scalers, classifiers=cfg.classifier_instances, seed=42),
            },
            datasets=["electricity"],
            task="classification",
        )
        suite.run()
        suite.print_table()
    """

    scalers:                List
    model_pool:             List
    hyperparameters:        Dict
    classifier_instances:   List

    def asml_config_dict(self) -> dict:
        """Config dict in the format AutoStreamClassifier expects."""
        from samlb.framework.base import SelectKBest, VarianceThreshold
        from samlb.framework.classification.asml.helper import range_gen
        return {
            "models":          self.model_pool,
            "preprocessors":   self.scalers,
            "features": [
                VarianceThreshold(threshold=0),
                SelectKBest(),
            ],
            "hyperparameters": {
                **self.hyperparameters,
                "MinMaxScaler":   {},
                "StandardScaler": {"with_std": [True, False]},
                "VarianceThreshold": {
                    "threshold":   range_gen(0.0, 1.0, step=0.1, float_n=True),
                    "min_samples": range_gen(1, 10, step=1),
                },
                "SelectKBest": {"k": range_gen(1, 25, step=1)},
            },
        }

    def autoclass_config_dict(self) -> dict:
        """Config dict in the format AutoClass expects."""
        return {
            "algorithms":      self.model_pool,
            "hyperparameters": self.hyperparameters,
        }

    def eaml_param_grid(self) -> dict:
        """param_grid in the format EvolutionaryBaggingClassifier expects."""
        return {
            "Scaler":     self.scalers,
            "Classifier": self.classifier_instances,
        }

# ── Default config (the shared C++ pool) ──────────────────────────────────────

DEFAULT_CLASSIFICATION_CONFIG = ClassificationConfig(
    scalers=SHARED_PREPROCESSORS,
    model_pool=SHARED_MODEL_POOL,
    hyperparameters=SHARED_HYPERPARAMETERS,
    classifier_instances=SHARED_CLASSIFIER_INSTANCES,
)

# ── Ensemble-baseline config (ARF / SRP / Leveraging Bagging / HAT pool) ──────
# Same shape, different candidate set: every model in this pool is already a
# drift-adaptive ensemble on its own, so this is what you pass a framework to
# search *over* baselines instead of tuning plain single models from scratch.

ENSEMBLE_CLASSIFICATION_CONFIG = ClassificationConfig(
    scalers=SHARED_PREPROCESSORS,
    model_pool=ENSEMBLE_MODEL_POOL,
    hyperparameters=SHARED_HYPERPARAMETERS,
    classifier_instances=ENSEMBLE_CLASSIFIER_INSTANCES,
)


def get_classification_config(pool: str = "normal") -> ClassificationConfig:
    """Return the :class:`ClassificationConfig` for the requested model pool.

    Parameters
    ----------
    pool : str
        ``"normal"`` / ``"plain"`` (default) — single models (Naive Bayes,
        Perceptron, Hoeffding Tree, ...), the pool a search framework tunes
        and combines from scratch.
        ``"ensemble"`` / ``"baseline"`` — drift-adaptive ensembles (ARF, SRP,
        Leveraging Bagging, Hoeffding Adaptive Tree) as the candidates
        instead, for measuring whether search still adds value once every
        candidate is already a strong baseline on its own.

    Examples
    --------
        from samlb.framework.classification.shared_config import get_classification_config
        from samlb.framework.classification.asml import AutoStreamClassifier

        cfg = get_classification_config(pool="ensemble")
        model = AutoStreamClassifier(config_dict=cfg.asml_config_dict(), seed=42)
    """
    normalized = pool.strip().lower()
    if normalized in ("normal", "plain", "default"):
        return DEFAULT_CLASSIFICATION_CONFIG
    if normalized in ("ensemble", "baseline", "ensemble_baseline"):
        return ENSEMBLE_CLASSIFICATION_CONFIG
    raise ValueError(
        f"pool must be 'normal' or 'ensemble', got {pool!r}."
    )
