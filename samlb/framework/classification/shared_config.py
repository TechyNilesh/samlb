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
from typing import Dict, List, Optional

from river import preprocessing

from samlb.framework.base._cpp_wrappers import (
    HoeffdingTreeClassifier,
    KNNClassifier,
    LogisticRegression,
    NaiveBayes,
    PassiveAggressiveClassifier,
    Perceptron,
    SGTClassifier,
    SoftmaxRegression,
)
from samlb.framework.classification.asml.helper import range_gen

# ── Preprocessors ─────────────────────────────────────────────────────────────
# Same for all frameworks.

SHARED_PREPROCESSORS = [
    preprocessing.MinMaxScaler(),
    preprocessing.StandardScaler(),
    preprocessing.MaxAbsScaler(),
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
        from river import preprocessing

        cfg = ClassificationConfig(
            scalers=[preprocessing.MinMaxScaler(), preprocessing.StandardScaler()],
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
        from river import feature_selection, stats
        from samlb.framework.classification.asml.helper import range_gen
        return {
            "models":          self.model_pool,
            "preprocessors":   self.scalers,
            "features": [
                feature_selection.VarianceThreshold(threshold=0),
                feature_selection.SelectKBest(similarity=stats.PearsonCorr()),
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
