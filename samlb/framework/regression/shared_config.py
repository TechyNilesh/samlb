"""
samlb.framework.regression.shared_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Single source of truth for the C++ algorithm pool used by the regression
frameworks (ASML, EvoAutoML). Mirrors
:mod:`samlb.framework.classification.shared_config` — same
``RegressionConfig`` shape, same ``pool="normal"|"ensemble"`` switch — so
classification and regression are configured the same way.

Only SAMLB C++ wrappers are used — no River Python regressors.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List

from samlb.framework.base import MaxAbsScaler, MinMaxScaler, SelectKBest, StandardScaler

from samlb.framework.base._cpp_wrappers import (
    ARFRegressor,
    BayesianLinearRegression,
    HoeffdingTreeRegressor,
    KNNRegressor,
    LinearRegression,
    PassiveAggressiveRegressor,
    SRPRegressor,
)
from samlb.framework.regression.asml.helper import range_gen

# ── Preprocessors ─────────────────────────────────────────────────────────────

SHARED_PREPROCESSORS = [
    MinMaxScaler(),
    StandardScaler(),
    MaxAbsScaler(),
]

# ── Base regressor pool (one default instance per type) ───────────────────────
# Used by ASML (ARDNS search).

SHARED_MODEL_POOL = [
    LinearRegression(),
    BayesianLinearRegression(),
    PassiveAggressiveRegressor(),
    KNNRegressor(),
    HoeffdingTreeRegressor(),
]

# ── Ensemble baseline pool ──────────────────────────────────────────────────
# Drift-adaptive ensembles (ARF, SRP) instead of the plain single-model pool
# above. Pass this via ``get_regression_config(pool="ensemble")`` when a
# search framework should pick among strong streaming baselines rather than
# tune simple models from scratch.

ENSEMBLE_MODEL_POOL = [
    ARFRegressor(),
    SRPRegressor(),
]

# ── Hyperparameter search spaces ──────────────────────────────────────────────
# Keyed by class name — used by ASML (ARDNS).

SHARED_HYPERPARAMETERS = {
    "LinearRegression": {
        "learning_rate": range_gen(0.001, 0.1, step=0.005, float_n=True),
        "l2":            range_gen(0.0, 0.01, step=0.001, float_n=True),
    },
    "BayesianLinearRegression": {
        "alpha": range_gen(0.1, 10.0, step=0.5, float_n=True),
        "beta":  range_gen(0.1, 10.0, step=0.5, float_n=True),
    },
    "PassiveAggressiveRegressor": {
        "C":       range_gen(0.1, 10.0, step=0.5, float_n=True),
        "epsilon": range_gen(0.0, 0.5, step=0.05, float_n=True),
    },
    "KNNRegressor": {
        "n_neighbors": range_gen(2, 20, step=2),
        "window_size": range_gen(200, 2000, step=200),
        "p":           [1, 2],
    },
    "HoeffdingTreeRegressor": {
        "grace_period":     range_gen(50, 500, step=50),
        "split_confidence": [1e-9, 1e-7, 1e-4, 1e-2],
        "tie_threshold":    range_gen(0.02, 0.08, step=0.01, float_n=True),
        "max_depth":        range_gen(10, 100, step=10),
        "learning_rate":    range_gen(0.001, 0.1, step=0.005, float_n=True),
    },
    "ARFRegressor": {
        "n_models":     range_gen(5, 30, step=5),
        "lambda_value": range_gen(1.0, 10.0, step=1.0, float_n=True),
        "grace_period": range_gen(50, 500, step=50),
        "max_depth":    range_gen(10, 100, step=10),
    },
    "SRPRegressor": {
        "n_models":          range_gen(5, 30, step=5),
        "lambda_value":      range_gen(1.0, 10.0, step=1.0, float_n=True),
        "subspace_fraction": range_gen(0.2, 0.9, step=0.1, float_n=True),
        "training_method":   ["patches", "subspaces", "resampling"],
    },
    # Preprocessors
    "MinMaxScaler":   {},
    "StandardScaler": {"with_std": [True, False]},
    "MaxAbsScaler":   {},
    # Feature selectors
    "SelectKBest": {
        "k": range_gen(2, 25, step=1),
    },
}

# ── Pre-configured instances (all hyper combos) ───────────────────────────────
# Used by EvoAutoML (param_grid).

SHARED_REGRESSOR_INSTANCES = [
    # Linear Regression variants
    LinearRegression(learning_rate=0.001),
    LinearRegression(learning_rate=0.01),
    LinearRegression(learning_rate=0.1),
    LinearRegression(learning_rate=0.01, l2=0.001),
    # Bayesian Linear Regression variants
    BayesianLinearRegression(alpha=0.1, beta=1.0),
    BayesianLinearRegression(alpha=0.5, beta=1.0),
    BayesianLinearRegression(alpha=1.0, beta=1.0),
    BayesianLinearRegression(alpha=1.0, beta=0.1),
    # Passive Aggressive Regressor variants
    PassiveAggressiveRegressor(C=0.1,  epsilon=0.01),
    PassiveAggressiveRegressor(C=0.5,  epsilon=0.05),
    PassiveAggressiveRegressor(C=1.0,  epsilon=0.1),
    PassiveAggressiveRegressor(C=5.0,  epsilon=0.1),
    # Hoeffding Tree Regressor variants
    HoeffdingTreeRegressor(grace_period=50,  max_depth=10, learning_rate=0.01),
    HoeffdingTreeRegressor(grace_period=100, max_depth=20, learning_rate=0.01),
    HoeffdingTreeRegressor(grace_period=200, max_depth=30, learning_rate=0.01),
    HoeffdingTreeRegressor(grace_period=500, max_depth=50, learning_rate=0.001),
    # KNN Regressor variants
    KNNRegressor(n_neighbors=3,  window_size=500),
    KNNRegressor(n_neighbors=5,  window_size=1000),
    KNNRegressor(n_neighbors=10, window_size=2000),
    KNNRegressor(n_neighbors=5,  window_size=500,  p=1),
]

# Ensemble-baseline counterpart of SHARED_REGRESSOR_INSTANCES, for EvoAutoML
# (param_grid) when the pool should be drift-adaptive ensembles rather than
# plain single models.

ENSEMBLE_REGRESSOR_INSTANCES = [
    # ARF
    ARFRegressor(n_models=10),
    ARFRegressor(n_models=20, lambda_value=8.0),
    ARFRegressor(n_models=10, grace_period=200),
    # SRP
    SRPRegressor(n_models=10),
    SRPRegressor(n_models=10, training_method="subspaces"),
    SRPRegressor(n_models=10, subspace_fraction=0.4),
]


# ── RegressionConfig — unified config object ──────────────────────────────────

@dataclasses.dataclass
class RegressionConfig:
    """Single config object passed to the regression frameworks.

    Mirrors :class:`samlb.framework.classification.shared_config.ClassificationConfig`.

    Attributes
    ----------
    scalers : list
        Preprocessor instances (MinMaxScaler, StandardScaler, ...).
    model_pool : list
        One default instance per algorithm type. Used by ASML (ARDNS search).
    hyperparameters : dict
        Hyperparameter search spaces keyed by class name. Used by ASML.
    regressor_instances : list
        Pre-configured instances at various hyperparameter settings. Used by
        EvoAutoML (param_grid).
    """

    scalers:             List
    model_pool:          List
    hyperparameters:     Dict
    regressor_instances: List

    def asml_config_dict(self) -> dict:
        """Config dict in the format AutoStreamRegressor expects."""
        return {
            "models":          self.model_pool,
            "preprocessors":   self.scalers,
            "features":        [SelectKBest()],
            "hyperparameters": self.hyperparameters,
        }

    def eaml_param_grid(self) -> dict:
        """param_grid in the format EvolutionaryBaggingRegressor expects."""
        return {
            "Scaler":    self.scalers,
            "Regressor": self.regressor_instances,
        }


# ── Default config (the shared C++ pool) ──────────────────────────────────────

DEFAULT_REGRESSION_CONFIG = RegressionConfig(
    scalers=SHARED_PREPROCESSORS,
    model_pool=SHARED_MODEL_POOL,
    hyperparameters=SHARED_HYPERPARAMETERS,
    regressor_instances=SHARED_REGRESSOR_INSTANCES,
)

# ── Ensemble-baseline config (ARF / SRP pool) ─────────────────────────────────

ENSEMBLE_REGRESSION_CONFIG = RegressionConfig(
    scalers=SHARED_PREPROCESSORS,
    model_pool=ENSEMBLE_MODEL_POOL,
    hyperparameters=SHARED_HYPERPARAMETERS,
    regressor_instances=ENSEMBLE_REGRESSOR_INSTANCES,
)


def get_regression_config(pool: str = "normal") -> RegressionConfig:
    """Return the :class:`RegressionConfig` for the requested model pool.

    Parameters
    ----------
    pool : str
        ``"normal"`` / ``"plain"`` (default) — single models (Linear
        Regression, KNN, Hoeffding Tree, ...), the pool a search framework
        tunes and combines from scratch.
        ``"ensemble"`` / ``"baseline"`` — drift-adaptive ensembles (ARF, SRP)
        as the candidates instead.

    Examples
    --------
        from samlb.framework.regression.shared_config import get_regression_config
        from samlb.framework.regression.asml import AutoStreamRegressor

        cfg = get_regression_config(pool="ensemble")
        model = AutoStreamRegressor(config_dict=cfg.asml_config_dict())
    """
    normalized = pool.strip().lower()
    if normalized in ("normal", "plain", "default"):
        return DEFAULT_REGRESSION_CONFIG
    if normalized in ("ensemble", "baseline", "ensemble_baseline"):
        return ENSEMBLE_REGRESSION_CONFIG
    raise ValueError(
        f"pool must be 'normal' or 'ensemble', got {pool!r}."
    )
