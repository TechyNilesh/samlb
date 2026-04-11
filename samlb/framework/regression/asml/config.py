"""
ASML Regression — search space configuration.

Base models use SAMLB C++ backends where available.
River's AdaptiveRandomForestRegressor is kept as it has no C++ equivalent.
"""
from river import preprocessing, feature_selection, stats, forest

from samlb.framework.base._cpp_wrappers import (
    LinearRegression,
    BayesianLinearRegression,
    PassiveAggressiveRegressor,
    KNNRegressor,
    HoeffdingTreeRegressor,
)
from .helper import range_gen

# ── model pool ────────────────────────────────────────────────────────────────

model_options = [
    LinearRegression(),
    BayesianLinearRegression(),
    PassiveAggressiveRegressor(),
    KNNRegressor(),
    HoeffdingTreeRegressor(),
    forest.ARFRegressor(),   # River ARF — no C++ equivalent
]

# ── preprocessing ─────────────────────────────────────────────────────────────

preprocessor_options = [
    preprocessing.MinMaxScaler(),
    preprocessing.MaxAbsScaler(),
]

# ── feature selection ─────────────────────────────────────────────────────────

feature_selection_options = [
    feature_selection.SelectKBest(similarity=stats.PearsonCorr()),
]

# ── hyperparameter search spaces ──────────────────────────────────────────────

hyperparameters_options = {
    # C++ wrappers
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
    # River ARF
    "ARFRegressor": {
        "n_models":          range_gen(3, 9, step=2),
        "max_depth":         range_gen(5, 30, step=5),
        "grace_period":      range_gen(50, 500, step=50),
        "aggregation_method": ["mean", "median"],
        "leaf_prediction":   ["mean", "model", "adaptive"],
    },
    # River preprocessors
    "MinMaxScaler":  {},
    "MaxAbsScaler":  {},
    # River feature selectors
    "SelectKBest": {
        "k": range_gen(2, 25, step=1),
    },
}

# ── default config dict ───────────────────────────────────────────────────────

default_config_dict = {
    "models":          model_options,
    "preprocessors":   preprocessor_options,
    "features":        feature_selection_options,
    "hyperparameters": hyperparameters_options,
}
