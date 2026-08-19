"""
StreamingAutoGluon Regression — base learner configuration.

Mirrors the classification side: one base learner type per algorithm in the
shared C++ regression pool, with the shared preprocessors handed out
round-robin across those types. This is the same pool the other regression
frameworks search, so the only difference between results is the strategy.
"""
from samlb.framework.base import (
    BayesianLinearRegression,
    HoeffdingTreeRegressor,
    KNNRegressor,
    LinearRegression,
    MaxAbsScaler,
    MinMaxScaler,
    PassiveAggressiveRegressor,
    StandardScaler,
)

# One base learner type per algorithm in the C++ regression pool.
SAG_REG_BASE_LEARNERS: list = [
    LinearRegression(),
    BayesianLinearRegression(),
    PassiveAggressiveRegressor(),
    KNNRegressor(),
    HoeffdingTreeRegressor(),
]

# Assigned round-robin across the base types (one scaler per type).
SAG_REG_SCALERS: list = [
    MinMaxScaler(),
    StandardScaler(),
    MaxAbsScaler(),
]

# Cheaper, deliberately diverse subset: a tree, a linear and a distance model.
SAG_REG_BASE_LEARNERS_SMALL: list = [
    HoeffdingTreeRegressor(),
    LinearRegression(),
    KNNRegressor(),
]

__all__ = [
    "SAG_REG_BASE_LEARNERS",
    "SAG_REG_BASE_LEARNERS_SMALL",
    "SAG_REG_SCALERS",
]
