"""
samlb.algorithms.regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fast C++ streaming regression algorithms with River-compatible API.
All classes expose: predict_one(x), learn_one(x, y), reset()
"""
from samlb._samlb_core import (
    LinearRegression,
    BayesianLinearRegression,
    PassiveAggressiveRegressor,
    KNNRegressor,
    HoeffdingTreeRegressor,
)

__all__ = [
    "LinearRegression",
    "BayesianLinearRegression",
    "PassiveAggressiveRegressor",
    "KNNRegressor",
    "HoeffdingTreeRegressor",
]
