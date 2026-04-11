"""
EvoAutoML Regression — search space configuration.

The param_grid has two top-level keys:
  "Scaler"    — list of scaler instances (any may be swapped in on mutation)
  "Regressor" — list of regressor instances at various hyperparameter settings

Using SAMLB C++ wrappers for maximum throughput.
"""
from river import preprocessing

from samlb.framework.base._cpp_wrappers import (
    BayesianLinearRegression,
    HoeffdingTreeRegressor,
    KNNRegressor,
    LinearRegression,
    PassiveAggressiveRegressor,
)

EAML_REG_PARAM_GRID: dict = {
    "Scaler": [
        preprocessing.MinMaxScaler(),
        preprocessing.StandardScaler(),
        preprocessing.MaxAbsScaler(),
    ],
    "Regressor": [
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
    ],
}
