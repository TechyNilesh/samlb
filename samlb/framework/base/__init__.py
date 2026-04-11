"""samlb.framework.base — abstract base class and River-compatible C++ wrappers."""
from ._framework import BaseStreamFramework
from ._cpp_wrappers import (
    # classification
    NaiveBayes,
    Perceptron,
    LogisticRegression,
    PassiveAggressiveClassifier,
    SoftmaxRegression,
    KNNClassifier,
    HoeffdingTreeClassifier,
    SGTClassifier,
    # regression
    LinearRegression,
    BayesianLinearRegression,
    PassiveAggressiveRegressor,
    KNNRegressor,
    HoeffdingTreeRegressor,
)

__all__ = [
    "BaseStreamFramework",
    "NaiveBayes", "Perceptron", "LogisticRegression",
    "PassiveAggressiveClassifier", "SoftmaxRegression",
    "KNNClassifier", "HoeffdingTreeClassifier", "SGTClassifier",
    "LinearRegression", "BayesianLinearRegression",
    "PassiveAggressiveRegressor", "KNNRegressor", "HoeffdingTreeRegressor",
]
