"""samlb.framework.base — abstract framework base and C++ component wrappers."""
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
    EFDTClassifier,
    SGTClassifier,
    ARFClassifier,
    SRPClassifier,
    LeveragingBaggingClassifier,
    HoeffdingAdaptiveTreeClassifier,
    # regression
    LinearRegression,
    BayesianLinearRegression,
    PassiveAggressiveRegressor,
    KNNRegressor,
    HoeffdingTreeRegressor,
    ARFRegressor,
    SRPRegressor,
    # preprocessing / feature selection
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    VarianceThreshold,
    SelectKBest,
)

__all__ = [
    "BaseStreamFramework",
    "NaiveBayes", "Perceptron", "LogisticRegression",
    "PassiveAggressiveClassifier", "SoftmaxRegression",
    "KNNClassifier", "HoeffdingTreeClassifier", "EFDTClassifier", "SGTClassifier",
    "ARFClassifier", "SRPClassifier",
    "LeveragingBaggingClassifier", "HoeffdingAdaptiveTreeClassifier",
    "LinearRegression", "BayesianLinearRegression",
    "PassiveAggressiveRegressor", "KNNRegressor", "HoeffdingTreeRegressor",
    "ARFRegressor", "SRPRegressor",
    "StandardScaler", "MinMaxScaler", "MaxAbsScaler",
    "VarianceThreshold", "SelectKBest",
]
