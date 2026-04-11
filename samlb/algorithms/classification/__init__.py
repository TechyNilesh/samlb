"""
samlb.algorithms.classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fast C++ streaming classification algorithms with River-compatible API.
All classes expose: predict_one(x), learn_one(x, y), predict_proba_one(x), reset()
"""
from samlb._samlb_core import (
    NaiveBayes,
    Perceptron,
    LogisticRegressionClassifier,
    PassiveAggressiveClassifier,
    SoftmaxRegression,
    KNNClassifier,
    HoeffdingTreeClassifier,
    EFDTClassifier,
    SGTClassifier,
)

__all__ = [
    "NaiveBayes",
    "Perceptron",
    "LogisticRegressionClassifier",
    "PassiveAggressiveClassifier",
    "SoftmaxRegression",
    "KNNClassifier",
    "HoeffdingTreeClassifier",
    "EFDTClassifier",
    "SGTClassifier",
]
