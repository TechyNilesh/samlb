"""
Tests for SGBT / SGBR, the C++ streaming gradient boosting learners.

Reference behaviour is checked against CapyMOA where it is installed; those
tests skip otherwise, so the suite still runs without a JVM.
"""
import itertools

import pytest

from samlb.datasets import stream
from samlb.framework.base import SGBTClassifier, SGBRRegressor


@pytest.fixture(scope="module")
def cls_data():
    return list(itertools.islice(stream("electricity", task="classification"), 3000))


@pytest.fixture(scope="module")
def reg_data():
    return list(itertools.islice(stream("fried", task="regression"), 3000))


def _accuracy(model, data):
    correct = 0
    for x, y in data:
        correct += model.predict_one(x) == y
        model.learn_one(x, y)
    return correct / len(data)


def _r2(model, data):
    se = 0.0
    n = 0
    mean = 0.0
    m2 = 0.0
    for x, y in data:
        y = float(y)
        p = model.predict_one(x)
        se += (p - y) ** 2
        n += 1
        d = y - mean
        mean += d / n
        m2 += d * (y - mean)
    return 1.0 - se / m2 if m2 > 0 else float("nan")


def _fit_r2(model, data):
    se = 0.0
    n = 0
    mean = 0.0
    m2 = 0.0
    for x, y in data:
        y = float(y)
        p = model.predict_one(x)
        se += (p - y) ** 2
        n += 1
        d = y - mean
        mean += d / n
        m2 += d * (y - mean)
        model.learn_one(x, y)
    return 1.0 - se / m2 if m2 > 0 else float("nan")


# ── defaults are the two papers' defaults, and they differ ───────────────────

def test_classifier_uses_the_2024_classification_defaults():
    m = SGBTClassifier()
    assert (m.n_models, m.learning_rate, m.bag_size) == (100, 0.0125, 1)


def test_regressor_uses_the_2025_sgbr_defaults():
    """SGBR is a separate method, not the classifier retargeted: a bagged base
    learner, ten iterations, and a learning rate of 1.0."""
    m = SGBRRegressor()
    assert (m.n_models, m.learning_rate, m.bag_size) == (10, 1.0, 10)
    assert (m.grace_period, m.split_confidence) == (50, 0.01)


# ── behaviour ────────────────────────────────────────────────────────────────

def test_classifier_learns(cls_data):
    assert _accuracy(SGBTClassifier(n_models=25, n_classes=2, seed=1), cls_data) > 0.75


def test_classifier_deterministic(cls_data):
    a = _accuracy(SGBTClassifier(n_models=10, n_classes=2, seed=1), cls_data)
    b = _accuracy(SGBTClassifier(n_models=10, n_classes=2, seed=1), cls_data)
    assert a == b


def test_classifier_handles_multiclass_by_discovery(cls_data):
    """With no schema the label set is discovered online, one booster per class."""
    m = SGBTClassifier(n_models=5, seed=1)
    for x, y in cls_data[:500]:
        m.learn_one(x, y)
    proba = m.predict_proba_one(cls_data[500][0])
    assert set(proba) == {0, 1}
    assert abs(sum(proba.values()) - 1.0) < 1e-9


def test_regressor_learns(reg_data):
    assert _fit_r2(SGBRRegressor(n_models=5, bag_size=3, seed=1), reg_data) > 0.3


def test_regressor_shrinkage_is_what_the_learning_rate_controls(reg_data):
    """The classifier's learning rate on a continuous target reaches only
    1-(1-lr)^n of the target, and the resulting bias destroys R^2. This is the
    bug the SGBR defaults exist to avoid — keep it pinned."""
    good = _fit_r2(SGBRRegressor(n_models=10, bag_size=3, seed=1), reg_data)
    shrunk = _fit_r2(
        SGBRRegressor(n_models=10, bag_size=3, learning_rate=0.0125, seed=1), reg_data)
    assert good > 0.3
    assert shrunk < good


def test_regressor_predictions_are_finite(reg_data):
    import math
    m = SGBRRegressor(n_models=5, bag_size=3, seed=1)
    for x, y in reg_data:
        assert math.isfinite(m.predict_one(x))
        m.learn_one(x, y)


def test_reset_clears_state(cls_data):
    m = SGBTClassifier(n_models=5, n_classes=2, seed=1)
    for x, y in cls_data[:300]:
        m.learn_one(x, y)
    m.reset()
    assert m.predict_proba_one(cls_data[0][0]) == {}


@pytest.mark.parametrize("cls", [SGBTClassifier, SGBRRegressor])
def test_invalid_leaf_prediction_rejected(cls):
    with pytest.raises(ValueError):
        cls(leaf_prediction="linear")


def test_base_learner_is_fimtdd_configured_as_the_reference():
    """SGBT/SGBR are defined over FIMT-DD with mean leaves — MOA's ``-e``."""
    assert SGBTClassifier().leaf_prediction == "mean"
    assert SGBRRegressor().leaf_prediction == "mean"


# ── pool membership ──────────────────────────────────────────────────────────

def test_present_in_both_ensemble_pools():
    from samlb.framework.classification.shared_config import (
        ENSEMBLE_MODEL_POOL as cls_pool, SHARED_HYPERPARAMETERS as cls_hp)
    from samlb.framework.regression.shared_config import (
        ENSEMBLE_MODEL_POOL as reg_pool, SHARED_HYPERPARAMETERS as reg_hp)
    assert any(isinstance(m, SGBTClassifier) for m in cls_pool)
    assert any(isinstance(m, SGBRRegressor) for m in reg_pool)
    # A search framework needs a space to tune over, keyed by class name.
    assert "SGBTClassifier" in cls_hp and "SGBRRegressor" in reg_hp


@pytest.mark.parametrize("cls,params", [
    (SGBTClassifier, {"n_models": 20, "bag_size": 5, "learning_rate": 0.05}),
    (SGBRRegressor, {"n_models": 5, "bag_size": 20}),
])
def test_clone_with_new_params(cls, params):
    """Every search framework reconfigures candidates through clone()."""
    m = cls().clone(params)
    for k, v in params.items():
        assert getattr(m, k) == v
