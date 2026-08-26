"""
Tests for FIMTDDRegressor and the adapted LeveragingBaggingRegressor.
"""
import itertools
import random

import pytest

from samlb.datasets import stream
from samlb.framework.base import (
    FIMTDDRegressor,
    HoeffdingTreeRegressor,
    LeveragingBaggingRegressor,
)


def _step_stream(n=6000, seed=0):
    """y is +1 inside a band of x0 and -1 outside: trivial for a tree with real
    split candidates, unfittable by any global linear model."""
    rng = random.Random(seed)
    for _ in range(n):
        x = {"x0": rng.random(), "x1": rng.random()}
        yield x, (1.0 if 0.3 < x["x0"] < 0.7 else -1.0)


def _late_mae(model, data, skip=2000):
    err = n = 0.0
    for i, (x, y) in enumerate(data):
        if i > skip:
            err += abs(model.predict_one(x) - y)
            n += 1
        model.learn_one(x, y)
    return err / n


def _r2(model, data):
    se = 0.0
    n = 0
    mean = 0.0
    m2 = 0.0
    peak = 0.0
    for x, y in data:
        y = float(y)
        p = model.predict_one(x)
        peak = max(peak, abs(p))
        se += (p - y) ** 2
        n += 1
        d = y - mean
        mean += d / n
        m2 += d * (y - mean)
        model.learn_one(x, y)
    return (1.0 - se / m2 if m2 > 0 else float("nan")), peak


# ── FIMT-DD ──────────────────────────────────────────────────────────────────

def test_fimtdd_splits_where_hoeffding_tree_regressor_does_not():
    """The E-BST observers are the point: exact split candidates instead of a
    Gaussian summary. A constant predictor scores 1.0 here."""
    data = list(_step_stream())
    fimtdd = FIMTDDRegressor(grace_period=200, leaf_prediction="mean")
    assert _late_mae(fimtdd, data) < 0.05
    assert fimtdd.n_splits() >= 2
    assert _late_mae(HoeffdingTreeRegressor(grace_period=200), list(data)) > 0.5


def test_fimtdd_deterministic():
    data = list(_step_stream())
    a = _late_mae(FIMTDDRegressor(seed=1), list(data))
    b = _late_mae(FIMTDDRegressor(seed=1), list(data))
    assert a == b


@pytest.mark.parametrize("dataset", ["ailerons", "elevators"])
def test_reference_perceptron_diverges_and_the_guard_holds(dataset):
    """MOA's FIMT-DD perceptron normalises by running global feature statistics
    and can leave the target's range entirely. Pinning both halves: that the
    unguarded form still reproduces the defect, and that the default does not."""
    # 20k, not a shorter prefix: on elevators the unguarded perceptron has not
    # left the target's range yet at 8k, so a shorter run would not exercise
    # the very thing this test pins.
    data = list(itertools.islice(stream(dataset, task="regression"), 20000))
    y_max = max(abs(float(y)) for _, y in data)

    raw_r2, raw_peak = _r2(FIMTDDRegressor(leaf_prediction="perceptron"), data)
    safe_r2, safe_peak = _r2(FIMTDDRegressor(leaf_prediction="adaptive"), data)

    assert raw_peak > y_max            # predicts outside anything ever observed
    assert safe_peak <= y_max + 1e-9   # bounded to the leaf's own target range
    assert safe_r2 > raw_r2


def test_fimtdd_rejects_unknown_leaf_prediction():
    with pytest.raises(ValueError):
        FIMTDDRegressor(leaf_prediction="linear")


# ── Leveraging Bagging (adapted) ─────────────────────────────────────────────

def test_leveraging_bagging_regressor_learns():
    data = list(_step_stream())
    assert _late_mae(LeveragingBaggingRegressor(n_models=5, seed=1), data) < 0.3


def test_leveraging_bagging_regressor_deterministic():
    data = list(_step_stream())
    a = _late_mae(LeveragingBaggingRegressor(n_models=5, seed=1), list(data))
    b = _late_mae(LeveragingBaggingRegressor(n_models=5, seed=1), list(data))
    assert a == b


def test_leveraging_bagging_default_lambda_is_the_leveraged_one():
    """lambda 6, not online bagging's 1 — that is what 'leveraging' means."""
    assert LeveragingBaggingRegressor().lambda_value == 6.0


# ── pool symmetry ────────────────────────────────────────────────────────────

def test_regression_pool_matches_the_classification_pool():
    from samlb.framework.classification.shared_config import (
        ENSEMBLE_MODEL_POOL as cls_pool)
    from samlb.framework.regression.shared_config import (
        ENSEMBLE_MODEL_POOL as reg_pool, SHARED_HYPERPARAMETERS as reg_hp)
    assert len(reg_pool) == len(cls_pool)
    names = {type(m).__name__ for m in reg_pool}
    assert {"LeveragingBaggingRegressor", "FIMTDDRegressor"} <= names
    assert "FIMTDDRegressor" in reg_hp and "LeveragingBaggingRegressor" in reg_hp
