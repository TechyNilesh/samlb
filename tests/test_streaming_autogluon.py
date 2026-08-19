"""
Tests for StreamingAutoGluon.

These use samlb.datasets.stream rather than the raw CSV/ARFF fixtures in
conftest, so they are self-contained.
"""
import itertools

import pytest

from samlb.datasets import stream
from samlb.framework.classification.sag import (
    SAG_BASE_LEARNERS_SMALL,
    StreamingAutoGluon,
)
from samlb.metrics import WindowMetric


@pytest.fixture(scope="module")
def data():
    return list(itertools.islice(stream("electricity", task="classification"), 1500))


def _prequential(model, data):
    correct = 0
    for x, y in data:
        correct += model.predict_one(x) == y
        model.learn_one(x, y)
    return correct / len(data)


def test_learns_better_than_chance(data):
    acc = _prequential(StreamingAutoGluon(seed=42, learners=SAG_BASE_LEARNERS_SMALL), data)
    assert acc > 0.6


def test_deterministic(data):
    a = _prequential(StreamingAutoGluon(seed=42, learners=SAG_BASE_LEARNERS_SMALL), data)
    b = _prequential(StreamingAutoGluon(seed=42, learners=SAG_BASE_LEARNERS_SMALL), data)
    assert a == b


def test_holdout_rotation_is_uniform(data):
    """Every fold learner must miss exactly 1/k of the instances."""
    k = 3
    model = StreamingAutoGluon(seed=42, n_folds=k, learners=SAG_BASE_LEARNERS_SMALL)
    counts = [0] * k

    for idx, learner in enumerate(model._fold_learners[0]):
        original = learner.learn_one

        def wrapper(x, y, _idx=idx, _orig=original):
            counts[_idx] += 1
            return _orig(x, y)

        learner.learn_one = wrapper

    n = 900
    for x, y in data[:n]:
        model.learn_one(x, y)

    assert counts == [n - n // k] * k


def test_meta_features_are_added_per_type_and_class(data):
    model = StreamingAutoGluon(seed=42, learners=SAG_BASE_LEARNERS_SMALL)
    for x, y in data[:200]:
        model.learn_one(x, y)

    x = data[200][0]
    augmented = model._augment(x, model._aggregate(model._fold_votes(x)))
    meta = [k for k in augmented if k.startswith("P_")]

    assert len(meta) == len(model._names) * len(model._classes)
    assert all(k in augmented for k in x)                     # raw features kept
    for name in model._names:                                 # per type, votes sum to ~1
        total = sum(augmented[f"P_{name}_{c}"] for c in model._classes)
        assert total == pytest.approx(1.0, abs=1e-9)


def test_fold_votes_cached_per_instance(data):
    """predict_one followed by learn_one must only run the fold learners once."""
    model = StreamingAutoGluon(seed=42, learners=SAG_BASE_LEARNERS_SMALL)
    for x, y in data[:100]:
        model.learn_one(x, y)

    calls = [0]
    learner = model._fold_learners[0][0]
    original = learner.predict_proba_one

    def counting(x, _orig=original):
        calls[0] += 1
        return _orig(x)

    learner.predict_proba_one = counting

    x, y = data[100]
    model.predict_one(x)
    model.learn_one(x, y)
    assert calls[0] == 1


def test_reset_clears_state(data):
    model = StreamingAutoGluon(seed=42, learners=SAG_BASE_LEARNERS_SMALL)
    first = _prequential(model, data)
    model.reset()
    assert model._seen == 0
    assert model._classes == []
    assert _prequential(model, data) == first


@pytest.mark.parametrize("n_folds", [2, 3, 5])
def test_n_folds(data, n_folds):
    model = StreamingAutoGluon(seed=42, n_folds=n_folds, learners=SAG_BASE_LEARNERS_SMALL)
    assert _prequential(model, data) > 0.6
    assert all(len(row) == n_folds for row in model._fold_learners)


@pytest.mark.parametrize("metric", ["accuracy", "f1"])
def test_combination_metric(data, metric):
    model = StreamingAutoGluon(seed=42, metric=metric, learners=SAG_BASE_LEARNERS_SMALL)
    assert _prequential(model, data) > 0.6
    assert all(0.0 <= w <= 1.0 for w in model.stacked_weights().values())


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        StreamingAutoGluon(n_folds=1)
    with pytest.raises(ValueError):
        StreamingAutoGluon(metric="nope")
    with pytest.raises(ValueError):
        StreamingAutoGluon(learners=[])


def test_predicts_before_any_training():
    model = StreamingAutoGluon(seed=42, learners=SAG_BASE_LEARNERS_SMALL)
    assert model.predict_one({"a": 1.0, "b": 2.0}) == 0
    assert model.predict_proba_one({"a": 1.0, "b": 2.0}) == {}


# ── WindowMetric ─────────────────────────────────────────────────────────────

def test_window_metric_slides():
    m = WindowMetric(window_size=4)
    for _ in range(4):
        m.update(1, 1)
    assert m.accuracy() == 1.0
    for _ in range(4):                       # push the correct ones out
        m.update(0, 1)
    assert m.size == 4
    assert m.accuracy() == 0.0


def test_window_metric_macro_f1_ignores_unobserved_classes():
    m = WindowMetric(window_size=10)
    m.update(0, 0)
    m.update(1, 1)
    assert m.macro_f1() == pytest.approx(1.0)
    assert m.get(WindowMetric.METRIC_ACCURACY) == pytest.approx(1.0)
    assert m.get(WindowMetric.METRIC_F1) == pytest.approx(1.0)


def test_window_metric_grows_class_counters():
    m = WindowMetric(window_size=10)
    m.update(7, 7)                            # class index far beyond the initial size
    assert m.accuracy() == 1.0
    assert m.macro_f1() == pytest.approx(1.0)


# ── Regression variant ───────────────────────────────────────────────────────

from samlb.framework.regression.sag import (            # noqa: E402
    SAG_REG_BASE_LEARNERS_SMALL,
    StreamingAutoGluonRegressor,
)
from samlb.metrics import WindowRegressionMetric        # noqa: E402


@pytest.fixture(scope="module")
def reg_data():
    return list(itertools.islice(stream("bike", task="regression"), 1500))


def _prequential_reg(model, data):
    """Mean absolute error over a prequential pass."""
    total = 0.0
    for x, y in data:
        total += abs(model.predict_one(x) - y)
        model.learn_one(x, y)
    return total / len(data)


def test_reg_beats_predicting_the_mean(reg_data):
    ys = [y for _, y in reg_data]
    baseline = sum(abs(y - sum(ys) / len(ys)) for y in ys) / len(ys)
    model = StreamingAutoGluonRegressor(seed=42, learners=SAG_REG_BASE_LEARNERS_SMALL)
    assert _prequential_reg(model, reg_data) < baseline


def test_reg_deterministic(reg_data):
    a = _prequential_reg(
        StreamingAutoGluonRegressor(seed=42, learners=SAG_REG_BASE_LEARNERS_SMALL), reg_data)
    b = _prequential_reg(
        StreamingAutoGluonRegressor(seed=42, learners=SAG_REG_BASE_LEARNERS_SMALL), reg_data)
    assert a == b


def test_reg_holdout_rotation_is_uniform(reg_data):
    k = 3
    model = StreamingAutoGluonRegressor(
        seed=42, n_folds=k, learners=SAG_REG_BASE_LEARNERS_SMALL)
    counts = [0] * k
    for idx, learner in enumerate(model._fold_learners[0]):
        original = learner.learn_one

        def wrapper(x, y, _idx=idx, _orig=original):
            counts[_idx] += 1
            return _orig(x, y)

        learner.learn_one = wrapper

    n = 900
    for x, y in reg_data[:n]:
        model.learn_one(x, y)
    assert counts == [n - n // k] * k


def test_reg_meta_features_one_per_type(reg_data):
    model = StreamingAutoGluonRegressor(seed=42, learners=SAG_REG_BASE_LEARNERS_SMALL)
    for x, y in reg_data[:200]:
        model.learn_one(x, y)

    x = reg_data[200][0]
    augmented = model._augment(x, model._aggregate(model._fold_predictions(x)))
    meta = [k for k in augmented if k.startswith("P_")]
    assert len(meta) == len(model._names)
    assert all(k in augmented for k in x)


def test_reg_clip_bounds_predictions(reg_data):
    model = StreamingAutoGluonRegressor(
        seed=42, clip=True, learners=SAG_REG_BASE_LEARNERS_SMALL)
    seen = []
    for x, y in reg_data[:500]:
        p = model.predict_one(x)
        if seen:
            assert min(seen) <= p <= max(seen)
        model.learn_one(x, y)
        seen.append(y)


def test_reg_lower_error_gets_more_weight(reg_data):
    model = StreamingAutoGluonRegressor(seed=42, learners=SAG_REG_BASE_LEARNERS_SMALL)
    for x, y in reg_data[:800]:
        model.learn_one(x, y)
    errors = model.stacked_errors()
    weights = model.stacked_weights()
    best = min(errors, key=errors.get)
    worst = max(errors, key=errors.get)
    assert weights[best] >= weights[worst]
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", ["mae", "rmse"])
def test_reg_metric_option(reg_data, metric):
    model = StreamingAutoGluonRegressor(
        seed=42, metric=metric, learners=SAG_REG_BASE_LEARNERS_SMALL)
    assert _prequential_reg(model, reg_data) > 0


def test_reg_rejects_bad_arguments():
    with pytest.raises(ValueError):
        StreamingAutoGluonRegressor(n_folds=1)
    with pytest.raises(ValueError):
        StreamingAutoGluonRegressor(metric="accuracy")
    with pytest.raises(ValueError):
        StreamingAutoGluonRegressor(learners=[])


def test_window_regression_metric_slides():
    m = WindowRegressionMetric(window_size=3)
    for yt, yp in [(1, 1), (2, 4), (3, 3), (10, 0)]:
        m.update(yt, yp)
    assert m.size == 3
    assert m.mae() == pytest.approx((2 + 0 + 10) / 3)
    assert m.rmse() == pytest.approx(((4 + 0 + 100) / 3) ** 0.5)
    assert m.get(WindowRegressionMetric.METRIC_MAE) == pytest.approx(m.mae())
    assert m.get(WindowRegressionMetric.METRIC_RMSE) == pytest.approx(m.rmse())
