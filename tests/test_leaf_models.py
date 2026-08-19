"""
Leaf-model regression tests.

Both trees pick their leaf predictor adaptively — the classifier between
majority class and naive Bayes, the regressor between the leaf mean and a
linear model. Two failure modes are easy to reintroduce and hard to notice:

* a leaf predictor that is disabled by default, so the tree degenerates to a
  constant predictor (this is what made the classifier score exactly its
  majority-class baseline on `insects`);
* a leaf linear model that extrapolates wildly when the same instance is
  learned repeatedly, which is what bagging frameworks do. A single spike
  leaves MAE almost untouched while destroying R², so it hides from
  single-learner testing.
"""
import itertools

import pytest

from samlb.datasets import stream
from samlb.framework.base import HoeffdingTreeClassifier, HoeffdingTreeRegressor


@pytest.fixture(scope="module")
def insects():
    return list(itertools.islice(stream("insects", task="classification"), 5000))


@pytest.fixture(scope="module")
def ailerons():
    return list(itertools.islice(stream("ailerons", task="regression"), 5000))


def _majority_baseline(data):
    from collections import Counter
    seen, correct = Counter(), 0
    for _, y in data:
        if seen:
            correct += seen.most_common(1)[0][0] == y
        seen[y] += 1
    return correct / len(data)


def test_classifier_leaf_beats_a_constant_predictor(insects):
    """The tree must do better than always predicting the majority class.

    It scored 0.1876 against a 0.1872 baseline while naive Bayes leaves were
    switched off.
    """
    model = HoeffdingTreeClassifier()
    correct = 0
    for x, y in insects:
        correct += model.predict_one(x) == y
        model.learn_one(x, y)
    accuracy = correct / len(insects)
    assert accuracy > _majority_baseline(insects) + 0.20


def test_classifier_leaf_prediction_modes(insects):
    """"mc" is the degenerate mode; "nba" must be clearly better."""
    scores = {}
    for mode in ("mc", "nba"):
        model = HoeffdingTreeClassifier()
        model._cpp.leaf_prediction = mode      # must be exposed, not silently ignored
        correct = 0
        for x, y in insects:
            correct += model.predict_one(x) == y
            model.learn_one(x, y)
        scores[mode] = correct / len(insects)
    assert scores["nba"] > scores["mc"] + 0.20


def _r2(preds, ys):
    mean = sum(ys) / len(ys)
    ss_res = sum((y - p) ** 2 for p, y in zip(preds, ys))
    ss_tot = sum((y - mean) ** 2 for y in ys)
    return 1 - ss_res / ss_tot if ss_tot else 0.0


def test_regressor_beats_predicting_the_mean(ailerons):
    model = HoeffdingTreeRegressor()
    preds, ys = [], []
    for x, y in ailerons:
        preds.append(model.predict_one(x))
        model.learn_one(x, y)
        ys.append(y)
    assert _r2(preds, ys) > 0.0


def _run_reg(data, repeats=1, mode=None):
    model = HoeffdingTreeRegressor()
    if mode is not None:
        model._cpp.leaf_prediction = mode
    preds, ys = [], []
    for x, y in data:
        preds.append(model.predict_one(x))
        for _ in range(repeats):
            model.learn_one(x, y)
        ys.append(y)
    return preds, ys


@pytest.mark.parametrize("repeats", [6, 15])
def test_regressor_survives_repeated_updates(ailerons, repeats):
    """Bagging frameworks learn the same instance k times (Poisson weights).

    The leaf linear model used to be judged against data it had just fitted,
    then extrapolate wildly — R² collapsed while MAE stayed fine, so it hid
    from single-learner testing. The property that matters is relative:
    repeating an instance must not make the tree markedly worse than seeing it
    once. An absolute R² threshold would measure how hard the dataset is
    instead.
    """
    base_preds, ys = _run_reg(ailerons, repeats=1)
    preds, _ = _run_reg(ailerons, repeats=repeats)

    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    # No prediction may sit more than one full target range outside the data.
    assert all(lo - span <= p <= hi + span for p in preds)
    assert _r2(preds, ys) > _r2(base_preds, ys) - 0.15


def test_regressor_leaf_prediction_modes(ailerons):
    """Every mode must stay finite and bounded, including pure "linear"."""
    for mode in ("adaptive", "mean", "linear"):
        preds, ys = _run_reg(ailerons, mode=mode)
        lo, hi = min(ys), max(ys)
        span = (hi - lo) or 1.0
        assert all(lo - span <= p <= hi + span for p in preds), mode
