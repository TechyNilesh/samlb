"""
Tests for the River / CapyMOA adapters.

Both backends are optional, so each half skips when its library is absent.
Everything runs on a synthetic stream — no dataset files needed.
"""
import random

import pytest

from samlb.framework.adapters import (
    CapyMOAClassifier,
    CapyMOARegressor,
    RiverClassifier,
    RiverRegressor,
)


def _stream(n=600, seed=0):
    rng = random.Random(seed)
    rows = [{"a": rng.random(), "b": rng.random()} for _ in range(n)]
    cls = [(x, int(x["a"] + 0.3 * x["b"] > 0.65)) for x in rows]
    reg = [(x, 2 * x["a"] + x["b"]) for x in rows]
    return cls, reg


def _prequential(model, data, regression=False):
    """Returns (score, n_scored) — accuracy, or MSE when regression."""
    total, scored = 0.0, 0
    for x, y in data:
        pred = model.predict_one(x)
        if pred is not None:
            total += (pred - y) ** 2 if regression else float(pred == y)
            scored += 1
        model.learn_one(x, y)
    return total / max(scored, 1), scored


@pytest.fixture(scope="module")
def data():
    return _stream()


# ── River ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not RiverClassifier.is_available(), reason="river not installed")
class TestRiverAdapter:
    def test_classifier_learns(self, data):
        from river import tree
        model = RiverClassifier(tree.HoeffdingTreeClassifier(), name="river-HT")
        acc, scored = _prequential(model, data[0])
        assert acc > 0.8
        assert scored > 0
        assert str(model) == "river-HT"

    def test_regressor_learns(self, data):
        from river import linear_model, preprocessing
        model = RiverRegressor(
            preprocessing.StandardScaler() | linear_model.LinearRegression())
        mse, _ = _prequential(model, data[1], regression=True)
        assert mse < 1.0

    def test_reset_restores_untrained_state(self, data):
        from river import tree
        model = RiverClassifier(tree.HoeffdingTreeClassifier())
        first, _ = _prequential(model, data[0])
        model.reset()
        second, _ = _prequential(model, data[0])
        assert first == second

    def test_prototype_is_never_trained(self, data):
        from river import tree
        prototype = tree.HoeffdingTreeClassifier()
        _prequential(RiverClassifier(prototype), data[0])
        assert prototype.predict_one(data[0][0][0]) is None

    def test_accepts_a_factory(self, data):
        from river import forest
        model = RiverClassifier(lambda: forest.ARFClassifier(seed=1))
        acc, _ = _prequential(model, data[0])
        assert acc > 0.8

    def test_task_mismatch_is_rejected(self):
        from river import tree
        with pytest.raises(TypeError):
            RiverRegressor(tree.HoeffdingTreeClassifier())

    def test_rejects_non_estimator(self):
        with pytest.raises(TypeError):
            RiverClassifier(object())

    def test_is_available(self):
        assert RiverClassifier.is_available() is True


# ── CapyMOA ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not CapyMOAClassifier.is_available(), reason="capymoa not installed")
class TestCapyMOAAdapter:
    def test_classifier_learns(self, data):
        model = CapyMOAClassifier("HoeffdingTree", grace_period=50)
        acc, scored = _prequential(model, data[0])
        assert acc > 0.8
        assert scored > 0

    def test_regressor_learns(self, data):
        model = CapyMOARegressor("FIMTDD")
        mse, _ = _prequential(model, data[1], regression=True)
        assert mse < 1.0

    def test_learner_kwargs_reach_the_backend(self, data):
        model = CapyMOAClassifier(
            "AdaptiveRandomForestClassifier", ensemble_size=3, seed=1)
        _prequential(model, data[0][:100])
        assert model.capymoa_model is not None
        acc, _ = _prequential(model, data[0])
        assert acc > 0.8

    def test_accepts_a_class(self, data):
        from capymoa.classifier import NaiveBayes
        acc, _ = _prequential(CapyMOAClassifier(NaiveBayes), data[0])
        assert acc > 0.7

    def test_reset_rebuilds_lazily(self, data):
        model = CapyMOAClassifier("HoeffdingTree")
        first, _ = _prequential(model, data[0])
        model.reset()
        assert model.capymoa_model is None
        second, _ = _prequential(model, data[0])
        assert first == second

    def test_non_integer_labels_round_trip(self, data):
        labelled = [(x, "yes" if y else "no") for x, y in data[0]]
        model = CapyMOAClassifier("NaiveBayes", classes=["no", "yes"])
        acc, _ = _prequential(model, labelled)
        assert acc > 0.8

    def test_labels_are_discovered_when_classes_unset(self, data):
        labelled = [(x, "yes" if y else "no") for x, y in data[0]]
        acc, _ = _prequential(CapyMOAClassifier("NaiveBayes"), labelled)
        assert acc > 0.8

    def test_undeclared_label_is_rejected(self):
        model = CapyMOAClassifier("NaiveBayes", classes=["no", "maybe"])
        with pytest.raises(ValueError, match="not in the declared classes"):
            model.learn_one({"a": 1.0, "b": 2.0}, "yes")

    def test_more_labels_than_slots_is_rejected(self):
        model = CapyMOAClassifier("NaiveBayes", max_classes=2)
        x = {"a": 1.0, "b": 2.0}
        model.learn_one(x, "a")
        model.learn_one(x, "b")
        with pytest.raises(ValueError, match="max_classes"):
            model.learn_one(x, "c")

    def test_unknown_learner_name_lists_alternatives(self):
        with pytest.raises(ValueError, match="AdaptiveRandomForestClassifier"):
            CapyMOAClassifier("NoSuchLearner")

    def test_built_learner_is_rejected(self):
        from capymoa.classifier import NaiveBayes
        from capymoa.stream import Schema
        schema = Schema.from_custom(
            features=["a", "b", "c"], target="c", categories={"c": ["0", "1"]})
        with pytest.raises(TypeError, match="need their schema"):
            CapyMOAClassifier(NaiveBayes(schema=schema))

    def test_invalid_max_classes(self):
        with pytest.raises(ValueError, match="max_classes"):
            CapyMOAClassifier("NaiveBayes", max_classes=1)

    def test_is_available(self):
        assert CapyMOAClassifier.is_available() is True
