"""
Tests for the SRPClassifier wrapper (Streaming Random Patches, Gomes et al. 2019).

Synthetic stream only — no dataset files needed.
"""
import random

import pytest

from samlb.framework.base import SRPClassifier, StandardScaler


def _stream(n=1500, seed=0):
    rng = random.Random(seed)
    rows = [{"a": rng.random(), "b": rng.random(), "c": rng.random()} for _ in range(n)]
    return [(x, int(x["a"] + 0.3 * x["b"] > 0.65)) for x in rows]


def _prequential(model, data):
    hits = scored = 0
    for x, y in data:
        pred = model.predict_one(x)
        if pred is not None:
            hits += pred == y
            scored += 1
        model.learn_one(x, y)
    return hits / max(scored, 1)


@pytest.fixture(scope="module")
def data():
    return _stream()


class TestSRPClassifier:
    def test_learns(self, data):
        assert _prequential(SRPClassifier(n_models=5, seed=1), data) > 0.8

    @pytest.mark.parametrize("method", ["patches", "subspaces", "resampling"])
    def test_every_training_method_learns(self, method, data):
        model = SRPClassifier(n_models=5, seed=1, training_method=method)
        assert _prequential(model, data) > 0.75

    def test_seed_is_reproducible(self, data):
        a = _prequential(SRPClassifier(n_models=5, seed=1), data)
        b = _prequential(SRPClassifier(n_models=5, seed=1), data)
        assert a == b

    def test_reset_clears_state(self, data):
        model = SRPClassifier(n_models=5, seed=1)
        first = _prequential(model, data)
        model.reset()
        assert _prequential(model, data) == first

    def test_predict_proba_is_a_distribution(self, data):
        model = SRPClassifier(n_models=5, seed=1)
        for x, y in data[:300]:
            model.learn_one(x, y)
        proba = model.predict_proba_one(data[0][0])
        assert proba
        assert all(0.0 <= p <= 1.0 for p in proba.values())
        assert abs(sum(proba.values()) - 1.0) < 1e-6

    def test_clone_is_untrained_and_keeps_params(self, data):
        model = SRPClassifier(n_models=3, seed=7, training_method="subspaces")
        _prequential(model, data)
        fresh = model.clone()
        assert fresh._get_params() == model._get_params()
        assert _prequential(fresh, data) == _prequential(SRPClassifier(
            n_models=3, seed=7, training_method="subspaces"), data)

    def test_fuses_into_a_pipeline(self, data):
        pipeline = StandardScaler() | SRPClassifier(n_models=5, seed=1)
        assert list(pipeline.steps) == ["StandardScaler", "SRPClassifier"]
        assert _prequential(pipeline, data) > 0.75

    def test_invalid_training_method(self):
        with pytest.raises(ValueError, match="training_method"):
            SRPClassifier(training_method="nope")
