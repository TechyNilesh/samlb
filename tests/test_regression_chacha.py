import run_regression as rr

from samlb.framework.regression.chacha import model as chacha_model


class FakeAutoVW:
    AUTOMATIC = object()

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.learn_calls = []
        self.predict_calls = []

    def predict(self, example):
        self.predict_calls.append(example)
        return 1.25

    def learn(self, example):
        self.learn_calls.append(example)


def test_chacha_predict_and_learn(monkeypatch):
    monkeypatch.setattr(
        chacha_model,
        "_load_backend",
        lambda: (FakeAutoVW, lambda lower, upper: ("loguniform", lower, upper)),
    )

    model = chacha_model.ChaChaRegressor(seed=7)
    x = {"b": 2.0, "a": 1.0}

    pred = model.predict_one(x)
    model.learn_one(x, 3.5)

    assert pred == 1.25
    assert model._model.predict_calls == ["|f 0:1.0 1:2.0"]
    assert model._model.learn_calls == ["3.5 |f 0:1.0 1:2.0"]


def test_build_models_adds_chacha_when_available(monkeypatch):
    class FakeChaCha:
        def __init__(self, seed):
            self.seed = seed

        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(rr, "ChaChaRegressor", FakeChaCha)

    models = rr._build_models(seed=11)

    assert "ChaCha" in models
    assert models["ChaCha"].seed == 11


def test_build_models_skips_chacha_when_unavailable(monkeypatch):
    class FakeChaCha:
        def __init__(self, seed):
            self.seed = seed

        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(rr, "ChaChaRegressor", FakeChaCha)

    models = rr._build_models(seed=11)

    assert "ChaCha" not in models
