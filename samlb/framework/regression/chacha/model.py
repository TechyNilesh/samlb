"""
ChaCha Regression — FLAML AutoVW wrapper for streaming regression.

ChaCha in this repo is implemented via FLAML's online ``AutoVW`` API.
It performs progressive online hyperparameter search for a Vowpal Wabbit
regressor while exposing SAMLB's standard stream framework interface.
"""
from __future__ import annotations

import importlib.util
import warnings
from typing import Any, Dict, Optional

from samlb.framework.base._framework import BaseStreamFramework


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_backend():
    if not _has_module("flaml") or not _has_module("vowpalwabbit"):
        raise ImportError(
            "ChaChaRegressor requires FLAML AutoVW and Vowpal Wabbit. "
            "Install with `pip install \"flaml[vw]\"`."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"flaml\.automl is not available.*",
            category=UserWarning,
        )
        from flaml import AutoVW
        from flaml.tune import loguniform

    return AutoVW, loguniform


class ChaChaRegressor(BaseStreamFramework):
    """Online regression wrapper around FLAML's AutoVW.

    Parameters
    ----------
    max_live_model_num : int
        Maximum number of live AutoVW models updated per iteration.
    search_space : dict | None
        AutoVW search space. Defaults to interactions + learning_rate search.
    init_config : dict | None
        Initial configuration for the AutoVW search.
    metric : str
        Progressive validation loss used by AutoVW.
    seed : int
        Random seed for reproducibility.
    """

    budget: int = 5

    def __init__(
        self,
        max_live_model_num: int = 5,
        search_space: Optional[Dict[str, Any]] = None,
        init_config: Optional[Dict[str, Any]] = None,
        metric: str = "mae_clipped",
        seed: int = 42,
    ):
        AutoVW, loguniform = _load_backend()

        self.max_live_model_num = max_live_model_num
        self.metric = metric
        self.seed = seed
        self.search_space = (
            search_space
            if search_space is not None
            else {
                "interactions": AutoVW.AUTOMATIC,
                "learning_rate": loguniform(lower=2e-10, upper=1.0),
            }
        )
        self.init_config = (
            init_config
            if init_config is not None
            else {"interactions": set(), "learning_rate": 0.5}
        )

        self._AutoVW = AutoVW
        self._model = self._new_model()

    @staticmethod
    def is_available() -> bool:
        return _has_module("flaml") and _has_module("vowpalwabbit")

    def _new_model(self):
        return self._AutoVW(
            max_live_model_num=self.max_live_model_num,
            search_space=self.search_space,
            init_config=self.init_config,
            metric=self.metric,
            random_seed=self.seed,
        )

    @staticmethod
    def _to_vw_example(x: Dict[str, Any], y: Optional[float] = None) -> str:
        parts = []
        for idx, key in enumerate(sorted(x)):
            value = x[key]
            if value is None:
                continue
            try:
                parts.append(f"{idx}:{float(value)}")
            except (TypeError, ValueError):
                token = str(value).replace(" ", "_").replace("|", "_").replace(":", "_")
                parts.append(f"f{idx}={token}")

        feature_part = " ".join(parts)
        if y is None:
            return f"|f {feature_part}".rstrip()
        return f"{float(y)} |f {feature_part}".rstrip()

    def predict_one(self, x: Dict[str, float]) -> float:
        try:
            pred = self._model.predict(self._to_vw_example(x))
        except Exception:
            return 0.0

        if pred is None:
            return 0.0
        try:
            return float(pred)
        except (TypeError, ValueError):
            return 0.0

    def learn_one(self, x: Dict[str, float], y: Any) -> None:
        try:
            self._model.learn(self._to_vw_example(x, float(y)))
        except Exception:
            return

    def reset(self) -> None:
        self._model = self._new_model()
