"""
samlb.framework.base._framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for all SAMLB AutoML streaming frameworks.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStreamFramework(ABC):
    """
    Abstract base for streaming AutoML frameworks in SAMLB.

    Every framework (ASML, AutoClass, OAML, EvoAutoML …) must implement
    the three core methods below.  The remaining attributes are conventions
    used by :class:`samlb.benchmark.runner.BenchmarkRunner`.

    Parameters
    ----------
    exploration_window : int
        Number of instances per exploration/adaptation cycle.
    budget : int
        Total number of pipelines / configurations to explore.
    seed : int or None
        Random seed for reproducibility.
    """

    exploration_window: int = 1000
    budget: int = 10
    seed: int | None = 42

    # ── required interface ────────────────────────────────────────────────────

    @abstractmethod
    def predict_one(self, x: Dict[str, float]) -> Any:
        """Return a prediction for a single instance *before* learning."""

    @abstractmethod
    def learn_one(self, x: Dict[str, float], y: Any) -> None:
        """Update the model with a single labelled instance."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the framework to its initial (untrained) state."""

    # ── optional convenience ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        )
        return f"{type(self).__name__}({params})"
