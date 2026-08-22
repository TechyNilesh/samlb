"""samlb.framework.regression — streaming AutoML frameworks for regression."""
from .asml.model import AutoStreamRegressor
from .chacha.model import ChaChaRegressor
from .eaml.model import EvolutionaryBaggingRegressor
from .shared_config import get_regression_config

__all__ = [
    "AutoStreamRegressor",
    "ChaChaRegressor",
    "EvolutionaryBaggingRegressor",
    "get_regression_config",
]
