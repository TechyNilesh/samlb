"""samlb.framework.regression — streaming AutoML frameworks for regression."""
from .asml.model import AutoStreamRegressor
from .chacha.model import ChaChaRegressor
from .eaml.model import EvolutionaryBaggingRegressor
from .sag.model import StreamingAutoGluonRegressor

__all__ = [
    "AutoStreamRegressor",
    "ChaChaRegressor",
    "EvolutionaryBaggingRegressor",
    "StreamingAutoGluonRegressor",
]
