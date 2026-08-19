"""samlb.framework.classification — streaming AutoML frameworks for classification."""
from .asml.model import AutoStreamClassifier
from .autoclass.model import AutoClass
from .eaml.model import EvolutionaryBaggingClassifier
from .oaml.model import OAMLClassifier
from .sag.model import StreamingAutoGluon

__all__ = [
    "AutoStreamClassifier",
    "AutoClass",
    "EvolutionaryBaggingClassifier",
    "OAMLClassifier",
    "StreamingAutoGluon",
]
