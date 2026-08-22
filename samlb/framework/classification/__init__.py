"""samlb.framework.classification — streaming AutoML frameworks for classification."""
from .asml.model import AutoStreamClassifier
from .autoclass.model import AutoClass
from .eaml.model import EvolutionaryBaggingClassifier
from .oaml.model import OAMLClassifier
from .shared_config import get_classification_config

__all__ = [
    "AutoStreamClassifier",
    "AutoClass",
    "EvolutionaryBaggingClassifier",
    "OAMLClassifier",
    "get_classification_config",
]
