"""
samlb.framework.classification.oaml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OAML — Online AutoML classifier with GAMA-style drift-triggered pipeline search.

Configurable via scalers / classifiers / hyperparameters:
    from samlb.framework.classification.oaml import OAMLClassifier
    from samlb.framework.classification.oaml.config import (
        OAML_SCALERS, OAML_CLASSIFIERS, OAML_HYPERPARAMETERS,
    )

    model = OAMLClassifier(
        scalers=OAML_SCALERS,           # or pass your own lists
        classifiers=OAML_CLASSIFIERS,
        hyperparameters=OAML_HYPERPARAMETERS,
        initial_batch_size=200,
        population_size=10,
        generations=3,
        seed=42,
    )
"""
from .model  import OAMLClassifier
from .config import OAML_SCALERS, OAML_CLASSIFIERS, OAML_HYPERPARAMETERS

__all__ = [
    "OAMLClassifier",
    "OAML_SCALERS",
    "OAML_CLASSIFIERS",
    "OAML_HYPERPARAMETERS",
]
