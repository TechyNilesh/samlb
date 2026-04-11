"""
samlb.framework.classification.oaml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OAML — Online AutoML classifier with drift-triggered pipeline search.

Configurable via scalers / classifiers:
    from samlb.framework.classification.oaml import OAMLClassifier
    from samlb.framework.classification.oaml.config import OAML_SCALERS, OAML_CLASSIFIERS

    model = OAMLClassifier(
        scalers=OAML_SCALERS,           # or pass your own lists
        classifiers=OAML_CLASSIFIERS,
        initial_batch_size=200,
        budget=20,
        seed=42,
    )
"""
from .model  import OAMLClassifier
from .config import OAML_SCALERS, OAML_CLASSIFIERS

__all__ = ["OAMLClassifier", "OAML_SCALERS", "OAML_CLASSIFIERS"]
