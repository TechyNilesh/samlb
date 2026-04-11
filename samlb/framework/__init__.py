"""
samlb.framework
~~~~~~~~~~~~~~~
Streaming AutoML frameworks for SAMLB.

Classification
--------------
    from samlb.framework.classification.asml    import AutoStreamClassifier
    from samlb.framework.classification.autoclass import AutoClass
    from samlb.framework.classification.eaml   import EvolutionaryBaggingClassifier
    from samlb.framework.classification.oaml   import OAMLClassifier

Regression
----------
    from samlb.framework.regression.asml import AutoStreamRegressor
    from samlb.framework.regression.chacha import ChaChaRegressor
    from samlb.framework.regression.eaml import EvolutionaryBaggingRegressor

All frameworks expose the same interface:
    model.predict_one(x: dict) -> label / float
    model.learn_one(x: dict, y) -> None
    model.reset() -> None
"""
from .base._framework import BaseStreamFramework
from .classification.asml.model import AutoStreamClassifier
from .classification.autoclass.model import AutoClass
from .classification.eaml.model import EvolutionaryBaggingClassifier
from .classification.oaml.model import OAMLClassifier
from .regression.asml.model import AutoStreamRegressor
from .regression.chacha.model import ChaChaRegressor
from .regression.eaml.model import EvolutionaryBaggingRegressor

__all__ = [
    "BaseStreamFramework",
    # classification
    "AutoStreamClassifier",
    "AutoClass",
    "EvolutionaryBaggingClassifier",
    "OAMLClassifier",
    # regression
    "AutoStreamRegressor",
    "ChaChaRegressor",
    "EvolutionaryBaggingRegressor",
]
