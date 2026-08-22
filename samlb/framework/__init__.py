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

Baselines (task-agnostic)
-------------------------
    from samlb.framework.random_search import RandomSearch

Model pool selection
---------------------
Every classification/regression framework's candidate pool can be swapped
between plain single models and drift-adaptive ensemble baselines
(ARF / SRP / Leveraging Bagging / Hoeffding Adaptive Tree):

    from samlb.framework import get_classification_config
    from samlb.framework.classification.asml import AutoStreamClassifier

    cfg = get_classification_config(pool="ensemble")   # or pool="normal" (default)
    model = AutoStreamClassifier(config_dict=cfg.asml_config_dict(), seed=42)

Same for regression via ``get_regression_config`` (pool of ARF/SRP vs. plain
regressors), and for EvoAutoML/OAML via ``cfg.eaml_param_grid()`` /
``cfg.classifier_instances`` — see
``samlb.framework.classification.shared_config.ClassificationConfig`` and
``samlb.framework.regression.shared_config.RegressionConfig``.

All frameworks expose the same interface:
    model.predict_one(x: dict) -> label / float
    model.learn_one(x: dict, y) -> None
    model.reset() -> None
"""
from .base._framework import BaseStreamFramework
from .random_search import RandomSearch
from .classification.asml.model import AutoStreamClassifier
from .classification.autoclass.model import AutoClass
from .classification.eaml.model import EvolutionaryBaggingClassifier
from .classification.oaml.model import OAMLClassifier
from .classification.shared_config import get_classification_config
from .regression.asml.model import AutoStreamRegressor
from .regression.chacha.model import ChaChaRegressor
from .regression.eaml.model import EvolutionaryBaggingRegressor
from .regression.shared_config import get_regression_config

__all__ = [
    "BaseStreamFramework",
    # baselines
    "RandomSearch",
    # classification
    "AutoStreamClassifier",
    "AutoClass",
    "EvolutionaryBaggingClassifier",
    "OAMLClassifier",
    "get_classification_config",
    # regression
    "AutoStreamRegressor",
    "ChaChaRegressor",
    "EvolutionaryBaggingRegressor",
    "get_regression_config",
]
