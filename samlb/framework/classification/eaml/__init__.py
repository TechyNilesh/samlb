"""
samlb.framework.classification.eaml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
EvoAutoML — Evolutionary Bagging classifier.

Configurable via param_grid:
    from samlb.framework.classification.eaml import EvolutionaryBaggingClassifier
    from samlb.framework.classification.eaml.config import EAML_CLF_PARAM_GRID

    model = EvolutionaryBaggingClassifier(
        param_grid=EAML_CLF_PARAM_GRID,   # or pass your own
        population_size=10,
        sampling_rate=1000,
        seed=42,
    )
"""
from .model  import EvolutionaryBaggingClassifier
from .config import EAML_CLF_PARAM_GRID

__all__ = ["EvolutionaryBaggingClassifier", "EAML_CLF_PARAM_GRID"]
