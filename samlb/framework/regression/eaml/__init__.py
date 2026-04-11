"""
samlb.framework.regression.eaml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
EvoAutoML — Evolutionary Bagging regressor.

Configurable via param_grid:
    from samlb.framework.regression.eaml import EvolutionaryBaggingRegressor
    from samlb.framework.regression.eaml.config import EAML_REG_PARAM_GRID

    model = EvolutionaryBaggingRegressor(
        param_grid=EAML_REG_PARAM_GRID,
        population_size=10,
        sampling_rate=1000,
        seed=42,
    )
"""
from .model  import EvolutionaryBaggingRegressor
from .config import EAML_REG_PARAM_GRID

__all__ = ["EvolutionaryBaggingRegressor", "EAML_REG_PARAM_GRID"]
