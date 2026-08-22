"""
EvoAutoML Regression — search space configuration.
Uses the shared C++ algorithm pool defined in shared_config.

The param_grid has two top-level keys:
  "Scaler"    — list of scaler instances (any may be swapped in on mutation)
  "Regressor" — list of regressor instances at various hyperparameter settings
"""
from samlb.framework.regression.shared_config import (
    SHARED_PREPROCESSORS,
    SHARED_REGRESSOR_INSTANCES,
)

EAML_REG_PARAM_GRID: dict = {
    "Scaler":    SHARED_PREPROCESSORS,
    "Regressor": SHARED_REGRESSOR_INSTANCES,
}
