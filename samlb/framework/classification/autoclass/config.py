"""
AutoClass Classification — search space configuration.
Uses the shared C++ algorithm pool defined in shared_config.
"""
from samlb.framework.classification.shared_config import (
    SHARED_HYPERPARAMETERS,
    SHARED_MODEL_POOL,
)

model_options          = SHARED_MODEL_POOL
hyperparameters_options = SHARED_HYPERPARAMETERS

default_config_dict = {
    "algorithms":      model_options,
    "hyperparameters": hyperparameters_options,
}
