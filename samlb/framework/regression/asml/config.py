"""
ASML Regression — search space configuration.
Uses the shared C++ algorithm pool defined in shared_config.
"""
from samlb.framework.base import SelectKBest

from samlb.framework.regression.shared_config import (
    SHARED_HYPERPARAMETERS,
    SHARED_MODEL_POOL,
    SHARED_PREPROCESSORS,
)

# Expose under the names ASML code references
model_options              = SHARED_MODEL_POOL
preprocessor_options        = SHARED_PREPROCESSORS
feature_selection_options   = [SelectKBest()]
hyperparameters_options     = SHARED_HYPERPARAMETERS

# ── default config dict ───────────────────────────────────────────────────────

default_config_dict = {
    "models":          model_options,
    "preprocessors":   preprocessor_options,
    "features":        feature_selection_options,
    "hyperparameters": hyperparameters_options,
}
