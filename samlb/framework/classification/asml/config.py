"""
ASML Classification — search space configuration.
Uses the shared C++ algorithm pool defined in shared_config.
"""
from river import feature_selection, stats

from samlb.framework.classification.shared_config import (
    SHARED_HYPERPARAMETERS,
    SHARED_MODEL_POOL,
    SHARED_PREPROCESSORS,
)
from .helper import range_gen

# Expose under the names ASML code references
model_options        = SHARED_MODEL_POOL
preprocessor_options = SHARED_PREPROCESSORS

feature_selection_options = [
    feature_selection.VarianceThreshold(threshold=0.01),
    feature_selection.SelectKBest(similarity=stats.PearsonCorr()),
]

hyperparameters_options = {
    **SHARED_HYPERPARAMETERS,
    # Preprocessors (no tunable params beyond selection)
    "MinMaxScaler":   {},
    "StandardScaler": {"with_std": [True, False]},
    # Feature selectors — expanded range for high-dimensional data
    "VarianceThreshold": {
        "threshold":   range_gen(0.0, 1.0, step=0.1, float_n=True),
        "min_samples": range_gen(1, 10, step=1),
    },
    "SelectKBest": {
        "k": range_gen(5, 100, step=5),
    },
}

default_config_dict = {
    "models":          model_options,
    "preprocessors":   preprocessor_options,
    "features":        feature_selection_options,
    "hyperparameters": hyperparameters_options,
}
