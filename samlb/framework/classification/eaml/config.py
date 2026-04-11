"""
EvoAutoML Classification — search space configuration.
Uses the shared C++ algorithm pool defined in shared_config.
"""
from samlb.framework.classification.shared_config import (
    SHARED_CLASSIFIER_INSTANCES,
    SHARED_PREPROCESSORS,
)

EAML_CLF_PARAM_GRID: dict = {
    "Scaler":     SHARED_PREPROCESSORS,
    "Classifier": SHARED_CLASSIFIER_INSTANCES,
}
