"""
OAML Classification — pipeline search space configuration.
Uses the shared C++ algorithm pool defined in shared_config.
"""
from samlb.framework.classification.shared_config import (
    SHARED_CLASSIFIER_INSTANCES,
    SHARED_PREPROCESSORS,
)

OAML_SCALERS     = SHARED_PREPROCESSORS
OAML_CLASSIFIERS = SHARED_CLASSIFIER_INSTANCES
