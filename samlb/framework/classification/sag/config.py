"""
StreamingAutoGluon — base learner configuration.

Everything here comes from :mod:`samlb.framework.classification.shared_config`,
the single pool every classification framework in the benchmark draws from, so
the only difference between results is the AutoML strategy and not the set of
algorithms available.

The Java reference stacks three stream *ensembles* (AdaptiveRandomForest, ARTE
and SORE). Those have no counterpart in this codebase, so the stacking
structure is kept and the base learner types become the shared pool.
"""
from samlb.framework.base import (
    HoeffdingTreeClassifier,
    LogisticRegression,
    NaiveBayes,
)
from samlb.framework.classification.shared_config import (
    SHARED_CLASSIFIER_INSTANCES,
    SHARED_MODEL_POOL,
    SHARED_PREPROCESSORS,
)

# One base learner type per algorithm in the shared pool — the same list ASML
# and AutoClass search over.
SAG_BASE_LEARNERS: list = list(SHARED_MODEL_POOL)

# The shared preprocessor pool, assigned round-robin across the base types (one
# scaler per type, not a search). Using the whole pool rather than a single
# fixed scaler adds diversity to the meta features, which is what stacking
# feeds on.
SAG_SCALERS: list = list(SHARED_PREPROCESSORS)

# A cheaper, deliberately diverse subset: a tree, a probabilistic model and a
# linear model. Pass as learners=SAG_BASE_LEARNERS_SMALL. Cost scales with
# len(learners) * (n_folds + 1).
SAG_BASE_LEARNERS_SMALL: list = [
    HoeffdingTreeClassifier(),
    NaiveBayes(),
    LogisticRegression(),
]

__all__ = [
    "SAG_BASE_LEARNERS",
    "SAG_BASE_LEARNERS_SMALL",
    "SAG_SCALERS",
    "SHARED_CLASSIFIER_INSTANCES",
    "SHARED_MODEL_POOL",
    "SHARED_PREPROCESSORS",
]
