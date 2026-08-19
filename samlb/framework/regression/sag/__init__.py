"""
samlb.framework.regression.sag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
StreamingAutoGluon Regression — online stacking of k-fold cross-validated
stream regressors.

    from samlb.framework.regression.sag import StreamingAutoGluonRegressor

    model = StreamingAutoGluonRegressor(
        n_folds=3,        # k fold learners per type, plus one stacked learner
        metric="mae",     # or "rmse" — weights the stacked regressors
        window_size=1000,
        clip=True,
        seed=42,
    )

Base learners and preprocessors both come from the shared C++ regression pool.
Pass ``learners=[...]`` / ``scalers=[...]`` to override, or ``scalers=None``
for raw features.
"""
from .model  import StreamingAutoGluonRegressor
from .config import (
    SAG_REG_BASE_LEARNERS,
    SAG_REG_BASE_LEARNERS_SMALL,
    SAG_REG_SCALERS,
)

__all__ = [
    "StreamingAutoGluonRegressor",
    "SAG_REG_BASE_LEARNERS",
    "SAG_REG_BASE_LEARNERS_SMALL",
    "SAG_REG_SCALERS",
]
