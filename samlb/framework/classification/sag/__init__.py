"""
samlb.framework.classification.sag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
StreamingAutoGluon — online stacking of k-fold cross-validated stream learners.

    from samlb.framework.classification.sag import StreamingAutoGluon

    model = StreamingAutoGluon(
        n_folds=3,           # k fold learners per type, plus one stacked learner
        metric="accuracy",   # or "f1" — weights the stacked learners' votes
        window_size=1000,
        seed=42,
    )

Base learners and preprocessors both come from the shared pool. Pass
``learners=[...]`` / ``scalers=[...]`` to override, or ``scalers=None`` to feed
raw features as the Java reference does.
"""
from .model  import StreamingAutoGluon
from .config import SAG_BASE_LEARNERS, SAG_BASE_LEARNERS_SMALL, SAG_SCALERS

__all__ = ["StreamingAutoGluon", "SAG_BASE_LEARNERS",
           "SAG_BASE_LEARNERS_SMALL", "SAG_SCALERS"]
