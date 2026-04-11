"""
samlb.framework.classification.asml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ASML — Adaptive Streaming Machine Learning classifier.

Configurable via config_dict:
    from samlb.framework.classification.asml import AutoStreamClassifier
    from samlb.framework.classification.asml.config import default_config_dict

    model = AutoStreamClassifier(
        config_dict=default_config_dict,   # or pass your own
        exploration_window=1000,
        budget=10,
        seed=42,
    )
"""
from .model  import AutoStreamClassifier
from .config import default_config_dict

__all__ = ["AutoStreamClassifier", "default_config_dict"]
