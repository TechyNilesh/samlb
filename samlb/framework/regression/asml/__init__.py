"""
samlb.framework.regression.asml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ASML — Adaptive Streaming Machine Learning regressor.

Configurable via config_dict:
    from samlb.framework.regression.asml import AutoStreamRegressor
    from samlb.framework.regression.asml.config import default_config_dict

    model = AutoStreamRegressor(
        config_dict=default_config_dict,
        exploration_window=1000,
        budget=10,
        seed=42,
    )
"""
from .model  import AutoStreamRegressor
from .config import default_config_dict

__all__ = ["AutoStreamRegressor", "default_config_dict"]
