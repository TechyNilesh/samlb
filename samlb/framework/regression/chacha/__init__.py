"""
samlb.framework.regression.chacha
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ChaCha — FLAML AutoVW online regressor.

Configurable via AutoVW search_space/init_config:
    from samlb.framework.regression.chacha import ChaChaRegressor

    model = ChaChaRegressor(
        max_live_model_num=5,
        seed=42,
    )
"""
from .model import ChaChaRegressor

__all__ = ["ChaChaRegressor"]
