"""
samlb.framework.classification.autoclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AutoClass — genetic algorithm AutoML classifier.

Configurable via config_dict:
    from samlb.framework.classification.autoclass import AutoClass
    from samlb.framework.classification.autoclass.config import default_config_dict

    model = AutoClass(
        config_dict=default_config_dict,   # or pass your own
        exploration_window=1000,
        population_size=10,
        seed=42,
    )
"""
from .model  import AutoClass
from .config import default_config_dict

__all__ = ["AutoClass", "default_config_dict"]
