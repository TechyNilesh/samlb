"""
samlb.framework.adapters
~~~~~~~~~~~~~~~~~~~~~~~~
Adapters that let external streaming libraries run inside a SAMLB benchmark.

Both backends are optional dependencies — nothing here is imported until you
construct an adapter, and every adapter exposes ``is_available()`` so a suite
can skip it cleanly when the backend is missing.

    from samlb.benchmark import BenchmarkSuite
    from samlb.framework.adapters import CapyMOAClassifier, RiverClassifier
    from river import forest, preprocessing

    models = {"SAMLB-ARF": ARFClassifier(seed=42)}
    if RiverClassifier.is_available():
        models["river-ARF"] = RiverClassifier(
            preprocessing.StandardScaler() | forest.ARFClassifier(seed=42))
    if CapyMOAClassifier.is_available():
        models["moa-ARF"] = CapyMOAClassifier(
            "AdaptiveRandomForestClassifier", ensemble_size=10, seed=42)

    BenchmarkSuite(models=models, datasets=["electricity"],
                   task="classification").run()

Install the backends with ``pip install river`` and ``pip install capymoa``
(CapyMOA also needs a JVM), or with the ``river`` / ``capymoa`` extras.
"""
from ._capymoa import CapyMOAClassifier, CapyMOARegressor
from ._river import RiverClassifier, RiverRegressor

__all__ = [
    "CapyMOAClassifier",
    "CapyMOARegressor",
    "RiverClassifier",
    "RiverRegressor",
]
