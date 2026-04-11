"""
samlb.benchmark
~~~~~~~~~~~~~~~
Framework-agnostic benchmarking runner for streaming AutoML.

The benchmark knows nothing about any specific framework.
Any object that implements the three-method interface can be plugged in:

    predict_one(x: dict) -> label | float
    learn_one(x: dict, y) -> None
    reset() -> None

Usage
-----
    from samlb.benchmark import BenchmarkSuite

    # Bring your own frameworks — benchmark doesn't care which ones
    from samlb.framework.classification.asml  import AutoStreamClassifier
    from samlb.framework.classification.eaml  import EvolutionaryBaggingClassifier
    from samlb.framework.classification.oaml  import OAMLClassifier
    from samlb.framework.classification.autoclass import AutoClass

    suite = BenchmarkSuite(
        models={
            "ASML":      AutoStreamClassifier(seed=42),
            "EvoAutoML": EvolutionaryBaggingClassifier(seed=42),
            "OAML":      OAMLClassifier(seed=42),
            "AutoClass": AutoClass(seed=42),
        },
        datasets=["electricity", "covtype"],
        task="classification",
    )
    suite.run()
    suite.print_table()
    suite.to_csv("results/cls.csv")
"""
from .suite import BenchmarkSuite

__all__ = ["BenchmarkSuite"]
