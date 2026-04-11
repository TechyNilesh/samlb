"""
samlb — Streaming AutoML Benchmark
====================================
Fast C++ base algorithms + framework-agnostic benchmark runner.

Architecture
------------
  samlb/benchmark/    — framework-agnostic BenchmarkSuite
  samlb/evaluation/   — framework-agnostic PrequentialEvaluator
  samlb/datasets/     — dataset loader (stream, list_datasets)
  samlb/framework/    — independent AutoML frameworks (plug any in)
    classification/
      asml/           — AutoStreamClassifier    (config_dict=)
      autoclass/      — AutoClass               (config_dict=)
      eaml/           — EvolutionaryBaggingClassifier (param_grid=)
      oaml/           — OAMLClassifier          (scalers=, classifiers=)
    regression/
      asml/           — AutoStreamRegressor     (config_dict=)
      chacha/         — ChaChaRegressor         (FLAML AutoVW)
      eaml/           — EvolutionaryBaggingRegressor (param_grid=)

Quick start
-----------
    from samlb.benchmark import BenchmarkSuite
    from samlb.framework.classification.asml  import AutoStreamClassifier
    from samlb.framework.classification.eaml  import EvolutionaryBaggingClassifier

    suite = BenchmarkSuite(
        models={
            "ASML":      AutoStreamClassifier(seed=42),
            "EvoAutoML": EvolutionaryBaggingClassifier(seed=42),
        },
        datasets=["electricity"],
        task="classification",
    )
    suite.run()
    suite.print_table()
"""
__version__ = "0.1.0"
