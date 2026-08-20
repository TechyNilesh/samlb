# SAMLB — Streaming AutoML Benchmark

SAMLB benchmarks streaming AutoML methods on evolving data streams, on a level
playing field: every framework searches the same pool of base learners and
preprocessors, every method is scored by the same prequential evaluator, and
every per-instance component is implemented in C++ so the comparison measures
the search strategy rather than the host language.

```python
from samlb.benchmark import BenchmarkSuite
from samlb.framework.classification.asml import AutoStreamClassifier
from samlb.framework.classification.eaml import EvolutionaryBaggingClassifier

suite = BenchmarkSuite(
    models={
        "ASML":      AutoStreamClassifier(seed=42),
        "EvoAutoML": EvolutionaryBaggingClassifier(seed=42),
    },
    datasets=["electricity", "covertype"],
    task="classification",
    n_runs=10,
)
suite.run()
suite.print_table()
```

## Pages

| Page | What it covers |
|------|----------------|
| [[Installation]] | Install from PyPI or source, optional backends, build troubleshooting |
| [[Quick Start]] | First benchmark, the model contract, reading the output |
| [[Benchmark API]] | `BenchmarkSuite`, `PrequentialEvaluator`, `RunResult`, parallel runs |
| [[Datasets]] | The 30 bundled streams, `stream()` / `load()`, adding your own |
| [[Frameworks]] | The AutoML methods that ship with SAMLB, and their configuration |
| [[Base Algorithms]] | The C++ learners, scalers, selectors, metrics, drift detectors, fused pipelines |
| [[External Algorithms]] | Benchmarking River and CapyMOA/MOA learners through the adapters |
| [[Extending SAMLB]] | Writing your own framework, adapter, dataset or C++ learner |
| [[FAQ]] | Common questions and failure modes |

## The one contract that matters

Anything with these three methods is a SAMLB model — built-in framework,
adapter, or your own class:

```python
model.predict_one(x)   # x is {feature_name: float}; predict BEFORE learning
model.learn_one(x, y)  # update with one labelled instance
model.reset()          # back to untrained; called before every run
```

Everything else in SAMLB — the suite, the evaluator, the result objects, the
CSV/JSON writers — is built on top of exactly that.

## Project links

- Repository: <https://github.com/TechyNilesh/samlb>
- Issues: <https://github.com/TechyNilesh/samlb/issues>
- Contributing: [CONTRIBUTING.md](https://github.com/TechyNilesh/samlb/blob/main/CONTRIBUTING.md)
