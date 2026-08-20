# Quick Start

## Your first benchmark

```python
from samlb.benchmark import BenchmarkSuite
from samlb.framework.classification.asml import AutoStreamClassifier
from samlb.framework.classification.eaml import EvolutionaryBaggingClassifier

suite = BenchmarkSuite(
    models={
        "ASML":      AutoStreamClassifier(seed=42),
        "EvoAutoML": EvolutionaryBaggingClassifier(seed=42),
    },
    datasets=["electricity"],
    task="classification",
    n_runs=1,
    max_samples=10_000,      # drop this for the full stream
)
suite.run()
suite.print_table()
```

`print_table()` writes a fixed-width table to stdout — one row per
(framework × dataset), with `accuracy`, `f1`, `precision`, `recall` and wall
clock for classification, `r2`, `mae`, `rmse` for regression. With `n_runs > 1`
it adds Run and Seed columns and a mean row per group.

Start with `max_samples` set. A full 15-dataset × 10-run sweep is hours of
compute; a 10k-instance smoke run tells you whether the wiring is right in
seconds.

## Regression

Same call, different task. Regression targets are z-score normalised online
before they reach the model, and predictions are mapped back before scoring —
so a learner never sees a raw target range that would make SGD diverge.

```python
from samlb.framework.regression.asml import AutoStreamRegressor
from samlb.framework.regression.eaml import EvolutionaryBaggingRegressor

suite = BenchmarkSuite(
    models={
        "ASML":      AutoStreamRegressor(seed=42),
        "EvoAutoML": EvolutionaryBaggingRegressor(seed=42),
    },
    datasets=["bike", "california_housing"],
    task="regression",
    normalize=True,          # min-max scale features while streaming
    n_runs=1,
)
suite.run()
suite.print_table()
```

## Without the suite

Any model can be driven directly against a stream — useful for debugging a
framework you are writing.

```python
from samlb.datasets import stream
from samlb.framework.base import HoeffdingTreeClassifier, StandardScaler

model = StandardScaler() | HoeffdingTreeClassifier()

hits = n = 0
for x, y in stream("electricity", task="classification", max_samples=5000):
    pred = model.predict_one(x)      # predict BEFORE learning
    if pred is not None:
        hits += pred == y
        n += 1
    model.learn_one(x, y)

print(f"prequential accuracy: {hits / n:.4f}")
```

That predict-then-learn order is the whole point: every prediction is made on
an instance the model has not seen, so no test set is needed.

## The model contract

A SAMLB model is anything with three methods:

```python
model.predict_one(x)   # x is {feature_name: float}; returns a label, a float, or None
model.learn_one(x, y)  # update with one labelled instance
model.reset()          # back to untrained state
```

`reset()` is called before every run, which is what keeps state from leaking
between datasets and seeds. Returning `None` from `predict_one` is legal and
means "not ready yet" — the evaluator counts the instance but does not score
it, which is how warm-up phases are handled.

Frameworks conventionally subclass `BaseStreamFramework`, but the benchmark
only checks for the three methods.

## Command line

The example scripts are the CLI:

```bash
# full classification benchmark (all frameworks × all datasets × 10 runs)
python examples/run_benchmark.py

# a quick subset
python examples/run_benchmark.py --n_runs 5 --max_samples 50000 \
    --datasets electricity covertype

# spread the seeds across cores
python examples/run_benchmark.py --n_runs 100 --parallel --cpu_utilization 0.8

# regression
python examples/run_regression.py --n_runs 5 --datasets bike california_housing

# River / CapyMOA learners against SAMLB's own ARF
python examples/run_external_baselines.py --task classification --n_runs 10
```

Every script writes JSON and CSV under `--output_dir` (default `results/`).

## Saving results

```python
suite.to_csv("results/my_run.csv")     # one flat row per run
suite.to_json("results")               # per-run JSON + aggregate.json
```

See [[Benchmark API]] for the layout of both, and for working with
`RunResult` objects directly.

## Where to go next

- [[Frameworks]] — what ships, and how to configure each method
- [[Base Algorithms]] — the C++ learners and fused pipelines
- [[External Algorithms]] — benchmarking River and CapyMOA learners
- [[Datasets]] — the 30 bundled streams
