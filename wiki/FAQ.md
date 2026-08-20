# FAQ

## Why is there no train/test split?

Streams are evaluated **prequentially**: every instance is predicted before it
is learned from, so every prediction is already on unseen data. The cumulative
metric over the whole stream is the score, and the windowed series is the
learning curve. Splitting a stream would destroy the temporal order that the
drift lives in.

## Why did my model score `None` / why is `n_scored` lower than `n_samples`?

`predict_one` may return `None` while a model is warming up. Those instances
are counted but not scored. A handful at the start of a stream is normal; a
large number means the model never became ready — check the warm-up logic.

## Why are my results identical across seeds?

Either the method is deterministic, or `reset()` is not clearing the random
state. `BenchmarkSuite` sets `model.seed` before every run, but a framework
that builds its RNG once in `__init__` and never rebuilds it in `reset()` will
ignore that.

## Why do results differ between two runs of the same seed?

Almost always leftover state: `reset()` did not fully return the model to its
untrained form. The test for it is in [[Extending SAMLB]] — run the same
stream twice with a `reset()` between and compare.

## `AttributeError: module 'samlb._samlb_core' has no attribute ...`

The compiled extension is stale. An editable install does not recompile when
C++ changes; rebuild:

```bash
cmake -S . -B build/local -DCMAKE_BUILD_TYPE=Release
cmake --build build/local -j
cp build/local/_samlb_core.*.so samlb/
```

## Do I have to use `samlb.datasets`?

No. Any iterable of `(x_dict, y)` works with any model — see [[Datasets]].
`BenchmarkSuite`'s tables and JSON output, though, are built on the dataset
registry, so a stream you want in the suite has to be added as an `.npz`.

## Should I set `normalize=True`?

For regression, yes — gradient-based learners diverge on raw target ranges. It
min-max scales *features* using only statistics seen so far, so it leaks no
future information. Regression *targets* are separately z-score normalised by
the evaluator by default, with predictions mapped back before scoring, so
reported errors are always in the original units.

## How long does a full benchmark take?

Hours, for all frameworks × 15 datasets × 10 runs. Develop against
`max_samples=10_000` and a couple of datasets, then run the full sweep once.
`examples/run_benchmark.py --parallel --cpu_utilization 0.8` spreads seeds
across cores.

## A run failed — did I lose the whole benchmark?

No. Exceptions are caught per run; the traceback lands on `RunResult.error`,
the table shows `ERROR` for that row, and aggregates exclude failed runs
instead of counting them as zeros.

## Why is River not a dependency any more?

Everything SAMLB used from River — scalers, feature selection, metrics,
ADWIN/EDDM, ARF — is now native C++, verified bit-identical to River's
implementations before the dependency was dropped. River is still fully usable
as an *optional* backend through the adapters; see [[External Algorithms]].

## Can I benchmark a scikit-learn model?

Not directly — scikit-learn estimators are batch learners with no `learn_one`.
Wrap one in a class that implements the three-method contract (buffering and
refitting periodically, say), or use River's or CapyMOA's streaming
equivalents through the adapters.

## Which metric ranks the table?

Accuracy for classification, R² for regression
(`RunResult.primary_metric()`). All of `accuracy` / `f1` / `precision` /
`recall` and `mae` / `rmse` / `r2` are recorded regardless; F1, precision and
recall are macro-averaged.

## CapyMOA fails to import

CapyMOA starts a JVM at import time. Check `java -version` — a JDK 17+ must be
on the path before `pip install capymoa`. `CapyMOAClassifier.is_available()`
returns `False` rather than raising when the JVM cannot start, so a suite can
skip it.

## Where do the numbers in the paper come from?

`benchmark_results/` holds the recorded sweeps (CSV plus JSON summaries), and
`results/` holds per-run JSON. See [[Benchmark API]] for the layout.
