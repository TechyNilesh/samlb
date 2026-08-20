# Benchmark API

Three layers, each usable on its own:

| Layer | Class | Job |
|-------|-------|-----|
| Orchestration | `samlb.benchmark.BenchmarkSuite` | frameworks × datasets × seeds, tables, CSV/JSON |
| Evaluation | `samlb.evaluation.PrequentialEvaluator` | one model on one stream, test-then-train |
| Result | `samlb.evaluation.results.RunResult` | everything one run produced |

## BenchmarkSuite

```python
BenchmarkSuite(
    models,                 # {display_name: model}
    datasets,               # [name, ...] or None for every dataset of the task
    task,                   # "classification" | "regression"
    n_runs=1,               # independent runs per (framework × dataset)
    seeds=None,             # explicit seeds; len(seeds) must equal n_runs
    window_size=1000,       # instances per evaluation window
    max_samples=None,       # cap instances per dataset per run
    normalize=False,        # min-max scale features while streaming
    verbose=True,           # progress line per run
)
```

`reset()` is called on every model before every run, and each model's `seed`
attribute is set to the run's seed, so one model instance is safely reused
across the whole sweep.

### Running

```python
results = suite.run()          # list[RunResult], also kept on the suite
suite.print_table()
```

A run that raises does not stop the sweep: the exception is caught, its
traceback is stored on the `RunResult.error` field, and the table shows
`ERROR` for that row.

For a live progress feed — a notebook widget, a queue in a parallel runner —
pass a callback:

```python
def on_event(event):
    # event["event"] is "task_started" or "task_finished"
    # finished events also carry event["result"] (a RunResult)
    print(event["event"], event["framework_name"], event["dataset_name"])

suite.run(progress_callback=on_event)
```

### Output

```python
suite.print_table()
suite.to_csv("results/run.csv")
suite.to_json("results")
```

`to_csv` writes one flat row per run: framework, dataset, seed, `n_samples`,
every cumulative metric, and runtime. Windowed series are not included — they
are in the JSON.

`to_json` writes a directory tree. With `n_runs=1`:

```
results/classification/electricity/ASML.json     # metrics + windowed + runtime
results/classification/summary.json              # flat list of all runs
```

With `n_runs > 1`:

```
results/classification/electricity/ASML/run_00.json
results/classification/electricity/ASML/run_01.json
results/classification/electricity/ASML/aggregate.json   # mean ± std
results/classification/summary.json
```

`aggregate.json` carries `framework`, `dataset`, `task`, `n_runs`,
`n_successful_runs`, `n_failed_runs`, `seeds`, `n_samples`, `metrics` and
`windowed_metrics` as mean ± std. Failed runs are excluded from the averages
rather than silently counted as zeros.

### Merging results from elsewhere

Useful when seeds were run in separate processes (this is how
`examples/run_benchmark.py --parallel` works):

```python
suite.load_results(all_results)    # replace whatever the suite holds
suite.merge_results(more_results)  # append
suite.print_table()
suite.to_json("results")
```

## PrequentialEvaluator

One model, one stream, no orchestration.

```python
from samlb.evaluation.evaluator import PrequentialEvaluator

ev = PrequentialEvaluator(
    task="classification",
    window_size=1000,
    max_samples=None,
    normalize=False,
    normalize_target=None,   # default: True for regression, False for classification
    sample_runtime_every=100,
    verbose=False,
)
result = ev.run(model, "electricity", framework_name="ASML")
print(result.metrics)
print(result.windowed_metrics["accuracy"])    # the learning curve
```

The loop is exactly:

1. predict on the instance,
2. update metrics if the prediction is not `None`,
3. learn from the instance.

`WindowedEvaluator` is the same class with `verbose=True` by default, for
watching a learning curve as it happens.

The evaluator does **not** call `reset()` — `BenchmarkSuite` does that. Reset
the model yourself when driving the evaluator directly.

### Target normalisation

For regression, targets are z-score normalised online with Welford statistics
computed from instances seen so far, and predictions are denormalised before
scoring. So the model trains on standardised `y`, but every reported number is
in the original target space. Turn it off with `normalize_target=False` if a
method handles raw scales itself.

## RunResult

```python
r = results[0]
r.framework_name, r.dataset_name, r.task
r.n_samples                 # instances processed
r.metrics                   # {"accuracy": ..., "f1": ...} or {"r2": ..., "mae": ...}
r.windowed_metrics          # {metric: [per-window values]}
r.total_runtime_s
r.runtime_per_instance_ms   # sampled, every sample_runtime_every-th instance
r.run_id, r.seed
r.error                     # traceback string, or None
r.primary_metric()          # accuracy (cls) or r2 (reg)
r.as_dict()                 # flat dict, CSV-shaped, no windowed data
```

Plotting a learning curve:

```python
import matplotlib.pyplot as plt

for r in suite.results:
    plt.plot(r.windowed_metrics["accuracy"], label=r.framework_name)
plt.xlabel(f"window (of {suite.window_size} instances)")
plt.ylabel("accuracy")
plt.legend()
```

## Metrics

Classification reports `accuracy`, `f1`, `precision`, `recall` — the last
three macro-averaged. Regression reports `mae`, `rmse`, `r2`. All are computed
incrementally in C++ and updated on every scored instance.

Ranking uses accuracy for classification and R² for regression
(`RunResult.primary_metric()`).

The metric classes are usable directly, with River's calling convention:

```python
from samlb.metrics import Accuracy, MacroF1, MAE, RMSE, R2

m = Accuracy()
m.update(y_true, y_pred)
m.get()
```

See [[Base Algorithms]] for the windowed variants and the drift detectors.
