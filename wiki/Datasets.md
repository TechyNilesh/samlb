# Datasets

30 real and synthetic streams ship with SAMLB — 15 classification, 15
regression. They are downloaded from GitHub on first use and cached locally, so
the first run of a dataset needs network access and later runs do not.

```python
from samlb.datasets import list_datasets, load, stream

list_datasets("classification")
list_datasets("regression")
```

## Classification

| Dataset | Notes |
|---------|-------|
| `electricity` | Australian electricity market, the standard drift benchmark |
| `covertype` | Forest cover type, 7 classes |
| `insects` | Optical insect classification, gradual and abrupt drift |
| `adult` | Census income |
| `credit_card` | Fraud detection, heavily imbalanced |
| `nomao` | Place-data deduplication |
| `poker_hand` | Poker hands, many classes |
| `shuttle` | NASA shuttle telemetry, imbalanced |
| `vehicle_sensIT` | Vehicle acoustic/seismic sensors |
| `new_airlines` | Flight delay |
| `movingRBF`, `moving_squares` | Synthetic, continuous drift |
| `synth_RandomRBFDrift`, `synth_agrawal` | Synthetic generators |
| `sea_high_abrupt_drift` | Synthetic, abrupt concept changes |

## Regression

| Dataset | Notes |
|---------|-------|
| `california_housing`, `House8L`, `kings_county` | Housing prices |
| `bike`, `MetroTraffic` | Demand and traffic counts, strong seasonality |
| `ailerons`, `elevators` | Flight control surfaces |
| `superconductivity` | Critical-temperature prediction |
| `diamonds`, `fifa`, `cps88wages` | Tabular real-world regression |
| `wave_energy` | Wave farm power output |
| `fried`, `FriedmanGra`, `hyperA` | Synthetic, including drifting variants |

## Streaming

`stream()` yields `(x_dict, y)` pairs in the dataset's natural order — the
order carries the drift, so it is never shuffled.

```python
for x, y in stream("electricity", task="classification", max_samples=5000):
    ...   # x: {"feature": float, ...}, y: int (cls) or float (reg)
```

| Argument | Meaning |
|----------|---------|
| `name` | dataset name |
| `task` | `"classification"` (default) or `"regression"` |
| `max_samples` | stop after N instances |
| `normalize` | min-max scale features online, from the range seen so far |

`normalize=True` scales using statistics from instances seen so far only —
never the whole file — so it leaks no future information. It is what
`BenchmarkSuite(normalize=True)` passes through, and is recommended for
gradient-based regressors.

## Loading in bulk

When you want arrays rather than a stream:

```python
X, y, meta = load("electricity", task="classification", max_samples=None)

meta["feature_names"]   # list[str]
meta["target_name"]
meta["n_samples"], meta["n_features"]
meta["n_classes"]       # classification only
```

`X` is `float32` of shape `(n_samples, n_features)`; `y` is `int32` for
classification and `float32` for regression.

## Adding your own dataset

Write an `.npz` with four arrays and drop it in the right directory:

```python
import numpy as np

np.savez_compressed(
    "samlb/datasets/classification/my_stream.npz",
    X=X.astype("float32"),                 # (n_samples, n_features)
    y=y.astype("int32"),                   # int32 cls / float32 reg
    feature_names=np.array(feature_names), # (n_features,)
    target_name=np.array("target"),
)
```

`list_datasets()` and `load()` pick it up automatically — no registry to edit.

Keep the rows in their real temporal order. A shuffled stream measures
something else entirely.

## Using a stream SAMLB does not ship

Nothing in the benchmark requires `samlb.datasets`; the evaluator only needs an
iterable of `(x_dict, y)`. Drive a model directly over your own iterator:

```python
def my_stream():
    for row in my_source:
        yield {"f1": row.a, "f2": row.b}, row.label

hits = n = 0
for x, y in my_stream():
    pred = model.predict_one(x)
    if pred is not None:
        hits += pred == y
        n += 1
    model.learn_one(x, y)
```

To get `BenchmarkSuite`'s tables and JSON output for such a stream, add it as
an `.npz` as above — that is the only path the suite knows.
