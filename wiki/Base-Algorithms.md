# Base Algorithms

Every component that runs per instance is C++, exposed through thin Python
wrappers. A framework searching over these pays Python overhead once per
`learn_one`, not once per arithmetic operation.

```python
from samlb.framework.base import (
    # classification
    NaiveBayes, Perceptron, LogisticRegression, PassiveAggressiveClassifier,
    SoftmaxRegression, KNNClassifier, HoeffdingTreeClassifier, EFDTClassifier,
    SGTClassifier, ARFClassifier, SRPClassifier,
    # regression
    LinearRegression, BayesianLinearRegression, PassiveAggressiveRegressor,
    KNNRegressor, HoeffdingTreeRegressor, ARFRegressor,
    # preprocessing / feature selection
    StandardScaler, MinMaxScaler, MaxAbsScaler, VarianceThreshold, SelectKBest,
)
```

## The estimator surface

Every wrapper is a `samlb.base.Estimator`:

```python
model = HoeffdingTreeClassifier(grace_period=100)

model.learn_one(x, y)
model.predict_one(x)
model.predict_proba_one(x)      # classifiers
model.reset()

model._get_params()             # {"grace_period": 100, ...}
fresh = model.clone()           # same hyperparameters, no learned state
tuned = model.clone({"grace_period": 500})
```

`clone()` is what frameworks use to spawn candidates, and `_get_params()`
introspects hyperparameters without per-class boilerplate — it reads the
`__init__` signature, so an attribute must be named exactly like its argument.

## Selected signatures

```python
HoeffdingTreeClassifier(grace_period=200, split_confidence=1e-7, tie_threshold=0.05,
                        nb_threshold=0, max_depth=20, split_criterion="info_gain", ...)
KNNClassifier(n_neighbors=5, window_size=1000, p=2)
ARFClassifier(n_models=10, seed=0, lambda_value=6.0, drift_delta=0.001,
              warning_delta=0.01, grace_period=50, max_depth=20,
              split_confidence=0.01, subspace_size=-1)
SRPClassifier(n_models=10, seed=0, lambda_value=6.0, drift_delta=0.001,
              warning_delta=0.01, grace_period=50, max_depth=20,
              split_confidence=0.01, subspace_fraction=0.6,
              training_method="patches")   # or "subspaces" / "resampling"

HoeffdingTreeRegressor(grace_period=200, split_confidence=1e-7,
                       tie_threshold=0.05, max_depth=20, learning_rate=0.01)
ARFRegressor(n_models=10, seed=0, lambda_value=6.0, drift_delta=0.001, ...)

StandardScaler(with_std=True)
VarianceThreshold(threshold=0.0, min_samples=2)
SelectKBest(k=10, use_abs=False)
```

Leaf prediction in `HoeffdingTreeClassifier` is Naive Bayes Adaptive, matching
MOA and River. `HoeffdingTreeRegressor` fits its leaf linear model in
standardised space, chooses adaptively between that model and the leaf mean,
and cannot extrapolate beyond the leaf's own target spread.

## Fused pipelines

The `|` operator builds a pipeline that is stitched together **in C++**, so an
instance crosses the Python boundary once per call instead of once per stage.

```python
from samlb.framework.base import (
    HoeffdingTreeClassifier, SelectKBest, StandardScaler,
)

pipeline = StandardScaler() | SelectKBest(k=8) | HoeffdingTreeClassifier()

pipeline.learn_one(x, y)
pipeline.predict_one(x)
pipeline.predict_proba_one(x)
pipeline.reset()
pipeline.clone()
pipeline.steps          # {"StandardScaler": ..., "SelectKBest": ..., ...}
```

Rules:

- Every step but the last must be a transformer; the last must be a learner.
- `scaler | selector` on its own is a valid *partial* chain — composable, but
  not runnable until a learner is appended.
- `None` steps are skipped, so an optional selector needs no branching:
  `StandardScaler() | maybe_selector | model`.

Feature order is handed to C++ once, from the first instance, because
`SelectKBest` breaks ties by it and a C++ hash map does not preserve order.

## Metrics

`samlb.metrics` replaces the `river.metrics` / `river.drift` surface SAMLB used
to depend on, with River's calling convention preserved.

```python
from samlb.metrics import (
    Accuracy, MacroF1, MacroPrecision, MacroRecall,   # classification
    MAE, RMSE, R2,                                    # regression
    WindowMetric, WindowRegressionMetric,             # sliding window
    ADWIN, EDDM,                                      # drift detection
)

m = Accuracy()
m.update(y_true, y_pred)
m.get()
```

`WindowMetric` tracks accuracy or macro-F1 over a sliding window and
`WindowRegressionMetric` tracks MAE or RMSE — useful for weighting an ensemble
member by its *recent* performance rather than its lifetime average.

## Drift detection

```python
from samlb.metrics import ADWIN, EDDM

detector = ADWIN(delta=0.002)
for x, y in data_stream:
    correct = model.predict_one(x) == y
    detector.update(0 if correct else 1)
    if detector.drift_detected:
        model = model.clone()      # react however your strategy demands
    model.learn_one(x, y)
```

ADWIN maintains an adaptive window and signals when the mean of recent
observations differs significantly from older ones; EDDM watches the distance
between errors and is more sensitive to gradual drift.

## A note on the build

`-ffast-math` is deliberately **not** enabled. It implies
`-ffinite-math-only`, under which the `-inf` sentinels used throughout the
learners are undefined behaviour — with it, results stopped being reproducible
across unrelated work in the same process.
