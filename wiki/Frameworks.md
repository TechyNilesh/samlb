# Frameworks

Every framework implements `predict_one` / `learn_one` / `reset`, and every one
searches the **same** pool of C++ base learners and preprocessors. That shared
pool is the point: the only thing a benchmark result compares is the search
strategy.

## Classification

| Framework | Class | Strategy |
|-----------|-------|----------|
| ASML | `AutoStreamClassifier` | Adaptive Random Drift Nearby Search — ADWIN drift detection, recency-weighted ensemble, adaptive budget |
| AutoClass | `AutoClass` | Genetic algorithm with an ARF meta-regressor guiding hyperparameter mutation |
| EvoAutoML | `EvolutionaryBaggingClassifier` | Evolutionary bagging — population, tournament selection, Poisson(6) sampling |
| OAML | `OAMLClassifier` | Drift-triggered random search — EDDM detector, warm-up phase |
| RandomSearch | `RandomSearch` | Baseline: keeps the whole pool warm, picks one pipeline per exploration window |

```python
from samlb.framework.classification.asml      import AutoStreamClassifier, default_config_dict
from samlb.framework.classification.autoclass import AutoClass
from samlb.framework.classification.eaml      import EvolutionaryBaggingClassifier, EAML_CLF_PARAM_GRID
from samlb.framework.classification.oaml      import OAMLClassifier, OAML_SCALERS, OAML_CLASSIFIERS
from samlb.framework.random_search            import RandomSearch
from samlb.framework.classification.shared_config import (
    SHARED_PREPROCESSORS, SHARED_CLASSIFIER_INSTANCES,
)

models = {
    "ASML":         AutoStreamClassifier(config_dict=default_config_dict, seed=42),
    "AutoClass":    AutoClass(seed=42),
    "EvoAutoML":    EvolutionaryBaggingClassifier(param_grid=EAML_CLF_PARAM_GRID, seed=42),
    "OAML":         OAMLClassifier(scalers=OAML_SCALERS, classifiers=OAML_CLASSIFIERS, seed=42),
    "RandomSearch": RandomSearch(scalers=SHARED_PREPROCESSORS,
                                 models=SHARED_CLASSIFIER_INSTANCES, seed=42),
}
```

## Regression

| Framework | Class | Strategy |
|-----------|-------|----------|
| ASML | `AutoStreamRegressor` | Same search as the classifier, plus Welford target normalisation and prediction clipping |
| EvoAutoML | `EvolutionaryBaggingRegressor` | Evolutionary bagging over regression pipelines |
| ChaCha | `ChaChaRegressor` | FLAML AutoVW — Vowpal Wabbit online HPO, progressive validation loss. Optional (`pip install "samlb[vw]"`) |
| RandomSearch | `RandomSearch` | Same baseline, with `clip=True` for regression |

```python
from samlb.framework.regression.asml   import AutoStreamRegressor, default_config_dict
from samlb.framework.regression.eaml   import EvolutionaryBaggingRegressor, EAML_REG_PARAM_GRID
from samlb.framework.regression.chacha import ChaChaRegressor

models = {
    "ASML":      AutoStreamRegressor(config_dict=default_config_dict, seed=42),
    "EvoAutoML": EvolutionaryBaggingRegressor(param_grid=EAML_REG_PARAM_GRID, seed=42),
}
if ChaChaRegressor.is_available():
    models["ChaCha"] = ChaChaRegressor(seed=42)
```

## SOTA baselines

Not AutoML — single methods, included so AutoML results have a floor to be read
against.

```python
from samlb.framework.base import ARFClassifier, ARFRegressor

ARFClassifier(n_models=10, seed=42)   # Gomes et al. 2017
ARFRegressor(n_models=10, seed=42)
```

SRP (Gomes et al. 2019) is implemented in the C++ core but has no Python
wrapper yet; it is reachable as `samlb._samlb_core.SRPClassifier`, without the
`clone()` / pipeline surface the wrapped learners have.

## Constructor parameters

```python
AutoStreamClassifier(config_dict=None, metric=None, exploration_window=1000,
                     budget=10, ensemble_size=3, prediction_mode="ensemble",
                     verbose=False, seed=42)

AutoClass(config_dict=None, metric=None, exploration_window=1000,
          population_size=10, seed=42)

EvolutionaryBaggingClassifier(population_size=10, sampling_size=1,
                              sampling_rate=1000, seed=42, param_grid=None)

OAMLClassifier(initial_batch_size=200, window_size=500, population_size=10,
               generations=3, train_split=0.8, force_research_interval=50000,
               min_research_gap=1000, ...)

RandomSearch(scalers, models, exploration_window=1000, clip=False, seed=42)
```

`AutoStreamRegressor` adds `feature_selection=True`;
`EvolutionaryBaggingRegressor` mirrors the classifier.

Three conventions hold across all of them:

- `exploration_window` — instances between search steps
- `budget` — configurations explored per step
- `seed` — set by `BenchmarkSuite` before every run, so pass it but do not rely
  on the value you passed surviving a sweep

## The shared pool

`samlb.framework.classification.shared_config` is the single source of truth:

```python
from samlb.framework.classification.shared_config import (
    SHARED_PREPROCESSORS,          # MinMax, Standard, MaxAbs
    SHARED_MODEL_POOL,             # one default instance per learner type
    SHARED_HYPERPARAMETERS,        # search space per class name
    SHARED_CLASSIFIER_INSTANCES,   # the pool expanded into concrete configs
    DEFAULT_CLASSIFICATION_CONFIG, # all four bundled together
)
```

To change the pool for **every** framework at once, build a
`ClassificationConfig` and feed each framework from it:

```python
from samlb.framework.classification.shared_config import ClassificationConfig
from samlb.framework.base import HoeffdingTreeClassifier, MinMaxScaler, Perceptron

cfg = ClassificationConfig(
    scalers=[MinMaxScaler()],
    model_pool=[HoeffdingTreeClassifier(), Perceptron()],
    hyperparameters={
        "HoeffdingTreeClassifier": {"grace_period": [100, 200, 500]},
        "Perceptron":              {"learning_rate": [0.01, 0.1]},
    },
    classifier_instances=[
        HoeffdingTreeClassifier(grace_period=100),
        HoeffdingTreeClassifier(grace_period=500),
        Perceptron(learning_rate=0.01),
    ],
)

models = {
    "ASML":      AutoStreamClassifier(config_dict=cfg.asml_config_dict(), seed=42),
    "AutoClass": AutoClass(config_dict=cfg.autoclass_config_dict(), seed=42),
    "EvoAutoML": EvolutionaryBaggingClassifier(param_grid=cfg.eaml_param_grid(), seed=42),
    "OAML":      OAMLClassifier(scalers=cfg.scalers,
                                classifiers=cfg.classifier_instances, seed=42),
}
```

Each `*_config_dict()` / `*_param_grid()` translates one config into the shape
that framework expects, so a pool change stays consistent across all of them.

Writing your own framework: [[Extending SAMLB]].
Benchmarking a River or MOA method instead: [[External Algorithms]].
