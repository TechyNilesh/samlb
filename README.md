<p align="center">
  <img src="https://raw.githubusercontent.com/TechyNilesh/samlb/main/assets/samlb_logo.png" alt="SAMLB Logo" width="400">
</p>
<p align="center">A unified benchmark framework for evaluating AutoML systems on data streams with fast C++ base algorithms and rigorous prequential evaluation.</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://pypi.org/project/samlb/"><img src="https://img.shields.io/pypi/v/samlb.svg" alt="PyPI"></a>
  <a href="https://pepy.tech/project/samlb"><img src="https://static.pepy.tech/badge/samlb" alt="Downloads"></a>
  <a href="https://github.com/TechyNilesh/samlb/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

---

## Why SAMLB?

Streaming AutoML methods are hard to compare fairly. Different papers use different datasets, evaluation protocols, and algorithm pools. **SAMLB** solves this by providing:

- **Pure C++ core** — base learners, preprocessing, feature selection, metrics and drift detection are all native, with no Python ML dependency (Naive Bayes, Hoeffding Trees, KNN, Perceptron, Logistic Regression, and more)
- **Framework-agnostic benchmarking** -- plug in any streaming AutoML method with just 3 methods
- **Standardized prequential evaluation** (test-then-train) with windowed metric snapshots for learning curves
- **30 curated datasets** (15 classification + 15 regression) spanning real-world and synthetic drift scenarios
- **Parallel execution** for large-scale experiments across multiple seeds

## Installation

### From PyPI

```bash
pip install samlb
```

### From source

```bash
git clone https://github.com/TechyNilesh/samlb.git
cd samlb
pip install -e ".[dev]"
```

### Optional: Vowpal Wabbit support (for ChaCha regressor)

```bash
pip install "samlb[vw]"
```

> **Requirements:** Python >= 3.9, a C++ compiler (for the native extension), CMake

## Quick Start

### Python API

```python
from samlb.benchmark import BenchmarkSuite
from samlb.framework.classification.asml import AutoStreamClassifier
from samlb.framework.classification.eaml import EvolutionaryBaggingClassifier
from samlb.framework.classification.sag import StreamingAutoGluon
from samlb.framework.random_search import RandomSearch
from samlb.framework.classification.shared_config import (
    SHARED_PREPROCESSORS, SHARED_CLASSIFIER_INSTANCES,
)

suite = BenchmarkSuite(
    models={
        "ASML":         AutoStreamClassifier(seed=42),
        "EvoAutoML":    EvolutionaryBaggingClassifier(seed=42),
        "StreamingAutoGluon": StreamingAutoGluon(seed=42),
        # RandomSearch baseline over the same shared learner pool
        "RandomSearch": RandomSearch(
            scalers=SHARED_PREPROCESSORS,
            models=SHARED_CLASSIFIER_INSTANCES,
            seed=42,
        ),
    },
    datasets=["electricity", "covertype"],
    task="classification",
    n_runs=10,
    window_size=1000,
)
suite.run()
suite.print_table()
suite.to_csv("results/classification.csv")
```

> RandomSearch is task-agnostic: for regression, pass the regression pool
> (`EAML_REG_PARAM_GRID["Scaler"]` / `["Regressor"]` from
> `samlb.framework.regression.eaml.config`) and set `clip=True`.

### Dataset Streaming

```python
from samlb.datasets import stream, list_datasets

# See all available datasets
print(list_datasets("classification"))
print(list_datasets("regression"))

# Stream instance by instance
for x, y in stream("electricity", task="classification"):
    pred = model.predict_one(x)
    model.learn_one(x, y)
```

### CLI

```bash
# Full classification benchmark (5 frameworks x 15 datasets x 10 runs)
python examples/run_benchmark.py

# Custom subset
python examples/run_benchmark.py --n_runs 5 --max_samples 50000 --datasets electricity covertype

# Parallel execution across CPU cores
python examples/run_benchmark.py --n_runs 100 --parallel --cpu_utilization 0.8

# Regression benchmark (4 frameworks x 15 datasets x 10 runs)
python examples/run_regression.py
python examples/run_regression.py --n_runs 5 --datasets bike california_housing
```

## Included Frameworks

### Classification

| Framework | Strategy | Key Features |
|-----------|----------|--------------|
| **ASML** | Adaptive Random Drift Nearby Search | ADWIN drift detection, recency-weighted ensemble, adaptive budget |
| **AutoClass** | Genetic Algorithm + Meta-Regressor | Fitness-proportionate selection, ARF surrogate for HP mutation |
| **EvoAutoML** | Evolutionary Bagging | Population-based, tournament selection, Poisson(6) sampling |
| **OAML** | Drift-triggered Random Search | EDDM drift detector, warm-up phase, random search |
| **StreamingAutoGluon** | Online stacking of k-fold CV learners | k-fold cross-validated base learners feeding out-of-fold meta features to per-type stacked learners, votes weighted by windowed accuracy / macro-F1 |

### Regression

| Framework | Strategy | Key Features |
|-----------|----------|--------------|
| **ASML** | Adaptive Random Drift Nearby Search | Online target normalization (Welford), prediction clipping |
| **ChaCha** | FLAML AutoVW | Vowpal Wabbit online HPO, progressive validation loss |
| **EvoAutoML** | Evolutionary Bagging | Population-based ensemble, mutation-driven search |
| **StreamingAutoGluon** | Online stacking of k-fold CV learners | Out-of-fold meta features from k-fold cross-validated regressors, stacked predictions weighted by inverse windowed MAE / RMSE |

### Baseline (classification & regression)

| Baseline | Strategy | Key Features |
|----------|----------|--------------|
| **RandomSearch** | Random per-window selection | Keeps the full shared learner pool warm, randomly picks one pipeline per exploration window |

### StreamingAutoGluon

Online stacking, implementing the method from its original Java reference
implementation (`StreamingAutoGluon.java`).

For each of `T` base learner types, `k + 1` learners are built. The `k` **fold
learners** are trained with online k-fold cross-validation — the learner whose
index equals `instances_seen % k` is held out from the current instance — and
always see the raw features. The remaining **stacked learner** is trained on
the instance augmented with the average class-probability prediction of all
fold learners of every type; because every fold learner predicts before any of
them is trained on that instance, the meta features stay out-of-fold. At
prediction time the stacked learners' votes are combined, each weighted by its
accuracy or macro-F1 over a sliding window.

Base learners and preprocessors both come from the same shared pool every other
framework searches — one base type per algorithm in `SHARED_MODEL_POOL`, with
`SHARED_PREPROCESSORS` handed out round-robin across those types.

```python
from samlb.framework.classification.sag import StreamingAutoGluon, SAG_BASE_LEARNERS_SMALL
from samlb.framework.classification.shared_config import DEFAULT_CLASSIFICATION_CONFIG as cfg

model = StreamingAutoGluon(
    **cfg.sag_kwargs(),     # learners + scalers from the shared pool (the default)
    n_folds=3,              # k  (-k in the Java reference)
    metric="accuracy",      # or "f1"  (-m in the Java reference)
    window_size=1000,       # (-w in the Java reference)
    seed=42,
)

# cheaper: 3 base types instead of the full pool
fast = StreamingAutoGluon(learners=SAG_BASE_LEARNERS_SMALL, seed=42)

# raw features, as the Java reference does
raw = StreamingAutoGluon(scalers=None, seed=42)
```

Cost scales with `len(learners) * (n_folds + 1)`, so the base-type list is the
main accuracy/speed dial. Two deviations from the Java reference: the base types
are drawn from the shared SAMLB C++ pool rather than ARF/ARTE/SORE, and the
class set is discovered online instead of being read from an ARFF header, so the meta-feature block grows as labels appear.

A regression counterpart is available as
`samlb.framework.regression.sag.StreamingAutoGluonRegressor`. The structure is
identical; only the two class-specific pieces change — each base type
contributes one meta feature (the average prediction of its fold learners)
instead of one per class, and the stacked regressors are weighted by
`1 / (eps + windowed MAE or RMSE)` so lower error earns a larger share.

```python
from samlb.framework.regression.sag import StreamingAutoGluonRegressor

model = StreamingAutoGluonRegressor(n_folds=3, metric="mae", clip=True, seed=42)
```

Note that StreamingAutoGluon is deterministic (the Java reference reports
`isRandomizable() == false`), so repeated runs with different seeds give
identical results and its standard deviation across runs is always zero.

## C++ Base Algorithms

Every per-instance component is implemented in C++ and exposed through thin Python wrappers:

**Classification:** Naive Bayes, Perceptron, Logistic Regression, Passive Aggressive, Softmax Regression, KNN, Hoeffding Tree, EFDT, SGT

**Regression:** Linear Regression, Bayesian Linear Regression, Passive Aggressive, Hoeffding Tree, KNN

**Preprocessing:** MinMaxScaler, StandardScaler, MaxAbsScaler, VarianceThreshold, SelectKBest (Pearson)

**Metrics:** Accuracy, MacroF1, MacroPrecision, MacroRecall, MAE, RMSE, R²

**Drift detection:** ADWIN, EDDM

Pipelines are *fused*: `scaler | selector | model` is executed as a single C++
object, so an instance crosses the Python/C++ boundary once per `learn_one` /
`predict_one` rather than once per stage.

## Evaluation Methodology

SAMLB uses **prequential evaluation** (test-then-train):

1. For each instance in the stream:
   - **Predict** -- get the model's prediction *before* seeing the label
   - **Evaluate** -- score the prediction against the true label
   - **Learn** -- update the model with the labelled instance
2. Metrics are captured at configurable window intervals for learning curve analysis
3. Runtime is sampled per-instance for performance profiling

**Classification metrics:** Accuracy, Macro-F1, Macro-Precision, Macro-Recall

**Regression metrics:** MAE, RMSE, R^2

## Datasets

### Classification (15 datasets -- 2.5M+ total instances)

| Dataset | Samples | Features | Classes | Type | Description |
|---------|--------:|---------:|--------:|------|-------------|
| `adult` | 48,842 | 14 | 4 | Real | Income prediction (Census) |
| `covertype` | 100,000 | 54 | 7 | Real | Forest cover type (cartographic) |
| `credit_card` | 284,807 | 30 | 2 | Real | Credit card fraud detection |
| `electricity` | 45,312 | 8 | 2 | Real | Electricity price direction (NSW, Australia) |
| `insects` | 52,848 | 33 | 6 | Real | Insect species with concept drift |
| `new_airlines` | 539,383 | 7 | 2 | Real | Flight delay prediction |
| `nomao` | 34,465 | 118 | 2 | Real | Nomao place deduplication |
| `poker_hand` | 1,025,009 | 10 | 10 | Real | Poker hand classification |
| `shuttle` | 58,000 | 9 | 7 | Real | NASA Space Shuttle radiator |
| `vehicle_sensIT` | 98,528 | 100 | 3 | Real | Vehicle type from seismic sensors |
| `movingRBF` | 200,000 | 10 | 5 | Synthetic | Moving radial basis functions |
| `moving_squares` | 200,000 | 2 | 4 | Synthetic | Moving class boundaries |
| `sea_high_abrupt_drift` | 500,000 | 3 | 2 | Synthetic | SEA generator with abrupt drift |
| `synth_RandomRBFDrift` | 100,000 | 4 | 4 | Synthetic | RBF generator with gradual drift |
| `synth_agrawal` | 100,000 | 9 | 2 | Synthetic | Agrawal generator |

### Regression (15 datasets -- 1M+ total instances)

| Dataset | Samples | Features | Type | Description |
|---------|--------:|---------:|------|-------------|
| `ailerons` | 13,750 | 40 | Real | Aircraft control surface deflection |
| `bike` | 17,379 | 12 | Real | Bike sharing hourly demand |
| `california_housing` | 20,640 | 8 | Real | California median house values |
| `cps88wages` | 28,155 | 6 | Real | Wage prediction (CPS 1988) |
| `diamonds` | 53,940 | 9 | Real | Diamond price prediction |
| `elevators` | 16,599 | 18 | Real | Aircraft elevator control |
| `fifa` | 19,178 | 28 | Real | FIFA player overall rating |
| `House8L` | 22,784 | 8 | Real | House price (8-feature variant) |
| `kings_county` | 21,613 | 21 | Real | King County house sales price |
| `MetroTraffic` | 48,204 | 7 | Real | Interstate traffic volume (Minneapolis) |
| `superconductivity` | 21,263 | 81 | Real | Superconductor critical temperature |
| `wave_energy` | 72,000 | 48 | Real | Wave energy converter power output |
| `fried` | 40,768 | 10 | Synthetic | Friedman function |
| `FriedmanGra` | 100,000 | 10 | Synthetic | Friedman with gradual drift |
| `hyperA` | 500,000 | 10 | Synthetic | Hyperplane with drift |

## Output Formats

```
results/
  classification/
    summary.json                  # Flat JSON: one row per (framework x dataset x run)
    <dataset>/<framework>/
      run_00.json                 # Raw per-run JSON with full learning curves
      ...
      run_09.json
      aggregate.json              # Aggregated mean +/- std across 10 runs
  regression/
    summary.json
    <dataset>/<framework>/
      run_00.json
      ...
      run_09.json
      aggregate.json
```

The released repository includes the raw JSON results used for the paper under `results/classification/` and `results/regression/`.

## Project Structure

```
.
├── pyproject.toml             # Package metadata & build config
├── CMakeLists.txt             # C++ build configuration
├── LICENSE                    # MIT License
├── README.md                  # This file
├── _cpp/                      # C++ source (9 classifiers, 5 regressors)
│   ├── classification/
│   ├── regression/
│   ├── core/                  # Shared headers
│   └── bindings/              # PyBind11 module
├── samlb/                     # Python package
│   ├── __init__.py            # Version: 0.3.0
│   ├── algorithms/            # C++ algorithm Python bindings
│   ├── benchmark/             # BenchmarkSuite orchestrator
│   ├── evaluation/            # PrequentialEvaluator, metrics, results
│   ├── datasets/              # 30 datasets (15 clf + 15 reg NPZ files)
│   └── framework/             # AutoML framework implementations
│       ├── base/              # BaseStreamFramework + C++ wrappers
│       ├── random_search.py   # RandomSearch baseline (task-agnostic)
│       ├── classification/    # ASML, AutoClass, EvoAutoML, OAML
│       └── regression/        # ASML, ChaCha, EvoAutoML
├── results/                   # Raw paper results as JSON files
│   ├── classification/        # Classification run_*.json + aggregate.json
│   └── regression/            # Regression run_*.json + aggregate.json
├── tests/                     # Test suite
└── examples/                  # Benchmark runner scripts
    ├── run_benchmark.py       # Classification benchmark CLI
    └── run_regression.py      # Regression benchmark CLI
```

---

## Contributing

We welcome contributions! Whether you are adding a new AutoML framework, new datasets, or fixing bugs.

### Development Setup

```bash
git clone https://github.com/TechyNilesh/samlb.git
cd samlb
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
ruff check samlb/
ruff format samlb/
```

---

### Adding a New Streaming AutoML Framework

This is the primary way to contribute. Every framework in SAMLB implements the same 3-method interface, making it easy to add your own.

#### Step 1 -- Create your framework directory

```
samlb/framework/classification/my_method/    # (or regression/)
    __init__.py
    model.py
    config.py        # optional: search space / hyperparameter config
```

#### Step 2 -- Implement `BaseStreamFramework`

```python
# samlb/framework/classification/my_method/model.py

from __future__ import annotations
from typing import Any, Dict
from samlb.framework.base import BaseStreamFramework


class MyStreamingAutoML(BaseStreamFramework):
    """My new streaming AutoML method."""

    def __init__(self, seed: int = 42, exploration_window: int = 1000, budget: int = 10):
        self.seed = seed
        self.exploration_window = exploration_window
        self.budget = budget
        self._init_state()

    def predict_one(self, x: Dict[str, float]) -> Any:
        """
        Return prediction for one instance BEFORE learning.

        x : dict mapping feature_name -> float value
        Returns: class label (int) for classification, value (float) for regression
        """
        return self._current_model_predict(x)

    def learn_one(self, x: Dict[str, float], y: Any) -> None:
        """
        Update the model with one labelled instance.

        This is where your AutoML logic lives:
        - Update base learners
        - Evaluate pipeline candidates
        - Detect drift and adapt
        - Explore new configurations
        """
        self._update(x, y)

    def reset(self) -> None:
        """Reset to initial untrained state (called before each run)."""
        self._init_state()
```

#### Step 3 -- Register in `__init__.py`

```python
# samlb/framework/classification/__init__.py

from .my_method.model import MyStreamingAutoML

__all__ = [
    "AutoStreamClassifier",
    "AutoClass",
    "EvolutionaryBaggingClassifier",
    "OAMLClassifier",
    "MyStreamingAutoML",       # <-- add here
]
```

#### Step 4 -- Use available building blocks

SAMLB provides the full set of C++ components as building blocks:

```python
# Learners, preprocessing and feature selection (all C++)
from samlb.framework.base import (
    NaiveBayes,
    Perceptron,
    LogisticRegression,
    HoeffdingTreeClassifier,
    KNNClassifier,
    SGTClassifier,
    MinMaxScaler,
    StandardScaler,
    VarianceThreshold,
    SelectKBest,
)

# Metrics and drift detection (also C++)
from samlb.metrics import Accuracy, MAE, ADWIN, EDDM

# Compose a pipeline with the | operator; the chain is fused into one C++ object
pipeline = MinMaxScaler() | HoeffdingTreeClassifier(grace_period=200)
pipeline.predict_one(x)
pipeline.learn_one(x, y)
```

#### Step 5 -- Run it in the benchmark

```python
from samlb.benchmark import BenchmarkSuite
from samlb.framework.classification.my_method import MyStreamingAutoML

suite = BenchmarkSuite(
    models={
        "MyMethod": MyStreamingAutoML(seed=42),
    },
    datasets=["electricity", "covertype", "insects"],
    task="classification",
    n_runs=10,
)
suite.run()
suite.print_table()
```

#### Step 6 -- Add tests

```python
# tests/test_my_method.py

from samlb.framework.classification.my_method import MyStreamingAutoML
from samlb.datasets import stream


def test_predict_and_learn():
    model = MyStreamingAutoML(seed=42)
    for x, y in stream("electricity", task="classification", max_samples=500):
        pred = model.predict_one(x)
        model.learn_one(x, y)
    assert pred is not None


def test_reset():
    model = MyStreamingAutoML(seed=42)
    for x, y in stream("electricity", task="classification", max_samples=100):
        model.learn_one(x, y)
    model.reset()
    # Should be back to untrained state
```

### Adding a New Dataset

1. Prepare your data as a NumPy NPZ file with this schema:
   - `X` -- `float32` array of shape `(n_samples, n_features)`
   - `y` -- `int32` (classification) or `float32` (regression) array of shape `(n_samples,)`
   - `feature_names` -- string array of shape `(n_features,)`
   - `target_name` -- string scalar
2. Place the `.npz` file in `samlb/datasets/classification/` or `samlb/datasets/regression/`
3. It will be automatically discovered by `list_datasets()` and `load()`

### PR Checklist

- [ ] Code passes `ruff check samlb/`
- [ ] Tests pass with `pytest tests/`
- [ ] New framework implements all 3 methods of `BaseStreamFramework`
- [ ] Include a brief description of the AutoML strategy
- [ ] Reference any papers if applicable
- [ ] Include benchmark results on at least 3 datasets

---

## Citation

If you use SAMLB in your research, please cite:

```bibtex
@inproceedings{verma2026samlb,
  title     = {SAMLB: A Streaming AutoML Benchmark},
  author    = {Verma, Nilesh and Bifet, Albert and Pfahringer, Bernhard and Bahri, Maroua},
  booktitle = {Proceedings of the International Conference on Automated Machine Learning (AutoML)},
  year      = {2026},
  url       = {https://github.com/TechyNilesh/samlb}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
