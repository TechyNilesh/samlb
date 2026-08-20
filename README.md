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

### Optional backends

```bash
pip install "samlb[vw]"       # Vowpal Wabbit, for the ChaCha regressor
pip install "samlb[river]"    # River algorithms, via the River adapters
pip install "samlb[capymoa]"  # CapyMOA/MOA algorithms (needs a JVM)
```

> **Requirements:** Python >= 3.9, a C++ compiler (for the native extension), CMake

## Quick Start

### Python API

```python
from samlb.benchmark import BenchmarkSuite
from samlb.framework.classification.asml import AutoStreamClassifier
from samlb.framework.classification.eaml import EvolutionaryBaggingClassifier
from samlb.framework.random_search import RandomSearch
from samlb.framework.classification.shared_config import (
    SHARED_PREPROCESSORS, SHARED_CLASSIFIER_INSTANCES,
)

suite = BenchmarkSuite(
    models={
        "ASML":         AutoStreamClassifier(seed=42),
        "EvoAutoML":    EvolutionaryBaggingClassifier(seed=42),
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

### Regression

| Framework | Strategy | Key Features |
|-----------|----------|--------------|
| **ASML** | Adaptive Random Drift Nearby Search | Online target normalization (Welford), prediction clipping |
| **ChaCha** | FLAML AutoVW | Vowpal Wabbit online HPO, progressive validation loss |
| **EvoAutoML** | Evolutionary Bagging | Population-based ensemble, mutation-driven search |

### Baseline (classification & regression)

| Baseline | Strategy | Key Features |
|----------|----------|--------------|
| **RandomSearch** | Random per-window selection | Keeps the full shared learner pool warm, randomly picks one pipeline per exploration window |

## External Algorithms (River & CapyMOA)

Benchmarks often need to place SAMLB's frameworks next to algorithms from
[River](https://riverml.xyz) or [CapyMOA](https://capymoa.org). Two adapters
make any learner from either library usable as a SAMLB model — same
`predict_one` / `learn_one` / `reset` contract, so it drops straight into
`BenchmarkSuite` and is scored by the same prequential evaluator.

Both libraries are optional. Nothing is imported until an adapter is
constructed, and `is_available()` lets a suite skip a backend that is not
installed rather than fail.

```python
from river import forest, preprocessing
from samlb.benchmark import BenchmarkSuite
from samlb.framework.adapters import CapyMOAClassifier, RiverClassifier
from samlb.framework.base import ARFClassifier

models = {"SAMLB-ARF": ARFClassifier(n_models=10, seed=42)}

if RiverClassifier.is_available():
    models["River-ARF"] = RiverClassifier(
        preprocessing.StandardScaler() | forest.ARFClassifier(n_models=10, seed=42),
        name="River-ARF",
    )

if CapyMOAClassifier.is_available():
    models["MOA-ARF"] = CapyMOAClassifier(
        "AdaptiveRandomForestClassifier", ensemble_size=10, seed=42, name="MOA-ARF",
    )

BenchmarkSuite(models=models, datasets=["electricity"],
               task="classification", n_runs=10).run()
```

`examples/run_external_baselines.py` runs exactly this comparison from the
command line, for either task:

```bash
python3 examples/run_external_baselines.py --task classification --n_runs 10
python3 examples/run_external_baselines.py --task regression --datasets abalone
```

### River adapters

`RiverClassifier` / `RiverRegressor` take a River estimator or a pipeline ending
in one. The object you pass is a **prototype**: it is cloned before every run and
never trained in place, so one adapter can be reused across seeds and datasets.
Pass a zero-argument callable instead when an estimator cannot be cloned.

```python
from samlb.framework.adapters import RiverRegressor

RiverRegressor(preprocessing.StandardScaler() | linear_model.LinearRegression())
RiverRegressor(lambda: forest.ARFRegressor(seed=1), name="River-ARF")
```

River classifiers return `None` until they have seen a label; the evaluator
counts those instances but does not score them, exactly as it does for OAML's
warm-up.

### CapyMOA adapters

`CapyMOAClassifier` / `CapyMOARegressor` take a CapyMOA **class**, or its name in
`capymoa.classifier` / `capymoa.regressor`, plus any learner keyword arguments.
A class rather than an instance, because a CapyMOA learner is bound to a MOA
`Schema` at construction and the schema is not known until the stream starts.
The adapter derives it from the first instance, builds the learner then, and
converts each `{feature: value}` dict to the dense array MOA expects.

```python
from samlb.framework.adapters import CapyMOAClassifier, CapyMOARegressor

CapyMOAClassifier("HoeffdingTree", grace_period=50, seed=42)
CapyMOAClassifier("AdaptiveRandomForestClassifier", ensemble_size=10)
CapyMOARegressor("AdaptiveRandomForestRegressor", ensemble_size=10)
```

MOA works in class *indices*, so the adapter keeps the label mapping and hands
back the original SAMLB labels. Labels are discovered as they arrive, against
`max_classes` reserved nominal slots (100 by default; only MOA's per-class
memory scales with it). Pass `classes=[...]` when the label set is known up
front — the schema is then exact and an unexpected label raises instead of
being silently absorbed.

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
├── CONTRIBUTING.md            # Contributor guide
├── wiki/                      # Wiki page sources (published to the GitHub wiki)
├── scripts/                   # publish_wiki.sh
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
│       ├── adapters/          # River & CapyMOA adapters (optional backends)
│       ├── random_search.py   # RandomSearch baseline (task-agnostic)
│       ├── classification/    # ASML, AutoClass, EvoAutoML, OAML
│       └── regression/        # ASML, ChaCha, EvoAutoML
├── results/                   # Raw paper results as JSON files
│   ├── classification/        # Classification run_*.json + aggregate.json
│   └── regression/            # Regression run_*.json + aggregate.json
├── tests/                     # Test suite
└── examples/                  # Benchmark runner scripts
    ├── run_benchmark.py       # Classification benchmark CLI
    ├── run_regression.py      # Regression benchmark CLI
    └── run_external_baselines.py  # River / CapyMOA comparison CLI
```

---

## Documentation

Full usage documentation lives in the [SAMLB wiki](https://github.com/TechyNilesh/samlb/wiki):

| Page | What it covers |
|------|----------------|
| [Installation](https://github.com/TechyNilesh/samlb/wiki/Installation) | Install, optional backends, build troubleshooting |
| [Quick Start](https://github.com/TechyNilesh/samlb/wiki/Quick-Start) | First benchmark, the model contract, CLI |
| [Benchmark API](https://github.com/TechyNilesh/samlb/wiki/Benchmark-API) | `BenchmarkSuite`, evaluator, `RunResult`, output formats |
| [Datasets](https://github.com/TechyNilesh/samlb/wiki/Datasets) | The 30 bundled streams, and adding your own |
| [Frameworks](https://github.com/TechyNilesh/samlb/wiki/Frameworks) | The bundled AutoML methods and their configuration |
| [Base Algorithms](https://github.com/TechyNilesh/samlb/wiki/Base-Algorithms) | C++ learners, fused pipelines, metrics, drift detectors |
| [External Algorithms](https://github.com/TechyNilesh/samlb/wiki/External-Algorithms) | Benchmarking River and CapyMOA learners |
| [Extending SAMLB](https://github.com/TechyNilesh/samlb/wiki/Extending-SAMLB) | Writing a framework, adapter, dataset or C++ learner |
| [FAQ](https://github.com/TechyNilesh/samlb/wiki/FAQ) | Common questions and failure modes |

The pages are version-controlled in [`wiki/`](wiki/) — edit them there and open
a PR; `./scripts/publish_wiki.sh` pushes them to the wiki.

---

## Contributing

Contributions are welcome — a new streaming AutoML framework, a dataset, an
adapter for another library, a bug fix, or a documentation correction.

```bash
git clone https://github.com/TechyNilesh/samlb.git
cd samlb
pip install -e ".[dev]"
pytest tests/
ruff check samlb/
```

Adding a framework means implementing three methods:

```python
from samlb.framework.base import BaseStreamFramework

class MyStreamingAutoML(BaseStreamFramework):
    def predict_one(self, x): ...   # predict BEFORE learning
    def learn_one(self, x, y): ...  # your AutoML logic lives here
    def reset(self): ...            # back to untrained; called before every run
```

[**CONTRIBUTING.md**](CONTRIBUTING.md) has the full walkthrough — development
setup, rebuilding the C++ extension, the step-by-step guide to adding a
framework, dataset, adapter or C++ learner, and the PR checklist.

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
