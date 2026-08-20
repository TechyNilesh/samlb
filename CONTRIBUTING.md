# Contributing to SAMLB

Thanks for your interest in SAMLB. Contributions of every size are welcome —
a new streaming AutoML framework, a dataset, an adapter for another library,
a bug fix, or a correction to the docs.

- **Detailed usage documentation:** [the SAMLB wiki](https://github.com/TechyNilesh/samlb/wiki)
  (source lives in [`wiki/`](wiki/))
- **Bugs and feature requests:** [open an issue](https://github.com/TechyNilesh/samlb/issues)

---

## Table of contents

- [Development setup](#development-setup)
- [Running the tests](#running-the-tests)
- [Code style](#code-style)
- [Adding a new streaming AutoML framework](#adding-a-new-streaming-automl-framework)
- [Adding a new dataset](#adding-a-new-dataset)
- [Adding an adapter for another library](#adding-an-adapter-for-another-library)
- [Adding a C++ algorithm](#adding-a-c-algorithm)
- [Editing the documentation](#editing-the-documentation)
- [Reporting a bug](#reporting-a-bug)
- [Pull request checklist](#pull-request-checklist)

---

## Development setup

SAMLB ships a native extension, so an editable install compiles C++ the first
time. You need Python >= 3.9, a C++17 compiler and CMake.

```bash
git clone https://github.com/TechyNilesh/samlb.git
cd samlb
pip install -e ".[dev]"
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -c "import samlb; print(samlb.__version__)"
```

Optional backends, only needed if you work on the corresponding adapter:

```bash
pip install -e ".[river]"     # River adapters
pip install -e ".[capymoa]"   # CapyMOA adapters (also needs a JVM)
pip install -e ".[vw]"        # Vowpal Wabbit, for the ChaCha regressor
```

### Rebuilding the C++ extension

An editable install does **not** recompile automatically when you edit a file
under `_cpp/`. Re-run the install, or build in place:

```bash
pip install -e .                            # simplest
# or, for a faster loop:
cmake -S . -B build/local -DCMAKE_BUILD_TYPE=Release
cmake --build build/local -j
cp build/local/_samlb_core.*.so samlb/
```

If an import fails with a missing attribute on `samlb._samlb_core`, the
compiled extension is stale — rebuild before looking any further.

## Running the tests

```bash
pytest tests/
```

Some tests stream real datasets, which are downloaded on first use and cached.
The tests for the optional adapters skip themselves when their backend is not
installed, so a plain `pytest` run is expected to report skips.

Test suites that import the benchmark CLIs need `examples/` on the path:

```bash
PYTHONPATH=examples pytest tests/
```

## Code style

```bash
ruff check samlb/
ruff format samlb/
```

Line length is 100. Match the surrounding code: type hints on public
signatures, NumPy-style docstrings on public classes, and comments that
explain *why* rather than restating the code.

---

## Adding a new streaming AutoML framework

This is the primary way to contribute. Every framework implements the same
three methods, so the benchmark treats yours exactly like the built-in ones.

### Step 1 — create the framework directory

```
samlb/framework/classification/my_method/    # (or regression/)
    __init__.py
    model.py
    config.py        # optional: search space / hyperparameter config
```

### Step 2 — implement `BaseStreamFramework`

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
        """Return a prediction for one instance BEFORE learning from it.

        x : dict mapping feature_name -> float value
        Returns: class label (int) for classification, value (float) for regression.
        Return None while the model is still warming up; the evaluator counts
        the instance but does not score it.
        """
        return self._current_model_predict(x)

    def learn_one(self, x: Dict[str, float], y: Any) -> None:
        """Update the model with one labelled instance.

        This is where your AutoML logic lives:
        update base learners, evaluate pipeline candidates, detect drift and
        adapt, explore new configurations.
        """
        self._update(x, y)

    def reset(self) -> None:
        """Reset to the initial untrained state (called before every run)."""
        self._init_state()
```

### Step 3 — register it

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

### Step 4 — build on the C++ components

Everything that runs per instance is already native; reuse it rather than
writing Python inner loops.

```python
from samlb.framework.base import (
    NaiveBayes, Perceptron, LogisticRegression,
    HoeffdingTreeClassifier, KNNClassifier, SGTClassifier,
    MinMaxScaler, StandardScaler, VarianceThreshold, SelectKBest,
)
from samlb.metrics import Accuracy, MAE, ADWIN, EDDM

# The | operator fuses the chain into a single C++ object, so an instance
# crosses the Python/C++ boundary once per call rather than once per stage.
pipeline = MinMaxScaler() | HoeffdingTreeClassifier(grace_period=200)
pipeline.predict_one(x)
pipeline.learn_one(x, y)
```

Draw candidate learners and preprocessors from the shared pools in
`samlb.framework.classification.shared_config` when your method searches over
a pool — that is what makes a comparison against the other frameworks fair.

### Step 5 — run it in the benchmark

```python
from samlb.benchmark import BenchmarkSuite
from samlb.framework.classification.my_method import MyStreamingAutoML

suite = BenchmarkSuite(
    models={"MyMethod": MyStreamingAutoML(seed=42)},
    datasets=["electricity", "covertype", "insects"],
    task="classification",
    n_runs=10,
)
suite.run()
suite.print_table()
```

### Step 6 — add tests

```python
# tests/test_my_method.py

from samlb.datasets import stream
from samlb.framework.classification.my_method import MyStreamingAutoML


def test_predict_and_learn():
    model = MyStreamingAutoML(seed=42)
    pred = None
    for x, y in stream("electricity", task="classification", max_samples=500):
        pred = model.predict_one(x)
        model.learn_one(x, y)
    assert pred is not None


def test_reset_makes_runs_reproducible():
    """reset() must clear learned state — two runs of the same stream match."""
    data = list(stream("electricity", task="classification", max_samples=500))
    model = MyStreamingAutoML(seed=42)

    def prequential():
        hits = 0
        for x, y in data:
            hits += model.predict_one(x) == y
            model.learn_one(x, y)
        return hits

    first = prequential()
    model.reset()
    assert prequential() == first
```

## Adding a new dataset

1. Prepare the data as a NumPy `.npz` file with this schema:
   - `X` — `float32` array of shape `(n_samples, n_features)`
   - `y` — `int32` (classification) or `float32` (regression), shape `(n_samples,)`
   - `feature_names` — string array of shape `(n_features,)`
   - `target_name` — string scalar
2. Place it in `samlb/datasets/classification/` or `samlb/datasets/regression/`.
3. `list_datasets()` and `load()` discover it automatically.

Keep the stream in its natural order — do **not** shuffle. Order carries the
concept drift that the whole benchmark is about.

## Adding an adapter for another library

`samlb/framework/adapters/` holds the River and CapyMOA adapters; a third
library follows the same shape:

- Subclass `BaseStreamFramework` and implement `predict_one` / `learn_one` /
  `reset`.
- Keep the dependency **optional**: import inside the functions that need it,
  and expose a static `is_available()` so a suite can skip the backend rather
  than fail.
- Never train the object the user handed you — clone or rebuild it, so one
  adapter instance can be reused across seeds and datasets.
- Add the dependency as an extra in `pyproject.toml`.
- Guard the tests with `@pytest.mark.skipif(not X.is_available(), ...)` and
  test on synthetic data, so they run without dataset downloads.

## Adding a C++ algorithm

1. Add the header and source under `_cpp/<area>/`.
2. Add the `.cpp` file to `CPP_SOURCES` in `CMakeLists.txt`.
3. Bind it in `_cpp/bindings/pybind_module.cpp`.
4. Add the Python wrapper in `samlb/framework/base/_cpp_wrappers.py` and export
   it from `samlb/framework/base/__init__.py`.
5. Validate it against a reference implementation (River or MOA) before
   claiming parity, and say so in the PR.

Two build constraints worth knowing: `-ffast-math` is deliberately **not**
enabled, because the learners use `-inf` sentinels that it makes undefined;
and any new learner must be deterministic given its seed.

## Editing the documentation

The wiki pages are version-controlled in [`wiki/`](wiki/) so they can be
reviewed in a PR like any other change. Edit the Markdown there, and a
maintainer publishes it to the GitHub wiki with:

```bash
./scripts/publish_wiki.sh
```

See [`wiki/README.md`](wiki/README.md) for the details.

## Reporting a bug

Open an issue with:

- SAMLB version (`python -c "import samlb; print(samlb.__version__)"`), Python
  version and OS
- the smallest snippet that reproduces the problem
- what you expected and what happened, with the full traceback

For a wrong-looking benchmark number, include the dataset, the seed, and the
metric — a result that is merely surprising is often drift, not a bug.

## Pull request checklist

- [ ] `ruff check samlb/` passes
- [ ] `pytest tests/` passes (skips for uninstalled optional backends are fine)
- [ ] New framework implements all three `BaseStreamFramework` methods
- [ ] New optional dependency is an extra, imported lazily, with `is_available()`
- [ ] Brief description of the AutoML strategy, and a reference if there is a paper
- [ ] Benchmark results on at least 3 datasets for a new framework
- [ ] Docs updated — README for the surface, `wiki/` for the detail
