# Extending SAMLB

Four things people usually add: a framework, an adapter for another library, a
dataset, or a C++ learner. The full contributor workflow — setup, tests, style,
PR checklist — is in
[CONTRIBUTING.md](https://github.com/TechyNilesh/samlb/blob/main/CONTRIBUTING.md);
this page is the design side.

## A new AutoML framework

Implement three methods and the whole benchmark works on your method.

```python
# samlb/framework/classification/my_method/model.py
from __future__ import annotations

from typing import Any, Dict

from samlb.framework.base import BaseStreamFramework


class MyStreamingAutoML(BaseStreamFramework):
    """One-line description of the search strategy."""

    def __init__(self, seed: int = 42, exploration_window: int = 1000, budget: int = 10):
        self.seed = seed
        self.exploration_window = exploration_window
        self.budget = budget
        self._init_state()

    def predict_one(self, x: Dict[str, float]) -> Any:
        return self._best.predict_one(x)

    def learn_one(self, x: Dict[str, float], y: Any) -> None:
        self._best.learn_one(x, y)
        self._maybe_explore(x, y)

    def reset(self) -> None:
        self._init_state()
```

Then register it in `samlb/framework/classification/__init__.py` (or
`regression/`).

Five things the benchmark assumes:

1. **`reset()` really resets.** It is called before every run; anything it
   leaves behind leaks across datasets and seeds. The test for this is that two
   runs of the same stream, with a `reset()` between, produce identical numbers.
2. **`predict_one` never learns.** Prequential evaluation depends on the
   prediction happening first.
3. **`None` means "not ready".** Return it during warm-up; the instance is
   counted but not scored. Do not return a fake label instead.
4. **`self.seed` is honoured.** `BenchmarkSuite` sets it before each run and
   expects a different seed to produce a different — but reproducible — search.
5. **Exceptions are survivable but visible.** A raising run is caught and
   reported as `ERROR` in the table, not silently dropped.

### Build on the C++ components

Do not write Python inner loops. Compose the native learners instead:

```python
from samlb.framework.base import (
    HoeffdingTreeClassifier, SelectKBest, StandardScaler,
)
from samlb.metrics import ADWIN, WindowMetric

candidate = StandardScaler() | SelectKBest(k=10) | HoeffdingTreeClassifier()
fresh = candidate.clone()          # untrained copy, same hyperparameters
```

`clone()` on any estimator or pipeline gives a fresh untrained copy — that is
the primitive most search strategies are built from. `WindowMetric` scores a
candidate on its recent behaviour rather than its lifetime average, and `ADWIN`
/ `EDDM` tell you when the stream moved.

### Search the shared pool

A framework that searches its own private pool cannot be fairly compared with
the others. Draw from `shared_config` instead:

```python
from samlb.framework.classification.shared_config import (
    SHARED_CLASSIFIER_INSTANCES, SHARED_HYPERPARAMETERS, SHARED_PREPROCESSORS,
)
```

If your method needs a differently shaped view of the pool, add a
`ClassificationConfig` method that translates it — that is what
`asml_config_dict()`, `autoclass_config_dict()` and `eaml_param_grid()` do.

## A new adapter for another library

Follow `samlb/framework/adapters/`:

```python
class MyLibClassifier(BaseStreamFramework):
    def __init__(self, model, name=None, seed=42):
        if not _has_module("mylib"):
            raise ImportError("... install with `pip install mylib`")
        ...

    @staticmethod
    def is_available() -> bool:
        return _has_module("mylib")
```

Four rules, learned from the River and CapyMOA ones:

- **Import lazily.** A missing optional backend must not break
  `import samlb.framework.adapters`.
- **Expose `is_available()`.** Suites should skip a backend, not crash on it.
- **Never train the caller's object.** Clone it, or rebuild from a factory, so
  one adapter survives a whole sweep.
- **Translate honestly.** If the library speaks a different vocabulary — class
  indices, instance objects, a fixed schema — the adapter converts, and
  converts back on the way out.

Add the dependency as an extra in `pyproject.toml`, and guard the tests with
`@pytest.mark.skipif(not X.is_available(), ...)` on synthetic data so they run
anywhere.

## A new dataset

See [[Datasets]] — write an `.npz` with `X`, `y`, `feature_names`,
`target_name`, drop it in `samlb/datasets/<task>/`, and it is discovered
automatically. Keep the natural order.

## A new C++ learner

1. Header and source under `_cpp/<area>/`.
2. Add the `.cpp` to `CPP_SOURCES` in `CMakeLists.txt`.
3. Bind it in `_cpp/bindings/pybind_module.cpp`.
4. Wrap it in `samlb/framework/base/_cpp_wrappers.py` and export it from
   `samlb/framework/base/__init__.py`.
5. Validate against a reference implementation (River or MOA) before claiming
   parity.

Two constraints: the build deliberately avoids `-ffast-math` (it breaks the
`-inf` sentinels the learners rely on), and a learner must be deterministic
given its seed.

Remember to rebuild — an editable install does not recompile on its own:

```bash
cmake -S . -B build/local -DCMAKE_BUILD_TYPE=Release
cmake --build build/local -j
cp build/local/_samlb_core.*.so samlb/
```

## Testing what you added

```python
def test_reset_makes_runs_reproducible():
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

That single test catches the most common bug in a new framework: state that
survives `reset()` and quietly inflates every result after the first.
