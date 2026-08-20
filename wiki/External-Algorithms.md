# External Algorithms (River & CapyMOA)

Benchmarks usually need SAMLB's frameworks placed next to algorithms from
[River](https://riverml.xyz) or [CapyMOA](https://capymoa.org). Two adapters
make any learner from either library a SAMLB model — same `predict_one` /
`learn_one` / `reset` contract, so it drops into `BenchmarkSuite` and is scored
by the same prequential evaluator, on the same streams, against the same
metrics.

```python
from samlb.framework.adapters import (
    RiverClassifier, RiverRegressor,
    CapyMOAClassifier, CapyMOARegressor,
)
```

Both libraries are optional:

```bash
pip install "samlb[river]"
pip install "samlb[capymoa]"    # also needs a JVM
```

Nothing is imported until an adapter is constructed, and `is_available()` lets
a suite skip a missing backend instead of failing.

## A full comparison

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

suite = BenchmarkSuite(models=models, datasets=["electricity"],
                       task="classification", n_runs=10)
suite.run()
suite.print_table()
```

`examples/run_external_baselines.py` runs exactly this from the command line,
for either task:

```bash
python examples/run_external_baselines.py --task classification --n_runs 10
python examples/run_external_baselines.py --task regression --datasets abalone
```

## River adapters

`RiverClassifier` and `RiverRegressor` take a River estimator, or a pipeline
ending in one.

```python
from river import forest, linear_model, preprocessing, tree
from samlb.framework.adapters import RiverClassifier, RiverRegressor

RiverClassifier(tree.HoeffdingTreeClassifier(), name="River-HT")
RiverClassifier(preprocessing.StandardScaler() | forest.ARFClassifier(seed=42))
RiverRegressor(preprocessing.StandardScaler() | linear_model.LinearRegression())
```

The object you pass is a **prototype**. It is `clone()`d before every run and
never trained in place, so one adapter instance is safe to reuse across every
seed and dataset in a sweep — the same guarantee the built-in frameworks give
through `reset()`.

For an estimator that cannot be cloned, pass a zero-argument callable:

```python
RiverClassifier(lambda: forest.ARFClassifier(seed=1), name="River-ARF")
```

The task is checked at construction with River's own inspector (which
understands pipelines), so a classifier handed to `RiverRegressor` fails
immediately rather than halfway through a sweep:

```python
RiverRegressor(tree.HoeffdingTreeClassifier())
# TypeError: RiverRegressor expects a River regressor; HoeffdingTreeClassifier is not one.
```

River classifiers return `None` until they have seen a label. The evaluator
counts those instances but does not score them, exactly as it does for OAML's
warm-up.

Seeding stays River's business — set it on the estimator (`seed=42`), not on
the adapter.

### Reaching the live estimator

```python
adapter = RiverClassifier(tree.HoeffdingTreeClassifier())
...
adapter.river_model      # the trained River object, for post-run inspection
```

## CapyMOA adapters

`CapyMOAClassifier` and `CapyMOARegressor` take a CapyMOA **class**, or its
name in `capymoa.classifier` / `capymoa.regressor`, plus any learner keyword
arguments.

```python
from capymoa.classifier import HoeffdingTree
from samlb.framework.adapters import CapyMOAClassifier, CapyMOARegressor

CapyMOAClassifier("HoeffdingTree", grace_period=50, seed=42)
CapyMOAClassifier("AdaptiveRandomForestClassifier", ensemble_size=10)
CapyMOAClassifier(HoeffdingTree, grace_period=50)          # the class works too
CapyMOARegressor("AdaptiveRandomForestRegressor", ensemble_size=10)
CapyMOARegressor("FIMTDD")
```

A class rather than an instance, because a CapyMOA learner is bound to a MOA
`Schema` at construction, and the schema is not known until the stream starts.
The adapter derives it from the first instance it sees, builds the learner at
that point, and converts each `{feature: value}` dict into the dense array MOA
expects. Feature order is fixed by that first instance; later instances are
projected onto it, and a missing feature reads as `0.0`.

An unknown learner name lists the alternatives:

```python
CapyMOAClassifier("NoSuchLearner")
# ValueError: capymoa.classifier has no learner 'NoSuchLearner'.
# Available: AdaptiveRandomForestClassifier, CSMOTE, ...
```

`seed` is forwarded as `random_seed`, but only to learners whose signature
accepts one.

### Labels

MOA works in class *indices*; the adapter keeps the mapping and hands back your
original labels — including non-integer ones.

```python
# labels discovered as they arrive, against reserved nominal slots
CapyMOAClassifier("NaiveBayes")

# label set known up front — the schema is exact
CapyMOAClassifier("NaiveBayes", classes=["no", "yes"])

# more slots for a many-class stream
CapyMOAClassifier("HoeffdingTree", max_classes=500)
```

With `classes` set, a label outside the declared set raises rather than being
silently absorbed. Without it, labels are discovered against `max_classes`
reserved slots (100 by default) and exceeding them raises with a message
telling you which knob to turn. Only MOA's per-class memory scales with
`max_classes`; unseen labels carry no counts, so they do not affect accuracy.

### Reaching the live learner

```python
adapter = CapyMOAClassifier("HoeffdingTree")
adapter.capymoa_model    # None until the first instance builds it
```

`reset()` drops the learner and the schema; both are rebuilt lazily on the next
instance.

## Version notes

The CapyMOA adapters are written against CapyMOA's modern `Schema.from_custom`
API and are tested against 0.14. The `capymoa` extra pins `>=0.14` for that
reason. The River adapters are tested against 0.23 and the extra allows
`>=0.21`.

## Adding another library

The adapters are ordinary `BaseStreamFramework` subclasses in
`samlb/framework/adapters/`. See [[Extending SAMLB]] for the pattern —
lazy import, `is_available()`, never train the caller's object.
