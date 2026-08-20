# Installation

SAMLB is a Python package with a compiled C++ core. A wheel installs in
seconds; a source install compiles the extension.

**Requirements:** Python >= 3.9, and for a source install a C++17 compiler and
CMake.

## From PyPI

```bash
pip install samlb
```

## From source

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

## Optional backends

None of these are needed for the core benchmark; install only what you use.

```bash
pip install "samlb[river]"     # River algorithms, via the River adapters
pip install "samlb[capymoa]"   # CapyMOA / MOA algorithms — also needs a JVM
pip install "samlb[vw]"        # Vowpal Wabbit, for the ChaCha regressor
```

Each optional integration is imported lazily and exposes `is_available()`, so
code can degrade cleanly:

```python
from samlb.framework.adapters import CapyMOAClassifier, RiverClassifier
from samlb.framework.regression.chacha import ChaChaRegressor

print(RiverClassifier.is_available())    # river importable?
print(CapyMOAClassifier.is_available())  # capymoa importable, JVM usable?
print(ChaChaRegressor.is_available())    # flaml + vowpalwabbit?
```

CapyMOA runs MOA on a JVM through JPype. Install a JDK (17 or newer) and make
sure `java -version` works before `pip install capymoa`.

## Verifying the install

```python
import samlb
from samlb.datasets import list_datasets, stream
from samlb.framework.base import HoeffdingTreeClassifier

print(samlb.__version__)
print(len(list_datasets("classification")), "classification datasets")

model = HoeffdingTreeClassifier()
hits = n = 0
for x, y in stream("electricity", max_samples=2000):
    hits += model.predict_one(x) == y
    n += 1
    model.learn_one(x, y)
print(f"accuracy {hits / n:.3f}")
```

## Rebuilding after editing C++

An editable install does not recompile when you change a file under `_cpp/`.
Re-run `pip install -e .`, or build in place for a faster loop:

```bash
cmake -S . -B build/local -DCMAKE_BUILD_TYPE=Release
cmake --build build/local -j
cp build/local/_samlb_core.*.so samlb/
```

## Troubleshooting

**`AttributeError: module 'samlb._samlb_core' has no attribute ...`**
The compiled extension is stale — it predates the Python code you are running.
Rebuild as above. This is by far the most common source of confusing errors in
a source checkout.

**`ModuleNotFoundError: No module named 'samlb._samlb_core'`**
The extension was never built. Reinstall with `pip install -e .` and read the
build log; a missing compiler or CMake shows up there.

**CapyMOA import hangs or raises a JVM error**
Check `java -version`. CapyMOA starts a JVM at import time, and a missing or
mismatched JDK surfaces as an import failure rather than a clean message.

**Dataset download fails**
Datasets are fetched from GitHub on first use and cached. Behind a proxy, set
`HTTPS_PROXY`, or pre-place the `.npz` files under
`samlb/datasets/classification/` and `samlb/datasets/regression/`.
