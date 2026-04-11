"""
samlb.datasets
~~~~~~~~~~~~~~
Unified dataset loader for all SAMLB benchmark datasets (NPZ format).

Datasets are auto-downloaded from GitHub on first use and cached locally
in ``~/.samlb/datasets/``.

All datasets follow a consistent schema:
  X            float32  (n_samples, n_features)  — feature matrix
  y            float32  or int32 (n_samples,)    — labels / targets
  feature_names str[]   (n_features,)
  target_name  str      scalar

Usage
-----
    from samlb.datasets import load, list_datasets, stream

    # Load as numpy arrays
    X, y, meta = load("electricity", task="classification")

    # Stream as (x_dict, y) tuples — matches predict_one / learn_one API
    for x, y in stream("electricity", task="classification"):
        pred = model.predict_one(x)
        model.learn_one(x, y)
"""
from __future__ import annotations

import os
import sys
import urllib.request
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any

# ── dataset registry ─────────────────────────────────────────────────────────

_DATASETS = {
    "classification": [
        "adult", "covertype", "credit_card", "electricity", "insects",
        "movingRBF", "moving_squares", "new_airlines", "nomao", "poker_hand",
        "sea_high_abrupt_drift", "shuttle", "synth_RandomRBFDrift",
        "synth_agrawal", "vehicle_sensIT",
    ],
    "regression": [
        "FriedmanGra", "House8L", "MetroTraffic", "ailerons", "bike",
        "california_housing", "cps88wages", "diamonds", "elevators", "fifa",
        "fried", "hyperA", "kings_county", "superconductivity", "wave_energy",
    ],
}
_DATASETS["clustering"] = _DATASETS["classification"]

_BASE_URL = (
    "https://github.com/TechyNilesh/samlb/raw/main/"
    "samlb/datasets/{task}/{name}.npz"
)

# Local cache directory
_CACHE_DIR = Path.home() / ".samlb" / "datasets"

# Package-bundled directory (used in development / editable installs)
_PKG_DIR = Path(__file__).parent


# ── public API ────────────────────────────────────────────────────────────────

def list_datasets(task: str = "classification") -> list[str]:
    """Return sorted list of available dataset names for a task."""
    _validate_task(task)
    base_task = "classification" if task == "clustering" else task
    return sorted(_DATASETS[base_task])


def load(name: str, task: str = "classification",
         max_samples: int | None = None) -> tuple:
    """
    Load a dataset by name. Downloads from GitHub on first use.

    Parameters
    ----------
    name        : dataset name (without .npz), e.g. "electricity"
    task        : "classification" | "regression" | "clustering"
    max_samples : optional row limit

    Returns
    -------
    X            : np.ndarray  float32  (n_samples, n_features)
    y            : np.ndarray  int32 (classification) or float32 (regression)
    meta         : dict with keys feature_names, target_name, task, n_samples,
                   n_features, n_classes (classification only)
    """
    path = _resolve(name, task)
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    feature_names = list(data["feature_names"])
    target_name   = str(data["target_name"])

    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]

    if task in ("classification", "clustering"):
        y = y.astype("int32")

    meta: Dict[str, Any] = {
        "name":          name,
        "task":          task,
        "feature_names": feature_names,
        "target_name":   target_name,
        "n_samples":     X.shape[0],
        "n_features":    X.shape[1],
    }
    if task in ("classification", "clustering"):
        meta["n_classes"] = int(len(np.unique(y)))

    return X, y, meta


def stream(name: str, task: str = "classification",
           max_samples: int | None = None,
           normalize: bool = False
           ) -> Iterator[Tuple[Dict[str, float], Any]]:
    """
    Stream a dataset as (x_dict, y) tuples.
    Matches the predict_one / learn_one interface.

    Parameters
    ----------
    name        : dataset name
    task        : "classification" | "regression" | "clustering"
    max_samples : optional row limit
    normalize   : if True, min-max scale features online while streaming
                  (uses observed range so far; avoids future-data leakage)

    Yields
    ------
    (x_dict, y) where x_dict maps feature name → float value
    """
    X, y, meta = load(name, task=task, max_samples=max_samples)
    feat_names  = meta["feature_names"]

    x_min = None
    x_max = None

    for i in range(len(X)):
        x_row = X[i]
        if normalize:
            if x_min is None:
                x_min = x_row.copy()
                x_max = x_row.copy()
            else:
                x_min = np.minimum(x_min, x_row)
                x_max = np.maximum(x_max, x_row)
            x_range = x_max - x_min
            x_range[x_range == 0] = 1.0
            x_row = (x_row - x_min) / x_range

        x_dict = {feat_names[j]: float(x_row[j]) for j in range(len(feat_names))}
        label  = int(y[i]) if task in ("classification", "clustering") else float(y[i])
        yield x_dict, label


# ── helpers ───────────────────────────────────────────────────────────────────

def _validate_task(task: str) -> None:
    if task not in _DATASETS:
        raise ValueError(f"Unknown task {task!r}. Choose from {list(_DATASETS)}")


def _resolve(name: str, task: str) -> str:
    """Find dataset file: check package dir first, then cache, then download."""
    _validate_task(task)
    base_task = "classification" if task == "clustering" else task

    if name not in _DATASETS[base_task]:
        raise FileNotFoundError(
            f"Dataset {name!r} not found for task={task!r}.\n"
            f"Available: {list_datasets(task)}"
        )

    # 1. Check package-bundled directory (editable / dev install)
    pkg_path = _PKG_DIR / base_task / f"{name}.npz"
    if pkg_path.exists():
        return str(pkg_path)

    # 2. Check local cache
    cache_path = _CACHE_DIR / base_task / f"{name}.npz"
    if cache_path.exists():
        return str(cache_path)

    # 3. Download from GitHub
    _download(name, base_task, cache_path)
    return str(cache_path)


def _download(name: str, task: str, dest: Path) -> None:
    """Download a single dataset from GitHub to the local cache."""
    url = _BASE_URL.format(task=task, name=name)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name} ({task})... ", end="", flush=True)
    try:
        tmp = str(dest) + ".tmp"
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        os.replace(tmp, dest)
        print(" done.", flush=True)
    except Exception as e:
        # Clean up partial download
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"Failed to download dataset {name!r} from {url}.\n"
            f"Error: {e}\n"
            f"You can manually download it and place it at: {dest}"
        ) from e


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    """Simple progress indicator for urllib downloads."""
    if total_size > 0:
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading... {pct}% ({mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()
