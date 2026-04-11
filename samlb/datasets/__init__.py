"""
samlb.datasets
~~~~~~~~~~~~~~
Unified dataset loader for all SAMLB benchmark datasets (NPZ format).

All datasets are stored as compressed NPZ files with a consistent schema:
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
import numpy as np
from typing import Iterator, Tuple, Dict, Any

_ROOT = os.path.dirname(__file__)
_CLS_DIR = os.path.join(_ROOT, "classification")
_REG_DIR = os.path.join(_ROOT, "regression")

_TASK_DIRS = {
    "classification": _CLS_DIR,
    "clustering":     _CLS_DIR,   # same datasets as classification
    "regression":     _REG_DIR,
}


# ── public API ────────────────────────────────────────────────────────────────

def list_datasets(task: str = "classification") -> list[str]:
    """Return sorted list of available dataset names for a task."""
    d = _task_dir(task)
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npz"))


def load(name: str, task: str = "classification",
         max_samples: int | None = None) -> tuple:
    """
    Load a dataset by name.

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
            # Update per-feature bounds using the current sample, then scale.
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

def _task_dir(task: str) -> str:
    if task not in _TASK_DIRS:
        raise ValueError(f"Unknown task {task!r}. Choose from {list(_TASK_DIRS)}")
    return _TASK_DIRS[task]


def _resolve(name: str, task: str) -> str:
    d    = _task_dir(task)
    path = os.path.join(d, f"{name}.npz")
    if not os.path.exists(path):
        available = list_datasets(task)
        raise FileNotFoundError(
            f"Dataset {name!r} not found for task={task!r}.\n"
            f"Available: {available}"
        )
    return path
