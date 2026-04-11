"""
samlb.evaluation.results
~~~~~~~~~~~~~~~~~~~~~~~~
RunResult — single result object for one (framework × dataset × run) evaluation.
"""
from __future__ import annotations

import dataclasses
import math
import statistics
from typing import Dict, List, Any, Optional


@dataclasses.dataclass
class RunResult:
    """All outcomes for a single framework × dataset × run prequential evaluation.

    Attributes
    ----------
    framework_name          : display name of the framework
    dataset_name            : dataset name (as passed to samlb.datasets.stream)
    task                    : "classification" or "regression"
    n_samples               : total instances processed
    metrics                 : cumulative final metric values
                              cls → accuracy, f1, precision, recall
                              reg → mae, rmse, r2
    windowed_metrics        : per-window snapshot lists (one entry per window)
    total_runtime_s         : wall-clock seconds for the full run
    runtime_per_instance_ms : sampled per-instance times in ms
    run_id                  : 0-based index of this run (for multi-run benchmarks)
    seed                    : random seed used for this run (None = not set)
    error                   : traceback string if the run raised an exception
    """
    framework_name          : str
    dataset_name            : str
    task                    : str
    n_samples               : int
    metrics                 : Dict[str, float]
    windowed_metrics        : Dict[str, List[float]]
    total_runtime_s         : float
    runtime_per_instance_ms : List[float]
    run_id                  : int = 0
    seed                    : Optional[int] = None
    error                   : Optional[str] = None

    def primary_metric(self) -> float:
        """Accuracy (cls) or R² (reg) — used for ranking in print_table."""
        if self.task == "classification":
            return self.metrics.get("accuracy", float("nan"))
        return self.metrics.get("r2", float("nan"))

    def as_dict(self) -> Dict[str, Any]:
        """Flat dict suitable for a CSV row or summary JSON (no windowed data)."""
        row: Dict[str, Any] = {
            "run_id":       self.run_id,
            "seed":         self.seed,
            "framework":    self.framework_name,
            "dataset":      self.dataset_name,
            "task":         self.task,
            "n_samples":    self.n_samples,
            "total_time_s": round(self.total_runtime_s, 4),
            "error":        self.error or "",
        }
        for k, v in self.metrics.items():
            row[k] = round(v, 6) if isinstance(v, float) else v
        return row

    def to_json_dict(self) -> Dict[str, Any]:
        """Full dict including windowed curves and per-instance runtimes.

        Suitable for individual per-(framework × dataset × run) JSON files.
        All floats are rounded to 6 decimal places.
        """
        def _round_list(lst: List[float]) -> List[float]:
            return [round(v, 6) for v in lst]

        return {
            "run_id":                  self.run_id,
            "seed":                    self.seed,
            "framework":               self.framework_name,
            "dataset":                 self.dataset_name,
            "task":                    self.task,
            "n_samples":               self.n_samples,
            "metrics":                 {k: round(v, 6) for k, v in self.metrics.items()},
            "windowed_metrics":        {k: _round_list(v)
                                        for k, v in self.windowed_metrics.items()},
            "total_time_s":            round(self.total_runtime_s, 4),
            "runtime_per_instance_ms": _round_list(self.runtime_per_instance_ms),
            "error":                   self.error,
        }


def aggregate_runs(runs: List[RunResult]) -> Dict[str, Any]:
    """Compute mean ± std statistics across multiple RunResult objects.

    All RunResult objects must share the same framework_name and dataset_name.

    Returns a dict suitable for writing directly as aggregate.json.

    Parameters
    ----------
    runs : list of RunResult
        All runs for a single (framework × dataset) combination.

    Returns
    -------
    dict with keys:
        framework, dataset, task, n_runs, n_successful_runs, n_failed_runs,
        seeds, n_samples, metrics, windowed_metrics, and optional errors.

    Notes
    -----
    Failed runs (where ``error`` is populated) are excluded from the metric
    and windowed aggregates.
    """
    if not runs:
        raise ValueError("runs must not be empty.")

    n_total = len(runs)
    valid_runs = [r for r in runs if not r.error]
    if not valid_runs:
        return {
            "framework": runs[0].framework_name,
            "dataset": runs[0].dataset_name,
            "task": runs[0].task,
            "n_runs": n_total,
            "n_successful_runs": 0,
            "n_failed_runs": n_total,
            "seeds": [r.seed for r in runs],
            "n_samples": 0,
            "metrics": {},
            "windowed_metrics": {},
            "errors": [r.error for r in runs if r.error],
        }

    n = len(valid_runs)

    def _stats(vals: List[float]) -> Dict[str, float]:
        finite_vals = [v for v in vals if math.isfinite(v)]
        if not finite_vals:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": round(statistics.mean(finite_vals), 6),
            "std":  round(statistics.stdev(finite_vals), 6) if len(finite_vals) > 1 else 0.0,
            "min":  round(min(finite_vals), 6),
            "max":  round(max(finite_vals), 6),
        }

    # ── per-metric aggregate ───────────────────────────────────────────────────
    metrics_agg: Dict[str, Any] = {}
    for k in valid_runs[0].metrics:
        vals = [r.metrics.get(k, float("nan")) for r in valid_runs]
        metrics_agg[k] = _stats(vals)

    # ── windowed metrics — element-wise mean and std ───────────────────────────
    windowed_agg: Dict[str, Any] = {}
    for k in valid_runs[0].windowed_metrics:
        curves = [r.windowed_metrics[k] for r in valid_runs]
        non_empty_curves = [c for c in curves if c]
        if not non_empty_curves:
            windowed_agg[k] = {"mean": [], "std": []}
            continue
        n_windows = min(len(c) for c in non_empty_curves)
        means = [round(statistics.mean(c[i] for c in non_empty_curves), 6)
                 for i in range(n_windows)]
        stds  = [
            round(statistics.stdev(c[i] for c in non_empty_curves), 6)
            if len(non_empty_curves) > 1 else 0.0
            for i in range(n_windows)
        ]
        windowed_agg[k] = {"mean": means, "std": stds}

    failed_errors = [r.error for r in runs if r.error]
    result: Dict[str, Any] = {
        "framework":       valid_runs[0].framework_name,
        "dataset":         valid_runs[0].dataset_name,
        "task":            valid_runs[0].task,
        "n_runs":          n_total,
        "n_successful_runs": n,
        "n_failed_runs":   n_total - n,
        "seeds":           [r.seed for r in runs],
        "n_samples":       valid_runs[0].n_samples,
        "metrics":         metrics_agg,
        "windowed_metrics": windowed_agg,
    }
    if failed_errors:
        result["errors"] = failed_errors

    return result
