"""
samlb.benchmark.suite
~~~~~~~~~~~~~~~~~~~~~
BenchmarkSuite — orchestrates multiple frameworks × multiple datasets × multiple runs,
collects RunResult objects, and renders results as a table, CSV, or JSON.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from samlb.datasets import list_datasets as _list_datasets
from samlb.evaluation.evaluator import PrequentialEvaluator
from samlb.evaluation.results import RunResult, aggregate_runs


class BenchmarkSuite:
    """Run a cartesian product of frameworks × datasets × seeds and collect results.

    Parameters
    ----------
    models : dict[str, model]
        ``{display_name: model_instance}``.  ``model.reset()`` is called
        before every (framework, dataset, seed) run so state never leaks.
    datasets : list[str] | None
        Dataset names for ``samlb.datasets.stream()``.
        ``None`` → all datasets available for the task.
    task : str
        "classification" or "regression".
    n_runs : int
        Number of independent runs per (framework × dataset) combination.
        Each run uses a different random seed.  Default 1.
    seeds : list[int] | None
        Explicit seed list — length must equal ``n_runs``.
        ``None`` → auto-generate ``[0, 1, ..., n_runs-1]``.
        Ignored when ``n_runs=1`` and not provided (seed not touched).
    window_size : int
        Instances per evaluation window (default 1000).
    max_samples : int | None
        Optional row cap — useful for quick trial runs.
    normalize : bool
        Min-max scale features before streaming (recommended for regressors).
    verbose : bool
        Print progress per framework/dataset/run combination.

    Example — single run
    --------------------
        from samlb.benchmark import BenchmarkSuite

        suite = BenchmarkSuite(
            models={"ModelA": model_a, "ModelB": model_b},
            datasets=["electricity"],
            task="classification",
        )
        suite.run()
        suite.print_table()
        suite.to_json("results")

    Example — multiple runs with auto seeds
    ----------------------------------------
        suite = BenchmarkSuite(
            models={"ASML": asml, "OAML": oaml},
            datasets=["electricity"],
            task="classification",
            n_runs=5,           # seeds 0,1,2,3,4 auto-generated
        )
        suite.run()
        suite.print_table()
        suite.to_json("results")   # writes run_00.json...run_04.json + aggregate.json

    Example — explicit seeds
    ------------------------
        suite = BenchmarkSuite(
            models={"ASML": asml},
            datasets=["electricity"],
            task="classification",
            n_runs=3,
            seeds=[42, 123, 999],
        )
    """

    def __init__(
        self,
        models: Dict[str, Any],
        datasets: Optional[List[str]],
        task: str,
        n_runs: int = 1,
        seeds: Optional[List[int]] = None,
        window_size: int = 1000,
        max_samples: Optional[int] = None,
        normalize: bool = False,
        verbose: bool = True,
    ):
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got {task!r}"
            )
        if not models:
            raise ValueError("models dict must not be empty.")
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}.")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        if max_samples is not None and max_samples < 1:
            raise ValueError(f"max_samples must be >= 1 when set, got {max_samples}.")
        if seeds is not None and len(seeds) != n_runs:
            raise ValueError(
                f"len(seeds)={len(seeds)} must equal n_runs={n_runs}."
            )

        self.models      = models
        self.task        = task
        self.n_runs      = n_runs
        self.window_size = window_size
        self.max_samples = max_samples
        self.normalize   = normalize
        self.verbose     = verbose
        self.datasets    = list(_list_datasets(task) if datasets is None else datasets)

        # resolve seeds
        if seeds is not None:
            self.seeds: List[Optional[int]] = list(seeds)
        elif n_runs > 1:
            self.seeds = list(range(n_runs))
        else:
            self.seeds = [None]   # single run — don't touch model.seed

        if not self.datasets:
            raise ValueError(f"No datasets found for task={task!r}.")

        self._results: List[RunResult] = []

    # ── main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[RunResult]:
        """Execute all (framework × dataset × seed) runs sequentially.

        Each model's seed is updated and reset() is called before every run.
        Exceptions are caught per-run; the suite continues and the
        RunResult carries the traceback in its ``error`` field.

        Parameters
        ----------
        progress_callback : callable | None
            Optional callback invoked before and after each
            (framework × dataset × run) task. The callback receives a dict with
            an ``event`` key set to ``"task_started"`` or ``"task_finished"``
            plus run metadata. Finished events also carry the ``RunResult`` as
            ``event["result"]``.

        Returns the list of RunResult objects (also stored in self._results).
        """
        evaluator = PrequentialEvaluator(
            task=self.task,
            window_size=self.window_size,
            max_samples=self.max_samples,
            normalize=self.normalize,
            verbose=False,
        )

        total = len(self.seeds) * len(self.models) * len(self.datasets)
        idx   = 0

        def _emit_progress(event: Dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(event)
            except Exception as exc:
                if self.verbose:
                    print(f"  WARNING: progress callback raised: {exc}", flush=True)

        for run_id, seed in enumerate(self.seeds):
            if self.verbose and self.n_runs > 1:
                seed_label = f"seed={seed}" if seed is not None else "no seed"
                print(f"\n{'─'*60}")
                print(f"  Run {run_id + 1}/{self.n_runs}  ({seed_label})")
                print(f"{'─'*60}")

            for fw_name, model in self.models.items():
                for ds_name in self.datasets:
                    idx += 1
                    _emit_progress(
                        {
                            "event": "task_started",
                            "task_index": idx,
                            "task_total": total,
                            "run_id": run_id,
                            "seed": seed,
                            "framework_name": fw_name,
                            "dataset_name": ds_name,
                        }
                    )
                    if self.verbose:
                        run_label = (
                            f"  run {run_id+1}/{self.n_runs}  "
                            if self.n_runs > 1 else "  "
                        )
                        print(
                            f"\n[{idx}/{total}]{run_label}{fw_name}  ×  {ds_name}",
                            flush=True,
                        )

                    # update seed then reset
                    if seed is not None and hasattr(model, "seed"):
                        model.seed = seed
                    try:
                        model.reset()
                    except Exception as exc:
                        print(f"  WARNING: {fw_name}.reset() raised: {exc}")

                    result = evaluator.run(
                        model=model,
                        dataset_name=ds_name,
                        framework_name=fw_name,
                    )
                    # stamp run metadata onto the result
                    result = dataclasses.replace(result, run_id=run_id, seed=seed)
                    self._results.append(result)
                    _emit_progress(
                        {
                            "event": "task_finished",
                            "task_index": idx,
                            "task_total": total,
                            "run_id": run_id,
                            "seed": seed,
                            "framework_name": fw_name,
                            "dataset_name": ds_name,
                            "result": result,
                        }
                    )

                    if result.error:
                        print(f"  ✗ ERROR:\n{result.error}", flush=True)
                    elif self.verbose:
                        pm_label = "acc" if self.task == "classification" else "r2"
                        pm_val   = result.primary_metric()
                        print(
                            f"  n={result.n_samples:,}  "
                            f"{pm_label}={pm_val:.4f}  "
                            f"time={result.total_runtime_s:.1f}s",
                            flush=True,
                        )

        return self._results

    # ── output ────────────────────────────────────────────────────────────────

    def print_table(self) -> None:
        """Print a fixed-width ASCII results table to stdout.

        Single run  → no Run/Seed columns.
        Multi-run   → Run and Seed columns added; a mean row per group.
        """
        if not self._results:
            print("No results yet — call suite.run() first.")
            return

        metric_keys = (
            ["accuracy", "f1", "precision", "recall"]
            if self.task == "classification"
            else ["r2", "mae", "rmse"]
        )

        multi = self.n_runs > 1

        fw_w = max(len("Framework"), max(len(r.framework_name) for r in self._results))
        ds_w = max(len("Dataset"),   max(len(r.dataset_name)   for r in self._results))
        n_w  = 9
        m_w  = 10
        t_w  = 9
        SEP  = "  "

        def _header() -> str:
            parts = [f"{'Framework':<{fw_w}}", f"{'Dataset':<{ds_w}}"]
            if multi:
                parts += [f"{'Run':>4}", f"{'Seed':>6}"]
            parts += [f"{'N':>{n_w}}"]
            parts += [f"{k:>{m_w}}" for k in metric_keys]
            parts += [f"{'time_s':>{t_w}}"]
            return SEP.join(parts)

        def _row(r: RunResult, run_label: str = "", seed_label: str = "") -> str:
            parts = [
                f"{r.framework_name:<{fw_w}}",
                f"{r.dataset_name:<{ds_w}}",
            ]
            if multi:
                parts += [f"{run_label:>4}", f"{seed_label:>6}"]
            parts += [f"{r.n_samples:>{n_w},}"]
            if r.error:
                parts += [f"{'ERROR':>{m_w}}"] * len(metric_keys)
            else:
                parts += [
                    f"{r.metrics.get(k, float('nan')):>{m_w}.4f}"
                    for k in metric_keys
                ]
            parts += [f"{r.total_runtime_s:>{t_w}.2f}"]
            return SEP.join(parts)

        header = _header()
        print()
        print(header)
        print("─" * len(header))

        if not multi:
            for r in self._results:
                print(_row(r))
        else:
            import statistics as _stats

            groups: Dict[tuple, List[RunResult]] = defaultdict(list)
            for r in self._results:
                groups[(r.framework_name, r.dataset_name)].append(r)

            for (fw, ds), runs in groups.items():
                for r in sorted(runs, key=lambda x: x.run_id):
                    seed_lbl = str(r.seed) if r.seed is not None else "-"
                    print(_row(r, run_label=str(r.run_id), seed_label=seed_lbl))

                # mean summary row
                valid = [r for r in runs if not r.error]
                if valid and len(valid) > 1:
                    mean_metrics = {
                        k: _stats.mean(r.metrics.get(k, float("nan")) for r in valid)
                        for k in metric_keys
                    }
                    mean_time = _stats.mean(r.total_runtime_s for r in valid)
                    mean_r    = dataclasses.replace(
                        valid[0],
                        metrics=mean_metrics,
                        total_runtime_s=mean_time,
                    )
                    print(_row(mean_r, run_label="avg", seed_label="-"))
                print()

        print()

    def to_csv(self, path: str) -> None:
        """Write flat results to a CSV file (one row per run).

        Windowed metrics are NOT written here — access them via
        ``suite.results[i].windowed_metrics`` for plotting.

        Parameters
        ----------
        path : str
            Output file path.  Parent directory is created if it doesn't exist.
        """
        if not self._results:
            raise RuntimeError("No results to write — call suite.run() first.")

        metric_keys = (
            ["accuracy", "f1", "precision", "recall"]
            if self.task == "classification"
            else ["r2", "mae", "rmse"]
        )
        fieldnames = (
            ["run_id", "seed", "framework", "dataset", "task", "n_samples"]
            + metric_keys
            + ["total_time_s", "error"]
        )

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in self._results:
                row: Dict[str, Any] = {
                    "run_id":       r.run_id,
                    "seed":         r.seed if r.seed is not None else "",
                    "framework":    r.framework_name,
                    "dataset":      r.dataset_name,
                    "task":         r.task,
                    "n_samples":    r.n_samples,
                    "total_time_s": round(r.total_runtime_s, 4),
                    "error":        r.error or "",
                }
                for k in metric_keys:
                    row[k] = (
                        round(r.metrics.get(k, float("nan")), 6)
                        if not r.error else ""
                    )
                writer.writerow(row)

        print(f"Saved {len(self._results)} rows → {path}")

    def to_json(self, output_dir: str) -> None:
        """Save results as JSON files.

        Single run (n_runs=1)
        ~~~~~~~~~~~~~~~~~~~~~
        output_dir/
          {task}/
            {dataset}/
              {framework}.json    ← full result: metrics + windowed + runtime
            summary.json          ← flat list of all runs

        Multiple runs (n_runs > 1)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        output_dir/
          {task}/
            {dataset}/
              {framework}/
                run_00.json       ← run 0 full result
                run_01.json       ← run 1 full result
                ...
                aggregate.json    ← mean ± std across all runs
            summary.json          ← flat list of every individual run

        Parameters
        ----------
        output_dir : str
            Root directory.  Sub-directories are created automatically.
        """
        if not self._results:
            raise RuntimeError("No results to write — call suite.run() first.")

        task_dir = os.path.join(output_dir, self.task)

        def _safe(s: str) -> str:
            return s.replace(os.sep, "_").replace(" ", "_")

        n_written = 0

        if self.n_runs == 1:
            # single-run layout: task/dataset/framework.json
            for r in self._results:
                ds_dir = os.path.join(task_dir, r.dataset_name)
                os.makedirs(ds_dir, exist_ok=True)
                fpath = os.path.join(ds_dir, _safe(r.framework_name) + ".json")
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(r.to_json_dict(), f, indent=2)
                n_written += 1
        else:
            # multi-run layout: task/dataset/framework/run_NN.json + aggregate.json
            groups: Dict[tuple, List[RunResult]] = defaultdict(list)
            for r in self._results:
                groups[(r.dataset_name, r.framework_name)].append(r)

            for (ds_name, fw_name), runs in groups.items():
                fw_dir = os.path.join(task_dir, ds_name, _safe(fw_name))
                os.makedirs(fw_dir, exist_ok=True)

                for r in sorted(runs, key=lambda x: x.run_id):
                    fpath = os.path.join(fw_dir, f"run_{r.run_id:02d}.json")
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(r.to_json_dict(), f, indent=2)
                    n_written += 1

                agg_path = os.path.join(fw_dir, "aggregate.json")
                with open(agg_path, "w", encoding="utf-8") as f:
                    json.dump(aggregate_runs(runs), f, indent=2)

        # summary.json — flat list of every individual run
        os.makedirs(task_dir, exist_ok=True)
        summary_path = os.path.join(task_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump([r.as_dict() for r in self._results], f, indent=2)

        print(f"Saved {n_written} result file(s) + summary → {task_dir}/")

    def load_results(self, results: List[RunResult]) -> None:
        """Replace in-memory results with externally produced RunResult items.

        Useful for parallel execution patterns where workers evaluate subsets
        and the caller later loads merged outputs into a single suite object.
        """
        self._results = []
        self.merge_results(results)

    def merge_results(self, results: List[RunResult]) -> None:
        """Append externally produced RunResult items after validation."""
        if not isinstance(results, list):
            raise TypeError(f"results must be a list[RunResult], got {type(results).__name__}.")

        for r in results:
            if not isinstance(r, RunResult):
                raise TypeError(f"All items must be RunResult, got {type(r).__name__}.")
            if r.task != self.task:
                raise ValueError(
                    f"RunResult task {r.task!r} does not match suite task {self.task!r}."
                )

        self._results.extend(results)

    @property
    def results(self) -> List[RunResult]:
        """Read-only list of RunResult objects collected so far."""
        return list(self._results)
