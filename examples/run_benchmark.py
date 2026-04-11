"""
Classification benchmark — all 4 frameworks × all 33 datasets.

Usage
-----
    # defaults: 10 runs, no sample cap
    python3 run_benchmark.py

    # custom options
    python3 run_benchmark.py --n_runs 5 --max_samples 50000
    python3 run_benchmark.py --n_runs 1 --max_samples 10000 --datasets electricity noaa_weather
    python3 run_benchmark.py --n_runs 10 --max_samples 100000 --output_dir my_results
    python3 run_benchmark.py --n_runs 100 --parallel --cpu_utilization 0.8

Arguments
---------
    --n_runs        Number of independent runs per (framework × dataset).  Default 10.
    --max_samples   Cap instances per dataset per run.  Default: no cap (full dataset).
    --datasets      Space-separated list of dataset names.  Default: all 33.
    --output_dir    Root directory for JSON/CSV output.  Default: results.
    --seeds         Space-separated explicit seed list (length must equal n_runs).
    --parallel      Run seeds in parallel processes.
    --cpu_utilization  Fraction of detected CPU cores to use when --parallel is set.
                       Default 0.8 (80%).
    --max_workers   Optional hard cap for parallel workers.
"""
import argparse
import dataclasses
import math
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager
from queue import Empty
from typing import Any, Dict, Optional, Tuple

from samlb.benchmark import BenchmarkSuite
from samlb.framework.classification.asml      import AutoStreamClassifier, default_config_dict as asml_cfg
from samlb.framework.classification.autoclass import AutoClass, default_config_dict as ac_cfg
from samlb.framework.classification.eaml      import EvolutionaryBaggingClassifier, EAML_CLF_PARAM_GRID
from samlb.framework.classification.oaml      import OAMLClassifier, OAML_SCALERS, OAML_CLASSIFIERS


def parse_args():
    p = argparse.ArgumentParser(description="SAMLB classification benchmark")
    p.add_argument("--n_runs",      type=int,   default=10,   help="independent runs per (framework × dataset)")
    p.add_argument("--max_samples", type=int,   default=None, help="cap instances per dataset per run")
    p.add_argument("--datasets",    nargs="+",  default=None, help="dataset names (default: all 33)")
    p.add_argument("--output_dir",  type=str,   default="results", help="root output directory")
    p.add_argument("--seeds",       nargs="+",  type=int, default=None, help="explicit seed list")
    p.add_argument("--parallel", action="store_true", help="run seeds in parallel worker processes")
    p.add_argument("--cpu_utilization", type=float, default=0.8, help="CPU fraction for parallel mode (0,1]")
    p.add_argument("--max_workers", type=int, default=None, help="optional max parallel workers")
    return p.parse_args()


def _build_models(seed: int):
    return {
        "ASML":      AutoStreamClassifier(config_dict=asml_cfg, seed=seed),
        "AutoClass": AutoClass(config_dict=ac_cfg, seed=seed),
        "EvoAutoML": EvolutionaryBaggingClassifier(param_grid=EAML_CLF_PARAM_GRID, seed=seed),
        "OAML":      OAMLClassifier(scalers=OAML_SCALERS, classifiers=OAML_CLASSIFIERS, seed=seed),
    }


def _resolve_seeds(n_runs: int, seeds_arg):
    if seeds_arg is None:
        return list(range(n_runs))
    if len(seeds_arg) != n_runs:
        raise ValueError(f"len(seeds)={len(seeds_arg)} must equal n_runs={n_runs}.")
    return list(seeds_arg)


def _auto_workers(cpu_utilization: float, max_workers: Optional[int]) -> Tuple[int, int]:
    if not (0 < cpu_utilization <= 1):
        raise ValueError(f"cpu_utilization must be in (0,1], got {cpu_utilization}.")
    cpu_total = os.cpu_count() or 1
    workers = max(1, math.floor(cpu_total * cpu_utilization))
    if max_workers is not None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >=1, got {max_workers}.")
        workers = min(workers, max_workers)
    return cpu_total, workers


def _queue_progress_event(progress_queue, run_id: int, seed: int, event: Dict[str, Any]):
    if progress_queue is None:
        return

    payload: Dict[str, Any] = {
        "event": event["event"],
        "run_id": run_id,
        "seed": seed,
        "task_index": event["task_index"],
        "task_total": event["task_total"],
        "framework_name": event["framework_name"],
        "dataset_name": event["dataset_name"],
    }
    if event["event"] == "task_finished":
        result = event["result"]
        payload.update(
            {
                "status": "error" if result.error else "ok",
                "metric": result.primary_metric(),
                "n_samples": result.n_samples,
                "total_runtime_s": result.total_runtime_s,
            }
        )

    progress_queue.put(payload)


def _print_parallel_event(event: Dict[str, Any]):
    prefix = (
        f"  [seed {event['seed']} | "
        f"{event['task_index']}/{event['task_total']}]"
    )
    target = f"{event['framework_name']} × {event['dataset_name']}"

    if event["event"] == "task_started":
        print(f"{prefix} start {target}", flush=True)
        return

    if event["status"] == "error":
        print(
            f"{prefix} error {target}  "
            f"time={event['total_runtime_s']:.2f}s",
            flush=True,
        )
        return

    print(
        f"{prefix} done  {target}  "
        f"acc={event['metric']:.4f}  "
        f"n={event['n_samples']:,}  "
        f"time={event['total_runtime_s']:.2f}s",
        flush=True,
    )


def _drain_progress_queue(progress_queue) -> None:
    while True:
        try:
            event = progress_queue.get_nowait()
        except Empty:
            break
        _print_parallel_event(event)


def _run_single_seed(
    run_id: int,
    seed: int,
    datasets,
    max_samples: Optional[int],
    progress_queue=None,
):
    suite = BenchmarkSuite(
        models=_build_models(seed=seed),
        datasets=datasets,
        task="classification",
        n_runs=1,
        seeds=[seed],
        max_samples=max_samples,
        verbose=False,
    )
    progress_callback = None
    if progress_queue is not None:
        def progress_callback(event):
            _queue_progress_event(progress_queue, run_id, seed, event)

    results = suite.run(progress_callback=progress_callback)
    return [dataclasses.replace(r, run_id=run_id, seed=seed) for r in results]


def main():
    args = parse_args()
    seeds = _resolve_seeds(args.n_runs, args.seeds)

    print("SAMLB Classification Benchmark", flush=True)
    print(f"  n_runs:      {args.n_runs}", flush=True)
    print(
        f"  max_samples: {args.max_samples if args.max_samples else 'no cap (full dataset)'}",
        flush=True,
    )
    print(f"  datasets:    {'all 33' if args.datasets is None else args.datasets}", flush=True)
    print(f"  output_dir:  {args.output_dir}", flush=True)
    print(f"  seeds:       {seeds}", flush=True)
    print(f"  parallel:    {args.parallel}", flush=True)
    if args.parallel:
        cpu_total, workers = _auto_workers(args.cpu_utilization, args.max_workers)
        print(f"  cpu_total:   {cpu_total}", flush=True)
        print(f"  workers:     {workers} ({args.cpu_utilization*100:.0f}% target)", flush=True)
    print(flush=True)

    if args.parallel:
        _, workers = _auto_workers(args.cpu_utilization, args.max_workers)
        all_results = []
        with Manager() as manager:
            progress_queue = manager.Queue()
            with ProcessPoolExecutor(max_workers=workers) as ex:
                future_to_seed = {
                    ex.submit(
                        _run_single_seed,
                        run_id,
                        seed,
                        args.datasets,
                        args.max_samples,
                        progress_queue,
                    ): (run_id, seed)
                    for run_id, seed in enumerate(seeds)
                }
                pending = set(future_to_seed)
                done_count = 0

                while pending:
                    _drain_progress_queue(progress_queue)
                    done, pending = wait(
                        pending,
                        timeout=0.2,
                        return_when=FIRST_COMPLETED,
                    )
                    for fut in done:
                        run_id, seed = future_to_seed[fut]
                        results = fut.result()
                        _drain_progress_queue(progress_queue)
                        all_results.extend(results)
                        done_count += 1
                        ok = [r for r in results if not r.error]
                        err = [r for r in results if r.error]
                        if ok:
                            avg_acc = sum(r.primary_metric() for r in ok) / len(ok)
                            print(
                                f"  [seed {seed}] done ({done_count}/{len(seeds)})  "
                                f"{len(ok)} ok, {len(err)} errors  "
                                f"avg_acc={avg_acc:.4f}",
                                flush=True,
                            )
                        else:
                            print(
                                f"  [seed {seed}] done ({done_count}/{len(seeds)})  "
                                f"ALL {len(err)} ERRORED",
                                flush=True,
                            )

                _drain_progress_queue(progress_queue)

        all_results.sort(key=lambda r: (r.run_id, r.framework_name, r.dataset_name))
        suite = BenchmarkSuite(
            models=_build_models(seed=seeds[0] if seeds else 0),
            datasets=args.datasets,
            task="classification",
            n_runs=args.n_runs,
            seeds=seeds,
            max_samples=args.max_samples,
            verbose=False,
        )
        suite.load_results(all_results)
    else:
        suite = BenchmarkSuite(
            models=_build_models(seed=42),
            datasets=args.datasets,
            task="classification",
            n_runs=args.n_runs,
            seeds=seeds,
            max_samples=args.max_samples,
            verbose=True,
        )
        suite.run()

    suite.print_table()

    cap_tag = f"_cap{args.max_samples}" if args.max_samples else ""
    suite.to_json(args.output_dir)
    suite.to_csv(f"{args.output_dir}/classification_{args.n_runs}runs{cap_tag}.csv")


if __name__ == "__main__":
    main()
