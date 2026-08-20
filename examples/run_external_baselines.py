"""
Benchmark River and CapyMOA learners alongside SAMLB's own C++ baselines.

Both libraries are optional: whichever is installed contributes its models,
whichever is not is skipped with a note.

Usage
-----
    # defaults: classification, 1 run, no sample cap
    python3 run_external_baselines.py

    # custom options
    python3 run_external_baselines.py --task regression --datasets abalone
    python3 run_external_baselines.py --n_runs 5 --max_samples 20000

Arguments
---------
    --task          "classification" (default) or "regression".
    --n_runs        Number of independent runs per (model × dataset).  Default 1.
    --max_samples   Cap instances per dataset per run.  Default: no cap.
    --datasets      Space-separated dataset names.  Default: all for the task.
    --output_dir    Root directory for JSON/CSV output.  Default: results.

Install the backends with:
    pip install river
    pip install capymoa      # also needs a JVM
"""
import argparse

from samlb.benchmark import BenchmarkSuite
from samlb.framework.adapters import (
    CapyMOAClassifier,
    CapyMOARegressor,
    RiverClassifier,
    RiverRegressor,
)
from samlb.framework.base import ARFClassifier, ARFRegressor


def parse_args():
    p = argparse.ArgumentParser(description="SAMLB external-library baselines")
    p.add_argument("--task", choices=("classification", "regression"),
                   default="classification")
    p.add_argument("--n_runs", type=int, default=1, help="runs per (model × dataset)")
    p.add_argument("--max_samples", type=int, default=None, help="cap instances per run")
    p.add_argument("--datasets", nargs="+", default=None, help="dataset names")
    p.add_argument("--output_dir", type=str, default="results", help="output directory")
    return p.parse_args()


def build_classification_models(seed: int) -> dict:
    # SAMLB's own ARF is the reference point the external ones are read against.
    models = {"SAMLB-ARF": ARFClassifier(n_models=10, seed=seed)}

    if RiverClassifier.is_available():
        from river import forest, preprocessing, tree
        models["River-ARF"] = RiverClassifier(
            forest.ARFClassifier(n_models=10, seed=seed), name="River-ARF")
        models["River-HT"] = RiverClassifier(
            preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier(),
            name="River-HT")
    else:
        print("  skipping River models — `pip install river`", flush=True)

    if CapyMOAClassifier.is_available():
        models["MOA-ARF"] = CapyMOAClassifier(
            "AdaptiveRandomForestClassifier", ensemble_size=10, seed=seed,
            name="MOA-ARF")
        models["MOA-HT"] = CapyMOAClassifier("HoeffdingTree", seed=seed, name="MOA-HT")
    else:
        print("  skipping CapyMOA models — `pip install capymoa`", flush=True)

    return models


def build_regression_models(seed: int) -> dict:
    models = {"SAMLB-ARF": ARFRegressor(n_models=10, seed=seed)}

    if RiverRegressor.is_available():
        from river import forest, linear_model, preprocessing
        models["River-ARF"] = RiverRegressor(
            forest.ARFRegressor(n_models=10, seed=seed), name="River-ARF")
        models["River-LR"] = RiverRegressor(
            preprocessing.StandardScaler() | linear_model.LinearRegression(),
            name="River-LR")
    else:
        print("  skipping River models — `pip install river`", flush=True)

    if CapyMOARegressor.is_available():
        models["MOA-ARF"] = CapyMOARegressor(
            "AdaptiveRandomForestRegressor", ensemble_size=10, seed=seed,
            name="MOA-ARF")
        models["MOA-FIMTDD"] = CapyMOARegressor("FIMTDD", seed=seed, name="MOA-FIMTDD")
    else:
        print("  skipping CapyMOA models — `pip install capymoa`", flush=True)

    return models


def main():
    args = parse_args()

    print(f"SAMLB External Baselines — {args.task}", flush=True)
    models = (
        build_classification_models(seed=42) if args.task == "classification"
        else build_regression_models(seed=42)
    )
    print(f"  models:      {list(models)}", flush=True)
    print(f"  datasets:    {args.datasets or 'all'}", flush=True)
    print(f"  n_runs:      {args.n_runs}", flush=True)
    print(flush=True)

    suite = BenchmarkSuite(
        models=models,
        datasets=args.datasets,
        task=args.task,
        n_runs=args.n_runs,
        max_samples=args.max_samples,
        normalize=(args.task == "regression"),
        verbose=True,
    )
    suite.run()
    suite.print_table()

    cap_tag = f"_cap{args.max_samples}" if args.max_samples else ""
    suite.to_json(args.output_dir)
    suite.to_csv(f"{args.output_dir}/external_{args.task}_{args.n_runs}runs{cap_tag}.csv")


if __name__ == "__main__":
    main()
