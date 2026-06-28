"""
Random-search baseline benchmark — classification and regression.

Maintains the full shared learner pool warm and randomly selects one pipeline
per exploration window to serve predictions.  Produces JSON in the same layout
as run_benchmark.py / run_regression.py so results slot alongside the other
frameworks.

    python3 run_random_search.py --task classification --n_runs 10
    python3 run_random_search.py --task regression     --n_runs 10
"""
import argparse

from samlb.benchmark import BenchmarkSuite
from samlb.framework.random_search import RandomSearch

from samlb.framework.classification.shared_config import (
    SHARED_PREPROCESSORS, SHARED_CLASSIFIER_INSTANCES,
)
from samlb.framework.regression.eaml.config import EAML_REG_PARAM_GRID

PAPER_CLF_DATASETS = [
    "adult", "covertype", "credit_card", "electricity", "insects",
    "new_airlines", "nomao", "poker_hand", "shuttle", "vehicle_sensIT",
    "movingRBF", "moving_squares", "sea_high_abrupt_drift",
    "synth_RandomRBFDrift", "synth_agrawal",
]
PAPER_REG_DATASETS = [
    "ailerons", "bike", "california_housing", "cps88wages", "diamonds",
    "elevators", "fifa", "House8L", "kings_county", "MetroTraffic",
    "superconductivity", "wave_energy", "fried", "FriedmanGra", "hyperA",
]


def parse_args():
    p = argparse.ArgumentParser(description="SAMLB random-search baseline")
    p.add_argument("--task", choices=["classification", "regression"], required=True)
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=None, help="run a single specific seed")
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    if args.task == "classification":
        scalers, models = SHARED_PREPROCESSORS, SHARED_CLASSIFIER_INSTANCES
        datasets = args.datasets or PAPER_CLF_DATASETS
    else:
        scalers = EAML_REG_PARAM_GRID["Scaler"]
        models = EAML_REG_PARAM_GRID["Regressor"]
        datasets = args.datasets or PAPER_REG_DATASETS

    if args.seed is not None:
        seeds = [args.seed]; args.n_runs = 1
    else:
        seeds = list(range(args.n_runs))
    print(f"SAMLB Random-Search Baseline ({args.task})", flush=True)
    print(f"  pool size:  {len(models)} models x {len(scalers)} scalers", flush=True)
    print(f"  window:     {args.window}", flush=True)
    print(f"  n_runs:     {args.n_runs}", flush=True)
    print(f"  datasets:   {len(datasets)}", flush=True)

    suite = BenchmarkSuite(
        models={"RandomSearch": RandomSearch(scalers=scalers, models=models,
                                             exploration_window=args.window,
                                             clip=(args.task=="regression"))},
        datasets=datasets,
        task=args.task,
        normalize=(args.task == "regression"),
        n_runs=args.n_runs,
        seeds=seeds,
        verbose=True,
    )
    suite.run()
    suite.print_table()
    suite.to_json(args.output_dir)
    suite.to_csv(f"{args.output_dir}/{args.task}_random_search_{args.n_runs}runs.csv")


if __name__ == "__main__":
    main()
