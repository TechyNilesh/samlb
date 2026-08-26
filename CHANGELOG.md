# Changelog

All notable changes to SAMLB are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.6.0] - 2026-08-26

### Added
- `SGBTClassifier` / `SGBRRegressor` — Streaming Gradient Boosted Trees, the
  first boosting method in the pool; everything else (ARF, SRP, Leveraging
  Bagging) is bagging-family. These are **two different methods sharing one
  boosting machine**, not one method with two output types. Classification
  follows Gunasekara, Pfahringer, Gomes & Bifet (*Machine Learning* 2024): 100
  boosting iterations, learning rate 0.0125, a single tree per iteration.
  Regression follows SGBR (same authors, *DMKD* 2025): 10 iterations, learning
  rate 1.0, and a Poisson(1) bag of 10 trees per iteration, the paper's
  SGB(Oza) variant. The learning rate is the part that cannot be shared: at
  0.0125 the raw score reaches only `1 - (1 - lr)^n` of the target, 0.71 after
  100 iterations, and that 29% shrinkage is a bias term that destroys R² on any
  target whose mean is large relative to its spread.
- `FIMTDDRegressor` — Fast Incremental Model Tree with Drift Detection
  (Ikonomovska, Gama & Džeroski, *DMKD* 2011): E-BST attribute observers, the
  ratio form of the Hoeffding test compared across attributes, and
  Page-Hinckley drift detection with alternate subtrees. It exists because
  `HoeffdingTreeRegressor` barely splits — it summarises each feature with a
  Gaussian and tries a few interpolated cut points, where FIMT-DD keeps every
  observed value as an exact candidate. On a step function the older tree
  scores MAE 0.68 against a constant predictor's 1.0; FIMT-DD scores 0.005.
- **MOA's FIMT-DD perceptron diverges, and the fix is carried here.** Its leaf
  model normalises by *running* global feature statistics, so an instance seen
  while a feature's spread is still near zero overshoots the weights and
  nothing bounds the output afterwards. `leaf_prediction="adaptive"` (default)
  bounds the perceptron to the leaf's own observed target range and uses it
  only while it beats the leaf mean. `"perceptron"` keeps the reference form,
  divergence included, so the defect stays reproducible.
- `LeveragingBaggingRegressor` — an **adaptation**, not a port. Leveraging
  Bagging (Bifet, Holmes & Pfahringer, ICDM 2010) is classification-only;
  neither MOA nor River has a regression version. Poisson(6) resampling and
  per-member ADWIN-with-reset carry over unchanged; the random output codes are
  dropped, being defined over class labels, which MOA itself makes optional.
  The one addition a continuous target forces is feeding ADWIN the absolute
  error over the target's running spread, since a raw residual would tie the
  detector's sensitivity to the units of `y`. Cite it as an adaptation.
- The regression and classification ensemble pools are now symmetric, five
  candidates each: ARF, SRP, Leveraging Bagging, the drift-adaptive tree
  (`HoeffdingAdaptiveTreeClassifier` / `FIMTDDRegressor`), and the boosting
  method (SGBT / SGBR).
- `RunResult.cpu_time_s` and `RunResult.peak_memory_mb` — process CPU time and
  peak resident memory are now recorded natively by `PrequentialEvaluator`,
  alongside the existing wall-clock. CPU time is the figure to quote on a
  shared machine, since wall-clock reflects whatever else is running.

### Changed
- **`HoeffdingTreeClassifier` split search now matches MOA's reference**
  (`GaussianNumericAttributeClassObserver` with Info/Gini gain). Per (class,
  feature) Gaussian summaries carry the observed min/max, so split-weight
  estimates are exact at the tails instead of trusting the Gaussian CDF where
  it is least reliable; candidate split points are evenly spaced between the
  observed min and max with `n_split_points` raised from 1 to 10 (MOA's
  `numBinsOption` default); and merit is computed on the exact resulting class
  distributions rather than a mean ± 3σ heuristic. Evaluating a single
  threshold at the weighted mean was badly under-splitting. `no_pre_prune` is
  exposed through the bindings. Thanks to Daniel Nowak.
  **This changes results.** `HoeffdingTreeClassifier` is in
  `SHARED_MODEL_POOL`, so every benchmark number that involves it is expected
  to move. The published tables in `benchmark_results/` predate the fix and
  need regenerating before they are quoted again.

### Fixed
- ASML regression: the divergence guard was six orders of magnitude too loose,
  letting a diverged pipeline's prediction through effectively unclipped. On
  `fifa` this cost 0.6076 R² against 0.7552 once tightened, and on
  `superconductivity` 0.7728 against 0.8569.

### Removed
- The benchmark tables, raw result files and changelog carried an entry for a
  stacking framework that was added and then removed during the 0.2.0 cycle and
  is not part of this package. Its rows were being published as though they
  described something installable here.

## [0.5.0] - 2026-08-22

### Added
- `SRPRegressor` — regression counterpart of `SRPClassifier` (Streaming Random
  Patches, Gomes et al. 2019): per-member fixed random feature subspace +
  Poisson resampling, with per-member ADWIN background-tree swap on drift.
- `LeveragingBaggingClassifier` (Bifet, Holmes & Pfahringer, ICDM 2010):
  diversity via higher Poisson(6) resampling rather than feature subspacing;
  each member carries its own ADWIN and is reset outright on drift (no
  background-tree promotion, unlike ARF/SRP).
- `HoeffdingAdaptiveTreeClassifier` (Bifet & Gavaldà, SDM 2009) — a whole-tree
  simplification of the node-level original: one ADWIN warning/drift pair
  tracks overall tree accuracy, with a background tree trained in parallel and
  promoted on drift. Documented as a simplification in the class docstring;
  benchmarked against CapyMOA's node-level implementation.
- Ensemble-vs-normal model pool selection for every classification and
  regression search framework (ASML, AutoClass, EvoAutoML, OAML), via
  `get_classification_config(pool=...)` / `get_regression_config(pool=...)`
  — `"normal"` (plain single models) or `"ensemble"` (ARF / SRP / Leveraging
  Bagging / Hoeffding Adaptive Tree) as the candidate pool.
- `samlb/framework/regression/shared_config.py` — a `RegressionConfig`
  dataclass mirroring the existing classification `shared_config`, giving
  regression the same single-source-of-truth algorithm pool that
  classification already had.

### Changed
- `samlb/framework/regression/asml/config.py` and
  `samlb/framework/regression/eaml/config.py` now re-export their pool and
  hyperparameter definitions from the new shared config (same public names,
  no behaviour change for existing callers).

## [0.4.0] - 2026-08-22

### Added
- `SRPClassifier` framework wrapper (C++ backend), exposed through
  `samlb.framework.base`.
- River and CapyMOA adapters (`RiverClassifier`/`RiverRegressor`,
  `CapyMOAClassifier`/`CapyMOARegressor`), letting either library's models
  drop into `BenchmarkSuite` with the same `predict_one`/`learn_one`/`reset`
  contract as native SAMLB models.
- Contributor guide and wiki documentation.

### Fixed
- Refreshed the `uv.lock` lockfile.

## [0.3.0] - 2026-06-29

### Added
- `RandomSearch` baseline — a task-agnostic sanity-check floor that randomly
  selects one pipeline per exploration window from a shared warm learner
  pool. Any real AutoML search strategy is expected to beat it.

### Removed
- The trivial majority-class / mean-prediction baselines, superseded by
  `RandomSearch` as a stronger, still-non-intelligent floor.

## [0.2.0] - 2026-05-14

### Added
- Full C++ core: base learners (Naive Bayes, Perceptron, Logistic Regression,
  Passive Aggressive, Softmax Regression, KNN, Hoeffding Tree, EFDT, SGT),
  preprocessing (Standard/MinMax/MaxAbs scalers, VarianceThreshold,
  SelectKBest), metrics, ADWIN/EDDM drift detection, and an ARF regressor —
  all native, with fused `scaler | selector | model` pipelines that cross the
  Python/C++ boundary once per `learn_one`/`predict_one` instead of once per
  stage.
- SOTA streaming ensemble baselines: ARF (Gomes et al. 2017) and SRP (Gomes
  et al. 2019), validated against River's reference implementations.
- OAML reimplemented with GAMA-style pipeline search.
- Benchmark results published in the README.

### Changed
- River dependency dropped entirely in favour of the native C++ core.
- Windows build fixed; deprecated `macos-13` CI runner dropped in favour of
  current runners.
- ARF removed from the regression shared learner pool.

### Fixed
- `HoeffdingTreeClassifier` previously only ever predicted the majority
  class — naive Bayes leaves were gated behind `nb_threshold > 0` (default
  `0`). Naive Bayes Adaptive is now the default leaf predictor, matching MOA
  and River.
- `HoeffdingTreeRegressor` fitted its leaf linear model on raw targets with a
  scale-blind gradient clip, diverging badly on datasets with large target
  scales (R² of -0.72 on `california_housing`, worse than a constant
  predictor). The leaf model now fits in standardised space, is selected
  adaptively against the leaf mean, and cannot extrapolate beyond the leaf's
  own target spread.
- Build no longer uses `-ffast-math`: it implies `-ffinite-math-only`, under
  which the `-inf` sentinels used throughout the learners are undefined
  behaviour, which was making results non-reproducible.

## [0.1.0] - 2026-04-11

### Added
- Initial release: streaming AutoML benchmark framework with prequential
  (test-then-train) evaluation, windowed metric snapshots, and curated
  classification/regression datasets.
- Datasets auto-download from GitHub on first use and are cached locally.

[0.6.0]: https://github.com/TechyNilesh/samlb/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/TechyNilesh/samlb/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/TechyNilesh/samlb/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/TechyNilesh/samlb/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/TechyNilesh/samlb/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/TechyNilesh/samlb/releases/tag/v0.1.0
