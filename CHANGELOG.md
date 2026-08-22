# Changelog

All notable changes to SAMLB are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
- StreamingAutoGluon framework — online stacking of k-fold cross-validated
  stream learners, ported from the original Java reference implementation.
  (Later removed in this same cycle; see below.)
- OAML reimplemented with GAMA-style pipeline search.
- Benchmark results published in the README.

### Changed
- River dependency dropped entirely in favour of the native C++ core.
- Windows build fixed; deprecated `macos-13` CI runner dropped in favour of
  current runners.
- ARF removed from the regression shared learner pool.

### Removed
- StreamingAutoGluon, after being added earlier in this cycle.

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

[0.5.0]: https://github.com/TechyNilesh/samlb/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/TechyNilesh/samlb/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/TechyNilesh/samlb/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/TechyNilesh/samlb/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/TechyNilesh/samlb/releases/tag/v0.1.0
