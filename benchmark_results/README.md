# Benchmark results

Prequential (test-then-train) results on 15 classification + 15 regression
datasets, 10 seeds each, produced on a 128-core EPYC.

| file | contents |
|---|---|
| `classification_v06.csv` | 15 datasets x 7 frameworks x 10 seeds (1050 runs) |
| `regression_v08.csv` | 15 datasets x 6 frameworks x 10 seeds (900 runs) |
| `*_summary.json` | same runs with per-window learning curves |

Both were produced after the leaf-model fixes to `HoeffdingTreeClassifier`
(naive Bayes adaptive leaves) and `HoeffdingTreeRegressor` (standardised leaf
SGD, adaptive mean/linear selection, bounded extrapolation). Earlier result
sets are not comparable with these.

## Headline

Classification — mean accuracy / mean rank / wins over 15 datasets:

| framework | accuracy | rank | wins | CPU h |
|---|---:|---:|---:|---:|
| StreamingAutoGluon | 0.8510 | 2.20 | 6/15 | 13.19 |
| ARF | 0.8222 | 2.80 | 5/15 | 3.55 |
| EvoAutoML | 0.8377 | 3.40 | 1/15 | 8.58 |
| ASML | 0.8205 | 3.80 | 1/15 | 4.46 |
| OAML | 0.8142 | 4.27 | 1/15 | 2.84 |
| AutoClass | 0.8269 | 4.53 | 1/15 | 6.87 |
| RandomSearch | 0.6930 | 7.00 | 0/15 | 2.21 |

Regression — mean R2 / mean rank / wins:

| framework | R2 | rank | wins | CPU h |
|---|---:|---:|---:|---:|
| StreamingAutoGluon | 0.7753 | 1.27 | 11/15 | 0.97 |
| ASML | 0.7393 | 2.67 | 2/15 | 1.05 |
| EvoAutoML | 0.7089 | 2.93 | 0/15 | 1.66 |
| ChaCha | 0.6180 | 4.20 | 1/15 | 2.60 |
| ARF | 0.5681 | 4.33 | 1/15 | 0.22 |
| RandomSearch | 0.4073 | 5.60 | 0/15 | 0.37 |

StreamingAutoGluon is deterministic, so its standard deviation across seeds is
zero on every dataset.
