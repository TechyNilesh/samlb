"""
Shared fixtures for SAMLB tests.
Loaders for real-world CSV (classification) and ARFF (regression) datasets.
"""
import os
import pytest
import pandas as pd

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
CLS_DIR  = os.path.join(DATASETS_DIR, "classification")
REG_DIR  = os.path.join(DATASETS_DIR, "regression")

# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv_classification(filename, label_col="class", max_rows=10_000):
    """Load a classification CSV. Returns list of (x_dict, y_int) tuples."""
    path = os.path.join(CLS_DIR, filename)
    df   = pd.read_csv(path, nrows=max_rows)
    df   = df.dropna()
    # encode label as int
    df[label_col] = pd.Categorical(df[label_col]).codes
    feature_cols  = [c for c in df.columns if c != label_col]
    # keep only numeric features
    df = df[feature_cols + [label_col]]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rows = []
    for _, row in df.iterrows():
        x = {c: float(row[c]) for c in feature_cols}
        y = int(row[label_col])
        rows.append((x, y))
    return rows


def load_arff_regression(filename, max_rows=10_000, normalize=True):
    """
    Load a regression ARFF. Returns list of (x_dict, y_float) tuples.
    normalize=True applies min-max scaling to features and target so that
    all values are in [0, 1] — prevents overflow in gradient-based algorithms
    on high-magnitude datasets like cpu_activity.
    """
    path = os.path.join(REG_DIR, filename)
    attrs, data_lines = [], []
    in_data = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                parts = line.split()
                attrs.append(parts[1])
            elif low.startswith("@data"):
                in_data = True
            elif in_data:
                data_lines.append(line)
                if len(data_lines) >= max_rows:
                    break

    feat_names = attrs[:-1]
    raw = []
    for line in data_lines:
        vals = line.split(",")
        if len(vals) != len(attrs):
            continue
        try:
            raw.append([float(v) for v in vals])
        except ValueError:
            continue

    if not raw:
        return []

    # --- min-max normalise each column (features + target) ---
    if normalize:
        n_cols = len(attrs)
        col_min = [min(r[c] for r in raw) for c in range(n_cols)]
        col_max = [max(r[c] for r in raw) for c in range(n_cols)]
        col_range = [
            (col_max[c] - col_min[c]) if col_max[c] != col_min[c] else 1.0
            for c in range(n_cols)
        ]
        rows = []
        for r in raw:
            x = {feat_names[i]: (r[i] - col_min[i]) / col_range[i]
                 for i in range(len(feat_names))}
            y = (r[-1] - col_min[-1]) / col_range[-1]
            rows.append((x, y))
    else:
        rows = [
            ({feat_names[i]: r[i] for i in range(len(feat_names))}, r[-1])
            for r in raw
        ]
    return rows


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def electricity():
    return load_csv_classification("electricity.csv", label_col="class")

@pytest.fixture(scope="session")
def adult():
    return load_csv_classification("adult.csv", label_col="class")

@pytest.fixture(scope="session")
def sea_abrupt():
    return load_csv_classification("sea_high_abrupt_drift.csv", label_col="class")

@pytest.fixture(scope="session")
def hyperplane():
    return load_csv_classification("hyperplane_high_gradual_drift.csv", label_col="class")

@pytest.fixture(scope="session")
def airlines():
    return load_csv_classification("new_airlines.csv", label_col="class")

@pytest.fixture(scope="session")
def fried():
    return load_arff_regression("fried.arff")

@pytest.fixture(scope="session")
def ailerons():
    return load_arff_regression("ailerons.arff")

@pytest.fixture(scope="session")
def cpu_activity():
    return load_arff_regression("cpu_activity.arff")

@pytest.fixture(scope="session")
def kin8nm():
    return load_arff_regression("kin8nm.arff")

@pytest.fixture(scope="session")
def elevators():
    return load_arff_regression("elevators.arff")
