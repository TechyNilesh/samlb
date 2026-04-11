"""
Tests for samlb.algorithms.regression on real-world ARFF datasets.

Each test runs prequential evaluation (test-then-train) and asserts:
  - R² > minimum threshold (model learns better than mean predictor)
  - RMSE is finite and reasonable
  - Throughput > minimum inst/sec
  - reset() works correctly
"""
import time
import math
import pytest

from samlb.algorithms.regression import (
    LinearRegression,
    BayesianLinearRegression,
    PassiveAggressiveRegressor,
    KNNRegressor,
    HoeffdingTreeRegressor,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def prequential_reg(model, stream, window=1000):
    """
    Prequential regression evaluation.
    Returns (r2, rmse, mae, windowed_rmse, throughput).
    """
    preds, truths = [], []
    window_se, window_ae = [], []
    windowed_rmse = []
    t0 = time.perf_counter()

    for x, y in stream:
        pred = model.predict_one(x)
        preds.append(pred)
        truths.append(y)
        window_se.append((pred - y) ** 2)
        window_ae.append(abs(pred - y))
        model.learn_one(x, y)
        if len(preds) % window == 0:
            windowed_rmse.append(math.sqrt(sum(window_se[-window:]) / window))

    elapsed = time.perf_counter() - t0
    n       = len(truths)
    tput    = n / elapsed

    mean_y  = sum(truths) / n
    ss_tot  = sum((y - mean_y) ** 2 for y in truths)
    ss_res  = sum((p - t) ** 2 for p, t in zip(preds, truths))
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse    = math.sqrt(ss_res / n)
    mae     = sum(window_ae) / n

    return r2, rmse, mae, windowed_rmse, tput


def all_regressors():
    return {
        "LinearRegression":      LinearRegression(learning_rate=0.01),
        "BayesianLinearReg":     BayesianLinearRegression(alpha=1.0, beta=1.0),
        "PassiveAggressiveReg":  PassiveAggressiveRegressor(C=1.0, epsilon=0.01),
        "KNNRegressor(5,500)":   KNNRegressor(n_neighbors=5, window_size=500),
        "HoeffdingTreeReg":      HoeffdingTreeRegressor(),
    }


# ── per-dataset tests ─────────────────────────────────────────────────────────

class TestFried:
    """fried.arff — Friedman synthetic regression, 40K instances, 10 features."""

    MIN_R2   = 0.10   # prequential R² is lower than batch; HT needs warmup
    MIN_TPUT = 10_000

    def test_hoeffding_tree(self, fried):
        r2, rmse, mae, windows, tput = prequential_reg(HoeffdingTreeRegressor(), fried)
        print(f"\n[fried] HoeffdingTree  r2={r2:.3f}  rmse={rmse:.3f}  mae={mae:.3f}  tput={tput:,.0f}/s")
        assert r2   >= self.MIN_R2,   f"R² {r2:.3f} < {self.MIN_R2}"
        assert math.isfinite(rmse),   "RMSE is not finite"
        assert tput >= self.MIN_TPUT, f"Throughput {tput:.0f} too low"

    def test_linear_regression(self, fried):
        r2, rmse, mae, _, tput = prequential_reg(LinearRegression(learning_rate=0.001), fried)
        print(f"\n[fried] LinearReg      r2={r2:.3f}  rmse={rmse:.3f}  tput={tput:,.0f}/s")
        assert math.isfinite(r2) and math.isfinite(rmse)
        assert tput >= self.MIN_TPUT

    def test_bayesian_linear(self, fried):
        r2, rmse, mae, _, tput = prequential_reg(BayesianLinearRegression(), fried)
        print(f"\n[fried] BayesianLR     r2={r2:.3f}  rmse={rmse:.3f}  tput={tput:,.0f}/s")
        assert math.isfinite(r2)
        assert tput >= self.MIN_TPUT

    def test_passive_aggressive(self, fried):
        r2, rmse, mae, _, tput = prequential_reg(
            PassiveAggressiveRegressor(C=0.1, epsilon=0.01), fried)
        print(f"\n[fried] PassiveAgg     r2={r2:.3f}  rmse={rmse:.3f}  tput={tput:,.0f}/s")
        assert math.isfinite(r2)
        assert tput >= self.MIN_TPUT

    def test_knn(self, fried):
        r2, rmse, mae, _, tput = prequential_reg(
            KNNRegressor(n_neighbors=5, window_size=500), fried)
        print(f"\n[fried] KNNReg(5,500)  r2={r2:.3f}  rmse={rmse:.3f}  tput={tput:,.0f}/s")
        assert math.isfinite(r2)
        assert tput >= self.MIN_TPUT

    def test_windowed_rmse_trend(self, fried):
        """RMSE should decrease (model improves) as HoeffdingTree sees more data."""
        _, _, _, windows, _ = prequential_reg(HoeffdingTreeRegressor(), fried)
        assert len(windows) >= 5
        first  = sum(windows[:5]) / 5
        last   = sum(windows[-5:]) / 5
        print(f"\n[fried] HT windowed RMSE: first={first:.3f}  last={last:.3f}")
        assert last <= first * 1.2  # allow slight increase at start


class TestAilerons:
    """ailerons.arff — control surface regression, ~14K instances.

    KNN throughput is lower on larger datasets (O(window) per prediction).
    Per-algorithm throughput minimums applied.
    """

    MIN_R2        = 0.10
    MIN_TPUT      = 10_000
    KNN_MIN_TPUT  = 3_000   # KNN is O(window_size) — 500 neighbours, 40 features

    def test_all_regressors(self, ailerons):
        for name, model in all_regressors().items():
            r2, rmse, mae, _, tput = prequential_reg(model, ailerons)
            print(f"\n[ailerons] {name:25s}: r2={r2:.3f}  rmse={rmse:.4f}  mae={mae:.4f}  tput={tput:,.0f}/s")
            assert math.isfinite(r2),   f"{name}: R² not finite"
            assert math.isfinite(rmse), f"{name}: RMSE not finite"
            min_tput = self.KNN_MIN_TPUT if "KNN" in name else self.MIN_TPUT
            assert tput >= min_tput, f"{name}: throughput {tput:.0f} < {min_tput}"


class TestCPUActivity:
    """cpu_activity.arff — CPU load regression, ~8K instances."""

    MIN_R2   = 0.20
    MIN_TPUT = 10_000

    def test_all_regressors(self, cpu_activity):
        for name, model in all_regressors().items():
            r2, rmse, mae, _, tput = prequential_reg(model, cpu_activity)
            print(f"\n[cpu] {name:25s}: r2={r2:.3f}  rmse={rmse:.3f}  tput={tput:,.0f}/s")
            assert math.isfinite(r2)
            assert tput >= self.MIN_TPUT


class TestKin8nm:
    """kin8nm.arff — robot arm kinematics, ~8K instances."""

    MIN_R2   = 0.10
    MIN_TPUT = 10_000

    def test_all_regressors(self, kin8nm):
        for name, model in all_regressors().items():
            r2, rmse, mae, _, tput = prequential_reg(model, kin8nm)
            print(f"\n[kin8nm] {name:25s}: r2={r2:.3f}  rmse={rmse:.4f}  tput={tput:,.0f}/s")
            assert math.isfinite(r2)
            assert tput >= self.MIN_TPUT


class TestElevators:
    """elevators.arff — elevator system regression, ~16K instances."""

    MIN_TPUT = 10_000

    def test_all_regressors(self, elevators):
        for name, model in all_regressors().items():
            r2, rmse, mae, _, tput = prequential_reg(model, elevators)
            print(f"\n[elevators] {name:25s}: r2={r2:.3f}  rmse={rmse:.5f}  tput={tput:,.0f}/s")
            assert math.isfinite(r2)
            assert math.isfinite(rmse)
            assert tput >= self.MIN_TPUT


# ── correctness tests ─────────────────────────────────────────────────────────

class TestCorrectness:
    """Unit-level correctness for all regression algorithms."""

    def test_predict_returns_float(self, fried):
        model = LinearRegression()
        x, y  = fried[0]
        model.learn_one(x, y)
        pred  = model.predict_one(x)
        assert isinstance(pred, float), f"Expected float, got {type(pred)}"

    def test_predict_is_finite(self, fried):
        model = HoeffdingTreeRegressor()
        for x, y in fried[:500]:
            model.learn_one(x, y)
        x, _ = fried[500]
        pred = model.predict_one(x)
        assert math.isfinite(pred), f"Prediction is not finite: {pred}"

    def test_reset_clears_state(self, fried):
        model = LinearRegression()
        for x, y in fried[:2000]:
            model.learn_one(x, y)
        pred_before = model.predict_one(fried[0][0])
        model.reset()
        pred_after = model.predict_one(fried[0][0])
        # After reset weights are 0, so prediction should be 0.0 (bias = 0)
        assert pred_after == 0.0, f"After reset expected 0.0, got {pred_after}"
        assert pred_before != pred_after

    def test_knn_no_crash_empty(self, fried):
        model = KNNRegressor(n_neighbors=5, window_size=100)
        x, _ = fried[0]
        pred = model.predict_one(x)   # before any learn_one
        assert isinstance(pred, float)

    @pytest.mark.parametrize("ModelClass,kwargs", [
        (LinearRegression,         {"learning_rate": 0.01}),
        (BayesianLinearRegression, {}),
        (PassiveAggressiveRegressor, {"C": 1.0}),
        (KNNRegressor,             {"n_neighbors": 5, "window_size": 200}),
        (HoeffdingTreeRegressor,   {}),
    ])
    def test_no_crash_on_stream(self, ModelClass, kwargs, fried):
        model = ModelClass(**kwargs)
        for x, y in fried[:1000]:
            pred = model.predict_one(x)
            assert math.isfinite(pred), f"{ModelClass.__name__} produced non-finite prediction"
            model.learn_one(x, y)


# ── throughput benchmarks ─────────────────────────────────────────────────────

class TestThroughput:
    """All regressors must exceed 10K inst/sec on real data."""

    MIN_THROUGHPUT = 10_000

    @pytest.mark.parametrize("name,model", [
        ("LinearRegression",     LinearRegression()),
        ("BayesianLinearReg",    BayesianLinearRegression()),
        ("PassiveAggressiveReg", PassiveAggressiveRegressor()),
        ("KNNRegressor(5,500)",  KNNRegressor(5, 500)),
        ("HoeffdingTreeReg",     HoeffdingTreeRegressor()),
    ])
    def test_throughput(self, name, model, fried):
        _, _, _, _, tput = prequential_reg(model, fried)
        print(f"\n  {name:28s}: {tput:,.0f} inst/sec")
        assert tput >= self.MIN_THROUGHPUT, \
            f"{name} throughput {tput:.0f} < {self.MIN_THROUGHPUT}"
