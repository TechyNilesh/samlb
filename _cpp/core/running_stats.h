#pragma once
#include <cmath>
#include <limits>

// Incremental statistics, numerically identical to the River implementations
// they replace (river.stats.{Mean,Var,Min,Max,AbsMax,Cov,PearsonCorr}).
// Update formulas are ported verbatim so that results stay comparable with
// runs produced by the River-based version of SAMLB.

struct RMean {
    double n = 0.0;
    double mean = 0.0;

    inline void update(double x, double w = 1.0) {
        n += w;
        mean += (w / n) * (x - mean);
    }
    inline double get() const { return mean; }
};

// river.stats.Var — Welford, ddof = 1 by default.
struct RVar {
    RMean mean;
    double S = 0.0;
    double ddof = 1.0;

    RVar() = default;
    explicit RVar(double ddof_) : ddof(ddof_) {}

    inline void update(double x, double w = 1.0) {
        const double old_mean = mean.mean;
        mean.update(x, w);
        S += w * (x - old_mean) * (x - mean.mean);
    }
    inline double get() const {
        return mean.n > ddof ? S / (mean.n - ddof) : 0.0;
    }
    inline double n() const { return mean.n; }
};

// NOTE: no +/-inf sentinels here. The extension is built with -ffast-math
// (-ffinite-math-only), under which infinities are undefined behaviour and the
// optimiser is free to fold comparisons against them incorrectly. A seen-flag
// costs nothing and is always well defined.
struct RMin {
    bool   seen = false;
    double v = 0.0;
    inline void update(double x) { if (!seen || x < v) { v = x; seen = true; } }
    inline double get() const { return v; }
};

struct RMax {
    bool   seen = false;
    double v = 0.0;
    inline void update(double x) { if (!seen || x > v) { v = x; seen = true; } }
    inline double get() const { return v; }
};

struct RAbsMax {
    double v = 0.0;
    inline void update(double x) { const double a = std::fabs(x); if (a > v) v = a; }
    inline double get() const { return v; }
};

// river.stats.Cov
struct RCov {
    RMean mean_x, mean_y;
    double cov = 0.0;
    double ddof = 1.0;

    inline void update(double x, double y, double w = 1.0) {
        const double dx = x - mean_x.mean;
        mean_x.update(x, w);
        mean_y.update(y, w);
        const double denom = mean_x.n - ddof;
        cov += w * (dx * (y - mean_y.mean) - cov) / (denom > 1.0 ? denom : 1.0);
    }
    inline double get() const { return cov; }
};

// river.stats.PearsonCorr
struct RPearsonCorr {
    RVar var_x, var_y;
    RCov cov_xy;

    inline void update(double x, double y) {
        var_x.update(x);
        var_y.update(y);
        cov_xy.update(x, y);
    }
    inline double get() const {
        const double vx = var_x.get();
        const double vy = var_y.get();
        if (vx != 0.0 && vy != 0.0) return cov_xy.get() / std::sqrt(vx * vy);
        return 0.0;
    }
};
