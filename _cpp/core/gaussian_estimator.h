#pragma once
#include <cmath>
#include <limits>

// Incremental Gaussian sufficient statistics (Welford's algorithm).
// Tracks n, mean, M2 for online variance computation.
struct GaussianEstimator {
    double n    = 0.0;
    double mean = 0.0;
    double M2   = 0.0;   // sum of squared deviations

    void update(double x, double weight = 1.0) {
        n += weight;
        double delta  = x - mean;
        mean         += weight * delta / n;
        double delta2 = x - mean;
        M2           += weight * delta * delta2;
    }

    double variance() const {
        return (n > 1.0) ? M2 / (n - 1.0) : 0.0;
    }

    double std_dev() const {
        return std::sqrt(variance());
    }

    // P(x | class) — Gaussian probability density
    double probability_density(double x) const {
        if (n < 2.0) return 0.0;
        double var = variance();
        if (var <= 0.0) return (x == mean) ? 1.0 : 0.0;
        double diff = x - mean;
        return std::exp(-0.5 * diff * diff / var) /
               std::sqrt(2.0 * M_PI * var);
    }

    void reset() { n = 0.0; mean = 0.0; M2 = 0.0; }
};
