#include "bayesian_linear_reg.h"
#include <cmath>
#include <algorithm>

static constexpr double ERR_CLIP = 10.0;

static double clip(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

BayesianLinearRegression::BayesianLinearRegression(double alpha, double beta)
    : alpha(alpha), beta(beta) {}

double BayesianLinearRegression::predict_one(
    const std::unordered_map<std::string, double>& x) const {
    double s = mean_bias;
    for (auto& [k, v] : x) {
        auto it = mean.find(k);
        if (it != mean.end()) s += it->second * v;
    }
    return s;
}

void BayesianLinearRegression::learn_one(
    const std::unordered_map<std::string, double>& x, double y) {
    // Diagonal Bayesian update (rank-1 sequential update, Kalman-style)
    double pred = predict_one(x);
    double err  = clip(y - pred, -ERR_CLIP, ERR_CLIP);  // clipped error

    // Update bias
    double s_bias = 1.0 / (precision_bias + beta);
    mean_bias      += s_bias * beta * err;
    precision_bias += beta;

    // Update each feature weight
    for (auto& [k, v] : x) {
        double prec = precision.count(k) ? precision[k] : alpha;
        double s    = 1.0 / (prec + beta * v * v);
        double update = clip(s * beta * v * err, -ERR_CLIP, ERR_CLIP);
        mean[k]      = (mean.count(k) ? mean[k] : 0.0) + update;
        precision[k] = prec + beta * v * v;
    }
}

void BayesianLinearRegression::reset() {
    precision.clear(); mean.clear();
    precision_bias = 0.0; mean_bias = 0.0;
}
