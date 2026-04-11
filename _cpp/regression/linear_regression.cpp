#include "linear_regression.h"
#include <cmath>
#include <algorithm>

static constexpr double GRAD_CLIP = 10.0;

static double clip(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

LinearRegression::LinearRegression(double lr, double l2)
    : learning_rate(lr), l2(l2) {}

double LinearRegression::predict_one(
    const std::unordered_map<std::string, double>& x) const {
    double s = bias;
    for (auto& [k, v] : x) {
        auto it = weights.find(k);
        if (it != weights.end()) s += it->second * v;
    }
    return s;
}

void LinearRegression::learn_one(
    const std::unordered_map<std::string, double>& x, double y) {
    double pred = predict_one(x);
    double err  = clip(y - pred, -GRAD_CLIP, GRAD_CLIP);  // clipped residual
    for (auto& [k, v] : x) {
        double grad = clip(err * v, -GRAD_CLIP, GRAD_CLIP);
        weights[k] += learning_rate * (grad - l2 * weights[k]);
    }
    bias += learning_rate * err;
}

void LinearRegression::reset() { weights.clear(); bias = 0.0; }
