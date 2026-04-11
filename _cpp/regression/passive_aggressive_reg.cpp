#include "passive_aggressive_reg.h"
#include <cmath>
#include <algorithm>

PassiveAggressiveRegressor::PassiveAggressiveRegressor(double C, double eps)
    : C(C), epsilon(eps) {}

double PassiveAggressiveRegressor::predict_one(
    const std::unordered_map<std::string, double>& x) const {
    double s = bias;
    for (auto& [k, v] : x) {
        auto it = weights.find(k);
        if (it != weights.end()) s += it->second * v;
    }
    return s;
}

void PassiveAggressiveRegressor::learn_one(
    const std::unordered_map<std::string, double>& x, double y) {
    double pred  = predict_one(x);
    double loss  = std::max(0.0, std::abs(y - pred) - epsilon);
    if (loss == 0.0) return;  // passive

    // norm squared of x (+ 1 for bias)
    double ns = 1.0;
    for (auto& [k, v] : x) ns += v * v;

    double tau = std::min(C, loss / ns);
    double sign = (y - pred > 0) ? 1.0 : -1.0;

    for (auto& [k, v] : x) weights[k] += tau * sign * v;
    bias += tau * sign;
}

void PassiveAggressiveRegressor::reset() { weights.clear(); bias = 0.0; }
