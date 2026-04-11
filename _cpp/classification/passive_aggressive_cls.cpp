#include "passive_aggressive_cls.h"
#include <cmath>
#include <limits>
#include <algorithm>

PassiveAggressiveClassifier::PassiveAggressiveClassifier(double C) : C(C) {}

double PassiveAggressiveClassifier::dot(
    int cls, const std::unordered_map<std::string, double>& x) const {
    double s = 0.0;
    auto it = weights.find(cls);
    if (it != weights.end())
        for (auto& [k, v] : x) {
            auto wit = it->second.find(k);
            if (wit != it->second.end()) s += wit->second * v;
        }
    auto bit = bias.find(cls);
    return s + (bit != bias.end() ? bit->second : 0.0);
}

double PassiveAggressiveClassifier::norm_sq(
    const std::unordered_map<std::string, double>& x) const {
    double s = 0.0;
    for (auto& [k, v] : x) s += v * v;
    return s + 1.0;  // +1 for bias
}

int PassiveAggressiveClassifier::predict_one(
    const std::unordered_map<std::string, double>& x) const {
    if (seen_classes.empty()) return 0;
    double best = -std::numeric_limits<double>::infinity();
    int pred = *seen_classes.begin();
    for (int cls : seen_classes) {
        double s = dot(cls, x);
        if (s > best) { best = s; pred = cls; }
    }
    return pred;
}

std::unordered_map<int, double> PassiveAggressiveClassifier::predict_proba_one(
    const std::unordered_map<std::string, double>& x) const {
    std::unordered_map<int, double> proba;
    double sum = 0.0;
    for (int cls : seen_classes) {
        double s = std::exp(dot(cls, x));
        proba[cls] = s;
        sum += s;
    }
    if (sum > 0) for (auto& [k, v] : proba) v /= sum;
    return proba;
}

void PassiveAggressiveClassifier::learn_one(
    const std::unordered_map<std::string, double>& x, int y) {
    seen_classes.insert(y);
    if (weights.find(y) == weights.end()) { weights[y]; bias[y] = 0.0; }

    // PA-I: hinge loss update
    double margin = dot(y, x);
    // penalise any runner-up class
    double max_other = -std::numeric_limits<double>::infinity();
    int    runner_up = y;
    for (int cls : seen_classes) {
        if (cls == y) continue;
        if (weights.find(cls) == weights.end()) { weights[cls]; bias[cls] = 0.0; }
        double s = dot(cls, x);
        if (s > max_other) { max_other = s; runner_up = cls; }
    }
    if (runner_up == y) return;  // only one class seen

    double loss = std::max(0.0, 1.0 - (margin - max_other));
    if (loss == 0.0) return;  // passive

    // PA-I step size (clipped by C)
    double ns = norm_sq(x);
    double tau = std::min(C, loss / (2.0 * ns));

    // Update correct class up, runner-up down
    for (auto& [k, v] : x) {
        weights[y][k]        += tau * v;
        weights[runner_up][k] -= tau * v;
    }
    bias[y]        += tau;
    bias[runner_up] -= tau;
}

void PassiveAggressiveClassifier::reset() {
    weights.clear(); bias.clear(); seen_classes.clear();
}
