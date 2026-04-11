#include "logistic_regression.h"
#include <cmath>
#include <limits>

LogisticRegression::LogisticRegression(double lr, double l2)
    : learning_rate(lr), l2(l2) {}

double LogisticRegression::sigmoid(double z) const {
    return 1.0 / (1.0 + std::exp(-std::max(-500.0, std::min(500.0, z))));
}

double LogisticRegression::score(int cls,
    const std::unordered_map<std::string, double>& x) const {
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

int LogisticRegression::predict_one(
    const std::unordered_map<std::string, double>& x) const {
    if (seen_classes.empty()) return 0;
    double best = -std::numeric_limits<double>::infinity();
    int    pred = *seen_classes.begin();
    for (int cls : seen_classes) {
        double s = score(cls, x);
        if (s > best) { best = s; pred = cls; }
    }
    return pred;
}

std::unordered_map<int, double> LogisticRegression::predict_proba_one(
    const std::unordered_map<std::string, double>& x) const {
    std::unordered_map<int, double> proba;
    double sum = 0.0;
    for (int cls : seen_classes) {
        double s = std::exp(score(cls, x));
        proba[cls] = s;
        sum += s;
    }
    if (sum > 0) for (auto& [k, v] : proba) v /= sum;
    return proba;
}

void LogisticRegression::learn_one(
    const std::unordered_map<std::string, double>& x, int y) {
    seen_classes.insert(y);
    if (weights.find(y) == weights.end()) { weights[y]; bias[y] = 0.0; }

    // One-vs-rest SGD update for each class
    for (int cls : seen_classes) {
        double target = (cls == y) ? 1.0 : 0.0;
        double pred   = sigmoid(score(cls, x));
        double err    = target - pred;
        for (auto& [k, v] : x) {
            weights[cls][k] += learning_rate * (err * v - l2 * weights[cls][k]);
        }
        bias[cls] += learning_rate * err;
    }
}

void LogisticRegression::reset() {
    weights.clear(); bias.clear(); seen_classes.clear();
}
