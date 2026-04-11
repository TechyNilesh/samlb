#include "softmax.h"
#include <cmath>
#include <limits>
#include <algorithm>

SoftmaxRegression::SoftmaxRegression(double lr, double l2)
    : learning_rate(lr), l2(l2) {}

std::unordered_map<int, double> SoftmaxRegression::softmax(
    const std::unordered_map<std::string, double>& x) const {
    std::unordered_map<int, double> scores;
    double max_score = -std::numeric_limits<double>::infinity();
    for (int cls : seen_classes) {
        double s = bias.count(cls) ? bias.at(cls) : 0.0;
        if (weights.count(cls))
            for (auto& [k, v] : x) {
                auto it = weights.at(cls).find(k);
                if (it != weights.at(cls).end()) s += it->second * v;
            }
        scores[cls] = s;
        max_score = std::max(max_score, s);
    }
    double sum = 0.0;
    for (auto& [cls, s] : scores) { s = std::exp(s - max_score); sum += s; }
    if (sum > 0) for (auto& [cls, s] : scores) s /= sum;
    return scores;
}

int SoftmaxRegression::predict_one(
    const std::unordered_map<std::string, double>& x) const {
    if (seen_classes.empty()) return 0;
    auto proba = softmax(x);
    int best = seen_classes.empty() ? 0 : *seen_classes.begin();
    double best_p = -1.0;
    for (auto& [cls, p] : proba) if (p > best_p) { best_p = p; best = cls; }
    return best;
}

std::unordered_map<int, double> SoftmaxRegression::predict_proba_one(
    const std::unordered_map<std::string, double>& x) const {
    return softmax(x);
}

void SoftmaxRegression::learn_one(
    const std::unordered_map<std::string, double>& x, int y) {
    seen_classes.insert(y);
    if (!weights.count(y)) { weights[y]; bias[y] = 0.0; }

    auto proba = softmax(x);

    for (int cls : seen_classes) {
        if (!weights.count(cls)) { weights[cls]; bias[cls] = 0.0; }
        double target = (cls == y) ? 1.0 : 0.0;
        double err    = target - proba[cls];
        for (auto& [k, v] : x)
            weights[cls][k] += learning_rate * (err * v - l2 * weights[cls][k]);
        bias[cls] += learning_rate * err;
    }
}

void SoftmaxRegression::reset() {
    weights.clear(); bias.clear(); seen_classes.clear();
}
