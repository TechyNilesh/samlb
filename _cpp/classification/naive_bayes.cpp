#include "naive_bayes.h"
#include <cmath>
#include <stdexcept>

NaiveBayes::NaiveBayes() {}

void NaiveBayes::learn_one(const Features& x, int y) {
    class_counts[y]++;
    total_count++;
    for (auto& [feat, val] : x)
        estimators[y][feat].update(val);
}

int NaiveBayes::predict_one(const Features& x) const {
    if (class_counts.empty()) return -1;
    double best_score = -std::numeric_limits<double>::infinity();
    int    best_class = class_counts.begin()->first;
    for (auto& [cls, cnt] : class_counts) {
        double log_prob = std::log(static_cast<double>(cnt) / total_count);
        auto it = estimators.find(cls);
        if (it != estimators.end()) {
            for (auto& [feat, val] : x) {
                auto fit = it->second.find(feat);
                if (fit != it->second.end()) {
                    double p = fit->second.probability_density(val);
                    log_prob += std::log(p > 1e-300 ? p : 1e-300);
                }
            }
        }
        if (log_prob > best_score) { best_score = log_prob; best_class = cls; }
    }
    return best_class;
}

std::unordered_map<int, double> NaiveBayes::predict_proba_one(const Features& x) const {
    std::unordered_map<int, double> scores;
    double total = 0.0;
    for (auto& [cls, cnt] : class_counts) {
        double prob = static_cast<double>(cnt) / total_count;
        auto it = estimators.find(cls);
        if (it != estimators.end()) {
            for (auto& [feat, val] : x) {
                auto fit = it->second.find(feat);
                if (fit != it->second.end()) {
                    double p = fit->second.probability_density(val);
                    prob *= (p > 1e-300 ? p : 1e-300);
                }
            }
        }
        scores[cls] = prob;
        total += prob;
    }
    if (total > 0)
        for (auto& [k, v] : scores) v /= total;
    return scores;
}

void NaiveBayes::reset() {
    class_counts.clear();
    estimators.clear();
    total_count = 0;
}
