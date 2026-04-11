#include "perceptron.h"
#include <cmath>
#include <limits>
#include <algorithm>

Perceptron::Perceptron(double lr) : learning_rate(lr) {}

double Perceptron::dot(const std::unordered_map<std::string, double>& w,
                       const std::unordered_map<std::string, double>& x) const {
    double s = 0.0;
    for (auto& [k, v] : x) {
        auto it = w.find(k);
        if (it != w.end()) s += it->second * v;
    }
    return s;
}

int Perceptron::predict_one(const std::unordered_map<std::string, double>& x) const {
    if (weights.empty()) return 0;
    double best = -std::numeric_limits<double>::infinity();
    int    pred = weights.begin()->first;
    for (auto& [cls, w] : weights) {
        double score = dot(w, x) + bias.at(cls);
        if (score > best) { best = score; pred = cls; }
    }
    return pred;
}

std::unordered_map<int, double> Perceptron::predict_proba_one(
    const std::unordered_map<std::string, double>& x) const {
    std::unordered_map<int, double> scores;
    double sum = 0.0;
    for (auto& [cls, w] : weights) {
        double s = std::exp(dot(w, x) + bias.at(cls));
        scores[cls] = s;
        sum += s;
    }
    if (sum > 0) for (auto& [k, v] : scores) v /= sum;
    return scores;
}

void Perceptron::learn_one(const std::unordered_map<std::string, double>& x, int y) {
    // Ensure class weights exist
    if (weights.find(y) == weights.end()) { weights[y]; bias[y] = 0.0; }

    int pred = predict_one(x);
    if (pred != y) {
        // Update: increment correct class, decrement predicted class
        for (auto& [k, v] : x) {
            weights[y][k]    += learning_rate * v;
            weights[pred][k] -= learning_rate * v;
        }
        bias[y]    += learning_rate;
        bias[pred] -= learning_rate;
    }
}

void Perceptron::reset() { weights.clear(); bias.clear(); }
