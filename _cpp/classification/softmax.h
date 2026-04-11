#pragma once
#include <unordered_map>
#include <string>
#include <set>

// Multiclass Softmax Regression via SGD.
class SoftmaxRegression {
public:
    explicit SoftmaxRegression(double learning_rate = 0.01, double l2 = 0.0);
    void learn_one(const std::unordered_map<std::string, double>& x, int y);
    int  predict_one(const std::unordered_map<std::string, double>& x) const;
    std::unordered_map<int, double> predict_proba_one(
        const std::unordered_map<std::string, double>& x) const;
    void reset();

    double learning_rate, l2;

private:
    std::unordered_map<int, std::unordered_map<std::string, double>> weights;
    std::unordered_map<int, double> bias;
    std::set<int> seen_classes;

    std::unordered_map<int, double> softmax(
        const std::unordered_map<std::string, double>& x) const;
};
