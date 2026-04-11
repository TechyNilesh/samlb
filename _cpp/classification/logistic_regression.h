#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <set>

// Binary / multiclass logistic regression via SGD (one-vs-rest for multiclass).
class LogisticRegression {
public:
    explicit LogisticRegression(double learning_rate = 0.01, double l2 = 0.0);
    void learn_one(const std::unordered_map<std::string, double>& x, int y);
    int  predict_one(const std::unordered_map<std::string, double>& x) const;
    std::unordered_map<int, double> predict_proba_one(
        const std::unordered_map<std::string, double>& x) const;
    void reset();

    double learning_rate;
    double l2;

private:
    std::unordered_map<int, std::unordered_map<std::string, double>> weights;
    std::unordered_map<int, double> bias;
    std::set<int> seen_classes;

    double sigmoid(double z) const;
    double score(int cls, const std::unordered_map<std::string, double>& x) const;
};
