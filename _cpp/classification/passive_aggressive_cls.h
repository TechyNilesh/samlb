#pragma once
#include <unordered_map>
#include <string>
#include <set>

// Passive Aggressive Classifier (PA-I variant, Crammer et al. JMLR 2006).
// Supports binary and multiclass (one-vs-rest).
class PassiveAggressiveClassifier {
public:
    explicit PassiveAggressiveClassifier(double C = 1.0);
    void learn_one(const std::unordered_map<std::string, double>& x, int y);
    int  predict_one(const std::unordered_map<std::string, double>& x) const;
    std::unordered_map<int, double> predict_proba_one(
        const std::unordered_map<std::string, double>& x) const;
    void reset();

    double C;  // aggressiveness

private:
    std::unordered_map<int, std::unordered_map<std::string, double>> weights;
    std::unordered_map<int, double> bias;
    std::set<int> seen_classes;

    double dot(int cls, const std::unordered_map<std::string, double>& x) const;
    double norm_sq(const std::unordered_map<std::string, double>& x) const;
};
