#pragma once
#include <unordered_map>
#include <string>

// Passive Aggressive Regressor (PA-I, epsilon-insensitive loss).
class PassiveAggressiveRegressor {
public:
    explicit PassiveAggressiveRegressor(double C = 1.0, double epsilon = 0.1);
    void   learn_one(const std::unordered_map<std::string, double>& x, double y);
    double predict_one(const std::unordered_map<std::string, double>& x) const;
    void   reset();

    double C, epsilon;

private:
    std::unordered_map<std::string, double> weights;
    double bias = 0.0;
};
