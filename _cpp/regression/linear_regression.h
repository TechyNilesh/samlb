#pragma once
#include <unordered_map>
#include <string>

// Online Linear Regression via SGD on MSE loss.
class LinearRegression {
public:
    explicit LinearRegression(double learning_rate = 0.01, double l2 = 0.0);
    void   learn_one(const std::unordered_map<std::string, double>& x, double y);
    double predict_one(const std::unordered_map<std::string, double>& x) const;
    void   reset();

    double learning_rate, l2;

private:
    std::unordered_map<std::string, double> weights;
    double bias = 0.0;
};
