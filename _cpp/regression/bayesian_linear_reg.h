#pragma once
#include <unordered_map>
#include <string>
#include <vector>

// Online Bayesian Linear Regression.
// Maintains posterior mean and precision matrix (diagonal approximation).
class BayesianLinearRegression {
public:
    explicit BayesianLinearRegression(double alpha = 1.0, double beta = 1.0);
    void   learn_one(const std::unordered_map<std::string, double>& x, double y);
    double predict_one(const std::unordered_map<std::string, double>& x) const;
    void   reset();

    double alpha;  // prior precision
    double beta;   // noise precision

private:
    // Diagonal posterior precision per feature + bias
    std::unordered_map<std::string, double> precision;  // S_inv (diagonal)
    std::unordered_map<std::string, double> mean;       // posterior mean
    double precision_bias = 0.0;
    double mean_bias      = 0.0;
};
