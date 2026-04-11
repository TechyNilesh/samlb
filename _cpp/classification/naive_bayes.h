#pragma once
#include <unordered_map>
#include <string>
#include <limits>
#include "../core/gaussian_estimator.h"

class NaiveBayes {
public:
    using Features = std::unordered_map<std::string, double>;

    NaiveBayes();
    void learn_one(const Features& x, int y);
    int  predict_one(const Features& x) const;
    std::unordered_map<int, double> predict_proba_one(const Features& x) const;
    void reset();

private:
    std::unordered_map<int, long long> class_counts;
    std::unordered_map<int, std::unordered_map<std::string, GaussianEstimator>> estimators;
    long long total_count = 0;
};
