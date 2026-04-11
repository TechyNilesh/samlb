#pragma once
#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include "../core/gaussian_estimator.h"
#include "../core/hoeffding_bound.h"

// Hoeffding Tree Regressor — splits on variance reduction (SDR)
// Ikonomovska et al., 2011

struct HTRegNode {
    bool is_leaf = true;
    GaussianEstimator target_stats;   // for y
    std::unordered_map<std::string, GaussianEstimator> feat_stats;  // for x
    double total_weight = 0.0;
    long long n = 0;

    // Leaf linear model: y_hat = sum_i(w_i * x_i) + bias
    std::unordered_map<std::string, double> weights;
    double bias = 0.0;

    // Split fields
    std::string split_feature;
    double split_value = 0.0;
    std::unique_ptr<HTRegNode> left;
    std::unique_ptr<HTRegNode> right;
};

class HoeffdingTreeRegressor {
public:
    HoeffdingTreeRegressor(
        int    grace_period     = 200,
        double split_confidence = 1e-7,
        double tie_threshold    = 0.05,
        int    max_depth        = 20,
        double learning_rate    = 0.01
    );

    void   learn_one(const std::unordered_map<std::string, double>& x, double y);
    double predict_one(const std::unordered_map<std::string, double>& x) const;
    void   reset();

    int    grace_period;
    double split_confidence;
    double tie_threshold;
    int    max_depth;
    double learning_rate;

private:
    std::unique_ptr<HTRegNode> root;

    HTRegNode* traverse(const std::unordered_map<std::string, double>& x) const;
    void       update_leaf(HTRegNode* node,
                           const std::unordered_map<std::string, double>& x, double y);
    void       try_split(HTRegNode* node, int depth);
    double     sdr(HTRegNode* node, const std::string& feat, double threshold) const;
};
