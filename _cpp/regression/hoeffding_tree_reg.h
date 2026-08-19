#pragma once
#include "../core/estimator.h"
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

    // Leaf linear model, fitted in standardised space:
    //   z = bias + sum_i(w_i * (x_i - mu_i) / sd_i),  y_hat = mu_y + z * sd_y
    // Working in standardised units keeps the update independent of the target
    // scale — raw-target SGD diverges (or, with a fixed gradient clip, crawls)
    // whenever |y| is large.
    std::unordered_map<std::string, double> weights;
    double bias = 0.0;

    // Squared error each leaf predictor would have accumulated, scored before
    // learning from the instance. The better one answers at prediction time.
    double mean_sse = 0.0;
    double lin_sse  = 0.0;

    // Split fields
    std::string split_feature;
    double split_value = 0.0;
    std::unique_ptr<HTRegNode> left;
    std::unique_ptr<HTRegNode> right;
};

class HoeffdingTreeRegressor : public IRegressor {
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

    // Leaf predictor: "adaptive" (whichever of mean/linear has lower error so
    // far), "mean", or "linear". Adaptive matches FIMT-DD and river's default,
    // and is what stops a diverging linear model from destroying the leaf.
    std::string leaf_prediction = "adaptive";

    int    grace_period;
    double split_confidence;
    double tie_threshold;
    int    max_depth;
    double learning_rate;

private:
    // Leaf linear model evaluated on the target scale.
    double linear_predict(const HTRegNode* node,
                          const std::unordered_map<std::string, double>& x) const;
    std::unique_ptr<HTRegNode> root;

    HTRegNode* traverse(const std::unordered_map<std::string, double>& x) const;
    void       update_leaf(HTRegNode* node,
                           const std::unordered_map<std::string, double>& x, double y);
    void       try_split(HTRegNode* node, int depth);
    double     sdr(HTRegNode* node, const std::string& feat, double threshold) const;
};
