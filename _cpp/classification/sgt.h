#pragma once
#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include "../core/gaussian_estimator.h"

// Stochastic Gradient Tree Classifier (Gouk et al., ECML-PKDD 2019)
// A differentiable streaming decision tree that uses gradient-based
// split finding rather than Hoeffding bound.

struct SGTNode {
    bool is_leaf = true;
    // Leaf: gradient and hessian accumulators per feature x split
    std::unordered_map<std::string, GaussianEstimator> grad_stats;
    std::unordered_map<std::string, GaussianEstimator> hess_stats;
    // Leaf weight (prediction)
    double weight   = 0.0;
    double grad_sum = 0.0;
    double hess_sum = 0.0;
    long long n     = 0;

    // Split fields
    std::string split_feature;
    double split_value = 0.0;
    std::unique_ptr<SGTNode> left;
    std::unique_ptr<SGTNode> right;
};

class SGTClassifier {
public:
    SGTClassifier(
        double learning_rate = 0.1,
        double lambda        = 0.1,   // L2 regularisation on leaf weights
        int    grace_period  = 200,
        int    max_depth     = 6
    );

    void learn_one(const std::unordered_map<std::string, double>& x, int y);
    int  predict_one(const std::unordered_map<std::string, double>& x) const;
    std::unordered_map<int, double> predict_proba_one(
        const std::unordered_map<std::string, double>& x) const;
    void reset();

    double learning_rate, lambda;
    int    grace_period, max_depth;

private:
    std::unique_ptr<SGTNode> root;
    std::set<int> seen_classes;

    SGTNode* traverse(const std::unordered_map<std::string, double>& x) const;
    void     update_leaf(SGTNode* node,
                         const std::unordered_map<std::string, double>& x,
                         double grad, double hess);
    void     try_split(SGTNode* node, int depth);
    double   gain(double G_L, double H_L, double G_R, double H_R) const;
};
