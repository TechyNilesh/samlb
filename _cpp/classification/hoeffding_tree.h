#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include "../core/gaussian_estimator.h"
#include "../core/hoeffding_bound.h"

// Hoeffding Tree Classifier (VFDT — Very Fast Decision Tree)
// Domingos & Hulten, KDD 2000

struct HTNode {
    bool is_leaf = true;
    // Leaf: per-class, per-feature Gaussian estimators
    std::unordered_map<int, std::unordered_map<std::string, GaussianEstimator>> stats;
    std::unordered_map<int, double> class_counts;
    double total_weight = 0.0;

    // Split node fields
    std::string split_feature;
    double split_value = 0.0;
    std::unique_ptr<HTNode> left;   // <= split_value
    std::unique_ptr<HTNode> right;  // >  split_value
};

class HoeffdingTreeClassifier {
public:
    HoeffdingTreeClassifier(
        int    grace_period      = 200,
        double split_confidence  = 1e-7,
        double tie_threshold     = 0.05,
        int    nb_threshold      = 0,
        int    max_depth         = 20,
        const std::string& split_criterion = "info_gain"
    );

    void learn_one(const std::unordered_map<std::string, double>& x, int y);
    int  predict_one(const std::unordered_map<std::string, double>& x) const;
    std::unordered_map<int, double> predict_proba_one(
        const std::unordered_map<std::string, double>& x) const;
    void reset();

    int    grace_period;
    double split_confidence;
    double tie_threshold;
    int    nb_threshold;
    int    max_depth;
    std::string split_criterion;

private:
    std::unique_ptr<HTNode> root;
    std::set<int> seen_classes;

    HTNode* traverse(const std::unordered_map<std::string, double>& x) const;
    void    update_leaf(HTNode* node,
                        const std::unordered_map<std::string, double>& x, int y);
    void    try_split(HTNode* node, int depth);
    double  info_gain(HTNode* node, const std::string& feat, double threshold) const;
    double  gini(HTNode* node, const std::string& feat, double threshold) const;
    double  node_entropy(const std::unordered_map<int, double>& counts, double total) const;
    int     majority_class(const HTNode* node) const;
};
