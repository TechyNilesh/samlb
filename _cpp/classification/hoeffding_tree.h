#pragma once
#include "../core/estimator.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <random>
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

    // Naive Bayes Adaptive: how often each leaf predictor would have been
    // right on the instances that reached this leaf. Whichever is ahead is
    // used at prediction time (MOA's NaiveBayesAdaptive, river's 'nba').
    double mc_correct = 0.0;
    double nb_correct = 0.0;

    // Split node fields
    std::string split_feature;
    double split_value = 0.0;
    std::unique_ptr<HTNode> left;   // <= split_value
    std::unique_ptr<HTNode> right;  // >  split_value
};

class HoeffdingTreeClassifier : public IClassifier {
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

    // Candidate split points evaluated per numeric feature. 1 = a single
    // threshold at the weighted mean.
    int    n_split_points = 1;

    // Leaf predictor: "mc" (majority class), "nb" (naive Bayes) or "nba"
    // (adaptive — per leaf, whichever has been more accurate so far).
    // "nba" matches MOA and river's default; "mc" is the original behaviour.
    std::string leaf_prediction = "nba";

    // Random subspace: number of features sampled at each split attempt.
    // 0 = evaluate every feature (plain VFDT). ARF sets this to sqrt(M)+1 and
    // resamples at every node, which is what separates it from SRP's single
    // fixed per-learner subspace.
    void set_subspace(int size, unsigned int seed);

private:
    int          subspace_size_ = 0;
    mutable std::mt19937 split_rng_{0};

    std::unique_ptr<HTNode> root;
    std::set<int> seen_classes;

    HTNode* traverse(const std::unordered_map<std::string, double>& x) const;
    // Class log-posteriors at a leaf under the leaf's Gaussian statistics.
    std::unordered_map<int, double> nb_log_proba(
        const HTNode* node, const std::unordered_map<std::string, double>& x) const;
    // True when this leaf should answer with naive Bayes rather than majority.
    bool use_nb(const HTNode* node) const;
    void    update_leaf(HTNode* node,
                        const std::unordered_map<std::string, double>& x, int y);
    void    try_split(HTNode* node, int depth);
    double  info_gain(HTNode* node, const std::string& feat, double threshold) const;
    double  gini(HTNode* node, const std::string& feat, double threshold) const;
    double  node_entropy(const std::unordered_map<int, double>& counts, double total) const;
    int     majority_class(const HTNode* node) const;
};
