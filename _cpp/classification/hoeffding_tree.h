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
//
// Split search mirrors MOA's moa.classifiers.trees.HoeffdingTree with its
// default GaussianNumericAttributeClassObserver / InfoGainSplitCriterion /
// GiniSplitCriterion: per (class, feature) Gaussian summaries plus the
// observed min/max, evenly-spaced split-point candidates between the
// global min/max, and merit computed on the exact resulting class
// distributions (estimated via the Gaussian CDF) rather than on a
// mean±3sd heuristic.

// Per (class, feature) numeric attribute observer. Mirrors MOA's
// GaussianNumericAttributeClassObserver: a running Gaussian plus the
// min/max value seen for this class, which let the split-weight estimate
// shortcut to an exact answer at the tails instead of trusting the CDF
// out where the Gaussian approximation is least reliable.
struct AttrObserver {
    GaussianEstimator est;
    double min_val = 0.0;
    double max_val = 0.0;

    void update(double x) {
        if (est.n <= 0.0) {
            min_val = x;
            max_val = x;
        } else if (x < min_val) {
            min_val = x;
        } else if (x > max_val) {
            max_val = x;
        }
        est.update(x);
    }
};

struct HTNode {
    bool is_leaf = true;
    // Leaf: per-class, per-feature Gaussian attribute observers
    std::unordered_map<int, std::unordered_map<std::string, AttrObserver>> stats;
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

    // Candidate split points evaluated per numeric feature, evenly spaced
    // between the observed min and max (MOA's numBinsOption, default 10).
    int    n_split_points = 10;

    // Leaf predictor: "mc" (majority class), "nb" (naive Bayes) or "nba"
    // (adaptive — per leaf, whichever has been more accurate so far).
    // "nba" matches MOA and river's default; "mc" is the original behaviour.
    std::string leaf_prediction = "nba";

    // Pre-pruning: reject a split whose winning merit doesn't beat the
    // "don't split" option (MOA's noPrePruneOption, inverted). Default false
    // matches MOA's default (pre-pruning on).
    bool no_pre_prune = false;

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
    int     majority_class(const HTNode* node) const;
};
