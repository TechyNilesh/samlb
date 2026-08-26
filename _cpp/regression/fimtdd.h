#pragma once
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core/estimator.h"

// FIMT-DD — Fast Incremental Model Tree with Drift Detection
// (Ikonomovska, Gama & Džeroski, DMKD 2011), ported from MOA's FIMTDD.java.
//
// The regression counterpart of the Hoeffding Adaptive Tree, and the base
// learner both SGBT and SGBR are defined over. Three things distinguish it
// from HoeffdingTreeRegressor, and they are why that tree under-splits:
//
//  1. **E-BST attribute observers.** Every distinct value a feature has taken
//     in a leaf is kept in a binary search tree carrying the prefix sums of
//     (n, sum(y), sum(y^2)) either side of it, so *every observed value* is an
//     exact split candidate. HoeffdingTreeRegressor instead summarises each
//     feature with a Gaussian and tries a handful of interpolated points.
//  2. **The ratio form of the Hoeffding test.** A split happens when
//     SDR(second) / SDR(best) < 1 - epsilon, not when the difference of merits
//     exceeds epsilon. On variance reduction, where merits are unbounded and
//     scale with the target, the ratio is the scale-free comparison.
//  3. **Page-Hinckley drift detection with alternate subtrees.** Each inner
//     node watches its own normalised absolute error; on drift it grows a
//     replacement subtree in the background and promotes it once the
//     faded-error ratio favours it. This is the "tree replacement strategy"
//     SGBT's drift recovery is credited to.
//
// Leaves carry a perceptron fitted in a globally normalised space, ((x-mu)/3s
// and (y-mu)/3s), matching the reference; leaf_prediction="mean" reproduces
// MOA's regressionTree flag (-e), which turns the perceptron off.
//
// The perceptron diverges. Because it normalises by *running* global feature
// statistics, an early instance seen while sigma is still near zero produces a
// huge normalised input, the weight update overshoots, and nothing afterwards
// bounds the output: on ailerons the reference form predicts 53.6 where the
// stream never exceeds 1.0, for an R^2 of -21. Both SGBT and SGBR pass "-e",
// which sidesteps it by not using the perceptron at all.
//   leaf_prediction:
//     "adaptive"   (default) whichever of mean/perceptron has the lower error
//                  in this leaf so far, with the perceptron bounded to the
//                  leaf's own observed target range. Same remedy already
//                  applied to HoeffdingTreeRegressor in 0.2.0.
//     "mean"       the reference's -e, and what SGBT/SGBR use.
//     "perceptron" the reference's model tree verbatim, divergence included.
//                  Kept so the defect stays reproducible, not for use.
//
// Not ported: E-BST pruning of unpromising split points (removeBadSplits), a
// memory optimisation that does not change which split is chosen; and nominal
// attributes, which the reference also ignores.
class FIMTDDRegressor : public IRegressor {
public:
    FIMTDDRegressor(int grace_period = 200,
                    double split_confidence = 1e-7,
                    double tie_threshold = 0.05,
                    int max_depth = 20,
                    std::string leaf_prediction = "adaptive",
                    double learning_ratio = 0.02,
                    double learning_rate_decay = 0.001,
                    bool learning_ratio_const = false,
                    double page_hinckley_alpha = 0.005,
                    double page_hinckley_threshold = 50.0,
                    double alternate_tree_fading_factor = 0.995,
                    int alternate_tree_t_min = 150,
                    int alternate_tree_time = 1500,
                    bool drift_detection = true,
                    int seed = 1);

    void   learn_one(const Features& x, double y) override;
    double predict_one(const Features& x) const override;
    void   reset() override;

    int    n_leaves() const;
    int    n_splits() const { return split_count_; }

    int    grace_period;
    double split_confidence;
    double tie_threshold;
    int    max_depth;
    std::string leaf_prediction;
    double learning_ratio;
    double learning_rate_decay;
    bool   learning_ratio_const;
    double page_hinckley_alpha;
    double page_hinckley_threshold;
    double alternate_tree_fading_factor;
    int    alternate_tree_t_min;
    int    alternate_tree_time;
    bool   drift_detection;
    int    seed;

private:
    // ── E-BST: every observed value is an exact split candidate ─────────────
    struct EBSTNode {
        double value = 0.0;
        // Prefix sums of (weight, sum y, sum y^2) for instances that went left
        // (value <= this) and right (value > this) at this node.
        double left_n = 0.0, left_sum = 0.0, left_sq = 0.0;
        double right_n = 0.0, right_sum = 0.0, right_sq = 0.0;
        std::unique_ptr<EBSTNode> left, right;
    };

    struct Stats {
        double n = 0.0, sum = 0.0, sq = 0.0;
        void add(double y, double w) { n += w; sum += w * y; sq += w * y * y; }
        double sd() const;
    };

    struct Perceptron {
        std::unordered_map<std::string, double> weights;
        double bias = 0.0;
        double instances = 0.0;
        bool   initialised = false;
    };

    struct Node {
        // Leaves and inner nodes share the statistics the drift test needs.
        bool   is_leaf = true;
        Stats  stats;
        double abs_errors = 0.0;       // sum of normalised |error| on the path
        double examples = 0.0;

        // Leaf only
        std::unordered_map<std::string, std::unique_ptr<EBSTNode>> observers;
        double examples_at_last_split = 0.0;
        Perceptron model;
        // Squared error each leaf predictor would have accumulated, scored
        // before learning from the instance, plus the target range this leaf
        // has actually seen — the perceptron may not predict outside it.
        double mean_sse = 0.0;
        double perceptron_sse = 0.0;
        double y_min = 0.0;
        double y_max = 0.0;
        bool   y_seen = false;

        // Inner only
        std::string split_feature;
        double      split_value = 0.0;
        std::unique_ptr<Node> left, right;

        // Drift state (inner only)
        bool   change_detection = true;
        double ph_sum = 0.0;
        double ph_min = 0.0;
        bool   ph_started = false;
        std::unique_ptr<Node> alternate;
        double loss_examples = 0.0;
        double loss_faded_original = 0.0;
        double loss_faded_alternate = 0.0;
        double previous_weight = 0.0;
    };

    struct SplitSuggestion {
        std::string feature;
        double      value = 0.0;
        double      merit = -1.0;
    };

    // tree-level statistics, used to normalise the perceptron's inputs/target
    double normalise_target(double value) const;
    double normalised_error(double y, double prediction) const;

    static void  ebst_insert(std::unique_ptr<EBSTNode>& node, double value,
                             double y, double w);
    SplitSuggestion best_split(const Node& leaf) const;
    void            ebst_scan(const EBSTNode* node, const Stats& total,
                              const std::string& feature, Stats running,
                              SplitSuggestion& best, SplitSuggestion& second) const;

    double perceptron_predict(const Perceptron& model, const Features& x) const;
    void   perceptron_update(Perceptron& model, const Features& x, double y);

    double node_predict(const Node* node, const Features& x) const;
    void   learn_at(Node* node, const Features& x, double y, double prediction,
                    double normal_error, bool growth_allowed, bool in_alternate,
                    int depth);
    void   learn_leaf(Node* node, const Features& x, double y, bool growth_allowed,
                      int depth);
    void   attempt_split(Node* leaf, int depth);
    std::unique_ptr<Node> new_leaf(const Perceptron* inherit) const;

    bool building_model_tree() const { return leaf_prediction != "mean"; }
    double leaf_predict(const Node* leaf, const Features& x) const;

    std::unique_ptr<Node> root_;
    Stats  target_;                                   // global target stats
    std::unordered_map<std::string, Stats> attr_;     // global per-feature stats
    int    split_count_ = 0;
    mutable std::mt19937 rng_;
};
