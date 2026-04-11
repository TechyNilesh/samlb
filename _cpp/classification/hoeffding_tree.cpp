#include "hoeffding_tree.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor / reset
// ---------------------------------------------------------------------------

HoeffdingTreeClassifier::HoeffdingTreeClassifier(
    int    grace_period_,
    double split_confidence_,
    double tie_threshold_,
    int    nb_threshold_,
    int    max_depth_,
    const std::string& split_criterion_)
    : grace_period(grace_period_)
    , split_confidence(split_confidence_)
    , tie_threshold(tie_threshold_)
    , nb_threshold(nb_threshold_)
    , max_depth(max_depth_)
    , split_criterion(split_criterion_)
{
    root = std::make_unique<HTNode>();
}

void HoeffdingTreeClassifier::reset() {
    root = std::make_unique<HTNode>();
    seen_classes.clear();
}

// ---------------------------------------------------------------------------
// Traversal: walk split nodes until we reach a leaf
// ---------------------------------------------------------------------------

HTNode* HoeffdingTreeClassifier::traverse(
    const std::unordered_map<std::string, double>& x) const
{
    HTNode* node = root.get();
    while (!node->is_leaf) {
        auto it = x.find(node->split_feature);
        double val = (it != x.end()) ? it->second : 0.0;
        node = (val <= node->split_value) ? node->left.get()
                                          : node->right.get();
    }
    return node;
}

// ---------------------------------------------------------------------------
// update_leaf: accumulate statistics at a leaf
// ---------------------------------------------------------------------------

void HoeffdingTreeClassifier::update_leaf(
    HTNode* node,
    const std::unordered_map<std::string, double>& x,
    int y)
{
    node->class_counts[y] += 1.0;
    node->total_weight     += 1.0;
    for (auto& [feat, val] : x)
        node->stats[y][feat].update(val);
}

// ---------------------------------------------------------------------------
// Helper: entropy of a count distribution
// ---------------------------------------------------------------------------

double HoeffdingTreeClassifier::node_entropy(
    const std::unordered_map<int, double>& counts, double total) const
{
    if (total <= 0.0) return 0.0;
    double h = 0.0;
    for (auto& [cls, cnt] : counts) {
        if (cnt > 0.0) {
            double p = cnt / total;
            h -= p * std::log2(p);
        }
    }
    return h;
}

// ---------------------------------------------------------------------------
// info_gain: H(parent) - (n_L/n)*H(left) - (n_R/n)*H(right)
// Uses the per-class feature Gaussian mean as a proxy split distribution.
// For a candidate split on (feat, threshold):
//   - instances from class c with estimator mean <= threshold go left
//   - we approximate left/right class counts proportionally using the
//     Gaussian CDF fraction of each class's estimator
// ---------------------------------------------------------------------------

static double gaussian_cdf(double x, double mean, double std_dev) {
    if (std_dev <= 0.0) return (x >= mean) ? 1.0 : 0.0;
    return 0.5 * (1.0 + std::erf((x - mean) / (std_dev * std::sqrt(2.0))));
}

double HoeffdingTreeClassifier::info_gain(
    HTNode* node, const std::string& feat, double threshold) const
{
    double n_total = node->total_weight;
    if (n_total <= 0.0) return 0.0;

    // Parent entropy
    double parent_h = node_entropy(node->class_counts, n_total);

    // Build approximate left/right class counts
    std::unordered_map<int, double> left_counts, right_counts;
    double n_left = 0.0, n_right = 0.0;

    for (auto& [cls, cnt] : node->class_counts) {
        auto it = node->stats.find(cls);
        double frac_left = 0.5; // default: unknown feature → split 50/50
        if (it != node->stats.end()) {
            auto fit = it->second.find(feat);
            if (fit != it->second.end() && fit->second.n > 0.0) {
                frac_left = gaussian_cdf(threshold,
                                         fit->second.mean,
                                         fit->second.std_dev());
            }
        }
        double l = cnt * frac_left;
        double r = cnt * (1.0 - frac_left);
        left_counts[cls]  = l;
        right_counts[cls] = r;
        n_left  += l;
        n_right += r;
    }

    double h_left  = node_entropy(left_counts,  n_left);
    double h_right = node_entropy(right_counts, n_right);

    double gain = parent_h
                - (n_left  / n_total) * h_left
                - (n_right / n_total) * h_right;
    return gain;
}

// ---------------------------------------------------------------------------
// gini: Gini(parent) - (n_L/n)*Gini(left) - (n_R/n)*Gini(right)
// ---------------------------------------------------------------------------

static double gini_impurity(const std::unordered_map<int, double>& counts,
                             double total) {
    if (total <= 0.0) return 0.0;
    double g = 1.0;
    for (auto& [cls, cnt] : counts) {
        double p = cnt / total;
        g -= p * p;
    }
    return g;
}

double HoeffdingTreeClassifier::gini(
    HTNode* node, const std::string& feat, double threshold) const
{
    double n_total = node->total_weight;
    if (n_total <= 0.0) return 0.0;

    double parent_g = gini_impurity(node->class_counts, n_total);

    std::unordered_map<int, double> left_counts, right_counts;
    double n_left = 0.0, n_right = 0.0;

    for (auto& [cls, cnt] : node->class_counts) {
        auto it = node->stats.find(cls);
        double frac_left = 0.5;
        if (it != node->stats.end()) {
            auto fit = it->second.find(feat);
            if (fit != it->second.end() && fit->second.n > 0.0) {
                frac_left = gaussian_cdf(threshold,
                                         fit->second.mean,
                                         fit->second.std_dev());
            }
        }
        double l = cnt * frac_left;
        double r = cnt * (1.0 - frac_left);
        left_counts[cls]  = l;
        right_counts[cls] = r;
        n_left  += l;
        n_right += r;
    }

    double g_reduction = parent_g
                       - (n_left  / n_total) * gini_impurity(left_counts,  n_left)
                       - (n_right / n_total) * gini_impurity(right_counts, n_right);
    return g_reduction;
}

// ---------------------------------------------------------------------------
// majority_class: return the class with the highest count at a leaf
// ---------------------------------------------------------------------------

int HoeffdingTreeClassifier::majority_class(const HTNode* node) const {
    if (node->class_counts.empty()) return 0;
    int    best_cls   = node->class_counts.begin()->first;
    double best_count = -1.0;
    for (auto& [cls, cnt] : node->class_counts) {
        if (cnt > best_count) { best_count = cnt; best_cls = cls; }
    }
    return best_cls;
}

// ---------------------------------------------------------------------------
// try_split: attempt to split a leaf using Hoeffding bound
// ---------------------------------------------------------------------------

void HoeffdingTreeClassifier::try_split(HTNode* node, int depth) {
    if (depth >= max_depth) return;
    if (node->total_weight < 2.0) return;

    // Collect all feature names seen at this leaf
    std::set<std::string> features;
    for (auto& [cls, feat_map] : node->stats)
        for (auto& [feat, est] : feat_map)
            features.insert(feat);

    if (features.empty()) return;

    // Number of classes seen at this node
    int n_classes = static_cast<int>(seen_classes.size());
    if (n_classes < 2) return;

    // Hoeffding bound range depends on criterion
    // For info_gain: range = log2(n_classes) (max possible entropy)
    // For gini:      range = 1.0
    double range = (split_criterion == "gini")
                 ? 1.0
                 : std::log2(static_cast<double>(n_classes));

    double epsilon = hoeffding_bound(range, split_confidence, node->total_weight);

    // Evaluate each feature at its overall weighted mean as the split threshold
    double best_score  = -std::numeric_limits<double>::infinity();
    double second_best = -std::numeric_limits<double>::infinity();
    std::string best_feature;
    double       best_threshold = 0.0;

    for (const auto& feat : features) {
        // Compute overall feature mean across all classes (weighted by class count)
        double w_mean = 0.0, w_total = 0.0;
        for (auto& [cls, cnt] : node->class_counts) {
            auto it = node->stats.find(cls);
            if (it == node->stats.end()) continue;
            auto fit = it->second.find(feat);
            if (fit == it->second.end() || fit->second.n <= 0.0) continue;
            w_mean  += cnt * fit->second.mean;
            w_total += cnt;
        }
        if (w_total <= 0.0) continue;
        double threshold = w_mean / w_total;

        double score = (split_criterion == "gini")
                     ? gini(node, feat, threshold)
                     : info_gain(node, feat, threshold);

        if (score > best_score) {
            second_best    = best_score;
            best_score     = score;
            best_feature   = feat;
            best_threshold = threshold;
        } else if (score > second_best) {
            second_best = score;
        }
    }

    if (best_feature.empty()) return;

    // Check Hoeffding bound condition
    bool do_split = false;
    if (best_score <= 0.0) return; // No gain — nothing to split on

    double delta = best_score - second_best;
    if (delta > epsilon) {
        do_split = true;
    } else if (delta < tie_threshold && epsilon < tie_threshold) {
        // Tie-breaking: split anyway when bound is tight
        do_split = true;
    }

    if (!do_split) return;

    // Perform the split: convert leaf → split node, create two child leaves
    node->is_leaf      = false;
    node->split_feature = best_feature;
    node->split_value   = best_threshold;

    node->left  = std::make_unique<HTNode>();
    node->right = std::make_unique<HTNode>();

    // Redistribute accumulated stats proportionally to children
    for (auto& [cls, cnt] : node->class_counts) {
        auto it = node->stats.find(cls);
        double frac_left = 0.5;
        if (it != node->stats.end()) {
            auto fit = it->second.find(best_feature);
            if (fit != it->second.end() && fit->second.n > 0.0) {
                frac_left = gaussian_cdf(best_threshold,
                                          fit->second.mean,
                                          fit->second.std_dev());
            }
        }
        double l = cnt * frac_left;
        double r = cnt * (1.0 - frac_left);
        if (l > 0.0) node->left->class_counts[cls]  = l;
        if (r > 0.0) node->right->class_counts[cls] = r;
        node->left->total_weight  += l;
        node->right->total_weight += r;
        // Copy feature stats proportionally
        if (it != node->stats.end()) {
            for (auto& [feat2, est] : it->second) {
                if (l > 0.0) {
                    // Approximate: copy estimator scaled by fraction
                    GaussianEstimator g_l;
                    g_l.n    = est.n * frac_left;
                    g_l.mean = est.mean;
                    g_l.M2   = est.M2 * frac_left;
                    node->left->stats[cls][feat2] = g_l;
                }
                if (r > 0.0) {
                    GaussianEstimator g_r;
                    g_r.n    = est.n * (1.0 - frac_left);
                    g_r.mean = est.mean;
                    g_r.M2   = est.M2 * (1.0 - frac_left);
                    node->right->stats[cls][feat2] = g_r;
                }
            }
        }
    }

    // Clear leaf-level data from the now-internal node
    node->stats.clear();
    node->class_counts.clear();
    node->total_weight = 0.0;
}

// ---------------------------------------------------------------------------
// learn_one
// ---------------------------------------------------------------------------

void HoeffdingTreeClassifier::learn_one(
    const std::unordered_map<std::string, double>& x, int y)
{
    seen_classes.insert(y);
    if (!root) root = std::make_unique<HTNode>();

    // We need depth information during traversal for try_split,
    // so we do a manual traversal that also records depth.
    HTNode* node  = root.get();
    int     depth = 0;
    while (!node->is_leaf) {
        auto it  = x.find(node->split_feature);
        double v = (it != x.end()) ? it->second : 0.0;
        node = (v <= node->split_value) ? node->left.get()
                                        : node->right.get();
        ++depth;
    }

    update_leaf(node, x, y);

    // Try to split every grace_period instances at this leaf
    if (static_cast<long long>(node->total_weight) % grace_period == 0) {
        try_split(node, depth);
    }
}

// ---------------------------------------------------------------------------
// predict_one
// ---------------------------------------------------------------------------

int HoeffdingTreeClassifier::predict_one(
    const std::unordered_map<std::string, double>& x) const
{
    if (!root) return 0;
    HTNode* node = traverse(x);

    // Use Naive Bayes at leaf if enough samples and nb_threshold > 0
    if (nb_threshold > 0 && node->total_weight >= nb_threshold) {
        int    best_cls  = 0;
        double best_prob = -std::numeric_limits<double>::infinity();
        double n_total   = node->total_weight;
        for (auto& [cls, cnt] : node->class_counts) {
            double log_p = std::log(cnt / n_total);
            auto it = node->stats.find(cls);
            if (it != node->stats.end()) {
                for (auto& [feat, val] : x) {
                    auto fit = it->second.find(feat);
                    if (fit != it->second.end()) {
                        double p = fit->second.probability_density(val);
                        log_p += std::log(p > 1e-300 ? p : 1e-300);
                    }
                }
            }
            if (log_p > best_prob) { best_prob = log_p; best_cls = cls; }
        }
        return best_cls;
    }

    return majority_class(node);
}

// ---------------------------------------------------------------------------
// predict_proba_one
// ---------------------------------------------------------------------------

std::unordered_map<int, double> HoeffdingTreeClassifier::predict_proba_one(
    const std::unordered_map<std::string, double>& x) const
{
    std::unordered_map<int, double> proba;
    if (!root) return proba;
    HTNode* node   = traverse(x);
    double  n_total = node->total_weight;

    if (n_total <= 0.0) {
        // Return uniform over seen classes
        if (!seen_classes.empty()) {
            double u = 1.0 / seen_classes.size();
            for (int c : seen_classes) proba[c] = u;
        }
        return proba;
    }

    if (nb_threshold > 0 && node->total_weight >= nb_threshold) {
        // Naive Bayes probabilities
        double total_prob = 0.0;
        for (auto& [cls, cnt] : node->class_counts) {
            double p = cnt / n_total;
            auto it = node->stats.find(cls);
            if (it != node->stats.end()) {
                for (auto& [feat, val] : x) {
                    auto fit = it->second.find(feat);
                    if (fit != it->second.end()) {
                        double pd = fit->second.probability_density(val);
                        p *= (pd > 1e-300 ? pd : 1e-300);
                    }
                }
            }
            proba[cls]  = p;
            total_prob += p;
        }
        if (total_prob > 0.0)
            for (auto& [k, v] : proba) v /= total_prob;
    } else {
        // Majority-based: normalised class counts
        for (auto& [cls, cnt] : node->class_counts)
            proba[cls] = cnt / n_total;
    }

    // Ensure all seen classes are present
    for (int c : seen_classes)
        if (proba.find(c) == proba.end())
            proba[c] = 0.0;

    return proba;
}
