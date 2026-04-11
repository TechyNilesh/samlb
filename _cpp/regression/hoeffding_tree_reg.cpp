#include "hoeffding_tree_reg.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <set>

// ---------------------------------------------------------------------------
// Constructor / reset
// ---------------------------------------------------------------------------

HoeffdingTreeRegressor::HoeffdingTreeRegressor(
    int    grace_period_,
    double split_confidence_,
    double tie_threshold_,
    int    max_depth_,
    double learning_rate_)
    : grace_period(grace_period_)
    , split_confidence(split_confidence_)
    , tie_threshold(tie_threshold_)
    , max_depth(max_depth_)
    , learning_rate(learning_rate_)
{
    root = std::make_unique<HTRegNode>();
}

void HoeffdingTreeRegressor::reset() {
    root = std::make_unique<HTRegNode>();
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

HTRegNode* HoeffdingTreeRegressor::traverse(
    const std::unordered_map<std::string, double>& x) const
{
    HTRegNode* node = root.get();
    while (!node->is_leaf) {
        auto it   = x.find(node->split_feature);
        double v  = (it != x.end()) ? it->second : 0.0;
        node = (v <= node->split_value) ? node->left.get()
                                        : node->right.get();
    }
    return node;
}

// ---------------------------------------------------------------------------
// update_leaf: accumulate statistics and update linear model via SGD
// ---------------------------------------------------------------------------

void HoeffdingTreeRegressor::update_leaf(
    HTRegNode* node,
    const std::unordered_map<std::string, double>& x, double y)
{
    node->target_stats.update(y);
    node->total_weight += 1.0;
    node->n            += 1;

    for (auto& [feat, val] : x)
        node->feat_stats[feat].update(val);

    // Online SGD update for the leaf linear model (MSE loss)
    // with gradient clipping to prevent divergence on high-dim data.
    static constexpr double GRAD_CLIP = 10.0;
    double y_hat = node->bias;
    for (auto& [feat, val] : x) {
        auto it = node->weights.find(feat);
        if (it != node->weights.end()) y_hat += it->second * val;
    }
    double err = y - y_hat;
    err = std::max(-GRAD_CLIP, std::min(GRAD_CLIP, err));
    for (auto& [feat, val] : x) {
        double grad = std::max(-GRAD_CLIP, std::min(GRAD_CLIP, err * val));
        node->weights[feat] += learning_rate * grad;
    }
    node->bias += learning_rate * err;
}

// ---------------------------------------------------------------------------
// sdr: Standard Deviation Reduction
//   SDR = std(parent) - (n_L/n)*std(left) - (n_R/n)*std(right)
//
// We approximate child std devs using the conditional Gaussian assumption:
// For a feature F with overall mean mu_f and std sigma_f:
//   - instances with F <= threshold are "left" (fraction p_L ~ Gaussian CDF)
//   - We split target_stats proportionally for the child variance estimate.
//
// A more accurate approach requires joint (y, f) statistics; here we use
// the unconditional y variance scaled by the remaining variance fraction
// after the split (a practical approximation used in streaming settings).
// ---------------------------------------------------------------------------

static double gaussian_cdf_reg(double x, double mean, double std_dev) {
    if (std_dev <= 0.0) return (x >= mean) ? 1.0 : 0.0;
    return 0.5 * (1.0 + std::erf((x - mean) / (std_dev * std::sqrt(2.0))));
}

double HoeffdingTreeRegressor::sdr(
    HTRegNode* node, const std::string& feat, double threshold) const
{
    double n_total = node->total_weight;
    if (n_total < 2.0) return 0.0;

    double parent_std = node->target_stats.std_dev();
    if (parent_std <= 0.0) return 0.0;

    // Fraction of instances going left, estimated from feature distribution
    auto fit = node->feat_stats.find(feat);
    double frac_left = 0.5;
    if (fit != node->feat_stats.end() && fit->second.n > 0.0) {
        frac_left = gaussian_cdf_reg(threshold,
                                     fit->second.mean,
                                     fit->second.std_dev());
    }
    frac_left = std::max(0.01, std::min(0.99, frac_left));
    double frac_right = 1.0 - frac_left;

    // Estimate child standard deviations.
    // We use a two-component Gaussian mixture decomposition:
    //   var(Y) = frac_L*(var_L + mu_L^2) + frac_R*(var_R + mu_R^2) - mu_Y^2
    // Assuming the split is on an independent feature, we approximate child
    // variances as equal to the parent variance (worst case), giving:
    //   SDR = parent_std * (1 - frac_L - frac_R) = 0 — not useful.
    //
    // Instead we use the feature-correlated variance reduction proxy:
    //   std_L ≈ parent_std * sqrt(frac_L)
    //   std_R ≈ parent_std * sqrt(frac_R)
    // This is equivalent to treating each child as receiving a proportion
    // of the total spread, which rewards more balanced splits.
    double std_left  = parent_std * std::sqrt(frac_left);
    double std_right = parent_std * std::sqrt(frac_right);

    double sdr_val = parent_std
                   - frac_left  * std_left
                   - frac_right * std_right;
    return sdr_val;
}

// ---------------------------------------------------------------------------
// try_split
// ---------------------------------------------------------------------------

void HoeffdingTreeRegressor::try_split(HTRegNode* node, int depth) {
    if (depth >= max_depth) return;
    if (node->n < 2) return;

    double parent_std = node->target_stats.std_dev();
    if (parent_std <= 0.0) return;  // no variance → nothing to split

    // Collect candidate features
    std::set<std::string> features;
    for (auto& [feat, est] : node->feat_stats)
        features.insert(feat);
    if (features.empty()) return;

    // Hoeffding bound: range = parent_std (an estimate of the random variable range)
    double range   = parent_std;
    double epsilon = hoeffding_bound(range, split_confidence,
                                     static_cast<double>(node->n));

    double best_sdr    = -std::numeric_limits<double>::infinity();
    double second_sdr  = -std::numeric_limits<double>::infinity();
    std::string best_feature;
    double       best_threshold = 0.0;

    for (const auto& feat : features) {
        auto fit = node->feat_stats.find(feat);
        if (fit == node->feat_stats.end() || fit->second.n <= 0.0) continue;
        double threshold = fit->second.mean;  // split at feature mean

        double s = sdr(node, feat, threshold);
        if (s > best_sdr) {
            second_sdr    = best_sdr;
            best_sdr      = s;
            best_feature  = feat;
            best_threshold = threshold;
        } else if (s > second_sdr) {
            second_sdr = s;
        }
    }

    if (best_feature.empty() || best_sdr <= 0.0) return;

    double delta = best_sdr - second_sdr;
    bool do_split = false;
    if (delta > epsilon) {
        do_split = true;
    } else if (delta < tie_threshold && epsilon < tie_threshold) {
        do_split = true;
    }
    if (!do_split) return;

    // Estimate left fraction
    auto fit = node->feat_stats.find(best_feature);
    double frac_left = 0.5;
    if (fit != node->feat_stats.end() && fit->second.n > 0.0) {
        frac_left = gaussian_cdf_reg(best_threshold,
                                     fit->second.mean,
                                     fit->second.std_dev());
    }
    frac_left = std::max(0.01, std::min(0.99, frac_left));
    double frac_right = 1.0 - frac_left;

    // Convert leaf to split node
    node->is_leaf       = false;
    node->split_feature = best_feature;
    node->split_value   = best_threshold;

    node->left  = std::make_unique<HTRegNode>();
    node->right = std::make_unique<HTRegNode>();

    // Distribute target stats to children proportionally
    {
        auto& p = node->target_stats;

        // Left child
        auto& L = node->left->target_stats;
        L.n    = p.n    * frac_left;
        L.mean = p.mean;               // same mean approximation
        L.M2   = p.M2   * frac_left;

        // Right child
        auto& R = node->right->target_stats;
        R.n    = p.n    * frac_right;
        R.mean = p.mean;
        R.M2   = p.M2   * frac_right;
    }

    node->left->total_weight  = node->total_weight * frac_left;
    node->right->total_weight = node->total_weight * frac_right;
    node->left->n  = static_cast<long long>(node->n * frac_left);
    node->right->n = static_cast<long long>(node->n * frac_right);

    // Distribute feature stats proportionally
    for (auto& [feat2, est] : node->feat_stats) {
        GaussianEstimator gl, gr;
        gl.n = est.n * frac_left;  gl.mean = est.mean; gl.M2 = est.M2 * frac_left;
        gr.n = est.n * frac_right; gr.mean = est.mean; gr.M2 = est.M2 * frac_right;
        node->left->feat_stats[feat2]  = gl;
        node->right->feat_stats[feat2] = gr;
    }

    // Copy linear model to children (they start from the same model and
    // adapt further with new data from their respective sub-populations)
    node->left->weights  = node->weights;
    node->left->bias     = node->bias;
    node->right->weights = node->weights;
    node->right->bias    = node->bias;

    // Clear parent leaf data
    node->target_stats.reset();
    node->feat_stats.clear();
    node->weights.clear();
    node->bias         = 0.0;
    node->total_weight = 0.0;
    node->n            = 0;
}

// ---------------------------------------------------------------------------
// learn_one
// ---------------------------------------------------------------------------

void HoeffdingTreeRegressor::learn_one(
    const std::unordered_map<std::string, double>& x, double y)
{
    if (!root) root = std::make_unique<HTRegNode>();

    HTRegNode* node  = root.get();
    int        depth = 0;
    while (!node->is_leaf) {
        auto it  = x.find(node->split_feature);
        double v = (it != x.end()) ? it->second : 0.0;
        node = (v <= node->split_value) ? node->left.get()
                                        : node->right.get();
        ++depth;
    }

    update_leaf(node, x, y);

    if (node->n % grace_period == 0) {
        try_split(node, depth);
    }
}

// ---------------------------------------------------------------------------
// predict_one
// ---------------------------------------------------------------------------

double HoeffdingTreeRegressor::predict_one(
    const std::unordered_map<std::string, double>& x) const
{
    if (!root) return 0.0;
    HTRegNode* node = traverse(x);

    // Use linear model once we have enough data (at least grace_period samples)
    if (node->n >= grace_period) {
        double y_hat = node->bias;
        for (auto& [feat, val] : x) {
            auto it = node->weights.find(feat);
            if (it != node->weights.end())
                y_hat += it->second * val;
        }
        return y_hat;
    }

    // Fall back to target mean
    return node->target_stats.mean;
}
