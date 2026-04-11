#include "sgt.h"
#include <cmath>
#include <limits>
#include <algorithm>

// ---------------------------------------------------------------------------
// Utility: sigmoid
// ---------------------------------------------------------------------------

static inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// ---------------------------------------------------------------------------
// Constructor / reset
// ---------------------------------------------------------------------------

SGTClassifier::SGTClassifier(double lr, double lam, int gp, int md)
    : learning_rate(lr), lambda(lam), grace_period(gp), max_depth(md)
{
    root = std::make_unique<SGTNode>();
}

void SGTClassifier::reset() {
    root = std::make_unique<SGTNode>();
    seen_classes.clear();
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

SGTNode* SGTClassifier::traverse(
    const std::unordered_map<std::string, double>& x) const
{
    SGTNode* node = root.get();
    while (!node->is_leaf) {
        auto it   = x.find(node->split_feature);
        double v  = (it != x.end()) ? it->second : 0.0;
        node = (v <= node->split_value) ? node->left.get()
                                        : node->right.get();
    }
    return node;
}

// ---------------------------------------------------------------------------
// update_leaf: accumulate gradient / hessian statistics
// ---------------------------------------------------------------------------

void SGTClassifier::update_leaf(
    SGTNode* node,
    const std::unordered_map<std::string, double>& x,
    double grad, double hess)
{
    node->grad_sum += grad;
    node->hess_sum += hess;
    node->n        += 1;

    // Update per-feature gradient and hessian accumulators using the
    // feature value as the "observation" and grad/hess as weights.
    // We record whether each observation went left (<= mean) or right
    // by accumulating into two separate GaussianEstimators per feature.
    // Here we store the raw feature values weighted by |grad| to identify
    // good split thresholds.
    for (auto& [feat, val] : x) {
        node->grad_stats[feat].update(val, std::abs(grad) + 1e-10);
        node->hess_stats[feat].update(val, hess + 1e-10);
    }

    // Newton step: update leaf weight
    // w  = -G / (H + lambda)
    if (node->hess_sum + lambda > 0.0)
        node->weight -= learning_rate * node->grad_sum /
                        (node->hess_sum + lambda);
}

// ---------------------------------------------------------------------------
// gain: XGBoost-style leaf gain
//   gain = G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)
// ---------------------------------------------------------------------------

double SGTClassifier::gain(double G_L, double H_L,
                            double G_R, double H_R) const
{
    double g = G_L * G_L / (H_L + lambda)
             + G_R * G_R / (H_R + lambda)
             - (G_L + G_R) * (G_L + G_R) / (H_L + H_R + lambda);
    return 0.5 * g;
}

// ---------------------------------------------------------------------------
// try_split: gradient-based split finding
// ---------------------------------------------------------------------------

void SGTClassifier::try_split(SGTNode* node, int depth) {
    if (depth >= max_depth) return;
    if (node->n < grace_period) return;

    // The split threshold for each feature is its weighted-mean value
    // (using the gradient accumulator mean as a proxy for the data centroid).

    double G_total = node->grad_sum;
    double H_total = node->hess_sum;

    double best_gain = 0.0;     // only split if gain > 0
    std::string best_feat;
    double       best_threshold = 0.0;
    double       best_G_L = 0.0, best_H_L = 0.0;
    double       best_G_R = 0.0, best_H_R = 0.0;

    for (auto& [feat, g_est] : node->grad_stats) {
        if (g_est.n <= 0.0) continue;

        // Use the feature mean as split threshold
        double threshold = g_est.mean;

        // Approximate fraction of data going left vs right using the
        // hessian estimator's Gaussian CDF (hess_sum ≈ n for log-loss).
        auto h_it = node->hess_stats.find(feat);
        double frac_left = 0.5;
        if (h_it != node->hess_stats.end() && h_it->second.n > 0.0) {
            double sd = h_it->second.std_dev();
            if (sd > 0.0) {
                frac_left = 0.5 * (1.0 + std::erf(
                    (threshold - h_it->second.mean) / (sd * std::sqrt(2.0))));
            }
        }
        frac_left = std::max(0.01, std::min(0.99, frac_left));

        // Approximate G and H for left and right children
        double G_L = G_total * frac_left;
        double G_R = G_total * (1.0 - frac_left);
        double H_L = H_total * frac_left;
        double H_R = H_total * (1.0 - frac_left);

        double g = gain(G_L, H_L, G_R, H_R);
        if (g > best_gain) {
            best_gain      = g;
            best_feat      = feat;
            best_threshold = threshold;
            best_G_L = G_L; best_H_L = H_L;
            best_G_R = G_R; best_H_R = H_R;
        }
    }

    if (best_feat.empty() || best_gain <= 0.0) return;

    // Perform split
    node->is_leaf       = false;
    node->split_feature = best_feat;
    node->split_value   = best_threshold;

    node->left  = std::make_unique<SGTNode>();
    node->right = std::make_unique<SGTNode>();

    // Initialise child leaf weights using Newton step on the split gradient sums
    if (best_H_L + lambda > 0.0)
        node->left->weight  = -best_G_L / (best_H_L + lambda);
    if (best_H_R + lambda > 0.0)
        node->right->weight = -best_G_R / (best_H_R + lambda);

    // Carry over proportional gradient / hessian sums to children
    double frac_left  = (H_total > 0.0) ? best_H_L / H_total : 0.5;
    double frac_right = 1.0 - frac_left;

    node->left->grad_sum  = best_G_L;
    node->left->hess_sum  = best_H_L;
    node->left->n         = static_cast<long long>(node->n * frac_left);

    node->right->grad_sum = best_G_R;
    node->right->hess_sum = best_H_R;
    node->right->n        = static_cast<long long>(node->n * frac_right);

    // Distribute feature stats to children proportionally
    for (auto& [feat, g_est] : node->grad_stats) {
        GaussianEstimator gl, gr;
        gl.n = g_est.n * frac_left;  gl.mean = g_est.mean; gl.M2 = g_est.M2 * frac_left;
        gr.n = g_est.n * frac_right; gr.mean = g_est.mean; gr.M2 = g_est.M2 * frac_right;
        node->left->grad_stats[feat]  = gl;
        node->right->grad_stats[feat] = gr;
    }
    for (auto& [feat, h_est] : node->hess_stats) {
        GaussianEstimator hl, hr;
        hl.n = h_est.n * frac_left;  hl.mean = h_est.mean; hl.M2 = h_est.M2 * frac_left;
        hr.n = h_est.n * frac_right; hr.mean = h_est.mean; hr.M2 = h_est.M2 * frac_right;
        node->left->hess_stats[feat]  = hl;
        node->right->hess_stats[feat] = hr;
    }

    // Clear parent leaf data
    node->grad_stats.clear();
    node->hess_stats.clear();
    node->grad_sum = 0.0;
    node->hess_sum = 0.0;
    node->n        = 0;
}

// ---------------------------------------------------------------------------
// learn_one
// ---------------------------------------------------------------------------

void SGTClassifier::learn_one(
    const std::unordered_map<std::string, double>& x, int y)
{
    seen_classes.insert(y);
    if (!root) root = std::make_unique<SGTNode>();

    // Traverse and track depth
    SGTNode* node  = root.get();
    int      depth = 0;
    while (!node->is_leaf) {
        auto it  = x.find(node->split_feature);
        double v = (it != x.end()) ? it->second : 0.0;
        node = (v <= node->split_value) ? node->left.get()
                                        : node->right.get();
        ++depth;
    }

    // Compute log-loss gradient and hessian for binary classification.
    // For multiclass we use a one-vs-all heuristic: treat the current leaf
    // weight as the raw score for the positive class (y == 1) in each binary
    // problem.  For simplicity we map all classes to {0, 1} using the label
    // directly (works for binary naturally; for multiclass labels > 1 the
    // gradient still pushes the weight in the right direction).
    double pred  = sigmoid(node->weight);
    double y_bin = (y > 0) ? 1.0 : 0.0;  // treat label 0 as negative class
    double grad  = pred - y_bin;           // dL/dw  (log-loss)
    double hess  = std::max(pred * (1.0 - pred), 1e-6);  // d²L/dw²

    update_leaf(node, x, grad, hess);

    // Try split every grace_period instances
    if (node->n % grace_period == 0) {
        try_split(node, depth);
    }
}

// ---------------------------------------------------------------------------
// predict_one
// ---------------------------------------------------------------------------

int SGTClassifier::predict_one(
    const std::unordered_map<std::string, double>& x) const
{
    if (!root) return 0;
    SGTNode* node = traverse(x);
    // Leaf weight is the raw logit; sigmoid > 0.5 ⟺ weight > 0
    return (node->weight >= 0.0) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// predict_proba_one
// ---------------------------------------------------------------------------

std::unordered_map<int, double> SGTClassifier::predict_proba_one(
    const std::unordered_map<std::string, double>& x) const
{
    std::unordered_map<int, double> proba;
    if (!root) {
        proba[0] = 0.5; proba[1] = 0.5;
        return proba;
    }
    SGTNode* node = traverse(x);
    double p1 = sigmoid(node->weight);
    proba[1] = p1;
    proba[0] = 1.0 - p1;
    return proba;
}
