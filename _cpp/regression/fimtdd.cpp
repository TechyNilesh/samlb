#include "fimtdd.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr double kMinBranchWeight = 5.0;   // MOA's VarianceReductionSplitCriterion
}

double FIMTDDRegressor::Stats::sd() const {
    if (n > 1.0) {
        const double v = (sq - (sum * sum) / n) / n;
        return v > 0.0 ? std::sqrt(v) : 0.0;
    }
    return 0.0;
}

FIMTDDRegressor::FIMTDDRegressor(int grace_period, double split_confidence,
                                 double tie_threshold, int max_depth,
                                 std::string leaf_prediction, double learning_ratio,
                                 double learning_rate_decay, bool learning_ratio_const,
                                 double page_hinckley_alpha,
                                 double page_hinckley_threshold,
                                 double alternate_tree_fading_factor,
                                 int alternate_tree_t_min, int alternate_tree_time,
                                 bool drift_detection, int seed)
    : grace_period(grace_period),
      split_confidence(split_confidence),
      tie_threshold(tie_threshold),
      max_depth(max_depth),
      leaf_prediction(std::move(leaf_prediction)),
      learning_ratio(learning_ratio),
      learning_rate_decay(learning_rate_decay),
      learning_ratio_const(learning_ratio_const),
      page_hinckley_alpha(page_hinckley_alpha),
      page_hinckley_threshold(page_hinckley_threshold),
      alternate_tree_fading_factor(alternate_tree_fading_factor),
      alternate_tree_t_min(alternate_tree_t_min),
      alternate_tree_time(alternate_tree_time),
      drift_detection(drift_detection),
      seed(seed),
      rng_(static_cast<unsigned>(seed)) {}

// ── normalisation, shared by the perceptron and the drift test ───────────────

double FIMTDDRegressor::normalise_target(double value) const {
    if (target_.n > 1.0) {
        const double sd = target_.sd();
        if (sd > 0.0) return (value - target_.sum / target_.n) / (3.0 * sd);
    }
    return 0.0;
}

double FIMTDDRegressor::normalised_error(double y, double prediction) const {
    return std::fabs(normalise_target(y) - normalise_target(prediction));
}

// ── E-BST ───────────────────────────────────────────────────────────────────

void FIMTDDRegressor::ebst_insert(std::unique_ptr<EBSTNode>& node, double value,
                                  double y, double w) {
    if (!node) {
        node = std::make_unique<EBSTNode>();
        node->value = value;
        node->left_n = w;
        node->left_sum = w * y;
        node->left_sq = w * y * y;
        return;
    }
    EBSTNode* cur = node.get();
    while (true) {
        if (value <= cur->value) {
            cur->left_n += w;
            cur->left_sum += w * y;
            cur->left_sq += w * y * y;
            if (value == cur->value) return;
            if (!cur->left) {
                cur->left = std::make_unique<EBSTNode>();
                cur->left->value = value;
                cur->left->left_n = w;
                cur->left->left_sum = w * y;
                cur->left->left_sq = w * y * y;
                return;
            }
            cur = cur->left.get();
        } else {
            cur->right_n += w;
            cur->right_sum += w * y;
            cur->right_sq += w * y * y;
            if (!cur->right) {
                cur->right = std::make_unique<EBSTNode>();
                cur->right->value = value;
                cur->right->left_n = w;
                cur->right->left_sum = w * y;
                cur->right->left_sq = w * y * y;
                return;
            }
            cur = cur->right.get();
        }
    }
}

// In-order walk accumulating everything to the left of the current candidate.
// Standard deviation reduction:  SDR = sd(all) - nL/n sd(L) - nR/n sd(R),
// with both branches required to hold at least five instances.
//
// `running` is the distribution strictly left of this subtree. A node's own
// left_* already covers every instance in its left subtree, so the left
// distribution at a candidate is running + node->left_* — adding the subtree
// totals again would double-count them.
void FIMTDDRegressor::ebst_scan(const EBSTNode* node, const Stats& total,
                                const std::string& feature, Stats running,
                                SplitSuggestion& best,
                                SplitSuggestion& second) const {
    if (!node) return;

    if (node->left) ebst_scan(node->left.get(), total, feature, running, best, second);

    Stats left = running;
    left.n += node->left_n;
    left.sum += node->left_sum;
    left.sq += node->left_sq;

    Stats right;
    right.n = total.n - left.n;
    right.sum = total.sum - left.sum;
    right.sq = total.sq - left.sq;

    if (left.n >= kMinBranchWeight && right.n >= kMinBranchWeight) {
        const double merit = total.sd()
                           - (left.n / total.n) * left.sd()
                           - (right.n / total.n) * right.sd();
        if (merit > best.merit) best = {feature, node->value, merit};
    }

    if (node->right) ebst_scan(node->right.get(), total, feature, left, best, second);
}

FIMTDDRegressor::SplitSuggestion FIMTDDRegressor::best_split(const Node& leaf) const {
    // One candidate per *attribute*, then rank the attributes — as the
    // reference does. Ranking raw candidate values instead would make the top
    // two almost always adjacent cut points on the same feature, with a merit
    // ratio near 1, and the tree would never split.
    SplitSuggestion best, second;
    for (const auto& kv : leaf.observers) {
        SplitSuggestion per_feature, ignored;
        ebst_scan(kv.second.get(), leaf.stats, kv.first, Stats{}, per_feature, ignored);
        if (per_feature.merit <= 0.0) continue;
        if (per_feature.merit > best.merit) {
            second = best;
            best = per_feature;
        } else if (per_feature.merit > second.merit) {
            second = per_feature;
        }
    }
    if (best.merit <= 0.0) return SplitSuggestion{};
    if (second.merit <= 0.0) return best;              // only one usable feature

    // MOA's ratio form of the Hoeffding test: on variance reduction the merits
    // are unbounded and scale with the target, so best-minus-second is not a
    // scale-free comparison but second-over-best is.
    const double bound = std::sqrt(std::log(1.0 / split_confidence)
                                   / (2.0 * leaf.stats.n));
    if (second.merit / best.merit < 1.0 - bound || bound < tie_threshold) {
        return best;
    }
    return SplitSuggestion{};
}

// ── leaf perceptron, fitted in globally normalised space ────────────────────

double FIMTDDRegressor::perceptron_predict(const Perceptron& model,
                                           const Features& x) const {
    if (!model.initialised) return 0.0;
    double z = model.bias;
    for (const auto& kv : x) {
        auto stat = attr_.find(kv.first);
        if (stat == attr_.end() || stat->second.n <= 1.0) continue;
        const double sd = stat->second.sd();
        if (sd <= 0.0) continue;
        const double normalised = (kv.second - stat->second.sum / stat->second.n)
                                / (3.0 * sd);
        auto w = model.weights.find(kv.first);
        if (w != model.weights.end()) z += w->second * normalised;
    }
    // back to the target's scale
    if (target_.n > 1.0) {
        const double sd = target_.sd();
        if (sd > 0.0) return z * 3.0 * sd + target_.sum / target_.n;
    }
    return target_.n > 0.0 ? target_.sum / target_.n : 0.0;
}

void FIMTDDRegressor::perceptron_update(Perceptron& model, const Features& x,
                                        double y) {
    if (!model.initialised) {
        std::uniform_real_distribution<double> init(-1.0, 1.0);
        for (const auto& kv : x) model.weights[kv.first] = init(rng_);
        model.bias = init(rng_);
        model.initialised = true;
    }
    model.instances += 1.0;

    const double ratio = learning_ratio_const
        ? learning_ratio
        : learning_ratio / (1.0 + model.instances * learning_rate_decay);

    // delta is taken in normalised space, as in the reference
    double z = model.bias;
    std::vector<std::pair<const std::string*, double>> normalised;
    normalised.reserve(x.size());
    for (const auto& kv : x) {
        auto stat = attr_.find(kv.first);
        double v = 0.0;
        if (stat != attr_.end() && stat->second.n > 1.0) {
            const double sd = stat->second.sd();
            if (sd > 0.0) v = (kv.second - stat->second.sum / stat->second.n)
                             / (3.0 * sd);
        }
        normalised.emplace_back(&kv.first, v);
        z += model.weights[kv.first] * v;
    }
    const double delta = normalise_target(y) - z;
    for (const auto& item : normalised) {
        model.weights[*item.first] += delta * ratio * item.second;
    }
    model.bias += delta * ratio;
}

// ── prediction ──────────────────────────────────────────────────────────────

double FIMTDDRegressor::leaf_predict(const Node* leaf, const Features& x) const {
    const double mean = leaf->stats.n > 0.0 ? leaf->stats.sum / leaf->stats.n
                      : (target_.n > 0.0 ? target_.sum / target_.n : 0.0);
    if (leaf_prediction == "mean") return mean;

    double model = perceptron_predict(leaf->model, x);
    if (leaf_prediction == "perceptron") return model;   // reference, unbounded

    // "adaptive": bound the perceptron to what this leaf has actually seen, so
    // a diverged weight vector cannot leave the target's range, then use it
    // only while it is beating the leaf mean.
    if (leaf->y_seen) {
        model = std::max(leaf->y_min, std::min(leaf->y_max, model));
    }
    return leaf->perceptron_sse < leaf->mean_sse ? model : mean;
}

double FIMTDDRegressor::node_predict(const Node* node, const Features& x) const {
    while (node && !node->is_leaf) {
        auto it = x.find(node->split_feature);
        const double v = it == x.end() ? 0.0 : it->second;
        node = (v <= node->split_value ? node->left : node->right).get();
    }
    if (!node) return 0.0;
    return leaf_predict(node, x);
}

double FIMTDDRegressor::predict_one(const Features& x) const {
    if (!root_) return 0.0;
    return node_predict(root_.get(), x);
}

// ── training ────────────────────────────────────────────────────────────────

std::unique_ptr<FIMTDDRegressor::Node> FIMTDDRegressor::new_leaf(
    const Perceptron* inherit) const {
    auto leaf = std::make_unique<Node>();
    // Children inherit the splitting leaf's perceptron, as in the reference —
    // a fresh random one would throw away everything the parent had learnt.
    if (inherit) leaf->model = *inherit;
    return leaf;
}

void FIMTDDRegressor::learn_one(const Features& x, double y) {
    if (!root_) root_ = new_leaf(nullptr);

    target_.add(y, 1.0);
    for (const auto& kv : x) attr_[kv.first].add(kv.second, 1.0);

    const double prediction = node_predict(root_.get(), x);
    learn_at(root_.get(), x, y, prediction, normalised_error(y, prediction),
             true, false, 0);
}

void FIMTDDRegressor::learn_leaf(Node* node, const Features& x, double y,
                                 bool growth_allowed, int depth) {
    // Score both leaf predictors on this instance before learning from it, so
    // the choice between them is made out-of-sample.
    if (building_model_tree()) {
        const double mean = node->stats.n > 0.0 ? node->stats.sum / node->stats.n
                          : (target_.n > 0.0 ? target_.sum / target_.n : 0.0);
        double model = perceptron_predict(node->model, x);
        if (node->y_seen) model = std::max(node->y_min, std::min(node->y_max, model));
        node->mean_sse += (mean - y) * (mean - y);
        node->perceptron_sse += (model - y) * (model - y);
    }
    if (!node->y_seen) {
        node->y_min = node->y_max = y;
        node->y_seen = true;
    } else {
        node->y_min = std::min(node->y_min, y);
        node->y_max = std::max(node->y_max, y);
    }

    node->stats.add(y, 1.0);
    node->examples += 1.0;
    if (building_model_tree()) perceptron_update(node->model, x, y);

    for (const auto& kv : x) {
        ebst_insert(node->observers[kv.first], kv.second, y, 1.0);
    }

    if (growth_allowed && depth < max_depth &&
        node->stats.n - node->examples_at_last_split >= grace_period) {
        attempt_split(node, depth);
        node->examples_at_last_split = node->stats.n;
    }
}

void FIMTDDRegressor::attempt_split(Node* leaf, int depth) {
    (void)depth;
    const SplitSuggestion split = best_split(*leaf);
    if (split.merit <= 0.0 || split.feature.empty()) return;

    leaf->is_leaf = false;
    leaf->split_feature = split.feature;
    leaf->split_value = split.value;
    const Perceptron* inherit = building_model_tree() ? &leaf->model : nullptr;
    leaf->left = new_leaf(inherit);
    leaf->right = new_leaf(inherit);
    leaf->observers.clear();
    ++split_count_;
}

void FIMTDDRegressor::learn_at(Node* node, const Features& x, double y,
                               double prediction, double normal_error,
                               bool growth_allowed, bool in_alternate, int depth) {
    while (true) {
        if (node->is_leaf) {
            learn_leaf(node, x, y, growth_allowed, depth);
            return;
        }

        node->examples += 1.0;
        node->abs_errors += normal_error;

        if (!in_alternate && node->alternate) {
            bool keep_alternate = true;
            const double loss_o = std::pow(y - prediction, 2);
            const double loss_a = std::pow(y - node_predict(node->alternate.get(), x), 2);
            node->loss_faded_original =
                loss_o + alternate_tree_fading_factor * node->loss_faded_original;
            node->loss_faded_alternate =
                loss_a + alternate_tree_fading_factor * node->loss_faded_alternate;
            node->loss_examples += 1.0;

            if (node->loss_examples - node->previous_weight >= alternate_tree_t_min) {
                node->previous_weight = node->loss_examples;
                const double qi = std::log(
                    (node->loss_faded_original + 1e-12) /
                    (node->loss_faded_alternate + 1e-12));
                if (qi > 0.0) {
                    // The background subtree is doing better: promote it.
                    auto promoted = std::move(node->alternate);
                    *node = std::move(*promoted);
                    node->change_detection = true;
                    node->ph_sum = 0.0;
                    node->ph_started = false;
                    keep_alternate = false;
                    continue;                      // carry on inside the promotion
                }
                if (node->loss_examples >= alternate_tree_time) {
                    node->alternate.reset();
                    node->change_detection = true;
                    node->ph_sum = 0.0;
                    node->ph_started = false;
                    keep_alternate = false;
                }
            }
            if (keep_alternate) {
                growth_allowed = false;
                learn_at(node->alternate.get(), x, y, prediction, normal_error,
                         true, true, depth);
            }
        }

        if (drift_detection && node->change_detection && !in_alternate) {
            // Page-Hinckley on the node's own normalised error, against its
            // running mean, with a tolerance of alpha.
            const double signal =
                normal_error - node->abs_errors / node->examples - page_hinckley_alpha;
            node->ph_sum += signal;
            if (!node->ph_started) {
                node->ph_min = node->ph_sum;
                node->ph_started = true;
            }
            node->ph_min = std::min(node->ph_min, node->ph_sum);
            if (node->ph_sum - node->ph_min > page_hinckley_threshold) {
                node->alternate = new_leaf(nullptr);
                node->loss_examples = 0.0;
                node->loss_faded_original = 0.0;
                node->loss_faded_alternate = 0.0;
                node->previous_weight = 0.0;
                node->change_detection = false;   // suspended below this node
            }
        }

        auto it = x.find(node->split_feature);
        const double v = it == x.end() ? 0.0 : it->second;
        node = (v <= node->split_value ? node->left : node->right).get();
        ++depth;
        if (!node) return;
    }
}

void FIMTDDRegressor::reset() {
    root_.reset();
    target_ = Stats{};
    attr_.clear();
    split_count_ = 0;
    rng_.seed(static_cast<unsigned>(seed));
}

int FIMTDDRegressor::n_leaves() const { return split_count_ + 1; }
