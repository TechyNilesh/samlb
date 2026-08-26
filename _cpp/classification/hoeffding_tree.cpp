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

std::unordered_map<int, double> HoeffdingTreeClassifier::nb_log_proba(
    const HTNode* node, const std::unordered_map<std::string, double>& x) const
{
    std::unordered_map<int, double> log_p;
    const double n_total = node->total_weight;
    if (n_total <= 0.0) return log_p;

    for (const auto& [cls, cnt] : node->class_counts) {
        double lp = std::log(cnt / n_total);
        const auto it = node->stats.find(cls);
        if (it != node->stats.end()) {
            for (const auto& [feat, val] : x) {
                const auto fit = it->second.find(feat);
                if (fit != it->second.end()) {
                    const double p = fit->second.est.probability_density(val);
                    lp += std::log(p > 1e-300 ? p : 1e-300);
                }
            }
        }
        log_p[cls] = lp;
    }
    return log_p;
}

bool HoeffdingTreeClassifier::use_nb(const HTNode* node) const {
    if (leaf_prediction == "mc") return false;
    if (node->total_weight < static_cast<double>(nb_threshold)) return false;
    if (leaf_prediction == "nb") return true;
    return node->nb_correct > node->mc_correct;      // "nba"
}

void HoeffdingTreeClassifier::update_leaf(
    HTNode* node,
    const std::unordered_map<std::string, double>& x,
    int y)
{
    // Score both leaf predictors on this instance *before* learning from it,
    // so the adaptive choice is based on out-of-sample performance.
    if (leaf_prediction != "mc" && node->total_weight > 0.0) {
        if (majority_class(node) == y) node->mc_correct += 1.0;
        const auto log_p = nb_log_proba(node, x);
        if (!log_p.empty()) {
            int best = 0;
            double best_lp = -std::numeric_limits<double>::infinity();
            for (const auto& [cls, lp] : log_p)
                if (lp > best_lp) { best_lp = lp; best = cls; }
            if (best == y) node->nb_correct += 1.0;
        }
    }

    node->class_counts[y] += 1.0;
    node->total_weight     += 1.0;
    for (auto& [feat, val] : x)
        node->stats[y][feat].update(val);
}

// ---------------------------------------------------------------------------
// Split-merit machinery — mirrors MOA's InfoGainSplitCriterion /
// GiniSplitCriterion operating on exact resulting class distributions, and
// GaussianNumericAttributeClassObserver for how those distributions are
// estimated for a numeric split candidate.
// ---------------------------------------------------------------------------

namespace {

using ClassDist = std::unordered_map<int, double>;

double sum_of(const ClassDist& d) {
    double s = 0.0;
    for (auto& [cls, w] : d) s += w;
    return s;
}

double gaussian_cdf(double x, double mean, double std_dev) {
    if (std_dev <= 0.0) return (x >= mean) ? 1.0 : 0.0;
    return 0.5 * (1.0 + std::erf((x - mean) / (std_dev * std::sqrt(2.0))));
}

// MOA's GaussianEstimator.estimatedWeight_LessThan_EqualTo_GreaterThan_Value:
// treats the density at the split point as a point mass ("equal"), then
// splits the remaining weight by the normal CDF.
void estimated_weights(const AttrObserver& obs, double split_value,
                        double& less, double& equal, double& greater)
{
    equal = obs.est.probability_density(split_value) * obs.est.n;
    const double sd = obs.est.std_dev();
    if (sd > 0.0) {
        less = gaussian_cdf(split_value, obs.est.mean, sd) * obs.est.n - equal;
    } else {
        less = (split_value < obs.est.mean) ? obs.est.n - equal : 0.0;
    }
    double g = obs.est.n - equal - less;
    if (g < 0.0) g = 0.0;
    greater = g;
}

// MOA's GaussianNumericAttributeClassObserver.getClassDistsResultingFromBinarySplit:
// values <= split_value go left. Per-class min/max shortcut the estimate
// exactly whenever split_value falls outside a class's observed range.
void class_dists_from_binary_split(const HTNode* node, const std::string& feat,
                                    double split_value,
                                    ClassDist& left, ClassDist& right)
{
    for (auto& [cls, feat_map] : node->stats) {
        auto fit = feat_map.find(feat);
        if (fit == feat_map.end() || fit->second.est.n <= 0.0) continue;
        const AttrObserver& obs = fit->second;
        if (split_value < obs.min_val) {
            right[cls] += obs.est.n;
        } else if (split_value >= obs.max_val) {
            left[cls] += obs.est.n;
        } else {
            double less, equal, greater;
            estimated_weights(obs, split_value, less, equal, greater);
            left[cls]  += less + equal;
            right[cls] += greater;
        }
    }
}

// MOA's GaussianNumericAttributeClassObserver.getSplitPointSuggestions:
// `num_bins` points evenly spaced strictly between the global observed
// min/max for this feature at this leaf (default 10 bins, like MOA).
std::vector<double> split_point_suggestions(const HTNode* node,
                                             const std::string& feat,
                                             int num_bins)
{
    double min_v = std::numeric_limits<double>::infinity();
    double max_v = -std::numeric_limits<double>::infinity();
    for (auto& [cls, feat_map] : node->stats) {
        auto fit = feat_map.find(feat);
        if (fit == feat_map.end() || fit->second.est.n <= 0.0) continue;
        min_v = std::min(min_v, fit->second.min_val);
        max_v = std::max(max_v, fit->second.max_val);
    }

    std::vector<double> out;
    if (!(min_v < max_v)) return out;  // no spread, or nothing observed

    std::set<double> suggestions;
    const double range = max_v - min_v;
    for (int i = 0; i < num_bins; ++i) {
        double split_value = range / (num_bins + 1.0) * (i + 1) + min_v;
        if (split_value > min_v && split_value < max_v)
            suggestions.insert(split_value);
    }
    out.assign(suggestions.begin(), suggestions.end());
    return out;
}

double compute_entropy(const ClassDist& dist, double total) {
    if (total <= 0.0) return 0.0;
    double h = 0.0;
    for (auto& [cls, w] : dist) {
        if (w > 0.0) {
            double p = w / total;
            h -= p * std::log2(p);
        }
    }
    return h;
}

// MOA's InfoGainSplitCriterion.numSubsetsGreaterThanFrac (minBranchFrac=0.01
// default): a split only counts if at least two branches carry a
// non-trivial share of the weight.
constexpr double MIN_BRANCH_FRAC = 0.01;

using Branch = std::pair<const ClassDist*, double>;

int num_subsets_greater_than_frac(const std::vector<Branch>& branches, double min_frac) {
    double total = 0.0;
    for (auto& [d, w] : branches) total += w;
    if (total <= 0.0) return 0;
    int count = 0;
    for (auto& [d, w] : branches)
        if (w / total > min_frac) ++count;
    return count;
}

double merit_info_gain(const ClassDist& pre, double pre_total,
                        const std::vector<Branch>& branches)
{
    if (num_subsets_greater_than_frac(branches, MIN_BRANCH_FRAC) < 2)
        return -std::numeric_limits<double>::infinity();
    double total = 0.0;
    for (auto& [d, w] : branches) total += w;
    if (total <= 0.0) return 0.0;
    double post_h = 0.0;
    for (auto& [d, w] : branches) post_h += w * compute_entropy(*d, w);
    post_h /= total;
    return compute_entropy(pre, pre_total) - post_h;
}

double gini_impurity(const ClassDist& dist, double total) {
    if (total <= 0.0) return 0.0;
    double g = 1.0;
    for (auto& [cls, w] : dist) {
        double p = w / total;
        g -= p * p;
    }
    return g;
}

double merit_gini(const std::vector<Branch>& branches) {
    double total = 0.0;
    for (auto& [d, w] : branches) total += w;
    if (total <= 0.0) return 0.0;
    double g = 0.0;
    for (auto& [d, w] : branches) g += (w / total) * gini_impurity(*d, w);
    return 1.0 - g;
}

double merit_of_split(const std::string& criterion,
                       const ClassDist& pre, double pre_total,
                       const std::vector<Branch>& branches)
{
    return (criterion == "gini") ? merit_gini(branches)
                                  : merit_info_gain(pre, pre_total, branches);
}

} // namespace

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
// try_split: attempt to split a leaf using the Hoeffding bound, mirroring
// MOA's HoeffdingTree.attemptToSplit — one best candidate per attribute (via
// class_dists_from_binary_split / split_point_suggestions above), a null
// "don't split" candidate for pre-pruning, and a split iff the best merit
// clears the second-best by more than the bound (or the bound itself is
// already tight, MOA's tie-break).
// ---------------------------------------------------------------------------

void HoeffdingTreeClassifier::set_subspace(int size, unsigned int seed) {
    subspace_size_ = size;
    split_rng_.seed(seed);
}

void HoeffdingTreeClassifier::try_split(HTNode* node, int depth) {
    if (depth >= max_depth) return;
    if (node->total_weight < 2.0) return;

    int n_classes = static_cast<int>(seen_classes.size());
    if (n_classes < 2) return;

    std::set<std::string> features;
    for (auto& [cls, feat_map] : node->stats)
        for (auto& [feat, obs] : feat_map)
            features.insert(feat);
    if (features.empty()) return;

    // Hoeffding bound range depends on criterion:
    // info_gain -> log2(n_classes) (max possible entropy); gini -> 1.0
    double range = (split_criterion == "gini")
                 ? 1.0
                 : std::log2(static_cast<double>(std::max(n_classes, 2)));
    double epsilon = hoeffding_bound(range, split_confidence, node->total_weight);

    struct Candidate {
        bool        is_null   = false;
        std::string feature;
        double      threshold = 0.0;
        double      merit     = -std::numeric_limits<double>::infinity();
        ClassDist   left, right;
    };
    std::vector<Candidate> candidates;

    // Pre-pruning "null" option: the merit of not splitting at all.
    if (!no_pre_prune) {
        Candidate c;
        c.is_null = true;
        c.merit = merit_of_split(split_criterion, node->class_counts, node->total_weight,
                                  {{&node->class_counts, node->total_weight}});
        candidates.push_back(std::move(c));
    }

    // Random subspace, resampled at every split attempt (ARF).
    std::vector<std::string> cand_feats(features.begin(), features.end());
    if (subspace_size_ > 0 && static_cast<int>(cand_feats.size()) > subspace_size_) {
        std::shuffle(cand_feats.begin(), cand_feats.end(), split_rng_);
        cand_feats.resize(static_cast<size_t>(subspace_size_));
    }

    for (const auto& feat : cand_feats) {
        std::vector<double> thresholds = split_point_suggestions(node, feat, n_split_points);
        if (thresholds.empty()) continue;

        Candidate best_for_feat;
        best_for_feat.feature = feat;
        for (double threshold : thresholds) {
            ClassDist left, right;
            class_dists_from_binary_split(node, feat, threshold, left, right);
            double left_total  = sum_of(left);
            double right_total = sum_of(right);
            double merit = merit_of_split(split_criterion, node->class_counts, node->total_weight,
                                           {{&left, left_total}, {&right, right_total}});
            if (merit > best_for_feat.merit) {
                best_for_feat.merit     = merit;
                best_for_feat.threshold = threshold;
                best_for_feat.left      = std::move(left);
                best_for_feat.right     = std::move(right);
            }
        }
        candidates.push_back(std::move(best_for_feat));
    }

    if (candidates.empty()) return;

    size_t best_idx = 0;
    for (size_t i = 1; i < candidates.size(); ++i)
        if (candidates[i].merit > candidates[best_idx].merit) best_idx = i;

    double second_best_merit = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i == best_idx) continue;
        if (candidates[i].merit > second_best_merit) second_best_merit = candidates[i].merit;
    }

    const Candidate& best = candidates[best_idx];
    if (best.merit == -std::numeric_limits<double>::infinity()) return;

    bool should_split;
    if (candidates.size() < 2) {
        should_split = true;   // only one suggestion at all -> take it (matches MOA)
    } else {
        double delta = best.merit - second_best_merit;
        should_split = (delta > epsilon) || (epsilon < tie_threshold);
    }
    if (!should_split) return;
    if (best.is_null) return;  // pre-prune: "don't split" won

    // Perform the split: convert leaf -> split node, create two child leaves
    node->is_leaf       = false;
    node->split_feature  = best.feature;
    node->split_value    = best.threshold;

    node->left  = std::make_unique<HTNode>();
    node->right = std::make_unique<HTNode>();

    double left_total = 0.0, right_total = 0.0;
    for (auto& [cls, w] : best.left)
        if (w > 0.0) { node->left->class_counts[cls] = w; left_total += w; }
    for (auto& [cls, w] : best.right)
        if (w > 0.0) { node->right->class_counts[cls] = w; right_total += w; }
    node->left->total_weight  = left_total;
    node->right->total_weight = right_total;

    // Children start with empty attribute observers and rebuild them from
    // scratch as new instances arrive — matching MOA, which does not carry a
    // parent's per-attribute statistics into its new leaves.
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

    if (use_nb(node)) {
        const auto log_p = nb_log_proba(node, x);
        int    best     = majority_class(node);
        double best_lp  = -std::numeric_limits<double>::infinity();
        for (const auto& [cls, lp] : log_p)
            if (lp > best_lp) { best_lp = lp; best = cls; }
        return best;
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

    if (use_nb(node)) {
        // Naive Bayes probabilities
        double total_prob = 0.0;
        for (auto& [cls, cnt] : node->class_counts) {
            double p = cnt / n_total;
            auto it = node->stats.find(cls);
            if (it != node->stats.end()) {
                for (auto& [feat, val] : x) {
                    auto fit = it->second.find(feat);
                    if (fit != it->second.end()) {
                        double pd = fit->second.est.probability_density(val);
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
