#include "sgbt.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

// Below this, p*(1-p) is numerically zero and the pseudo-label g/h blows up.
// MOA leaves the division unguarded (clipPredictions is passed false), which
// only survives because a saturated score is rare; here it is floored.
constexpr double kMinHessian = 1e-12;

inline double sigmoid(double z) {
    // softmax over (raw, 0) at index 0 — the reference's SoftmaxCrossEntropy
    // transfer, written for the one-output committee it always uses.
    if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
    const double e = std::exp(z);
    return e / (1.0 + e);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
//  SGBTBooster
// ─────────────────────────────────────────────────────────────────────────────

SGBTBooster::SGBTBooster(int n_iterations, double learning_rate,
                         int percentage_of_features, int multiply_hessian_by,
                         int skip_training, bool squared_loss, int bag_size,
                         int grace_period, double split_confidence, int max_depth,
                         std::string leaf_prediction, unsigned seed)
    : n_iterations_(std::max(1, n_iterations)),
      percentage_of_features_(percentage_of_features),
      multiply_hessian_by_(multiply_hessian_by),
      skip_training_(skip_training),
      bag_size_(std::max(1, bag_size)),
      learning_rate_(learning_rate),
      squared_loss_(squared_loss),
      grace_period_(grace_period),
      max_depth_(max_depth),
      split_confidence_(split_confidence),
      leaf_prediction_(std::move(leaf_prediction)),
      seed_(seed),
      rng_(seed) {}

void SGBTBooster::init(const Features& x) {
    // Feature order is lost in an unordered_map, so the canonical order is the
    // sorted key list of the first instance. Deterministic given the seed.
    std::vector<std::string> names;
    names.reserve(x.size());
    for (const auto& kv : x) names.push_back(kv.first);
    std::sort(names.begin(), names.end());

    const int n = static_cast<int>(names.size());
    int k = static_cast<int>(std::lround(n * (percentage_of_features_ / 100.0)));
    if (k < 2) k += 1;                       // MOA: bump a degenerate subspace
    k = std::max(1, std::min(k, n));

    subspaces_.clear();
    trees_.clear();
    subspaces_.reserve(n_iterations_);
    trees_.reserve(n_iterations_);

    std::vector<int> index(names.size());
    std::iota(index.begin(), index.end(), 0);
    for (int m = 0; m < n_iterations_; ++m) {
        // Partial shuffle: the first k entries are a uniform k-subset.
        for (int i = 0; i < k; ++i) {
            std::uniform_int_distribution<int> pick(i, n - 1);
            std::swap(index[i], index[pick(rng_)]);
        }
        std::vector<std::string> subspace;
        subspace.reserve(k);
        for (int i = 0; i < k; ++i) subspace.push_back(names[index[i]]);
        std::sort(subspace.begin(), subspace.end());
        subspaces_.push_back(std::move(subspace));

        std::vector<std::unique_ptr<FIMTDDRegressor>> bag;
        bag.reserve(bag_size_);
        for (int b = 0; b < bag_size_; ++b) {
            auto tree = std::make_unique<FIMTDDRegressor>(
                grace_period_, split_confidence_, 0.05, max_depth_, leaf_prediction_);
            bag.push_back(std::move(tree));
        }
        trees_.push_back(std::move(bag));
    }
    initialised_ = true;
}

void SGBTBooster::project(const Features& x,
                          const std::vector<std::string>& subspace,
                          Features& out) const {
    out.clear();
    out.reserve(subspace.size());
    for (const auto& name : subspace) {
        auto it = x.find(name);
        if (it != x.end()) out.emplace(name, it->second);
    }
}

void SGBTBooster::learn(const Features& x, double target) {
    if (skip_training_ > 1) {
        // MOA skips when nextInt(S) == 0, i.e. one instance in S is dropped.
        std::uniform_int_distribution<int> skip(0, skip_training_ - 1);
        if (skip(rng_) == 0) return;
    }
    if (!initialised_) init(x);

    double raw = 0.0;
    for (int m = 0; m < n_iterations_; ++m) {
        double g, h;
        if (squared_loss_) {
            g = target - raw;
            h = 1.0;
        } else {
            const double p = sigmoid(raw);      // P(output 0)
            g = target - p;
            h = std::max(p * (1.0 - p), kMinHessian);
        }
        project(x, subspaces_[m], scratch_);
        const Features& xs = scratch_;

        // Weighted squared loss: the base regressor is fitted to g/h. MOA gives
        // the hessian weight by training ceil(h * M) times rather than by
        // passing an instance weight.
        const double pseudo = g / h;
        int times = 1;
        if (multiply_hessian_by_ > 1) {
            times = std::max(1, static_cast<int>(std::ceil(h * multiply_hessian_by_)));
        }

        double sum = 0.0;
        for (auto& tree : trees_[m]) {
            // Online bagging (Oza & Russell): each bag member sees the instance
            // Poisson(1) times. A bag of one degenerates to plain training,
            // which is the classifier's configuration.
            int weight = 1;
            if (bag_size_ > 1) {
                std::poisson_distribution<int> poisson(1.0);
                weight = poisson(rng_);
            }
            for (int t = 0; t < times * weight; ++t) tree->learn_one(xs, pseudo);
            sum += tree->predict_one(xs);
        }
        raw += learning_rate_ * (sum / trees_[m].size());
    }
}

double SGBTBooster::raw_score(const Features& x, bool scale_by_lr) const {
    if (!initialised_) return 0.0;
    double raw = 0.0;
    for (int m = 0; m < n_iterations_; ++m) {
        project(x, subspaces_[m], scratch_);
        double sum = 0.0;
        for (const auto& tree : trees_[m]) sum += tree->predict_one(scratch_);
        raw += sum / trees_[m].size();
    }
    // The reference sums the base learners' outputs unscaled here while scaling
    // them by the learning rate during training. That is harmless in both of
    // its configurations — classification decides on sign(raw), and regression
    // runs at a learning rate of 1.0, where the scaling is the identity — so
    // the default reproduces it. scale_by_lr makes the two paths agree for any
    // other learning rate.
    return scale_by_lr ? raw * learning_rate_ : raw;
}

void SGBTBooster::reset() {
    initialised_ = false;
    subspaces_.clear();
    trees_.clear();
    rng_.seed(seed_);
}

// ─────────────────────────────────────────────────────────────────────────────
//  SGBTClassifier
// ─────────────────────────────────────────────────────────────────────────────

SGBTClassifier::SGBTClassifier(int n_models, double learning_rate,
                               int percentage_of_features, int multiply_hessian_by,
                               int skip_training, bool use_squared_loss, int bag_size,
                               int n_classes, bool scale_prediction_by_lr, int grace_period,
                               double split_confidence, int max_depth,
                               std::string leaf_prediction, int seed)
    : n_models(n_models),
      learning_rate(learning_rate),
      percentage_of_features(percentage_of_features),
      multiply_hessian_by(multiply_hessian_by),
      skip_training(skip_training),
      use_squared_loss(use_squared_loss),
      bag_size(bag_size),
      n_classes(n_classes),
      scale_prediction_by_lr(scale_prediction_by_lr),
      seed(seed),
      grace_period_(grace_period),
      max_depth_(max_depth),
      split_confidence_(split_confidence),
      leaf_prediction_(std::move(leaf_prediction)) {}

std::unique_ptr<SGBTBooster> SGBTClassifier::new_booster(size_t index) const {
    // MOA copies one prepared booster per class, and the copies carry the same
    // RNG state — so every booster draws the same subspaces. Reproduced by
    // seeding them all identically.
    (void)index;
    return std::make_unique<SGBTBooster>(
        n_models, learning_rate, percentage_of_features, multiply_hessian_by,
        skip_training, use_squared_loss, bag_size, grace_period_, split_confidence_,
        max_depth_, leaf_prediction_, static_cast<unsigned>(seed));
}

size_t SGBTClassifier::class_slot(int y) {
    auto it = std::lower_bound(classes_.begin(), classes_.end(), y);
    const size_t slot = static_cast<size_t>(it - classes_.begin());
    if (it == classes_.end() || *it != y) {
        classes_.insert(it, y);
        // The binary path keeps a single booster whatever the label set; the
        // one-vs-all path grows one booster per class, inserted in label order.
        if (!binary_mode()) {
            boosters_.insert(boosters_.begin() + slot, new_booster(slot));
        }
    }
    return slot;
}

void SGBTClassifier::learn_one(const Features& x, int y) {
    const size_t slot = class_slot(y);

    if (binary_mode()) {
        // The reference's binary path: a single booster whose output 0 is the
        // logit of the first class. Ground truth of output 0 is 1 when the
        // instance belongs to it. A third label would have nowhere to go, so
        // n_classes=2 is an assertion that the stream is binary; anything past
        // the first two labels is folded into the positive side.
        if (boosters_.empty()) boosters_.push_back(new_booster(0));
        boosters_[0]->learn(x, slot == 0 ? 1.0 : 0.0);
        return;
    }

    // One-vs-all: booster i is trained on "does this belong to classes_[i]",
    // and its output 0 is the ground truth of the *negative* side.
    for (size_t i = 0; i < boosters_.size(); ++i) {
        boosters_[i]->learn(x, i == slot ? 0.0 : 1.0);
    }
}

std::unordered_map<int, double> SGBTClassifier::predict_proba_one(const Features& x) const {
    std::unordered_map<int, double> out;
    if (classes_.empty()) return out;
    if (classes_.size() == 1) {
        out[classes_[0]] = 1.0;
        return out;
    }

    if (binary_mode()) {
        if (boosters_.empty()) return out;
        const double p0 = sigmoid(boosters_[0]->raw_score(x, scale_prediction_by_lr));
        out[classes_[0]] = p0;
        out[classes_[1]] = 1.0 - p0;
        return out;
    }

    double total = 0.0;
    std::vector<double> votes(boosters_.size(), 0.0);
    for (size_t i = 0; i < boosters_.size(); ++i) {
        // Vote for the positive class: 1 - softmax(raw, 0)[0].
        votes[i] = 1.0 - sigmoid(boosters_[i]->raw_score(x, scale_prediction_by_lr));
        total += votes[i];
    }
    for (size_t i = 0; i < classes_.size(); ++i) {
        out[classes_[i]] = total > 0.0 ? votes[i] / total : 0.0;
    }
    return out;
}

int SGBTClassifier::predict_one(const Features& x) const {
    const auto proba = predict_proba_one(x);
    if (proba.empty()) return 0;
    int best = proba.begin()->first;
    double best_p = -1.0;
    // Iterate classes_ rather than the map so ties break on the lower label.
    for (int c : classes_) {
        auto it = proba.find(c);
        if (it != proba.end() && it->second > best_p) {
            best_p = it->second;
            best = c;
        }
    }
    return best;
}

void SGBTClassifier::reset() {
    classes_.clear();
    boosters_.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
//  SGBRRegressor
// ─────────────────────────────────────────────────────────────────────────────

SGBRRegressor::SGBRRegressor(int n_models, double learning_rate,
                             int percentage_of_features, int multiply_hessian_by,
                             int skip_training, int bag_size,
                             int grace_period, double split_confidence, int max_depth,
                             std::string leaf_prediction, int seed)
    : n_models(n_models),
      learning_rate(learning_rate),
      percentage_of_features(percentage_of_features),
      multiply_hessian_by(multiply_hessian_by),
      skip_training(skip_training),
      bag_size(bag_size),
      seed(seed),
      grace_period_(grace_period),
      max_depth_(max_depth),
      split_confidence_(split_confidence),
      leaf_prediction_(std::move(leaf_prediction)) {}

void SGBRRegressor::learn_one(const Features& x, double y) {
    if (!booster_) {
        booster_ = std::make_unique<SGBTBooster>(
            n_models, learning_rate, percentage_of_features, multiply_hessian_by,
            skip_training, /*squared_loss=*/true, bag_size, grace_period_,
            split_confidence_, max_depth_, leaf_prediction_,
            static_cast<unsigned>(seed));
    }
    booster_->learn(x, y);
}

double SGBRRegressor::predict_one(const Features& x) const {
    if (!booster_) return 0.0;
    // learning_rate is 1.0 by default here, so the reference's unscaled sum is
    // already the score the trees were fitted against.
    return booster_->raw_score(x, /*scale_by_lr=*/false);
}

void SGBRRegressor::reset() { booster_.reset(); }
