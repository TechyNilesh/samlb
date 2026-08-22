#include "leveraging_bagging_cls.h"

LeveragingBaggingClassifier::LeveragingBaggingClassifier(
    int n_models_, int seed_, double lambda_value_, double drift_delta,
    int grace_period, int max_depth, double split_confidence)
    : n_models(n_models_),
      seed(seed_),
      lambda_value(lambda_value_),
      drift_delta_(drift_delta),
      split_confidence_(split_confidence),
      grace_period_(grace_period),
      max_depth_(max_depth),
      rng_(static_cast<unsigned int>(seed_)) {
    reset();
}

std::unique_ptr<HoeffdingTreeClassifier> LeveragingBaggingClassifier::new_tree() const {
    return std::unique_ptr<HoeffdingTreeClassifier>(new HoeffdingTreeClassifier(
        grace_period_, split_confidence_, 0.05, 0, max_depth_, "info_gain"));
}

void LeveragingBaggingClassifier::reset() {
    rng_.seed(static_cast<unsigned int>(seed));
    members_.clear();
    members_.resize(static_cast<size_t>(n_models < 1 ? 1 : n_models));
    for (auto& m : members_) {
        m.tree = new_tree();
        m.drift.reset(new ADWIN(drift_delta_));
    }
}

void LeveragingBaggingClassifier::learn_one(const Features& x, int y) {
    std::poisson_distribution<int> poisson(lambda_value);

    for (auto& m : members_) {
        // Error signal for this member's own detector, measured before update.
        const bool correct = (m.tree->predict_one(x) == y);
        const double err   = correct ? 0.0 : 1.0;

        const int k = poisson(rng_);
        for (int j = 0; j < k; ++j) m.tree->learn_one(x, y);

        m.drift->update(err);
        if (m.drift->drift_detected()) {
            // Leveraging bagging resets the member outright rather than
            // growing a background replacement first.
            m.tree = new_tree();
            m.drift.reset(new ADWIN(drift_delta_));
        }
    }
}

std::unordered_map<int, double> LeveragingBaggingClassifier::predict_proba_one(
        const Features& x) const {
    std::unordered_map<int, double> votes;
    double total = 0.0;

    for (const auto& m : members_) {
        if (!m.tree) continue;
        const auto proba = m.tree->predict_proba_one(x);
        double sum = 0.0;
        for (const auto& kv : proba) sum += kv.second;
        if (sum <= 0.0) continue;
        for (const auto& kv : proba) {
            votes[kv.first] += kv.second / sum;
            total += kv.second / sum;
        }
    }

    if (total > 0.0) {
        for (auto& kv : votes) kv.second /= total;
    }
    return votes;
}

int LeveragingBaggingClassifier::predict_one(const Features& x) const {
    const auto votes = predict_proba_one(x);
    int best = 0;
    double best_v = -1.0;
    for (const auto& kv : votes) {
        if (kv.second > best_v) { best_v = kv.second; best = kv.first; }
    }
    return best;
}
