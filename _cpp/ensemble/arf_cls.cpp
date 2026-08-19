#include "arf_cls.h"

#include <cmath>

ARFClassifier::ARFClassifier(int n_models_, int seed_, double lambda_value_,
                             double drift_delta, double warning_delta,
                             int grace_period, int max_depth,
                             double split_confidence, int subspace_size_)
    : n_models(n_models_),
      seed(seed_),
      lambda_value(lambda_value_),
      subspace_size(subspace_size_),
      drift_delta_(drift_delta),
      warning_delta_(warning_delta),
      split_confidence_(split_confidence),
      grace_period_(grace_period),
      max_depth_(max_depth),
      rng_(static_cast<unsigned int>(seed_)) {
    reset();
}

std::unique_ptr<HoeffdingTreeClassifier> ARFClassifier::new_tree(
        unsigned int subspace_seed, int n_features) const {
    auto tree = std::unique_ptr<HoeffdingTreeClassifier>(new HoeffdingTreeClassifier(
        grace_period_, split_confidence_, 0.05, 0, max_depth_, "info_gain"));

    int k = subspace_size;
    if (k < 0) {
        // ARF default: sqrt(M) + 1 features considered per split.
        k = static_cast<int>(std::sqrt(static_cast<double>(n_features))) + 1;
    }
    if (k > 0 && k < n_features) tree->set_subspace(k, subspace_seed);
    return tree;
}

void ARFClassifier::reset() {
    rng_.seed(static_cast<unsigned int>(seed));
    members_.clear();
    members_.resize(static_cast<size_t>(n_models < 1 ? 1 : n_models));
    for (auto& m : members_) {
        m.tree.reset();
        m.background.reset();
        m.drift.reset(new ADWIN(drift_delta_));
        m.warning.reset(new ADWIN(warning_delta_));
        m.correct = 0.0;
        m.seen = 0.0;
        m.ready = false;
    }
}

void ARFClassifier::configure(Member& m, size_t index, const Features& x) {
    // The feature count is only known once the first instance arrives.
    const int n_features = static_cast<int>(x.size());
    m.tree = new_tree(static_cast<unsigned int>(seed + 1000 * (index + 1)), n_features);
    m.ready = true;
}

void ARFClassifier::learn_one(const Features& x, int y) {
    std::poisson_distribution<int> poisson(lambda_value);

    for (size_t i = 0; i < members_.size(); ++i) {
        Member& m = members_[i];
        if (!m.ready) configure(m, i, x);

        // Error signal for the detectors, measured before this update.
        const int pred = m.tree->predict_one(x);
        const bool correct = (pred == y);
        m.correct += correct ? 1.0 : 0.0;
        m.seen += 1.0;

        const int k = poisson(rng_);
        for (int j = 0; j < k; ++j) m.tree->learn_one(x, y);
        if (m.background) {
            for (int j = 0; j < k; ++j) m.background->learn_one(x, y);
        }

        const double err = correct ? 0.0 : 1.0;

        m.warning->update(err);
        if (m.warning->drift_detected() && !m.background) {
            m.background = new_tree(
                static_cast<unsigned int>(seed + 7919 * (i + 1) + static_cast<int>(m.seen)),
                static_cast<int>(x.size()));
            m.warning.reset(new ADWIN(warning_delta_));
        }

        m.drift->update(err);
        if (m.drift->drift_detected()) {
            if (m.background) m.tree = std::move(m.background);
            else              m.tree = new_tree(
                static_cast<unsigned int>(seed + 104729 * (i + 1) + static_cast<int>(m.seen)),
                static_cast<int>(x.size()));
            m.background.reset();
            m.drift.reset(new ADWIN(drift_delta_));
            m.warning.reset(new ADWIN(warning_delta_));
            m.correct = 0.0;
            m.seen = 0.0;
        }
    }
}

std::unordered_map<int, double> ARFClassifier::predict_proba_one(const Features& x) const {
    std::unordered_map<int, double> votes;
    double total = 0.0;

    for (const auto& m : members_) {
        if (!m.ready || !m.tree) continue;
        auto proba = m.tree->predict_proba_one(x);
        double sum = 0.0;
        for (const auto& kv : proba) sum += kv.second;
        if (sum <= 0.0) continue;

        // Accuracy-weighted vote; an untested tree still counts, weakly.
        const double weight = m.seen > 0.0 ? m.correct / m.seen : 1.0;
        if (weight <= 0.0) continue;
        for (const auto& kv : proba) {
            votes[kv.first] += (kv.second / sum) * weight;
            total += (kv.second / sum) * weight;
        }
    }

    if (total > 0.0) {
        for (auto& kv : votes) kv.second /= total;
    }
    return votes;
}

int ARFClassifier::predict_one(const Features& x) const {
    auto votes = predict_proba_one(x);
    int best = 0;
    double best_v = -1.0;
    for (const auto& kv : votes) {
        if (kv.second > best_v) { best_v = kv.second; best = kv.first; }
    }
    return best;
}
