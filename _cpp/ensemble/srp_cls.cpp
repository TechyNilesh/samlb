#include "srp_cls.h"

#include <algorithm>
#include <cmath>

SRPClassifier::SRPClassifier(int n_models_, int seed_, double lambda_value_,
                             double drift_delta, double warning_delta,
                             int grace_period, int max_depth,
                             double split_confidence, double subspace_fraction_,
                             const std::string& training_method_)
    : n_models(n_models_),
      seed(seed_),
      lambda_value(lambda_value_),
      subspace_fraction(subspace_fraction_),
      training_method(training_method_),
      drift_delta_(drift_delta),
      warning_delta_(warning_delta),
      split_confidence_(split_confidence),
      grace_period_(grace_period),
      max_depth_(max_depth),
      rng_(static_cast<unsigned int>(seed_)) {
    reset();
}

std::unique_ptr<HoeffdingTreeClassifier> SRPClassifier::new_tree() const {
    // Plain tree: SRP does the feature randomisation outside the learner.
    return std::unique_ptr<HoeffdingTreeClassifier>(new HoeffdingTreeClassifier(
        grace_period_, split_confidence_, 0.05, 0, max_depth_, "info_gain"));
}

void SRPClassifier::reset() {
    rng_.seed(static_cast<unsigned int>(seed));
    members_.clear();
    members_.resize(static_cast<size_t>(n_models < 1 ? 1 : n_models));
    for (auto& m : members_) {
        m.tree.reset();
        m.background.reset();
        m.drift.reset(new ADWIN(drift_delta_));
        m.warning.reset(new ADWIN(warning_delta_));
        m.subspace.clear();
        m.correct = 0.0;
        m.seen = 0.0;
        m.ready = false;
    }
}

void SRPClassifier::configure(Member& m, size_t index, const Features& x) {
    m.tree = new_tree();

    if (training_method != "resampling") {
        std::vector<std::string> keys;
        keys.reserve(x.size());
        for (const auto& kv : x) keys.push_back(kv.first);
        // Sort first: unordered_map iteration order is not reproducible across
        // runs, and the subspace must be a function of the seed alone.
        std::sort(keys.begin(), keys.end());

        size_t k = static_cast<size_t>(
            std::round(subspace_fraction * static_cast<double>(keys.size())));
        k = std::max<size_t>(1, std::min(k, keys.size()));

        std::mt19937 member_rng(static_cast<unsigned int>(seed + 7919 * (index + 1)));
        std::shuffle(keys.begin(), keys.end(), member_rng);
        keys.resize(k);
        std::sort(keys.begin(), keys.end());
        m.subspace = std::move(keys);
    }
    m.ready = true;
}

Features SRPClassifier::project(const Member& m, const Features& x) const {
    if (m.subspace.empty()) return x;
    Features out;
    out.reserve(m.subspace.size());
    for (const auto& f : m.subspace) {
        const auto it = x.find(f);
        if (it != x.end()) out.emplace(f, it->second);
    }
    return out;
}

void SRPClassifier::learn_one(const Features& x, int y) {
    std::poisson_distribution<int> poisson(lambda_value);
    const bool resample = (training_method != "subspaces");

    for (size_t i = 0; i < members_.size(); ++i) {
        Member& m = members_[i];
        if (!m.ready) configure(m, i, x);

        const Features xs = project(m, x);

        const bool correct = (m.tree->predict_one(xs) == y);
        m.correct += correct ? 1.0 : 0.0;
        m.seen += 1.0;

        const int k = resample ? poisson(rng_) : 1;
        for (int j = 0; j < k; ++j) m.tree->learn_one(xs, y);
        if (m.background) {
            for (int j = 0; j < k; ++j) m.background->learn_one(xs, y);
        }

        const double err = correct ? 0.0 : 1.0;

        m.warning->update(err);
        if (m.warning->drift_detected() && !m.background) {
            m.background = new_tree();
            m.warning.reset(new ADWIN(warning_delta_));
        }

        m.drift->update(err);
        if (m.drift->drift_detected()) {
            m.tree = m.background ? std::move(m.background) : new_tree();
            m.background.reset();
            m.drift.reset(new ADWIN(drift_delta_));
            m.warning.reset(new ADWIN(warning_delta_));
            m.correct = 0.0;
            m.seen = 0.0;
        }
    }
}

std::unordered_map<int, double> SRPClassifier::predict_proba_one(const Features& x) const {
    std::unordered_map<int, double> votes;
    double total = 0.0;

    for (const auto& m : members_) {
        if (!m.ready || !m.tree) continue;
        const auto proba = m.tree->predict_proba_one(project(m, x));
        double sum = 0.0;
        for (const auto& kv : proba) sum += kv.second;
        if (sum <= 0.0) continue;

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

int SRPClassifier::predict_one(const Features& x) const {
    const auto votes = predict_proba_one(x);
    int best = 0;
    double best_v = -1.0;
    for (const auto& kv : votes) {
        if (kv.second > best_v) { best_v = kv.second; best = kv.first; }
    }
    return best;
}
