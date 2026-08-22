#include "srp_reg.h"

#include <algorithm>
#include <cmath>

SRPRegressor::SRPRegressor(int n_models_, int seed_, double lambda_value_,
                           double drift_delta, double warning_delta,
                           int grace_period, int max_depth, double learning_rate,
                           double subspace_fraction_,
                           const std::string& training_method_)
    : n_models(n_models_),
      seed(seed_),
      lambda_value(lambda_value_),
      subspace_fraction(subspace_fraction_),
      training_method(training_method_),
      learning_rate_(learning_rate),
      grace_period_(grace_period),
      max_depth_(max_depth),
      drift_delta_(drift_delta),
      warning_delta_(warning_delta),
      rng_(static_cast<unsigned int>(seed_)) {
    reset();
}

std::unique_ptr<HoeffdingTreeRegressor> SRPRegressor::new_tree() const {
    // Plain tree: SRP does the feature randomisation outside the learner.
    return std::unique_ptr<HoeffdingTreeRegressor>(
        new HoeffdingTreeRegressor(grace_period_, 1e-7, 0.05, max_depth_, learning_rate_));
}

void SRPRegressor::reset() {
    rng_.seed(static_cast<unsigned int>(seed));
    members_.clear();
    members_.resize(static_cast<size_t>(n_models < 1 ? 1 : n_models));
    for (auto& m : members_) {
        m.tree.reset();
        m.background.reset();
        m.drift.reset(new ADWIN(drift_delta_));
        m.warning.reset(new ADWIN(warning_delta_));
        m.subspace.clear();
        m.ready = false;
    }
}

void SRPRegressor::configure(Member& m, size_t index, const Features& x) {
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

Features SRPRegressor::project(const Member& m, const Features& x) const {
    if (m.subspace.empty()) return x;
    Features out;
    out.reserve(m.subspace.size());
    for (const auto& f : m.subspace) {
        const auto it = x.find(f);
        if (it != x.end()) out.emplace(f, it->second);
    }
    return out;
}

void SRPRegressor::learn_one(const Features& x, double y) {
    std::poisson_distribution<int> poisson(lambda_value);
    const bool resample = (training_method != "subspaces");

    for (size_t i = 0; i < members_.size(); ++i) {
        Member& m = members_[i];
        if (!m.ready) configure(m, i, x);

        const Features xs = project(m, x);

        // Error signal for the drift detectors, measured before this update.
        const double err = std::fabs(y - m.tree->predict_one(xs));

        const int k = resample ? poisson(rng_) : 1;
        for (int j = 0; j < k; ++j) m.tree->learn_one(xs, y);
        if (m.background) {
            for (int j = 0; j < k; ++j) m.background->learn_one(xs, y);
        }

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
        }
    }
}

double SRPRegressor::predict_one(const Features& x) const {
    if (members_.empty()) return 0.0;
    double acc = 0.0;
    int n = 0;
    for (const auto& m : members_) {
        if (!m.ready || !m.tree) continue;
        acc += m.tree->predict_one(project(m, x));
        ++n;
    }
    return n > 0 ? acc / static_cast<double>(n) : 0.0;
}
