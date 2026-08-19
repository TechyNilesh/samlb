#include "arf_reg.h"

#include <algorithm>
#include <cmath>

ARFRegressor::ARFRegressor(int n_models_, int seed_, double lambda_value_,
                           double drift_delta, double warning_delta,
                           int grace_period, int max_depth, double learning_rate)
    : n_models(n_models_),
      seed(seed_),
      lambda_value(lambda_value_),
      drift_delta_(drift_delta),
      warning_delta_(warning_delta),
      grace_period_(grace_period),
      max_depth_(max_depth),
      learning_rate_(learning_rate),
      rng_(static_cast<unsigned int>(seed_)) {
    reset();
}

std::unique_ptr<HoeffdingTreeRegressor> ARFRegressor::new_tree() const {
    return std::unique_ptr<HoeffdingTreeRegressor>(
        new HoeffdingTreeRegressor(grace_period_, 1e-7, 0.05, max_depth_, learning_rate_));
}

void ARFRegressor::reset() {
    rng_.seed(static_cast<unsigned int>(seed));
    members_.clear();
    members_.resize(static_cast<size_t>(n_models < 1 ? 1 : n_models));
    for (auto& m : members_) {
        m.tree = new_tree();
        m.background.reset();
        m.drift.reset(new ADWIN(drift_delta_));
        m.warning.reset(new ADWIN(warning_delta_));
        m.subspace.clear();
        m.subspace_ready = false;
    }
}

void ARFRegressor::init_subspace(Member& m, const Features& x) {
    std::vector<std::string> keys;
    keys.reserve(x.size());
    for (const auto& kv : x) keys.push_back(kv.first);
    // Deterministic base order: the hash-map iteration order is not stable
    // across runs, so sort before sampling to keep seeds reproducible.
    std::sort(keys.begin(), keys.end());

    const size_t k = std::max<size_t>(
        1, static_cast<size_t>(std::round(std::sqrt(static_cast<double>(keys.size())))));
    std::shuffle(keys.begin(), keys.end(), rng_);
    keys.resize(std::min(k, keys.size()));

    m.subspace = std::move(keys);
    m.subspace_ready = true;
}

Features ARFRegressor::project(const Member& m, const Features& x) const {
    if (m.subspace.empty()) return x;
    Features out;
    out.reserve(m.subspace.size());
    for (const auto& f : m.subspace) {
        const auto it = x.find(f);
        if (it != x.end()) out.emplace(f, it->second);
    }
    return out;
}

void ARFRegressor::learn_one(const Features& x, double y) {
    std::poisson_distribution<int> poisson(lambda_value);

    for (auto& m : members_) {
        if (!m.subspace_ready) init_subspace(m, x);
        const Features xs = project(m, x);

        // Error signal for the drift detectors, measured before the update.
        const double err = std::fabs(y - m.tree->predict_one(xs));

        const int k = poisson(rng_);
        for (int i = 0; i < k; ++i) m.tree->learn_one(xs, y);
        if (m.background) {
            for (int i = 0; i < k; ++i) m.background->learn_one(xs, y);
        }

        m.warning->update(err);
        if (m.warning->drift_detected() && !m.background) {
            m.background = new_tree();                    // start a replacement
            m.warning.reset(new ADWIN(warning_delta_));
        }

        m.drift->update(err);
        if (m.drift->drift_detected()) {
            if (m.background) m.tree = std::move(m.background);
            else              m.tree = new_tree();
            m.background.reset();
            m.drift.reset(new ADWIN(drift_delta_));
            m.warning.reset(new ADWIN(warning_delta_));
        }
    }
}

double ARFRegressor::predict_one(const Features& x) const {
    if (members_.empty()) return 0.0;
    double acc = 0.0;
    for (const auto& m : members_) {
        acc += m.tree->predict_one(project(m, x));
    }
    return acc / static_cast<double>(members_.size());
}
