#include "leveraging_bagging_reg.h"

#include <algorithm>
#include <cmath>

LeveragingBaggingRegressor::LeveragingBaggingRegressor(
    int n_models, int seed, double lambda_value, double drift_delta,
    int grace_period, int max_depth, double split_confidence,
    std::string leaf_prediction)
    : n_models(n_models),
      seed(seed),
      lambda_value(lambda_value),
      drift_delta_(drift_delta),
      split_confidence_(split_confidence),
      grace_period_(grace_period),
      max_depth_(max_depth),
      leaf_prediction_(std::move(leaf_prediction)),
      rng_(static_cast<unsigned>(seed)) {
    reset();
}

std::unique_ptr<FIMTDDRegressor> LeveragingBaggingRegressor::new_tree(int index) const {
    // Each member gets its own seed so their leaf models differ; the
    // resampling weights already differ through the shared RNG.
    auto tree = std::make_unique<FIMTDDRegressor>(
        grace_period_, split_confidence_, 0.05, max_depth_, leaf_prediction_);
    tree->seed = seed + index;
    tree->reset();
    return tree;
}

void LeveragingBaggingRegressor::reset() {
    members_.clear();
    members_.reserve(static_cast<size_t>(n_models));
    for (int i = 0; i < n_models; ++i) {
        members_.push_back({new_tree(i), std::make_unique<ADWIN>(drift_delta_)});
    }
    n_ = sum_ = sq_ = 0.0;
    rng_.seed(static_cast<unsigned>(seed));
}

void LeveragingBaggingRegressor::learn_one(const Features& x, double y) {
    n_ += 1.0;
    sum_ += y;
    sq_ += y * y;

    // Absolute error over the target's running spread. The classifier feeds
    // ADWIN a 0/1 error, already scale-free; a raw residual is not, and would
    // make the detector's sensitivity depend on the units of y.
    double scale = 1.0;
    if (n_ > 1.0) {
        const double var = (sq_ - (sum_ * sum_) / n_) / n_;
        if (var > 0.0) scale = std::sqrt(var);
    }

    std::poisson_distribution<int> poisson(lambda_value);
    for (auto& member : members_) {
        const double error = std::fabs(member.tree->predict_one(x) - y) / scale;
        member.drift->update(error);

        const int weight = poisson(rng_);
        for (int k = 0; k < weight; ++k) member.tree->learn_one(x, y);

        if (member.drift->drift_detected()) {
            // Reset outright, as the classifier does — no background tree.
            const int index = static_cast<int>(&member - members_.data());
            member.tree = new_tree(index);
            member.drift = std::make_unique<ADWIN>(drift_delta_);
        }
    }
}

double LeveragingBaggingRegressor::predict_one(const Features& x) const {
    if (members_.empty()) return 0.0;
    double total = 0.0;
    for (const auto& member : members_) total += member.tree->predict_one(x);
    return total / static_cast<double>(members_.size());
}
