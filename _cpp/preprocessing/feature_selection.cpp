#include "feature_selection.h"

#include <algorithm>
#include <cmath>

// ── VarianceThreshold ────────────────────────────────────────────────────────

VarianceThreshold::VarianceThreshold(double threshold_, int min_samples_)
    : threshold(threshold_), min_samples(min_samples_) {}

void VarianceThreshold::learn_one(const Features& x) {
    for (const auto& kv : x) variances_[kv.first].update(kv.second);
}

bool VarianceThreshold::keep(const std::string& f) const {
    const auto it = variances_.find(f);
    if (it == variances_.end()) return true;                    // unseen -> keep
    if (it->second.n() < static_cast<double>(min_samples)) return true;
    return it->second.get() > threshold;
}

void VarianceThreshold::transform_inplace(Features& x) const {
    for (auto it = x.begin(); it != x.end(); ) {
        if (keep(it->first)) ++it;
        else it = x.erase(it);
    }
}

Features VarianceThreshold::transform_one(const Features& x) const {
    Features out;
    out.reserve(x.size());
    for (const auto& kv : x) if (keep(kv.first)) out.emplace(kv.first, kv.second);
    return out;
}

void VarianceThreshold::reset() { variances_.clear(); }

// ── SelectKBest (PearsonCorr) ────────────────────────────────────────────────

SelectKBest::SelectKBest(int k_, bool use_abs_) : k(k_), use_abs(use_abs_) {}

void SelectKBest::set_feature_order(const std::vector<std::string>& order) {
    // Declared up front, so ranks line up with the dataset's column order —
    // the order River's Counter would have used.
    for (const auto& f : order) {
        if (!rank_.count(f)) rank_[f] = next_rank_++;
    }
    dirty_ = true;
}

void SelectKBest::learn_one_sup(const Features& x, double y) {
    for (const auto& kv : x) {
        auto it = sims_.find(kv.first);
        if (it == sims_.end()) {
            it = sims_.emplace(kv.first, RPearsonCorr{}).first;
            keys_.push_back(kv.first);
            // Any feature not covered by set_feature_order falls back to
            // first-seen order.
            if (!rank_.count(kv.first)) rank_[kv.first] = next_rank_++;
        }
        it->second.update(kv.second, y);
        const double s = it->second.get();
        score_[kv.first] = use_abs ? std::fabs(s) : s;
    }
    dirty_ = true;
}

void SelectKBest::refresh_best() const {
    best_.clear();
    if (!keys_.empty()) {
        // Mirrors Counter.most_common(k): descending score, ties broken by
        // declared feature order.
        std::vector<const std::string*> keys;
        keys.reserve(keys_.size());
        for (const auto& f : keys_) keys.push_back(&f);
        std::sort(keys.begin(), keys.end(),
                  [this](const std::string* a, const std::string* b) {
                      const double sa = score_.at(*a), sb = score_.at(*b);
                      if (sa != sb) return sa > sb;
                      return rank_.at(*a) < rank_.at(*b);
                  });
        const size_t n = std::min<size_t>(keys.size(), static_cast<size_t>(k < 0 ? 0 : k));
        best_.reserve(n);
        for (size_t i = 0; i < n; ++i) best_.insert(*keys[i]);
    }
    dirty_ = false;
}

void SelectKBest::transform_inplace(Features& x) const {
    if (dirty_) refresh_best();
    if (keys_.empty()) return;                                   // river: pass through
    for (auto it = x.begin(); it != x.end(); ) {
        if (best_.count(it->first)) ++it;
        else it = x.erase(it);
    }
}

Features SelectKBest::transform_one(const Features& x) const {
    if (dirty_) refresh_best();
    if (keys_.empty()) return x;
    Features out;
    out.reserve(std::min(x.size(), best_.size()));
    for (const auto& kv : x) if (best_.count(kv.first)) out.emplace(kv.first, kv.second);
    return out;
}

void SelectKBest::reset() {
    sims_.clear();
    score_.clear();
    keys_.clear();
    best_.clear();
    dirty_ = true;
    // rank_/next_rank_ survive reset: the declared column order is a property
    // of the stream, not of the fitted state.
}
