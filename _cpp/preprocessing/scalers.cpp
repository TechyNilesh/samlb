#include "scalers.h"

#include <cmath>

// ── StandardScaler ───────────────────────────────────────────────────────────

StandardScaler::StandardScaler(bool with_std_) : with_std(with_std_) {}

void StandardScaler::learn_one(const Features& x) {
    if (with_std) {
        for (const auto& kv : x) {
            Stat& s = stats_[kv.first];
            s.count += 1.0;
            const double old_mean = s.mean;
            s.mean += (kv.second - old_mean) / s.count;
            s.var  += ((kv.second - old_mean) * (kv.second - s.mean) - s.var) / s.count;
        }
    } else {
        for (const auto& kv : x) {
            Stat& s = stats_[kv.first];
            s.count += 1.0;
            s.mean += (kv.second - s.mean) / s.count;
        }
    }
}

void StandardScaler::transform_inplace(Features& x) const {
    if (with_std) {
        for (auto& kv : x) {
            const auto it = stats_.find(kv.first);
            if (it == stats_.end()) { kv.second = 0.0; continue; }  // river: var 0 -> 0.0
            const double v = it->second.var;
            kv.second = (v != 0.0) ? (kv.second - it->second.mean) / std::sqrt(v) : 0.0;
        }
    } else {
        for (auto& kv : x) {
            const auto it = stats_.find(kv.first);
            kv.second -= (it == stats_.end()) ? 0.0 : it->second.mean;
        }
    }
}

Features StandardScaler::transform_one(const Features& x) const {
    Features out(x);
    transform_inplace(out);
    return out;
}

void StandardScaler::reset() { stats_.clear(); }

// ── MinMaxScaler ─────────────────────────────────────────────────────────────

MinMaxScaler::MinMaxScaler() {}

void MinMaxScaler::learn_one(const Features& x) {
    for (const auto& kv : x) {
        Stat& s = stats_[kv.first];
        s.lo.update(kv.second);
        s.hi.update(kv.second);
    }
}

void MinMaxScaler::transform_inplace(Features& x) const {
    for (auto& kv : x) {
        const auto it = stats_.find(kv.first);
        if (it == stats_.end()) {
            // river yields NaN here (its unseen min/max are +/-inf). NaN is not
            // representable under -ffast-math, and a NaN feature is useless to
            // every downstream learner, so an unseen feature scales to 0 —
            // consistent with StandardScaler and MaxAbsScaler. This can only
            // differ from river on a predict that precedes any learn.
            kv.second = 0.0;
            continue;
        }
        const double lo = it->second.lo.get();
        const double d  = it->second.hi.get() - lo;
        kv.second = (d != 0.0) ? (kv.second - lo) / d : 0.0;
    }
}

Features MinMaxScaler::transform_one(const Features& x) const {
    Features out(x);
    transform_inplace(out);
    return out;
}

void MinMaxScaler::reset() { stats_.clear(); }

// ── MaxAbsScaler ─────────────────────────────────────────────────────────────

MaxAbsScaler::MaxAbsScaler() {}

void MaxAbsScaler::learn_one(const Features& x) {
    for (const auto& kv : x) stats_[kv.first].update(kv.second);
}

void MaxAbsScaler::transform_inplace(Features& x) const {
    for (auto& kv : x) {
        const auto it = stats_.find(kv.first);
        const double m = (it == stats_.end()) ? 0.0 : it->second.get();
        kv.second = (m != 0.0) ? kv.second / m : 0.0;
    }
}

Features MaxAbsScaler::transform_one(const Features& x) const {
    Features out(x);
    transform_inplace(out);
    return out;
}

void MaxAbsScaler::reset() { stats_.clear(); }
