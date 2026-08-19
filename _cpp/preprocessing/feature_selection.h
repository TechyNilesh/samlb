#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../core/estimator.h"
#include "../core/running_stats.h"

// Drop-in C++ replacements for river.feature_selection.{VarianceThreshold,
// SelectKBest}. SelectKBest is hard-wired to PearsonCorr similarity, which is
// the only similarity SAMLB configures.

class VarianceThreshold : public ITransformer {
public:
    explicit VarianceThreshold(double threshold = 0.0, int min_samples = 2);

    void     learn_one(const Features& x) override;
    Features transform_one(const Features& x) const override;
    void     transform_inplace(Features& x) const override;
    void     reset() override;

    double threshold;
    int    min_samples;

private:
    bool keep(const std::string& f) const;
    std::unordered_map<std::string, RVar> variances_;
};

class SelectKBest : public ITransformer {
public:
    explicit SelectKBest(int k = 10, bool use_abs = false);

    void     learn_one(const Features& x) override { (void)x; }  // supervised only
    void     learn_one_sup(const Features& x, double y) override;
    Features transform_one(const Features& x) const override;
    void     transform_inplace(Features& x) const override;
    bool     is_supervised() const override { return true; }
    void     set_feature_order(const std::vector<std::string>& order) override;
    void     reset() override;

    int  k;
    bool use_abs;

private:
    void refresh_best() const;

    std::unordered_map<std::string, RPearsonCorr> sims_;
    std::unordered_map<std::string, double>       score_;
    std::vector<std::string>                      keys_;    // every feature seen
    std::unordered_map<std::string, int>          rank_;    // tie-break priority
    int                                           next_rank_ = 0;

    // Recomputing the leaderboard is deferred to the first transform after a
    // learn, so predict+learn on one instance costs a single sort.
    mutable std::unordered_set<std::string> best_;
    mutable bool dirty_ = true;
};
