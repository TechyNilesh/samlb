#pragma once
#include <string>
#include <unordered_map>

#include "../core/estimator.h"
#include "../core/running_stats.h"

// Drop-in C++ replacements for river.preprocessing.{StandardScaler,
// MinMaxScaler, MaxAbsScaler}. Update/transform formulas match River exactly,
// including its behaviour for unseen features and zero-variance columns.

class StandardScaler : public ITransformer {
public:
    explicit StandardScaler(bool with_std = true);

    void     learn_one(const Features& x) override;
    Features transform_one(const Features& x) const override;
    void     transform_inplace(Features& x) const override;
    void     reset() override;

    bool with_std;

private:
    // river keeps counts/means/vars in defaultdicts; one struct per feature is
    // the same state with one hash lookup instead of three.
    struct Stat { double count = 0.0; double mean = 0.0; double var = 0.0; };
    std::unordered_map<std::string, Stat> stats_;
};

class MinMaxScaler : public ITransformer {
public:
    MinMaxScaler();

    void     learn_one(const Features& x) override;
    Features transform_one(const Features& x) const override;
    void     transform_inplace(Features& x) const override;
    void     reset() override;

private:
    struct Stat { RMin lo; RMax hi; };
    std::unordered_map<std::string, Stat> stats_;
};

class MaxAbsScaler : public ITransformer {
public:
    MaxAbsScaler();

    void     learn_one(const Features& x) override;
    Features transform_one(const Features& x) const override;
    void     transform_inplace(Features& x) const override;
    void     reset() override;

private:
    std::unordered_map<std::string, RAbsMax> stats_;
};
