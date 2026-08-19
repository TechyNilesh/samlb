#pragma once
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../classification/hoeffding_tree.h"
#include "../core/estimator.h"
#include "../drift/adwin.h"

// Streaming Random Patches (Gomes et al., 2019).
//
// The distinction from ARF is where the feature randomisation happens. ARF
// resamples a subspace at *every split attempt* inside the tree. SRP draws one
// random subset of features per ensemble member and keeps it for that member's
// whole life, so the base learner is an unmodified tree fed a projected
// instance. Combined with Poisson resampling that gives a random *patch*
// (subset of features x subset of instances).
//
// training_method:
//   "patches"    — subspace + resampling (the paper's default, RP)
//   "subspaces"  — subspace only, every member sees every instance once (RS)
//   "resampling" — resampling only, all features (RE)
class SRPClassifier : public IClassifier {
public:
    explicit SRPClassifier(int n_models = 10,
                           int seed = 0,
                           double lambda_value = 6.0,
                           double drift_delta = 0.001,
                           double warning_delta = 0.01,
                           int grace_period = 50,
                           int max_depth = 20,
                           double split_confidence = 0.01,
                           double subspace_fraction = 0.6,
                           const std::string& training_method = "patches");

    void                            learn_one(const Features& x, int y) override;
    int                             predict_one(const Features& x) const override;
    std::unordered_map<int, double> predict_proba_one(const Features& x) const override;
    void                            reset() override;

    int         n_models;
    int         seed;
    double      lambda_value;
    double      subspace_fraction;
    std::string training_method;

private:
    struct Member {
        std::unique_ptr<HoeffdingTreeClassifier> tree;
        std::unique_ptr<HoeffdingTreeClassifier> background;
        std::unique_ptr<ADWIN>                   drift;
        std::unique_ptr<ADWIN>                   warning;
        std::vector<std::string>                 subspace;   // empty = all features
        double correct = 0.0;
        double seen    = 0.0;
        bool   ready   = false;
    };

    std::unique_ptr<HoeffdingTreeClassifier> new_tree() const;
    void     configure(Member& m, size_t index, const Features& x);
    Features project(const Member& m, const Features& x) const;

    double drift_delta_, warning_delta_, split_confidence_;
    int    grace_period_, max_depth_;

    std::vector<Member>  members_;
    mutable std::mt19937 rng_;
};
