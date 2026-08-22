#pragma once
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../core/estimator.h"
#include "../drift/adwin.h"
#include "../regression/hoeffding_tree_reg.h"

// Streaming Random Patches — regression variant (Gomes et al., 2019).
//
// Same distinction from ARFRegressor as SRPClassifier has from ARFClassifier:
// ARF resamples a subspace at *every split attempt* inside the tree, SRP draws
// one random feature subset per ensemble member and keeps it for that
// member's whole life, feeding an unmodified regression tree a projected
// instance. Combined with Poisson resampling that gives a random *patch*.
//
// training_method:
//   "patches"    — subspace + resampling (the paper's default, RP)
//   "subspaces"  — subspace only, every member sees every instance once (RS)
//   "resampling" — resampling only, all features (RE)
class SRPRegressor : public IRegressor {
public:
    explicit SRPRegressor(int n_models = 10,
                          int seed = 0,
                          double lambda_value = 6.0,
                          double drift_delta = 0.001,
                          double warning_delta = 0.01,
                          int grace_period = 200,
                          int max_depth = 20,
                          double learning_rate = 0.01,
                          double subspace_fraction = 0.6,
                          const std::string& training_method = "patches");

    void   learn_one(const Features& x, double y) override;
    double predict_one(const Features& x) const override;
    void   reset() override;

    int         n_models;
    int         seed;
    double      lambda_value;
    double      subspace_fraction;
    std::string training_method;

private:
    struct Member {
        std::unique_ptr<HoeffdingTreeRegressor> tree;
        std::unique_ptr<HoeffdingTreeRegressor> background;
        std::unique_ptr<ADWIN>                  drift;
        std::unique_ptr<ADWIN>                  warning;
        std::vector<std::string>                subspace;   // empty = all features
        bool                                    ready = false;
    };

    std::unique_ptr<HoeffdingTreeRegressor> new_tree() const;
    void     configure(Member& m, size_t index, const Features& x);
    Features project(const Member& m, const Features& x) const;

    double learning_rate_;
    int    grace_period_, max_depth_;
    double drift_delta_, warning_delta_;

    std::vector<Member>  members_;
    mutable std::mt19937 rng_;
};
