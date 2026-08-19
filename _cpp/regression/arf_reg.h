#pragma once
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../core/estimator.h"
#include "../drift/adwin.h"
#include "hoeffding_tree_reg.h"

// Adaptive Random Forest regressor — replaces river.forest.ARFRegressor, which
// AutoClass uses as a surrogate fitness model.
//
// Faithful to the ARF scheme: Poisson(lambda) online bagging, a random feature
// subspace per tree, and an ADWIN warning/drift pair per tree that trains a
// background tree on warning and promotes it on drift. Aggregation is the
// mean of the member predictions.
class ARFRegressor : public IRegressor {
public:
    explicit ARFRegressor(int n_models = 10,
                          int seed = 0,
                          double lambda_value = 6.0,
                          double drift_delta = 0.001,
                          double warning_delta = 0.01,
                          int grace_period = 200,
                          int max_depth = 20,
                          double learning_rate = 0.01);

    void   learn_one(const Features& x, double y) override;
    double predict_one(const Features& x) const override;
    void   reset() override;

    int    n_models;
    int    seed;
    double lambda_value;

private:
    struct Member {
        std::unique_ptr<HoeffdingTreeRegressor> tree;
        std::unique_ptr<HoeffdingTreeRegressor> background;
        std::unique_ptr<ADWIN>                  drift;
        std::unique_ptr<ADWIN>                  warning;
        std::vector<std::string>                subspace;
        bool                                    subspace_ready = false;
    };

    std::unique_ptr<HoeffdingTreeRegressor> new_tree() const;
    void     init_subspace(Member& m, const Features& x);
    Features project(const Member& m, const Features& x) const;

    double drift_delta_;
    double warning_delta_;
    int    grace_period_;
    int    max_depth_;
    double learning_rate_;

    std::vector<Member>  members_;
    mutable std::mt19937 rng_;
};
