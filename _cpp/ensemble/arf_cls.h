#pragma once
#include <memory>
#include <random>
#include <vector>

#include "../classification/hoeffding_tree.h"
#include "../core/estimator.h"
#include "../drift/adwin.h"

// Adaptive Random Forest (Gomes et al., Machine Learning 2017).
//
// The standard streaming baseline: online bagging over Hoeffding trees, each
// tree resampling a random feature subspace at every split attempt, with a
// warning/drift detector pair per tree. On warning a background tree starts
// training; on drift it replaces the foreground tree. Votes are weighted by
// each tree's running accuracy.
//
// The per-node subspace resampling (rather than one fixed subspace per tree)
// is what separates ARF from SRP.
class ARFClassifier : public IClassifier {
public:
    explicit ARFClassifier(int n_models = 10,
                           int seed = 0,
                           double lambda_value = 6.0,
                           double drift_delta = 0.001,
                           double warning_delta = 0.01,
                           int grace_period = 50,
                           int max_depth = 20,
                           double split_confidence = 0.01,
                           int subspace_size = -1);   // -1 = sqrt(M) + 1

    void                            learn_one(const Features& x, int y) override;
    int                             predict_one(const Features& x) const override;
    std::unordered_map<int, double> predict_proba_one(const Features& x) const override;
    void                            reset() override;

    int    n_models;
    int    seed;
    double lambda_value;
    int    subspace_size;

private:
    struct Member {
        std::unique_ptr<HoeffdingTreeClassifier> tree;
        std::unique_ptr<HoeffdingTreeClassifier> background;
        std::unique_ptr<ADWIN>                   drift;
        std::unique_ptr<ADWIN>                   warning;
        double correct = 0.0;
        double seen    = 0.0;
        bool   ready   = false;   // subspace assigned once the width is known
    };

    std::unique_ptr<HoeffdingTreeClassifier> new_tree(unsigned int subspace_seed,
                                                      int n_features) const;
    void configure(Member& m, size_t index, const Features& x);

    double drift_delta_, warning_delta_, split_confidence_;
    int    grace_period_, max_depth_;

    std::vector<Member>  members_;
    mutable std::mt19937 rng_;
};
