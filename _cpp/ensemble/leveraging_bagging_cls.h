#pragma once
#include <memory>
#include <random>
#include <vector>

#include "../classification/hoeffding_tree.h"
#include "../core/estimator.h"
#include "../drift/adwin.h"

// Leveraging Bagging (Bifet, Holmes & Pfahringer, ICDM 2010).
//
// Differs from ARF/SRP in both where diversity comes from and how drift is
// handled:
//   - Diversity: plain online bagging resamples each instance with
//     Poisson(1); leveraging bagging "leverages" that by using a higher
//     lambda (default 6, matching MOA's LeveragingBag), which increases the
//     input-space diversity across members without any feature subspacing.
//   - Drift: each member carries its own ADWIN on its own error. ARF/SRP grow
//     a background replacement and swap it in once ready; leveraging bagging
//     instead resets the member immediately (a fresh, empty tree) the moment
//     its own ADWIN fires — matching MOA's per-classifier ADWINChangeDetector
//     reset behaviour.
//   - Voting: unweighted majority across members (no accuracy weighting).
class LeveragingBaggingClassifier : public IClassifier {
public:
    explicit LeveragingBaggingClassifier(int n_models = 10,
                                         int seed = 0,
                                         double lambda_value = 6.0,
                                         double drift_delta = 0.002,
                                         int grace_period = 50,
                                         int max_depth = 20,
                                         double split_confidence = 0.01);

    void                            learn_one(const Features& x, int y) override;
    int                             predict_one(const Features& x) const override;
    std::unordered_map<int, double> predict_proba_one(const Features& x) const override;
    void                            reset() override;

    int    n_models;
    int    seed;
    double lambda_value;

private:
    struct Member {
        std::unique_ptr<HoeffdingTreeClassifier> tree;
        std::unique_ptr<ADWIN>                   drift;
    };

    std::unique_ptr<HoeffdingTreeClassifier> new_tree() const;

    double drift_delta_, split_confidence_;
    int    grace_period_, max_depth_;

    std::vector<Member>  members_;
    mutable std::mt19937 rng_;
};
