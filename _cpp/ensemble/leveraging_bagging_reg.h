#pragma once
#include <memory>
#include <random>
#include <vector>

#include "../core/estimator.h"
#include "../drift/adwin.h"
#include "../regression/fimtdd.h"

// Leveraging Bagging for regression — the regression counterpart of
// LeveragingBaggingClassifier (Bifet, Holmes & Pfahringer, ICDM 2010).
//
// NOT a port: the published method is classification-only, and neither MOA
// (LeveragingBag implements MultiClassClassifier) nor River offers a
// regression version. Of the paper's two mechanisms, one carries over
// unchanged and one does not:
//
//   * **Leveraged resampling — carried over.** Online bagging draws each
//     member's instance weight from Poisson(1), which ignores about a third of
//     the stream per member; leveraging bagging raises lambda (default 6) so
//     members see far more of it, with more variance between them. Nothing
//     here refers to class labels.
//   * **Random output codes — dropped.** The paper's second source of
//     diversity assigns each member a random binary code over the *class
//     labels* and has it learn a relabelled problem. A continuous target has
//     no labels to code, and there is no accepted analogue, so this half is
//     simply absent. MOA's own implementation makes output codes optional
//     (-o), so dropping them stays inside the paper's own design space.
//   * **Drift — carried over.** Each member keeps its own ADWIN, fed its own
//     absolute error normalised by the ensemble's running target spread so the
//     detector's thresholds mean the same thing on any scale, and is reset
//     outright when that ADWIN fires. That is the classifier's behaviour, and
//     the normalisation is the one addition a continuous target forces.
//
// Aggregation is the unweighted mean of the members' predictions, mirroring
// the classifier's unweighted majority vote.
class LeveragingBaggingRegressor : public IRegressor {
public:
    explicit LeveragingBaggingRegressor(int n_models = 10,
                                        int seed = 0,
                                        double lambda_value = 6.0,
                                        double drift_delta = 0.002,
                                        int grace_period = 50,
                                        int max_depth = 20,
                                        double split_confidence = 0.01,
                                        std::string leaf_prediction = "adaptive");

    void   learn_one(const Features& x, double y) override;
    double predict_one(const Features& x) const override;
    void   reset() override;

    int    n_models;
    int    seed;
    double lambda_value;

private:
    struct Member {
        std::unique_ptr<FIMTDDRegressor> tree;
        std::unique_ptr<ADWIN>           drift;
    };

    std::unique_ptr<FIMTDDRegressor> new_tree(int index) const;

    double      drift_delta_, split_confidence_;
    int         grace_period_, max_depth_;
    std::string leaf_prediction_;

    // Running target statistics, used only to make the ADWIN input scale-free.
    double n_ = 0.0, sum_ = 0.0, sq_ = 0.0;

    std::vector<Member>  members_;
    mutable std::mt19937 rng_;
};
