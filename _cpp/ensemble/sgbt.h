#pragma once
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core/estimator.h"
#include "../regression/fimtdd.h"

// Streaming Gradient Boosted Trees — SGBT / SGBR.
//
// Classification: Gunasekara, Pfahringer, Gomes & Bifet, Machine Learning 2024.
// Regression:     Gunasekara, Pfahringer, Gomes & Bifet, DMKD 2025 — a separate
//                 method with its own defaults, not the classifier retargeted.
//                 Its base learner is a *bag* of trees (the paper's SGB(Oza)
//                 variant), it runs 10 boosting iterations rather than 100, and
//                 it uses a learning rate of 1.0 rather than 0.0125. That last
//                 one matters most: at 0.0125 the raw score reaches only
//                 1-(1-0.0125)^100 = 0.71 of the target, and a 29% shrinkage
//                 destroys R^2 on any target with a large mean.
//
// Gradient boosting for streams, using the weighted squared loss of XGBoost.
// Every instance is pushed through all M boosting iterations in one pass: at
// iteration m the gradient and hessian of the loss are taken at the raw score
// accumulated by iterations 0..m-1, the base regressor is trained on the
// pseudo-label g/h, and its output (times the learning rate) is added to the
// running raw score. Each iteration sees its own fixed random feature subset.
//
// Departures from the MOA reference, all forced by this codebase or by the
// stream interface, and all documented at the point they occur in sgbt.cpp:
//
//  * Base learner. The reference's is FIMTDD with mean leaves ("-e"), which
//    FIMTDDRegressor now provides natively, so this is no longer a
//    substitution. An earlier version used HoeffdingTreeRegressor and had to
//    fall back on its linear leaves, because that tree splits far too rarely
//    for mean leaves to carry any signal.
//  * Class count. MOA reads it from the ARFF schema. A stream does not announce
//    it, so classes are discovered online: pass n_classes=2 for the reference's
//    single-booster binary path, or leave it 0 to discover and run one-vs-all.
//  * Subspaces. MOA enumerates all k-combinations when there are <= 20
//    features and samples those without replacement; here each iteration draws
//    its own random k-subset directly.
//  * A floor on the hessian, so a saturated probability cannot make the
//    pseudo-label g/h infinite. MOA leaves this unguarded.
//  * The bagged base learner is Poisson(1) online bagging over bag_size trees
//    (Oza & Russell), which is what meta.OzaBag does, rather than MOA's OzaBag
//    object.
class SGBTBooster {
public:
    SGBTBooster(int n_iterations, double learning_rate, int percentage_of_features,
                int multiply_hessian_by, int skip_training, bool squared_loss,
                int bag_size, int grace_period, double split_confidence,
                int max_depth, std::string leaf_prediction, unsigned seed);

    // target is the ground truth of output 0: for classification the indicator
    // that the instance belongs to the *negative* class (MOA takes derivatives
    // at index 0, which is class 0), for regression the value of y.
    void   learn(const Features& x, double target);

    // Sum of the base learners' outputs. MOA does NOT scale this by the
    // learning rate even though training does — see scale_by_lr in sgbt.cpp.
    double raw_score(const Features& x, bool scale_by_lr) const;

    void   reset();
    bool   started() const { return initialised_; }

private:
    void     init(const Features& x);
    // Fills `out` rather than returning: one boosting iteration projects the
    // instance onto its own subspace, and at 100 iterations touched twice per
    // instance a fresh map each time is 200 allocations per instance.
    void project(const Features& x, const std::vector<std::string>& subspace,
                 Features& out) const;

    int         n_iterations_, percentage_of_features_, multiply_hessian_by_, skip_training_;
    int         bag_size_;
    double      learning_rate_;
    bool        squared_loss_;
    int         grace_period_, max_depth_;
    double      split_confidence_;
    std::string leaf_prediction_;
    unsigned    seed_;

    bool                                                  initialised_ = false;
    std::vector<std::vector<std::string>>                 subspaces_;
    // [iteration][bag member]; bag_size_ == 1 is a plain single-tree booster.
    std::vector<std::vector<std::unique_ptr<FIMTDDRegressor>>> trees_;
    mutable std::mt19937                                  rng_;
    // Scratch buffer for project(). Single-threaded by construction: one
    // booster is only ever driven by one thread at a time.
    mutable Features                                      scratch_;
};

class SGBTClassifier : public IClassifier {
public:
    explicit SGBTClassifier(int n_models = 100,
                            double learning_rate = 0.0125,
                            int percentage_of_features = 75,
                            int multiply_hessian_by = 1,
                            int skip_training = 1,
                            bool use_squared_loss = false,
                            int bag_size = 1,
                            int n_classes = 0,
                            bool scale_prediction_by_lr = false,
                            int grace_period = 25,
                            double split_confidence = 0.05,
                            int max_depth = 20,
                            std::string leaf_prediction = "mean",
                            int seed = 1);

    void                            learn_one(const Features& x, int y) override;
    int                             predict_one(const Features& x) const override;
    std::unordered_map<int, double> predict_proba_one(const Features& x) const override;
    void                            reset() override;

    int    n_models;
    double learning_rate;
    int    percentage_of_features;
    int    multiply_hessian_by;
    int    skip_training;
    bool   use_squared_loss;
    int    bag_size;
    int    n_classes;
    bool   scale_prediction_by_lr;
    int    seed;

private:
    std::unique_ptr<SGBTBooster> new_booster(size_t index) const;
    size_t                       class_slot(int y);          // inserts if unseen
    bool                         binary_mode() const { return n_classes == 2; }

    int         grace_period_, max_depth_;
    double      split_confidence_;
    std::string leaf_prediction_;

    std::vector<int>                          classes_;      // ascending
    std::vector<std::unique_ptr<SGBTBooster>> boosters_;      // parallel to classes_
};

class SGBRRegressor : public IRegressor {
public:
    explicit SGBRRegressor(int n_models = 10,
                           double learning_rate = 1.0,
                           int percentage_of_features = 75,
                           int multiply_hessian_by = 1,
                           int skip_training = 1,
                           int bag_size = 10,
                           int grace_period = 50,
                           double split_confidence = 0.01,
                           int max_depth = 20,
                           std::string leaf_prediction = "mean",
                           int seed = 1);

    void   learn_one(const Features& x, double y) override;
    double predict_one(const Features& x) const override;
    void   reset() override;

    int    n_models;
    double learning_rate;
    int    percentage_of_features;
    int    multiply_hessian_by;
    int    skip_training;
    int    bag_size;
    int    seed;

private:
    int         grace_period_, max_depth_;
    double      split_confidence_;
    std::string leaf_prediction_;

    std::unique_ptr<SGBTBooster> booster_;
};
