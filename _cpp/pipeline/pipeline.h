#pragma once
#include <memory>
#include <vector>

#include "../core/estimator.h"

// Fused pipelines: (transformer, ...) -> learner, executed entirely in C++.
//
// The whole point is marshalling. A Python-side pipeline converts the feature
// dict into a C++ map once per stage (scaler.learn, scaler.transform,
// learner.learn = 3 conversions plus 2 intermediate dict allocations). These
// classes take the features *by value* — pybind11 builds the map once — and
// then mutate it in place through every stage.

class ClassificationPipeline : public IClassifier {
public:
    ClassificationPipeline(std::vector<std::shared_ptr<ITransformer>> steps,
                           std::shared_ptr<IClassifier> learner);

    // Fast entry points bound to Python: features arrive by value.
    void                            learn_one_owned(Features x, int y);
    int                             predict_one_owned(Features x) const;
    std::unordered_map<int, double> predict_proba_one_owned(Features x) const;

    // IClassifier — used when a pipeline is nested inside another component.
    void                            learn_one(const Features& x, int y) override;
    int                             predict_one(const Features& x) const override;
    std::unordered_map<int, double> predict_proba_one(const Features& x) const override;
    void                            reset() override;

    // Forwarded to every step (see ITransformer::set_feature_order).
    void   set_feature_order(const std::vector<std::string>& order);
    size_t n_steps() const { return steps_.size(); }

private:
    void apply(Features& x) const;

    std::vector<std::shared_ptr<ITransformer>> steps_;
    std::shared_ptr<IClassifier>               learner_;
};

class RegressionPipeline : public IRegressor {
public:
    RegressionPipeline(std::vector<std::shared_ptr<ITransformer>> steps,
                       std::shared_ptr<IRegressor> learner);

    void   learn_one_owned(Features x, double y);
    double predict_one_owned(Features x) const;

    void   learn_one(const Features& x, double y) override;
    double predict_one(const Features& x) const override;
    void   reset() override;

    void   set_feature_order(const std::vector<std::string>& order);
    size_t n_steps() const { return steps_.size(); }

private:
    void apply(Features& x) const;

    std::vector<std::shared_ptr<ITransformer>> steps_;
    std::shared_ptr<IRegressor>                learner_;
};
