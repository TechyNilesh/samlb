#include "pipeline.h"

#include <utility>

// ── ClassificationPipeline ───────────────────────────────────────────────────

ClassificationPipeline::ClassificationPipeline(std::vector<std::shared_ptr<ITransformer>> steps,
                                               std::shared_ptr<IClassifier> learner)
    : steps_(std::move(steps)), learner_(std::move(learner)) {}

void ClassificationPipeline::apply(Features& x) const {
    for (const auto& t : steps_) t->transform_inplace(x);
}

void ClassificationPipeline::learn_one_owned(Features x, int y) {
    // river.compose.Pipeline order: each step learns on its input, then
    // transforms it for the next step.
    for (const auto& t : steps_) {
        if (t->is_supervised()) t->learn_one_sup(x, static_cast<double>(y));
        else                    t->learn_one(x);
        t->transform_inplace(x);
    }
    learner_->learn_one(x, y);
}

int ClassificationPipeline::predict_one_owned(Features x) const {
    apply(x);
    return learner_->predict_one(x);
}

std::unordered_map<int, double> ClassificationPipeline::predict_proba_one_owned(Features x) const {
    apply(x);
    return learner_->predict_proba_one(x);
}

void ClassificationPipeline::learn_one(const Features& x, int y) { learn_one_owned(x, y); }
int  ClassificationPipeline::predict_one(const Features& x) const { return predict_one_owned(x); }
std::unordered_map<int, double> ClassificationPipeline::predict_proba_one(const Features& x) const {
    return predict_proba_one_owned(x);
}

void ClassificationPipeline::set_feature_order(const std::vector<std::string>& order) {
    for (const auto& t : steps_) t->set_feature_order(order);
}

void ClassificationPipeline::reset() {
    for (const auto& t : steps_) t->reset();
    learner_->reset();
}

// ── RegressionPipeline ───────────────────────────────────────────────────────

RegressionPipeline::RegressionPipeline(std::vector<std::shared_ptr<ITransformer>> steps,
                                       std::shared_ptr<IRegressor> learner)
    : steps_(std::move(steps)), learner_(std::move(learner)) {}

void RegressionPipeline::apply(Features& x) const {
    for (const auto& t : steps_) t->transform_inplace(x);
}

void RegressionPipeline::learn_one_owned(Features x, double y) {
    for (const auto& t : steps_) {
        if (t->is_supervised()) t->learn_one_sup(x, y);
        else                    t->learn_one(x);
        t->transform_inplace(x);
    }
    learner_->learn_one(x, y);
}

double RegressionPipeline::predict_one_owned(Features x) const {
    apply(x);
    return learner_->predict_one(x);
}

void   RegressionPipeline::learn_one(const Features& x, double y) { learn_one_owned(x, y); }
double RegressionPipeline::predict_one(const Features& x) const { return predict_one_owned(x); }

void RegressionPipeline::set_feature_order(const std::vector<std::string>& order) {
    for (const auto& t : steps_) t->set_feature_order(order);
}

void RegressionPipeline::reset() {
    for (const auto& t : steps_) t->reset();
    learner_->reset();
}
