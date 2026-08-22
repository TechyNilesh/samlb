#include "hat_cls.h"

HoeffdingAdaptiveTreeClassifier::HoeffdingAdaptiveTreeClassifier(
    int grace_period_, double split_confidence_, double tie_threshold_,
    int nb_threshold_, int max_depth_, double drift_delta, double warning_delta,
    const std::string& split_criterion_)
    : grace_period(grace_period_),
      split_confidence(split_confidence_),
      tie_threshold(tie_threshold_),
      nb_threshold(nb_threshold_),
      max_depth(max_depth_),
      split_criterion(split_criterion_),
      drift_delta_(drift_delta),
      warning_delta_(warning_delta) {
    reset();
}

std::unique_ptr<HoeffdingTreeClassifier> HoeffdingAdaptiveTreeClassifier::new_tree() const {
    return std::unique_ptr<HoeffdingTreeClassifier>(new HoeffdingTreeClassifier(
        grace_period, split_confidence, tie_threshold, nb_threshold, max_depth,
        split_criterion));
}

void HoeffdingAdaptiveTreeClassifier::reset() {
    tree_ = new_tree();
    background_.reset();
    drift_.reset(new ADWIN(drift_delta_));
    warning_.reset(new ADWIN(warning_delta_));
}

void HoeffdingAdaptiveTreeClassifier::learn_one(const Features& x, int y) {
    // Error signal for the detectors, measured before this update.
    const bool correct = (tree_->predict_one(x) == y);
    const double err   = correct ? 0.0 : 1.0;

    tree_->learn_one(x, y);
    if (background_) background_->learn_one(x, y);

    warning_->update(err);
    if (warning_->drift_detected() && !background_) {
        background_ = new_tree();
        warning_.reset(new ADWIN(warning_delta_));
    }

    drift_->update(err);
    if (drift_->drift_detected()) {
        tree_ = background_ ? std::move(background_) : new_tree();
        background_.reset();
        drift_.reset(new ADWIN(drift_delta_));
        warning_.reset(new ADWIN(warning_delta_));
    }
}

int HoeffdingAdaptiveTreeClassifier::predict_one(const Features& x) const {
    return tree_->predict_one(x);
}

std::unordered_map<int, double> HoeffdingAdaptiveTreeClassifier::predict_proba_one(
        const Features& x) const {
    return tree_->predict_proba_one(x);
}
