#include "metrics.h"

#include <algorithm>
#include <cmath>

// ── Accuracy ─────────────────────────────────────────────────────────────────

void Accuracy::update(int y_true, int y_pred) {
    total_ += 1;
    if (y_true == y_pred) correct_ += 1;
}
double Accuracy::get() const { return total_ ? static_cast<double>(correct_) / total_ : 0.0; }
void   Accuracy::reset() { correct_ = 0; total_ = 0; }

// ── Macro-averaged precision / recall / F1 ───────────────────────────────────

void MacroBase::update(int y_true, int y_pred) {
    // Touch both labels so a class that only ever appears as a true label (or
    // only as a prediction) still contributes a 0 term to the macro average,
    // matching River's behaviour of averaging over every observed class.
    cm_[y_true];
    cm_[y_pred];
    if (y_true == y_pred) {
        cm_[y_true].tp += 1;
    } else {
        cm_[y_pred].fp += 1;
        cm_[y_true].fn += 1;
    }
}

void MacroBase::reset() { cm_.clear(); }

double MacroPrecision::get() const {
    if (cm_.empty()) return 0.0;
    double acc = 0.0;
    for (const auto& kv : cm_) {
        const double denom = static_cast<double>(kv.second.tp + kv.second.fp);
        acc += denom > 0.0 ? kv.second.tp / denom : 0.0;
    }
    return acc / static_cast<double>(cm_.size());
}

double MacroRecall::get() const {
    if (cm_.empty()) return 0.0;
    double acc = 0.0;
    for (const auto& kv : cm_) {
        const double denom = static_cast<double>(kv.second.tp + kv.second.fn);
        acc += denom > 0.0 ? kv.second.tp / denom : 0.0;
    }
    return acc / static_cast<double>(cm_.size());
}

double MacroF1::get() const {
    if (cm_.empty()) return 0.0;
    double acc = 0.0;
    for (const auto& kv : cm_) {
        const double denom = static_cast<double>(2 * kv.second.tp + kv.second.fp + kv.second.fn);
        acc += denom > 0.0 ? (2.0 * kv.second.tp) / denom : 0.0;
    }
    return acc / static_cast<double>(cm_.size());
}

// ── Regression ───────────────────────────────────────────────────────────────

void   MAE::update(double y_true, double y_pred) { mean_.update(std::fabs(y_true - y_pred)); }
double MAE::get() const { return mean_.get(); }
void   MAE::reset() { mean_ = RMean{}; }

void RMSE::update(double y_true, double y_pred) {
    const double d = y_true - y_pred;
    mean_.update(d * d);
}
double RMSE::get() const { return std::sqrt(mean_.get()); }
void   RMSE::reset() { mean_ = RMean{}; }

void R2::update(double y_true, double y_pred) {
    y_var_.update(y_true);
    const double d = y_true - y_pred;
    rss_ += d * d;
}
double R2::get() const {
    if (y_var_.n() > 1.0) {
        const double tss = (y_var_.n() - 1.0) * y_var_.get();
        if (tss != 0.0) return 1.0 - (rss_ / tss);
    }
    return 0.0;
}
void R2::reset() { y_var_ = RVar{}; rss_ = 0.0; }

// ── WindowMetric ─────────────────────────────────────────────────────────────

WindowMetric::WindowMetric(int window_size)
    : window_size_(window_size < 0 ? 0 : window_size),
      predicted_(static_cast<size_t>(window_size_), 0),
      actual_(static_cast<size_t>(window_size_), 0) {}

void WindowMetric::ensure(int cls) {
    if (cls < 0) return;
    if (static_cast<size_t>(cls) >= tp_.size()) {
        tp_.resize(static_cast<size_t>(cls) + 1, 0);
        fp_.resize(static_cast<size_t>(cls) + 1, 0);
        fn_.resize(static_cast<size_t>(cls) + 1, 0);
    }
}

void WindowMetric::apply(int predicted, int actual, int delta) {
    if (predicted == actual) {
        if (actual >= 0 && static_cast<size_t>(actual) < tp_.size()) tp_[actual] += delta;
        correct_ += delta;
    } else {
        if (predicted >= 0 && static_cast<size_t>(predicted) < fp_.size()) fp_[predicted] += delta;
        if (actual >= 0 && static_cast<size_t>(actual) < fn_.size()) fn_[actual] += delta;
    }
}

void WindowMetric::update(int predicted, int actual) {
    if (window_size_ == 0) return;
    ensure(predicted);
    ensure(actual);

    if (size_ == window_size_) {
        // Evict the oldest observation before recording the new one.
        apply(predicted_[next_], actual_[next_], -1);
    } else {
        size_ += 1;
    }
    predicted_[next_] = predicted;
    actual_[next_] = actual;
    apply(predicted, actual, 1);
    next_ = (next_ + 1) % window_size_;
}

double WindowMetric::accuracy() const {
    return size_ == 0 ? 0.0 : static_cast<double>(correct_) / size_;
}

double WindowMetric::macro_f1() const {
    double sum = 0.0;
    int observed = 0;
    for (size_t c = 0; c < tp_.size(); ++c) {
        const long long tp = tp_[c], fp = fp_[c], fn = fn_[c];
        if (tp + fp + fn == 0) continue;          // class absent from the window
        observed += 1;
        const double denom = 2.0 * tp + fp + fn;
        if (denom > 0.0) sum += (2.0 * tp) / denom;
    }
    return observed == 0 ? 0.0 : sum / observed;
}

double WindowMetric::get(int metric) const {
    return metric == METRIC_F1 ? macro_f1() : accuracy();
}

void WindowMetric::reset() {
    std::fill(predicted_.begin(), predicted_.end(), 0);
    std::fill(actual_.begin(), actual_.end(), 0);
    tp_.clear(); fp_.clear(); fn_.clear();
    next_ = 0;
    size_ = 0;
    correct_ = 0;
}

// ── WindowRegressionMetric ───────────────────────────────────────────────────

WindowRegressionMetric::WindowRegressionMetric(int window_size)
    : window_size_(window_size < 0 ? 0 : window_size),
      abs_err_(static_cast<size_t>(window_size_), 0.0),
      sq_err_(static_cast<size_t>(window_size_), 0.0) {}

void WindowRegressionMetric::recompute() {
    sum_abs_ = 0.0;
    sum_sq_ = 0.0;
    for (int i = 0; i < size_; ++i) {
        sum_abs_ += abs_err_[i];
        sum_sq_ += sq_err_[i];
    }
    since_recompute_ = 0;
}

void WindowRegressionMetric::update(double y_true, double y_pred) {
    if (window_size_ == 0) return;
    const double d = y_true - y_pred;
    const double a = std::fabs(d);
    const double q = d * d;

    if (size_ == window_size_) {
        sum_abs_ -= abs_err_[next_];
        sum_sq_ -= sq_err_[next_];
    } else {
        size_ += 1;
    }
    abs_err_[next_] = a;
    sq_err_[next_] = q;
    sum_abs_ += a;
    sum_sq_ += q;
    next_ = (next_ + 1) % window_size_;

    // Running add/subtract loses precision over millions of updates; refresh
    // the sums exactly once per full window.
    if (++since_recompute_ >= window_size_) recompute();
}

double WindowRegressionMetric::mae() const {
    return size_ == 0 ? 0.0 : sum_abs_ / size_;
}

double WindowRegressionMetric::rmse() const {
    return size_ == 0 ? 0.0 : std::sqrt(sum_sq_ / size_);
}

double WindowRegressionMetric::get(int metric) const {
    return metric == METRIC_RMSE ? rmse() : mae();
}

void WindowRegressionMetric::reset() {
    std::fill(abs_err_.begin(), abs_err_.end(), 0.0);
    std::fill(sq_err_.begin(), sq_err_.end(), 0.0);
    sum_abs_ = 0.0;
    sum_sq_ = 0.0;
    next_ = 0;
    size_ = 0;
    since_recompute_ = 0;
}
