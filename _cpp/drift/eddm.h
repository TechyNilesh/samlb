#pragma once
#include "../core/running_stats.h"

// EDDM (Early Drift Detection Method) — port of river.drift.binary.EDDM.
// update() takes 1 for a misclassification, 0 for a correct prediction.
class EDDM {
public:
    explicit EDDM(int warm_start = 30, double alpha = 0.95, double beta = 0.9);

    void update(int x);
    bool drift_detected() const { return drift_detected_; }
    bool warning_detected() const { return warning_detected_; }
    void reset();

    int    warm_start;
    double alpha;
    double beta;

private:
    RVar      error_distances_;
    long long n_ = 0;
    long long last_error_ = 0;
    long long n_errors_ = 0;
    double    p2s_prime_max_ = -1.0;
    bool      drift_detected_ = false;
    bool      warning_detected_ = false;
};
