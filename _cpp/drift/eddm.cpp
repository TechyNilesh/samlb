#include "eddm.h"

#include <cmath>
#include <stdexcept>

EDDM::EDDM(int warm_start_, double alpha_, double beta_)
    : warm_start(warm_start_), alpha(alpha_), beta(beta_) {
    if (alpha < beta) throw std::invalid_argument("'alpha' must be greater or equal to 'beta'.");
}

void EDDM::reset() {
    error_distances_ = RVar{};
    n_ = 0;
    last_error_ = 0;
    n_errors_ = 0;
    p2s_prime_max_ = -1.0;
    drift_detected_ = false;
    warning_detected_ = false;
}

void EDDM::update(int x) {
    if (drift_detected_) reset();

    n_ += 1;

    if (x != 1) return;

    n_errors_ += 1;
    error_distances_.update(static_cast<double>(n_ - last_error_));

    if (n_ > warm_start) {
        const double pi_prime = error_distances_.mean.get();
        const double si_prime = std::sqrt(error_distances_.get());
        const double p2s_prime = pi_prime + 2.0 * si_prime;

        if (p2s_prime > p2s_prime_max_) {
            p2s_prime_max_ = p2s_prime;
        } else if (n_errors_ > warm_start) {
            const double level = p2s_prime / p2s_prime_max_;
            if (level < beta) {
                drift_detected_ = true;
            } else if (level < alpha) {
                warning_detected_ = true;
            } else {
                warning_detected_ = false;
            }
        }
    }

    last_error_ = n_;
}
