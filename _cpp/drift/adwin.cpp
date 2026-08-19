#include "adwin.h"

#include <cmath>

ADWIN::ADWIN(double delta_, int clock_, int max_buckets_, int min_window_length_, int grace_period_)
    : delta(delta_),
      clock(clock_),
      max_buckets(max_buckets_),
      min_window_length(min_window_length_),
      grace_period(grace_period_) {
    rows_.emplace_back(max_buckets);
}

void ADWIN::reset() {
    rows_.clear();
    rows_.emplace_back(max_buckets);
    total_ = 0.0;
    variance_ = 0.0;
    width_ = 0.0;
    bucket_number_ = 0;
    time_ = 0;
    drift_detected_ = false;
}

void ADWIN::insert_element(double value) {
    width_ += 1.0;
    Row& head = rows_.front();
    head.total[head.size] = value;
    head.variance[head.size] = 0.0;
    head.size += 1;
    bucket_number_ += 1;

    if (width_ > 1.0) {
        const double d = value - total_ / (width_ - 1.0);
        variance_ += (width_ - 1.0) * d * d / width_;
    }
    total_ += value;
    compress_buckets();
}

void ADWIN::compress_buckets() {
    size_t i = 0;
    while (i < rows_.size()) {
        if (rows_[i].size != max_buckets + 1) break;

        // Grow first: emplace_back may reallocate, so no Row reference may be
        // held across this call.
        if (i + 1 == rows_.size()) rows_.emplace_back(max_buckets);
        Row& cursor = rows_[i];
        Row& next   = rows_[i + 1];

        // Merge the two oldest buckets of this row into one bucket of the next.
        const double n = bucket_size(static_cast<int>(i));
        const double u1 = cursor.total[0] / n;
        const double u2 = cursor.total[1] / n;
        const double inc_variance = n * n * (u1 - u2) * (u1 - u2) / (n + n);

        next.total[next.size] = cursor.total[0] + cursor.total[1];
        next.variance[next.size] = cursor.variance[0] + cursor.variance[1] + inc_variance;
        next.size += 1;
        bucket_number_ -= 1;

        // Shift the row down by two.
        for (int k = 2; k <= max_buckets + 1; ++k) {
            cursor.total[k - 2] = cursor.total[k];
            cursor.variance[k - 2] = cursor.variance[k];
        }
        for (int k = 0; k < 2; ++k) {
            cursor.total[max_buckets + 1 - k] = 0.0;
            cursor.variance[max_buckets + 1 - k] = 0.0;
        }
        cursor.size -= 2;

        if (next.size <= max_buckets) break;
        ++i;
    }
}

double ADWIN::delete_element() {
    Row& tail = rows_.back();
    const double n1 = bucket_size(static_cast<int>(rows_.size()) - 1);

    width_ -= n1;
    total_ -= tail.total[0];
    const double u1 = tail.total[0] / n1;
    const double d = u1 - (width_ > 0.0 ? total_ / width_ : 0.0);
    const double inc_variance = tail.variance[0] + n1 * width_ * d * d / (n1 + width_);
    variance_ -= inc_variance;

    for (int k = 1; k <= max_buckets + 1; ++k) {
        tail.total[k - 1] = tail.total[k];
        tail.variance[k - 1] = tail.variance[k];
    }
    tail.total[max_buckets + 1] = 0.0;
    tail.variance[max_buckets + 1] = 0.0;
    tail.size -= 1;
    bucket_number_ -= 1;

    if (tail.size == 0 && rows_.size() > 1) rows_.pop_back();
    return n1;
}

bool ADWIN::cut_expression(double n0, double n1, double diff) const {
    const double n = width_;
    if (n <= 1.0) return false;
    const double dd = std::log(2.0 * std::log(n) / delta);
    const double v = variance_ / n;
    const double m = 1.0 / (n0 - min_window_length + 1.0) + 1.0 / (n1 - min_window_length + 1.0);
    const double epsilon = std::sqrt(2.0 * m * v * dd) + (2.0 / 3.0) * dd * m;
    return std::fabs(diff) > epsilon;
}

void ADWIN::update(double value) {
    drift_detected_ = false;
    insert_element(value);
    time_ += 1;

    if (time_ % clock != 0 || width_ <= static_cast<double>(grace_period)) return;

    bool reduce_width = true;
    while (reduce_width) {
        reduce_width = false;
        bool exit_loop = false;

        double n0 = 0.0, n1 = width_;
        double u0 = 0.0, u1 = total_;

        for (int i = static_cast<int>(rows_.size()) - 1; i >= 0 && !exit_loop; --i) {
            const Row& cursor = rows_[static_cast<size_t>(i)];
            for (int k = 0; k <= cursor.size - 1; ++k) {
                if (i == 0 && k == cursor.size - 1) { exit_loop = true; break; }

                const double bs = bucket_size(i);
                n0 += bs;  n1 -= bs;
                u0 += cursor.total[k];  u1 -= cursor.total[k];

                if (n1 <= 0.0) { exit_loop = true; break; }

                const double diff = u0 / n0 - u1 / n1;
                if (n1 >= static_cast<double>(min_window_length) &&
                    n0 >= static_cast<double>(min_window_length) &&
                    cut_expression(n0, n1, diff)) {
                    reduce_width = true;
                    drift_detected_ = true;
                    if (width_ > 0.0) {
                        delete_element();
                        exit_loop = true;
                        break;
                    }
                }
            }
        }
    }
}
