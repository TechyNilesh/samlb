#pragma once
#include <vector>

// ADWIN2 (Bifet & Gavaldà) — port of the MOA/river adaptive-windowing detector.
// Defaults match river.drift.ADWIN so drift behaviour is unchanged.
class ADWIN {
public:
    explicit ADWIN(double delta = 0.002,
                   int clock = 32,
                   int max_buckets = 5,
                   int min_window_length = 5,
                   int grace_period = 10);

    void   update(double value);
    bool   drift_detected() const { return drift_detected_; }
    double estimation() const { return width_ > 0.0 ? total_ / width_ : 0.0; }
    double width() const { return width_; }
    double variance() const { return width_ > 0.0 ? variance_ / width_ : 0.0; }
    void   reset();

    double delta;
    int    clock;
    int    max_buckets;
    int    min_window_length;
    int    grace_period;

private:
    // One row per bucket capacity: row i holds buckets of 2^i observations.
    // Within a row, index 0 is the oldest bucket; rows further from index 0
    // hold older data.
    struct Row {
        int                 size = 0;
        std::vector<double> total;
        std::vector<double> variance;
        explicit Row(int max_buckets)
            : total(static_cast<size_t>(max_buckets) + 2, 0.0),
              variance(static_cast<size_t>(max_buckets) + 2, 0.0) {}
    };

    void   insert_element(double value);
    void   compress_buckets();
    double delete_element();
    bool   cut_expression(double n0, double n1, double diff) const;
    static double bucket_size(int row) { return static_cast<double>(1ULL << row); }

    std::vector<Row> rows_;
    double    total_ = 0.0;
    double    variance_ = 0.0;
    double    width_ = 0.0;
    long long bucket_number_ = 0;
    long long time_ = 0;
    bool      drift_detected_ = false;
};
