#pragma once
#include <unordered_map>
#include <vector>

#include "../core/running_stats.h"

// C++ replacements for the River metrics SAMLB uses:
//   classification: Accuracy, MacroF1, MacroPrecision, MacroRecall
//   regression:     MAE, RMSE, R2

class Accuracy {
public:
    void   update(int y_true, int y_pred);
    double get() const;
    void   reset();
private:
    long long correct_ = 0, total_ = 0;
};

// Shared multiclass confusion counts; the three macro metrics differ only in
// how they reduce per-class TP/FP/FN.
class MacroBase {
public:
    void update(int y_true, int y_pred);
    void reset();
protected:
    struct Cell { long long tp = 0, fp = 0, fn = 0; };
    std::unordered_map<int, Cell> cm_;
};

class MacroPrecision : public MacroBase {
public:
    double get() const;
};

class MacroRecall : public MacroBase {
public:
    double get() const;
};

class MacroF1 : public MacroBase {
public:
    double get() const;
};

class MAE {
public:
    void   update(double y_true, double y_pred);
    double get() const;
    void   reset();
private:
    RMean mean_;
};

class RMSE {
public:
    void   update(double y_true, double y_pred);
    double get() const;
    void   reset();
private:
    RMean mean_;
};

class R2 {
public:
    void   update(double y_true, double y_pred);
    double get() const;
    void   reset();
private:
    RVar   y_var_;
    double rss_ = 0.0;
};

// Accuracy / macro-F1 of a learner over a sliding window of predictions.
// Used by StreamingAutoGluon to weight each stacked learner's vote.
//
// Unlike the MOA original the class count is not known up front (a stream does
// not announce it), so the per-class counters grow as labels are observed.
class WindowMetric {
public:
    explicit WindowMetric(int window_size = 1000);

    void   update(int predicted, int actual);
    double accuracy() const;
    double macro_f1() const;
    double get(int metric) const;          // 0 = accuracy, 1 = macro F1
    int    size() const { return size_; }
    void   reset();

    static constexpr int METRIC_ACCURACY = 0;
    static constexpr int METRIC_F1 = 1;

private:
    void apply(int predicted, int actual, int delta);
    void ensure(int cls);

    int                    window_size_;
    std::vector<int>       predicted_, actual_;
    std::vector<long long> tp_, fp_, fn_;
    int                    next_ = 0;
    int                    size_ = 0;
    long long              correct_ = 0;
};

// Sliding-window MAE / RMSE — the regression counterpart of WindowMetric,
// used to weight the stacked regressors of StreamingAutoGluon.
class WindowRegressionMetric {
public:
    explicit WindowRegressionMetric(int window_size = 1000);

    void   update(double y_true, double y_pred);
    double mae() const;
    double rmse() const;
    double get(int metric) const;          // 0 = MAE, 1 = RMSE
    int    size() const { return size_; }
    void   reset();

    static constexpr int METRIC_MAE = 0;
    static constexpr int METRIC_RMSE = 1;

private:
    void recompute();                      // bounds float drift from add/subtract

    int                 window_size_;
    std::vector<double> abs_err_, sq_err_;
    double              sum_abs_ = 0.0, sum_sq_ = 0.0;
    int                 next_ = 0;
    int                 size_ = 0;
    long long           since_recompute_ = 0;
};
