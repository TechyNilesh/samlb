#pragma once
#include <unordered_map>
#include <string>
#include "../core/sliding_window.h"

class KNNRegressor {
public:
    explicit KNNRegressor(int n_neighbors = 5, int window_size = 1000, int p = 2);
    void   learn_one(const std::unordered_map<std::string, double>& x, double y);
    double predict_one(const std::unordered_map<std::string, double>& x) const;
    void   reset();

    int n_neighbors, p;

private:
    SlidingWindow<double> window;
};
