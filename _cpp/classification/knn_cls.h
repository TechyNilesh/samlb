#pragma once
#include <unordered_map>
#include <string>
#include "../core/sliding_window.h"

class KNNClassifier {
public:
    explicit KNNClassifier(int n_neighbors = 5, int window_size = 1000, int p = 2);
    void learn_one(const std::unordered_map<std::string, double>& x, int y);
    int  predict_one(const std::unordered_map<std::string, double>& x) const;
    std::unordered_map<int, double> predict_proba_one(
        const std::unordered_map<std::string, double>& x) const;
    void reset();

    int n_neighbors, p;

private:
    SlidingWindow<int> window;
};
