#include "knn_reg.h"
#include <algorithm>
#include <vector>
#include <numeric>

KNNRegressor::KNNRegressor(int k, int win, int p)
    : n_neighbors(k), p(p), window(win) {}

void KNNRegressor::learn_one(
    const std::unordered_map<std::string, double>& x, double y) {
    window.push(x, y);
}

double KNNRegressor::predict_one(
    const std::unordered_map<std::string, double>& x) const {
    if (window.empty()) return 0.0;

    // Convert query to dense vector
    auto& fidx = const_cast<SlidingWindow<double>&>(window).feature_index;
    int nf = const_cast<SlidingWindow<double>&>(window).n_features;
    std::vector<double> query(nf, 0.0);
    for (auto& [k, v] : x) {
        auto it = fidx.find(k);
        if (it != fidx.end()) query[it->second] = v;
    }

    std::vector<std::pair<double, double>> dists;
    dists.reserve(window.size());
    for (auto& [feat, label] : window.buffer)
        dists.emplace_back(SlidingWindow<double>::dense_distance(query, feat, p), label);

    int k = std::min(n_neighbors, static_cast<int>(dists.size()));
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

    // Inverse-distance weighted average
    double num = 0.0, den = 0.0;
    for (int i = 0; i < k; ++i) {
        double w = (dists[i].first > 1e-10) ? 1.0 / dists[i].first : 1e10;
        num += w * dists[i].second;
        den += w;
    }
    return (den > 0) ? num / den : 0.0;
}

void KNNRegressor::reset() {
    window.buffer.clear();
    window.feature_index.clear();
    window.n_features = 0;
}
