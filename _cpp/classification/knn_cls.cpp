#include "knn_cls.h"
#include <algorithm>
#include <vector>
#include <unordered_map>

KNNClassifier::KNNClassifier(int k, int win, int p)
    : n_neighbors(k), p(p), window(win) {}

void KNNClassifier::learn_one(
    const std::unordered_map<std::string, double>& x, int y) {
    window.push(x, y);
}

int KNNClassifier::predict_one(
    const std::unordered_map<std::string, double>& x) const {
    if (window.empty()) return -1;

    // Convert query to dense vector using the window's feature index
    auto& fidx = const_cast<SlidingWindow<int>&>(window).feature_index;
    int nf = const_cast<SlidingWindow<int>&>(window).n_features;
    std::vector<double> query(nf, 0.0);
    for (auto& [k, v] : x) {
        auto it = fidx.find(k);
        if (it != fidx.end()) query[it->second] = v;
    }

    // Build (distance, label) pairs using dense distance
    std::vector<std::pair<double, int>> dists;
    dists.reserve(window.size());
    for (auto& [feat, label] : window.buffer)
        dists.emplace_back(SlidingWindow<int>::dense_distance(query, feat, p), label);

    int k = std::min(n_neighbors, static_cast<int>(dists.size()));
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

    std::unordered_map<int, int> votes;
    for (int i = 0; i < k; ++i) votes[dists[i].second]++;

    int best = -1; int best_count = -1;
    for (auto& [cls, cnt] : votes)
        if (cnt > best_count) { best_count = cnt; best = cls; }
    return best;
}

std::unordered_map<int, double> KNNClassifier::predict_proba_one(
    const std::unordered_map<std::string, double>& x) const {
    std::unordered_map<int, double> proba;
    if (window.empty()) return proba;

    auto& fidx = const_cast<SlidingWindow<int>&>(window).feature_index;
    int nf = const_cast<SlidingWindow<int>&>(window).n_features;
    std::vector<double> query(nf, 0.0);
    for (auto& [k, v] : x) {
        auto it = fidx.find(k);
        if (it != fidx.end()) query[it->second] = v;
    }

    std::vector<std::pair<double, int>> dists;
    dists.reserve(window.size());
    for (auto& [feat, label] : window.buffer)
        dists.emplace_back(SlidingWindow<int>::dense_distance(query, feat, p), label);

    int k = std::min(n_neighbors, static_cast<int>(dists.size()));
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

    for (int i = 0; i < k; ++i) proba[dists[i].second] += 1.0 / k;
    return proba;
}

void KNNClassifier::reset() {
    window.buffer.clear();
    window.feature_index.clear();
    window.n_features = 0;
}
