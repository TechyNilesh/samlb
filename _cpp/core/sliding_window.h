#pragma once
#include <cmath>
#include <deque>
#include <vector>
#include <unordered_map>
#include <string>

// Fixed-capacity sliding window of (feature_map, label) pairs.
// Used by KNN and other instance-based algorithms.
//
// Internally converts feature dicts to dense vectors on first push,
// mapping feature names → contiguous indices for fast distance computation.
template <typename Label>
struct SlidingWindow {
    using Features = std::unordered_map<std::string, double>;
    using DenseEntry = std::pair<std::vector<double>, Label>;

    std::deque<DenseEntry> buffer;
    std::size_t max_size;

    // Feature name → dense index mapping (built lazily on first push)
    std::unordered_map<std::string, int> feature_index;
    int n_features = 0;

    explicit SlidingWindow(std::size_t max_size) : max_size(max_size) {}

    // Convert sparse feature dict to dense vector
    std::vector<double> to_dense(const Features& x) {
        // Register any new features
        for (auto& [k, v] : x) {
            if (feature_index.find(k) == feature_index.end()) {
                feature_index[k] = n_features++;
                // Extend all existing entries
                for (auto& [vec, lbl] : buffer)
                    vec.resize(n_features, 0.0);
            }
        }
        std::vector<double> vec(n_features, 0.0);
        for (auto& [k, v] : x) {
            vec[feature_index[k]] = v;
        }
        return vec;
    }

    void push(const Features& x, const Label& y) {
        if (buffer.size() >= max_size)
            buffer.pop_front();
        buffer.emplace_back(to_dense(x), y);
    }

    std::size_t size() const { return buffer.size(); }
    bool empty()       const { return buffer.empty(); }

    // Dense distance between a query vector and a stored vector
    static double dense_distance(const std::vector<double>& a,
                                 const std::vector<double>& b, int p = 2) {
        double dist = 0.0;
        std::size_t n = std::min(a.size(), b.size());
        if (p == 2) {
            for (std::size_t i = 0; i < n; ++i) {
                double diff = a[i] - b[i];
                dist += diff * diff;
            }
            return std::sqrt(dist);
        } else if (p == 1) {
            for (std::size_t i = 0; i < n; ++i)
                dist += std::abs(a[i] - b[i]);
            return dist;
        } else {
            for (std::size_t i = 0; i < n; ++i)
                dist += std::pow(std::abs(a[i] - b[i]), p);
            return std::pow(dist, 1.0 / p);
        }
    }

    // Legacy sparse distance (kept for backward compat)
    static double distance(const Features& a, const Features& b, int p = 2) {
        double dist = 0.0;
        for (auto& [k, v] : a) {
            auto it = b.find(k);
            double diff = v - (it != b.end() ? it->second : 0.0);
            if (p == 1)      dist += std::abs(diff);
            else if (p == 2) dist += diff * diff;
            else             dist += std::pow(std::abs(diff), p);
        }
        return (p == 2) ? std::sqrt(dist) : std::pow(dist, 1.0 / p);
    }
};
