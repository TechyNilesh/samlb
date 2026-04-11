#pragma once
#include <cmath>

// Hoeffding bound: the minimum number of samples needed so that the
// observed mean is within epsilon of the true mean with probability 1-delta.
//
// epsilon = sqrt( R^2 * ln(1/delta) / (2 * n) )
//
// R = range of the random variable (1.0 for normalised metrics).
inline double hoeffding_bound(double range, double confidence, double n) {
    if (n <= 0.0) return range;
    return std::sqrt((range * range * std::log(1.0 / confidence)) / (2.0 * n));
}
