#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Common abstract interfaces so that heterogeneous components (scalers,
// feature selectors, classifiers, regressors) can be composed inside a fused
// C++ pipeline without crossing the Python boundary between stages.

using Features = std::unordered_map<std::string, double>;

class ITransformer {
public:
    virtual ~ITransformer() = default;

    // Unsupervised update (scalers). Supervised transformers override
    // learn_one_sup and ignore this.
    virtual void learn_one(const Features& x) = 0;

    // Supervised update (feature selection). Default forwards to learn_one.
    virtual void learn_one_sup(const Features& x, double y) { (void)y; learn_one(x); }

    // Allocating transform — used by the standalone Python API.
    virtual Features transform_one(const Features& x) const = 0;

    // In-place transform — used by the fused pipeline to avoid one map
    // allocation per stage per instance.
    virtual void transform_inplace(Features& x) const = 0;

    virtual bool is_supervised() const { return false; }

    // Feature order is lost when a Python dict becomes an unordered_map, but
    // some components (SelectKBest) break ties by it. The owner declares the
    // canonical order once; components that don't care ignore it.
    virtual void set_feature_order(const std::vector<std::string>& order) { (void)order; }

    virtual void reset() = 0;
};

class IClassifier {
public:
    virtual ~IClassifier() = default;
    virtual void learn_one(const Features& x, int y) = 0;
    virtual int  predict_one(const Features& x) const = 0;
    virtual std::unordered_map<int, double> predict_proba_one(const Features& x) const = 0;
    virtual void reset() = 0;
};

class IRegressor {
public:
    virtual ~IRegressor() = default;
    virtual void   learn_one(const Features& x, double y) = 0;
    virtual double predict_one(const Features& x) const = 0;
    virtual void   reset() = 0;
};
