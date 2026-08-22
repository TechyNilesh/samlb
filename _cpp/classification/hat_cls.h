#pragma once
#include <memory>

#include "../core/estimator.h"
#include "../drift/adwin.h"
#include "hoeffding_tree.h"

// Hoeffding Adaptive Tree (Bifet & Gavaldà, SDM 2009).
//
// The paper's HAT monitors every *node* with its own ADWIN and grows a
// per-node alternate subtree the moment that node's local error starts
// drifting, promoting the alternate once it is demonstrably better. That is
// the same idea ARFClassifier/SRPClassifier apply per ensemble member, here
// collapsed to a single tree: one ADWIN pair (warning + drift) tracks the
// whole tree's accuracy, a background tree trains in parallel once a warning
// fires, and it is promoted wholesale on drift. This whole-tree
// simplification trades node-local granularity for a implementation that
// reuses HoeffdingTreeClassifier unchanged; expect it to react to drift more
// coarsely (and later) than MOA's/river's node-level HoeffdingAdaptiveTree,
// which resets only the affected subtree.
class HoeffdingAdaptiveTreeClassifier : public IClassifier {
public:
    explicit HoeffdingAdaptiveTreeClassifier(int grace_period = 200,
                                             double split_confidence = 1e-7,
                                             double tie_threshold = 0.05,
                                             int nb_threshold = 0,
                                             int max_depth = 20,
                                             double drift_delta = 0.002,
                                             double warning_delta = 0.02,
                                             const std::string& split_criterion = "info_gain");

    void                            learn_one(const Features& x, int y) override;
    int                             predict_one(const Features& x) const override;
    std::unordered_map<int, double> predict_proba_one(const Features& x) const override;
    void                            reset() override;

    int    grace_period;
    double split_confidence;
    double tie_threshold;
    int    nb_threshold;
    int    max_depth;
    std::string split_criterion;

private:
    std::unique_ptr<HoeffdingTreeClassifier> new_tree() const;

    double drift_delta_, warning_delta_;

    std::unique_ptr<HoeffdingTreeClassifier> tree_;
    std::unique_ptr<HoeffdingTreeClassifier> background_;
    std::unique_ptr<ADWIN>                   drift_;
    std::unique_ptr<ADWIN>                   warning_;
};
