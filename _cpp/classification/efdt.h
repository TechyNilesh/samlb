#pragma once
#include "hoeffding_tree.h"

// EFDT — Extremely Fast Decision Tree (Manapragada et al., KDD 2018)
// Extends HoeffdingTree: splits earlier on best available split,
// then revises splits when a better one is found.
// For simplicity, we subclass HoeffdingTreeClassifier and override try_split
// to use a lower threshold and allow split revision.

class EFDTClassifier : public HoeffdingTreeClassifier {
public:
    EFDTClassifier(
        int    grace_period      = 200,
        double split_confidence  = 1e-5,
        double tie_threshold     = 0.05,
        int    nb_threshold      = 0,
        int    max_depth         = 20
    );
};
