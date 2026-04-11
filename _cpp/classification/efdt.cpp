#include "efdt.h"

// EFDT uses a more aggressive (less conservative) split confidence than the
// standard Hoeffding Tree.  The default split_confidence is 1e-5 instead of
// 1e-7, which means the Hoeffding bound epsilon is larger → splits happen
// earlier (with less data) and are revised more readily when a better split
// is discovered.
//
// All tree-building, traversal, and prediction logic is inherited unchanged
// from HoeffdingTreeClassifier.  Only the constructor differs: it passes a
// higher split_confidence so the bound is less stringent.

EFDTClassifier::EFDTClassifier(
    int    grace_period,
    double split_confidence,
    double tie_threshold,
    int    nb_threshold,
    int    max_depth)
    : HoeffdingTreeClassifier(
          grace_period,
          split_confidence,   // 1e-5 by default — more aggressive
          tie_threshold,
          nb_threshold,
          max_depth,
          "info_gain")        // EFDT paper uses information gain
{}
