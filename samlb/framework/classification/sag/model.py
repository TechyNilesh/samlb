"""
StreamingAutoGluon — online stacking of k-fold cross-validated stream learners.

Implementation of StreamingAutoGluon on the SAMLB C++ core, following the
original Java reference implementation (``StreamingAutoGluon.java``).

Algorithm
---------
For every base learner type ``t`` (``T`` of them), ``k + 1`` learners are built:

* ``k`` **fold learners**, trained with online k-fold cross-validation exactly
  as ``EvaluatePrequentialCV`` does — the learner whose index equals
  ``instances_seen % k`` is *not* trained on the current instance, the rest
  are. Fold learners always see the raw features.
* one **stacked learner**, trained on the instance augmented with the average
  class-probability prediction of all ``k`` fold learners of *every* type. The
  meta features stay out-of-fold because every fold learner predicts before any
  of them is trained on the current instance.

Prediction aggregates the fold learners the same way training does: all fold
learners predict, their per-type averages augment the instance, the augmented
instance goes to the stacked learners, and their votes are combined weighted by
a performance metric (accuracy or macro F1) over a sliding window.

Differences from the Java reference
-----------------------------------
* Base types are SAMLB C++ learners rather than ARF/ARTE/SORE (see config.py).
* The Java reference reads the class count from the ARFF header; a stream does
  not announce it, so classes are discovered online and the meta-feature block
  grows as new labels appear.
* No thread budget: the ``-j`` option has no analogue here, learners run in
  process.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from samlb.base import Pipeline
from samlb.framework.base._framework import BaseStreamFramework
from samlb.metrics import WindowMetric

from .config import SAG_BASE_LEARNERS, SAG_SCALERS

_METRICS = {"accuracy": WindowMetric.METRIC_ACCURACY, "f1": WindowMetric.METRIC_F1}


class StreamingAutoGluon(BaseStreamFramework):
    """Online stacking of k-fold cross-validated stream learners.

    Parameters
    ----------
    learners : list, optional
        Base learner prototypes, one per "ensemble type". Defaults to
        :data:`SAG_BASE_LEARNERS` (the shared pool).
    scalers : list, estimator or None
        Preprocessors, assigned round-robin across the base types and fused
        into each learner's pipeline. Defaults to :data:`SAG_SCALERS` (the
        shared preprocessor pool); pass a single estimator to use one for every
        type, or ``None`` for raw features, as the Java reference does.
    n_folds : int
        Number of cross-validation folds ``k`` (``-k`` in the Java reference, default 3).
    metric : {"accuracy", "f1"}
        Metric used to weight the stacked learners' votes (``-m`` in the Java reference).
    window_size : int
        Sliding window over which that metric is estimated (``-w`` in the Java reference).
    seed : int or None
        Kept for API symmetry; the method itself is deterministic.
    """

    exploration_window: int = 1000

    def __init__(
        self,
        learners: Optional[List] = None,
        scalers: Any = "default",
        n_folds: int = 3,
        metric: str = "accuracy",
        window_size: int = 1000,
        seed: Optional[int] = 42,
    ):
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {sorted(_METRICS)}, got {metric!r}.")

        self.learners = learners if learners is not None else SAG_BASE_LEARNERS
        # "default" (not None) as the sentinel so scalers=None means "no scaler".
        if isinstance(scalers, str) and scalers == "default":
            scalers = SAG_SCALERS
        if scalers is None:
            self.scalers: List = []
        elif isinstance(scalers, (list, tuple)):
            self.scalers = list(scalers)
        else:
            self.scalers = [scalers]
        self.n_folds = n_folds
        self.metric = metric
        self.window_size = window_size
        self.seed = seed

        if not self.learners:
            raise ValueError("StreamingAutoGluon requires at least one base learner type.")

        self.reset()

    # ── construction ─────────────────────────────────────────────────────────

    @staticmethod
    def _unique_names(learners) -> List[str]:
        """Distinct name per base type — meta-feature keys are built from these.

        A pool may hold several instances of one algorithm (the shared
        classifier_instances list does), and identical names would make their
        meta features overwrite each other.
        """
        names, counts = [], {}
        for learner in learners:
            base = type(learner).__name__
            counts[base] = counts.get(base, 0) + 1
            names.append(base if counts[base] == 1 else f"{base}{counts[base]}")
        return names

    def _build(self, prototype, type_index: int):
        """A fresh learner, optionally behind a scaler, fused into one C++ object.

        Scalers are handed out round-robin over the base types, so every
        preprocessor in the pool is used and the assignment stays deterministic.
        """
        learner = prototype.clone()
        if not self.scalers:
            return learner
        scaler = self.scalers[type_index % len(self.scalers)].clone()
        return Pipeline(scaler, learner)

    def reset(self) -> None:
        self._names = self._unique_names(self.learners)
        self._n_types = len(self.learners)

        # [type][fold] trained on raw features; [type] trained on augmented ones.
        self._fold_learners = [
            [self._build(p, t) for _ in range(self.n_folds)]
            for t, p in enumerate(self.learners)
        ]
        self._stacked_learners = [
            self._build(p, t) for t, p in enumerate(self.learners)
        ]
        self._stacked_metrics = [WindowMetric(self.window_size) for _ in self.learners]

        self._classes: List[int] = []
        self._seen = 0
        self._clear_cache()

    # ── meta features ────────────────────────────────────────────────────────

    def _clear_cache(self) -> None:
        self._cached_x = None
        self._cached_votes = None

    def _fold_votes(self, x: dict) -> List[List[Dict[int, float]]]:
        """Normalised votes of every fold learner.

        Cached on the instance so a test-then-train step only predicts once,
        as the Java reference does.
        """
        if self._cached_votes is not None and self._cached_x is x:
            return self._cached_votes

        votes = []
        for row in self._fold_learners:
            per_fold = []
            for learner in row:
                proba = learner.predict_proba_one(x)
                total = sum(proba.values())
                per_fold.append({c: v / total for c, v in proba.items()} if total > 0 else {})
            votes.append(per_fold)

        self._cached_x = x
        self._cached_votes = votes
        return votes

    def _aggregate(self, fold_votes) -> List[Dict[int, float]]:
        """Average, per type, the votes of that type's k fold learners."""
        averages = []
        for per_fold in fold_votes:
            acc: Dict[int, float] = {}
            for vote in per_fold:
                for c, v in vote.items():
                    acc[c] = acc.get(c, 0.0) + v
            averages.append({c: v / self.n_folds for c, v in acc.items()})
        return averages

    def _augment(self, x: dict, aggregated) -> dict:
        """Raw features plus one meta feature per (type, class)."""
        out = dict(x)
        for name, avg in zip(self._names, aggregated):
            for c in self._classes:
                out[f"P_{name}_{c}"] = avg.get(c, 0.0)
        return out

    def _observe(self, y: int) -> None:
        if y not in self._classes:
            self._classes.append(y)
            self._classes.sort()

    # ── BaseStreamFramework interface ────────────────────────────────────────

    def learn_one(self, x: dict, y: Any) -> None:
        self._observe(y)

        # The fold learner held out for this instance, as in EvaluatePrequentialCV.
        held_out = self._seen % self.n_folds

        # 1) Fold learners predict before being trained on the instance.
        fold_votes = self._fold_votes(x)

        # 2) Stacked learners are scored, then trained, on the augmented instance.
        #    None of the fold learners has seen this instance yet, so the meta
        #    features are out-of-fold.
        augmented = self._augment(x, self._aggregate(fold_votes))
        for t in range(self._n_types):
            vote = self._stacked_learners[t].predict_proba_one(augmented)
            if sum(vote.values()) > 0.0:
                self._stacked_metrics[t].update(max(vote, key=vote.get), y)
            self._stacked_learners[t].learn_one(augmented, y)

        # 3) Fold learners train on the raw instance, except the held-out one.
        for row in self._fold_learners:
            for i, learner in enumerate(row):
                if i != held_out:
                    learner.learn_one(x, y)

        self._seen += 1
        self._clear_cache()

    def predict_proba_one(self, x: dict) -> Dict[int, float]:
        augmented = self._augment(x, self._aggregate(self._fold_votes(x)))

        stacked = []
        combined: Dict[int, float] = {}
        total_weight = 0.0
        metric = _METRICS[self.metric]

        for t in range(self._n_types):
            vote = self._stacked_learners[t].predict_proba_one(augmented)
            total = sum(vote.values())
            stacked.append((vote, total))
            if total > 0.0:
                weight = self._stacked_metrics[t].get(metric)
                for c, v in vote.items():
                    combined[c] = combined.get(c, 0.0) + (v / total) * weight
                total_weight += weight

        if total_weight <= 0.0:
            # No metric estimate yet (or every stacked learner abstained): vote
            # uniformly, as the Java reference does.
            combined = {}
            for vote, total in stacked:
                if total > 0.0:
                    for c, v in vote.items():
                        combined[c] = combined.get(c, 0.0) + v / total
        return combined

    def predict_one(self, x: dict) -> Any:
        combined = self.predict_proba_one(x)
        if not combined:
            return self._classes[0] if self._classes else 0
        return max(combined, key=combined.get)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def stacked_weights(self) -> Dict[str, float]:
        """Current vote weight of each stacked learner."""
        metric = _METRICS[self.metric]
        return {
            name: m.get(metric) for name, m in zip(self._names, self._stacked_metrics)
        }

    def __repr__(self) -> str:
        return (
            f"StreamingAutoGluon(types={self._names}, n_folds={self.n_folds}, "
            f"metric={self.metric!r}, window_size={self.window_size}, "
            f"scalers={[type(s).__name__ for s in self.scalers]})"
        )
