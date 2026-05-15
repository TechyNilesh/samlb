"""
OAML — Online AutoML Classifier (GAMA-style reimplementation).

A self-contained reimplementation of OAML's drift-triggered pipeline
optimisation. The original OAML builds on GAMA's genetic-programming
pipeline search; here that search is reimplemented natively over the
SAMLB shared C++ learner pool, so no external AutoML dependency is needed.

Architecture
------------
Phase 1 — Warm-up / initial search
  Buffer the first ``initial_batch_size`` instances, then run a
  genetic-programming search over (scaler | classifier) pipelines.
  Deploy the best pipeline, retrained on the full warm-up buffer, and
  seed the backup ensemble with it.

Phase 2 — Online streaming
  For every new instance: predict with the deployed pipeline, learn,
  update the backup ensemble, and feed the binary error to an EDDM
  drift detector. A sliding window of the most recent instances is kept.

Phase 3 — Drift-triggered re-search
  When EDDM signals drift — or no drift has been seen for
  ``force_research_interval`` instances — re-run the GP search on the
  sliding window. The freshly searched pipeline is compared against the
  backup ensemble on the window; the better of the two is deployed. The
  new pipeline is appended to the backup ensemble (capped at
  ``backup_ensemble_size``) and the drift detector is reset.

Genetic-programming search
--------------------------
An individual is a (scaler | classifier) pipeline. Fitness is hold-out
accuracy on a train/eval split of the buffer. Each generation applies
tournament selection, component crossover, and mutation (scaler swap,
classifier swap, or hyperparameter perturbation) and keeps the best
``population_size`` individuals (elitism).
"""
from __future__ import annotations

import collections
import copy
from typing import Any, Deque, List, Optional, Tuple

import numpy as np
from river import metrics
from river.drift.binary import EDDM as _EDDM

from samlb.framework.base._framework import BaseStreamFramework
from .config import OAML_CLASSIFIERS, OAML_HYPERPARAMETERS, OAML_SCALERS

# A genome is an (untrained scaler prototype, untrained classifier prototype) pair.
Genome = Tuple[Any, Any]


# ── Lightweight pipeline ──────────────────────────────────────────────────────

class _Pipeline:
    """A (scaler | classifier) pipeline."""

    def __init__(self, scaler, classifier):
        self.scaler = copy.deepcopy(scaler)
        self.classifier = copy.deepcopy(classifier)

    def predict_one(self, x: dict) -> Any:
        x_t = self.scaler.transform_one(x)
        return self.classifier.predict_one(x_t)

    def predict_proba_one(self, x: dict) -> dict:
        x_t = self.scaler.transform_one(x)
        return self.classifier.predict_proba_one(x_t)

    def learn_one(self, x: dict, y: Any) -> "_Pipeline":
        self.scaler.learn_one(x)
        x_t = self.scaler.transform_one(x)
        self.classifier.learn_one(x_t, y)
        return self


# ── Backup ensemble (GAMA's model store analogue) ─────────────────────────────

class _BackupEnsemble:
    """Majority-vote ensemble of past best pipelines, capped at ``max_size``."""

    def __init__(self, members: List[_Pipeline], max_size: int):
        self.members: List[_Pipeline] = list(members)
        self.max_size = max_size

    def add(self, pipeline: _Pipeline) -> None:
        self.members.append(pipeline)
        while len(self.members) > self.max_size:
            self.members.pop(0)

    def predict_one(self, x: dict) -> Any:
        votes: collections.Counter = collections.Counter()
        for m in self.members:
            try:
                p = m.predict_one(x)
            except Exception:
                p = None
            if p is not None:
                votes[p] += 1
        if not votes:
            return None
        return votes.most_common(1)[0][0]

    def learn_one(self, x: dict, y: Any) -> "_BackupEnsemble":
        for m in self.members:
            try:
                m.learn_one(x, y)
            except Exception:
                pass
        return self


# ── Genetic-programming pipeline search ───────────────────────────────────────

class _GamaSearch:
    """Evolutionary search over (scaler | classifier) pipelines."""

    def __init__(
        self,
        scalers: list,
        classifiers: list,
        hyperparameters: dict,
        *,
        population_size: int,
        generations: int,
        train_split: float,
        rng: np.random.RandomState,
    ):
        self.scalers = scalers
        self.classifiers = classifiers
        self.hyperparameters = hyperparameters
        self.population_size = population_size
        self.generations = generations
        self.train_split = train_split
        self.rng = rng

    # ── genome operators ──────────────────────────────────────────────────────

    def _random_genome(self) -> Genome:
        scaler = self.scalers[self.rng.randint(len(self.scalers))]
        clf = self.classifiers[self.rng.randint(len(self.classifiers))]
        return (scaler, clf)

    def _crossover(self, g1: Genome, g2: Genome) -> Genome:
        """Child takes the scaler of one parent and the classifier of the other."""
        return (g1[0], g2[1])

    def _mutate(self, genome: Genome) -> Genome:
        scaler, clf = genome
        r = self.rng.rand()
        if r < 1 / 3:
            scaler = self.scalers[self.rng.randint(len(self.scalers))]
        elif r < 2 / 3:
            clf = self.classifiers[self.rng.randint(len(self.classifiers))]
        else:
            clf = self._perturb_hyperparams(clf)
        return (scaler, clf)

    def _perturb_hyperparams(self, clf):
        """Return a clone of *clf* with truncated-normal-perturbed hyperparameters."""
        name = type(clf).__name__
        space = self.hyperparameters.get(name, {})
        if not space:
            return clf
        cur = clf._get_params()
        mutated: dict = {}
        for k, vals in space.items():
            if k not in cur or not vals:
                continue
            v = cur[k]
            if isinstance(v, bool):
                mutated[k] = (not v) if self.rng.rand() < 0.5 else v
            elif isinstance(v, (int, float)):
                lo, hi = min(vals), max(vals)
                sd = max((hi - lo) / 6.0, 1e-9)
                new_v = np.clip(self.rng.normal(v, sd * 0.7), lo, hi)
                mutated[k] = int(round(new_v)) if isinstance(v, int) else float(new_v)
            else:
                mutated[k] = vals[self.rng.randint(len(vals))]
        return clf.clone(new_params=mutated) if mutated else clf

    # ── fitness & selection ───────────────────────────────────────────────────

    @staticmethod
    def _evaluate(genome: Genome, train_data: list, eval_data: list) -> float:
        try:
            pipe = _Pipeline(genome[0], genome[1])
            for x, y in train_data:
                pipe.learn_one(x, y)
            correct = sum(1 for x, y in eval_data if pipe.predict_one(x) == y)
            return correct / len(eval_data)
        except Exception:
            return 0.0

    def _tournament(self, population: List[Genome], fitness: List[float], k: int = 3) -> Genome:
        idxs = self.rng.randint(0, len(population), size=min(k, len(population)))
        best = idxs[0]
        for i in idxs[1:]:
            if fitness[i] > fitness[best]:
                best = i
        return population[best]

    # ── public API ────────────────────────────────────────────────────────────

    def search(self, data: list) -> Optional[Genome]:
        """Run the GP search on *data*; return the best genome found."""
        n = len(data)
        if n < 4:
            return None

        split = max(1, min(n - 1, int(n * self.train_split)))
        train_data, eval_data = data[:split], data[split:]

        population = [self._random_genome() for _ in range(self.population_size)]
        fitness = [self._evaluate(g, train_data, eval_data) for g in population]

        for _ in range(self.generations):
            offspring = []
            for _ in range(self.population_size):
                p1 = self._tournament(population, fitness)
                p2 = self._tournament(population, fitness)
                child = self._mutate(self._crossover(p1, p2))
                offspring.append(child)
            off_fitness = [self._evaluate(g, train_data, eval_data) for g in offspring]

            # Elitist replacement: keep the best population_size of parents + offspring.
            combined = sorted(
                zip(population + offspring, fitness + off_fitness),
                key=lambda t: t[1],
                reverse=True,
            )
            population = [g for g, _ in combined[: self.population_size]]
            fitness = [f for _, f in combined[: self.population_size]]

        return population[0]


# ── OAML Classifier ───────────────────────────────────────────────────────────

class OAMLClassifier(BaseStreamFramework):
    """Online AutoML Classifier with GAMA-style drift-triggered pipeline search.

    Parameters
    ----------
    initial_batch_size : int
        Number of instances buffered before the initial GP search.
    window_size : int
        Size of the sliding window kept for drift-triggered re-search.
    population_size : int
        Number of pipelines per GP generation.
    generations : int
        Number of GP generations per search round.
    train_split : float
        Fraction of the buffer used for training during fitness evaluation
        (the remainder is the hold-out used to score pipelines).
    force_research_interval : int
        Re-run the search if this many instances pass without a drift signal.
    min_research_gap : int
        Minimum number of instances between two consecutive searches.
    backup_ensemble_size : int
        Maximum number of past pipelines retained in the backup ensemble.
    seed : int
        Random seed for reproducibility.
    scalers, classifiers : list, optional
        Pipeline component pools. Default to the shared C++ classifier pool.
    hyperparameters : dict, optional
        Hyperparameter search spaces keyed by classifier class name.
    """

    def __init__(
        self,
        initial_batch_size: int = 200,
        window_size: int = 500,
        population_size: int = 10,
        generations: int = 3,
        train_split: float = 0.8,
        force_research_interval: int = 50_000,
        min_research_gap: int = 1_000,
        backup_ensemble_size: int = 10,
        seed: int = 42,
        scalers: Optional[list] = None,
        classifiers: Optional[list] = None,
        hyperparameters: Optional[dict] = None,
    ):
        self.initial_batch_size = initial_batch_size
        self.window_size = window_size
        self.population_size = population_size
        self.generations = generations
        self.train_split = train_split
        self.force_research_interval = force_research_interval
        self.min_research_gap = min_research_gap
        self.backup_ensemble_size = backup_ensemble_size
        self.seed = seed
        self.scalers = scalers if scalers is not None else OAML_SCALERS
        self.classifiers = classifiers if classifiers is not None else OAML_CLASSIFIERS
        self.hyperparameters = (
            hyperparameters if hyperparameters is not None else OAML_HYPERPARAMETERS
        )

        self._init_state()

    # ── internal state ────────────────────────────────────────────────────────

    def _init_state(self) -> None:
        self._rng = np.random.RandomState(self.seed)
        self._search = _GamaSearch(
            self.scalers,
            self.classifiers,
            self.hyperparameters,
            population_size=self.population_size,
            generations=self.generations,
            train_split=self.train_split,
            rng=self._rng,
        )
        self._warm_buffer: list = []
        self._sliding_window: Deque[Tuple] = collections.deque(maxlen=self.window_size)
        self._current: Optional[Any] = None
        self._backup: Optional[_BackupEnsemble] = None
        self._drift_detector = _EDDM()
        self._warmed_up: bool = False
        self._since_research: int = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _deploy_from_genome(self, genome: Genome, data: list) -> _Pipeline:
        """Build a fresh pipeline from *genome* and train it on all of *data*."""
        pipe = _Pipeline(genome[0], genome[1])
        for x, y in data:
            try:
                pipe.learn_one(x, y)
            except Exception:
                pass
        return pipe

    @staticmethod
    def _prequential_score(model, data: list) -> float:
        """Test-then-train *model* over *data*; return prequential accuracy."""
        metric = metrics.Accuracy()
        for x, y in data:
            try:
                y_pred = model.predict_one(x)
                if y_pred is not None:
                    metric.update(y, y_pred)
                model.learn_one(x, y)
            except Exception:
                pass
        return metric.get()

    def _research(self) -> None:
        """Drift-triggered re-search: re-optimise on the sliding window."""
        data = list(self._sliding_window)
        self._since_research = 0
        self._drift_detector = _EDDM()

        genome = self._search.search(data)
        if genome is None:
            return

        # Compare the backup ensemble against a fresh candidate on the window.
        candidate = _Pipeline(genome[0], genome[1])
        acc_candidate = self._prequential_score(candidate, data)
        acc_backup = self._prequential_score(self._backup, data)

        deployed = self._deploy_from_genome(genome, data)
        self._current = self._backup if acc_backup > acc_candidate else deployed
        self._backup.add(copy.deepcopy(deployed))

    # ── BaseStreamFramework interface ─────────────────────────────────────────

    def predict_one(self, x: dict) -> Any:
        """Predict with the current best pipeline (None during warm-up)."""
        if self._current is None:
            return None
        try:
            return self._current.predict_one(x)
        except Exception:
            return None

    def learn_one(self, x: dict, y: Any) -> None:
        self._sliding_window.append((x, y))

        if not self._warmed_up:
            # ── Warm-up phase ─────────────────────────────────────────────────
            self._warm_buffer.append((x, y))
            if len(self._warm_buffer) >= self.initial_batch_size:
                genome = self._search.search(self._warm_buffer)
                if genome is not None:
                    self._current = self._deploy_from_genome(genome, self._warm_buffer)
                    self._backup = _BackupEnsemble(
                        [copy.deepcopy(self._current)], self.backup_ensemble_size
                    )
                    self._warmed_up = True
                    self._warm_buffer = []
            return

        # ── Online phase ──────────────────────────────────────────────────────
        y_pred = self.predict_one(x)
        self._current.learn_one(x, y)
        if self._backup is not self._current:
            self._backup.learn_one(x, y)

        if y_pred is not None:
            self._drift_detector.update(0 if y_pred == y else 1)

        self._since_research += 1
        drift = self._drift_detector.drift_detected
        forced = self._since_research >= self.force_research_interval
        if (drift or forced) and self._since_research >= self.min_research_gap:
            self._research()

    def reset(self) -> None:
        """Reset to the initial (untrained) state."""
        self._init_state()
