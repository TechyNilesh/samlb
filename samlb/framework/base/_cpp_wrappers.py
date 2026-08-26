"""
samlb.framework.base._cpp_wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thin Python wrappers around the SAMLB C++ components.

Every wrapper holds a ``_cpp`` handle and stores its hyperparameters as
attributes named after its ``__init__`` arguments. That convention is all
:class:`samlb.base.Estimator` needs to provide ``_get_params`` / ``clone``,
and the ``_cpp`` handle is what lets :class:`samlb.base.Pipeline` fuse the
whole chain into a single C++ object.

There is deliberately no dict sanitising here: the C++ side takes the feature
mapping directly, and NaN handling lives in the dataset loader.
"""
from __future__ import annotations

import samlb._samlb_core as _core

from samlb.base import Estimator


# ── Classification ────────────────────────────────────────────────────────────

class _Classifier(Estimator):
    _kind = "classifier"

    def learn_one(self, x, y):
        self._cpp.learn_one(x, y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(x)

    def predict_proba_one(self, x):
        return self._cpp.predict_proba_one(x)

    def reset(self):
        self._cpp.reset()
        return self


class NaiveBayes(_Classifier):
    """Gaussian Naïve Bayes (C++ backend)."""

    def __init__(self):
        self._cpp = _core.NaiveBayes()


class Perceptron(_Classifier):
    """Multiclass Perceptron (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self._cpp = _core.Perceptron(learning_rate=learning_rate)


class LogisticRegression(_Classifier):
    """One-vs-Rest Logistic Regression with SGD (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.l2 = l2
        self._cpp = _core.LogisticRegressionClassifier(learning_rate=learning_rate, l2=l2)


class PassiveAggressiveClassifier(_Classifier):
    """Passive-Aggressive Classifier (C++ backend)."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self._cpp = _core.PassiveAggressiveClassifier(C=C)


class SoftmaxRegression(_Classifier):
    """Multinomial Softmax Regression (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.l2 = l2
        self._cpp = _core.SoftmaxRegression(learning_rate=learning_rate, l2=l2)


class KNNClassifier(_Classifier):
    """K-Nearest Neighbours Classifier with sliding window (C++ backend)."""

    def __init__(self, n_neighbors: int = 5, window_size: int = 1000, p: int = 2):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.p = p
        self._cpp = _core.KNNClassifier(n_neighbors=n_neighbors, window_size=window_size, p=p)


class HoeffdingTreeClassifier(_Classifier):
    """Very Fast Decision Tree — VFDT (C++ backend)."""

    def __init__(
        self,
        grace_period: int = 200,
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        nb_threshold: int = 0,
        max_depth: int = 20,
        split_criterion: str = "info_gain",
    ):
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.nb_threshold = nb_threshold
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        self._cpp = _core.HoeffdingTreeClassifier(
            grace_period=grace_period,
            split_confidence=split_confidence,
            tie_threshold=tie_threshold,
            nb_threshold=nb_threshold,
            max_depth=max_depth,
            split_criterion=split_criterion,
        )


class EFDTClassifier(_Classifier):
    """Extremely Fast Decision Tree (C++ backend)."""

    def __init__(
        self,
        grace_period: int = 200,
        split_confidence: float = 1e-5,
        tie_threshold: float = 0.05,
        nb_threshold: int = 0,
        max_depth: int = 20,
    ):
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.nb_threshold = nb_threshold
        self.max_depth = max_depth
        self._cpp = _core.EFDTClassifier(
            grace_period=grace_period,
            split_confidence=split_confidence,
            tie_threshold=tie_threshold,
            nb_threshold=nb_threshold,
            max_depth=max_depth,
        )


class SGTClassifier(_Classifier):
    """Stochastic Gradient Tree Classifier (C++ backend)."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        lambda_: float = 0.1,
        grace_period: int = 200,
        max_depth: int = 6,
    ):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.grace_period = grace_period
        self.max_depth = max_depth
        self._cpp = _core.SGTClassifier(
            learning_rate=learning_rate,
            lambda_=lambda_,
            grace_period=grace_period,
            max_depth=max_depth,
        )


class ARFClassifier(_Classifier):
    """Adaptive Random Forest (C++ backend).

    Gomes et al., Machine Learning 2017 — the standard streaming baseline.
    Online bagging over Hoeffding trees, each resampling a random feature
    subspace at every split, with a warning/drift detector pair per tree and
    accuracy-weighted voting.
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int = 0,
        lambda_value: float = 6.0,
        drift_delta: float = 0.001,
        warning_delta: float = 0.01,
        grace_period: int = 50,
        max_depth: int = 20,
        split_confidence: float = 0.01,
        subspace_size: int = -1,
    ):
        self.n_models = n_models
        self.seed = seed
        self.lambda_value = lambda_value
        self.drift_delta = drift_delta
        self.warning_delta = warning_delta
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.split_confidence = split_confidence
        self.subspace_size = subspace_size
        self._cpp = _core.ARFClassifier(
            n_models=n_models, seed=seed, lambda_value=lambda_value,
            drift_delta=drift_delta, warning_delta=warning_delta,
            grace_period=grace_period, max_depth=max_depth,
            split_confidence=split_confidence, subspace_size=subspace_size,
        )


class SRPClassifier(_Classifier):
    """Streaming Random Patches (C++ backend).

    Gomes et al., ECML PKDD 2019. The difference from :class:`ARFClassifier` is
    where the feature randomisation happens: ARF resamples a subspace at every
    split attempt inside the tree, while SRP draws one random feature subset per
    ensemble member and keeps it for that member's whole life, so the base
    learner is an unmodified tree fed a projected instance. Combined with
    Poisson resampling that gives a random *patch* — a subset of features by a
    subset of instances.

    Parameters
    ----------
    training_method : str
        ``"patches"`` (default) — subspace and resampling, the paper's RP.
        ``"subspaces"`` — subspace only; every member sees every instance once.
        ``"resampling"`` — resampling only, over all features.
    subspace_fraction : float
        Share of the features each member is given.
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int = 0,
        lambda_value: float = 6.0,
        drift_delta: float = 0.001,
        warning_delta: float = 0.01,
        grace_period: int = 50,
        max_depth: int = 20,
        split_confidence: float = 0.01,
        subspace_fraction: float = 0.6,
        training_method: str = "patches",
    ):
        if training_method not in ("patches", "subspaces", "resampling"):
            raise ValueError(
                "training_method must be 'patches', 'subspaces' or 'resampling', "
                f"got {training_method!r}."
            )
        self.n_models = n_models
        self.seed = seed
        self.lambda_value = lambda_value
        self.drift_delta = drift_delta
        self.warning_delta = warning_delta
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.split_confidence = split_confidence
        self.subspace_fraction = subspace_fraction
        self.training_method = training_method
        self._cpp = _core.SRPClassifier(
            n_models=n_models, seed=seed, lambda_value=lambda_value,
            drift_delta=drift_delta, warning_delta=warning_delta,
            grace_period=grace_period, max_depth=max_depth,
            split_confidence=split_confidence,
            subspace_fraction=subspace_fraction,
            training_method=training_method,
        )


class LeveragingBaggingClassifier(_Classifier):
    """Leveraging Bagging (C++ backend).

    Bifet, Holmes & Pfahringer, ICDM 2010. Diversity comes from a higher
    Poisson resampling weight than plain online bagging (default lambda=6,
    matching MOA's LeveragingBag) rather than feature subspacing. Each member
    carries its own ADWIN and is reset outright the moment its own detector
    fires drift — no background-tree promotion, unlike ARF/SRP. Voting is
    unweighted majority across members.
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int = 0,
        lambda_value: float = 6.0,
        drift_delta: float = 0.002,
        grace_period: int = 50,
        max_depth: int = 20,
        split_confidence: float = 0.01,
    ):
        self.n_models = n_models
        self.seed = seed
        self.lambda_value = lambda_value
        self.drift_delta = drift_delta
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.split_confidence = split_confidence
        self._cpp = _core.LeveragingBaggingClassifier(
            n_models=n_models, seed=seed, lambda_value=lambda_value,
            drift_delta=drift_delta, grace_period=grace_period,
            max_depth=max_depth, split_confidence=split_confidence,
        )


class HoeffdingAdaptiveTreeClassifier(_Classifier):
    """Hoeffding Adaptive Tree (C++ backend).

    Bifet & Gavaldà, SDM 2009. The reference implementation monitors every
    *node* with its own ADWIN and grows a per-node alternate subtree once that
    node's local error drifts. This backend applies the same warning/drift +
    background-tree mechanism ARFClassifier uses per ensemble member, but to
    a single whole tree rather than per node: one ADWIN pair tracks overall
    accuracy, a background tree trains in parallel once a warning fires, and
    it replaces the foreground tree wholesale on drift. Expect coarser,
    later reactions to *local* drift than MOA's/river's node-level
    HoeffdingAdaptiveTree, which resets only the affected subtree.
    """

    def __init__(
        self,
        grace_period: int = 200,
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        nb_threshold: int = 0,
        max_depth: int = 20,
        drift_delta: float = 0.002,
        warning_delta: float = 0.02,
        split_criterion: str = "info_gain",
    ):
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.nb_threshold = nb_threshold
        self.max_depth = max_depth
        self.drift_delta = drift_delta
        self.warning_delta = warning_delta
        self.split_criterion = split_criterion
        self._cpp = _core.HoeffdingAdaptiveTreeClassifier(
            grace_period=grace_period, split_confidence=split_confidence,
            tie_threshold=tie_threshold, nb_threshold=nb_threshold,
            max_depth=max_depth, drift_delta=drift_delta,
            warning_delta=warning_delta, split_criterion=split_criterion,
        )


# ── Regression ────────────────────────────────────────────────────────────────

class _Regressor(Estimator):
    _kind = "regressor"

    def learn_one(self, x, y):
        self._cpp.learn_one(x, y)
        return self

    def predict_one(self, x):
        return self._cpp.predict_one(x)

    def reset(self):
        self._cpp.reset()
        return self


class LinearRegression(_Regressor):
    """Linear Regression with SGD (C++ backend)."""

    def __init__(self, learning_rate: float = 0.01, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.l2 = l2
        self._cpp = _core.LinearRegression(learning_rate=learning_rate, l2=l2)


class BayesianLinearRegression(_Regressor):
    """Bayesian Linear Regression (C++ backend)."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self._cpp = _core.BayesianLinearRegression(alpha=alpha, beta=beta)


class PassiveAggressiveRegressor(_Regressor):
    """Passive-Aggressive Regressor (C++ backend)."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.1):
        self.C = C
        self.epsilon = epsilon
        self._cpp = _core.PassiveAggressiveRegressor(C=C, epsilon=epsilon)


class KNNRegressor(_Regressor):
    """K-Nearest Neighbours Regressor with sliding window (C++ backend)."""

    def __init__(self, n_neighbors: int = 5, window_size: int = 1000, p: int = 2):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.p = p
        self._cpp = _core.KNNRegressor(n_neighbors=n_neighbors, window_size=window_size, p=p)


class HoeffdingTreeRegressor(_Regressor):
    """Hoeffding Tree Regressor with SDR splitting (C++ backend)."""

    def __init__(
        self,
        grace_period: int = 200,
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        max_depth: int = 20,
        learning_rate: float = 0.01,
    ):
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._cpp = _core.HoeffdingTreeRegressor(
            grace_period=grace_period,
            split_confidence=split_confidence,
            tie_threshold=tie_threshold,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )


class ARFRegressor(_Regressor):
    """Adaptive Random Forest Regressor (C++ backend).

    Used by AutoClass as a surrogate fitness model.
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int = 0,
        lambda_value: float = 6.0,
        drift_delta: float = 0.001,
        warning_delta: float = 0.01,
        grace_period: int = 200,
        max_depth: int = 20,
        learning_rate: float = 0.01,
    ):
        self.n_models = n_models
        self.seed = seed
        self.lambda_value = lambda_value
        self.drift_delta = drift_delta
        self.warning_delta = warning_delta
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._cpp = _core.ARFRegressor(
            n_models=n_models,
            seed=seed,
            lambda_value=lambda_value,
            drift_delta=drift_delta,
            warning_delta=warning_delta,
            grace_period=grace_period,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )


class FIMTDDRegressor(_Regressor):
    """Fast Incremental Model Tree with Drift Detection (C++ backend).

    Ikonomovska, Gama & Džeroski, *DMKD* 2011 — the regression counterpart of
    :class:`HoeffdingAdaptiveTreeClassifier`, and the base learner both
    :class:`SGBTClassifier` and :class:`SGBRRegressor` are defined over.

    Three things separate it from :class:`HoeffdingTreeRegressor`:

    * **E-BST attribute observers.** Every distinct value a feature has taken
      in a leaf is kept with the prefix sums either side of it, so every
      observed value is an exact split candidate. `HoeffdingTreeRegressor`
      summarises each feature with a Gaussian and tries a few interpolated
      points, which is why it splits far too rarely — on a step function it
      cannot fit, it scores MAE 0.68 against a constant predictor's 1.0, where
      this tree scores 0.005.
    * **The ratio form of the Hoeffding test.** A split happens when
      ``SDR(second) / SDR(best) < 1 - epsilon``, compared across *attributes*.
      On variance reduction the merits scale with the target, so the ratio is
      the scale-free comparison.
    * **Page-Hinckley drift detection with alternate subtrees.** Each inner
      node watches its own normalised error and grows a replacement subtree in
      the background, promoting it once the faded-error ratio favours it.

    Parameters
    ----------
    leaf_prediction : {"adaptive", "mean", "perceptron"}
        ``"perceptron"`` is the reference's model tree and **diverges**: it
        normalises by running global feature statistics, so an instance seen
        while a feature's spread is still near zero overshoots the weights, and
        nothing bounds the output afterwards — on `ailerons` it predicts 53.6
        where the stream never exceeds 1.0, for an R² of -21. It is kept only
        so the defect stays reproducible.
        ``"adaptive"`` (default) bounds the perceptron to the leaf's own
        observed target range and uses it only while it beats the leaf mean,
        the same remedy applied to `HoeffdingTreeRegressor` in 0.2.0.
        ``"mean"`` is the reference's ``-e`` flag, and what SGBT/SGBR use.
    drift_detection : bool
        Page-Hinckley plus alternate subtrees. Turning it off leaves a plain
        incremental model tree.
    """

    def __init__(
        self,
        grace_period: int = 200,
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        max_depth: int = 20,
        leaf_prediction: str = "adaptive",
        learning_ratio: float = 0.02,
        learning_rate_decay: float = 0.001,
        learning_ratio_const: bool = False,
        page_hinckley_alpha: float = 0.005,
        page_hinckley_threshold: float = 50.0,
        alternate_tree_fading_factor: float = 0.995,
        alternate_tree_t_min: int = 150,
        alternate_tree_time: int = 1500,
        drift_detection: bool = True,
        seed: int = 1,
    ):
        if leaf_prediction not in ("adaptive", "mean", "perceptron"):
            raise ValueError(
                "leaf_prediction must be 'adaptive', 'mean' or 'perceptron', "
                f"got {leaf_prediction!r}."
            )
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.max_depth = max_depth
        self.leaf_prediction = leaf_prediction
        self.learning_ratio = learning_ratio
        self.learning_rate_decay = learning_rate_decay
        self.learning_ratio_const = learning_ratio_const
        self.page_hinckley_alpha = page_hinckley_alpha
        self.page_hinckley_threshold = page_hinckley_threshold
        self.alternate_tree_fading_factor = alternate_tree_fading_factor
        self.alternate_tree_t_min = alternate_tree_t_min
        self.alternate_tree_time = alternate_tree_time
        self.drift_detection = drift_detection
        self.seed = seed
        self._cpp = _core.FIMTDDRegressor(
            grace_period=grace_period, split_confidence=split_confidence,
            tie_threshold=tie_threshold, max_depth=max_depth,
            leaf_prediction=leaf_prediction, learning_ratio=learning_ratio,
            learning_rate_decay=learning_rate_decay,
            learning_ratio_const=learning_ratio_const,
            page_hinckley_alpha=page_hinckley_alpha,
            page_hinckley_threshold=page_hinckley_threshold,
            alternate_tree_fading_factor=alternate_tree_fading_factor,
            alternate_tree_t_min=alternate_tree_t_min,
            alternate_tree_time=alternate_tree_time,
            drift_detection=drift_detection, seed=seed,
        )

    def n_splits(self) -> int:
        """Splits performed so far — a cheap check that the tree is growing."""
        return self._cpp.n_splits()


class LeveragingBaggingRegressor(_Regressor):
    """Leveraging Bagging for regression (C++ backend).

    The regression counterpart of :class:`LeveragingBaggingClassifier`, which
    the original paper (Bifet, Holmes & Pfahringer, ICDM 2010) does not define
    — it is classification-only, and neither MOA nor River ships a regression
    version. Of its two mechanisms one carries over and one does not:

    * **Leveraged resampling — kept.** Members draw their instance weight from
      Poisson(``lambda_value``), default 6 rather than online bagging's 1, so
      each sees far more of the stream and members differ more. Nothing about
      it refers to class labels.
    * **Random output codes — dropped.** The paper's other diversity source
      assigns each member a random binary code over the *class labels*. A
      continuous target has none, and there is no accepted analogue. MOA makes
      output codes optional (``-o``), so running without them stays inside the
      paper's own design space.
    * **Drift — kept.** Each member keeps its own ADWIN and is reset outright
      when it fires, exactly as in the classifier. The one addition a
      continuous target forces is that ADWIN is fed the absolute error divided
      by the target's running spread; the classifier feeds it a 0/1 error,
      which is already scale-free, whereas a raw residual would tie the
      detector's sensitivity to the units of ``y``.

    Because it is an adaptation rather than a port, cite it as such — it is not
    the published Leveraging Bagging.
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int = 0,
        lambda_value: float = 6.0,
        drift_delta: float = 0.002,
        grace_period: int = 50,
        max_depth: int = 20,
        split_confidence: float = 0.01,
        leaf_prediction: str = "adaptive",
    ):
        if leaf_prediction not in ("adaptive", "mean", "perceptron"):
            raise ValueError(
                "leaf_prediction must be 'adaptive', 'mean' or 'perceptron', "
                f"got {leaf_prediction!r}."
            )
        self.n_models = n_models
        self.seed = seed
        self.lambda_value = lambda_value
        self.drift_delta = drift_delta
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.split_confidence = split_confidence
        self.leaf_prediction = leaf_prediction
        self._cpp = _core.LeveragingBaggingRegressor(
            n_models=n_models, seed=seed, lambda_value=lambda_value,
            drift_delta=drift_delta, grace_period=grace_period,
            max_depth=max_depth, split_confidence=split_confidence,
            leaf_prediction=leaf_prediction,
        )


class SGBTClassifier(_Classifier):
    """Streaming Gradient Boosted Trees (C++ backend).

    Gunasekara, Pfahringer, Gomes & Bifet, *Machine Learning* 2024. Gradient
    boosting for streams under XGBoost's weighted squared loss: every instance
    is pushed through all ``n_models`` boosting iterations in one pass, each
    iteration taking the gradient and hessian at the raw score accumulated so
    far, training its base regressor on the pseudo-label ``g / h``, and adding
    its (learning-rate-scaled) output to that score.

    The only boosting method in the pool — everything else here (ARF, SRP,
    Leveraging Bagging) is bagging-family.

    Parameters
    ----------
    n_models : int
        Boosting iterations ``M``. The reference default is 100; cost per
        instance scales with ``M`` times the number of one-vs-all boosters, so
        this is by far the most expensive learner in the pool.
    learning_rate : float
        Shrinkage applied to each iteration's contribution.
    percentage_of_features : int
        Size, as a percentage, of the fixed random feature subset each
        iteration is given.
    multiply_hessian_by : int
        When above 1, an iteration trains ``ceil(hessian * this)`` times
        instead of once — the reference's way of applying the hessian weight.
    skip_training : int
        Drop one instance in this many from training. ``1`` trains on all.
    use_squared_loss : bool
        Squared error instead of softmax cross-entropy.
    bag_size : int
        Trees per boosting iteration, bagged with Poisson(1) online bagging.
        ``1`` is the classification paper's configuration;
        :class:`SGBRRegressor` uses a bag.
    n_classes : int
        ``2`` selects the reference's single-booster binary path. ``0``
        (default) discovers the labels online and runs one booster per class,
        which is what a stream without a schema allows.
    scale_prediction_by_lr : bool
        The reference sums the base learners' outputs *unscaled* at prediction
        time while scaling them by the learning rate during training. Only
        ``sign(raw)`` decides a class, so this is immaterial here and defaults
        to ``False``, matching the reference.
    leaf_prediction : str
        Leaf model of the base tree. ``"mean"`` matches the reference's
        FIMT-DD flag ``-e`` (regression tree rather than model tree).

    Notes
    -----
    The base learner is :class:`FIMTDDRegressor`, the same tree the reference
    uses, drift detection included.
    """

    def __init__(
        self,
        n_models: int = 100,
        learning_rate: float = 0.0125,
        percentage_of_features: int = 75,
        multiply_hessian_by: int = 1,
        skip_training: int = 1,
        use_squared_loss: bool = False,
        bag_size: int = 1,
        n_classes: int = 0,
        scale_prediction_by_lr: bool = False,
        grace_period: int = 25,
        split_confidence: float = 0.05,
        max_depth: int = 20,
        leaf_prediction: str = "mean",
        seed: int = 1,
    ):
        if leaf_prediction not in ("perceptron", "mean"):
            raise ValueError(
                f"leaf_prediction must be 'perceptron' or 'mean', got {leaf_prediction!r}."
            )
        if n_classes not in (0, 2) and n_classes < 2:
            raise ValueError(f"n_classes must be 0 (discover) or >= 2, got {n_classes}.")
        self.n_models = n_models
        self.learning_rate = learning_rate
        self.percentage_of_features = percentage_of_features
        self.multiply_hessian_by = multiply_hessian_by
        self.skip_training = skip_training
        self.use_squared_loss = use_squared_loss
        self.bag_size = bag_size
        self.n_classes = n_classes
        self.scale_prediction_by_lr = scale_prediction_by_lr
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.max_depth = max_depth
        self.leaf_prediction = leaf_prediction
        self.seed = seed
        self._cpp = _core.SGBTClassifier(
            n_models=n_models, learning_rate=learning_rate,
            percentage_of_features=percentage_of_features,
            multiply_hessian_by=multiply_hessian_by,
            skip_training=skip_training, use_squared_loss=use_squared_loss,
            bag_size=bag_size, n_classes=n_classes,
            scale_prediction_by_lr=scale_prediction_by_lr,
            grace_period=grace_period, split_confidence=split_confidence,
            max_depth=max_depth, leaf_prediction=leaf_prediction, seed=seed,
        )


class SGBRRegressor(_Regressor):
    """Streaming Gradient Boosted Regression — SGBR (C++ backend).

    Gunasekara, Pfahringer, Gomes & Bifet, *Data Mining and Knowledge Discovery*
    2025. A **separate method** from :class:`SGBTClassifier`, not the classifier
    retargeted at a continuous label: the boosting machinery is shared, but the
    paper gives regression its own configuration, and the differences are not
    cosmetic.

    * **Bagged base learner.** Each boosting iteration holds ``bag_size`` trees
      trained with Poisson(1) online bagging, rather than a single tree — the
      paper's SGB(Oza) variant, which is the one it reports as beating the
      state of the art.
    * **Learning rate 1.0**, against the classifier's 0.0125. At 0.0125 the raw
      score reaches only ``1 - (1 - lr)^n_models`` of the target — 0.71 after
      100 iterations — and a 29% systematic shrinkage destroys R² on any target
      whose mean is large relative to its spread.
    * **10 boosting iterations**, against the classifier's 100, and a base tree
      with grace period 50 and split confidence 0.01.

    Squared error, so the gradient is the residual ``y - raw`` and the hessian
    is 1: each iteration fits what the ones before it left behind.

    Notes
    -----
    The base learner is :class:`FIMTDDRegressor`, the same tree the reference
    uses, drift detection included.
    """

    def __init__(
        self,
        n_models: int = 10,
        learning_rate: float = 1.0,
        percentage_of_features: int = 75,
        multiply_hessian_by: int = 1,
        skip_training: int = 1,
        bag_size: int = 10,
        grace_period: int = 50,
        split_confidence: float = 0.01,
        max_depth: int = 20,
        leaf_prediction: str = "mean",
        seed: int = 1,
    ):
        if leaf_prediction not in ("perceptron", "mean"):
            raise ValueError(
                f"leaf_prediction must be 'perceptron' or 'mean', got {leaf_prediction!r}."
            )
        self.n_models = n_models
        self.learning_rate = learning_rate
        self.percentage_of_features = percentage_of_features
        self.multiply_hessian_by = multiply_hessian_by
        self.skip_training = skip_training
        self.bag_size = bag_size
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.max_depth = max_depth
        self.leaf_prediction = leaf_prediction
        self.seed = seed
        self._cpp = _core.SGBRRegressor(
            n_models=n_models, learning_rate=learning_rate,
            percentage_of_features=percentage_of_features,
            multiply_hessian_by=multiply_hessian_by,
            skip_training=skip_training, bag_size=bag_size,
            grace_period=grace_period, split_confidence=split_confidence,
            max_depth=max_depth, leaf_prediction=leaf_prediction, seed=seed,
        )


class SRPRegressor(_Regressor):
    """Streaming Random Patches Regressor (C++ backend).

    Gomes et al., ECML PKDD 2019 — the regression counterpart of
    :class:`SRPClassifier`. Each ensemble member gets one fixed random feature
    subspace for its whole life (rather than ARF's per-split resampling),
    combined with Poisson resampling for a random *patch* of the input.

    Parameters
    ----------
    training_method : str
        ``"patches"`` (default) — subspace and resampling, the paper's RP.
        ``"subspaces"`` — subspace only; every member sees every instance once.
        ``"resampling"`` — resampling only, over all features.
    subspace_fraction : float
        Share of the features each member is given.
    """

    def __init__(
        self,
        n_models: int = 10,
        seed: int = 0,
        lambda_value: float = 6.0,
        drift_delta: float = 0.001,
        warning_delta: float = 0.01,
        grace_period: int = 200,
        max_depth: int = 20,
        learning_rate: float = 0.01,
        subspace_fraction: float = 0.6,
        training_method: str = "patches",
    ):
        if training_method not in ("patches", "subspaces", "resampling"):
            raise ValueError(
                "training_method must be 'patches', 'subspaces' or 'resampling', "
                f"got {training_method!r}."
            )
        self.n_models = n_models
        self.seed = seed
        self.lambda_value = lambda_value
        self.drift_delta = drift_delta
        self.warning_delta = warning_delta
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subspace_fraction = subspace_fraction
        self.training_method = training_method
        self._cpp = _core.SRPRegressor(
            n_models=n_models, seed=seed, lambda_value=lambda_value,
            drift_delta=drift_delta, warning_delta=warning_delta,
            grace_period=grace_period, max_depth=max_depth,
            learning_rate=learning_rate,
            subspace_fraction=subspace_fraction,
            training_method=training_method,
        )


# ── Preprocessing / feature selection ─────────────────────────────────────────

class _Transformer(Estimator):
    _kind = "transformer"
    _supervised = False

    def __deepcopy__(self, memo):
        # Unlike learners (which cannot be deep-copied at all and so clone
        # fresh), River's preprocessors carried their fitted statistics through
        # a deepcopy. EvoAutoML's mutation step relies on that, so the copy
        # keeps the running statistics.
        new = self.clone()
        new._cpp = self._cpp.clone_state()
        memo[id(self)] = new
        return new

    def learn_one(self, x):
        self._cpp.learn_one(x)
        return self

    def transform_one(self, x):
        return self._cpp.transform_one(x)

    def reset(self):
        self._cpp.reset()
        return self


class StandardScaler(_Transformer):
    """Zero-mean / unit-variance scaling (C++ backend)."""

    def __init__(self, with_std: bool = True):
        self.with_std = with_std
        self._cpp = _core.StandardScaler(with_std=with_std)


class MinMaxScaler(_Transformer):
    """Min-max scaling to [0, 1] (C++ backend)."""

    def __init__(self):
        self._cpp = _core.MinMaxScaler()


class MaxAbsScaler(_Transformer):
    """Absolute-max scaling to [-1, 1] (C++ backend)."""

    def __init__(self):
        self._cpp = _core.MaxAbsScaler()


class VarianceThreshold(_Transformer):
    """Drop features whose running variance stays below ``threshold``."""

    def __init__(self, threshold: float = 0.0, min_samples: int = 2):
        self.threshold = threshold
        self.min_samples = min_samples
        self._cpp = _core.VarianceThreshold(threshold=threshold, min_samples=min_samples)


class SelectKBest(_Transformer):
    """Keep the ``k`` features most correlated with the target (Pearson).

    Supervised: ``learn_one`` takes ``(x, y)``.
    """

    _supervised = True

    def __init__(self, k: int = 10, use_abs: bool = False):
        self.k = k
        self.use_abs = use_abs
        self._cpp = _core.SelectKBest(k=k, use_abs=use_abs)
        self._order_declared = False

    def __deepcopy__(self, memo):
        new = super().__deepcopy__(memo)
        new._order_declared = self._order_declared
        return new

    def learn_one(self, x, y):
        if not self._order_declared:
            # Tie-break order, see samlb.base.Pipeline._declare_order.
            self._cpp.set_feature_order(list(x.keys()))
            self._order_declared = True
        self._cpp.learn_one(x, float(y))
        return self
