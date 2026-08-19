#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Core interfaces
#include "../core/estimator.h"

// Classification
#include "../classification/naive_bayes.h"
#include "../classification/perceptron.h"
#include "../classification/logistic_regression.h"
#include "../classification/passive_aggressive_cls.h"
#include "../classification/softmax.h"
#include "../classification/knn_cls.h"
#include "../classification/hoeffding_tree.h"
#include "../classification/efdt.h"
#include "../classification/sgt.h"

// Regression
#include "../regression/linear_regression.h"
#include "../regression/bayesian_linear_reg.h"
#include "../regression/passive_aggressive_reg.h"
#include "../regression/knn_reg.h"
#include "../regression/hoeffding_tree_reg.h"
#include "../regression/arf_reg.h"

// Ensembles (SOTA streaming baselines)
#include "../ensemble/arf_cls.h"
#include "../ensemble/srp_cls.h"

// Preprocessing / feature selection
#include "../preprocessing/scalers.h"
#include "../preprocessing/feature_selection.h"

// Metrics, drift detection, pipelines
#include "../metrics/metrics.h"
#include "../drift/adwin.h"
#include "../drift/eddm.h"
#include "../pipeline/pipeline.h"

namespace py = pybind11;

PYBIND11_MODULE(_samlb_core, m) {
    m.doc() = "SAMLB C++ core — streaming learners, preprocessing, metrics, drift, pipelines";

    // ------------------------------------------------------------------ //
    //  ABSTRACT INTERFACES  (needed so pipelines accept any component)
    // ------------------------------------------------------------------ //

    py::class_<ITransformer, std::shared_ptr<ITransformer>>(m, "ITransformer");
    py::class_<IClassifier,  std::shared_ptr<IClassifier>>(m, "IClassifier");
    py::class_<IRegressor,   std::shared_ptr<IRegressor>>(m, "IRegressor");

    // ------------------------------------------------------------------ //
    //  CLASSIFICATION
    // ------------------------------------------------------------------ //

    py::class_<NaiveBayes, IClassifier, std::shared_ptr<NaiveBayes>>(m, "NaiveBayes")
        .def(py::init<>())
        .def("learn_one",         &NaiveBayes::learn_one)
        .def("predict_one",       &NaiveBayes::predict_one)
        .def("predict_proba_one", &NaiveBayes::predict_proba_one)
        .def("reset",             &NaiveBayes::reset);

    py::class_<Perceptron, IClassifier, std::shared_ptr<Perceptron>>(m, "Perceptron")
        .def(py::init<double>(), py::arg("learning_rate") = 0.01)
        .def("learn_one",         &Perceptron::learn_one)
        .def("predict_one",       &Perceptron::predict_one)
        .def("predict_proba_one", &Perceptron::predict_proba_one)
        .def("reset",             &Perceptron::reset)
        .def_readwrite("learning_rate", &Perceptron::learning_rate);

    py::class_<LogisticRegression, IClassifier, std::shared_ptr<LogisticRegression>>(
            m, "LogisticRegressionClassifier")
        .def(py::init<double, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("l2")            = 0.0)
        .def("learn_one",         &LogisticRegression::learn_one)
        .def("predict_one",       &LogisticRegression::predict_one)
        .def("predict_proba_one", &LogisticRegression::predict_proba_one)
        .def("reset",             &LogisticRegression::reset)
        .def_readwrite("learning_rate", &LogisticRegression::learning_rate)
        .def_readwrite("l2",            &LogisticRegression::l2);

    py::class_<PassiveAggressiveClassifier, IClassifier,
               std::shared_ptr<PassiveAggressiveClassifier>>(m, "PassiveAggressiveClassifier")
        .def(py::init<double>(), py::arg("C") = 1.0)
        .def("learn_one",         &PassiveAggressiveClassifier::learn_one)
        .def("predict_one",       &PassiveAggressiveClassifier::predict_one)
        .def("predict_proba_one", &PassiveAggressiveClassifier::predict_proba_one)
        .def("reset",             &PassiveAggressiveClassifier::reset)
        .def_readwrite("C", &PassiveAggressiveClassifier::C);

    py::class_<SoftmaxRegression, IClassifier, std::shared_ptr<SoftmaxRegression>>(
            m, "SoftmaxRegression")
        .def(py::init<double, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("l2")            = 0.0)
        .def("learn_one",         &SoftmaxRegression::learn_one)
        .def("predict_one",       &SoftmaxRegression::predict_one)
        .def("predict_proba_one", &SoftmaxRegression::predict_proba_one)
        .def("reset",             &SoftmaxRegression::reset)
        .def_readwrite("learning_rate", &SoftmaxRegression::learning_rate)
        .def_readwrite("l2",            &SoftmaxRegression::l2);

    py::class_<KNNClassifier, IClassifier, std::shared_ptr<KNNClassifier>>(m, "KNNClassifier")
        .def(py::init<int, int, int>(),
             py::arg("n_neighbors")  = 5,
             py::arg("window_size")  = 1000,
             py::arg("p")            = 2)
        .def("learn_one",         &KNNClassifier::learn_one)
        .def("predict_one",       &KNNClassifier::predict_one)
        .def("predict_proba_one", &KNNClassifier::predict_proba_one)
        .def("reset",             &KNNClassifier::reset)
        .def_readwrite("n_neighbors", &KNNClassifier::n_neighbors)
        .def_readwrite("p",           &KNNClassifier::p);

    py::class_<HoeffdingTreeClassifier, IClassifier, std::shared_ptr<HoeffdingTreeClassifier>>(
            m, "HoeffdingTreeClassifier")
        .def(py::init<int, double, double, int, int, std::string>(),
             py::arg("grace_period")     = 200,
             py::arg("split_confidence") = 1e-7,
             py::arg("tie_threshold")    = 0.05,
             py::arg("nb_threshold")     = 0,
             py::arg("max_depth")        = 20,
             py::arg("split_criterion")  = "info_gain")
        .def("learn_one",         &HoeffdingTreeClassifier::learn_one)
        .def("predict_one",       &HoeffdingTreeClassifier::predict_one)
        .def("predict_proba_one", &HoeffdingTreeClassifier::predict_proba_one)
        .def("reset",             &HoeffdingTreeClassifier::reset)
        .def_readwrite("grace_period",     &HoeffdingTreeClassifier::grace_period)
        .def_readwrite("split_confidence", &HoeffdingTreeClassifier::split_confidence)
        .def_readwrite("tie_threshold",    &HoeffdingTreeClassifier::tie_threshold)
        .def_readwrite("max_depth",        &HoeffdingTreeClassifier::max_depth)
        .def_readwrite("split_criterion",  &HoeffdingTreeClassifier::split_criterion)
        .def_readwrite("leaf_prediction",  &HoeffdingTreeClassifier::leaf_prediction)
        .def_readwrite("n_split_points",   &HoeffdingTreeClassifier::n_split_points);

    py::class_<EFDTClassifier, HoeffdingTreeClassifier, std::shared_ptr<EFDTClassifier>>(
            m, "EFDTClassifier")
        .def(py::init<int, double, double, int, int>(),
             py::arg("grace_period")     = 200,
             py::arg("split_confidence") = 1e-5,
             py::arg("tie_threshold")    = 0.05,
             py::arg("nb_threshold")     = 0,
             py::arg("max_depth")        = 20);

    py::class_<SGTClassifier, IClassifier, std::shared_ptr<SGTClassifier>>(m, "SGTClassifier")
        .def(py::init<double, double, int, int>(),
             py::arg("learning_rate") = 0.1,
             py::arg("lambda_")       = 0.1,
             py::arg("grace_period")  = 200,
             py::arg("max_depth")     = 6)
        .def("learn_one",         &SGTClassifier::learn_one)
        .def("predict_one",       &SGTClassifier::predict_one)
        .def("predict_proba_one", &SGTClassifier::predict_proba_one)
        .def("reset",             &SGTClassifier::reset)
        .def_readwrite("learning_rate", &SGTClassifier::learning_rate)
        .def_readwrite("lambda_",       &SGTClassifier::lambda)
        .def_readwrite("grace_period",  &SGTClassifier::grace_period)
        .def_readwrite("max_depth",     &SGTClassifier::max_depth);

    py::class_<ARFClassifier, IClassifier, std::shared_ptr<ARFClassifier>>(m, "ARFClassifier")
        .def(py::init<int, int, double, double, double, int, int, double, int>(),
             py::arg("n_models")         = 10,
             py::arg("seed")             = 0,
             py::arg("lambda_value")     = 6.0,
             py::arg("drift_delta")      = 0.001,
             py::arg("warning_delta")    = 0.01,
             py::arg("grace_period")     = 50,
             py::arg("max_depth")        = 20,
             py::arg("split_confidence") = 0.01,
             py::arg("subspace_size")    = -1)
        .def("learn_one",         &ARFClassifier::learn_one)
        .def("predict_one",       &ARFClassifier::predict_one)
        .def("predict_proba_one", &ARFClassifier::predict_proba_one)
        .def("reset",             &ARFClassifier::reset)
        .def_readwrite("n_models", &ARFClassifier::n_models)
        .def_readwrite("seed",     &ARFClassifier::seed);

    py::class_<SRPClassifier, IClassifier, std::shared_ptr<SRPClassifier>>(m, "SRPClassifier")
        .def(py::init<int, int, double, double, double, int, int, double, double, std::string>(),
             py::arg("n_models")           = 10,
             py::arg("seed")               = 0,
             py::arg("lambda_value")       = 6.0,
             py::arg("drift_delta")        = 0.001,
             py::arg("warning_delta")      = 0.01,
             py::arg("grace_period")       = 50,
             py::arg("max_depth")          = 20,
             py::arg("split_confidence")   = 0.01,
             py::arg("subspace_fraction")  = 0.6,
             py::arg("training_method")    = "patches")
        .def("learn_one",         &SRPClassifier::learn_one)
        .def("predict_one",       &SRPClassifier::predict_one)
        .def("predict_proba_one", &SRPClassifier::predict_proba_one)
        .def("reset",             &SRPClassifier::reset)
        .def_readwrite("n_models", &SRPClassifier::n_models)
        .def_readwrite("seed",     &SRPClassifier::seed);

    // ------------------------------------------------------------------ //
    //  REGRESSION
    // ------------------------------------------------------------------ //

    py::class_<LinearRegression, IRegressor, std::shared_ptr<LinearRegression>>(
            m, "LinearRegression")
        .def(py::init<double, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("l2")            = 0.0)
        .def("learn_one",   &LinearRegression::learn_one)
        .def("predict_one", &LinearRegression::predict_one)
        .def("reset",       &LinearRegression::reset)
        .def_readwrite("learning_rate", &LinearRegression::learning_rate)
        .def_readwrite("l2",            &LinearRegression::l2);

    py::class_<BayesianLinearRegression, IRegressor, std::shared_ptr<BayesianLinearRegression>>(
            m, "BayesianLinearRegression")
        .def(py::init<double, double>(),
             py::arg("alpha") = 1.0,
             py::arg("beta")  = 1.0)
        .def("learn_one",   &BayesianLinearRegression::learn_one)
        .def("predict_one", &BayesianLinearRegression::predict_one)
        .def("reset",       &BayesianLinearRegression::reset)
        .def_readwrite("alpha", &BayesianLinearRegression::alpha)
        .def_readwrite("beta",  &BayesianLinearRegression::beta);

    py::class_<PassiveAggressiveRegressor, IRegressor,
               std::shared_ptr<PassiveAggressiveRegressor>>(m, "PassiveAggressiveRegressor")
        .def(py::init<double, double>(),
             py::arg("C")       = 1.0,
             py::arg("epsilon") = 0.1)
        .def("learn_one",   &PassiveAggressiveRegressor::learn_one)
        .def("predict_one", &PassiveAggressiveRegressor::predict_one)
        .def("reset",       &PassiveAggressiveRegressor::reset)
        .def_readwrite("C",       &PassiveAggressiveRegressor::C)
        .def_readwrite("epsilon", &PassiveAggressiveRegressor::epsilon);

    py::class_<KNNRegressor, IRegressor, std::shared_ptr<KNNRegressor>>(m, "KNNRegressor")
        .def(py::init<int, int, int>(),
             py::arg("n_neighbors") = 5,
             py::arg("window_size") = 1000,
             py::arg("p")           = 2)
        .def("learn_one",   &KNNRegressor::learn_one)
        .def("predict_one", &KNNRegressor::predict_one)
        .def("reset",       &KNNRegressor::reset)
        .def_readwrite("n_neighbors", &KNNRegressor::n_neighbors)
        .def_readwrite("p",           &KNNRegressor::p);

    py::class_<HoeffdingTreeRegressor, IRegressor, std::shared_ptr<HoeffdingTreeRegressor>>(
            m, "HoeffdingTreeRegressor")
        .def(py::init<int, double, double, int, double>(),
             py::arg("grace_period")     = 200,
             py::arg("split_confidence") = 1e-7,
             py::arg("tie_threshold")    = 0.05,
             py::arg("max_depth")        = 20,
             py::arg("learning_rate")    = 0.01)
        .def("learn_one",   &HoeffdingTreeRegressor::learn_one)
        .def("predict_one", &HoeffdingTreeRegressor::predict_one)
        .def("reset",       &HoeffdingTreeRegressor::reset)
        .def_readwrite("grace_period",     &HoeffdingTreeRegressor::grace_period)
        .def_readwrite("split_confidence", &HoeffdingTreeRegressor::split_confidence)
        .def_readwrite("tie_threshold",    &HoeffdingTreeRegressor::tie_threshold)
        .def_readwrite("max_depth",        &HoeffdingTreeRegressor::max_depth)
        .def_readwrite("learning_rate",    &HoeffdingTreeRegressor::learning_rate)
        .def_readwrite("leaf_prediction",  &HoeffdingTreeRegressor::leaf_prediction);

    py::class_<ARFRegressor, IRegressor, std::shared_ptr<ARFRegressor>>(m, "ARFRegressor")
        .def(py::init<int, int, double, double, double, int, int, double>(),
             py::arg("n_models")      = 10,
             py::arg("seed")          = 0,
             py::arg("lambda_value")  = 6.0,
             py::arg("drift_delta")   = 0.001,
             py::arg("warning_delta") = 0.01,
             py::arg("grace_period")  = 200,
             py::arg("max_depth")     = 20,
             py::arg("learning_rate") = 0.01)
        .def("learn_one",   &ARFRegressor::learn_one)
        .def("predict_one", &ARFRegressor::predict_one)
        .def("reset",       &ARFRegressor::reset)
        .def_readwrite("n_models", &ARFRegressor::n_models)
        .def_readwrite("seed",     &ARFRegressor::seed);

    // ------------------------------------------------------------------ //
    //  PREPROCESSING
    // ------------------------------------------------------------------ //

    py::class_<StandardScaler, ITransformer, std::shared_ptr<StandardScaler>>(m, "StandardScaler")
        .def(py::init<bool>(), py::arg("with_std") = true)
        .def("learn_one",     &StandardScaler::learn_one)
        .def("transform_one", &StandardScaler::transform_one)
        .def("clone_state",   [](const StandardScaler& self) { return std::make_shared<StandardScaler>(self); })
        .def("reset",         &StandardScaler::reset)
        .def_readwrite("with_std", &StandardScaler::with_std);

    py::class_<MinMaxScaler, ITransformer, std::shared_ptr<MinMaxScaler>>(m, "MinMaxScaler")
        .def(py::init<>())
        .def("learn_one",     &MinMaxScaler::learn_one)
        .def("transform_one", &MinMaxScaler::transform_one)
        .def("clone_state",   [](const MinMaxScaler& self) { return std::make_shared<MinMaxScaler>(self); })
        .def("clone_state",   [](const MinMaxScaler& self) { return std::make_shared<MinMaxScaler>(self); })
        .def("reset",         &MinMaxScaler::reset);

    py::class_<MaxAbsScaler, ITransformer, std::shared_ptr<MaxAbsScaler>>(m, "MaxAbsScaler")
        .def(py::init<>())
        .def("learn_one",     &MaxAbsScaler::learn_one)
        .def("transform_one", &MaxAbsScaler::transform_one)
        .def("clone_state",   [](const MaxAbsScaler& self) { return std::make_shared<MaxAbsScaler>(self); })
        .def("clone_state",   [](const MaxAbsScaler& self) { return std::make_shared<MaxAbsScaler>(self); })
        .def("reset",         &MaxAbsScaler::reset);

    py::class_<VarianceThreshold, ITransformer, std::shared_ptr<VarianceThreshold>>(
            m, "VarianceThreshold")
        .def(py::init<double, int>(),
             py::arg("threshold")   = 0.0,
             py::arg("min_samples") = 2)
        .def("learn_one",     &VarianceThreshold::learn_one)
        .def("transform_one", &VarianceThreshold::transform_one)
        .def("clone_state",   [](const VarianceThreshold& self) { return std::make_shared<VarianceThreshold>(self); })
        .def("reset",         &VarianceThreshold::reset)
        .def_readwrite("threshold",   &VarianceThreshold::threshold)
        .def_readwrite("min_samples", &VarianceThreshold::min_samples);

    py::class_<SelectKBest, ITransformer, std::shared_ptr<SelectKBest>>(m, "SelectKBest")
        .def(py::init<int, bool>(), py::arg("k") = 10, py::arg("use_abs") = false)
        // supervised: Python-side signature is learn_one(x, y)
        .def("learn_one",     &SelectKBest::learn_one_sup)
        .def("transform_one", &SelectKBest::transform_one)
        .def("set_feature_order", &SelectKBest::set_feature_order)
        .def("clone_state",   [](const SelectKBest& self) { return std::make_shared<SelectKBest>(self); })
        .def("reset",         &SelectKBest::reset)
        .def_readwrite("k",       &SelectKBest::k)
        .def_readwrite("use_abs", &SelectKBest::use_abs);

    // ------------------------------------------------------------------ //
    //  PIPELINES  (fused: one dict->map conversion per instance)
    // ------------------------------------------------------------------ //

    py::class_<ClassificationPipeline, IClassifier, std::shared_ptr<ClassificationPipeline>>(
            m, "ClassificationPipeline")
        .def(py::init<std::vector<std::shared_ptr<ITransformer>>, std::shared_ptr<IClassifier>>(),
             py::arg("steps"), py::arg("learner"))
        .def("learn_one",         &ClassificationPipeline::learn_one_owned)
        .def("predict_one",       &ClassificationPipeline::predict_one_owned)
        .def("predict_proba_one", &ClassificationPipeline::predict_proba_one_owned)
        .def("set_feature_order", &ClassificationPipeline::set_feature_order)
        .def("reset",             &ClassificationPipeline::reset);

    py::class_<RegressionPipeline, IRegressor, std::shared_ptr<RegressionPipeline>>(
            m, "RegressionPipeline")
        .def(py::init<std::vector<std::shared_ptr<ITransformer>>, std::shared_ptr<IRegressor>>(),
             py::arg("steps"), py::arg("learner"))
        .def("learn_one",   &RegressionPipeline::learn_one_owned)
        .def("predict_one", &RegressionPipeline::predict_one_owned)
        .def("set_feature_order", &RegressionPipeline::set_feature_order)
        .def("reset",       &RegressionPipeline::reset);

    // ------------------------------------------------------------------ //
    //  METRICS
    // ------------------------------------------------------------------ //

    py::class_<Accuracy>(m, "Accuracy")
        .def(py::init<>())
        .def("update", &Accuracy::update, py::arg("y_true"), py::arg("y_pred"))
        .def("get",    &Accuracy::get)
        .def("reset",  &Accuracy::reset);

    py::class_<MacroPrecision>(m, "MacroPrecision")
        .def(py::init<>())
        .def("update", &MacroPrecision::update, py::arg("y_true"), py::arg("y_pred"))
        .def("get",    &MacroPrecision::get)
        .def("reset",  &MacroPrecision::reset);

    py::class_<MacroRecall>(m, "MacroRecall")
        .def(py::init<>())
        .def("update", &MacroRecall::update, py::arg("y_true"), py::arg("y_pred"))
        .def("get",    &MacroRecall::get)
        .def("reset",  &MacroRecall::reset);

    py::class_<MacroF1>(m, "MacroF1")
        .def(py::init<>())
        .def("update", &MacroF1::update, py::arg("y_true"), py::arg("y_pred"))
        .def("get",    &MacroF1::get)
        .def("reset",  &MacroF1::reset);

    py::class_<MAE>(m, "MAE")
        .def(py::init<>())
        .def("update", &MAE::update, py::arg("y_true"), py::arg("y_pred"))
        .def("get",    &MAE::get)
        .def("reset",  &MAE::reset);

    py::class_<RMSE>(m, "RMSE")
        .def(py::init<>())
        .def("update", &RMSE::update, py::arg("y_true"), py::arg("y_pred"))
        .def("get",    &RMSE::get)
        .def("reset",  &RMSE::reset);

    py::class_<WindowMetric>(m, "WindowMetric")
        .def(py::init<int>(), py::arg("window_size") = 1000)
        .def("update",   &WindowMetric::update, py::arg("predicted"), py::arg("actual"))
        .def("get",      &WindowMetric::get, py::arg("metric") = 0)
        .def("accuracy", &WindowMetric::accuracy)
        .def("macro_f1", &WindowMetric::macro_f1)
        .def("reset",    &WindowMetric::reset)
        .def_property_readonly("size", &WindowMetric::size);

    py::class_<WindowRegressionMetric>(m, "WindowRegressionMetric")
        .def(py::init<int>(), py::arg("window_size") = 1000)
        .def("update", &WindowRegressionMetric::update,
             py::arg("y_true"), py::arg("y_pred"))
        .def("get",    &WindowRegressionMetric::get, py::arg("metric") = 0)
        .def("mae",    &WindowRegressionMetric::mae)
        .def("rmse",   &WindowRegressionMetric::rmse)
        .def("reset",  &WindowRegressionMetric::reset)
        .def_property_readonly("size", &WindowRegressionMetric::size);

    py::class_<R2>(m, "R2")
        .def(py::init<>())
        .def("update", &R2::update, py::arg("y_true"), py::arg("y_pred"))
        .def("get",    &R2::get)
        .def("reset",  &R2::reset);

    // ------------------------------------------------------------------ //
    //  DRIFT DETECTION
    // ------------------------------------------------------------------ //

    py::class_<ADWIN>(m, "ADWIN")
        .def(py::init<double, int, int, int, int>(),
             py::arg("delta")             = 0.002,
             py::arg("clock")             = 32,
             py::arg("max_buckets")       = 5,
             py::arg("min_window_length") = 5,
             py::arg("grace_period")      = 10)
        .def("update", &ADWIN::update)
        .def("reset",  &ADWIN::reset)
        .def_property_readonly("drift_detected", &ADWIN::drift_detected)
        .def_property_readonly("estimation",     &ADWIN::estimation)
        .def_property_readonly("width",          &ADWIN::width)
        .def_property_readonly("variance",       &ADWIN::variance)
        .def_readwrite("delta", &ADWIN::delta);

    py::class_<EDDM>(m, "EDDM")
        .def(py::init<int, double, double>(),
             py::arg("warm_start") = 30,
             py::arg("alpha")      = 0.95,
             py::arg("beta")       = 0.9)
        .def("update", &EDDM::update)
        .def("reset",  &EDDM::reset)
        .def_property_readonly("drift_detected",   &EDDM::drift_detected)
        .def_property_readonly("warning_detected", &EDDM::warning_detected)
        .def_readwrite("warm_start", &EDDM::warm_start)
        .def_readwrite("alpha",      &EDDM::alpha)
        .def_readwrite("beta",       &EDDM::beta);
}
