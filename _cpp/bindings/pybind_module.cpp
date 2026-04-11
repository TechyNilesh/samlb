#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <string>

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

namespace py = pybind11;
using Features = std::unordered_map<std::string, double>;

PYBIND11_MODULE(_samlb_core, m) {
    m.doc() = "SAMLB C++ core — fast streaming ML algorithms";

    // ------------------------------------------------------------------ //
    //  CLASSIFICATION
    // ------------------------------------------------------------------ //

    py::class_<NaiveBayes>(m, "NaiveBayes")
        .def(py::init<>())
        .def("learn_one",         &NaiveBayes::learn_one)
        .def("predict_one",       &NaiveBayes::predict_one)
        .def("predict_proba_one", &NaiveBayes::predict_proba_one)
        .def("reset",             &NaiveBayes::reset);

    py::class_<Perceptron>(m, "Perceptron")
        .def(py::init<double>(), py::arg("learning_rate") = 0.01)
        .def("learn_one",         &Perceptron::learn_one)
        .def("predict_one",       &Perceptron::predict_one)
        .def("predict_proba_one", &Perceptron::predict_proba_one)
        .def("reset",             &Perceptron::reset)
        .def_readwrite("learning_rate", &Perceptron::learning_rate);

    py::class_<LogisticRegression>(m, "LogisticRegressionClassifier")
        .def(py::init<double, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("l2")            = 0.0)
        .def("learn_one",         &LogisticRegression::learn_one)
        .def("predict_one",       &LogisticRegression::predict_one)
        .def("predict_proba_one", &LogisticRegression::predict_proba_one)
        .def("reset",             &LogisticRegression::reset)
        .def_readwrite("learning_rate", &LogisticRegression::learning_rate)
        .def_readwrite("l2",            &LogisticRegression::l2);

    py::class_<PassiveAggressiveClassifier>(m, "PassiveAggressiveClassifier")
        .def(py::init<double>(), py::arg("C") = 1.0)
        .def("learn_one",         &PassiveAggressiveClassifier::learn_one)
        .def("predict_one",       &PassiveAggressiveClassifier::predict_one)
        .def("predict_proba_one", &PassiveAggressiveClassifier::predict_proba_one)
        .def("reset",             &PassiveAggressiveClassifier::reset)
        .def_readwrite("C", &PassiveAggressiveClassifier::C);

    py::class_<SoftmaxRegression>(m, "SoftmaxRegression")
        .def(py::init<double, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("l2")            = 0.0)
        .def("learn_one",         &SoftmaxRegression::learn_one)
        .def("predict_one",       &SoftmaxRegression::predict_one)
        .def("predict_proba_one", &SoftmaxRegression::predict_proba_one)
        .def("reset",             &SoftmaxRegression::reset)
        .def_readwrite("learning_rate", &SoftmaxRegression::learning_rate)
        .def_readwrite("l2",            &SoftmaxRegression::l2);

    py::class_<KNNClassifier>(m, "KNNClassifier")
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

    py::class_<HoeffdingTreeClassifier>(m, "HoeffdingTreeClassifier")
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
        .def_readwrite("split_criterion",  &HoeffdingTreeClassifier::split_criterion);

    py::class_<EFDTClassifier, HoeffdingTreeClassifier>(m, "EFDTClassifier")
        .def(py::init<int, double, double, int, int>(),
             py::arg("grace_period")     = 200,
             py::arg("split_confidence") = 1e-5,
             py::arg("tie_threshold")    = 0.05,
             py::arg("nb_threshold")     = 0,
             py::arg("max_depth")        = 20);

    py::class_<SGTClassifier>(m, "SGTClassifier")
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

    // ------------------------------------------------------------------ //
    //  REGRESSION
    // ------------------------------------------------------------------ //

    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<double, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("l2")            = 0.0)
        .def("learn_one",   &LinearRegression::learn_one)
        .def("predict_one", &LinearRegression::predict_one)
        .def("reset",       &LinearRegression::reset)
        .def_readwrite("learning_rate", &LinearRegression::learning_rate)
        .def_readwrite("l2",            &LinearRegression::l2);

    py::class_<BayesianLinearRegression>(m, "BayesianLinearRegression")
        .def(py::init<double, double>(),
             py::arg("alpha") = 1.0,
             py::arg("beta")  = 1.0)
        .def("learn_one",   &BayesianLinearRegression::learn_one)
        .def("predict_one", &BayesianLinearRegression::predict_one)
        .def("reset",       &BayesianLinearRegression::reset)
        .def_readwrite("alpha", &BayesianLinearRegression::alpha)
        .def_readwrite("beta",  &BayesianLinearRegression::beta);

    py::class_<PassiveAggressiveRegressor>(m, "PassiveAggressiveRegressor")
        .def(py::init<double, double>(),
             py::arg("C")       = 1.0,
             py::arg("epsilon") = 0.1)
        .def("learn_one",   &PassiveAggressiveRegressor::learn_one)
        .def("predict_one", &PassiveAggressiveRegressor::predict_one)
        .def("reset",       &PassiveAggressiveRegressor::reset)
        .def_readwrite("C",       &PassiveAggressiveRegressor::C)
        .def_readwrite("epsilon", &PassiveAggressiveRegressor::epsilon);

    py::class_<KNNRegressor>(m, "KNNRegressor")
        .def(py::init<int, int, int>(),
             py::arg("n_neighbors") = 5,
             py::arg("window_size") = 1000,
             py::arg("p")           = 2)
        .def("learn_one",   &KNNRegressor::learn_one)
        .def("predict_one", &KNNRegressor::predict_one)
        .def("reset",       &KNNRegressor::reset)
        .def_readwrite("n_neighbors", &KNNRegressor::n_neighbors)
        .def_readwrite("p",           &KNNRegressor::p);

    py::class_<HoeffdingTreeRegressor>(m, "HoeffdingTreeRegressor")
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
        .def_readwrite("learning_rate",    &HoeffdingTreeRegressor::learning_rate);
}
