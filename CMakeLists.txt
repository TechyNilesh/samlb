cmake_minimum_required(VERSION 3.17)
project(samlb_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release optimisations
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  add_compile_options(/O2)
else()
  # NOTE: no -ffast-math. It implies -ffinite-math-only, under which the
  # -inf sentinels used throughout the learners (softmax.cpp, naive_bayes.cpp,
  # hoeffding_tree.cpp, ...) are undefined behaviour; the optimiser then folds
  # comparisons against them inconsistently and results become unreproducible.
  # -fno-math-errno keeps most of the speed without licensing that assumption.
  add_compile_options(-O3 -march=native -fno-math-errno)
  if(SAMLB_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=address)
  endif()
endif()

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# Collect all C++ sources
set(CPP_SOURCES
    _cpp/classification/naive_bayes.cpp
    _cpp/classification/perceptron.cpp
    _cpp/classification/logistic_regression.cpp
    _cpp/classification/passive_aggressive_cls.cpp
    _cpp/classification/softmax.cpp
    _cpp/classification/knn_cls.cpp
    _cpp/classification/hoeffding_tree.cpp
    _cpp/classification/efdt.cpp
    _cpp/classification/sgt.cpp
    _cpp/classification/hat_cls.cpp
    _cpp/regression/linear_regression.cpp
    _cpp/regression/bayesian_linear_reg.cpp
    _cpp/regression/passive_aggressive_reg.cpp
    _cpp/regression/knn_reg.cpp
    _cpp/regression/hoeffding_tree_reg.cpp
    _cpp/regression/fimtdd.cpp
    _cpp/regression/arf_reg.cpp
    _cpp/ensemble/arf_cls.cpp
    _cpp/ensemble/srp_cls.cpp
    _cpp/ensemble/srp_reg.cpp
    _cpp/ensemble/leveraging_bagging_cls.cpp
    _cpp/ensemble/leveraging_bagging_reg.cpp
    _cpp/ensemble/sgbt.cpp
    _cpp/preprocessing/scalers.cpp
    _cpp/preprocessing/feature_selection.cpp
    _cpp/metrics/metrics.cpp
    _cpp/drift/adwin.cpp
    _cpp/drift/eddm.cpp
    _cpp/pipeline/pipeline.cpp
    _cpp/bindings/pybind_module.cpp
)

pybind11_add_module(_samlb_core ${CPP_SOURCES})

target_include_directories(_samlb_core PRIVATE _cpp/core _cpp)

# Place .so next to samlb package so `import samlb` finds it
install(TARGETS _samlb_core DESTINATION samlb)
