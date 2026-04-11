cmake_minimum_required(VERSION 3.17)
project(samlb_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release optimisations
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native -ffast-math)

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
    _cpp/regression/linear_regression.cpp
    _cpp/regression/bayesian_linear_reg.cpp
    _cpp/regression/passive_aggressive_reg.cpp
    _cpp/regression/knn_reg.cpp
    _cpp/regression/hoeffding_tree_reg.cpp
    _cpp/bindings/pybind_module.cpp
)

pybind11_add_module(_samlb_core ${CPP_SOURCES})

target_include_directories(_samlb_core PRIVATE _cpp/core _cpp)

# Place .so next to samlb package so `import samlb` finds it
install(TARGETS _samlb_core DESTINATION samlb)
