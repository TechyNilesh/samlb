#pragma once
#include <unordered_map>
#include <string>

class Perceptron {
public:
    explicit Perceptron(double learning_rate = 0.01);
    void learn_one(const std::unordered_map<std::string, double>& x, int y);
    int  predict_one(const std::unordered_map<std::string, double>& x) const;
    std::unordered_map<int, double> predict_proba_one(
        const std::unordered_map<std::string, double>& x) const;
    void reset();

    double learning_rate;

private:
    // weights[class][feature]
    std::unordered_map<int, std::unordered_map<std::string, double>> weights;
    std::unordered_map<int, double> bias;

    double dot(const std::unordered_map<std::string, double>& w,
               const std::unordered_map<std::string, double>& x) const;
};
