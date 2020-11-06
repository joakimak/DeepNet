#include <iostream>
#include <fstream>

#include "linalg.h"
#include "nnetwork.h"
#include "data.h"
#include "operations.h"
#include "evaluation.h"
#include "timer.h"

using namespace Linalg;


int main(int argc, char* argv[])
{
    std::cout << "Loading data...\n";
    auto data = mnist<float>("./train.txt");
    auto samples = data.first.vsplit(static_cast<int>(data.first.rows() * 0.75));
    auto labels = data.second.vsplit(static_cast<int>(data.second.rows() * 0.75));
    auto train_x = samples.first;
    auto test_x = samples.second;
    auto train_y = labels.first;
    auto test_y = labels.second;

    std::cout << "\n==== Data summary ====\n";
    std::cout << "train_x: (" << train_x.rows() << ", " << train_x.cols() << ")\n";
    std::cout << "test_x: (" << test_x.rows() << ", " << test_x.cols() << ")\n";
    std::cout << "train_y: (" << train_y.rows() << ", " << train_y.cols() << ")\n";
    std::cout << "test_y: (" << test_y.rows() << ", " << test_y.cols() << ")\n";
    std::cout << "======================\n" << std::endl;

    Neural_network<float> network {};
    network.add_layer<Relu<float>>(784, 128);
    network.add_layer<Relu<float>>(128, 64);
    network.add_layer<Relu<float>>(64, 10);

    network.fit(train_x, train_y, 3000, 256, 0.1/256,
        Linalg::Policy::Heterogeneous{});

    auto pred = network.predict(test_x, Linalg::Policy::Heterogeneous{});

    auto acc = accuracy(test_y, pred);
    std::cout << "\nAccuracy: " << acc << std::endl;
}
