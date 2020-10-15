#include <iostream>
#include <fstream>

#include "linalg.h"
#include "deep_core.h"
#include "nnetwork.h"
#include "data.h"

using namespace Linalg;
using namespace std;


int main(int argc, char* argv[])
{
    // In case distributed execution policy is used.
    MPI_Init(&argc, &argv);

    cout << "Loading data...\n";
    cout.flush();
    auto data = mnist<float>("train.txt");
    auto samples = data.first.vsplit(static_cast<int>(data.first.rows() * 1));
    auto labels = data.second.vsplit(static_cast<int>(data.second.rows() * 1));
    auto train_x = samples.first;
    auto test_x = samples.second;
    auto train_y = labels.first;
    auto test_y = labels.second;

    cout << "\n==== Data summary ====\n";
    cout << "train_x: (" << train_x.rows() << ", " << train_x.cols() << ")\n";
    cout << "test_x: (" << test_x.rows() << ", " << test_x.cols() << ")\n";
    cout << "train_y: (" << train_y.rows() << ", " << train_y.cols() << ")\n";
    cout << "test_y: (" << test_y.rows() << ", " << test_y.cols() << ")\n";
    cout << "======================\n" << endl;

    Neural_network<float> nn;
    nn.add_layer<Relu<float>>(784, 128);
    nn.add_layer<Relu<float>>(128, 64);
    nn.add_layer<Softmax<float>>(64, 10);

    nn.fit(train_x, train_y, 10000, 256, 0.01/256,
        Linalg::Policy::Heterogeneous{});
}
