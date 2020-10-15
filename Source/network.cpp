/*
//#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "deep_core.h"
#include "timer.h"
#include "data.h"
#include "rand_utils.h"
#include "linalg.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <numeric>

int main2(int argc, char* argv[])
{
    int mpirank = 0;
    cout << "Loading data...\n";
    cout.flush();
    auto data = mnist<float>("/home/joakim/Source/C++/Lab2_Code/train.txt");
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

    size_t x_size = train_x.size();
    size_t y_size = train_y.size();

    int episodes = 10000;
    int batch_size = 256;
    float learn_rate = 0.01 / batch_size;

    auto w1 = random<Matrix<float>>(784, 128, 0.0, 0.05);
    auto w2 = random<Matrix<float>>(128, 64, 0.0, 0.05);
    auto w3 = random<Matrix<float>>(64, 10, 0.0, 0.05);

    #ifdef USE_MPI
        int mpisize;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        batch_size /= mpisize;
        srand (mpirank);

        Matrix<float> dw1_recv(w1.rows() * mpisize, w1.cols());
        Matrix<float> dw2_recv(w2.rows() * mpisize, w2.cols());
        Matrix<float> dw3_recv(w3.rows() * mpisize, w3.cols());
    #endif

    cout << "Training model...\n";
    Timing::Timer<Timing::Milliseconds> timer;
    for(auto i = 0; i != 10000; ++i){
        timer.start();
        auto r = random<int>(0, train_x.rows() - batch_size);
        Matrix<float> bx(
            train_x.row_begin(r),
            train_x.row_begin(r+batch_size),
            batch_size, train_x.cols()
        );
        Matrix<float> by(
            train_y.row_begin(r),
            train_y.row_begin(r+batch_size),
            batch_size, train_y.cols()
        );

        auto p = Linalg::Policy::DynamicParallel{4};
        // Feed forward
        auto a1 = relu(Linalg::dot(bx, w1, p));
        auto a2 = relu(Linalg::dot(a1, w2, p));
        auto yhat = softmax(Linalg::dot(a2, w3, p));

        // Backpropagation
        auto dyhat = (yhat - by);
        auto dw3 = Linalg::dot(a2.transpose(), dyhat, p);
        auto dz2 = Linalg::dot(dyhat, w3.transpose(), p) * relu_d(a2);
        auto dw2 = Linalg::dot(a1.transpose(), dz2, p);
        auto dz1 = Linalg::dot(dz2, w2.transpose(), p) * relu_d(a1);
        auto dw1 = Linalg::dot(bx.transpose(), dz1, p);

        w3 -= dw3 * learn_rate;
        w2 -= dw2 * learn_rate;
        w1 -= dw1 * learn_rate;

        #ifdef USE_MPI
            MPI_Gather(dw1[0], dw1.size(), MPI_FLOAT, dw1_recv[0], dw1_recv.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gather(dw2[0], dw2.size(), MPI_FLOAT, dw2_recv[0], dw2_recv.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gather(dw3[0], dw3.size(), MPI_FLOAT, dw3_recv[0], dw3_recv.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
            if(mpirank == 0){
                for (int k = 0; k < mpisize; ++k) {
                  if (k != mpirank) {
                      Matrix<float> dw3_k(dw3_recv.row_begin(k), dw3_recv.row_end(k), dw3.rows(), dw3.cols());
                      Matrix<float> dw2_k(dw2_recv.row_begin(k), dw2_recv.row_end(k), dw2.rows(), dw2.cols());
                      Matrix<float> dw1_k(dw1_recv.row_begin(k), dw1_recv.row_end(k), dw1.rows(), dw1.cols());
                      w3 -= dw3_k * learn_rate;
                      w2 -= dw2_k * learn_rate;
                      w1 -= dw1_k * learn_rate;
                  }
                }
            }
        #endif

        timer.stop();
        if(!((i+1) % 100) && mpirank==0){
            auto loss_m = yhat - by;
            auto loss = std::accumulate(loss_m.begin(), loss_m.end(), 0.0,
            [&](auto a, auto v){ return a + v * v; });
            cout << "Iteration #: " << i << '\n';
            cout << "Iteration time: " << timer.elapsed() << " ms\n";
            cout << "Loss: " << loss / batch_size << '\n';
            cout << "**********************" << endl;
        }
        timer.reset();
    }
    return 0;
}*/
