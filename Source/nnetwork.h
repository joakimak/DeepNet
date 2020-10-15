#ifndef NNETWORK_H
#define NNETWORK_H

#include "deep_core.h"
#include "linalg.h"
#include "rand_utils.h"
#include "timer.h"
#include <mpi.h>

// Required for specializing Neural_network::fit(...) based on execution policy
template<typename T>
struct Identity
{
    using type = T;
};


template<typename T>
class Neural_network
{
public:
    Neural_network(){}

    template<typename A>
    void add_layer(size_t in, size_t out);

    template<typename P>
    void fit(const Matrix<T>& x, const Matrix<T>& y, int n, size_t bsz, float lr, const P& p);

private:
    struct Layer
    {
        using func_type = decltype(Softmax<T>::eval);

        Layer(size_t in, size_t out, func_type act, func_type derv);

        template<typename P>
        Matrix<T> forward(const Matrix<T>& x, const P& policy);

        template<typename P>
        Matrix<T> backward(const Matrix<T>& delta, const P& policy, func_type derv);

        template<typename P>
        Matrix<T> update(const Matrix<T>& delta, float lr, const P& policy);

        Matrix<T> x_cache_;
        Matrix<T> weights_;
        func_type* act_;
        func_type* derv_;
    };

    // Local
    template<typename P>
    void fit(const Matrix<T>& samples, const Matrix<T>& labels, int episodes,
             size_t bsz, float lr, const P& p, Identity<P>);

    // Distributed
    template<typename P>
    void fit(
        const Matrix<T>& samples, const Matrix<T>& labels,
        int episodes, size_t bsz, float lr, const P& p,
        Identity<Linalg::Policy::Distributed>);

    int batch_size_;
    float learn_rate_;
    std::vector<Layer> layers_;
};


// =============== Layer::implementation ==============


template<typename T>
Neural_network<T>::Layer::Layer(
    size_t in, size_t out, Layer::func_type act, Layer::func_type derv)
: weights_{random<Matrix<T>>(in, out, 0.0, 0.05)},
  act_(act), derv_(derv){}


template<typename T>
template<typename P>
Matrix<T> Neural_network<T>::Layer::forward(const Matrix<T>& x, const P& policy)
{
    x_cache_ = x;
    return { act_(dot(x, weights_, policy)) };
}

template<typename T>
template<typename P>
Matrix<T> Neural_network<T>::Layer::backward(
    const Matrix<T>& delta, const P& policy, Layer::func_type derv)
{
    return dot(delta, weights_.transpose(), policy) * derv(x_cache_);
}

template<typename T>
template<typename P>
Matrix<T> Neural_network<T>::Layer::update(
    const Matrix<T>& delta, float lr, const P& policy)
{
    auto dw = dot(x_cache_.transpose(), delta, policy);
    weights_ -= dw * lr;
    return dw;
}


// ============== Neural_network::implementation ==============


template<typename T>
template<typename A>
void Neural_network<T>::add_layer(size_t in, size_t out)
{
    layers_.push_back(Layer(in, out, A::eval, A::derv));
}

template<typename T>
template<typename P>
void Neural_network<T>::fit(
    const Matrix<T>& samples, const Matrix<T>& labels, int episodes,
    size_t bsz, float lr, const P &p)
{
   fit(samples, labels, episodes, bsz, lr, p, Identity<P>{});
}

// Regular fit, for each execution policy except Distributed
template<typename T>
template<typename P>
void Neural_network<T>::fit(
    const Matrix<T>& samples, const Matrix<T>& labels,
    int episodes, size_t bsz, float lr, const P& p, Identity<P>)
{
    cout << "Training model...\n";
    Timing::Timer<Timing::Milliseconds> timer;
    for(auto i = 0; i != episodes; ++i){
        timer.start();

        // Select random samples
        auto r = random<int>(0, samples.rows() - bsz);
        Matrix<float> bx(
            samples.row_begin(r),
            samples.row_begin(r+bsz),
            bsz, samples.cols()
        );
        Matrix<float> by(
            labels.row_begin(r),
            labels.row_begin(r+bsz),
            bsz, labels.cols()
        );

        // Feed forward
        vector<Matrix<T>> acts;
        acts.push_back(bx);
        for(auto& l : layers_)
            acts.push_back(l.forward(acts.back(), p));

        // Backpropagation
        auto yhat = acts.back();
        acts.pop_back();
        auto loss = (yhat - by);
        for(auto l = layers_.rbegin(); l != layers_.rend() - 1; ++l){
            auto d = l->backward(loss, p, (l+1)->derv_);
            l->update(loss, lr, p);
            loss = d;
        }
        layers_.front().update(loss, lr, p);

        // Print status
        timer.stop();
        if(!((i+1) % 100)){
            auto loss_m = yhat - by;
            auto loss = std::accumulate(loss_m.begin(), loss_m.end(), 0.0,
                [&](auto a, auto v){ return a + v * v; });
            cout << "Iteration #: " << i << '\n';
            cout << "Iteration time: " << timer.elapsed() << " ms\n";
            cout << "Loss: " << loss / bsz << '\n';
            cout << "**********************" << endl;
        }
        timer.reset();
    }
}


// Distributed fit, let each processor update its weights locally
// and make a global update at the end of the episode
template<typename T>
template<typename P>
void Neural_network<T>::fit(
    const Matrix<T>& samples, const Matrix<T>& labels,
    int episodes, size_t bsz, float lr, const P&,
    Identity<Linalg::Policy::Distributed>)
{
    // Spawn other processes
    int mpirank = 0;
    int mpisize;
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    bsz /= mpisize;
    srand(mpirank);
    Linalg::Policy::Sequential p;

    vector<Matrix<float>> recv;
    for(auto& l : layers_)
        recv.push_back({l.weights_.rows() * mpisize, l.weights_.cols()});

    cout << "Training model...\n";
    Timing::Timer<Timing::Milliseconds> timer;
    for(auto i = 0; i != episodes; ++i){
        timer.start();

        // Select random samples
        auto r = random<int>(0, samples.rows() - bsz);
        Matrix<float> bx(
            samples.row_begin(r),
            samples.row_begin(r+bsz),
            bsz, samples.cols()
        );
        Matrix<float> by(
            labels.row_begin(r),
            labels.row_begin(r+bsz),
            bsz, labels.cols()
        );

        // Feed forward
        vector<Matrix<T>> acts;
        acts.push_back(bx);
        for(auto& l : layers_)
            acts.push_back(l.forward(acts.back(), p));

        // Backpropagation
        auto yhat = acts.back();
        acts.pop_back();
        auto loss = (yhat - by);
        vector<Matrix<T>> dws;
        for(auto l = layers_.rbegin(); l != layers_.rend() - 1; ++l){
            auto d = l->backward(loss, p, (l+1)->derv_);
            dws.push_back(l->update(loss, lr, p));
            loss = d;
        }
        layers_.front().update(loss, lr, p);

        // Gather weight updates from other processes
        for(size_t i = 0; i != dws.size(); ++i)
            MPI_Gather(dws[i][0], dws[i].size(), MPI_FLOAT, recv[i][0], recv[i].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        if(mpirank == 0){
            for (int k = 1; k < mpisize; ++k) {
                for(size_t i = 0; i != recv.size(); ++i){
                    Matrix<T> dw(
                        recv[i].row_begin(k),
                        recv[i].row_end(k),
                        recv[i].rows(),
                        recv[i].cols());
                    layers_[i].weights_ -= dw * lr;
                }
            }
        }

        timer.stop();
        if(!((i+1) % 100) && !mpirank){
            auto loss_m = yhat - by;
            auto loss = std::accumulate(loss_m.begin(), loss_m.end(), 0.0,
                [&](auto a, auto v){ return a + v * v; });
            cout << "Iteration #: " << i << '\n';
            cout << "Iteration time: " << timer.elapsed() << " ms\n";
            cout << "Loss: " << loss / bsz << '\n';
            cout << "**********************" << endl;
        }
        timer.reset();
    }
}




#endif // NNETWORK_H
