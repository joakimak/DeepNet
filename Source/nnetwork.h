#ifndef NNETWORK_H
#define NNETWORK_H

#include "operations.h"
#include "linalg.h"
#include "rand_utils.h"
#include "timer.h"


template<typename T>
struct Layer
{
    using func_type = decltype(Softmax<T>::eval);

    Layer(size_t in, size_t out, func_type act, func_type derv);

    template<typename P>
    Matrix<T> forward(const Matrix<T>& x, const P& policy);

    template<typename P>
    Matrix<T> backward(const Matrix<T>& weights, const Matrix<T>& delta, const P& policy);

    template<typename P>
    void update(const Matrix<T>& delta, float lr, const P& policy);

    Matrix<T> z_cache_;
    Matrix<T> x_cache_;
    Matrix<T> weights_;
    Matrix<T> bias_;
    func_type* act_;
    func_type* derv_;
};


template<typename T>
class Neural_network
{
public:
    Neural_network(){}

    template<typename A>
    void add_layer(size_t in, size_t out);

    template<typename P>
    void fit(const Matrix<T>& x, const Matrix<T>& y, int epochs,
            size_t batch_sz, float learn_rate, const P& p);

    template<typename P>
    Matrix<T> predict(const Matrix<T>& x, const P&);

    template<typename P>
    vector<Matrix<T>> backward(const Matrix<T>& error, const P& p);

private:
    std::vector<Layer<T>> layers_;
};


// =============== Layer::implementation ==============


template<typename T>
Layer<T>::Layer(
    size_t in, size_t out, Layer::func_type act, Layer::func_type derv)
: weights_{random<Matrix<T>>(out, in, 0.0, 0.01)},
  bias_{random<Matrix<T>>(out, 1, 0.0, 0.05)},
  act_(act), derv_(derv){}


template<typename T>
template<typename P>
Matrix<T> Layer<T>::forward(const Matrix<T>& x, const P& policy)
{
    x_cache_ = x;
    z_cache_ = dot(x, weights_.transpose(), policy);

    return { act_(z_cache_) };
}

template<typename T>
template<typename P>
Matrix<T> Layer<T>::backward(
    const Matrix<T>& weights, const Matrix<T>& delta, const P& policy)
{
    return dot(delta, weights, policy) * derv_(z_cache_);
}

template<typename T>
template<typename P>
void Layer<T>::update(const Matrix<T>& delta, float lr, const P& policy)
{
    auto dw = dot(delta.transpose(), x_cache_, policy);
    weights_ -= dw * lr;
}


// ============== Neural_network::implementation ==============


template<typename T>
template<typename A>
void Neural_network<T>::add_layer(size_t in, size_t out)
{
    layers_.push_back(Layer<T>(in, out, A::eval, A::derv));
}

template<typename T>
template<typename P>
Matrix<T> Neural_network<T>::predict(const Matrix<T>& x, const P& p)
{
    vector<Matrix<T>> acts;
    acts.push_back(x);
    for(auto& layer : layers_)
        acts.push_back(layer.forward(acts.back(), p));

    return acts.back();
}

template<typename T>
template<typename P>
vector<Matrix<T>> Neural_network<T>::backward(const Matrix<T>& error, const P& p)
{
    vector<Matrix<T>> deltas;
    deltas.push_back(error);
    for(auto layer = layers_.rbegin()+1; layer != layers_.rend(); ++layer)
        deltas.push_back(layer->backward((layer-1)->weights_, deltas.back(), p));

    return deltas;
}

template<typename T>
template<typename P>
void Neural_network<T>::fit(
    const Matrix<T>& x, const Matrix<T>& y, int epochs,
    size_t bsz, float lr, const P& p)
{
    std::cout << "Training model...\n";
    Timing::Timer<Timing::Milliseconds> timer;
    for(auto i = 0; i != epochs; ++i){
        timer.start();

        // Select random samples
        auto r = random<int>(0, x.rows() - bsz);
        Matrix<float> bx(x.row_begin(r), x.row_begin(r+bsz), bsz, x.cols());
        Matrix<float> by(y.row_begin(r), y.row_begin(r+bsz), bsz, y.cols());

        auto pred = predict(bx, p);
        Matrix<T> delta {};
        delta = (pred - by);
        auto deltas = backward(delta, p);
        for(size_t j = 0; j != layers_.size(); ++j)
            layers_[j].update(deltas[deltas.size()-j-1], lr, p);

        // Print status
        timer.stop();
        if(!((i+1) % 100)){
            auto loss_m = pred - by;
            auto loss = std::accumulate(loss_m.begin(), loss_m.end(), 0.0,
                [&](auto a, auto v){ return a + v * v; });
            std::cout << "Iteration #: " << i << '\n';
            std::cout << "Iteration time: " << timer.elapsed() << " ms\n";
            std::cout << "Loss: " << loss / bsz << '\n';
            std::cout << "**********************" << std::endl;
        }
        timer.reset();
    }
}


#endif // NNETWORK_H
