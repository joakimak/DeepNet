#ifndef OPERATIONS
#define OPERATIONS

#include "matrix.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

using Linalg::Matrix;

template<typename T>
struct Softmax
{
    Softmax() = delete;

    static Matrix<T> eval(const Matrix<T>& z)
    {
        auto res = z;
        for(size_t r = 0; r != res.rows(); ++r){
            float max = *std::max_element(res.row_begin(r), res.row_end(r));
            std::transform(res.row_begin(r), res.row_end(r), res.row_begin(r),
                [&](T v){ return std::exp(v - max); });
            float sum = std::accumulate(res.row_begin(r), res.row_end(r), 0.f);
            std::transform(res.row_begin(r), res.row_end(r), res.row_begin(r),
                [&](T v){ return v / sum; });
        }
        return res;
    }

    static Matrix<T> derv(const Matrix<T>& z){ return {}; }
};


template<typename T>
struct Sigmoid
{
    Sigmoid() = delete;

    static Matrix<T> eval(const Matrix<T>& z)
    {
        auto res = z;
        std::transform(res.begin(), res.end(), res.begin(),
            [](T v){ return 1 / (1 + std::exp(-v)); });
        return res;
    }

    static Matrix<T> derv(const Matrix<T>& z)
    {
        auto res = z;
        std::transform(res.begin(), res.end(), res.begin(),
            [](T v){ return v * (1 - v); });
        return res;
    }
};

template<typename T>
struct Relu
{
    Relu() = delete;

    static Matrix<T> eval(const Matrix<T>& z)
    {
        auto res = z;
        std::transform(res.begin(), res.end(), res.begin(),
            [](T v){ return std::clamp(v, T{}, v); });
        return res;
    }

    static Matrix<T> derv(const Matrix<T>& z)
    {
        auto res = z;
        std::transform(res.begin(), res.end(), res.begin(),
            [&](T v){ return v > 0.0; });
        return res;
    }
};


#endif
