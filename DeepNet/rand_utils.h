#ifndef RAND_UTILS_H
#define RAND_UTILS_H

#include "matrix.h"

#include <random>
#include <functional>
#include <algorithm>


static std::default_random_engine engine{std::random_device{}()};

template<typename T>
T random(T lo, T hi)
{
    return std::uniform_real_distribution<T>(lo, hi)(engine);
}

template<>
int random<int>(int lo, int hi)
{
    return std::uniform_int_distribution<>(lo, hi)(engine);
}


template<typename M>
typename std::enable_if<
std::is_base_of<Linalg::Matrix<typename M::value_type>, M>::value, M>::type
random(int rows, int cols, typename M::value_type lo, typename M::value_type hi)
{
    M matrix{rows, cols};
    std::generate(matrix.begin(), matrix.end(),
        [&](){ return random<typename M::value_type>(lo, hi); });
    return matrix;
}

#endif // RAND_UTILS_H

