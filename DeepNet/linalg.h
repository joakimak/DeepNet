#ifndef LINALG_H
#define LINALG_H

#include "matrix.h"
#include <omp.h>

#include <cmath>
#include <thread>
#include <algorithm>

using std::vector;

extern "C"
Linalg::Matrix<float> dot_device(const Linalg::Matrix<float>& a, const Linalg::Matrix<float>& b);

namespace Linalg
{

// Execution policies for linalg operations.
namespace Policy
{

struct Sequential{};

struct BlockTiled
{
    BlockTiled(int bsz) : block_size{bsz}{}
    int block_size;
};

struct StaticParallel
{
    StaticParallel(int n) : n_threads{n}{}
    int n_threads;
};

struct DynamicParallel
{
    DynamicParallel(int n) : n_dynamic_threads{n}{}
    int n_dynamic_threads;
};

struct Distributed{};

struct Heterogeneous{};

}

// Helper routines for multithreaded linalg operations.
namespace Thread_routines
{

template<typename T>
struct Dot_args
{
    const Matrix<T>& a;
    const Matrix<T>& b;
    size_t row_start;
    size_t row_end;
    Matrix<T>& out;
};

template<typename T>
void dot(const Dot_args<T>& args)
{
    for(size_t r = args.row_start; r != args.row_end; ++r)
        for(size_t c = 0; c != args.b.rows(); ++c)
            for(size_t k = 0; k < args.b.cols(); ++k)
                args.out[r][c] += args.a.row_begin(r)[k] * args.b.row_begin(c)[k];
}

template<typename T>
struct Conv_args
{
    const Matrix<T>& matrix;
    const std::vector<Matrix<T>>& filters;
    std::vector<Matrix<T>>& outs;
    size_t stride;
    size_t start;
    size_t end;
};

template<typename T>
void conv(const Conv_args<T>& args)
{
    assert(args.filters[0].rows() <= args.matrix.rows()
          && args.filters[0].cols() <= args.matrix.cols());

    for(auto n = args.start; n != args.end; ++n)
        for(auto r = 0; r != args.outs[n].rows(); ++r)
            for(auto c = 0; c != args.outs[n].cols(); ++c)
                for(auto i = 0; i != args.filters[n].rows(); ++i)
                    for(auto j = 0; j != args.filters[n].cols(); ++j)
                        args.outs[n][r][c] +=
                            args.matrix.row_begin(r*args.stride+i)[c*args.stride+j]
                            * args.filters[n].row_begin(i)[j];
}

}

// ========== dot(a,b) ==========

template<typename T>
Matrix<T> dot(const Matrix<T>& a, const Matrix<T>& b, Policy::Sequential={})
{
    assert(a.cols() == b.rows());

    Matrix<T> res{a.rows(), b.cols()};

    auto bt = b.transpose();
    for(size_t r = 0; r != a.rows(); ++r)
        for(size_t c = 0; c != bt.rows(); ++c)
            for(size_t k = 0; k != a.cols(); ++k)
                res[r][c] += a.row_begin(r)[k] * bt.row_begin(c)[k];

    return res;
}

template<typename T>
Matrix<T> dot(const Matrix<T>& a, const Matrix<T>& b, Policy::BlockTiled p)
{
    const int ar {a.rows()};
    const int ac {a.cols()};
    const int bc {b.cols()};
    auto bt = b.transpose();
    Matrix<T> res{ar, bc};
    for (int i0 = 0; i0 < ar; i0 += p.block_size) {
        int imax = i0 + p.block_size > ar ? ar : i0 + p.block_size;
        for (int j0 = 0; j0 < bc; j0 += p.block_size) {
            int jmax = j0 + p.block_size > bc ? bc : j0 + p.block_size;
            for (int k0 = 0; k0 < ac; k0 += p.block_size) {
                int kmax = k0 + p.block_size > ac ? ac : k0 + p.block_size;
                if(k0 < kmax)
                    for (int j1 = j0; j1 < jmax; ++j1)
                        for (int i1 = i0; i1 < imax; ++i1)
                            for (int k1 = k0; k1 < kmax; ++k1)
                                res[i1][j1] += a.row_begin(i1)[k1]
                                            * bt.row_begin(j1)[k1];
            }
        }
    }

    return res;
}

template<typename T>
Matrix<T> dot(const Matrix<T>& a, const Matrix<T>& b, Policy::StaticParallel p)
{
    assert(a.cols() == b.rows());

    auto bt = b.transpose();
    auto res = Matrix<T>{a.rows(), b.cols()};
    const int partition_sz = static_cast<int>(
        ceil(static_cast<float>(a.rows()) / p.n_threads)
    );
    std::vector<std::thread> threads{};
    for(size_t i = 0; i < p.n_threads; ++i){
        Thread_routines::Dot_args<T> args{
            a, bt, i * partition_sz, std::min((i+1)*partition_sz, a.rows()), res
        };
        threads.push_back(std::thread(Thread_routines::dot<T>, args));
    }
    for(auto& t : threads)
        t.join();

    return res;
}

template<typename T>
Matrix<T> dot(const Matrix<T>& a, const Matrix<T>& b, Policy::DynamicParallel p)
{
    assert(a.cols() == b.rows());

    Matrix<T> res{a.rows(), b.cols()};
    auto bt = b.transpose();
    omp_set_dynamic(p.n_dynamic_threads);
    #pragma omp parallel for
    for(auto r = 0; r < a.rows(); ++r)
        for( int c = 0; c < bt.rows(); ++c )
            for( int k = 0; k < a.cols(); ++k )
                res[r][c] += a.row_begin(r)[k] * bt.row_begin(c)[k];

    return res;
}

template<typename T>
Matrix<T> dot(const Matrix<T>& a, const Matrix<T>& b, Policy::Heterogeneous)
{
    return dot_device(a, b);
}


// ============== conv(m, fs) ==============


template<typename T>
vector<Matrix<T>> conv(
    const Matrix<T>& m,
    const vector<Matrix<T>>& filters,
    int stride,
    Policy::Sequential={})
{
    assert(filters[0].rows() <= m.rows()
           && filters[0].cols() <= m.cols());

    std::vector<Matrix<T>> res (
        filters.size(),
        Matrix<T>{
            (m.rows() - filters[0].rows()) / stride + 1,
            (m.cols() - filters[0].cols()) / stride + 1
        }
    );

    for(size_t n = 0; n != filters.size(); ++n)
        for(size_t r = 0; r != res[n].rows(); ++r)
            for(size_t c = 0; c != res[n].cols(); ++c)
                for(size_t i = 0; i != filters[n].rows(); ++i)
                    for(size_t j = 0; j != filters[n].cols(); ++j)
                        res[n][r][c] += m.row_begin(r*stride+i)[c*stride+j]
                                      * filters[n].row_begin(i)[j];

    return res;
}

template<typename T>
vector<Matrix<T>> conv(
    const Matrix<T>& m,
    const vector<Matrix<T>>& filters,
    int stride,
    Policy::StaticParallel p)
{
    assert(filters[0].rows() <= m.rows()
           && filters[0].cols() <= m.cols());

    std::vector<Matrix<T>> res (
        filters.size(),
        Matrix<T>{
            (m.rows() - filters[0].rows()) / stride + 1,
            (m.cols() - filters[0].cols()) / stride + 1
        }
    );

    const size_t partition_sz = static_cast<size_t>(
        ceil(static_cast<float>(filters.size()) / p.n_threads)
    );
    std::vector<std::thread> threads{};
    for(int i = 0; i != p.n_threads; ++i){
        size_t end = (i+1) * partition_sz;
        end = end < filters.size() ? end : filters.size();
        Thread_routines::Conv_args<T> args{
            m,
            filters,
            res,
            stride,
            i*partition_sz,
            end
        };
        threads.push_back(std::thread(Thread_routines::conv<T>, args));
    }
    for(auto& t : threads)
        t.join();

    return res;
}

template<typename T>
vector<Matrix<T>> conv(
    const Matrix<T>& m,
    const vector<Matrix<T>>& filters,
    int stride,
    Policy::DynamicParallel p)
{
    assert(filters[0].rows() <= m.rows()
           && filters[0].cols() <= m.cols());

    std::vector<Matrix<T>> res (
        filters.size(),
        Matrix<T>{
            (m.rows() - filters[0].rows()) / stride + 1,
            (m.cols() - filters[0].cols()) / stride + 1
        }
    );

    omp_set_dynamic(p.n_dynamic_threads);
    #pragma omp parallel for
    for(size_t n = 0; n < filters.size(); ++n)
        for(size_t r = 0; r != res[n].rows(); ++r)
            for(size_t c = 0; c != res[n].cols(); ++c)
                for(size_t i = 0; i != filters[n].rows(); ++i)
                    for(size_t j = 0; j != filters[n].cols(); ++j)
                        res[n][r][c] += m.row_begin(r*stride+i)[c*stride+j]
                                      * filters[n].row_begin(i)[j];

    return res;
}



}



#endif // LINALG_H
