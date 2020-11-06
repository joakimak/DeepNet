#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>
#include "linalg.h"
#include <string>

using namespace std;
using Linalg::Matrix;

extern "C"
Matrix<float> dot(const Matrix<float>& a, const Matrix<float>& b);

// Resource handle to device memory representing a matrix.
// Provides simple conversions to/from Linalg::Matrix.
// Uses shared memory semantics and can be copied/moved safely.
template<typename T>
class Device_matrix
{
public:
    Device_matrix(size_t rows, size_t cols);
    explicit Device_matrix(const Matrix<T>& m);
    Device_matrix(const Device_matrix&);
    Device_matrix& operator==(const Device_matrix&);
    Device_matrix(Device_matrix&& dm);
    Device_matrix& operator==(Device_matrix&& dm);
    ~Device_matrix();

    __device__ T* operator[](size_t r){ return &data_[r*cols_]; }
    __device__ const T* operator[](size_t r) const { return &data_[r*cols_]; }
    inline __device__ size_t rows() const { return rows_; }
    inline __device__ size_t cols() const { return cols_; }
    inline __device__ __host__ size_t size() const { return rows_ * cols_; }
    Matrix<T> to_host() const;

private:
    void destroy();

    T* data_ {nullptr};
    size_t rows_;
    size_t cols_;
    bool owner_;
};


// ========== Device_matrix::implementation ==========


template<typename T>
Device_matrix<T>::Device_matrix(size_t rows, size_t cols)
: rows_{rows}, cols_{cols}, owner_{true}
{
    auto e = cudaMalloc((void**)&data_, rows * cols * sizeof(T));
    if(e != cudaSuccess)
        throw std::runtime_error("failed to allocate device memory");
}

template<typename T>
Device_matrix<T>::Device_matrix(const Matrix<T>& m)
: rows_{m.rows()}, cols_{m.cols()}, owner_{true}
{
    auto e = cudaMalloc((void**)&data_, m.size() * sizeof(T));
    if(e != cudaSuccess)
        throw std::runtime_error("failed to allocate device memory");
    e = cudaMemcpy(data_, m.row_begin(0), m.size() * sizeof(T),
                   cudaMemcpyHostToDevice);
    if(e != cudaSuccess){
        destroy();
        throw std::runtime_error("failed to copy matrix to device");
    }
}

template<typename T>
Device_matrix<T>::Device_matrix(const Device_matrix& dm)
: data_{dm.data_}, rows_{dm.rows_}, cols_{dm.cols_}, owner_{false}
{}

template<typename T>
Device_matrix<T>& Device_matrix<T>::operator==(const Device_matrix& dm)
{
    data_ = dm.data_;
    rows_ = dm.rows_;
    cols_ = dm.cols_;
    owner_ = false;
}

template<typename T>
Device_matrix<T>::~Device_matrix()
{
    destroy();
}

template<typename T>
Device_matrix<T>::Device_matrix(Device_matrix&& dm)
: data_{dm.data_}, rows_{dm.rows_}, cols_{dm.cols_}, owner_{true}
{
    dm.data_ = nullptr;
    dm.owner_ = false;
}

template<typename T>
Device_matrix<T>& Device_matrix<T>::operator==(Device_matrix&& dm)
{
    if(this != dm){
        if(data_ != nullptr)
            destroy();
        data_ = dm.data_;
        dm.data_ = nullptr;
        dm.owner = false;
        rows_ = dm.rows_;
        cols_ = dm.cols_;
        owner_ = true;
    }
    return *this;
}

template<typename T>
Matrix<T> Device_matrix<T>::to_host() const
{
    Matrix<T> res(rows_, cols_);
    cudaError_t result = cudaMemcpy(
        res[0], data_, size() * sizeof(T), cudaMemcpyDeviceToHost
    );

    if (result != cudaSuccess)
        throw std::runtime_error(
            "failed to copy matrix to host ("
            + string(cudaGetErrorString(result)) + ")"
        );
    return res;
}

template<typename T>
void Device_matrix<T>::destroy()
{
    if(data_ != nullptr && owner_){
        cudaFree(data_);
    }

}


// ============== Kernel function(s) ==============


template<typename T>
__global__ void dot_kernel(
    const Device_matrix<T> a, const Device_matrix<T> b, Device_matrix<T> c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < a.rows()) && (col < b.cols()))
    {
        T sum {};
        for (size_t i = 0; i != a.cols(); ++i){
            sum += a[row][i] * b[i][col];
        }
        c[row][col] = sum;
    }
}


// ============== Host function(s) ==============

Matrix<float> dot_device(const Matrix<float>& a, const Matrix<float>& b)
{
    assert(a.cols() == b.rows());

    Device_matrix<float> dev_a {a};
    Device_matrix<float> dev_b {b};
    Device_matrix<float> dev_c {a.rows(), b.cols()};

    constexpr int bsz {16};
    dim3 threads(bsz, bsz);
    dim3 blocks(
        static_cast<int>(ceil(static_cast<float>(b.cols()) / bsz)),
        static_cast<int>(ceil(static_cast<float>(a.rows()) / bsz))
    );
    dot_kernel<float><<<blocks, threads>>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    return dev_c.to_host();
}

