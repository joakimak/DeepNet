#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
#include <algorithm>

using std::vector;

namespace Linalg
{

template<typename T>
class Matrix
{
public:
    using value_type = T;
    using row_type = T*;
    using const_row_type = const T*;
    using iterator = typename vector<value_type>::iterator;
    using const_iterator = typename vector<value_type>::const_iterator;

    Matrix(){}
    Matrix(size_t rows, size_t cols)
    : data_(rows*cols), rows_{rows}, cols_{cols}{}

    template<typename I>
    Matrix(I b, I e, size_t rows, size_t cols)
    : data_(b, e), rows_{rows}, cols_{cols}{}

    Matrix(const Matrix& other) : data_(other.data_), rows_(other.rows_), cols_(other.cols_){}

    template<typename V>
    explicit Matrix(const vector<V>& x)
    : data_(x), rows_{x.size()}, cols_{1}{}

    Matrix& operator=(const Matrix<T>& other);
    Matrix& operator=(value_type x);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }

    iterator begin(){ return data_.begin(); }
    iterator end(){ return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }
    row_type row_begin(size_t r){ return &data_[r*cols_]; }
    row_type row_end(size_t r){ return &data_[(r+1)*cols_]; }
    const_row_type row_begin(size_t r) const { return &data_[r*cols_]; }
    const_row_type row_end(size_t r) const { return &data_[(r+1)*cols_]; }
    row_type operator[](size_t r){ return &data_[r*cols_]; }

    bool operator==(const Matrix& other);

    Matrix operator+(const Matrix& other);
    Matrix operator-(const Matrix& other);
    Matrix operator*(const Matrix& other);
    Matrix operator/(const Matrix& other);
    Matrix operator+(value_type v);
    Matrix operator-(value_type v);
    Matrix operator*(value_type v);
    Matrix operator/(value_type v);

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(const Matrix& other);
    Matrix& operator/=(const Matrix& other);
    Matrix& operator+=(value_type v);
    Matrix& operator-=(value_type v);
    Matrix& operator*=(value_type v);
    Matrix& operator/=(value_type v);

    template<typename V>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<V>& m);

    template<typename V>
    friend std::istream& operator>>(std::istream& is, Matrix<V>& m);

    Matrix transpose() const;
    Matrix& reshape(size_t rows, size_t cols);
    std::pair<Matrix, Matrix> vsplit(int n) const;
    std::pair<Matrix, Matrix> hsplit(int n) const;



private:
    vector<value_type> data_;
    size_t rows_;
    size_t cols_;
};

// ========== Definitions ==========

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other)
{
    data_ = other.data_;
    rows_ = other.rows_;
    cols_ = other.cols_;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(value_type x)
{
    for(auto& e : data_)
        e = x;
    return *this;
}

template<typename T>
bool Matrix<T>::operator==(const Matrix<T>& other)
{
    return rows_ == other.rows_
        && cols_ == other.cols_
        && data_ == other.data_;
}

template<typename V>
std::ostream& operator<<(std::ostream& os, const Matrix<V>& m)
{
    for(size_t r = 0; r != m.rows(); ++r){
        for(auto it = m.row_begin(r); it != m.row_end(r); ++it)
            os << *it << " ";
        os << '\n';
    }
    return os;
}

template<typename V>
std::istream& operator>>(std::istream& is, Matrix<V>& m)
{
    typename Matrix<V>::value_type v;
    std::string line;
    while(std::getline(is, line, '\n')){
        std::istringstream row{line};
        size_t cols = 0;
        while(row >> v){
            m.data_.push_back(v);
            ++cols;
        }
        m.cols_ = cols;
    }
    m.rows_ = m.data_.size() / m.cols_;

    return is;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix& other)
{
    assert(rows() == other.rows() && cols() == other.cols());
    return Matrix<T>{*this} += other;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& other)
{
    assert(rows() == other.rows() && cols() == other.cols());
    return Matrix<T>{*this} -= other;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other)
{
    assert(rows() == other.rows() && cols() == other.cols());
    return Matrix<T>{*this} *= other;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(const Matrix& other)
{
    assert(rows() == other.rows() && cols() == other.cols());
    return Matrix<T>{*this} /= other;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix& other)
{
    assert(rows() == other.rows() && cols() == other.cols());

    for(size_t r = 0; r != rows(); ++r)
        for(size_t c = 0; c != cols(); ++c)
            row_begin(r)[c] += other.row_begin(r)[c];

    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix& other)
{
    assert(rows() == other.rows() && cols() == other.cols());

    for(auto r = 0; r != rows(); ++r)
        for(auto c = 0; c != cols(); ++c)
            row_begin(r)[c] -= other.row_begin(r)[c];

    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix& other)
{
    assert(rows() == other.rows() && cols() == other.cols());

    for(auto r = 0; r != rows(); ++r)
        for(auto c = 0; c != cols(); ++c)
            row_begin(r)[c] *= other.row_begin(r)[c];

    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator/=(const Matrix& other)
{
    assert(rows() == other.rows() && cols() == other.cols());

    for(auto r = 0; r != rows(); ++r)
        for(auto c = 0; c != cols(); ++c)
            row_begin(r)[c] /= other.row_begin(r)[c];

    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(value_type v)
{
    return Matrix<T>{*this} += v;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(value_type v)
{
    return Matrix<T>{*this} -= v;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(value_type v)
{
    return Matrix<T>{*this} *= v;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(value_type v)
{
    return Matrix<T>{*this} /= v;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(value_type v)
{
    for(auto& e : *this)
        e += v;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-=(value_type v)
{
    for(auto& e : *this)
        e -= v;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(value_type v)
{
    for(auto& e : *this)
        e *= v;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator/=(value_type v)
{
    for(auto& e : *this)
        e /= v;
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::transpose() const
{
    Matrix<T> res{cols_, rows_};
    for(size_t r = 0; r != rows_; ++r)
        for(size_t c = 0; c != cols_; ++c)
            res[c][r] = row_begin(r)[c];

    return res;
}

template<typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::vsplit(int n) const
{
    return {
        Matrix<T>(begin(), begin() + n*cols_, n, cols_),
        Matrix<T>(begin() + n*cols_, end(), rows_ - n, cols_)
    };
}

template<typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::hsplit(int n) const
{
    auto t = transpose().vsplit(n);
    return {
        t.first.transpose(),
        t.second.transpose()
    };
}

template<typename T>
Matrix<T>& Matrix<T>::reshape(size_t rows, size_t cols)
{
    assert(rows * cols == size());
    rows_ = rows;
    cols_ = cols;
    return *this;
}

}

#endif // LINALG_H
