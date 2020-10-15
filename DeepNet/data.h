#ifndef DATA_H
#define DATA_H

#include "matrix.h"

#include <fstream>

using Linalg::Matrix;

template<typename T>
std::pair<Matrix<T>, Matrix<T>> mnist(const std::string& fname)
{
    Matrix<T> m;
    std::ifstream file{fname};
    if(file.is_open()){
        file >> m;
        file.close();
    }
    file.close();

    auto p = m.hsplit(1);
    auto samples = p.second;
    auto labels = p.first;

    Matrix<T> ohot{labels.rows(), 10};
    for(size_t i = 0; i != labels.rows(); ++i){
        auto label = labels[i][0];
        for(size_t c = 0; c < 10; ++c){
            if(label == c)
                ohot[i][c] = 1.0;
            else
                ohot[i][c] = 0.0;
        }
    }

    return {samples/255.0, ohot};
}


#endif // DATA_UTILS_H
