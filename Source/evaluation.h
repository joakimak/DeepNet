#ifndef EVALUATION_H
#define EVALUATION_H

#include "matrix.h"

using Linalg::Matrix;

template<typename T>
float accuracy(const Matrix<T>& real, const Matrix<T>& pred)
{
    float correct = 0.0;
    for(size_t r = 0; r != pred.rows(); ++r){
        float max_c = 0.0;
        size_t pred_c = 0;
        size_t real_c = 0;
        for(size_t c = 0; c != pred.cols(); ++c){
            if(pred.row_begin(r)[c] > max_c){
                max_c = pred.row_begin(r)[c];
                pred_c = c;
            }
            if(real.row_begin(r)[c] == 1)
                real_c = c;
        }
        if(pred_c == real_c)
            ++correct;
    }

    return correct / real.rows();
}

#endif // EVALUATION_H
