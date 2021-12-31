#include <iostream>

#include "mat.hpp"
#include "macro.hpp"

Mat8F::Mat8F(size_t rows_, size_t cols_)
        : rows(rows_), cols(cols_)
{
    data = static_cast<float *>(std::aligned_alloc(256, sizeof(float) * rows * cols));
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; ++i)
        data[i] = 0.0f;
}

Mat8F::~Mat8F()
{
    setnull();
}

void Mat8F::init(size_t rows_, size_t cols_)
{
    rows = rows_;
    cols = cols_;
    data = static_cast<float *>(std::aligned_alloc(256, sizeof(float) * rows * cols));
#pragma omp parallel for
    for (size_t i = 0; i < rows * cols; ++i)
        data[i] = 0.0f;
}

void Mat8F::setnull()
{
    if (data != nullptr)
    {
        free(data);
        data = nullptr;
    }
    rows = cols = 0;
}

void Mat8F::transfer()
{
    auto *ptr = static_cast<float *>(aligned_alloc(256, sizeof(float) * cols * rows));
#pragma omp parallel for
    for (size_t i = 0; i < cols * rows; ++i)
        ptr[i] = 0.0f;
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            ptr[j * rows + i] = data[i * cols + j];
        }
    }
    free(data);
    data = ptr;
    size_t tmp = cols;
    cols = rows;
    rows = tmp;
}

void Mat8F::sgemm(const Mat8F &blobMat, const Mat8F &filterMat, Mat8F &resBlobMat)
{
    ASSERT(blobMat.isValid() && filterMat.isValid(), "Parameter [blobMat] or [filterMat] is invalid")
    ASSERT(blobMat.cols == filterMat.rows, "Inconsistent size for matrix sgemm")

    resBlobMat.setnull();
    resBlobMat.init(blobMat.rows, filterMat.cols);

    for (size_t i = 0; i < blobMat.rows; ++i)
    {
        for (size_t k = 0; k < blobMat.cols; ++k)
        {
            float tmp = blobMat.data[i * blobMat.cols + k];
            for (size_t j = 0; j < filterMat.cols; ++j)
            {
                resBlobMat.data[i * resBlobMat.cols + j] += tmp * filterMat.data[k * filterMat.cols + j];
            }
        }
    }
}

inline bool Mat8F::isValid() const
{
    return (rows > 0 && cols > 0 && data != nullptr);
}