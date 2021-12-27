#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-msc51-cpp"
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"
#pragma ide diagnostic ignored "openmp-use-default-none"

#include <iostream>
#include <fstream>
#include <thread>
#include <cstdarg>
#include <random>

#if defined(_ENABLE_OMP)
#include <omp.h>
#endif

#if defined(_ENABLE_AVX2)

#include <immintrin.h>

#elif defined(_ENABLE_NEON)
#include <arm_neon.h>
#endif

//region Mat

//region Constructor and Destructor
template<typename Tp>
Mat<Tp>::Mat()
        : rows(0), cols(0), steps(0), beginRowIndex(0), beginColIndex(0), dataRows(0), dataCols(0), channels(1),
          data(nullptr), refCntPtr(nullptr), dataType(util::dataType(typeid(Tp)))
{}

template<typename Tp>
Mat<Tp>::Mat(const Mat<Tp> &other)
        : rows(other.rows), cols(other.cols), steps(other.steps), beginRowIndex(other.beginRowIndex),
          beginColIndex(other.beginColIndex), dataRows(other.dataRows), dataCols(other.dataCols),
          channels(other.channels),
          data(other.data), refCntPtr(other.refCntPtr), dataType(util::dataType(typeid(Tp)))
{
    *(this->refCntPtr) += 1;
}

template<typename Tp>
Mat<Tp>::Mat(size_t rows_, size_t cols_, int channels_)
        : rows(rows_), cols(cols_), steps(cols_), beginRowIndex(0), beginColIndex(0), dataRows(rows_), dataCols(cols_),
          channels(channels_), data(nullptr), refCntPtr(nullptr), dataType(util::dataType(typeid(Tp)))
{
    this->data = util::allocate<Tp>(this->channels * this->rows, this->cols);
    setRefCntPtr((int *) (&this->data[this->channels * this->dataRows - 1][this->dataCols]));
    *this->getRefCntPtr() = 1;
}

template<typename Tp>
Mat<Tp>::Mat(size_t rows_, size_t cols_, int channels_, initializer_list<string> pathList)
        : rows(rows_), cols(cols_), steps(cols_), beginRowIndex(0), beginColIndex(0), dataRows(rows_),
          dataCols(cols_), channels(channels_), data(nullptr), refCntPtr(nullptr),
          dataType(util::dataType(typeid(Tp)))
{
    if (channels_ <= 0) {
#ifdef DEBUG
        fprintf(stderr, "\033[31m[File: %s, Function: %s]->Invalid number of channels(%zu). Exit.\n\033[0m", __FILE__,
                __FUNCTION__, channels_);
#endif
        exit(EXIT_FAILURE);
    }
    if (this->channels != pathList.size()) {
#ifdef DEBUG
        fprintf(stderr,
                "\033[31m[File: %s, Function: %s]->Inconsistent number of channels(%d) and path of files(%d). Exit.\n\033[0m",
                __FILE__, __FUNCTION__, channels_, pathList.size());
#endif
        exit(EXIT_FAILURE);
    }
    this->data = util::allocate<Tp>(this->channels * this->rows, this->cols);
    int channelIndex = 0;
    for (const string &beg: pathList) {
        if (this->rows != util::getRowsOfTXTFile(beg) || this->cols != util::getColsOfTXTFile(beg)) {
            cout << "Inconsistent specified size and data's size in file. Exit." << endl;
            exit(EXIT_FAILURE);
        }
        util::read(beg, &(this->data[this->rows * channelIndex]), this->rows, this->cols);
        ++channelIndex;
    }
    setRefCntPtr((int *) (&this->data[this->channels * this->dataRows - 1][this->dataCols]));
    *this->getRefCntPtr() = 1;
}

template<typename Tp>
Mat<Tp>::Mat(const string &path)
        : beginRowIndex(0), beginColIndex(0), channels(1), data(nullptr), refCntPtr(nullptr),
          dataType(util::dataType(typeid(Tp)))
{
    if ((string *) &path == nullptr) {
        cout << "-- Mat(const string &path): The parameter is nullptr. Exit." << endl;
    }
    this->rows = util::getRowsOfTXTFile(path);
    this->cols = util::getColsOfTXTFile(path);
    this->dataRows = this->rows;
    this->dataCols = this->cols;
    this->steps = this->cols;
    this->data = util::allocate<Tp>(this->rows, this->cols);
    util::read(path, this->data, this->rows, this->cols);
    setRefCntPtr((int *) (&this->data[this->dataRows - 1][this->dataCols]));
    *this->getRefCntPtr() = 1;
}

template<typename Tp>
Mat<Tp>::~Mat()
{
    if (this->data == nullptr) {
        cout << "-- [data] has not been allocated memory. Return." << endl;
        return;
    }
    if (this->refCntPtr == nullptr) {
        cout << "-- [refCntPtr] has not been allocated memory. Return." << endl;
    }
    cout << "-- \033[1m\033[34m~Mat()\033[0m: refCnt: " << *this->refCntPtr << " --> ";
    *this->refCntPtr -= 1;
    cout << *this->refCntPtr << endl;
    if (*this->refCntPtr <= 0) {
        free(this->data[0]);
        free(this->data);
        cout << "   \033[1m\033[34m~Mat()\033[0m: Free memory successfully." << endl;
    }
}
//endregion

//region Operator Overload


template<typename Tp>
Mat<Tp> Mat<Tp>::operator+(const Mat<Tp> &other) const
{
    if (this->rows != other.rows || this->cols != other.cols || this > channels != other.channels) {
        cout << "Inconsistent size of two matrices for addition. Exit" << endl;
        exit(EXIT_FAILURE);
    }
    Mat<Tp> res(other.rows, other.cols, other.channels);
    util::Timer timer("Matrix<Tp> Addition");
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) + other.at(i, j, k);
            }
        }
    }
    return res;
}

template<>
Mat<float> Mat<float>::operator+(const Mat<float> &other) const
{
    if (this->rows != other.rows || this->cols != other.cols || this->channels != other.channels) {
        cout << "Inconsistent size of two matrices for addition. Exit" << endl;
        exit(EXIT_FAILURE);
    }
    Mat<float> res(other.rows, other.cols, other.channels);
    util::Timer timer("Mat<float> Addition");
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
#if defined(_ENABLE_AVX2)
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; j += 8) {
                __m256 v1, v2;
                __m256 r;
                v1 = _mm256_loadu_ps(this->rowPtr(i, k) + j);
                v2 = _mm256_loadu_ps(other.rowPtr(i, k) + j);
                r = _mm256_add_ps(v1, v2);
                _mm256_storeu_ps(res.rowPtr(i, k) + j, r);
            }
            for (size_t j = this->cols / 8 * 8; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) + other.at(i, j, k);
            }
        }
    }
#elif defined(_ENABLE_NEON)
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; j += 4) {
                float32x4_t v1 = vdupq_n_f32(0.0f);
                float32x4_t v2 = vdupq_n_f32(0.0f);
                float32x4_t r = vdupq_n_f32(0.0f);
                v1 = vld1q_f32(this->rowPtr(i, k) + j);
                v2 = vld1q_f32(other.rowPtr(i, k) + j);
                r = vaddq_f32(v1, v2);
                vst1q_f32(res.rowPtr(i, k) + j, r);
            }
            for (size_t j = this->cols / 4 * 4; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) + other.at(i, j, k);
            }
        }
    }
#else
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) + other.at(i, j, k);
            }
        }
    }
#endif
    return res;
}

template<>
Mat<int> Mat<int>::operator+(const Mat<int> &other) const
{
    if (this->rows != other.rows || this->cols != other.cols) {
        cout << "Inconsistent size of two matrices for addition. Exit" << endl;
        exit(EXIT_FAILURE);
    }
    Mat<int> res(other.rows, other.cols, other.channels);
    util::Timer timer("Mat<int> Addition");
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
#if defined(_ENABLE_AVX2)
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; j += 8) {
                __m256i v1, v2;
                __m256i r;
                v1 = _mm256_loadu_si256((__m256i *) (this->rowPtr(i, k) + j));
                v2 = _mm256_loadu_si256((__m256i *) (other.rowPtr(i, k) + j));
                r = _mm256_add_epi32(v1, v2);
                _mm256_storeu_si256((__m256i *) (res.rowPtr(i, k) + j), r);
            }
            for (size_t j = this->cols / 8 * 8; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) + other.at(i, j, k);
            }
        }
    }
#elif defined(_ENABLE_NEON)
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; j += 4) {
                int32x4_t v1 = vdupq_n_s32(0.0f);
                int32x4_t v2 = vdupq_n_s32(0.0f);
                int32x4_t r = vdupq_n_s32(0.0f);
                v1 = vld1q_s32(this->rowPtr(i, k) + j);
                v2 = vld1q_s32(other.rowPtr(i, k) + j);
                r = vaddq_s32(v1, v2);
                vst1q_s32(res.rowPtr(i, k) + j, r);
            }
            for (size_t j = this->cols / 4 * 4; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) + other.at(i, j, k);
            }
        }
    }
#else
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) + other.at(i, j, k);
            }
        }
    }
#endif
    return res;
}

template<>
Mat<double> Mat<double>::operator+(const Mat<double> &other) const
{
    if (this->rows != other.rows || this->cols != other.cols) {
        cout << "Inconsistent size of two matrices for addition. Exit" << endl;
        exit(EXIT_FAILURE);
    }
    Mat<double> res(other.rows, other.cols, other.channels);
    util::Timer timer("Mat<double> Addition");
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
#if defined(_ENABLE_AVX2)
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; j += 4) {
                __m256d v1, v2;
                __m256d r;
                v1 = _mm256_loadu_pd(this->rowPtr(i, k) + j);
                v2 = _mm256_loadu_pd(other.rowPtr(i, k) + j);
                r = _mm256_add_pd(v1, v2);
                _mm256_storeu_pd(res.rowPtr(i, k) + j, r);
            }
            for (size_t j = this->cols / 4 * 4; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) + other.at(i, j, k);
            }
        }
    }
    for (size_t i = (this->rows * this->cols * this->channels) / 4 * 4;
         i < this->rows * this->cols * this->channels; ++i) {
        *(res.data[0] + i) = *(this->data[0] + i) + *(other.data[0] + i);
    }
#elif defined(_ENABLE_NEON)
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; j += 2) {
                float64x2_t v1 = vdupq_n_f64(0.0f);
                float64x2_t v2 = vdupq_n_f64(0.0f);
                float64x2_t r = vdupq_n_f64(0.0f);
                v1 = vld1q_f64(this->rowPtr(i, k) + j);
                v2 = vld1q_f64(other.rowPtr(i, k) + j);
                r = vaddq_f64(v1, v2);
                vst1q_f64(res.rowPtr(i, k) + j, r);
            }
            for (size_t j = this->cols / 2 * 2; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) + other.at(i, j, k);
            }
        }
    }
#else
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) + other.at(i, j, k);
            }
        }
    }
#endif
    return res;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::operator-(const Mat<Tp> &other) const
{
    if (this->rows != other.rows || this->cols != other.cols) {
        cout << "Inconsistent size of two matrices for addition. Exit" << endl;
        exit(EXIT_FAILURE);
    }
    Mat<Tp> res(other.rows, other.cols, other.channels);
    util::Timer timer("Mat<Tp> subtraction");
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) - other.at(i, j, k);
            }
        }
    }
    return res;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::operator*(Mat<Tp> &other) const
{
    if (this->cols != other.rows) {
        cout << "Inconsistent matrices for multiplication. Exit." << endl;
        exit(EXIT_FAILURE);
    }
    Mat<Tp> res(other.rows, other.cols, other.channels);
    if (thread::hardware_concurrency() >= 1) {
        util::Timer timer("Mat<Tp> Multiplication-Threads");
        this->thread_product(other, res);
        return res;
    }
    util::Timer timer("Mat<Tp> Multiplication");
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
    for (int p = 0; p < this->channels; ++p) {
        for (size_t i = 0; i < this->rows; i++) {
            for (size_t k = 0; k < this->cols; ++k) {
                Tp tmp = this->at(i, k, p);
                for (size_t j = 0; j < other.cols; ++j) {
                    res.at(i, j, p) += tmp * other.at(k, j, p);
                }
            }
        }
    }
    return res;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::operator*(const int &multiplier) const
{
    Mat<Tp> res(this->rows, this->cols, this->channels);
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k) * multiplier;
            }
        }
    }
    return res;
}

template<typename Tp>
Mat<Tp> &Mat<Tp>::operator=(const Mat &other)
{
    if (this == &other) {
        return *this;
    }
    cout << "-- ";
    if (this->data != nullptr && this->refCntPtr != nullptr) {
        cout << "\033[1m\033[34moperator=()\033[0m: refCnt: " << *this->refCntPtr << " --> ";
        *this->refCntPtr -= 1;
        cout << *this->refCntPtr << endl;
        if (*this->refCntPtr <= 0) {
            free(this->data[0]);
            free(this->data);
            cout << "   \033[1m\033[34moperator=()\033[0m: Free memory successfully." << endl;
        }
    }
    (*this)(other.rows, other.cols, other.steps, other.beginRowIndex, other.beginColIndex,
            other.dataRows, other.dataCols, other.channels, other.data, other.refCntPtr);
    cout << "\033[1m\033[34moperator=()\033[0m: refCnt: " << *this->refCntPtr << " --> ";
    *(this->refCntPtr) += 1;
    cout << *this->refCntPtr << endl;
    return *this;
}

template<typename Tp>
bool Mat<Tp>::operator==(const Mat &other) const
{
    if (this->rows != other.rows || this->cols != other.cols || this->channels != other.channels) {
        return false;
    }
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < other.rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                if (this->at(i, j, k) != other.at(i, j, k)) {
                    return false;
                }
            }
        }
    }
    return true;
}

template<typename Tp>
bool Mat<Tp>::operator!=(const Mat &other) const
{
    if (&other == nullptr) {
        cout << "Mat<Tp>::operator+(const Mat<Tp> other): The parameter [\033[34mother\033[0m] is nullptr. Exit."
             << endl;
        exit(EXIT_FAILURE);
    }
    if (this->rows != other.rows || this->cols != other.cols || this->channels != other.channels) {
        return false;
    }
    if (*this == other) {
        return false;
    } else {
        return true;
    }
}

template<typename Tp>
void Mat<Tp>::operator()(size_t row, size_t col, size_t step, size_t rowst, size_t colst, size_t dRow, size_t dCol,
                         int channel, Tp **dt, int *ref)
{
    (*this)(row, col, step, rowst, colst, dRow, dCol, channel);
    this->data = dt;
    this->setRefCntPtr(ref);
}

template<typename Tp>
void Mat<Tp>::operator()(size_t row, size_t col, size_t step, size_t rowst, size_t colst, size_t dRow, size_t dCol,
                         int channel)
{
    this->rows = row;
    this->cols = col;
    this->steps = step;
    this->beginRowIndex = rowst;
    this->beginColIndex = colst;
    this->dataRows = dRow;
    this->dataCols = dCol;
    this->channels = channel;
}

template<typename Tp_>
Mat<Tp_> operator*(const int &multiplier, const Mat<Tp_> &mat)
{
    return mat * multiplier;
}

template<typename Tp_>
ostream &operator<<(ostream &os, const Mat<Tp_> &mat)
{
    for (int k = 0; k < mat.channels; ++k) {
        os << "Data type is [" << util::tpToString(mat.dataType) << "]. Elements in channel[" << k << "]:" << endl;
        for (size_t j = 0; j < mat.cols; ++j) {
            os << "________";
        }
        os << "___" << endl;
        for (size_t i = 0; i < mat.rows; ++i) {
            os << "|";
            for (size_t j = 0; j < mat.cols; ++j) {
                os << mat.at(i, j, k) << "\t";
            }
            os << "|" << endl;
        }
        for (size_t j = 0; j < mat.cols; ++j) {
            os << "--------";
        }
        os << "---" << endl;
    }
    os << "rows = " << mat.rows << ". cols = " << mat.cols << endl;
    os << "steps = " << mat.steps << endl;
    os << "beginRowIndex = " << mat.beginRowIndex << ". beginColIndex = " << mat.beginColIndex << endl;
    os << "dataRows = " << mat.dataRows << ". dataCols = " << mat.dataCols << endl;
    os << "channels = " << mat.channels << endl;
    os << "refCntPtr = " << mat.refCntPtr << ". refCnt = " << *(mat.refCntPtr) << endl;
    return os;
}

template<typename Tp>
template<typename Tp_>
Mat<Tp>::operator Mat<Tp_>() const
{
    Mat<Tp_> res(this->rows, this->cols, this->channels);
    for (int k = 0; k < res.getChannels(); ++k) {
        for (size_t i = 0; i < res.getRows(); ++i) {
            for (size_t j = 0; j < res.getCols(); ++j) {
                res.at(i, j, k) = static_cast<Tp_>(this->at(i, j, k));
            }
        }
    }
    return res;
}
//endregion

template<typename Tp>
Mat<Tp> Mat<Tp>::subMat(size_t _rows_, size_t _cols_, size_t _beginRowIndex_, size_t _beginColIndex_) const
{
    Mat<Tp> res(_rows_, _cols_, this->channels);
    for (int k = 0; k < res.channels; ++k) {
        for (size_t i = 0; i < res.rows; i++) {
            for (size_t j = 0; j < res.cols; ++j) {
                res.at(i, j, k) = this->at(_beginRowIndex_ + i, _beginColIndex_ + j, k);
            }
        }
    }
    return res;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::roi(size_t rowIndex, size_t colIndex, size_t rows_, size_t cols_) const
{
    if (rows_ >= this->rows || cols_ >= this->cols) {
        cout << "Invalid size of sub-matrix. Return [nullptr]." << endl;
        return (Mat<Tp> *) nullptr;
    }
    Mat<Tp> res;
    res(rows_, cols_, this->steps, this->beginRowIndex + rowIndex, this->beginColIndex + colIndex,
        this->dataRows, this->dataCols, this->channels, this->data, this->refCntPtr);
    *(this->refCntPtr) += 1;
    return res;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::clone() const
{
    Mat<Tp> res;
    res(rows, cols, cols, 0, 0, rows, cols, channels);
    res.data = util::allocate<Tp>(res.rows * res.channels, res.cols);
    for (int k = 0; k < res.channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(i, j, k);
            }
        }
    }
    res.setRefCntPtr((int *) (&(res.data[res.channels * res.dataRows - 1][res.dataCols])));
    *(res.getRefCntPtr()) = 1;
    return res;
}

template<typename Tp>
inline Tp &Mat<Tp>::at(size_t i, size_t j, int channelIndex) const
{
    return *(&(this->data[channelIndex * this->dataRows + this->beginRowIndex][this->beginColIndex]) + j +
             i * this->steps);
}

template<typename Tp>
Tp *Mat<Tp>::rowPtr(size_t rowIndex, int channelIndex) const
{
    return &(this->at(rowIndex, 0, channelIndex));
}

template<typename Tp>
Tp *Mat<Tp>::colPtr(size_t colIndex, int channelIndex) const
{
    return &(this->at(0, colIndex, channelIndex));
}

template<typename Tp>
void Mat<Tp>::print() const
{
    cout << "-- data = " << this->data << endl;
    for (int k = 0; k < this->channels; ++k) {
        cout << "   Data type is [" << util::tpToString(this->dataType) << "]. Elements in channel[" << k << "]:"
             << endl << "   ";
        for (size_t j = 0; j < this->cols; ++j) {
            cout << "_________";
        }
        cout << "___" << endl;
        for (size_t i = 0; i < this->rows; ++i) {
            cout << "   | ";
            for (size_t j = 0; j < this->cols; ++j) {
                printf("\033[1m\033[32m%-9.1f\033[0m", static_cast<float>(this->at(i, j, k)));
            }
            cout << " |" << endl;
        }
        cout << "   ";
        for (size_t j = 0; j < this->cols; ++j) {
            cout << "---------";
        }
        cout << "---" << endl;
    }
    cout << "   rows = " << this->rows << ". cols = " << this->cols << endl;
    cout << "   steps = " << this->steps << endl;
    cout << "   beginRowIndex = " << this->beginRowIndex << ". beginColIndex = " << this->beginColIndex << endl;
    cout << "   dataRows = " << this->dataRows << ". dataCols = " << this->dataCols << endl;
    cout << "   channels = " << this->channels << endl;
    cout << "   refCntPtr = " << this->refCntPtr << ". refCnt = " << *(this->refCntPtr) << endl;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::transfer()
{
    util::Timer timer("Mat<Tp> transformation");
    if (this->rows == this->cols) {
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
        for (int k = 0; k < this->channels; ++k) {
            for (size_t i = 0; i < this->rows; ++i) {
                for (size_t j = 0; j < this->cols; ++j) {
                    if (j > i) {
                        Tp temp = this->at(i, j, k);
                        this->at(i, j, k) = this->at(j, i, k);
                        this->at(j, i, k) = temp;
                    }
                }
            }
        }
        return *this;
    }
    Mat<Tp> res(this->cols, this->rows, this->channels);
    for (int k = 0; k < this->channels; ++k) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                res.at(i, j, k) = this->at(j, i, k);
                res.at(j, i, k) = this->at(i, j, k);
            }
        }
    }
    return res;
}

template<typename Tp>
void Mat<Tp>::writeTo(const char *path) const
{
    if (path == nullptr) {
        cout << "Mat<Tp>::writeTo(char *path): The parameter [\033[34mpath\033[0m] is nullptr. Exit." << endl;
        exit(EXIT_FAILURE);
    }
    FILE *fp;
    if ((fp = fopen(path, "w")) == nullptr) {
        printf("Fail to open or create output file -> %s. Exit.", path);
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k < this->channels; ++k) {
        if (this->channels > 1) {
            fputs("Channel[", fp);
            const char channelIndex = k + '0';
            fputs((const char *) (&(channelIndex)), fp);
            fputs("]:\r\n", fp);
        }
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                char *buffer = (char *) malloc(sizeof(char) * 32);
                sprintf(buffer, "%.1f", static_cast<float>(this->at(i, j, k)));
                fputs(buffer, fp);
                if (j < this->cols - 1) fputs(" ", fp);
                else fputs("\r\n", fp);
                free(buffer);
            }
        }
    }
    fclose(fp);
}

template<typename Tp>
Mat<Tp> Mat<Tp>::txtRead(const string &path)
{
    Mat<Tp> res(1, path);
    return res;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::rand(size_t rows_, size_t cols_, int channels_, Tp bottom, Tp top)
{
    Mat<Tp> res(rows_, cols_, channels_);
    default_random_engine engine(time(nullptr));
    for (int k = 0; k < res.channels; ++k) {
        for (size_t i = 0; i < res.rows; ++i) {
            for (size_t j = 0; j < res.cols; ++j) {
                uniform_real_distribution<float> generator(bottom, top);
                res.at(i, j, k) = static_cast<Tp>(generator(engine));
            }
        }
    }
    return res;
}

template<typename Tp>
void Mat<Tp>::singleThreadMul(const Mat<Tp> &mat_A, const Mat<Tp> &mat_B, Mat<Tp> &mat_C, const size_t _beginRowIndex_,
                              const size_t rowNumToCal)
{
    for (int p = 0; p < mat_A.channels; ++p) {
        for (size_t i = 0; i < rowNumToCal; ++i) {
            for (size_t k = 0; k < mat_A.cols; ++k) {
                Tp temp = mat_A.at(_beginRowIndex_ + i, k);
                for (size_t j = 0; j < mat_B.cols; ++j) {
                    mat_C.at(_beginRowIndex_ + i, j, p) += temp * mat_B.at(k, j, p);
                }
            }
        }
    }
}

template<typename Tp>
inline void Mat<Tp>::thread_product(const Mat<Tp> &other, Mat<Tp> &res) const
{
    size_t threadsNum = thread::hardware_concurrency();
    size_t rowNumOfEachThread = this->rows / threadsNum;
    auto *threads = new thread[threadsNum]{};
    for (size_t i = 0; i < threadsNum; ++i) {
        threads[i] = std::thread(Mat<Tp>::singleThreadMul, ref(*this), ref(other), ref(res), i * rowNumOfEachThread,
                                 rowNumOfEachThread);
    }
    for (size_t i = 0; i < threadsNum; ++i) {
        threads[i].join();
    }
}

template<typename Tp>
Mat<Tp> Mat<Tp>::strassen(const Mat<Tp> &mat_A, const Mat<Tp> &mat_B)
{
    size_t n = mat_A.rows;
    if (n <= 256) {
        Mat<Tp> res(mat_A.rows, mat_B.cols, mat_A.channels);
        for (int p = 0; p < mat_A.channels; ++p) {
            for (size_t i = 0; i < mat_A.rows; ++i) {
                for (size_t k = 0; k < mat_A.cols; ++k) {
                    Tp tmp = mat_A.at(i, k, p);
                    for (size_t j = 0; j < mat_B.cols; ++j) {
                        res.at(i, j, p) += tmp * mat_B.at(k, j, p);
                    }
                }
            }
        }
        return res;
    }

    n /= 2;

    Mat<Tp> A11 = mat_A.subMat(n, n, 0, 0);
    Mat<Tp> A12 = mat_A.subMat(n, n, 0, n);
    Mat<Tp> A21 = mat_A.subMat(n, n, n, 0);
    Mat<Tp> A22 = mat_A.subMat(n, n, n, n);

    Mat<Tp> B11 = mat_B.subMat(n, n, 0, 0);
    Mat<Tp> B12 = mat_B.subMat(n, n, 0, n);
    Mat<Tp> B21 = mat_B.subMat(n, n, n, 0);
    Mat<Tp> B22 = mat_B.subMat(n, n, n, n);

    Mat<Tp> S1 = B11 - B22;
    Mat<Tp> S2 = A11 + A12;
    Mat<Tp> S3 = A21 + A22;
    Mat<Tp> S4 = B21 - B11;
    Mat<Tp> S5 = A11 + A22;
    Mat<Tp> S6 = B11 + B22;
    Mat<Tp> S7 = A12 - A22;
    Mat<Tp> S8 = B21 + B22;
    Mat<Tp> S9 = A11 - A21;
    Mat<Tp> S10 = B11 + B12;

    Mat<Tp> P1 = strassen(A11, S1);
    Mat<Tp> P2 = strassen(S2, B22);
    Mat<Tp> P3 = strassen(S3, B11);
    Mat<Tp> P4 = strassen(A22, S4);
    Mat<Tp> P5 = strassen(S5, S6);
    Mat<Tp> P6 = strassen(S7, S8);
    Mat<Tp> P7 = strassen(S9, S10);

    Mat<Tp> C11 = P5 + P4 - P2 + P6;
    Mat<Tp> C12 = P1 + P2;
    Mat<Tp> C21 = P3 + P4;
    Mat<Tp> C22 = P5 + P1 - P3 - P7;

    return merge(C11, C12, C21, C22);
}

template<typename Tp>
Mat<Tp> Mat<Tp>::merge(const Mat<Tp> &C11, const Mat<Tp> &C12, const Mat<Tp> &C21, const Mat<Tp> &C22)
{
    Mat<Tp> res(C11.rows + C21.rows, C11.cols + C12.cols, C11.channels);
    for (int r = 0; r < C11.channels; ++r) {
        for (size_t i = 0; i < C11.rows; ++i) {
            for (size_t j = 0; j < C11.cols; ++j) {
                res.at(i, j, r) = C11.at(i, j, r);
                res.at(i, j + C11.cols, r) = C12.at(i, j, r);
                res.at(i + C11.rows, j, r) = C21.at(i, j, r);
                res.at(i + C11.rows, j + C11.cols, r) = C22.at(i, j, r);
            }
        }
    }
    return res;
}
//endregion
#pragma clang diagnostic pop