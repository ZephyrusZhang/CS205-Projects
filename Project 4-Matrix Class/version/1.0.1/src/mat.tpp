#include <iostream>
#include <fstream>
#include <thread>
#include <mat.hpp>


#if defined(_ENABLE_OMP)
#include <omp.h>
#endif

#if defined(_ENABLE_AVX2) || defined(_ENABLE_AVX512)
#include <immintrin.h>
typedef __m256 float32x8_t;
typedef __m512 float32x16_t;
#endif

#if defined(_ENABLE_NEON)
#include <arm_neon.h>
#endif

//region Mat

//region Constructor and Destructor
template<typename Tp>
Mat<Tp>::Mat()
        : rows(0), cols(0), steps(0), beginRowIndex(0), beginColIndex(0), dataRows(0),
          dataCols(0), data(nullptr), refCntPtr(nullptr)
{}

template<typename Tp>
Mat<Tp>::Mat(const Mat<Tp> &other)
        : rows(other.rows), cols(other.cols), steps(other.steps), beginRowIndex(other.beginRowIndex),
          beginColIndex(other.beginColIndex), dataRows(other.dataRows), dataCols(other.dataCols), data(other.data),
          refCntPtr(other.refCntPtr)
{
    *(this->refCntPtr) += 1;
}

template<typename Tp>
Mat<Tp>::Mat(size_t rows_, size_t cols_)
        : rows(rows_), cols(cols_), steps(cols_), beginRowIndex(0), beginColIndex(0),
          dataRows(rows_), dataCols(cols_)
{
    this->data = util::allocate<Tp>(rows_, cols_);
    setRefCntPtr((int *) (&this->data[this->dataRows - 1][this->dataCols]));
    *this->getRefCntPtr() = 1;
}

template<typename Tp>
Mat<Tp>::Mat(const string &filePath_)
        : beginRowIndex(0), beginColIndex(0)
{
    this->rows = util::getRowsOfTXTFile(filePath_);
    this->cols = util::getColsOfTXTFile(filePath_);
    this->dataRows = this->rows;
    this->dataCols = this->cols;
    this->steps = this->cols;
    this->data = util::allocate<Tp>(this->rows, this->cols);
    util::read(filePath_, this->data, this->rows, this->cols);
    setRefCntPtr((int *) (&this->data[this->dataRows - 1][this->dataCols]));
    *this->getRefCntPtr() = 1;
}

template<typename Tp>
Mat<Tp>::Mat(size_t rows_, size_t cols_, size_t steps_, size_t beginRowIndex_, size_t beginColIndex_, size_t dataRows_,
             size_t dataCols_, Tp **src, int *refCntPtr_)
        : rows(rows_), cols(cols_), steps(steps_), beginRowIndex(beginRowIndex_), beginColIndex(beginColIndex_),
          dataRows(dataRows_), dataCols(dataCols_), data(src), refCntPtr(refCntPtr_)
{
    *this->refCntPtr += 1;
}

template<typename Tp>
Mat<Tp>::~Mat()
{
    if (this->data == nullptr) {
        cout << "[data] has not been allocated memory. Return." << endl;
        return;
    }
    cout << "~Mat(): refCnt: " << *this->refCntPtr << " --> ";
    *this->refCntPtr -= 1;
    cout << *this->refCntPtr << endl;
    if (*this->refCntPtr <= 0) {
        free(this->data[0]);
        free(this->data);
        cout << "~Mat(): Free memory successfully." << endl;
    }
}
//endregion

//region Operator Overload
template<typename Tp>
inline Tp &Mat<Tp>::operator()(size_t i, size_t j)
{
    return *(&(this->data[this->beginRowIndex][this->beginColIndex]) + j + i * this->steps);
}

template<typename Tp>
Mat<Tp> Mat<Tp>::operator+(const Mat<Tp> &other)
{
    if (this->rows != other.rows || this->cols != other.cols) {
        cout << "Inconsistent size of two matrices for addition. Exit" << endl;
        exit(EXIT_FAILURE);
    }

    Mat<Tp> res(other.rows, other.cols);
    util::Timer timer("Matrix addition");
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
    for (size_t i = 0; i < this->rows; i++) {
#if defined(_ENABLE_AVX2)
        for (size_t j = 0; j < this->cols; j += 8) {
            float32x8_t v1, v2;
            float32x8_t r = _mm256_setzero_ps();
            v1 = _mm256_load_ps(&(this->rowPtr(i)[j]));
            v2 = _mm256_load_ps(&(other.rowPtr(i)[j]));
            r = _mm256_add_ps(v1, v2);
            _mm256_store_ps(&(res.rowPtr(i)[j]), r);
        }
        for (size_t j = this->cols / 8 * 8; j < this->cols; j++) {
            res.at(i, j) = this->at(i, j) + other.at(i, j);
        }
#elif defined(_ENABLE_AVX512)
        for (size_t j = 0; j < this->cols; j += 16) {
            float32x16_t v1, v2;
            float32x16_t r = _mm512_setzero_ps();
            v1 = _mm512_load_ps(&(this->rowPtr(i)[j]));
            v2 = _mm512_load_ps(&(other.rowPtr(i)[j]));
            r = _mm512_add_ps(v1, v2);
            _mm512_store_ps(&(res.rowPtr(i)[j]), r);
        }
        for (size_t j = this->cols / 16 * 16; j < this->cols; j++) {
            res.at(i, j) = this->at(i, j) + other.at(i, j);
        }
#elif defined(_ENABLE_NEON)
        for (size_t j = 0; j < this->cols; j += 4) {
            float32x4_t v1 = vdupq_n_f32(0.0f);
            float32x4_t v2 = vdupq_n_f32(0.0f);
            float32x4_t r = vdupq_n_f32(0.0f);
            v1 = vld1q_f32(&(this->rowPtr(i)[j]));
            v2 = vld1q_f32(&(other.rowPtr(i)[j]));
            r = vaddq_f32(v1, v2);
            vst1q_f32(&(res.rowPtr(i)[j]), r);
        }
        for (size_t j = this->cols / 4 * 4; j < this->cols; j++) {
            res.at(i, j) = this->at(i, j) + other.at(i, j);
        }
#else
        for (size_t j = 0; j < this->cols; j++) {
            res.at(i, j) = this->at(i, j) + other.at(i, j);
        }
#endif
    }
    return res;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::operator-(const Mat &other)
{
    if (this->rows != other.rows || this->cols != other.cols) {
        cout << "Inconsistent size of two matrices for addition. Exit" << endl;
        exit(EXIT_FAILURE);
    }

    Mat<Tp> res(other.rows, other.cols);
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
    for (size_t i = 0; i < this->rows; i++) {
#if defined(_ENABLE_AVX2)
        for (size_t j = 0; j < this->cols; j += 8) {
            float32x8_t v1, v2;
            float32x8_t r = _mm256_setzero_ps();
            v1 = _mm256_load_ps(&(this->rowPtr(i)[j]));
            v2 = _mm256_load_ps(&(other.rowPtr(i)[j]));
            r = _mm256_sub_ps(v1,v2);
            _mm256_store_ps(&(res.rowPtr(i)[j]), r);
        }
        for (size_t j = this->cols / 8 * 8; j < this->cols; j++) {
            res.at(i, j) = this->at(i, j) - other(i, j);
        }
#elif defined(_ENABLE_AVX512)
        for (size_t j = 0; j < this->cols; j += 16) {
            float32x16_t v1, v2;
            float32x16_t r = _mm512_setzero_ps();
            v1 = _mm512_load_ps(&(this->rowPtr(i)[j]));
            v2 = _mm512_load_ps(&(other.rowPtr(i)[j]));
            r = _mm512_sub_ps(v1, v2);
            _mm512_store_ps(&(res.rowPtr(i)[j]), r);
        }
        for (size_t j = this->cols / 16 * 16; j < this->cols; j++) {
            res.at(i, j) = this->at(i, j) + other.at(i, j);
        }
#elif defined(_ENABLE_NEON)
        for (size_t j = 0; j < this->cols; j += 4) {
            float32x4_t v1 = vdupq_n_f32(0.0f);
            float32x4_t v2 = vdupq_n_f32(0.0f);
            float32x4_t r = vdupq_n_f32(0.0f);
            v1 = vld1q_f32(&(this->rowPtr(i)[j]));
            v2 = vld1q_f32(&(other.rowPtr(i)[j]));
            r = vsubq_f32(v1, v2);
            vst1q_f32(&(res.rowPtr(i)[j]), r);
        }
        for (size_t j = this->cols / 4 * 4; j < this->cols; j++) {
            res.at(i, j) = this->at(i, j) + other.at(i, j);
        }
#else
        for (size_t j = 0; j < this->cols; j++) {
            res.at(i, j) = this->at(i, j) - other.at(i, j);
        }
#endif
    }
    return res;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::operator*(Mat &other)
{
    if (this->cols != other.rows) {
        cout << "Inconsistent matrices for multiplication. Exit." << endl;
        exit(EXIT_FAILURE);
    }
    Mat<Tp> res(other.rows, other.cols);

    util::Timer timer("Matrix Multiplication");

    //region Multiplication using multi-threads
    size_t threadsNum = thread::hardware_concurrency();
    size_t rowNumOfEachThread = this->rows / threadsNum;
    auto *threads = new thread[threadsNum]{};
    for (size_t i = 0; i < threadsNum; i++) {
        threads[i] = std::thread(Mat<Tp>::singleThreadMul, ref(*this), ref(other), ref(res), i * rowNumOfEachThread,
                                 rowNumOfEachThread);
    }
    for (size_t i = 0; i < threadsNum; i++) {
        threads[i].join();
    }
    //endregion

    return res;
}

template<typename Tp>
bool Mat<Tp>::operator==(const Mat &other) const
{
    if (this->rows != other.rows || this->cols != other.cols) {
        return false;
    }
    for (size_t i = 0; i < other.rows; i++) {
        for (size_t j = 0; j < other.cols; j++) {
            if (this->at(i, j) != other.at(i, j)) {
                return false;
            }
        }
    }
    return true;
}

template<typename Tp>
bool Mat<Tp>::operator!=(const Mat &other) const
{
    if (this->rows != other.rows || this->cols != other.cols) {
        return false;
    }
    if (*this == other) {
        return false;
    } else {
        return true;
    }
}

template<typename Tp>
Mat<Tp> &Mat<Tp>::operator=(const Mat &other)
{
    if (this == &other) {
        return *this;
    }
    this->rows = other.rows;
    this->cols = other.cols;
    this->steps = other.steps;
    this->beginRowIndex = other.beginRowIndex;
    this->beginColIndex = other.beginColIndex;
    this->dataRows = other.dataRows;
    this->dataCols = other.dataCols;
    this->data = other.data;
    this->refCntPtr = other.refCntPtr;
    cout << "operator=(): refCnt: " << *this->refCntPtr << " --> ";
    *(this->refCntPtr) += 1;
    cout << *this->refCntPtr << endl;
    return *this;
}
//endregion

//region Auxiliary Functions
template<typename Tp>
Mat<Tp> Mat<Tp>::txtRead(const string &path)
{
    Mat<Tp> res(1, path);
    return res;
}

template<typename Tp>
void Mat<Tp>::print()
{
    cout << "data = " << this->data << ". Elements :" << endl;
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            printf("%-9.1f", (float) this->at(i, j));
        }
        cout << endl;
    }
    cout << "rows = " << this->rows << ". cols = " << this->cols << endl;
    cout << "steps = " << this->steps << endl;
    cout << "beginRowIndex = " << this->beginRowIndex << ". beginColIndex = " << this->beginColIndex << endl;
    cout << "dataRows = " << this->dataRows << ". dataCols = " << this->dataCols << endl;
    cout << "refCntPtr = " << this->refCntPtr << ". refCnt = " << *(this->refCntPtr) << endl;
}

template<typename Tp>
void Mat<Tp>::transfer()
{
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            if (j > i) {
                Tp temp = this->at(i, j);
                this->at(i, j) = this->at(j, i);
                this->at(j, i) = temp;
            }
        }
    }
}

template<typename Tp>
inline Tp &Mat<Tp>::at(size_t i, size_t j) const
{
    return *(&(this->data[this->beginRowIndex][this->beginColIndex]) + j + i * this->steps);
}

template<typename Tp>
void Mat<Tp>::singleThreadMul(const Mat<Tp> &mat_A, const Mat<Tp> &mat_B, Mat<Tp> &mat_C, const size_t _beginRowIndex_,
                              const size_t rowNumToCal)
{
    for (size_t i = 0; i < rowNumToCal; i++) {
        for (size_t k = 0; k < mat_A.cols; k++) {
            Tp temp = mat_A.at(_beginRowIndex_ + i, k);
            for (size_t j = 0; j < mat_B.cols; j++) {
                mat_C.at(_beginRowIndex_ + i, j) += temp * mat_B.at(k, j);
            }
        }
    }
}

template<typename Tp>
Mat<Tp> Mat<Tp>::clone() const
{
    Mat<Tp> res;
    res.rows = this->rows;
    res.cols = this->cols;
    res.steps = this->cols;
    res.beginRowIndex = 0;
    res.beginColIndex = 0;
    res.dataRows = this->rows;
    res.dataCols = this->cols;
    res.data = util::allocate<Tp>(res.rows, res.cols);
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            res.at(i, j) = this->at(i, j);
        }
    }
    res.setRefCntPtr((int *) (&(res.data[res.dataRows - 1][res.dataCols])));
    *(res.getRefCntPtr()) = 1;
    return res;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::roi(size_t rowIndex, size_t colIndex, size_t rows_, size_t cols_)
{
    cout << "roi(): rowIndex = " << rowIndex << "; colIndex = " << colIndex << endl;
    cout << "roi(): this->data = " << this->data << endl;
    return this->subMat(rows_, cols_, rowIndex, colIndex).clone();
}

template<typename Tp>
Mat<Tp> Mat<Tp>::subMat(size_t _rows_, size_t _cols_, size_t _beginRowIndex_, size_t _beginColIndex_)
{
    Mat res;
    res.rows = _rows_;
    res.cols = _cols_;
    res.steps = this->steps;
    res.beginRowIndex = this->beginRowIndex + _beginRowIndex_;
    res.beginColIndex = this->beginColIndex + _beginColIndex_;
    res.dataRows = this->dataRows;
    res.dataCols = this->dataCols;
    res.data = this->data;
    res.refCntPtr = this->refCntPtr;
    *(this->refCntPtr) += 1;
    cout << "subMat(): res.data = " << res.data << endl;
    cout << "subMat(): res.beginRowIndex = " << res.beginRowIndex << "; this->beginColIndex = " << res.beginColIndex
         << endl;
    return res;
}

template<typename Tp>
Tp *Mat<Tp>::rowPtr(size_t rowIndex) const
{
    return &(this->at(rowIndex, 0));
}

template<typename Tp>
Tp *Mat<Tp>::colPtr(size_t colIndex) const
{
    return &(this->at(0, colIndex));
}
//endregion

//region ARM NEON
#if defined(_ENABLE_NEON)

template<typename Tp>
Mat<Tp> Mat<Tp>::mul_neon(Mat<Tp> &mat_A, Mat<Tp> &mat_B)
{
    Mat<Tp> res(mat_A.rows, mat_B.cols);
    util::Timer timer("Matrix multiplication using NEON IS");
    for (size_t i = 0; i < mat_A.rows; i++) {
        for (size_t j = 0; j < mat_B.cols; j++) {
            float sum[4] = {0.0f};
            float32x4_t v1 = vdupq_n_f32(0.0f);
            float32x4_t v2 = vdupq_n_f32(0.0f);
            float32x4_t r = vdupq_n_f32(0.0f);
            for (size_t k = 0; k < mat_A.cols; k += 4) {
                v1 = vld1q_f32(&(mat_A.rowPtr(i)[k]));
                v2 = vld1q_f32(&(mat_B.rowPtr(j)[k]));
                r = vmlaq_f32(r, v1, v2);
            }
//            cout << "sum = " << (sum[0] + sum[1] + sum[2] + sum[3]) << endl;
            vst1q_f32(sum, r);
            res.at(i, j) = sum[0] + sum[1] + sum[2] + sum[3];
//            cout << "res.at()" << res.at(i, j) << endl;
        }
    }
    return res;
}

#endif
//endregion

//endregion

//region Mat_

//region Constructor & Destructor
template<typename Tp>
Mat_<Tp>::Mat_()
        : channels(3), red(nullptr), green(nullptr), blue(nullptr),
          refCntPtr_R(nullptr), refCntPtr_G(nullptr), refCntPtr_B(nullptr)
{}

template<typename Tp>
Mat_<Tp>::Mat_(const string &pathR, const string &pathG, const string &pathB)
{
    size_t rows1 = util::getRowsOfTXTFile(pathR);
    size_t rows2 = util::getRowsOfTXTFile(pathG);
    size_t rows3 = util::getRowsOfTXTFile(pathB);
    if (!(rows1 == rows2 && rows2 == rows3)) {
        cout << "Inconsistent rows of data in different channels. Exit." << endl;
        exit(EXIT_FAILURE);
    }
    size_t cols1 = util::getColsOfTXTFile(pathR);
    size_t cols2 = util::getColsOfTXTFile(pathG);
    size_t cols3 = util::getColsOfTXTFile(pathB);
    if (!(cols1 == cols2 && cols2 == cols3)) {
        cout << "Inconsistent cols of data in different channels. Exit." << endl;
        exit(EXIT_FAILURE);
    }
    this->rows = rows1;
    this->cols = cols1;
    this->steps = cols1;
    this->beginColIndex = 0;
    this->dataRows = rows1;
    this->dataCols = cols1;
    this->red = util::allocate<Tp>(this->rows, this->cols);
    util::read(pathB, this->red, this->rows, this->cols);
    this->setRefCntPtrR((int *) (&this->red[this->dataRows - 1][this->dataCols]));
    *this->getRefCntPtrR() = 1;
    this->green = util::allocate<Tp>(this->rows, this->cols);
    util::read(pathG, this->green, this->rows, this->cols);
    this->setRefCntPtrG((int *) (&this->green[this->dataRows - 1][this->dataCols]));
    *this->getRefCntPtrG() = 1;
    this->blue = util::allocate<Tp>(this->rows, this->cols);
    util::read(pathB, this->blue, this->rows, this->cols);
    this->setRefCntPtrB((int *) (&this->blue[this->dataRows - 1][this->dataCols]));
    *this->getRefCntPtrB() = 1;
}

template<typename Tp>
Mat_<Tp>::Mat_(size_t rows_, size_t cols_, int channels_)
        : channels(channels_)
{
    this->rows = rows_;
    this->cols = cols_;
    this->steps = cols_;
    this->beginRowIndex = 0;
    this->beginColIndex = 0;
    this->dataRows = rows_;
    this->dataCols = cols_;
    this->red = util::allocate<Tp>(rows_, cols_);
    setRefCntPtrR((int *) (&this->red[this->dataRows - 1][this->dataCols]));
    *this->refCntPtr_R = 1;
    this->green = util::allocate<Tp>(rows_, cols_);
    setRefCntPtrG((int *) (&this->green[this->dataRows - 1][this->dataCols]));
    *this->refCntPtr_G = 1;
    this->blue = util::allocate<Tp>(rows_, cols_);
    setRefCntPtrB((int *) (&this->blue[this->dataRows - 1][this->dataCols]));
    *this->refCntPtr_B = 1;
}

template<typename Tp>
Mat_<Tp>::~Mat_()
{
    printf("~Mat()_: (refCnt_R, refCnt_G, refCnt_B): (%d, %d, %d) --> ", *refCntPtr_R, *refCntPtr_G, *refCntPtr_B);
    *this->refCntPtr_R -= 1;
    *this->refCntPtr_G -= 1;
    *this->refCntPtr_B -= 1;
    printf("(%d, %d, %d)\n", *refCntPtr_R, *refCntPtr_G, *refCntPtr_B);
    if (*this->refCntPtr_R <= 0) {
        free(this->red[0]);
        free(this->red);
        cout << "~Mat()_: Free memory of [red] successfully." << endl;
    }
    if (*this->refCntPtr_G <= 0) {
        free(this->green[0]);
        free(this->green);
        cout << "~Mat(): Free memory of [green] successfully." << endl;
    }
    if (*this->refCntPtr_B <= 0) {
        free(this->blue[0]);
        free(this->blue);
        cout << "~Mat(): Free memory of [blue] successfully." << endl;
    }
}
//endregion

template<typename Tp>
Mat_<Tp> Mat_<Tp>::operator*(const Mat_<Tp> &other)
{
    Mat_<Tp> res;

    Mat<Tp> matR1(this->rows, this->cols, this->steps, this->beginRowIndex, this->beginColIndex, this->dataRows,
                  this->dataCols, this->data, this->refCntPtr_R);
    Mat<Tp> matR2(other.rows, other.cols, other.steps, other.beginRowIndex, other.beginColIndex, other.dataRows,
                  other.dataCols, other.data, other.refCntPtr_R);
    Mat<Tp> resR = matR1 * matR2;
    resR

}

template<typename Tp>
void Mat_<Tp>::print()
{
    cout << "red = " << this->red << "; green = " << this->green << "; blue = " << this->blue << endl;
    cout << "Elements: " << endl;
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            printf("[%.1f, %.1f, %.1f]\t", this->at_R(i, j), this->at_G(i, j), this->at_B(i, j));
        }
        cout << endl;
    }
    cout << "rows = " << this->rows << ". cols = " << this->cols << endl;
    cout << "steps = " << this->steps << endl;
    cout << "beginRowIndex = " << this->beginRowIndex << ". beginColIndex = " << this->beginColIndex << endl;
    cout << "dataRows = " << this->dataRows << ". dataCols = " << this->dataCols << endl;
    cout << "refCntPtr = " << this->refCntPtr << ". refCnt = " << *(this->refCntPtr) << endl;
}

template<typename Tp>
Tp &Mat_<Tp>::at_R(size_t i, size_t j) const
{
    return *(&(this->red[this->beginRowIndex][this->beginColIndex]) + j + i * this->steps);
}

template<typename Tp>
Tp &Mat_<Tp>::at_G(size_t i, size_t j) const
{
    return *(&(this->green[this->beginRowIndex][this->beginRowIndex]) + j + i * this->steps);
}

template<typename Tp>
Tp &Mat_<Tp>::at_B(size_t i, size_t j) const
{
    return *(&(this->blue[this->beginRowIndex][this->beginColIndex]) + j + i * this->steps);
}

//endregion