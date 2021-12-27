#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"
#pragma ide diagnostic ignored "openmp-use-default-none"

#include <iostream>
#include <fstream>
#include <thread>
#include <cstdarg>

#if defined(_ENABLE_OMP)
#include <omp.h>
#endif

#if defined(_ENABLE_AVX2)

#include <immintrin.h>

#endif

#if defined(_ENABLE_NEON)
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
        cout << "Invalid number of channels. Exit." << endl;
        exit(EXIT_FAILURE);
    }
    if (this->channels != pathList.size()) {
        cout << "Inconsistent number of channels and path of files. Exit." << endl;
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
        channelIndex++;
    }
    setRefCntPtr((int *) (&this->data[this->channels * this->dataRows - 1][this->dataCols]));
    *this->getRefCntPtr() = 1;
}

template<typename Tp>
[[maybe_unused]] Mat<Tp>::Mat(const string &path)
        : beginRowIndex(0), beginColIndex(0), channels(1), data(nullptr), refCntPtr(nullptr),
          dataType(util::dataType(typeid(Tp)))
{
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
        cout << "[data] has not been allocated memory. Return." << endl;
        return;
    }
    if (this->refCntPtr == nullptr) {
        cout << "[refCntPtr] has not been allocated memory. Return." << endl;
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
inline Tp &Mat<Tp>::operator()(size_t i, size_t j, int channelIndex)
{
    return *(&(this->data[channelIndex * this->rows + this->beginRowIndex][this->beginColIndex]) + j + i * this->steps);
}

template<typename Tp>
Mat<Tp> Mat<Tp>::operator+(const Mat<Tp> &other) const
{
    if (this->rows != other.rows || this->cols != other.cols) {
        cout << "Inconsistent size of two matrices for addition. Exit" << endl;
        exit(EXIT_FAILURE);
    }
    cout << "In Mat<Tp>::operator+(): ";
    Mat<Tp> res(other.rows, other.cols, other.channels);
    util::Timer timer("Matrix<Tp> addition");
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
    for (size_t i = 0; i < this->rows * this->cols * this->channels; i++) {
        *(res.data[0] + i) = *(this->data[0] + i) + *(other.data[0] + i);
    }
    return res;
}

template<>
Mat<float> Mat<float>::operator+(const Mat<float> &other) const
{
    if (this->rows != other.rows || this->cols != other.cols) {
        cout << "Inconsistent size of two matrices for addition. Exit" << endl;
        exit(EXIT_FAILURE);
    }
    Mat<float> res(other.rows, other.cols, other.channels);
    util::Timer timer("Mat<float> addition");
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
#if defined(_ENABLE_AVX2)
    for (size_t i = 0; i < this->rows * this->cols * this->channels; i += 8) {
        __m256 v1, v2;
        __m256 r = _mm256_setzero_ps();
        v1 = _mm256_load_ps(this->data[0] + i);
        v2 = _mm256_load_ps(other.data[0] + i);
        r = _mm256_add_ps(v1, v2);
        _mm256_store_ps(res.data[0] + i, r);
    }
    for (size_t i = (this->rows * this->cols * this->channels) / 8 * 8;
         i < this->rows * this->cols * this->channels; i++) {
        *(res.data[0] + i) = *(this->data[0] + i) + *(other.data[0] + i);
    }
#elif defined(_ENABLE_NEON)
    for (size_t i = 0; i < this->rows * this->cols * this->channels; i += 4) {
        float32x4_t v1 = vdupq_n_f32(0.0f);
        float32x4_t v2 = vdupq_n_f32(0.0f);
        float32x4_t r = vdupq_n_f32(0.0f);
        v1 = vld1q_f32(this->data[0] + i);
        v2 = vld1q_f32(other.data[0] + i);
        r = vaddq_f32(v1, v2);
        vst1q_f32(res.data[0] + i, r);
    }
    for (size_t i = (this->rows * this->cols * this->channels) / 4 * 4;
         i < this->rows * this->cols * this->channels; i++) {
        *(res.data[0] + i) = *(this->data[0] + i) + *(other.data[0] + i);
    }
#else
    for (size_t i = 0; i < this->rows * this->cols * this->channels; i++) {
        *(res.data[0] + i) = *(this->data[0] + i) + *(other.data[0] + i);
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
    util::Timer timer("Mat<int> addition");
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
#if defined(_ENABLE_AVX2)
    for (size_t i = 0; i < this->rows * this->cols * this->channels; i += 8) {
        __m256i v1 = _mm256_set_epi32(*(this->data[0] + i + 0), *(this->data[0] + i + 1),
                                      *(this->data[0] + i + 2), *(this->data[0] + i + 3),
                                      *(this->data[0] + i + 4), *(this->data[0] + i + 5),
                                      *(this->data[0] + i + 6), *(this->data[0] + i + 7));
        __m256i v2 = _mm256_set_epi32(*(other.data[0] + i + 0), *(other.data[0] + i + 1),
                                      *(other.data[0] + i + 2), *(other.data[0] + i + 3),
                                      *(other.data[0] + i + 4), *(other.data[0] + i + 5),
                                      *(other.data[0] + i + 6), *(other.data[0] + i + 7));
        __m256i r = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
        r = _mm256_add_epi32(v1, v2);
        _mm256_store_si256((__m256i *) (res.data[0] + i), r);
    }
    for (size_t i = (this->rows * this->cols * this->channels) / 8 * 8;
         i < this->rows * this->cols * this->channels; i++) {
        *(res.data[0] + i) = *(this->data[0] + i) + *(other.data[0] + i);
    }
#elif defined(_ENABLE_NEON)
    for (size_t i = 0; i < this->rows * this->cols * this->channels; i += 4) {
        int32x4_t v1 = vdupq_n_s32(0);
        int32x4_t v2 = vdupq_n_s32(0);
        int32x4_t r = vdupq_n_s32(0);
        v1 = vld1q_s32(this->data[0] + i);
        v2 = vld1q_s32(other.data[0] + i);
        r = vaddq_s32(v1, v2);
        vst1q_s32(res.data[0] + i, r);
    }
    for (size_t i = (this->rows * this->cols * this->channels) / 4 * 4;
         i < this->rows * this->cols * this->channels; i++) {
        *(res.data[0] + i) = *(this->data[0] + i) + *(other.data[0] + i);
    }
#else
    for (size_t i = 0; i < this->rows * this->cols * this->channels; i++) {
        *(res.data[0] + i) = *(this->data[0] + i) + *(other.data[0] + i);
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
    for (size_t i = 0; i < this->rows * this->cols * this->channels; i++) {
        *(res.data[0] + i) = *(this->data[0] + i) - *(other.data[0] + i);
    }
    return res;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::operator*(const Mat<Tp> &other) const
{
    if (this->cols != other.rows) {
        cout << "Inconsistent matrices for multiplication. Exit." << endl;
        exit(EXIT_FAILURE);
    }
    Mat<Tp> res(other.rows, other.cols, other.channels);

    util::Timer timer("Matrix Multiplication");
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
    return res;
//    return strassen(*this, other);
}

template<typename Tp>
Mat<Tp> &Mat<Tp>::operator=(const Mat &other)
{
    if (this == &other) {
        return *this;
    }
    if (this->data != nullptr && this->refCntPtr != nullptr) {
        cout << "operator: refCnt: " << *this->refCntPtr << " --> ";
        *this->refCntPtr -= 1;
        cout << *this->refCntPtr << endl;
        if (*this->refCntPtr <= 0) {
            cout << "In operator(): ";
            free(this->data[0]);
            free(this->data);
            cout << "operator=(): Free memory successfully." << endl;
        }
    }
    this->rows = other.rows;
    this->cols = other.cols;
    this->steps = other.steps;
    this->beginRowIndex = other.beginRowIndex;
    this->beginColIndex = other.beginColIndex;
    this->dataRows = other.dataRows;
    this->dataCols = other.dataCols;
    this->channels = other.channels;
    this->data = other.data;
    this->refCntPtr = other.refCntPtr;
    cout << "operator=(): refCnt: " << *this->refCntPtr << " --> ";
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
    for (int k = 0; k < this->channels; k++) {
        for (size_t i = 0; i < other.rows; i++) {
            for (size_t j = 0; j < other.cols; j++) {
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
    if (this->rows != other.rows || this->cols != other.cols || this->channels != other.channels) {
        return false;
    }
    if (*this == other) {
        return false;
    } else {
        return true;
    }
}
//endregion

template<typename Tp>
Mat<Tp> Mat<Tp>::subMat(size_t _rows_, size_t _cols_, size_t _beginRowIndex_, size_t _beginColIndex_) const
{
    if (_rows_ >= this->rows || _cols_ >= this->cols) {
        cout << "Invalid size of sub-matrix. Return [nullptr]." << endl;
        return (Mat<Tp>) nullptr;
    }
    Mat res;
    res.rows = _rows_;
    res.cols = _cols_;
    res.steps = this->steps;
    res.beginRowIndex = this->beginRowIndex + _beginRowIndex_;
    res.beginColIndex = this->beginColIndex + _beginColIndex_;
    res.dataRows = this->dataRows;
    res.dataCols = this->dataCols;
    res.channels = this->channels;
    res.data = this->data;
    res.refCntPtr = this->refCntPtr;
    *(this->refCntPtr) += 1;
    return res;
}

template<typename Tp>
Mat<Tp> Mat<Tp>::roi(size_t rowIndex, size_t colIndex, size_t rows_, size_t cols_) const
{
    cout << "roi(): rowIndex = " << rowIndex << "; colIndex = " << colIndex << endl;
    cout << "roi(): this->data = " << this->data << endl;
    return this->subMat(rows_, cols_, rowIndex, colIndex).clone();
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
    res.channels = this->channels;
    res.data = util::allocate<Tp>(res.rows * res.channels, res.cols);
    for (int k = 0; k < res.channels; k++) {
        for (size_t i = 0; i < this->rows; i++) {
            for (size_t j = 0; j < this->cols; j++) {
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
    cout << "data = " << this->data << endl;
    for (int k = 0; k < this->channels; ++k) {
        cout << "Data type is [" << util::tpToString(this->dataType) << "]. Elements in channel[" << k << "]:" << endl;
        for (size_t j = 0; j < this->cols; j++) {
            cout << "_________";
        }
        cout << "___" << endl;
        for (size_t i = 0; i < this->rows; i++) {
            cout << "| ";
            for (size_t j = 0; j < this->cols; j++) {
                printf("%-9.1f", (float) this->at(i, j, k));
            }
            cout << " |" << endl;
        }
        for (size_t j = 0; j < this->cols; j++) {
            cout << "---------";
        }
        cout << "---" << endl;
    }
    cout << "rows = " << this->rows << ". cols = " << this->cols << endl;
    cout << "steps = " << this->steps << endl;
    cout << "beginRowIndex = " << this->beginRowIndex << ". beginColIndex = " << this->beginColIndex << endl;
    cout << "dataRows = " << this->dataRows << ". dataCols = " << this->dataCols << endl;
    cout << "channels = " << this->channels << endl;
    cout << "refCntPtr = " << this->refCntPtr << ". refCnt = " << *(this->refCntPtr) << endl;

}

template<typename Tp>
void Mat<Tp>::transfer()
{
#if defined(_ENABLE_OMP)
#pragma omp parallel for num_threads((int)thread::hardware_concurrency())
#endif
    for (int k = 0; k < this->channels; k++) {
        for (size_t i = 0; i < this->rows; i++) {
            for (size_t j = 0; j < this->cols; j++) {
                if (j > i) {
                    Tp temp = this->at(i, j, k);
                    this->at(i, j, k) = this->at(j, i, k);
                    this->at(j, i, k) = temp;
                }
            }
        }
    }
}

template<typename Tp>
void Mat<Tp>::writeTo(const char *path) const
{
    FILE *fp;
    if ((fp = fopen(path, "w")) == nullptr) {
        printf("Fail to open or create output file -> %s. Exit.", path);
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k < this->channels; ++k) {
        fputs("Channel[", fp);
        const char channelIndex = k + '0';
        fputs((const char *) (&(channelIndex)), fp);
        fputs("]:\r\n", fp);
        for (size_t i = 0; i < this->rows; i++) {
            for (size_t j = 0; j < this->cols; j++) {
                char *buffer = (char *) malloc(sizeof(char) * 32);
                sprintf(buffer, "%.1f", (float) this->at(i, j, k));
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
void Mat<Tp>::singleThreadMul(const Mat<Tp> &mat_A, const Mat<Tp> &mat_B, Mat<Tp> &mat_C, const size_t _beginRowIndex_,
                              const size_t rowNumToCal)
{
    for (int p = 0; p < mat_A.channels; p++) {
        for (size_t i = 0; i < rowNumToCal; i++) {
            for (size_t k = 0; k < mat_A.cols; k++) {
                Tp temp = mat_A.at(_beginRowIndex_ + i, k);
                for (size_t j = 0; j < mat_B.cols; j++) {
                    mat_C.at(_beginRowIndex_ + i, j, p) += temp * mat_B.at(k, j, p);
                }
            }
        }
    }
}

template<typename Tp>
Mat<Tp> Mat<Tp>::strassen(Mat<float> mat_A, const Mat<Tp> &mat_B)
{
    size_t n = mat_A.rows;
    if (n <= 256) {
        Mat<Tp> res(mat_A.rows, mat_B.cols, mat_A.channels);
        for (size_t i = 0; i < mat_A.rows; i++) {
            for (size_t k = 0; k < mat_A.cols; k++) {
                Tp tmp = mat_A.at(i, k, 0);
                for (size_t j = 0; j < mat_B.cols; j++) {
                    res.at(i, j, 0) += tmp * mat_B.at(k, j, 0);
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

    Mat<Tp> C11 = P4 + P5 - P2 + P6;
    Mat<Tp> C12 = P1 + P2;
    Mat<Tp> C21 = P3 + P4;
    Mat<Tp> C22 = P1 + P5 - P3 - P7;

    return merge(C11, C12, C21, C22);
}

template<typename Tp>
Mat<Tp> Mat<Tp>::merge(const Mat<Tp> &C11, const Mat<Tp> &C12, const Mat<Tp> &C21, const Mat<Tp> &C22)
{
    Mat<Tp> res(C11.rows + C21.rows, C11.cols + C12.cols, C11.channels);
    for (size_t i = 0; i < C11.rows; i++) {
        for (size_t j = 0; j < C11.cols; j++) {
            res.at(i, j, 0) = C11.at(i, j, 0);
            res.at(i, j + C11.cols, 0) = C12.at(i, j, 0);
            res.at(i + C21.rows, j, 0) = C21.at(i, j, 0);
            res.at(i + C21.rows, j + C22.cols, 0) = C22.at(i, j, 0);
        }
    }
    return res;
}
//endregion
#pragma clang diagnostic pop