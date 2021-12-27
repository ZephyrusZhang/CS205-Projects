#include <iostream>
#include <fstream>
#include <thread>

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

//region namespace util
bool util::is2ToNPower(size_t dest)
{
    if (dest & (dest - 1)) {
        return false;
    } else {
        return true;
    }
}

template<typename Tp>
Tp **util::allocate(size_t rows, size_t cols)
{
    auto *oneDimensionArray = static_cast<Tp *>(aligned_alloc(512, rows * cols * sizeof(Tp) + sizeof(int)));
    for (int i = 0; i < rows * cols; i++) {
        oneDimensionArray[i] = {};
    }
    auto **res = static_cast<Tp **>(malloc(sizeof(Tp *) * rows));
    for (size_t i = 0; i < rows; i++) {
        res[i] = oneDimensionArray + i * cols;
    }
    return res;
}

template<typename Tp>
void util::array_print(Tp **array, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            printf("%-9.1f", (float) array[i][j]);
        }
        cout << endl;
    }
}

template<typename Tp>
void util::array_write_cpp(Tp **array, size_t rows, size_t cols, const string &outputPath)
{
    ofstream ofs;
    ofs.open(outputPath, ios::out);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            ofs << array[i][j] << " ";
        }
        ofs << endl;
    }
    ofs.close();
}

template<typename Tp>
void util::array_write_c(Tp **array, size_t rows, size_t cols, const char *outputPath)
{
    FILE *fp;
    if ((fp = fopen(outputPath, "w")) == nullptr) {
        printf("Fail to open or create output file -> %s. Exit.", outputPath);
        exit(0);
    }
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            char *buffer = (char *) malloc(sizeof(char) * 32);
            sprintf(buffer, "%.1f", array[i][j]);
            fputs(buffer, fp);
            if (j < cols - 1) fputs(" ", fp);
            else fputs("\r\n", fp);
            free(buffer);
        }
    }
    fclose(fp);
}

size_t util::getRowsOfTXTFile(const string &filePath)
{
    ifstream ifs;
    ifs.open(filePath, ios::in);
    if (!ifs.is_open()) {
        return -1;
    }

    size_t rows = 0;
    string buffer;
    while (getline(ifs, buffer)) {
        rows++;
    }
    ifs.close();

    return rows;
}

size_t util::getColsOfTXTFile(const string &filePath)
{
    ifstream ifs;
    ifs.open(filePath, ios::in);
    if (!ifs.is_open()) {
        return -1;
    }

    size_t cols = 0;
    string buffer;
    if (getline(ifs, buffer)) {
        for (char i: buffer) {
            if (i == 32) {
                cols++;
            }
        }
        cols++;
    }
    ifs.close();

    return cols;
}

void util::error()
{
    cout << "Error" << endl;
}
//endregion

//region Constructor and Destructor
template<typename Tp>
Mat<Tp>::Mat()
        : rows(0), cols(0), steps(0), beginRowIndex(0), beginColIndex(0), channels(0),
          dataRows(0), dataCols(0), data(nullptr), refCntPtr(nullptr)
{}

template<typename Tp>
Mat<Tp>::Mat(const Mat<Tp> &other)
        : rows(other.rows), cols(other.cols), steps(other.steps), beginRowIndex(other.beginRowIndex),
          beginColIndex(other.beginColIndex), channels(other.channels), dataRows(other.dataRows),
          dataCols(other.dataCols), data(other.data), refCntPtr(other.refCntPtr)
{
    *(this->refCntPtr) += 1;
}

template<typename Tp>
Mat<Tp>::Mat(size_t rows_, size_t cols_)
        : rows(rows_), cols(cols_), steps(cols_), beginRowIndex(0), beginColIndex(0), channels(1),
          dataRows(rows_), dataCols(cols_)
{
    this->data = util::allocate<Tp>(rows_, cols_);
    setRefCntPtr((int *) (&this->data[this->dataRows - 1][this->dataCols]));
    *this->getRefCntPtr() = 1;
}

template<typename Tp>
Mat<Tp>::Mat(int channels_, const string &filePath_)
        : channels(channels_), beginRowIndex(0), beginColIndex(0)
{
    this->rows = util::getRowsOfTXTFile(filePath_);
    this->cols = util::getColsOfTXTFile(filePath_);
    this->dataRows = this->rows;
    this->dataCols = this->cols;
    this->steps = this->cols;

    ifstream ifs;
    ifs.open(filePath_);
    if (!ifs.is_open()) {
        cout << "Fail to read file <<" << filePath_ << ">>. Exit." << endl;
        exit((0));
    }
    Tp **resData = util::allocate<Tp>(this->rows, this->cols);
    char buffer[32] = {};
    size_t i = 0, j = 0;
    while (ifs >> buffer) {
        if (i == this->rows && j == this->cols) {
            break;
        }
        resData[i][j] = (Tp) stof(buffer);
        if (++j >= this->cols) {
            i++;
            j = 0;
        }
    }
    this->data = resData;
    ifs.close();

    setRefCntPtr((int *) (&this->data[this->dataRows - 1][this->dataCols]));
    *this->getRefCntPtr() = 1;
}

template<typename Tp>
Mat<Tp>::~Mat()
{
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
        exit(0);
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
        exit(0);
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
        exit(0);
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
    this->channels = other.channels;
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
    cout << "steps = " << this->steps << ". channels = " << this->channels << endl;
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
    res.channels = this->channels;
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
    res.channels = this->channels;
    res.dataRows = this->dataRows;
    res.dataCols = this->dataCols;
    res.data = this->data;
    res.refCntPtr = this->refCntPtr;
    *(this->refCntPtr) += 1;
    cout << "subMat(): res.data = " << res.data << endl;
    cout << "subMat(): res.beginRowIndex = " << res.beginRowIndex << "; this->beginColIndex = " << res.beginColIndex << endl;
    return res;
}

template<typename Tp>
Tp * Mat<Tp>::rowPtr(size_t rowIndex) const
{
    return &(this->at(rowIndex, 0));
}

template<typename Tp>
Tp *Mat<Tp>::colPtr(size_t colIndex) const
{
    return &(this->at(0, colIndex));
}
//endregion

//region Getter & Setter
template<typename Tp>
size_t Mat<Tp>::getRows() const
{
    return rows;
}

template<typename Tp>
size_t Mat<Tp>::getCols() const
{
    return cols;
}

template<typename Tp>
size_t Mat<Tp>::getSteps() const
{
    return steps;
}

template<typename Tp>
size_t Mat<Tp>::getBeginRowIndex() const
{
    return beginRowIndex;
}

template<typename Tp>
size_t Mat<Tp>::getBeginColIndex() const
{
    return beginColIndex;
}

template<typename Tp>
int Mat<Tp>::getChannels() const
{
    return channels;
}

template<typename Tp>
Tp **Mat<Tp>::getData() const
{
    return data;
}

template<typename Tp>
int *Mat<Tp>::getRefCntPtr() const
{
    return refCntPtr;
}

template<typename Tp>
void Mat<Tp>::setRefCntPtr(int *ptr)
{
    Mat::refCntPtr = ptr;
}

template<typename Tp>
void Mat<Tp>::setRows(size_t _rows_)
{
    Mat::rows = _rows_;
}

template<typename Tp>
void Mat<Tp>::setCols(size_t _cols_)
{
    Mat::cols = _cols_;
}

template<typename Tp>
void Mat<Tp>::setSteps(size_t _steps_)
{
    Mat::steps = _steps_;
}

template<typename Tp>
void Mat<Tp>::setBeginRowIndex(size_t _beginRowIndex_)
{
    Mat::beginRowIndex = _beginRowIndex_;
}

template<typename Tp>
void Mat<Tp>::setBeginColIndex(size_t _beginColIndex_)
{
    Mat::beginColIndex = _beginColIndex_;
}

template<typename Tp>
void Mat<Tp>::setChannels(int _channels_)
{
    Mat::channels = _channels_;
}

template<typename Tp>
void Mat<Tp>::setDataRows(size_t _dataRows_)
{
    Mat::dataRows = _dataRows_;
}

template<typename Tp>
void Mat<Tp>::setDataCols(size_t _dataCols_)
{
    Mat::dataCols = _dataCols_;
}

template<typename Tp>
void Mat<Tp>::setData(Tp **_data_)
{
    Mat::data = _data_;
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