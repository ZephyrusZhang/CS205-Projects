#ifndef PROJECT_4_MAT_HPP
#define PROJECT_4_MAT_HPP

#include <iostream>
#include <chrono>
#include <fstream>

using namespace std;
using namespace chrono;

template<typename Tp>
class Mat {

public:
    //region Constructor and Destructor
    Mat();

    Mat(const Mat<Tp> &);

    Mat(size_t rows_, size_t cols_);

    explicit Mat(const string &filePath_);

    Mat(size_t rows_, size_t cols_, size_t steps_, size_t beginRowIndex_, size_t beginColIndex_, size_t dataRows_,
        size_t dataCols_, Tp **src, int *refCntPtr_);

    ~Mat();
    //endregion

    //region Operator Overload Functions
    Tp &operator()(size_t i, size_t j);

    Mat<Tp> operator+(const Mat &);

    Mat<Tp> operator-(const Mat &);

    Mat<Tp> operator*(Mat &);

    bool operator==(const Mat &) const;

    bool operator!=(const Mat &) const;

    Mat<Tp> &operator=(const Mat &);

    //endregion

    //region Auxiliary Functions
    static Mat<Tp> txtRead(const string &);

    Mat<Tp> subMat(size_t _rows_, size_t _cols_, size_t _beginRowIndex_, size_t _beginColIndex_);

    Tp &at(size_t i, size_t j) const;

    Tp *rowPtr(size_t) const;

    Tp *colPtr(size_t) const;

    virtual void print();

    void transfer();

    Mat<Tp> clone() const;

    Mat<Tp> roi(size_t rowIndex, size_t colIndex, size_t rows_, size_t cols_);
    //endregion

    //region ARM NEON
#if defined(_ENABLE_NEON)
    static Mat<Tp> mul_neon(Mat &, Mat &);
#endif
    //endregion

    //region Getter & Setter
    [[nodiscard]] size_t

    getRows() const;

    [[nodiscard]] size_t getCols() const;

    [[nodiscard]] size_t getSteps() const;

    [[nodiscard]] size_t getBeginRowIndex() const;

    [[nodiscard]] size_t getBeginColIndex() const;

    Tp **getData() const;

    [[nodiscard]] int *getRefCntPtr() const;

    void setRefCntPtr(int *ptr);

    void setRows(size_t _rows_);

    void setCols(size_t _cols_);

    void setSteps(size_t _steps_);

    void setBeginRowIndex(size_t _beginRowIndex_);

    void setBeginColIndex(size_t _beginColIndex_);

    void setDataRows(size_t _dataRows_);

    void setDataCols(size_t _dataCols_);

    void setData(Tp **_data_);
    //endregion

protected:
    size_t rows{};
    size_t cols{};
    size_t steps{};
    size_t beginRowIndex{};
    size_t beginColIndex{};
    size_t dataRows{};
    size_t dataCols{};
    Tp **data{};
    int *refCntPtr{};

    static void singleThreadMul(const Mat &, const Mat &, Mat &, size_t, size_t);

//    static Mat<Tp> strassen(Mat &, Mat &);
//
//    static Mat<Tp> merge(Mat &, Mat &, Mat &, Mat &);

};

template<typename Tp>
class Mat_ : public Mat<Tp> {

public:
    Mat_();

    Mat_(const string &pathR, const string &pathG, const string &pathB);

    Mat_(size_t rows_, size_t cols_, int channels_);

    ~Mat_();

    Mat_<Tp> operator*(const Mat_<Tp> &);

    void print();

    Tp &at_R(size_t i, size_t j) const;

    Tp &at_G(size_t i, size_t j) const;

    Tp &at_B(size_t i, size_t j) const;

    //region Getter & Setter
    [[nodiscard]] int getChannels() const;

    void setChannels(int channels_);

    Tp **getRed() const;

    void setRed(Tp **red_);

    Tp **getGreen() const;

    void setGreen(Tp **green_);

    Tp **getBlue() const;

    void setBlue(Tp **blue_);

    [[nodiscard]] int *getRefCntPtrR() const;

    void setRefCntPtrR(int *refCntPtrR);

    [[nodiscard]] int *getRefCntPtrG() const;

    void setRefCntPtrG(int *refCntPtrG);

    [[nodiscard]] int *getRefCntPtrB() const;

    void setRefCntPtrB(int *refCntPtrB);
    //endregion

private:
    int channels{};
    Tp **red{};
    Tp **green{};
    Tp **blue{};
    int *refCntPtr_R{};
    int *refCntPtr_G{};
    int *refCntPtr_B{};

};

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

template<typename Tp>
int Mat_<Tp>::getChannels() const
{
    return channels;
}

template<typename Tp>
void Mat_<Tp>::setChannels(int channels_)
{
    Mat_::channels = channels_;
}

template<typename Tp>
Tp **Mat_<Tp>::getRed() const
{
    return red;
}

template<typename Tp>
void Mat_<Tp>::setRed(Tp **red_)
{
    Mat_::red = red_;
}

template<typename Tp>
Tp **Mat_<Tp>::getGreen() const
{
    return green;
}

template<typename Tp>
void Mat_<Tp>::setGreen(Tp **green_)
{
    Mat_::green = green_;
}

template<typename Tp>
Tp **Mat_<Tp>::getBlue() const
{
    return blue;
}

template<typename Tp>
void Mat_<Tp>::setBlue(Tp **blue_)
{
    Mat_::blue = blue_;
}

template<typename Tp>
int *Mat_<Tp>::getRefCntPtrR() const
{
    return refCntPtr_R;
}

template<typename Tp>
void Mat_<Tp>::setRefCntPtrR(int *refCntPtrR)
{
    refCntPtr_R = refCntPtrR;
}

template<typename Tp>
int *Mat_<Tp>::getRefCntPtrG() const
{
    return refCntPtr_G;
}

template<typename Tp>
void Mat_<Tp>::setRefCntPtrG(int *refCntPtrG)
{
    refCntPtr_G = refCntPtrG;
}

template<typename Tp>
int *Mat_<Tp>::getRefCntPtrB() const
{
    return refCntPtr_B;
}

template<typename Tp>
void Mat_<Tp>::setRefCntPtrB(int *refCntPtrB)
{
    refCntPtr_B = refCntPtrB;
}
//endregion

//region namespace util
namespace util {

    bool is2ToNPower(size_t dest);

    template<typename Tp>
    Tp **allocate(size_t rows, size_t cols);

    template<typename Tp>
    void read(const string &path, Tp **dest, size_t rows, size_t cols);

    template<typename Tp>
    Tp **array_mul(Tp **data1, Tp **data2, size_t rows, size_t cols, size_t dataRows, size_t dataCols,
                   size_t dataBeginRowIndex, size_t dataBeginColIndex, size_t steps);

    template<typename Tp>
    void array_print(Tp **array, size_t rows, size_t cols);

    template<typename Tp>
    void array_write_cpp(Tp **array, size_t rows, size_t cols, const string &outputPath);

    template<typename Tp>
    void array_write_c(Tp **array, size_t rows, size_t cols, const char *outputPath);

    size_t getRowsOfTXTFile(const string &filePath);

    size_t getColsOfTXTFile(const string &filePath);

    //region class Timer
    class Timer {

    private:
        string procedure;

        steady_clock::time_point _start;
        steady_clock::time_point _end;

    public:
        explicit Timer(const string &);

        ~Timer();

    };

    Timer::Timer(const string &name)
    {
        this->procedure = name;
        this->_start = steady_clock::now();
    }

    Timer::~Timer()
    {
        this->_end = steady_clock::now();
        duration<double> consumption = this->_end - this->_start;
        cout << "Procedure <<" << this->procedure << ">> consumed " << consumption.count() * 1000 << "ms." << endl;
    }

    void error();
    //endregion
}

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
void util::read(const string &path, Tp **dest, size_t rows, size_t cols)
{
    ifstream ifs;
    ifs.open(path);
    if (!ifs.is_open()) {
        cout << "Fail to read file <<" << path << ">>. Exit." << endl;
        exit(EXIT_FAILURE);
    }
    char buffer[32] = {};
    size_t i = 0, j = 0;
    while (ifs >> buffer) {
        if (i == rows && j == cols) {
            break;
        }
        dest[i][j] = (Tp) stof(buffer);
        if (++j >= cols) {
            i++;
            j = 0;
        }
    }
    ifs.close();
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
        exit(EXIT_FAILURE);
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

template<typename Tp>
Tp **util::array_mul(Tp **data1, Tp **data2, size_t rows, size_t cols, size_t dataRows, size_t dataCols,
                     size_t dataBeginRowIndex, size_t dataBeginColIndex, size_t steps)
{
    Tp **res = allocate<Tp>(rows, cols);

}

void util::error()
{
    cout << "Error" << endl;
}
//endregion

#include "../src/mat.tpp"

#endif //PROJECT_4_MAT_HPP