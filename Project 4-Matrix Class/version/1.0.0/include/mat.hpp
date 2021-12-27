#ifndef PROJECT_4_MAT_HPP
#define PROJECT_4_MAT_HPP

#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

namespace util {

    bool is2ToNPower(size_t dest);

    enum SourceFileType {
        TXT, IMAGE
    };

    template<typename Tp>
    Tp **allocate(size_t rows, size_t cols);

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

template<typename Tp>
class Mat {

public:
    //region Constructor and Destructor
    Mat();

    Mat(const Mat<Tp> &);

    Mat(size_t rows_, size_t cols_);

    Mat(int channels_, const string &filePath_);

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

    Tp * rowPtr(size_t) const;

    Tp *colPtr(size_t) const;

    void print();

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

    [[nodiscard]] int getChannels() const;

    Tp **getData() const;

    [[nodiscard]] int *getRefCntPtr() const;

    void setRefCntPtr(int *ptr);

    void setRows(size_t _rows_);

    void setCols(size_t _cols_);

    void setSteps(size_t _steps_);

    void setBeginRowIndex(size_t _beginRowIndex_);

    void setBeginColIndex(size_t _beginColIndex_);

    void setChannels(int _channels_);

    void setDataRows(size_t _dataRows_);

    void setDataCols(size_t _dataCols_);

    void setData(Tp **_data_);
    //endregion

private:
    size_t rows{};
    size_t cols{};
    size_t steps{};
    size_t beginRowIndex{};
    size_t beginColIndex{};
    int channels{};
    size_t dataRows{};
    size_t dataCols{};
    Tp **data{};
    int *refCntPtr{};

    static void singleThreadMul(const Mat &, const Mat &, Mat &, size_t, size_t);

    static Mat<Tp> strassen(Mat &, Mat &);

    static Mat<Tp> merge(Mat &, Mat &, Mat &, Mat &);

private:

};

template<typename Tp>
class Mat_ : public Mat<Tp> {

};

#include "../src/mat.tpp"

#endif //PROJECT_4_MAT_HPP