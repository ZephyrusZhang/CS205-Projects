#ifndef CNN_FACEDETECTION_FACEDETECTION_HPP
#define CNN_FACEDETECTION_FACEDETECTION_HPP

#include <opencv2/opencv.hpp>

#include "macro.hpp"
#include "facedetection-data.hpp"
#include "mat.hpp"

using namespace std;
using namespace cv;

class CDataBlob
{
public:
    int rows{};
    int cols{};
    int channels{};
    float *data{};

    CDataBlob() = default;

    CDataBlob(const string &srcPath, const Size &size);

    ~CDataBlob();

    void init(int _rows, int _cols, int _channels);

    void set(int rows_, int cols_, int channels_, float *&data_);

    void setnull();

    float operator()(int rowIndex, int colIndex, int channelIndex) const;

    [[nodiscard]] int total() const;

    [[nodiscard]] bool isValid() const;
};

class Filter
{
public:
    int rows{};
    int cols{};
    int channels{};
    int kernels{};
    int padding{};
    int stride{};
    float *const weights{};
    float *const bias{};

    Filter() = default;

    explicit Filter(const ConvParam &param);

    ~Filter();

    void setnull();

    float operator()(int rowIndex, int colIndex, int channelIndex, int kernelIndex) const;

    float operator()(int kernelIndex) const;

    [[nodiscard]] bool isValid() const;
};

namespace cnn
{
    float *allocate(int length, float initialValue);

    bool linearNormalization(float *data, int dataLength, float lowerBound, float upperBound);

    void facedetection128x128SimpleConv(const string &srcPath);

    void facedetection128x128UsingSgemm(const string &srcPath);

    bool convReLU(CDataBlob &blob, Filter &filter, CDataBlob &resBlob);

    bool maxPooling(CDataBlob &src, CDataBlob &res);

    bool im2colConvReLU(const CDataBlob &blob, const Filter &filter, CDataBlob &resBlob);
}

#endif //CNN_FACEDETECTION_FACEDETECTION_HPP