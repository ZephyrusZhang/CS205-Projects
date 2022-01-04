#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wunused-result"

#include <unistd.h>

#include "facedetection.hpp"

CDataBlob::CDataBlob(const string &srcPath, const Size &size)
{
    Mat img = imread(srcPath);

    ASSERT(!img.empty(), "Fail to read image")
    ASSERT(img.type() == CV_8UC3, "Invalid type of image for face detection")

    resize(img, img, size, 0, 0, INTER_LINEAR);
    cvtColor(img, img, COLOR_BGR2RGB);

    init(img.rows, img.cols, img.channels());

    ASSERT(data != nullptr, "Fail to allocate memory for data")

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            data[0 * rows * cols + i * rows + j] = force_cast(float) (img.at<Vec3b>(i, j)[0]);
            data[1 * rows * cols + i * rows + j] = force_cast(float) (img.at<Vec3b>(i, j)[1]);
            data[2 * rows * cols + i * rows + j] = force_cast(float) (img.at<Vec3b>(i, j)[2]);
        }
    }

    ASSERT(cnn::linearNormalization(data, channels * rows * cols, 0.0f, 1.0f),
           "Fail to normalize CDataBlob data")
}

CDataBlob::~CDataBlob()
{
    setnull();
}

void CDataBlob::init(int _rows, int _cols, int _channels)
{
    ASSERT((_rows >= 1 && _cols >= 1 && _channels >= 1),
           "Invalid size for initializing a data blob")

    setnull();
    rows = _rows;
    cols = _cols;
    channels = _channels;
    data = cnn::allocate(total(), 0.0f);
}

void CDataBlob::set(int rows_, int cols_, int channels_, float *&data_)
{
    rows = rows_;
    cols = cols_;
    channels = channels_;
    data = data_;
    data_ = nullptr;
}

void CDataBlob::setnull()
{
    if (data != nullptr)
    {
        free(data);
        data = nullptr;
    }
    rows = cols = channels = 0;
}

inline float CDataBlob::operator()(int rowIndex, int colIndex, int channelIndex) const
{
    return (rowIndex >= 0 && rowIndex < rows && colIndex >= 0 && colIndex < cols) ?
           data[channelIndex * rows * cols + rowIndex * cols + colIndex] : 0.0f;
}

bool CDataBlob::isValid() const
{
    return rows > 0 && cols > 0 && channels > 0 && data != nullptr;
}

Filter::Filter(const ConvParam &param)
        : weights(param.p_weight), bias(param.p_bias)
{
    ASSERT(param.isValid(), "Parameter [param] is invalid")

    rows = cols = param.kernel_size;
    channels = param.channels;
    kernels = param.kernelsNum;
    padding = param.pad;
    stride = param.stride;

    ASSERT((weights != nullptr && bias != nullptr), "Fail to load weights or bias for Filter")

}

Filter::~Filter()
{
    setnull();
}

void Filter::setnull()
{
    rows = cols = channels = kernels = padding = stride = 0;
}

inline float Filter::operator()(int rowIndex, int colIndex, int channelIndex, int kernelIndex) const
{
    return weights[kernelIndex * rows * cols * channels + channelIndex * rows * cols + rowIndex * cols + colIndex];
}

inline float Filter::operator()(int kernelIndex) const
{
    return bias[kernelIndex];
}

inline int CDataBlob::total() const
{
    return channels * rows * cols;
}

inline bool Filter::isValid() const
{
    return rows > 0 && cols > 0 && channels > 0 && kernels > 0 && weights != nullptr && bias != nullptr;
}

inline float *cnn::allocate(int length, float initialValue)
{
    auto res = static_cast<float *>(aligned_alloc(256, length * sizeof(float)));
#pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        res[i] = initialValue;
    }
    return res;
}

bool cnn::linearNormalization(float *data, int dataLength, float lowerBound, float upperBound)
{
    ASSERT(data != nullptr, "[data] has not been allocated")
    ASSERT(dataLength > 0, "Invalid size of data for normalization")

    float max, min;
    max = min = 0.0f;
#pragma omp parallel for
    for (int i = 0; i < dataLength; ++i)
    {
        max = MAX(max, data[i]);
        min = MIN(min, data[i]);
    }
#pragma omp parallel for
    for (int i = 0; i < dataLength; ++i)
    {
        data[i] = lowerBound + ((upperBound - lowerBound) / (max - min)) * (data[i] - min);
    }

    return true;
}

void cnn::facedetection128x128SimpleConv(const string &srcPath)
{
    TickMeter tick;
    tick.start();

    Filter conv_1_filter(conv_params[0]);
    Filter conv_2_filter(conv_params[1]);
    Filter conv_3_filter(conv_params[2]);
    CDataBlob img(srcPath, Size(128, 128));
    CDataBlob conv_1_blob, conv_2_blob, conv_3_blob;      //CDataBlob before max pooling
    CDataBlob maxPool_1_blob, maxPool_2_blob;             //CDataBlob after max pooling

    /*<------------------Layer 1------------------>*/
    cnn::convReLU(img, conv_1_filter, conv_1_blob);
    cnn::maxPooling(conv_1_blob, maxPool_1_blob);

    /*<------------------Layer 2------------------>*/
    cnn::convReLU(maxPool_1_blob, conv_2_filter, conv_2_blob);
    cnn::maxPooling(conv_2_blob, maxPool_2_blob);

    /*<------------------Layer 3------------------>*/
    cnn::convReLU(maxPool_2_blob, conv_3_filter, conv_3_blob);

    /*<------------------Fully-Connected Layer------------------>*/
    auto conf = new float[2]{0.0f};
    for (int i = 0; i < 2048; ++i)
    {
        conf[0] += fc_params[0].p_weight[i] * conv_3_blob.data[i];
        conf[1] += fc_params[0].p_weight[i + 2048] * conv_3_blob.data[i];
    }
    conf[0] += fc_params[0].p_bias[0];
    conf[1] += fc_params[0].p_bias[1];

    /*<------------------Softmax------------------>*/
    conf[0] = exp(conf[0]) / (exp(conf[0]) + exp(conf[1]));
    conf[1] = exp(conf[1]) / (exp(conf[0]) + exp(conf[1]));


    printf("\033[32mImage <%s>: [Background Confidence = %.6f, ", srcPath.c_str(), conf[0]);
    printf("Human Face Confidence = %.6f] \033[0m", conf[1]);

    tick.stop();
    printf("=> \033[35mTotal time consumption [simple convolution] = %g ms\033[0m\n",
           tick.getTimeMilli());
}

void cnn::facedetection128x128UsingSgemm(const string &srcPath)
{
    TickMeter tick;
    tick.start();

    Filter conv_1_filter(conv_params[0]);
    Filter conv_2_filter(conv_params[1]);
    Filter conv_3_filter(conv_params[2]);
    CDataBlob img(srcPath, Size(128, 128));
    CDataBlob conv_1_blob, conv_2_blob, conv_3_blob;    //CDataBlob before max pooling
    CDataBlob maxPool_1_blob, maxPool_2_blob;           //CDataBlob after max pooling

    /*<------------------Layer 1------------------>*/
    cnn::im2colConvReLU(img, conv_1_filter, conv_1_blob);
    cnn::maxPooling(conv_1_blob, maxPool_1_blob);

    /*<------------------Layer 2------------------>*/
    cnn::im2colConvReLU(maxPool_1_blob, conv_2_filter, conv_2_blob);
    cnn::maxPooling(conv_2_blob, maxPool_2_blob);

    /*<------------------Layer 3------------------>*/
    cnn::im2colConvReLU(maxPool_2_blob, conv_3_filter, conv_3_blob);

    /*<------------------Fully-Connected Layer------------------>*/
    auto conf = new float[2]{0.0f};
    for (int i = 0; i < 2048; ++i)
    {
        conf[0] += fc_params[0].p_weight[i] * conv_3_blob.data[i];
        conf[1] += fc_params[0].p_weight[i + 2048] * conv_3_blob.data[i];
    }
    conf[0] += fc_params[0].p_bias[0];
    conf[1] += fc_params[0].p_bias[1];

    /*<------------------Softmax------------------>*/
    conf[0] = exp(conf[0]) / (exp(conf[0]) + exp(conf[1]));
    conf[1] = exp(conf[1]) / (exp(conf[0]) + exp(conf[1]));


    printf("\033[32mImage <%s>: [Background Confidence = %.6f, ", srcPath.c_str(), conf[0]);
    printf("Human Face Confidence = %.6f] \033[0m", conf[1]);

    tick.stop();
    printf("=> \033[35mTotal time consumption using [matrix sgemm] = %g ms\033[0m\n",
           tick.getTimeMilli());
}

bool cnn::convReLU(CDataBlob &blob, Filter &filter, CDataBlob &resBlob)
{
    ASSERT((blob.isValid() && filter.isValid()), "Invalid input blob or filter")
    ASSERT((blob.channels == filter.channels),
           "Inconsistent channels for input CDataBlob and Filter")

    resBlob.setnull();

    int out_rows = floor((blob.rows + 2 * filter.padding - filter.rows) / filter.stride + 1);
    int out_cols = floor((blob.cols + 2 * filter.padding - filter.cols) / filter.stride + 1);
    resBlob.init(out_rows, out_cols, filter.kernels);

    /*---------------------------Convolution---------------------------*/
    size_t index = 0;
    for (int k = 0; k < filter.kernels; ++k)
    {
        for (int i = -filter.padding, x = 0; x < resBlob.rows; i += filter.stride, ++x)
        {
            for (int j = -filter.padding, y = 0; y < resBlob.cols; j += filter.stride, ++y)
            {
                for (int c = 0; c < filter.channels; ++c)
                {
                    resBlob.data[index] +=
                            blob(i + 0, j + 0, c) * filter(0, 0, c, k) +
                            blob(i + 0, j + 1, c) * filter(0, 1, c, k) +
                            blob(i + 0, j + 2, c) * filter(0, 2, c, k) +
                            blob(i + 1, j + 0, c) * filter(1, 0, c, k) +
                            blob(i + 1, j + 1, c) * filter(1, 1, c, k) +
                            blob(i + 1, j + 2, c) * filter(1, 2, c, k) +
                            blob(i + 2, j + 0, c) * filter(2, 0, c, k) +
                            blob(i + 2, j + 1, c) * filter(2, 1, c, k) +
                            blob(i + 2, j + 2, c) * filter(2, 2, c, k);
                }
                resBlob.data[index++] += filter(k);
            }
        }
    }
    /*-----------------------------------------------------------------*/

    /*--------------------------ReLU Function--------------------------*/
#pragma omp parallel for
    for (int i = 0; i < resBlob.channels * resBlob.rows * resBlob.cols; ++i)
    {
        resBlob.data[i] = 0.0f + force_cast(float) (resBlob.data[i] >= 0) * resBlob.data[i];
    }
    /*-----------------------------------------------------------------*/

    ONLY4DEBUG(
            printf("\033[34mConvBNReLU: Batch<%dx%dx%d> * Filter[%d, %d, <%dx%dx%dx%d>] => Batch<%dx%dx%d>\033[0m\n",
                   blob.channels, blob.rows, blob.cols, filter.padding, filter.stride,
                   filter.kernels, filter.channels, filter.rows, filter.cols, resBlob.channels,
                   resBlob.rows, resBlob.cols);)

    return true;
}

bool cnn::maxPooling(CDataBlob &src, CDataBlob &res)
{
    ASSERT(src.isValid(), "Parameter [src] is invalid")

    res.setnull();
    res.init(src.rows / 2, src.cols / 2, src.channels);

    for (int k = 0; k < src.channels; ++k)
    {
        for (int i = 0, x = 0; i < src.rows; i += 2, ++x)
        {
            for (int j = 0, y = 0; j < src.cols; j += 2, ++y)
            {
                float max = src.data[k * src.rows * src.cols + i * src.cols + j];
                if (src.data[k * src.rows * src.cols + i * src.cols + (j + 1)] > max)
                    max = src.data[k * src.rows * src.cols + i * src.cols + (j + 1)];
                if (src.data[k * src.rows * src.cols + (i + 1) * src.cols + j] > max)
                    max = src.data[k * src.rows * src.cols + (i + 1) * src.cols + j];
                if (src.data[k * src.rows * src.cols + (i + 1) * src.cols + (j + 1)] > max)
                    max = src.data[k * src.rows * src.cols + (i + 1) * src.cols + (j + 1)];
                res.data[k * res.rows * res.cols + x * res.cols + y] = max;
            }
        }
    }

    ONLY4DEBUG(printf("\033[34mMaxPooling Downsampled: <%dx%dx%d> => <%dx%dx%d>\033[0m\n",
                      src.channels, src.rows, src.cols, res.channels, res.rows, res.cols);)
    return true;
}

bool cnn::im2colConvReLU(const CDataBlob &blob, const Filter &filter, CDataBlob &resBlob)
{
    ASSERT(blob.isValid() || filter.isValid(), "Invalid blob or filter")
    ASSERT(blob.channels == filter.channels, "Inconsistent channels of blob and filter")

    /*<---------------------Filter Part--------------------->*/
    Mat4f filterMat(filter.kernels, filter.rows * filter.cols * filter.channels);
    memcpy(filterMat.data, filter.weights, sizeof(float) * filterMat.rows * filterMat.cols);
    filterMat.transfer();

    /*<--------------------CDataBlob Part-------------------->*/
    size_t out_rows = floor((blob.rows + 2 * filter.padding - filter.rows) / filter.stride + 1);
    size_t out_cols = floor((blob.cols + 2 * filter.padding - filter.cols) / filter.stride + 1);
    Mat4f blobMat(out_rows * out_cols, filterMat.rows);
    size_t index = 0;
    for (int i = -filter.padding, x = 0; x < out_rows; i += filter.stride, ++x)
    {
        for (int j = -filter.padding, y = 0; y < out_cols; j += filter.stride, ++y)
        {
            for (int c = 0; c < filter.channels; ++c)
            {
                blobMat.data[index++] = blob(i + 0, j + 0, c);
                blobMat.data[index++] = blob(i + 0, j + 1, c);
                blobMat.data[index++] = blob(i + 0, j + 2, c);
                blobMat.data[index++] = blob(i + 1, j + 0, c);
                blobMat.data[index++] = blob(i + 1, j + 1, c);
                blobMat.data[index++] = blob(i + 1, j + 2, c);
                blobMat.data[index++] = blob(i + 2, j + 0, c);
                blobMat.data[index++] = blob(i + 2, j + 1, c);
                blobMat.data[index++] = blob(i + 2, j + 2, c);
            }
        }
    }

    Mat4f resMat;
    Mat4f::sgemm(blobMat, filterMat, resMat);

    for (size_t j = 0; j < resMat.cols; ++j)
    {
        for (size_t i = 0; i < resMat.rows; ++i)
        {
            resMat.data[i * resMat.cols + j] += filter.bias[j];
        }
    }

    /*--------------------------ReLU Function--------------------------*/
#pragma omp parallel for
    for (size_t i = 0; i < resMat.rows * resMat.cols; ++i)
    {
        resMat.data[i] = force_cast(float) (resMat.data[i] > 0) * resMat.data[i] + 0.0f;
    }

    resMat.transfer();

    resBlob.set(force_cast(int) out_rows, force_cast(int) out_cols, filter.kernels, resMat.data);

    ONLY4DEBUG(
            printf("\033[34mConvBNReLU: Batch<%dx%dx%d> * Filter[%d, %d, <%dx%dx%dx%d>] => Batch<%dx%dx%d>\033[0m\n",
                   blob.channels, blob.rows, blob.cols, filter.padding, filter.stride,
                   filter.kernels, filter.channels, filter.rows, filter.cols, resBlob.channels,
                   resBlob.rows, resBlob.cols);
    )

    return true;
}

#pragma clang diagnostic pop