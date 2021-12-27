#ifndef CNN_FACEDETECTION_FACEDETECTION_DATA_HPP
#define CNN_FACEDETECTION_FACEDETECTION_DATA_HPP

class ConvParam
{
public:
    int pad;
    int stride;
    int kernel_size;
    int channels;
    int kernelsNum;
    float *p_weight;
    float *p_bias;

    [[nodiscard]] bool isValid() const;

    void print() const
    {
        printf("ConvParam: [pad, stride, kernel_size, channels, kernelNum, p_weight, p_bias] = "
               "[%d, %d, %d, %d, %d, %p, %p]\n", pad, stride, kernel_size, channels, kernelsNum, p_weight, p_bias);
    }
};

class FcParam
{
public:
    int in_features;
    int out_features;
    float *p_weight;
    float *p_bias;

    [[nodiscard]] bool isValid() const;
};

extern ConvParam conv_params[3];
extern FcParam fc_params[1];

#endif //CNN_FACEDETECTION_FACEDETECTION_DATA_HPP