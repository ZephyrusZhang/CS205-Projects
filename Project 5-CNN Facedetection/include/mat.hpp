#ifndef CNN_FACEDETECTION_MAT_HPP
#define CNN_FACEDETECTION_MAT_HPP

class CMat
{

public:
    size_t rows{};
    size_t cols{};
    float *data{};

public:
    CMat() = default;

    CMat(size_t rows_, size_t cols_);

    ~CMat();

    void init(size_t rows_, size_t cols_);

    void setnull();

    void transfer();

    static void sgemm(const CMat &blobMat, const CMat &filterMat, CMat &resBlobMat);

    [[nodiscard]] bool isValid() const;

};

#endif //CNN_FACEDETECTION_MAT_HPP