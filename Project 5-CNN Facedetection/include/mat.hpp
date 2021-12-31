#ifndef CNN_FACEDETECTION_MAT_HPP
#define CNN_FACEDETECTION_MAT_HPP

class Mat8F
{

public:
    size_t rows{};
    size_t cols{};
    float *data{};

public:
    Mat8F() = default;

    Mat8F(size_t rows_, size_t cols_);

    ~Mat8F();

    void init(size_t rows_, size_t cols_);

    void setnull();

    void transfer();

    static void sgemm(const Mat8F &blobMat, const Mat8F &filterMat, Mat8F &resBlobMat);

    [[nodiscard]] bool isValid() const;

};

#endif //CNN_FACEDETECTION_MAT_HPP