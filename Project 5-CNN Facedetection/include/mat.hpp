#ifndef CNN_FACEDETECTION_MAT_HPP
#define CNN_FACEDETECTION_MAT_HPP

class Mat4f
{

public:
    size_t rows{};
    size_t cols{};
    float *data{};

public:
    Mat4f() = default;

    Mat4f(size_t rows_, size_t cols_);

    ~Mat4f();

    void init(size_t rows_, size_t cols_);

    void setnull();

    void transfer();

    static void sgemm(const Mat4f &blobMat, const Mat4f &filterMat, Mat4f &resBlobMat);

    [[nodiscard]] bool isValid() const;

};

#endif //CNN_FACEDETECTION_MAT_HPP