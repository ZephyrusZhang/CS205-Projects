#ifndef CNN_FACEDETECTION_MAT_HPP
#define CNN_FACEDETECTION_MAT_HPP

class Matf32
{

public:
    size_t rows{};
    size_t cols{};
    float *data{};

public:
    Matf32() = default;

    Matf32(size_t rows_, size_t cols_);

    ~Matf32();

    void init(size_t rows_, size_t cols_);

    void setnull();

    void transfer();

    static void sgemm(const Matf32 &blobMat, const Matf32 &filterMat, Matf32 &resBlobMat);

    [[nodiscard]] bool isValid() const;

};

#endif //CNN_FACEDETECTION_MAT_HPP