#include "facedetection.hpp"

int main()
{
//    string paths[12] = {"image/airport.jpg", "image/beach.jpg", "image/ChuanShanJia.jpg", "image/face.jpg",
//                        "image/galaxy.jpg", "image/grass.jpg", "image/miku.jpg", "image/Rengoku Kyoujurou.jpg",
//                        "image/sunset.jpg", "image/Tifa.jpg", "image/ysq.jpg", "image/Zephyrus.jpg"};
    string paths[8] = {"image/face.jpg", "image/airport.jpg", "image/Tifa.jpg", "image/beach.jpg",
                       "image/Zephyrus.jpg", "image/galaxy.jpg", "image/ChuanShanJia.jpg", "image/sunset.jpg"};
    cnn::facedetection128x128SimpleConv("image/miku.jpg");
    for (string &path: paths)
        cnn::facedetection128x128SimpleConv(path);
    for (string &path: paths)
        cnn::facedetection128x128UsingSgemm(path);
}