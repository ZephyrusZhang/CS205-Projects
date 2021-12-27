// #pragma GCC optimize(2)
#pragma GCC optimize(3,"Ofast","inline")

#include <iostream>
#include "mat.hpp"

using namespace std;

int main(int argc, char **argv)
{
    if (argc <= 2)
    {
        cout << "文件的数量不够。退出！" << endl;
        exit(100);
    }

    string ifile1_name = argv[1];
    string ifile2_name = argv[2];

    Matrix mat_A(ifile1_name);
    Matrix mat_B(ifile2_name);
    if (mat_A.getCol() != mat_B.getRow())
    {
        cout << "矩阵A的列数不等于矩阵B的行数，无法做矩阵乘法。退出！" << endl;
        exit(100);
    }

    mat_A.prepareMat();
    mat_B.prepareMat();
    mat_A.fmatMul(mat_B);
    mat_A.dmatMul(mat_B);
    
    return 0;
}