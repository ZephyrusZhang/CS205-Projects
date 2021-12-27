#include <cmath>

#include "mat.hpp"

using namespace std;

int main()
{
#if defined(_ENABLE_NEON)
    cout << "<---------------------------------------------------------ARM---------------------------------------------------------->" <<endl;
#endif

    Mat<float> mat_A(32, 32, 2, {"file/txt/mat-A-32.txt", "file/txt/mat-B-32.txt"});
    Mat<float> mat_B = mat_A.subMat(15, 7, 2, 3);
    cout << sizeof(mat_A) << endl;

#if defined(_ENABLE_NEON)
    cout << "<---------------------------------------------------------------------------------------------------------------------->" <<endl;
#endif
    return 0;
}