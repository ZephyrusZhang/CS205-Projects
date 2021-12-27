#include <cmath>

#include "mat.hpp"

using namespace std;

int main()
{
#if defined(_ENABLE_NEON)
    cout << "<---------------------------------------------------------ARM---------------------------------------------------------->" <<endl;
#endif

    Mat<float> mat_A(32, 32, 2, {"file/txt/mat-A-32.txt", "file/txt/mat-B-32.txt"});
    Mat<float> mat_B(32, 32, 2, {"file/txt/mat-B-32.txt", "file/txt/mat-A-32.txt"});
    (mat_A * mat_B).print();

#if defined(_ENABLE_NEON)
    cout << "<---------------------------------------------------------------------------------------------------------------------->" <<endl;
#endif
    return 0;
}