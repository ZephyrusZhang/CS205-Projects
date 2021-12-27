#include "mat.hpp"

using namespace std;

int main()
{
#if defined(_ENABLE_NEON)
    cout << "<---------------------------------------------------------ARM---------------------------------------------------------->" <<endl;
#endif

    Mat<float> mat_A("file/txt/mat-A-2048.txt");
    Mat<float> mat_B("file/txt/mat-B-2048.txt");
    (mat_A * mat_B).writeTo("float-2048-multiplication-ikj.txt");

#if defined(_ENABLE_NEON)
    cout << "<---------------------------------------------------------------------------------------------------------------------->" <<endl;
#endif
    return 0;
}