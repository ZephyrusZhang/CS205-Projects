#include "mat.hpp"

using namespace std;

int main()
{
#if defined(_ENABLE_NEON)
    cout << "<---------------------------------------------------------ARM---------------------------------------------------------->" <<endl;
#endif

    Mat<float> mat_A(1, "file/txt/mat-A-32.txt");
    cout << "main(): mat_A.getData() = " << mat_A.getData() << endl;
    Mat<float> mat_C = mat_A + mat_A;
    mat_C.print();

#if defined(_ENABLE_NEON)
    cout << "<---------------------------------------------------------------------------------------------------------------------->" <<endl;
#endif
    return 0;
}