#include <cmath>
#include "mat.hpp"

using namespace std;

int main()
{
#if defined(_ENABLE_NEON)
    cout << "<---------------------------------------------------------ARM---------------------------------------------------------->" <<endl;
#endif

    Mats<float> mats(32, 32, 10);

#if defined(_ENABLE_NEON)
    cout << "<---------------------------------------------------------------------------------------------------------------------->" <<endl;
#endif
    return 0;
}