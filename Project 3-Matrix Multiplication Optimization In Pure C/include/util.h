#include "matrix.h"

#ifndef PROJECT_3_UTIL_H
#define PROJECT_3_UTIL_H

typedef enum BOOLEAN {
    FALSE, TRUE
} bool;

/**
 * @brief <p>Copy v1_ and v2_ 's data to v1 and v2.</p>
 * @param v1_ The pointer of a certain row in matrix.
 * @param v2_ The pointer of a certain column's first element in matrix.
 * @param v1 Aligned dynamic array used in SIMD operation(row)
 * @param v2 Aligned dynamic array used in SIMD operation(column)
 * @param n Size of matrix.
 */
void alignedCopy(const float *v1_, const float *v2_, float *v1, float *v2, size_t n);

/**
 * @brief <p>Allocate memory in heap for an 2D array with specified
 *           size of row and column and then initialize the data.</p>
 */
float **allocate2DArray(size_t row, size_t col);

void free2DArray(float **arr);

__attribute__((unused)) void print2DArray(float **arr, size_t row, size_t col);

/**
 * @brief <p>Get file's path according to its name(the file should be
 *           stored in folder txt).</p>
 */
char *getFilePath(const char *root, char *filename);

/**
 * @brief <p>Get path of the output file.
 *         eg. "out-2048x2048.txt"(stored in bin)</p>
 */
char *getOuFileName(size_t row, size_t col, Method method);

/**
 * @brief <p>Write the data of arr into output file.</p>
 */
void writeMatrix(float **arr, size_t row, size_t col, char *outFileName);

/**
 * @brief <p>This struct is used to passes data and stores the result in multi-threads.</p>
 *        <p>mat_A and mat_B is the two matrices which are multiplied together.</p>
 *        <p>resMat is the result of matrices multiplication</p>
 *        <p>beginRowIndex is the index of the first row's index that is calculated in
 *        current thread.</p>
 *        <p>rowNumToCal is the row's number of the current thread to calculate.</p>
 */
typedef struct RESULT_SET {
    MATRIX *resMat;
    MATRIX *mat_A;
    MATRIX *mat_B;
    size_t beginRowIndex;
    size_t rowNumToCal;
} ResultSet;

/**
 * @brief <p>Allocate memory in heap for struct ResultSet</p>
 */
ResultSet *prepareResultSet(MATRIX *resMat, MATRIX *mat_A, MATRIX *mat_B, size_t beginRowIndex, size_t rowNumToCal);

__attribute__((unused)) float *convertTo1DArray(float **data, size_t row, size_t col);

#endif //PROJECT_3_UTIL_H
