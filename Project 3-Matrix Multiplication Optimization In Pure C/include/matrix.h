#ifndef PROJECT_3_MATRIX_H
#define PROJECT_3_MATRIX_H

typedef struct Matrix {
    float **data;
    size_t row;
    size_t col;
} MATRIX;

typedef enum METHOD {
    IJK, IKJ, STRASSEN, MULTI_THREADS, OPEN_BLAS_dotProduct, OPENMP, OPEN_BLAS
} Method;

/**
 * @brief <p>Create matrix's instantiation from the specific file,
 *        and then store its pointer into matrix.</p>
 * @param matrix
 * @param filepath
 */
void Matrix(MATRIX *matrix, char *filepath);

/**
 * @brief <p>Copy data from src to dest.</p>
 */
__attribute__((unused)) void matrixcpy(MATRIX *dest, const MATRIX *src);

/**
 * @brief <p>Allocate memory in heap for a certain matrix.</p>
 */
MATRIX *prepareMat(size_t row, size_t col);

void freeMatrix(MATRIX *matrix);

void setRow(MATRIX *matrix, char *filename);

void setCol(MATRIX *matrix, char *filename);

MATRIX *subMatrix(MATRIX *matrix, size_t beginRowIndex, size_t beginColIndex, size_t row, size_t col);

MATRIX *matAddition(MATRIX *mat_A, MATRIX *mat_B);

MATRIX *matSubtraction(MATRIX *mat_A, MATRIX *mat_B);

MATRIX *mergeMat(MATRIX *C11, MATRIX *C12, MATRIX *C21, MATRIX *C22);

void matMul_ijk(MATRIX *mat_A, MATRIX *mat_B);

void matMul_ikj(MATRIX *mat_A, MATRIX *mat_B);

__attribute__((unused)) void matMul_OpenMP(MATRIX *mat_A, MATRIX *mat_B);

void matMul_Strassen(MATRIX *mat_A, MATRIX *mat_B);

MATRIX *strassen_(MATRIX *mat_A, MATRIX *mat_B);

float avx2_dotProduct(const float *v1_, const float *v2_, size_t n);

void matMul_Threads(MATRIX *mat_A, MATRIX *mat_B);

MATRIX *threadMulWith_n_Threads(MATRIX *mat_A, MATRIX *mat_B, size_t threadsNum);

void *singleThreadMatMul(void *args);

__attribute__((unused)) void matMul_OpenBLAS_dotProd(MATRIX *mat_A, MATRIX *mat_B);

void matMul_OpenBLAS(MATRIX *mat_A, MATRIX *mat_B);

#endif //PROJECT_3_MATRIX_H
