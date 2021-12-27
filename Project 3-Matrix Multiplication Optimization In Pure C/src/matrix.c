#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wunused-result"
#pragma ide diagnostic ignored "cert-err34-c"
#pragma GCC optimize(3, "Ofast", "inline")

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <pthread.h>

#include "../include/matrix.h"
#include "../include/util.h"
#include "../include/timer.h"
#include "../xianyi-OpenBLAS-efe4248/cblas.h"

bool use_avx2 = FALSE;

TIMER timer = {0, 0, 0, 0, 0};

void Matrix(MATRIX *matrix, char *filepath) {
    setRow(matrix, filepath);
    setCol(matrix, filepath);

    FILE *fp;
    if ((fp = fopen(filepath, "rt")) == NULL) {
        printf("Fail to open fp \"%s\". Exit.\n", filepath);
        exit(0);
    }

    printf("Start to read file and construct matrix.\n");
    TIME_START
    float **array = allocate2DArray(matrix->row, matrix->col);
    for (size_t i = 0; i < matrix->row; i++) {
        for (size_t j = 0; j < matrix->col; j++) {
            fscanf(fp, "%f", &array[i][j]);
        }
    }
    matrix->data = array;
    TIME_END("Reading file and constructing matrix completed. ")

    printf("Construct matrix from file \"%s\" successfully.\n", filepath);
    fclose(fp);
}

__attribute__((unused)) void matrixcpy(MATRIX *dest, const MATRIX *src) {
    if (dest->data != NULL) {
        free2DArray(dest->data);
    }

    dest->data = allocate2DArray(src->row, src->col);
    for (size_t i = 0; i < src->row; i++) {
        for (size_t j = 0; j < src->col; j++) {
            dest->data[i][j] = src->data[i][j];
        }
    }
}

MATRIX *prepareMat(size_t row, size_t col) {
    MATRIX *resMat = (MATRIX *) malloc(sizeof(MATRIX));
    resMat->row = row;
    resMat->col = col;
    resMat->data = allocate2DArray(row, col);
    return resMat;
}

void freeMatrix(MATRIX *matrix) {
    free2DArray(matrix->data);
    free(matrix);
}

void setRow(MATRIX *matrix, char *filename) {
    FILE *fp;
    if ((fp = fopen(filename, "rt")) == NULL) {
        printf("Fail to open fp \"%s\". Exit.\n", filename);
        exit(0);
    }

    char *line = NULL;
    size_t len = 0;
    size_t rowCnt = 0;
    while (getline(&line, &len, fp) != -1) rowCnt++;
    matrix->row = rowCnt;
    fclose(fp);
}

void setCol(MATRIX *matrix, char *filename) {
    FILE *fp;
    if ((fp = fopen(filename, "rt")) == NULL) {
        printf("Fail to open fp \"%s\". Exit.\n", filename);
        exit(0);
    }

    size_t colCnt = 0;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    if ((read = getline(&line, &len, fp)) == -1) {
        printf("Fail to read line.");
        exit(0);
    }
    for (ssize_t i = 0; i < read; i++) {
        if (line[i] == 32 || line[i] == 10) colCnt++;
    }
    matrix->col = colCnt;
    fclose(fp);
}

MATRIX *subMatrix(MATRIX *matrix, size_t beginRowIndex, size_t beginColIndex, size_t row, size_t col) {
    if (beginRowIndex >= matrix->row || beginColIndex >= matrix->col || row > matrix->row || col > matrix->col) {
        printf("Invalid parameter to construct sub-matrix. Exit.\n");
        exit(0);
    }

    MATRIX *subMat = (MATRIX *) malloc(sizeof(MATRIX));
    subMat->row = row;
    subMat->col = col;

    subMat->data = allocate2DArray(row, col);
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            subMat->data[i][j] = matrix->data[i + beginRowIndex][j + beginColIndex];
        }
    }

    return subMat;
}

MATRIX *matAddition(MATRIX *mat_A, MATRIX *mat_B) {
    if ((mat_A->row != mat_B->row) || (mat_A->col != mat_B->col)) {
        printf("Addition between two inconsistent matrices. Exit.\n");
        exit(0);
    }

    MATRIX *res = prepareMat(mat_A->row, mat_A->col);
    for (size_t i = 0; i < mat_A->row; i++) {
        for (size_t j = 0; j < mat_A->col; j++) {
            res->data[i][j] = mat_A->data[i][j] + mat_B->data[i][j];
        }
    }

    return res;
}

MATRIX *matSubtraction(MATRIX *mat_A, MATRIX *mat_B) {
    if ((mat_A->row != mat_B->row) || (mat_A->col != mat_B->col)) {
        printf("Subtraction between two inconsistent matrices. Exit.\n");
        exit(0);
    }

    MATRIX *res = prepareMat(mat_A->row, mat_A->col);
    for (size_t i = 0; i < mat_A->row; i++) {
        for (size_t j = 0; j < mat_A->col; j++) {
            res->data[i][j] = mat_A->data[i][j] - mat_B->data[i][j];
        }
    }

    return res;
}

void matMul_ijk(MATRIX *mat_A, MATRIX *mat_B) {
    float **resArr = allocate2DArray(mat_A->row, mat_B->col);

    printf("Start to execute multiplication between two %ldx%ld matrices in order i->j->k.\n", mat_A->row, mat_A->col);
    TIME_START
    for (size_t i = 0; i < mat_A->row; i++) {
        for (size_t j = 0; j < mat_B->col; j++) {
            if (use_avx2 == FALSE) {
                float temp = 0.0f;
                for (size_t k = 0; k < mat_A->col; k++) {
                    temp += mat_A->data[i][k] * mat_B->data[k][j];
                }
                resArr[i][j] = temp;
            } else {
                resArr[i][j] = avx2_dotProduct(mat_A->data[i], &(mat_B->data[0][j]), mat_A->row);
            }
        }
    }
    TIME_END("<<< Multiplication in order i->j->k completed. >>> ")

    writeMatrix(resArr, mat_A->row, mat_B->col, getOuFileName(mat_A->row, mat_B->col, IJK));

    free2DArray(resArr);
}

void matMul_ikj(MATRIX *mat_A, MATRIX *mat_B) {
    float **resArr = allocate2DArray(mat_A->row, mat_B->col);

    printf("Start to execute multiplication of two %ldx%ld matrices in order i->k->j.\n", mat_A->row, mat_A->col);
    TIME_START
    for (size_t i = 0; i < mat_A->row; i++) {
        for (size_t k = 0; k < mat_A->col; k++) {
            float temp = mat_A->data[i][k];
            for (size_t j = 0; j < mat_B->col; j++) {
                resArr[i][j] += temp * mat_B->data[k][j];
            }
        }
    }
    TIME_END("<<< Multiplication in order i->k->j completed. >>> ")

    writeMatrix(resArr, mat_A->row, mat_B->col, getOuFileName(mat_A->row, mat_B->col, IKJ));

    free2DArray(resArr);
}

__attribute__((unused)) void matMul_OpenMP(MATRIX *mat_A, MATRIX *mat_B) {
    float **resArr = allocate2DArray(mat_A->row, mat_B->col);

    printf("Start to execute multiplication of two %ldx%ld matrices using OpenMP.\n", mat_A->row, mat_A->col);
    TIME_START
#pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < mat_A->row; i++) {
        for (size_t k = 0; k < mat_A->col; k++) {
            float temp = mat_A->data[i][k];
            for (size_t j = 0; j < mat_B->col; j++) {
                resArr[i][j] += temp * mat_B->data[k][j];
            }
        }
    }
    TIME_END("<<< Multiplication using OpenMP completed. >>> ")

    writeMatrix(resArr, mat_A->row, mat_B->col, getOuFileName(mat_A->row, mat_B->col, OPENMP));

    free2DArray(resArr);
}

void matMul_Strassen(MATRIX *mat_A, MATRIX *mat_B) {
    MATRIX *res;

    printf("Start to execute multiplication of two %ldx%ld matrices using Strassen.\n", mat_A->row, mat_A->col);
    TIME_START
    res = strassen_(mat_A, mat_B);
    TIME_END("<<< Multiplication using Strassen completed. >>> ")

    writeMatrix(res->data, mat_A->row, mat_B->col, getOuFileName(mat_A->row, mat_B->col, STRASSEN));
    freeMatrix(res);
}

MATRIX *mergeMat(MATRIX *C11, MATRIX *C12, MATRIX *C21, MATRIX *C22) {
    MATRIX *mergeRes = prepareMat(C11->row + C21->row, C11->col + C12->col);

    for (size_t i = 0; i < C11->row; i++) {
        for (size_t j = 0; j < C11->col; j++) {
            mergeRes->data[i][j] = C11->data[i][j];
            mergeRes->data[i][j + C11->col] = C12->data[i][j];
            mergeRes->data[i + C21->row][j] = C21->data[i][j];
            mergeRes->data[i + C21->row][j + C22->col] = C22->data[i][j];
        }
    }

    //region Free Useless Matrix
    freeMatrix(C11);
    freeMatrix(C12);
    freeMatrix(C21);
    freeMatrix(C22);
    //endregion

    return mergeRes;
}

MATRIX *strassen_(MATRIX *mat_A, MATRIX *mat_B) {
    size_t n = mat_A->row;

    if (n <= 256) {
        MATRIX *res = prepareMat(mat_A->row, mat_B->col);

        if (use_avx2 == FALSE) {
            for (size_t i = 0; i < mat_A->row; i++) {
                for (size_t k = 0; k < mat_A->col; k++) {
                    float temp = mat_A->data[i][k];
                    for (size_t j = 0; j < mat_B->col; j++) {
                        res->data[i][j] += temp * mat_B->data[k][j];
                    }
                }
            }
        } else {
            if (n % 8 != 0) {
                printf("Size of matrices is not fit for SIMD optimization. Exit.\n");
                exit(0);
            }
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    res->data[i][j] = avx2_dotProduct(mat_A->data[i], &(mat_B->data[0][j]), n);
                }
            }
        }

        return res;
    }

    n /= 2;

    MATRIX *A11 = subMatrix(mat_A, 0, 0, n, n);
    MATRIX *A12 = subMatrix(mat_A, 0, n, n, n);
    MATRIX *A21 = subMatrix(mat_A, n, 0, n, n);
    MATRIX *A22 = subMatrix(mat_A, n, n, n, n);

    MATRIX *B11 = subMatrix(mat_B, 0, 0, n, n);
    MATRIX *B12 = subMatrix(mat_B, 0, n, n, n);
    MATRIX *B21 = subMatrix(mat_B, n, 0, n, n);
    MATRIX *B22 = subMatrix(mat_B, n, n, n, n);

    MATRIX *S1 = matSubtraction(B12, B22);
    MATRIX *S2 = matAddition(A11, A12);
    MATRIX *S3 = matAddition(A21, A22);
    MATRIX *S4 = matSubtraction(B21, B11);
    MATRIX *S5 = matAddition(A11, A22);
    MATRIX *S6 = matAddition(B11, B22);
    MATRIX *S7 = matSubtraction(A12, A22);
    MATRIX *S8 = matAddition(B21, B22);
    MATRIX *S9 = matSubtraction(A11, A21);
    MATRIX *S10 = matAddition(B11, B12);

    MATRIX *P1 = strassen_(A11, S1);
    MATRIX *P2 = strassen_(S2, B22);
    MATRIX *P3 = strassen_(S3, B11);
    MATRIX *P4 = strassen_(A22, S4);
    MATRIX *P5 = strassen_(S5, S6);
    MATRIX *P6 = strassen_(S7, S8);
    MATRIX *P7 = strassen_(S9, S10);

    //region Free Useless Matrix
    freeMatrix(A11);
    freeMatrix(A12);
    freeMatrix(A21);
    freeMatrix(A22);
    freeMatrix(B11);
    freeMatrix(B12);
    freeMatrix(B21);
    freeMatrix(B22);
    freeMatrix(S1);
    freeMatrix(S2);
    freeMatrix(S3);
    freeMatrix(S4);
    freeMatrix(S5);
    freeMatrix(S6);
    freeMatrix(S7);
    freeMatrix(S8);
    freeMatrix(S9);
    freeMatrix(S10);
    //endregion

    MATRIX *C11 = matAddition(matSubtraction(matAddition(P5, P4), P2), P6);
    MATRIX *C12 = matAddition(P1, P2);
    MATRIX *C21 = matAddition(P3, P4);
    MATRIX *C22 = matSubtraction(matSubtraction(matAddition(P5, P1), P3), P7);

    return mergeMat(C11, C12, C21, C22);
}

float avx2_dotProduct(const float *v1_, const float *v2_, size_t n) {
    __attribute__ ((aligned (32)))float sum[8] = {0.0f};

    __m256 r1, r2;
    __m256 r = _mm256_setzero_ps();
    float *v1 = (float *) aligned_alloc(32, sizeof(float) * n);
    float *v2 = (float *) aligned_alloc(32, sizeof(float) * n);
    alignedCopy(v1_, v2_, v1, v2, n);

    for (size_t k = 0; k < n; k += 8) {
        r1 = _mm256_load_ps(v1 + k);
        r2 = _mm256_load_ps(v2 + k);
        r = _mm256_add_ps(r, _mm256_mul_ps(r1, r2));
    }
    _mm256_store_ps(sum, r);

    free(v1);
    free(v2);

    return (sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]);
}

void matMul_Threads(MATRIX *mat_A, MATRIX *mat_B) {
    MATRIX *res;
    int threadsNum = 16;

    printf("Start to execute multiplication between two %ldx%ld matrices using multi-threadsNum (threadsNum >>> %d).\n",
           mat_A->row, mat_B->col, threadsNum);
    TIME_START
    res = threadMulWith_n_Threads(mat_A, mat_B, threadsNum);
    TIME_END("<<< Multiplication using multi-thread completed. >>> ")

    writeMatrix(res->data, res->row, res->col, getOuFileName(mat_A->row, mat_B->col, MULTI_THREADS));

    freeMatrix(res);
}

MATRIX *threadMulWith_n_Threads(MATRIX *mat_A, MATRIX *mat_B, size_t threadsNum) {
    size_t rowNumOfEveryThread = mat_A->row / threadsNum;
    MATRIX *resultMatrix = prepareMat(mat_A->row, mat_B->col);

    pthread_t *pthreads = (pthread_t *) malloc(threadsNum * sizeof(pthread_t));
    for (size_t i = 0; i < threadsNum; i++) pthreads[i] = (pthread_t) NULL;
    ResultSet **resultSets = (ResultSet **) malloc(sizeof(ResultSet *) * threadsNum);
    for (size_t i = 0; i < threadsNum; i++) {
        resultSets[i] = prepareResultSet(resultMatrix, mat_A, mat_B, i * rowNumOfEveryThread, rowNumOfEveryThread);
    }
    for (size_t i = 0; i < threadsNum; i++) {
        pthread_create(&pthreads[i], NULL, singleThreadMatMul, (void *) resultSets[i]);
    }
    for (size_t i = 0; i < threadsNum; i++) {
        if (pthread_join(pthreads[i], NULL) == 0) {
            printf("Thread[%ld] exits successfully. ", i);
        } else {
            printf("Thread[%ld] exits unsuccessfully. Exit.\n", i);
            exit(0);
        }
    }
    printf("\n");

    for (size_t i = 0; i < threadsNum; i++) {
        free(resultSets[i]);
    }
    free(resultSets);

    return resultMatrix;
}

void *singleThreadMatMul(void *args) {
    ResultSet *set = (ResultSet *) args;

    for (size_t i = 0; i < set->rowNumToCal; i++) {
        for (size_t k = 0; k < set->mat_A->col; k++) {
            float temp = set->mat_A->data[set->beginRowIndex + i][k];
            for (size_t j = 0; j < set->mat_B->col; j++) {
                set->resMat->data[set->beginRowIndex + i][j] += temp * set->mat_B->data[k][j];
            }
        }
    }

    pthread_exit(NULL);
}

__attribute__((unused)) void matMul_OpenBLAS_dotProd(MATRIX *mat_A, MATRIX *mat_B) {
    MATRIX *res = prepareMat(mat_A->row, mat_B->col);

    printf("Start to execute multiplication between two %ldx%ld matrices using OpenBLAS's dot product.\n", mat_A->row,
           mat_A->col);
    TIME_START
    for (size_t i = 0; i < mat_A->row; i++) {
        for (size_t j = 0; j < mat_B->col; j++) {
            res->data[i][j] = cblas_sdot((int) mat_A->row, mat_A->data[i], 1, &(mat_B->data[0][j]), (int) mat_B->row);
        }
    }
    TIME_END("<<< Multiplication using OpenBLAS's dot product completed. >>>")

    writeMatrix(res->data, res->row, res->col, getOuFileName(res->row, res->col, OPEN_BLAS_dotProduct));

    freeMatrix(res);
}

void matMul_OpenBLAS(MATRIX *mat_A, MATRIX *mat_B) {
    MATRIX *res = prepareMat(mat_A->row, mat_B->col);

    TIME_START
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) mat_A->row, (int) mat_B->col, (int) mat_A->col, 1,
                *mat_A->data, (int) mat_A->col, *mat_B->data, (int) mat_B->col, 0, *res->data, (int) mat_A->col);
    TIME_END("<<< Multiplication using OpenBLAS completed. >>>")

    writeMatrix(res->data, res->row, res->col, getOuFileName(res->row, res->col, OPEN_BLAS));

    freeMatrix(res);
}
#pragma clang diagnostic pop