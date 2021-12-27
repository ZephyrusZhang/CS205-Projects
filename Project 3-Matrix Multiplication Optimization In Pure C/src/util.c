#pragma GCC optimize(3, "Ofast", "inline")

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/util.h"
#include "../include/timer.h"

char prefix[64] = "out-";
char path[64] = {};
const char methods[7][64] = {"IJK", "IKJ", "STRASSEN", "MULTI_THREADS", "OpenBLAS_dotProduct", "OpenMP", "OpenBLAS"};

extern TIMER timer;
extern char suffix[];

inline void alignedCopy(const float *v1_, const float *v2_, float *v1, float *v2, size_t n) {
    for (size_t i = 0; i < n; i++) {
        v1[i] = v1_[i];
        v2[i] = v2_[i * n];
    }
}

float **allocate2DArray(size_t row, size_t col) {
    float **res = (float **) malloc(row * sizeof(float *));
    float *p = (float *) malloc(row * col * sizeof(float));
    if (res && p) {
        for (size_t i = 0; i < row; i++) {
            res[i] = p + i * col;
        }
    }

    //Initialization
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            res[i][j] = 0.0f;
        }
    }
    return res;
}

void free2DArray(float **arr) {
    free(arr[0]);
    free(arr);
}

__attribute__((unused)) void print2DArray(float **arr, size_t row, size_t col) {
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            printf("%-4.1f ", arr[i][j]);
        }
        printf("\n");
    }
}

char *getFilePath(const char *root, char *filename) {
//    char *path = (char *) malloc(sizeof(char) * strlen(root));
    for (size_t i = 0; i < strlen(root); i++) {
        path[i] = '\0';
    }
    strcpy(path, root);
    strcat(path, filename);
    return path;
}

char *getOuFileName(size_t row, size_t col, Method method) {
    char *name = (char *) malloc(sizeof(char) * 64);
    memcpy(name, prefix, sizeof(char) * 64);
    char buffer[64] = {};
    sprintf(buffer, "%ld", row);
    strcat(name, buffer);
    strcat(name, "x");
    sprintf(buffer, "%ld", col);
    strcat(name, buffer);
    strcat(name, "-");
    strcat(name, methods[(int)method]);
    strcat(name, suffix);
    return name;
}

void writeMatrix(float **arr, size_t row, size_t col, char *outFileName) {
//    printf("Start to write matrix into %s. \n", outFileName);

//    TIME_START
    FILE *fp;
    if ((fp = fopen(outFileName, "w")) == NULL) {
        printf("Fail to open or create output file -> %s. Exit.", outFileName);
        exit(0);
    }

    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            char *buffer = (char *) malloc(sizeof(char) * 32);
            sprintf(buffer, "%.1f", arr[i][j]);
            fputs(buffer, fp);
            if (j < col - 1) fputs(" ", fp);
            else fputs("\r\n", fp);
            free(buffer);
        }
    }
//    TIME_END("Writing matrix into file completed. ")

    fclose(fp);
}

ResultSet *prepareResultSet(MATRIX *resMat, MATRIX *mat_A, MATRIX *mat_B, size_t beginRowIndex, size_t rowNumToCal) {
    ResultSet *set = (ResultSet *) malloc(sizeof(ResultSet));
    set->resMat = resMat;
    set->mat_A = mat_A;
    set->mat_B = mat_B;
    set->beginRowIndex = beginRowIndex;
    set->rowNumToCal = rowNumToCal;
    return set;
}

__attribute__((unused)) float *convertTo1DArray(float **data, size_t row, size_t col) {
    float *res = (float *)malloc(sizeof(float) * row * col);
    for (size_t i = 0; i < row; i++) {
        for (size_t j = 0; j < col; j++) {
            *(res + j + i * col) = data[i][j];
        }
    }
    return res;
}