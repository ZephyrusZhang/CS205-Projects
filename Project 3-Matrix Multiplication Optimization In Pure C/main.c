#pragma GCC optimize(3, "Ofast", "inline")

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./include/matrix.h"
#include "./include/util.h"

char root[64] = "../txt/";
char suffix[64] = ".txt";
extern bool use_avx2;

int main(int argc, char **argv) {

    char filename0[64] = {};
    char filename1[64] = {};

    if (argc == 1) {
        while (TRUE) {
            printf("Please input names of two file or [Q] to quit: \n");
            if (scanf("%s", filename0) == EOF) {
                printf("Fail to read name from stdin. Exit.\n");
                exit(0);
            }
            if (filename0[0] == 'Q' || filename0[0] == 'q') {
                printf("Program exits.\n");
                exit(0);
            }
            if (scanf("%s", filename1) == EOF) {
                printf("Fail to read name from stdin. Exit.\n");
                exit(0);
            }

            MATRIX mat_A;
            Matrix(&mat_A, getFilePath(root, filename0));
            MATRIX mat_B;
            Matrix(&mat_B, getFilePath(root, filename1));

            char methodID;
            printf("\nWhich method do you want to use to calculate multiplication?\n");
            printf("    1 -->> Multiplication in order i->j->k.\n");
            printf("    2 -->> Multiplication in order i->k->j.\n");
            printf("    3 -->> Multiplication using Strassen.\n");
            printf("    4 -->> Multiplication using multi-thread.\n");
            printf("    5 -->> Multiplication using OpenBLAS.\n");
            if (scanf("\n%c", &methodID) == EOF) {
                printf("Fail to read methodID from stdin. Exit.\n");
                exit(0);
            }

            switch (methodID) {
                case '1':
                    matMul_ijk(&mat_A, &mat_B);
                    break;
                case '2':
                    matMul_ikj(&mat_A, &mat_B);
                    break;
                case '3':
                    matMul_Strassen(&mat_A, &mat_B);
                    break;
                case '4':
                    matMul_Threads(&mat_A, &mat_B);
                    break;
                case '5':
                    matMul_OpenBLAS(&mat_A, &mat_B);
                    break;
                default:
                    printf("Wrong ID. Exit.\n");
                    exit(0);
            }
        }
    } else {
        if (argc < 4) {
            puts("Not enough parameter. Exit.");
            exit(100);
        }

        char *ptr;
        if (strtol(argv[3], &ptr, 10)) {
            use_avx2 = TRUE;
        }

        strncpy(filename0, argv[1], 64);
        strncpy(filename1, argv[2], 64);
    }

    MATRIX mat_A;
    Matrix(&mat_A, getFilePath(root, filename0));
    MATRIX mat_B;
    Matrix(&mat_B, getFilePath(root, filename1));

    MATRIX *mat = matAddition(&mat_A, &mat_B);
    writeMatrix(mat->data, mat->row, mat->col, "addition-32x32.txt");

//    matMul_OpenBLAS(&mat_A, &mat_B);
//
//    matMul_OpenBLAS_dotProd(&mat_A, &mat_B);
//
//    matMul_ijk(&mat_A, &mat_B);
//
//    matMul_ikj(&mat_A, &mat_B);
//
//    matMul_Strassen(&mat_A, &mat_B);
//
//    matMul_Threads(&mat_A, &mat_B);
//
//    matMul_OpenMP(&mat_A, &mat_B);

    if (use_avx2 == TRUE) {
        printf("All multiplication above use AVX2 to speed up calculation.\n");
    } else {
        printf("All multiplication above don't use AVX2 to speed up calculation.\n");
    }

    return 0;
}