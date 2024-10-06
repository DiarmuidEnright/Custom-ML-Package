#include "matrix.h"

Matrix *matrix_create(int rows, int cols) {
    Matrix *matrix = (Matrix *) malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (double **) malloc(rows * sizeof(double *));
    int i;
    for (i = 0; i < rows; i++) {
        matrix->data[i] = (double *) malloc(cols * sizeof(double));
    }
    return matrix;
}

void matrix_free(Matrix *matrix) {
    int i;
    for (i = 0; i < matrix->rows; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}

double matrix_get(Matrix *matrix, int row, int col) {
    if (row < 0 || row >= matrix->rows || col < 0 || col >= matrix->cols) {
        printf("Error: Index out of bounds\n");
        exit(1);
    }
    return matrix->data[row][col];
}

void matrix_set(Matrix *matrix, int row, int col, double value) {
    if (row < 0 || row >= matrix->rows || col < 0 || col >= matrix->cols) {
        printf("Error: Index out of bounds\n");
        exit(1);
    }
    matrix->data[row][col] = value;
}

void matrix_print(Matrix *matrix) {
    int i, j;
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->cols; j++) {
            printf("%f ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

Matrix *matrix_transpose(Matrix *matrix) {
    Matrix *transpose = matrix_create(matrix->cols, matrix->rows);
    int i, j;
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->cols; j++) {
            matrix_set(transpose, j, i, matrix_get(matrix, i, j));
        }
    }
    return transpose;
}

Matrix *matrix_multiply(Matrix *matrix1, Matrix *matrix2) {
    if (matrix1->cols != matrix2->rows) {
        printf("Error: Matrices cannot be multiplied\n");
        exit(1);
    }
    Matrix *result = matrix_create(matrix1->rows, matrix2->cols);
    int i, j, k;
    for (i = 0; i < matrix1->rows; i++) {
        for (j = 0; j < matrix2->cols; j++) {
            double sum = 0;
            for (k = 0; k < matrix1->cols; k++) {
                sum += matrix_get(matrix1, i, k) * matrix_get(matrix2, k, j);
            }
            matrix_set(result, i, j, sum);
        }
    }
    return result;
}