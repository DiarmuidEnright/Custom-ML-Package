#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

Matrix *matrix_create(int rows, int cols);

void matrix_free(Matrix *matrix);

double matrix_get(Matrix *matrix, int row, int col);

void matrix_set(Matrix *matrix, int row, int col, double value);

void matrix_print(Matrix *matrix);

Matrix *matrix_transpose(Matrix *matrix);

Matrix *matrix_multiply(Matrix *matrix1, Matrix *matrix2);

#endif