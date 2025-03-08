#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>
#include "core/model.h"

typedef struct Dataset {
    double **X;
    double *y;
    size_t n_samples;
    size_t n_features;
} Dataset;

double** load_dataset(const char *filename, size_t *n_samples, size_t *n_features);
void split_dataset(double **X, double *y, size_t n_samples, double train_ratio, double ***X_train, double ***X_test, double **y_train, double **y_test);
void k_fold_split(double **X, double *y, size_t n_samples, size_t k, size_t fold_idx, double ***X_train, double ***X_val, double **y_train, double **y_val);

#endif
