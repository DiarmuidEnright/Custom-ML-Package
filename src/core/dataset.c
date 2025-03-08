#include "core/dataset.h"
#include <stdlib.h>

void k_fold_split(double **X, double *y, size_t n_samples, size_t k, size_t fold_idx, double ***X_train, double ***X_val, double **y_train, double **y_val) {
    size_t fold_size = n_samples / k;
    size_t val_start = fold_idx * fold_size;
    size_t val_end = val_start + fold_size;

    *X_train = (double **)malloc((n_samples - fold_size) * sizeof(double *));
    *X_val = (double **)malloc(fold_size * sizeof(double *));
    *y_train = (double *)malloc((n_samples - fold_size) * sizeof(double));
    *y_val = (double *)malloc(fold_size * sizeof(double));

    size_t train_idx = 0, val_idx = 0;
    for (size_t i = 0; i < n_samples; i++) {
        if (i >= val_start && i < val_end) {
            (*X_val)[val_idx] = X[i];
            (*y_val)[val_idx++] = y[i];
        } else {
            (*X_train)[train_idx] = X[i];
            (*y_train)[train_idx++] = y[i];
        }
    }
}
