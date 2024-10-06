#include "cross_validation.h"
#include "model.h"
#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>

void split_dataset(double **X, double *y, size_t n_samples, double train_ratio, double ***X_train, double ***X_test, double **y_train, double **y_test) {
    size_t train_size = (size_t)(train_ratio);
    size_t test_size = n_samples - train_size;

    *X_train = (double**)malloc(train_size * sizeof(double*));
    *X_test = (double**)malloc(test_size * sizeof(double*));
    *y_train = (double*)malloc(train_size * sizeof(double));
    *y_test = (double*)malloc(test_size * sizeof(double));

    for (size_t i = 0; i < train_size; i++) {
        (*X_train)[i] = X[i];
        (*y_train)[i] = y[i];
    }
    
    for (size_t i = 0; i < test_size; i++) {
        (*X_test)[i] = X[train_size + i];
        (*y_test)[i] = y[train_size + i];
    }
}

void cross_validation(Model *model, Dataset *data, int k) {
    int fold_size = data->n_samples / k;

    for (int i = 0; i < k; i++) {
        double **X_train, **X_test;
        double *y_train, *y_test;

        split_dataset(data->X, data->y, data->n_samples, (double)(fold_size * i), &X_train, &X_test, &y_train, &y_test);

        model->train(model, X_train, fold_size, data->n_features);

        double accuracy = model_evaluate(model, X_test, fold_size, data->n_features);
        printf("Fold %d: Accuracy = %.2f\n", i + 1, accuracy);

        free(X_train);
        free(X_test);
        free(y_train);
        free(y_test);
    }
}
