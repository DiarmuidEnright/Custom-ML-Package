#ifndef KNN_H
#define KNN_H

#include <stddef.h>

typedef struct {
    double **X_train;
    double *y_train;
    size_t n_samples;
    size_t n_features;
} kNN;

kNN* knn_train(double **X_train, double *y_train, size_t n_samples, size_t n_features);
double knn_predict(kNN *model, double *x);
void knn_free(kNN *model);

double euclidean_distance(double *a, double *b, size_t n);

#endif
