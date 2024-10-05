#include "preprocess.h"
#include <math.h>

void min_max_scaler(double **X, size_t n_samples, size_t n_features, double *min, double *max) {
    for (size_t i = 0; i < n_features; i++) {
        min[i] = X[0][i];
        max[i] = X[0][i];

        for (size_t j = 1; j < n_samples; j++) {
            if (X[j][i] < min[i]) min[i] = X[j][i];
            if (X[j][i] > max[i]) max[i] = X[j][i];
        }

        for (size_t j = 0; j < n_samples; j++) {
            X[j][i] = (X[j][i] - min[i]) / (max[i] - min[i]);
        }
    }
}

void standard_scaler(double **X, size_t n_samples, size_t n_features, double *mean, double *stddev) {
    for (size_t i = 0; i < n_features; i++) {
        mean[i] = 0.0;
        stddev[i] = 0.0;

        for (size_t j = 0; j < n_samples; j++) {
            mean[i] += X[j][i];
        }
        mean[i] /= n_samples;

        for (size_t j = 0; j < n_samples; j++) {
            stddev[i] += pow(X[j][i] - mean[i], 2);
        }
        stddev[i] = sqrt(stddev[i] / n_samples);

        for (size_t j = 0; j < n_samples; j++) {
            if (stddev[i] != 0) {
                X[j][i] = (X[j][i] - mean[i]) / stddev[i];
            } else {
                X[j][i] = 0;
            }
        }
    }
}
