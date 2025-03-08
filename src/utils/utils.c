#include "utils/utils.h"
#include <stdlib.h>
#include <math.h>
#include <stddef.h>

void normalize_data(double **X, size_t n_samples, size_t n_features) {
    for (size_t i = 0; i < n_features; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < n_samples; j++) {
            sum += X[j][i];
        }
        double mean = sum / n_samples;

        double sum_squared_diff = 0.0;
        for (size_t j = 0; j < n_samples; j++) {
            sum_squared_diff += (X[j][i] - mean) * (X[j][i] - mean);
        }
        double stddev = sqrt(sum_squared_diff / n_samples);

        for (size_t j = 0; j < n_samples; j++) {
            if (stddev != 0) {
                X[j][i] = (X[j][i] - mean) / stddev;
            } else {
                X[j][i] = 0;
            }
        }
    }
}
