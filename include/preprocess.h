#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <stddef.h>

void min_max_scaler(double **X, size_t n_samples, size_t n_features, double *min, double *max);
void standard_scaler(double **X, size_t n_samples, size_t n_features, double *mean, double *stddev);

#endif
