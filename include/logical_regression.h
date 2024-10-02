#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <stddef.h>

typedef struct {
    double *weights;
    double bias;
    size_t n_features;
} LogisticRegression;

LogisticRegression* logistic_regression_train(double **X, double *y, size_t n_samples, size_t n_features, double learning_rate, size_t max_iter);
double logistic_regression_predict(LogisticRegression *model, double *x);
void logistic_regression_free(LogisticRegression *model);

#endif
