#ifndef SVM_H
#define SVM_H

#include <stddef.h>

typedef struct {
    double *weights;
    double bias;
    size_t n_features;
} SVM;

SVM* svm_train(double **X, double *y, size_t n_samples, size_t n_features, double C, double tol, double max_iter);
double svm_predict(SVM *model, double *x);
void svm_free(SVM *model);

#endif
