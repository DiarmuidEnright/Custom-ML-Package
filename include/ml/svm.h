#ifndef SVM_H
#define SVM_H

#include "core/model.h"

Model* create_svm();
void svm_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split);
double svm_predict(Model *self, double *x, int n_features);
void svm_free(Model *self);

#endif
