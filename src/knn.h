#ifndef KNN_H
#define KNN_H

#include "model.h"

Model* create_knn();
void knn_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split);
double knn_predict(Model *self, double *x, int n_features);
void knn_free(Model *self);

#endif
