#ifndef GRADIENT_BOOSTING_H
#define GRADIENT_BOOSTING_H

#include "decision_tree.h"

typedef struct {
    DecisionTree **trees;
    size_t n_trees;
    double learning_rate;
} GradientBoosting;

GradientBoosting* gradient_boosting_train(double **X, double *y, size_t n_samples, size_t n_features, size_t n_trees, size_t max_depth, double learning_rate);
double gradient_boosting_predict(GradientBoosting *model, double *x);
void gradient_boosting_free(GradientBoosting *model);

#endif
