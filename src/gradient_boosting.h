#ifndef GRADIENT_BOOSTING_H
#define GRADIENT_BOOSTING_H

#include "decision_tree.h"
#include "model.h"

typedef struct GradientBoosting {
    size_t n_trees;
    double learning_rate;
    DecisionTree **trees;
} GradientBoosting;

GradientBoosting* gradient_boosting_train(Model *model, double **X, double *y, size_t n_samples, size_t n_features, size_t n_trees, size_t max_depth, double learning_rate, size_t min_samples_split);
double gradient_boosting_predict(GradientBoosting *model, double *x);
void gradient_boosting_free(GradientBoosting *model);

#endif