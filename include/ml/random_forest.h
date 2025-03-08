#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "core/model.h"
#include "ml/decision_tree.h"
#include <stddef.h>

typedef struct Forest {
    DecisionTree **trees;
    size_t n_trees;
} Forest;

// Training and prediction functions
Forest* random_forest_train(double **X, double *y, size_t n_samples, size_t n_features, size_t n_trees, size_t max_depth, size_t min_samples_split);
double random_forest_predict(Forest *forest, double *x);
void random_forest_free(Forest *forest);

#endif
