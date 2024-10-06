#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "model.h"

typedef struct Forest {
    DecisionTree **trees;
    int n_trees;
} Forest;

Forest* create_forest(int n_trees, int max_depth, int n_features, int n_samples, double **X, double *y);
void free_forest(Forest *forest);
double random_forest_predict(Forest *forest, double *x);

#endif