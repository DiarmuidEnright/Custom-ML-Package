#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "decision_tree.h"

typedef struct {
    DecisionTree **trees;
    size_t n_trees;
} RandomForest;

RandomForest* random_forest_train(double **X, double *y, size_t n_samples, size_t n_features, size_t n_trees, size_t max_depth, size_t min_samples_split);
double random_forest_predict(RandomForest *forest, double *x);
void random_forest_free(RandomForest *forest);

#endif
