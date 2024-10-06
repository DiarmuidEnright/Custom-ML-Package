#include "random_forest.h"
#include "decision_tree.h"
#include <stdlib.h>
#include <stdio.h>

RandomForest* random_forest_train(double **X, double *y, size_t n_samples, size_t n_features, size_t n_trees, size_t max_depth, size_t min_samples_split) {
    RandomForest *forest = (RandomForest *)malloc(sizeof(RandomForest));
    forest->n_trees = n_trees;
    forest->trees = (DecisionTree **)malloc(n_trees * sizeof(DecisionTree *));

    for (size_t i = 0; i < n_trees; i++) {
        double **bootstrap_X = (double **)malloc(n_samples * sizeof(double *));
        double *bootstrap_y = (double *)malloc(n_samples * sizeof(double));

        for (size_t j = 0; j < n_samples; j++) {
            size_t idx = rand() % n_samples;
            bootstrap_X[j] = X[idx];
            bootstrap_y[j] = y[idx];
        }

        Model *tree_model = (Model *)malloc(sizeof(Model));
        tree_model->root = NULL;
        decision_tree_train(tree_model, bootstrap_X, bootstrap_y, n_samples, n_features, max_depth, min_samples_split);

        forest->trees[i] = tree_model;

        free(bootstrap_X);
        free(bootstrap_y);
    }

    return forest;
}

double random_forest_predict(RandomForest *forest, double *x) {
    size_t votes[2] = {0, 0};

    for (size_t i = 0; i < forest->n_trees; i++) {
        double prediction = decision_tree_predict(forest->trees[i]->root, x);
        votes[(int)prediction]++;
    }

    return (votes[0] > votes[1]) ? 0 : 1;
}

void random_forest_free(RandomForest *forest) {
    for (size_t i = 0; i < forest->n_trees; i++) {
        decision_tree_free(forest->trees[i]);
    }
    free(forest->trees);
    free(forest);
}