#include "random_forest.h"
#include "model.h"
#include "decision_tree.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void random_forest_free(Forest *forest);

Forest* random_forest_train(double **X, double *y, size_t n_samples, size_t n_features, size_t n_trees, size_t max_depth, size_t min_samples_split) {
    Forest *forest = (Forest *)malloc(sizeof(Forest));
    if (forest == NULL) {
        printf("Memory allocation failed\n");
        return NULL;
    }
    forest->n_trees = n_trees;
    forest->trees = (DecisionTree **)malloc(n_trees * sizeof(DecisionTree *));
    if (forest->trees == NULL) {
        printf("Memory allocation failed\n");
        free(forest);
        return NULL;
    }

    for (size_t i = 0; i < n_trees; i++) {
        double **bootstrap_X = (double **)malloc(n_samples * sizeof(double *));
        if (bootstrap_X == NULL) {
            printf("Memory allocation failed\n");
            random_forest_free(forest);
            return NULL;
        }

        double *bootstrap_y = (double *)malloc(n_samples * sizeof(double));
        if (bootstrap_y == NULL) {
            printf("Memory allocation failed\n");
            free(bootstrap_X);
            random_forest_free(forest);
            return NULL;
        }

        DecisionTree *tree = (DecisionTree *)malloc(sizeof(DecisionTree));
        if (tree == NULL) {
            printf("Memory allocation failed\n");
            free(bootstrap_X);
            free(bootstrap_y);
            random_forest_free(forest);
            return NULL;
        }
        decision_tree_train((Model *)tree, bootstrap_X, bootstrap_y, n_samples, n_features, max_depth, min_samples_split);
        forest->trees[i] = tree;
    }
    return forest;
}

void random_forest_free(Forest *forest) {
    if (forest == NULL) return;

    for (size_t i = 0; i < forest->n_trees; i++) {
        if (forest->trees[i] != NULL) {
            decision_tree_free((TreeNode *)forest->trees[i]); // Casting to TreeNode* here
        }
    }
    free(forest->trees);
    free(forest);
}
