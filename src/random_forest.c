#include "random_forest.h"
#include "model.h"
#include "decision_tree.h"
#include <stdlib.h>
#include <stdio.h>

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

        for (size_t j = 0; j < n_samples; j++) {
            size_t idx = rand() % n_samples;
            bootstrap_X[j] = (double *)malloc(n_features * sizeof(double));
            if (bootstrap_X[j] == NULL) {
                printf("Memory allocation failed\n");
                for (size_t k = 0; k < j; k++) {
                    free(bootstrap_X[k]);
                }
                free(bootstrap_X);
                free(bootstrap_y);
                random_forest_free(forest);
                return NULL;
            }
            for (size_t k = 0; k < n_features; k++) {
                bootstrap_X[j][k] = X[idx][k];
            }
            bootstrap_y[j] = y[idx];
        }

        Model *tree_model = (Model *)malloc(sizeof(Model));
        if (tree_model == NULL) {
            printf("Memory allocation failed\n");
            for (size_t j = 0; j < n_samples; j++) {
                free(bootstrap_X[j]);
            }
            free(bootstrap_X);
            free(bootstrap_y);
            random_forest_free(forest);
            return NULL;
        }
        tree_model->root = NULL;
        if (tree_model->root == NULL) {
            printf("Error initializing tree model\n");
            for (size_t j = 0; j < n_samples; j++) {
                free(bootstrap_X[j]);
            }
            free(bootstrap_X);
            free(bootstrap_y);
            free(tree_model);
            random_forest_free(forest);
            return NULL;
        }
        decision_tree_train(tree_model, bootstrap_X, bootstrap_y, n_samples, n_features, max_depth, min_samples_split);

        forest->trees[i] = tree_model;

        for (size_t j = 0; j < n_samples; j++) {
            free(bootstrap_X[j]);
        }
        free(bootstrap_X);
        free(bootstrap_y);
    }

    return forest;
}

double random_forest_predict(Forest *forest, double *x) {
    if (forest == NULL || x == NULL) {
        printf("Error: forest or x is NULL\n");
        return -1;
    }

    size_t votes[2] = {0, 0};

    for (size_t i = 0; i < forest->n_trees; i++) {
        double prediction = decision_tree_predict(forest->trees[i]->root, x);
        votes[(int)prediction]++;
    }

    return (votes[0] > votes[1]) ? 0 : 1;
}

void random_forest_free(Forest *forest) {
    if (forest == NULL) {
        return;
    }

    for (size_t i = 0; i < forest->n_trees; i++) {
        if (forest->trees[i] != NULL) {
            decision_tree_free(forest->trees[i]);
        }
    }
    free(forest->trees);
    free(forest);
}