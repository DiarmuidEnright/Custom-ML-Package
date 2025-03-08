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

        Model *tree_model = (Model *)create_decision_tree();
        if (tree_model == NULL) {
            printf("Memory allocation failed\n");
            free(bootstrap_X);
            free(bootstrap_y);
            random_forest_free(forest);
            return NULL;
        }

        decision_tree_train(tree_model, bootstrap_X, bootstrap_y, n_samples, n_features, max_depth, min_samples_split);
        forest->trees[i] = (DecisionTree *)tree_model->current_tree;
        tree_model->current_tree = NULL;  // Prevent double free
        free(tree_model);  // Free the model structure but keep the tree

        free(bootstrap_X);
        free(bootstrap_y);
    }
    return forest;
}

double random_forest_predict(Forest *forest, double *x) {
    if (forest == NULL || forest->trees == NULL) {
        return -1;
    }

    double *predictions = (double *)malloc(forest->n_trees * sizeof(double));
    if (!predictions) {
        return -1;
    }

    for (size_t i = 0; i < forest->n_trees; i++) {
        predictions[i] = tree_predict(forest->trees[i]->root, x);
    }

    // Simple majority voting
    double sum = 0.0;
    for (size_t i = 0; i < forest->n_trees; i++) {
        sum += predictions[i];
    }
    free(predictions);

    return (sum / forest->n_trees) >= 0.5 ? 1.0 : 0.0;
}

void random_forest_free(Forest *forest) {
    if (forest == NULL) return;

    for (size_t i = 0; i < forest->n_trees; i++) {
        if (forest->trees[i] != NULL) {
            tree_free(forest->trees[i]->root);
            free(forest->trees[i]);
        }
    }
    free(forest->trees);
    free(forest);
}
