#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "model.h"
#include "decision_tree.h"

double entropy(double *y, int n_samples) {
    double *counts = calloc(n_samples, sizeof(double));
    int i;
    for (i = 0; i < n_samples; i++) {
        counts[(size_t) y[i]]++;
    }
    double entropy_val = 0.0;
    double sum = 0.0;
    for (i = 0; i < n_classes; i++) {
        if (counts[i] > 0.0) {
            sum += counts[i];
            entropy_val -= (counts[i] / sum) * log2((counts[i] / sum));
        }
    }
    free(counts);
    return entropy_val;
}

double gini_index(double *y, int n_samples) {
    double *counts = calloc(n_samples, sizeof(double));
    int i;
    for (i = 0; i < n_samples; i++) {
        counts[(size_t) y[i]]++;
    }
    double gini_val = 1.0;
    double sum = 0.0;
    for (i = 0; i < n_classes; i++) {
        if (counts[i] > 0.0) {
            sum += counts[i];
            gini_val -= ((counts[i] / sum) * (counts[i] / sum));
        }
    }
    free(counts);
    return gini_val;
}

double majority_error(double *y, int n_samples) {
    double majority_class = 0.0;
    double count = 0.0;
    int i;
    for (i = 0; i < n_samples; i++) {
        if (y[i] == 0.0) {
            count++;
        }
    }
    if (count > (n_samples / 2.0)) {
        majority_class = 0.0;
    } else {
        majority_class = 1.0;
    }
    double error = 0.0;
    for (i = 0; i < n_samples; i++) {
        if (y[i] != majority_class) {
            error++;
        }
    }
    return error / n_samples;
}

double decision_tree_predict(TreeNode *node, double *x) {
    if (node->value == -1) {
        return node->left ? decision_tree_predict(node->left, x) : node->right->value;
    } else {
        return node->value;
    }
}

void decision_tree_free(TreeNode *node) {
    if (node == NULL) {
        return;
    }
    decision_tree_free(node->left);
    decision_tree_free(node->right);
    free(node);
}

void decision_tree_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split) {
    DecisionTree *tree = (DecisionTree *) malloc(sizeof(DecisionTree));
    tree->max_depth = max_depth;
    tree->min_samples_split = min_samples_split;
    tree->root = decision_tree_create(X, y, n_samples, n_features, 0, max_depth, min_samples_split);
    self->tree = tree;
}

TreeNode *decision_tree_create(double **X, double *y, int n_samples, int n_features, size_t depth, size_t max_depth, size_t min_samples_split) {
    if (depth >= max_depth || n_samples < min_samples_split) {
        return create_leaf_node(majority_class(y, n_samples));
    } else {
        double *feature_values = malloc(n_samples * sizeof(double));
        int *feature_indices = malloc(n_samples * sizeof(int));
        int i;
        for (i = 0; i < n_samples; i++) {
            feature_values[i] = X[i][depth];
            feature_indices[i] = (int) X[i][depth];
        }
        double best_value = -1;
        int best_index = -1;
        for (i = 0; i < n_samples; i++) {
            if (feature_values[i] > best_value) {
                best_value = feature_values[i];
                best_index = feature_indices[i];
            }
        }
        TreeNode *node = (TreeNode *) malloc(sizeof(TreeNode));
        node->value = -1;
        node->left = decision_tree_create(X, y, n_samples, n_features, depth + 1, max_depth, min_samples_split);
        node->right = decision_tree_create(X, y, n_samples, n_features, depth + 1, max_depth, min_samples_split);
        node->feature_index = best_index;
        node->threshold = best_value;
        free(feature_values);
        free(feature_indices);
        return node;
    }
}

TreeNode *create_leaf_node(double value) {
    TreeNode *node = (TreeNode *) malloc(sizeof(TreeNode));
    node->value = value;
    node->left = NULL;
    node->right = NULL;
    return node;
}

double majority_class(double *y, int n_samples) {
    double count = 0.0;
    int i;
    for (i = 0; i < n_samples; i++) {
        if (y[i] == 0.0) {
            count++;
        }
    }
    if (count > (n_samples / 2.0)) {
        return 0.0;
    } else {
        return 1.0;
    }
}