#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "model.h"
#include "decision_tree.h"

extern int n_classes;

// Public utility functions for other models to use
double tree_predict(TreeNode *node, double *x) {
    if (node->value == -1) {
        return x[node->feature_idx] <= node->threshold
            ? tree_predict(node->left, x)
            : tree_predict(node->right, x);
    } else {
        return node->value;
    }
}

void tree_free(TreeNode *node) {
    if (node == NULL) return;
    tree_free(node->left);
    tree_free(node->right);
    free(node);
}

// Model interface functions
static double model_predict(Model *self, double *x, int n_features) {
    DecisionTree *tree = (DecisionTree *)self->current_tree;
    return tree_predict(tree->root, x);
}

static void model_free(Model *self) {
    DecisionTree *tree = (DecisionTree *)self->current_tree;
    if (tree) {
        tree_free(tree->root);
        free(tree);
    }
    free(self);
}

DecisionTree* create_decision_tree() {
    Model *model = (Model *)malloc(sizeof(Model));
    if (!model) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    model->train = decision_tree_train;
    model->predict = model_predict;
    model->free = model_free;
    model->current_tree = NULL;

    return (DecisionTree *)model;
}

static double entropy(double *y, int n_samples) {
    double *counts = calloc(n_classes, sizeof(double));
    if (!counts) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }

    for (int i = 0; i < n_samples; i++) {
        counts[(size_t)y[i]]++;
    }

    double entropy_val = 0.0;
    double sum = (double)n_samples;
    for (int i = 0; i < n_classes; i++) {
        if (counts[i] > 0.0) {
            double p = counts[i] / sum;
            entropy_val -= p * log2(p);
        }
    }

    free(counts);
    return entropy_val;
}

static TreeNode *create_leaf_node(double value) {
    TreeNode *node = (TreeNode *)malloc(sizeof(TreeNode));
    if (!node) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    node->value = value;
    node->left = NULL;
    node->right = NULL;
    return node;
}

static double majority_class(double *y, int n_samples) {
    double *counts = calloc(n_classes, sizeof(double));
    if (!counts) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }

    for (int i = 0; i < n_samples; i++) {
        counts[(size_t)y[i]]++;
    }

    double majority_class = 0.0;
    double max_count = 0.0;
    for (int i = 0; i < n_classes; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            majority_class = i;
        }
    }

    free(counts);
    return majority_class;
}

static TreeNode *tree_create_recursive(double **X, double *y, int n_samples, int n_features, size_t depth, size_t max_depth, size_t min_samples_split) {
    if (depth >= max_depth || n_samples < min_samples_split) {
        return create_leaf_node(majority_class(y, n_samples));
    }

    int best_feature = -1;
    double best_threshold = 0.0;
    double best_gain = -INFINITY;

    for (int f = 0; f < n_features; f++) {
        for (int i = 0; i < n_samples; i++) {
            double threshold = X[i][f];
            double gain = 0.0;
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_threshold = threshold;
            }
        }
    }

    if (best_feature == -1) {
        return create_leaf_node(majority_class(y, n_samples));
    }

    TreeNode *node = (TreeNode *)malloc(sizeof(TreeNode));
    if (!node) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    node->value = -1;
    node->feature_idx = best_feature;
    node->threshold = best_threshold;
    node->left = tree_create_recursive(X, y, n_samples / 2, n_features, depth + 1, max_depth, min_samples_split);
    node->right = tree_create_recursive(X, y, n_samples / 2, n_features, depth + 1, max_depth, min_samples_split);

    return node;
}

void decision_tree_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split) {
    DecisionTree *tree = (DecisionTree *)malloc(sizeof(DecisionTree));
    if (!tree) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

    tree->max_depth = max_depth;
    tree->min_samples_split = min_samples_split;
    tree->root = tree_create_recursive(X, y, n_samples, n_features, 0, max_depth, min_samples_split);
    self->current_tree = tree;
}
