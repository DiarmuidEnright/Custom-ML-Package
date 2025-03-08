#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <stddef.h>
#include "model.h"

typedef struct TreeNode {
    size_t feature_idx;
    double threshold;
    double value;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

typedef struct DecisionTree {
    TreeNode *root;
    size_t max_depth;
    size_t min_samples_split;
} DecisionTree;

// Training and prediction functions
void decision_tree_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split);

// Public utility functions for other models to use
double tree_predict(TreeNode *node, double *x);
void tree_free(TreeNode *node);

// Creation functions
DecisionTree* create_decision_tree();

#endif
