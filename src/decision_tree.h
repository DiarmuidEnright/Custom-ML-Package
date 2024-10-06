#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <stdlib.h>
#include "model.h"

typedef struct TreeNode {
    size_t feature_idx;
    double threshold;
    double value;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

typedef struct {
    TreeNode *root;
} DecisionTree;

void decision_tree_train(Model *self, double **X, int n_samples, int n_features);
void decision_tree_free(DecisionTree *tree);
double decision_tree_predict(TreeNode *node, double *x);

#endif