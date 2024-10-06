#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <stdlib.h>

typedef struct Model Model;

typedef struct TreeNode {
    size_t feature_idx;
    double threshold;
    double value;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

typedef struct DecisionTree {
    TreeNode *root;
} DecisionTree;

DecisionTree* decision_tree_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split);
void decision_tree_free(DecisionTree *tree);
double decision_tree_predict(TreeNode *node, double *x);

#endif