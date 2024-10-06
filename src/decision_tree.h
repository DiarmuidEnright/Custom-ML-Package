#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <stdlib.h>

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

DecisionTree* decision_tree_train(double **X, double *y, size_t n_samples, size_t n_features, size_t max_depth, size_t min_samples_split);
void decision_tree_free(DecisionTree *tree);
double decision_tree_predict(TreeNode *node, double *x);

#endif
