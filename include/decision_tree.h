#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <stddef.h>

typedef struct TreeNode TreeNode;

typedef struct {
    TreeNode *root;
} DecisionTree;

DecisionTree* decision_tree_train(double **X, double *y, size_t n_samples, size_t n_features, size_t max_depth, size_t min_samples_split);
double decision_tree_predict(TreeNode *node, double *x);
void decision_tree_free(DecisionTree *tree);

#endif
