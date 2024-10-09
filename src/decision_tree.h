#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <stddef.h>

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

typedef struct Model {
    void (*train)(struct Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split);
} Model;

void decision_tree_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split);
void decision_tree_free(TreeNode *node);

DecisionTree* create_decision_tree();

#endif
