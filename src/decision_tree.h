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

typedef struct DecisionTree {
    TreeNode *root;
    size_t max_depth;
    size_t min_samples_split;
} DecisionTree;


typedef struct Model {
    DecisionTree *current_tree;
    void (*train)(struct Model *self, double **data, double *target, int n_samples, int n_features, size_t max_depth, size_t min_samples_split);
    double (*predict)(struct Model *self, double *data, int n_features);
    void (*free)(struct Model *self);
} Model;

Model* create_decision_tree(void);
TreeNode* decision_tree_create(double **X, double *y, int n_samples, int n_features, size_t depth, size_t max_depth, size_t min_samples_split);
TreeNode* create_leaf_node(double value);
TreeNode* create_majority_class(double *y, int n_samples)

void decision_tree_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split);

void decision_tree_free(TreeNode *node);

double decision_tree_predict(TreeNode *node, double *x);

#endif
