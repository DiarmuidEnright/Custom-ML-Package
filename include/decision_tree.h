#ifndef DECISION_TREE_H
#define DECISION_TREE_H

typedef struct TreeNode {
    double threshold;
    int feature_index;
    double value;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

typedef struct {
    TreeNode *root;
} DecisionTree;

DecisionTree* decision_tree_train(double **X, double *y, size_t n_samples, size_t n_features, size_t max_depth, size_t min_samples_split);
double decision_tree_predict(TreeNode *node, double *x);  // Add this
void decision_tree_free(DecisionTree *tree);              // Add this

#endif
