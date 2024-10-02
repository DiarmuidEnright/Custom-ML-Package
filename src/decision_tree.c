#include "decision_tree.h"
#include <stdlib.h>

Node* decision_tree_train(double **X, double *y, size_t n_samples, size_t n_features) {
    Node *root = (Node*)malloc(sizeof(Node));
    // Pseudocode for training:
    // 1. Find the best feature and threshold to split on.
    // 2. Recursively create left and right subtrees.
    // 3. If a leaf node, assign a prediction value.
    
    return root;
}

double decision_tree_predict(Node *tree, double *x) {
    if (!tree->left && !tree->right) {
        return tree->prediction;
    }
    if (x[tree->feature_index] < tree->threshold) {
        return decision_tree_predict(tree->left, x);
    } else {
        return decision_tree_predict(tree->right, x);
    }
}

void decision_tree_free(Node *tree) {
    if (tree->left) decision_tree_free(tree->left);
    if (tree->right) decision_tree_free(tree->right);
    free(tree);
}
