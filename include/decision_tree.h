#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <stddef.h>

typedef struct Node {
    int feature_index;
    double threshold;
    double prediction;
    struct Node *left;
    struct Node *right;
} Node;

Node* decision_tree_train(double **X, double *y, size_t n_samples, size_t n_features);
double decision_tree_predict(Node *tree, double *x);
void decision_tree_free(Node *tree);

#endif